"""Drive the *real* vLLM scheduler on CPU to synthesise a shape-accurate trace.

Nothing here re-implements scheduling: ``Scheduler.schedule()`` decides
admissions, chunked-prefill splits, prefix-cache hits, block allocation,
preemption and MTP draft slots; ``Scheduler.update_from_output()`` advances the
state from a deterministic fake "model".  The only stand-ins are

* a ``SimpleNamespace`` ``VllmConfig`` (no model weights / tokenizer needed),
* ``StructuredOutputManager`` built with ``skip_tokenizer_init=True``,
* sampled token ids (constants; the harness studies CPU-side ``prepare_input``,
  not logits),
* the worker-side bookkeeping in :mod:`pi_harness.workload.worker_state`.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

from pi_harness.workload.generate import SynthConfig, arrival_plan, build_prompts
from pi_harness.workload.schema import (
    SCHEMA_VERSION,
    StepRecord,
    encode_scheduler_output,
    infer_phase,
)
from pi_harness.workload.worker_state import (
    WorkerBatch,
    apply_step,
    record_sampled_tokens,
    snapshot_input_batch,
    snapshot_req_states,
)


def build_fake_vllm_config(c: SynthConfig) -> Any:
    """Minimal stand-in for ``VllmConfig`` covering every field the scheduler
    reads (checked against ``scheduler.py`` of vLLM 0.26.0)."""
    scheduler_config = SimpleNamespace(
        policy="fcfs",
        max_num_seqs=c.max_num_seqs,
        max_num_batched_tokens=c.chunk_size,
        max_num_scheduled_tokens=None,
        async_scheduling=False,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=0,
        disable_chunked_mm_input=False,
        scheduler_reserve_full_isl=False,
        watermark=0.0,
    )
    cache_config = SimpleNamespace(
        block_size=c.block_size,
        num_gpu_blocks=c.num_blocks,
        enable_prefix_caching=bool(c.prefix_hit_ratio > 0.0),
        prefix_caching_hash_algo="sha256",
        mamba_cache_mode="none",
    )
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_index=0,
        distributed_executor_backend="uni",
    )
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        is_diffusion=False,
        max_model_len=c.max_model_len,
        enable_return_routed_experts=False,
        skip_tokenizer_init=True,
        runner_type="generate",
    )
    observability_config = SimpleNamespace(
        kv_cache_metrics=False,
        kv_cache_metrics_sample=0.01,
        enable_mfu_metrics=False,
        enable_logging_iteration_details=False,
    )
    structured_outputs_config = SimpleNamespace(
        reasoning_parser_plugin=None,
        reasoning_parser=None,
        enable_in_reasoning=False,
    )

    speculative_config = None
    if c.spec_k > 0:
        spec = SimpleNamespace(
            method="mtp",
            num_speculative_tokens_per_batch_size=None,
        )
        # mirrors vllm.config.speculative.SpeculativeConfig for method="mtp"
        spec.use_eagle = lambda: True
        spec.uses_draft_model = lambda: False
        spec.use_dflash = lambda: False
        spec.use_dspark = lambda: False
        spec.use_ngram_gpu = lambda: False
        speculative_config = spec

    return SimpleNamespace(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        model_config=model_config,
        observability_config=observability_config,
        structured_outputs_config=structured_outputs_config,
        lora_config=None,
        kv_events_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=speculative_config,
        num_speculative_tokens=c.spec_k,
        max_in_flight_tokens=c.max_model_len * max(c.max_num_seqs, 1),
        max_concurrent_batches=1,
        use_v2_model_runner=False,
    )


class SchedulerCtx:
    """Real ``Scheduler`` + ``KVCacheManager`` brought up on CPU."""

    def __init__(self, c: SynthConfig) -> None:
        import torch

        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
        )
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.structured_output import StructuredOutputManager

        self.cfg = c
        self.vllm_config = build_fake_vllm_config(c)
        self.kv_cache_config = KVCacheConfig(
            num_blocks=c.num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["model.layers.0.self_attn"],
                    kv_cache_spec=FullAttentionSpec(
                        block_size=c.block_size,
                        num_kv_heads=8,
                        head_size=128,
                        dtype=torch.bfloat16,
                    ),
                )
            ],
        )
        no_mm = type(
            "_NoMMRegistry", (), {"supports_multimodal_inputs": lambda _s, _c: False}
        )()
        self.scheduler = Scheduler(
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
            structured_output_manager=StructuredOutputManager(self.vllm_config),
            block_size=c.block_size,
            hash_block_size=c.block_size,
            mm_registry=no_mm,
        )
        caching_hash_fn = get_hash_fn_by_name("sha256")
        init_none_hash(caching_hash_fn)
        self.block_hasher = get_request_block_hasher(c.block_size, caching_hash_fn)
        self.num_groups = len(self.kv_cache_config.kv_cache_groups)

    def make_request(self, req_id: str, prompt_token_ids: list[int]) -> Any:
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        return Request(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=SamplingParams(
                max_tokens=max(self.cfg.osl, 1),
                ignore_eos=True,
                temperature=0.0,
                seed=0,
            ),
            pooling_params=None,
            block_hasher=self.block_hasher,
        )


def _fake_model_output(
    so_dict: dict, c: SynthConfig, rng: random.Random
) -> tuple[dict, dict[str, int]]:
    """Deterministic sampling result: ``1 + accepted drafts`` tokens per req.

    Token *values* are distinct per request: if every request emitted the same
    token, the KV blocks of the continuations would hash identically and the
    scheduler's ``num_common_prefix_blocks`` would report a shared prefix that
    cannot exist in a real run (a shape artefact).
    """
    req_ids = list((so_dict.get("num_scheduled_tokens") or {}).keys())
    spec = so_dict.get("scheduled_spec_decode_tokens") or {}
    sampled: list[list[int]] = []
    lens: dict[str, int] = {}
    vocab = max(int(c.vocab_size), 64)
    for rid in req_ids:
        k = len(spec.get(rid) or [])
        accepted = 0
        for _ in range(k):
            if rng.random() < c.accept_ratio:
                accepted += 1
            else:
                break
        n_out = accepted + 1
        seq = int(rid.split("-")[-1]) if rid.split("-")[-1].isdigit() else hash(rid) & 0xFFFF
        sampled.append(
            [
                (seq * 7919 + pos * 104729 + c.token_id_base) % (vocab - 1) + 1
                for pos in range(n_out)
            ]
        )
        lens[rid] = n_out
    out = {
        "req_ids": req_ids,
        "req_id_to_index": {rid: i for i, rid in enumerate(req_ids)},
        "sampled_token_ids": sampled,
    }
    return out, lens


def run(c: SynthConfig) -> list[StepRecord]:
    ctx = SchedulerCtx(c)
    sched = ctx.scheduler
    cfg = ctx.cfg
    rng = random.Random(cfg.seed)
    plan = arrival_plan(cfg, rng)
    prompts, prefix_hits = build_prompts(cfg, plan)

    batch = WorkerBatch()
    states: dict[str, dict] = {}
    recs: list[StepRecord] = []
    admitted = 0
    max_steps = cfg.steps if cfg.steps > 0 else cfg.steps_estimate()
    token_ids_shape = (cfg.max_num_seqs, cfg.max_model_len)

    for step in range(max_steps):
        while admitted < cfg.batch and plan[admitted] <= step:
            rid = f"{cfg.request_id_prefix}-{admitted:04d}"
            sched.add_request(ctx.make_request(rid, prompts[admitted]))
            admitted += 1
        if not sched.requests:
            if admitted < cfg.batch:
                # arrivals have not started yet (poisson / staircase)
                continue
            # all requests are done: the scheduler still has to *report* the
            # last finishes, so run one more (empty) step and record it as a
            # drain step.  This is what leaves the worker batch empty, which is
            # what a replayer needs to reuse a runner across runs.
            so = sched.schedule()
            if so.finished_req_ids:
                recs.append(
                    _drain_record(
                        len(recs), so, batch, states, ctx.num_groups, prefix_hits, cfg
                    )
                )
            break

        so = sched.schedule()
        so_dict = encode_scheduler_output(so)
        if so.total_num_scheduled_tokens == 0:
            if so.finished_req_ids:
                recs.append(
                    _drain_record(
                        len(recs), so, batch, states, ctx.num_groups, prefix_hits, cfg
                    )
                )
            if admitted < cfg.batch:
                continue
            break

        # worker view at _prepare_inputs() entry
        apply_step(so_dict, batch, states, num_groups=ctx.num_groups)
        new_ids = {rd["req_id"] for rd in so_dict.get("scheduled_new_reqs") or []}
        req_states = snapshot_req_states(batch, states, include_prompts_for=new_ids)
        input_batch = snapshot_input_batch(
            batch,
            states,
            token_ids_shape=token_ids_shape,
            num_groups=ctx.num_groups,
        )
        model_output, sampled_lens = _fake_model_output(so_dict, cfg, rng)
        rec = StepRecord(
            step_idx=len(recs),
            phase=infer_phase(so_dict, req_states),
            scheduler_output=so_dict,
            req_states=req_states,
            model_output=model_output,
            t_prepare_input_us=None,
            input_batch=input_batch,
            timings=None,
            meta={
                "source": "synth",
                "synth": True,
                "engine": "real",
                "schema_version": SCHEMA_VERSION,
                "prefix_hits": prefix_hits,
                "params": cfg.to_dict(),
            },
        )
        recs.append(rec)

        # advance the real scheduler with the fake model output
        from vllm.v1.outputs import ModelRunnerOutput

        sched.update_from_output(
            so,
            ModelRunnerOutput(
                req_ids=list(model_output["req_ids"]),
                req_id_to_index=dict(model_output["req_id_to_index"]),
                sampled_token_ids=[list(r) for r in model_output["sampled_token_ids"]],
            ),
        )
        record_sampled_tokens(so_dict, states, sampled_lens)

        # MTP: hand the next step its draft tokens (this is what the worker's
        # propose_draft_token_ids path does in the real run)
        if cfg.spec_k > 0:
            from vllm.v1.outputs import DraftTokenIds

            draft_ids: list[str] = []
            draft_tokens: list[list[int]] = []
            vocab = max(int(cfg.vocab_size), 64)
            for rid, req in sched.requests.items():
                if req.is_finished() or req.is_prefill_chunk:
                    continue
                draft_ids.append(rid)
                seq = (
                    int(rid.split("-")[-1])
                    if rid.split("-")[-1].isdigit()
                    else hash(rid) & 0xFFFF
                )
                draft_tokens.append(
                    [
                        (seq * 7919 + (i + 1) * 65537 + cfg.token_id_base)
                        % (vocab - 1)
                        + 1
                        for i in range(cfg.spec_k)
                    ]
                )
            if draft_ids:
                sched.update_draft_token_ids(DraftTokenIds(draft_ids, draft_tokens))

    return recs


def _drain_record(
    step_idx: int,
    so: Any,
    batch: Any,
    states: dict[str, dict],
    num_groups: int,
    prefix_hits: list[bool],
    cfg: SynthConfig,
) -> StepRecord:
    """Record an empty step that only reports finished requests."""
    so_dict = encode_scheduler_output(so)
    apply_step(so_dict, batch, states, num_groups=num_groups)
    return StepRecord(
        step_idx=step_idx,
        phase="drain",
        scheduler_output=so_dict,
        req_states=snapshot_req_states(batch, states),
        model_output=None,
        t_prepare_input_us=None,
        input_batch=snapshot_input_batch(
            batch,
            states,
            token_ids_shape=(cfg.max_num_seqs, cfg.max_model_len),
            num_groups=num_groups,
        ),
        timings=None,
        meta={
            "source": "synth",
            "synth": True,
            "engine": "real",
            "drain": True,
            "schema_version": SCHEMA_VERSION,
            "prefix_hits": prefix_hits,
            "params": cfg.to_dict(),
        },
    )
