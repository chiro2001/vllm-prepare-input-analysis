"""Pure-Python fallback synthesiser (used only when vllm cannot be imported).

It reproduces the *rules* that shape the load — FIFO admission under a
``max_num_batched_tokens`` budget, chunked prefill, one token per decode step,
optional MTP drafts, block-aligned allocation with prefix-cache reuse — without
running the real scheduler.  Traces made here carry
``meta.engine="pure(fallback)"`` so no claim of scheduler-level fidelity is ever
attached to them.
"""

from __future__ import annotations

import math
import random

from pi_harness.workload.generate import SynthConfig, arrival_plan, build_prompts
from pi_harness.workload.schema import SCHEMA_VERSION, StepRecord, infer_phase
from pi_harness.workload.worker_state import (
    WorkerBatch,
    apply_step,
    record_sampled_tokens,
    snapshot_input_batch,
    snapshot_req_states,
)


class _Req:
    __slots__ = (
        "req_id",
        "prompt",
        "num_computed",
        "num_output",
        "blocks",
        "gone",
        "spec_len",
    )

    def __init__(self, req_id: str, prompt: list[int]) -> None:
        self.req_id = req_id
        self.prompt = prompt
        self.num_computed = 0
        self.num_output = 0
        self.blocks: list[list[int]] = [[]]
        self.gone = False
        self.spec_len = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    @property
    def is_prefill_chunk(self) -> bool:
        return self.num_computed < self.prompt_len


def run(c: SynthConfig) -> list[StepRecord]:
    rng = random.Random(c.seed)
    plan = arrival_plan(c, rng)
    prompts, prefix_hits = build_prompts(c, plan)

    batch = WorkerBatch()
    states: dict[str, dict] = {}
    recs: list[StepRecord] = []
    counter = [1]  # next free block id
    # shared-prefix blocks are cached at block granularity, like a real prefix cache
    cached_prefix_blocks: list[int] = []
    waiting: list[_Req] = []
    running: list[_Req] = []
    admitted = 0
    # finished requests are reported one step late, like the real scheduler
    pending_finish: set[str] = set()

    max_steps = c.steps if c.steps > 0 else c.steps_estimate()
    for step in range(max_steps):
        while admitted < c.batch and plan[admitted] <= step:
            waiting.append(_Req(f"{c.request_id_prefix}-{admitted:04d}", prompts[admitted]))
            admitted += 1

        budget = c.chunk_size
        num_scheduled: dict[str, int] = {}
        new_reqs: list[dict] = []
        spec_tokens: dict[str, list[int]] = {}
        finished_req_ids = sorted(pending_finish)
        pending_finish.clear()

        # 1) running requests first (FIFO): finish the prompt, then decode
        for req in list(running):
            if req.gone:
                continue
            if budget <= 0:
                break
            if req.is_prefill_chunk:
                take = min(budget, req.prompt_len - req.num_computed)
            else:
                take = 1 + (c.spec_k if c.spec_k else 0)
            take = max(take, 1)
            num_scheduled[req.req_id] = take
            if c.spec_k and not req.is_prefill_chunk:
                spec_tokens[req.req_id] = [1] * c.spec_k
            budget -= take
            _grow_blocks(req, req.num_computed + take, c, counter)

        # 2) admit new requests while budget and slots remain
        while waiting and budget > 0 and len(running) < c.max_num_seqs:
            req = waiting[0]
            take = min(budget, max(req.prompt_len, 1))
            take = max(take, 1)
            hit = prefix_hits[int(req.req_id.split("-")[-1])]
            if hit:
                prefix_blocks = (req.prompt_len and c.prefix_len) // c.block_size
                while len(cached_prefix_blocks) < prefix_blocks:
                    cached_prefix_blocks.append(counter[0])
                    counter[0] += 1
                req.blocks[0] = list(cached_prefix_blocks[:prefix_blocks])
                req.num_computed = min(prefix_blocks * c.block_size, req.prompt_len)
                take = min(budget, max(req.prompt_len - req.num_computed, 1))
            waiting.pop(0)
            running.append(req)
            num_scheduled[req.req_id] = take
            budget -= take
            _grow_blocks(req, req.num_computed + take, c, counter)
            new_reqs.append(
                {
                    "req_id": req.req_id,
                    "prompt_token_ids": list(req.prompt),
                    "mm_features": [],
                    # same encoding the capture hook writes for a real
                    # SamplingParams, so replay decodes it with the same code
                    "sampling_params": {
                        "__type__": "SamplingParams",
                        "primitives": {
                            "n": 1,
                            "max_tokens": max(c.osl, 1),
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "ignore_eos": True,
                            "seed": 0,
                        },
                    },
                    "pooling_params": None,
                    "block_ids": [list(g) for g in req.blocks],
                    "num_computed_tokens": req.num_computed,
                    "lora_request": None,
                    "prompt_embeds_shape": None,
                    "prompt_is_token_ids": None,
                    "prefill_token_ids": None,
                }
            )

        if not num_scheduled:
            # nothing left to schedule: report the remaining finishes as a
            # drain step (see synth_real._drain_record) and stop
            # ids reported at the top of this (otherwise empty) step are still
            # outstanding, plus whatever finished at the very end
            pending_finished = sorted(set(finished_req_ids) | pending_finish)
            pending_finish.clear()
            if pending_finished:
                so_dict = _empty_scheduler_output(pending_finished)
                apply_step(so_dict, batch, states, num_groups=1)
                recs.append(
                    StepRecord(
                        step_idx=len(recs),
                        phase="drain",
                        scheduler_output=so_dict,
                        req_states=snapshot_req_states(batch, states),
                        model_output=None,
                        t_prepare_input_us=None,
                        input_batch=snapshot_input_batch(
                            batch,
                            states,
                            token_ids_shape=(c.max_num_seqs, c.max_model_len),
                            num_groups=1,
                        ),
                        timings=None,
                        meta={
                            "source": "synth",
                            "synth": True,
                            "engine": "pure",
                            "drain": True,
                            "schema_version": SCHEMA_VERSION,
                            "prefix_hits": prefix_hits,
                            "params": c.to_dict(),
                        },
                    )
                )
            break

        new_ids = {r["req_id"] for r in new_reqs}
        cached_req_ids = [
            req.req_id for req in running if req.req_id not in new_ids
        ]
        so_dict = {
            "scheduled_new_reqs": new_reqs,
            "scheduled_cached_reqs": {
                "req_ids": cached_req_ids,
                "resumed_req_ids": [],
                "new_token_ids": [],
                "all_token_ids": {},
                "new_block_ids": [
                    [
                        list(g)
                        for g in _new_blocks_for(req, num_scheduled[req.req_id])
                    ]
                    for req in running
                    if req.req_id not in new_ids
                ],
                "num_computed_tokens": [
                    req.num_computed
                    for req in running
                    if req.req_id not in new_ids
                ],
                "num_output_tokens": [
                    req.num_output for req in running if req.req_id not in new_ids
                ],
            },
            "num_scheduled_tokens": num_scheduled,
            "total_num_scheduled_tokens": sum(num_scheduled.values()),
            "scheduled_spec_decode_tokens": spec_tokens,
            "scheduled_encoder_inputs": {},
            "num_common_prefix_blocks": [len(cached_prefix_blocks)],
            "finished_req_ids": sorted(set(finished_req_ids)),
            "free_encoder_mm_hashes": [],
            "scheduled_encoder_input_stats": None,
            "preempted_req_ids": None,
            "has_structured_output_requests": False,
            "pending_structured_output_tokens": False,
            "num_invalid_spec_tokens": None,
            "kv_connector_metadata": None,
            "ec_connector_metadata": None,
            "new_block_ids_to_zero": None,
            "kv_cache_block_copies": None,
            "num_spec_tokens_to_schedule": c.spec_k,
        }

        apply_step(so_dict, batch, states, num_groups=1)
        for req in running:
            if req.req_id not in states:
                continue
            st = states[req.req_id]
            st["prompt_token_ids"] = list(req.prompt)
            st["num_prompt_tokens"] = req.prompt_len
            st["num_computed_tokens"] = req.num_computed
            st["block_ids"] = [list(g) for g in req.blocks]
            st["num_output_tokens"] = req.num_output
            st["spec_len"] = req.spec_len
        batch.condense()
        req_states = snapshot_req_states(
            batch, states, include_prompts_for={r["req_id"] for r in new_reqs}
        )
        input_batch = snapshot_input_batch(
            batch,
            states,
            token_ids_shape=(c.max_num_seqs, c.max_model_len),
            num_groups=1,
        )

        sampled: dict[str, int] = {}
        for rid, ntok in num_scheduled.items():
            req = next(r for r in running if r.req_id == rid)
            if req.is_prefill_chunk:
                req.num_computed += ntok
                if req.num_computed >= req.prompt_len:
                    req.num_computed = req.prompt_len
                sampled[rid] = 0
            else:
                acc = 0
                for _ in range(c.spec_k):
                    if rng.random() < c.accept_ratio:
                        acc += 1
                    else:
                        break
                req.num_computed += acc + 1
                req.num_output += acc + 1
                sampled[rid] = acc + 1
        model_output = {
            "req_ids": list(num_scheduled),
            "req_id_to_index": {rid: i for i, rid in enumerate(num_scheduled)},
            "sampled_token_ids": [
                # distinct values per request (see synth_real._fake_model_output)
                [
                    (int(r.split("-")[-1]) * 7919 + k * 104729 + 7)
                    % max(c.vocab_size - 1, 1)
                    + 1
                    for k in range(max(sampled[r], 1))
                ]
                for r in num_scheduled
            ],
        }

        recs.append(
            StepRecord(
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
                    "engine": "pure",
                    "schema_version": SCHEMA_VERSION,
                    "prefix_hits": prefix_hits,
                    "params": c.to_dict(),
                },
            )
        )
        record_sampled_tokens(so_dict, states, {k: v for k, v in sampled.items() if v})

        for req in list(running):
            if req.num_computed >= req.prompt_len + c.osl:
                req.gone = True
                pending_finish.add(req.req_id)
                running.remove(req)

    return recs


def _grow_blocks(req: _Req, up_to_tokens: int, c: SynthConfig, counter: list[int]) -> None:
    """Grow the request's block list (block-aligned) from the shared counter."""
    need = math.ceil(up_to_tokens / c.block_size) if up_to_tokens else 0
    while len(req.blocks[0]) < need:
        req.blocks[0].append(counter[0])
        counter[0] += 1


def _empty_scheduler_output(finished: list[str]) -> dict:
    return {
        "scheduled_new_reqs": [],
        "scheduled_cached_reqs": {
            "req_ids": [],
            "resumed_req_ids": [],
            "new_token_ids": [],
            "all_token_ids": {},
            "new_block_ids": [],
            "num_computed_tokens": [],
            "num_output_tokens": [],
        },
        "num_scheduled_tokens": {},
        "total_num_scheduled_tokens": 0,
        "scheduled_spec_decode_tokens": {},
        "scheduled_encoder_inputs": {},
        "num_common_prefix_blocks": [],
        "finished_req_ids": sorted(set(finished)),
        "free_encoder_mm_hashes": [],
        "scheduled_encoder_input_stats": None,
        "preempted_req_ids": None,
        "has_structured_output_requests": False,
        "pending_structured_output_tokens": False,
        "num_invalid_spec_tokens": None,
        "kv_connector_metadata": None,
        "ec_connector_metadata": None,
        "new_block_ids_to_zero": None,
        "kv_cache_block_copies": None,
        "num_spec_tokens_to_schedule": 0,
    }


def _new_blocks_for(req: _Req, _n: int) -> list[list[int]]:
    return [[] for _ in req.blocks]
