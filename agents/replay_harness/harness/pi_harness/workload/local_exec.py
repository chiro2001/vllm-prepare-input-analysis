"""CPU executor for the A/B harness: the same data path for every trace group.

``real`` / ``shuffled`` / ``synth`` traces are replayed through *this* executor,
so the only difference between the groups is the data inside the records.

The executor mirrors the CPU-side work of
``GPUModelRunner._update_states`` + ``_prepare_inputs``
(``vllm/v1/worker/gpu_model_runner.py``, and the Ascend overrides in
``vllm_ascend/worker/model_runner_v1.py``) on plain numpy buffers:

* ``decode``          — JSON dict -> real ``SchedulerOutput`` (via
                        ``pi_harness.trace.replay``; skipped when vllm is absent)
* ``update_states``   — per-request bookkeeping, token-id row copies, block-table
                        row append/commit (the ``InputBatch`` half)
* ``prepare_inputs``  — ``np.repeat``/``cumsum`` index arithmetic, position
                        computation, the big ``index_select`` gather of
                        ``token_ids_cpu`` into ``input_ids``, slot mapping,
                        spec-decode metadata (the ``_prepare_inputs`` half)

When the real runner (``pi_harness.runner.replay.PrepareInputReplay``) is
available, ``ab.py`` prefers it (``--executor runner``); this module is the
always-available fallback and the cross-check.  Deviations from the real
functions are listed in the report fidelity table.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pi_harness.workload.schema import StepRecord, shape_signature


def _get(so: Any, name: str, default: Any = None) -> Any:
    """Field access that works for both a real ``SchedulerOutput`` and the
    JSON dict form (the A/B runs either: real object when vllm is importable,
    dict otherwise)."""
    if isinstance(so, dict):
        return so.get(name, default)
    return getattr(so, name, default)


class LocalPrepareInputExecutor:
    def __init__(
        self,
        *,
        max_num_reqs: int = 256,
        max_model_len: int = 8192,
        block_size: int = 128,
        max_num_batched_tokens: int = 2048,
        decode_with_vllm: bool = True,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_blocks_per_req = max(1, max_model_len // max(block_size, 1) + 2)
        self.decode_with_vllm = decode_with_vllm
        # long-lived buffers, exactly like the real InputBatch/model runner
        self.token_ids_cpu = np.zeros(
            (max_num_reqs, max_model_len), dtype=np.int32
        )
        # real model runner: self.input_ids = _make_buffer(max_num_tokens, int32)
        self.input_ids = np.zeros(max_num_batched_tokens, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.block_table_cpu = np.zeros(
            (max_num_reqs, self.max_num_blocks_per_req), dtype=np.int32
        )
        self.block_table_gpu = np.zeros_like(self.block_table_cpu)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.arange_np = np.arange(
            max(max_model_len, max_num_batched_tokens) + 1, dtype=np.int64
        )
        self.query_pos = np.zeros(max(max_model_len, max_num_batched_tokens) + 1, np.int32)
        self.slot_mapping = np.zeros(max_num_batched_tokens, dtype=np.int64)
        self.logits_indices = np.zeros(max_num_reqs + 1, dtype=np.int64)
        self._req_ids: list[str] = []
        self.req_id_to_index: dict[str, int] = {}
        self._removed: list[int] = []
        self.prompt_cache: dict[str, list[int]] = {}

    # -- request bookkeeping (InputBatch) -----------------------------------
    @property
    def req_ids(self) -> list[str]:
        return self._req_ids

    def _add(self, rid: str, prompt_ids: list[int], num_computed: int, blocks: list[list[int]]) -> int:
        if self._removed:
            idx = self._removed.pop(0)
            self._req_ids[idx] = rid
        else:
            idx = len(self._req_ids)
            self._req_ids.append(rid)
        self.req_id_to_index[rid] = idx
        n = len(prompt_ids)
        self.num_prompt_tokens_cpu[idx] = n
        if n:
            self.token_ids_cpu[idx, :n] = prompt_ids
        self.num_computed_tokens_cpu[idx] = num_computed
        self.num_tokens_no_spec[idx] = n
        self.block_table_cpu[idx].fill(0)
        self.num_blocks_per_row[idx] = 0
        for group in blocks:
            self._append_blocks(idx, group)
        return idx

    def _append_blocks(self, idx: int, blocks: list[int]) -> None:
        if not blocks:
            return
        start = int(self.num_blocks_per_row[idx])
        end = min(start + len(blocks), self.max_num_blocks_per_req)
        self.block_table_cpu[idx, start:end] = blocks[: end - start]
        self.num_blocks_per_row[idx] = end

    def _remove(self, rid: str) -> None:
        idx = self.req_id_to_index.pop(rid, None)
        if idx is None:
            return
        self._req_ids[idx] = None  # type: ignore[assignment]
        self._removed.append(idx)
        self.num_blocks_per_row[idx] = 0

    def _condense(self) -> None:
        if not self._removed:
            return
        active = len(self._req_ids)
        if active == 0:
            self._req_ids.clear()
            self._removed.clear()
            return
        holes = sorted(self._removed)
        last = len(self._req_ids) - 1
        for hole in holes:
            while last >= 0 and self._req_ids[last] is None:
                last -= 1
            if last <= hole:
                break
            rid = self._req_ids[last]
            assert rid is not None
            self._req_ids[hole] = rid
            self._req_ids[last] = None
            self.req_id_to_index[rid] = hole
            for buf in (
                self.num_computed_tokens_cpu,
                self.num_prompt_tokens_cpu,
                self.num_tokens_no_spec,
                self.num_blocks_per_row,
            ):
                buf[hole] = buf[last]
            n_tok = int(self.num_tokens_no_spec[last])
            self.token_ids_cpu[hole, :n_tok] = self.token_ids_cpu[last, :n_tok]
            self.block_table_cpu[hole] = self.block_table_cpu[last]
            last -= 1
        self._req_ids = [r for r in self._req_ids if r is not None]  # type: ignore[misc]
        self._removed.clear()
        self.req_id_to_index = {r: i for i, r in enumerate(self._req_ids)}

    # -- the two timed phases ------------------------------------------------
    def update_states(self, so: Any, rec: StepRecord) -> None:
        """Mirror ``_update_states`` (CPU half)."""
        for rid in _get(so, "finished_req_ids", []) or []:
            self._remove(rid)
            self.prompt_cache.pop(rid, None)

        scheduled = list((_get(so, "num_scheduled_tokens", {}) or {}).keys())
        crd = _get(so, "scheduled_cached_reqs", None)
        resumed = set(_cget(crd, "resumed_req_ids", []) or [])
        scheduled_set = set(scheduled) - resumed
        for rid in list(self._req_ids):
            if rid is not None and rid not in scheduled_set:
                self._remove(rid)

        for rd in _get(so, "scheduled_new_reqs", []) or []:
            rid = _get(rd, "req_id")
            prompt_ids = list(_get(rd, "prompt_token_ids", []) or [])
            self.prompt_cache[rid] = prompt_ids
            self._add(
                rid,
                prompt_ids,
                int(_get(rd, "num_computed_tokens", 0)),
                [list(g) for g in (_get(rd, "block_ids", []) or [])],
            )

        for i, rid in enumerate(_cget(crd, "req_ids", []) or []):
            idx = self.req_id_to_index.get(rid)
            new_blocks = (_cget(crd, "new_block_ids", []) or [None] * (i + 1))[i]
            if idx is None:
                self._add(
                    rid,
                    list(self.prompt_cache.get(rid, [])),
                    int((_cget(crd, "num_computed_tokens", []) or [0] * (i + 1))[i]),
                    [],
                )
                idx = self.req_id_to_index[rid]
            if rid in resumed:
                self.block_table_cpu[idx].fill(0)
                self.num_blocks_per_row[idx] = 0
            for group in new_blocks or []:
                self._append_blocks(idx, list(group))
            nct = _cget(crd, "num_computed_tokens", []) or []
            if i < len(nct):
                self.num_computed_tokens_cpu[idx] = int(nct[i])
        self._condense()

    def prepare_inputs(self, so: Any) -> None:
        """Mirror the CPU-side hot loops of ``_prepare_inputs``."""
        total = int(_get(so, "total_num_scheduled_tokens", 0) or 0)
        nst = _get(so, "num_scheduled_tokens", {}) or {}
        req_ids = self.req_ids
        num_reqs = len(req_ids)
        if num_reqs == 0 or total <= 0:
            return
        # batch-table copy to the device buffer (first thing _prepare_inputs does)
        self.block_table_gpu[:num_reqs] = self.block_table_cpu[:num_reqs]

        tokens = np.empty(num_reqs, dtype=np.int32)
        for i, rid in enumerate(req_ids):
            tokens[i] = nst[rid]
        req_indices = np.repeat(self.arange_np[:num_reqs], tokens)
        cu_num_tokens = np.cumsum(tokens, dtype=np.int32)
        offsets = np.repeat(cu_num_tokens - tokens, tokens)
        np.subtract(
            self.arange_np[:total], offsets, out=self.query_pos[:total]
        )
        positions_np = (
            self.num_computed_tokens_cpu[req_indices] + self.query_pos[:total]
        )
        token_indices = positions_np + req_indices * self.token_ids_cpu.shape[1]
        np.take(
            self.token_ids_cpu.reshape(-1),
            token_indices.astype(np.intp, copy=False),
            out=self.input_ids[:total],
        )
        # slot mapping (block-table lookup per token)
        block_ids = self.block_table_gpu[req_indices, positions_np // self.block_size]
        self.slot_mapping[:total] = (
            block_ids * self.block_size + positions_np % self.block_size
        )
        # logits indices + spec-decode metadata
        spec = _get(so, "scheduled_spec_decode_tokens", {}) or {}
        if not spec:
            self.logits_indices[:num_reqs] = cu_num_tokens - 1
        else:
            draft = np.zeros(num_reqs, dtype=np.int32)
            for rid, toks in spec.items():
                idx = self.req_id_to_index.get(rid)
                if idx is not None:
                    draft[idx] = len(toks)
            sampled = draft + 1
            cu_sampled = np.cumsum(sampled, dtype=np.int32)
            logits = np.repeat(cu_num_tokens - sampled, sampled)
            logits += self.arange_np[: int(cu_sampled[-1])]
            self.logits_indices[: len(logits)] = logits


def decode_step(rec: StepRecord) -> tuple[Any, float]:
    """JSON -> SchedulerOutput (timed).  Falls back to the dict form when the
    vllm import is unavailable, so structure/pyramid checks stay comparable."""
    t0 = time.perf_counter_ns()
    try:
        from pi_harness.trace.replay import decode_scheduler_output

        so = decode_scheduler_output(rec.scheduler_output)
        us = (time.perf_counter_ns() - t0) / 1000.0
        return so, us
    except Exception:
        us = (time.perf_counter_ns() - t0) / 1000.0
        return rec.scheduler_output, us


def run_trace(
    recs: list[StepRecord],
    *,
    executor: LocalPrepareInputExecutor | None = None,
    max_num_reqs: int = 256,
    max_model_len: int = 8192,
    block_size: int = 128,
    max_num_batched_tokens: int = 2048,
) -> list[dict]:
    """Replay one trace; returns per-step timing dicts (µs)."""
    import gc

    ex = executor or LocalPrepareInputExecutor(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    out: list[dict] = []
    # GC pauses would show up as per-step outliers; the real worker runs with a
    # frozen/quiet heap (vllm freezes the startup heap), so disable it here too.
    gc.disable()
    try:
        for rec in recs:
            if int(rec.scheduler_output.get("total_num_scheduled_tokens") or 0) == 0:
                # drain step: the real worker only runs _update_states
                t0 = time.perf_counter_ns()
                c0 = time.process_time_ns()
                so, decode_us = decode_step(rec)
                ex.update_states(so, rec)
                t1 = time.perf_counter_ns()
                c1 = time.process_time_ns()
                out.append(
                    {
                        "step_idx": rec.step_idx,
                        "phase": rec.phase,
                        "num_reqs": 0,
                        "total_num_scheduled_tokens": 0,
                        "decode_us": decode_us,
                        "update_states_us": (t1 - t0) / 1000.0,
                        "prepare_inputs_us": 0.0,
                        "total_us": (t1 - t0) / 1000.0,
                        "update_states_cpu_us": (c1 - c0) / 1000.0,
                        "prepare_inputs_cpu_us": 0.0,
                        "total_cpu_us": (c1 - c0) / 1000.0,
                        "triton_launch_us": 0.0,
                        "drain": True,
                    }
                )
                continue
            t0 = time.perf_counter_ns()
            c0 = time.process_time_ns()
            so, decode_us = decode_step(rec)
            t1 = time.perf_counter_ns()
            c1 = time.process_time_ns()
            ex.update_states(so, rec)
            t2 = time.perf_counter_ns()
            c2 = time.process_time_ns()
            ex.prepare_inputs(so)
            t3 = time.perf_counter_ns()
            c3 = time.process_time_ns()
            sig = shape_signature(rec)
            out.append(
                {
                    "step_idx": rec.step_idx,
                    "phase": rec.phase,
                    "num_reqs": sig["num_reqs"],
                    "total_num_scheduled_tokens": sig["total_num_scheduled_tokens"],
                    "decode_us": decode_us,
                    "update_states_us": (t2 - t1) / 1000.0,
                    "prepare_inputs_us": (t3 - t2) / 1000.0,
                    "total_us": (t3 - t0) / 1000.0,
                    # CPU time is immune to preemption by the other tenants of
                    # this shared host -> used as the primary A/B statistic
                    "update_states_cpu_us": (c2 - c1) / 1000.0,
                    "prepare_inputs_cpu_us": (c3 - c2) / 1000.0,
                    "total_cpu_us": (c3 - c0) / 1000.0,
                    "triton_launch_us": 0.0,
                    "drain": False,
                }
            )
    finally:
        gc.enable()
    return out


def _cget(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
