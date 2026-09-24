"""Emulation of the *worker-side* bookkeeping that consumes a ``SchedulerOutput``.

The synthetic generator runs the real scheduler on CPU (see ``synth_real.py``)
but has no ``InputBatch``; this module reproduces, field by field, the part of
``vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states`` and
``vllm.v1.worker.gpu_input_batch.InputBatch`` that decides

* which requests sit in the persistent batch and in which row order
  (``InputBatch.add_request`` / ``remove_request`` / ``condense``),
* ``num_computed_tokens_cpu`` / ``num_prompt_tokens`` / ``num_tokens_no_spec``,
* the per-request block lists (append vs replace-on-resume).

It is an emulation, not the real code: everything it produces is derived from
the real ``SchedulerOutput``, and the deviations are listed in the report's
fidelity table.  Row order after removals is the only place where a mistake
would silently change shapes, so it is checked against the captured trace
(``meta.engine=real`` vs a real capture) whenever both are available.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class WorkerBatch:
    """``InputBatch`` index bookkeeping (lowest free index first, then condense)."""

    def __init__(self) -> None:
        self.slots: list[str | None] = []
        self.removed: deque[int] = deque()

    # -- mirrors InputBatch._register_add_request / remove_request/condense --
    def add(self, req_id: str) -> int:
        if self.removed:
            idx = self.removed.popleft()
            self.slots[idx] = req_id
            return idx
        self.slots.append(req_id)
        return len(self.slots) - 1

    def remove(self, req_id: str) -> int | None:
        try:
            idx = self.slots.index(req_id)
        except ValueError:
            return None
        self.slots[idx] = None
        self.removed.append(idx)
        return idx

    def condense(self) -> None:
        if not self.removed:
            return
        active = sum(1 for s in self.slots if s is not None)
        if active == 0:
            self.slots.clear()
            self.removed.clear()
            return
        holes = sorted(self.removed)
        last = len(self.slots) - 1
        for hole in holes:
            while last >= 0 and self.slots[last] is None:
                last -= 1
            if last <= hole:
                break
            self.slots[hole] = self.slots[last]
            self.slots[last] = None
            last -= 1
        self.slots = [s for s in self.slots if s is not None]
        self.removed.clear()

    @property
    def req_ids(self) -> list[str]:
        return [s for s in self.slots if s is not None]

    def __contains__(self, req_id: str) -> bool:
        return req_id in self.slots


def new_req_state(
    req_id: str,
    prompt_token_ids: list[int] | None,
    *,
    num_computed_tokens: int = 0,
    block_ids: tuple[list[int], ...] | list[list[int]] | None = None,
    sampling_params: Any = None,
    num_groups: int = 1,
) -> dict:
    prompt_len = len(prompt_token_ids or [])
    blocks = [list(g) for g in (block_ids or [])]
    while len(blocks) < num_groups:
        blocks.append([])
    return {
        "req_id": req_id,
        "prompt_token_ids": list(prompt_token_ids) if prompt_token_ids else None,
        "num_prompt_tokens": prompt_len,
        "num_computed_tokens": int(num_computed_tokens),
        "block_ids": blocks,
        "num_output_tokens": 0,
        "spec_len": 0,
        "sampling_params": sampling_params,
    }


def apply_step(
    so: dict,
    batch: WorkerBatch,
    states: dict[str, dict],
    *,
    num_groups: int = 1,
) -> None:
    """Apply one ``SchedulerOutput`` (dict form) to the emulated worker state.

    Must be called *before* snapshotting, i.e. it plays the role of
    ``_update_states``; the state it leaves behind is what ``_prepare_inputs``
    reads in the real worker.
    """
    # 1) finished requests: cached state dropped, row freed
    for rid in so.get("finished_req_ids") or []:
        states.pop(rid, None)
        batch.remove(rid)

    # 2) requests in the batch but not scheduled this step leave the batch
    #    (their cached state is kept for a later step)
    scheduled = set((so.get("num_scheduled_tokens") or {}).keys())
    resumed = set((so.get("scheduled_cached_reqs") or {}).get("resumed_req_ids") or [])
    for rid in list(batch.req_ids):
        if rid not in scheduled - resumed:
            batch.remove(rid)

    # 3) new requests -> CachedRequestState + persistent batch row
    for rd in so.get("scheduled_new_reqs") or []:
        rid = rd["req_id"]
        states[rid] = new_req_state(
            rid,
            rd.get("prompt_token_ids"),
            num_computed_tokens=int(rd.get("num_computed_tokens", 0)),
            block_ids=rd.get("block_ids") or [],
            sampling_params=rd.get("sampling_params"),
            num_groups=num_groups,
        )
        batch.add(rid)

    # 4) cached requests: counters + block list, re-add if they left the batch
    crd = so.get("scheduled_cached_reqs") or {}
    req_ids = list(crd.get("req_ids") or [])
    for i, rid in enumerate(req_ids):
        st = states.get(rid)
        if st is None:
            st = new_req_state(rid, None, num_groups=num_groups)
            states[rid] = st
        new_blocks = (crd.get("new_block_ids") or [None] * (i + 1))[i]
        if rid in resumed:
            st["block_ids"] = [list(g) for g in (new_blocks or [])]
        elif new_blocks:
            while len(st["block_ids"]) < len(new_blocks):
                st["block_ids"].append([])
            for g, blocks in enumerate(new_blocks):
                st["block_ids"][g] = list(st["block_ids"][g]) + list(blocks)
        nct = crd.get("num_computed_tokens") or []
        if i < len(nct):
            st["num_computed_tokens"] = int(nct[i])
        not_ = crd.get("num_output_tokens") or []
        if i < len(not_) and int(not_[i]) < st["num_output_tokens"]:
            # output tokens were discarded (sync KV-load failure / optimistic
            # spec-decode rollback): align the cached state
            st["num_output_tokens"] = int(not_[i])
        st["spec_len"] = len((so.get("scheduled_spec_decode_tokens") or {}).get(rid, []))
        if rid not in batch:
            batch.add(rid)

    batch.condense()


def snapshot_req_states(
    batch: WorkerBatch,
    states: dict[str, dict],
    *,
    include_prompts_for: set[str] | None = None,
) -> list[dict]:
    """``req_states`` for the trace, in worker batch order.

    ``prompt_token_ids`` is emitted only for the requests that enter the worker
    batch in this step (``include_prompts_for``); for the others it is ``None``
    and the reader is expected to have cached it.  That keeps traces small
    (a 32 x 2 k prompt repeated in every step would otherwise dominate the
    file) without losing information: the same ids also appear in
    ``scheduled_new_reqs[].prompt_token_ids`` of the entry step.
    """
    out = []
    for rid in batch.req_ids:
        st = states.get(rid)
        if st is None:
            continue
        first_seen = include_prompts_for is None or rid in include_prompts_for
        out.append(
            {
                "req_id": rid,
                "prompt_token_ids": (
                    [int(t) for t in (st.get("prompt_token_ids") or [])]
                    if first_seen and st.get("prompt_token_ids") is not None
                    else None
                ),
                "num_prompt_tokens": int(st.get("num_prompt_tokens", 0)),
                "num_computed_tokens": int(st.get("num_computed_tokens", 0)),
                "block_ids": [list(g) for g in st.get("block_ids") or []],
                "num_output_tokens": int(st.get("num_output_tokens", 0)),
                "spec_len": int(st.get("spec_len", 0)),
            }
        )
    return out


def snapshot_input_batch(
    batch: WorkerBatch,
    states: dict[str, dict],
    *,
    token_ids_shape: tuple[int, int],
    num_groups: int = 1,
) -> dict:
    """``InputBatch`` snapshot with the fields ``_prepare_inputs`` reads."""
    req_ids = batch.req_ids
    block_table: list[list[list[int]]] = []
    num_blocks_per_row: list[list[int]] = []
    for g in range(num_groups):
        rows = []
        widths = []
        for rid in req_ids:
            st = states.get(rid) or {}
            blocks = (st.get("block_ids") or [[]] * (g + 1))
            blocks = list(blocks[g]) if g < len(blocks) else []
            rows.append(blocks)
            widths.append(len(blocks))
        block_table.append(rows)
        num_blocks_per_row.append(widths)
    return {
        "req_ids": req_ids,
        "num_computed_tokens_cpu": [
            int((states.get(r) or {}).get("num_computed_tokens", 0)) for r in req_ids
        ],
        "num_prompt_tokens_cpu": [
            int((states.get(r) or {}).get("num_prompt_tokens", 0)) for r in req_ids
        ],
        "num_tokens_no_spec": [
            int((states.get(r) or {}).get("num_prompt_tokens", 0))
            + int((states.get(r) or {}).get("num_output_tokens", 0))
            for r in req_ids
        ],
        "block_table": block_table,
        "num_blocks_per_row": num_blocks_per_row,
        "token_ids_cpu_shape": list(token_ids_shape),
        "token_ids_cpu_dtype": "torch.int32",
        "spec_token_ids": [
            [0] * int((states.get(r) or {}).get("spec_len", 0)) for r in req_ids
        ],
        "num_reqs": len(req_ids),
        "emulated": True,
    }


def record_sampled_tokens(
    so: dict, states: dict[str, dict], per_req_sampled: dict[str, int]
) -> None:
    """Worker bookkeeping after sampling: output tokens are appended and the
    spec lengths reset (mirrors ``_bookkeeping_sync``/``update_async_output``)."""
    for rid, n in per_req_sampled.items():
        st = states.get(rid)
        if st is None:
            continue
        st["num_output_tokens"] = int(st.get("num_output_tokens", 0)) + int(n)
        st["num_tokens_no_spec"] = int(st.get("num_prompt_tokens", 0)) + int(
            st["num_output_tokens"]
        )
        st["spec_len"] = 0
