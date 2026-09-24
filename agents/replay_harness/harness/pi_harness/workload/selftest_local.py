"""Card-free self test for the trace/workload data path.

Runs with plain CPython + numpy (no vllm, no torch, no NPU):

    PYTHONPATH=harness python -m pi_harness.workload.selftest_local

It covers

* pure-engine synthesis + JSONL round-trip (schema),
* the synthetic worker/InputBatch bookkeeping invariants,
* ``shuffle_trace`` (same shapes, different values),
* the local executor (timings + buffers),
* the capture hook against a *stub* model runner (monkeypatch, buffering,
  ``PI_CAPTURE_MAX_STEPS``), which is how the hook is validated before the shim
  / real runner exist,
* ``pi_harness.trace.replay`` decode when vllm happens to be importable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

from pi_harness.trace.capture import Capture
from pi_harness.workload import ab as AB
from pi_harness.workload import generate as G
from pi_harness.workload.local_exec import LocalPrepareInputExecutor, run_trace
from pi_harness.workload.schema import (
    StepRecord,
    load_jsonl,
    save_jsonl,
    shape_signature,
    summarize_trace,
)

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


# ---------------------------------------------------------------------------
# stub model runner (mimics NPUModelRunner's interface for the capture hook)
# ---------------------------------------------------------------------------


class _StubReqState:
    def __init__(self, rid: str, n: int) -> None:
        self.req_id = rid
        self.prompt_token_ids = list(range(n))
        self.num_computed_tokens = 0
        self.block_ids = ([1, 2],)
        self.output_token_ids: list[int] = []
        self.spec_token_ids: list[int] = []
        self.num_prompt_tokens = n
        self.num_tokens = n


class _StubBlockTable:
    def __init__(self, rows: int, width: int) -> None:
        self.block_table = np.zeros((rows, width), dtype=np.int32)
        self.num_blocks_per_row = np.zeros(rows, dtype=np.int32)

    @property
    def np(self) -> np.ndarray:  # mirrors the NumpyTensor wrapper
        return self.block_table


class _StubMultiGroupBlockTable:
    def __init__(self) -> None:
        self.block_tables = [_StubBlockTable(4, 8)]


class _StubInputBatch:
    def __init__(self) -> None:
        self.req_ids = ["stub-0", "stub-1"]
        self.num_computed_tokens_cpu = np.array([0, 5], dtype=np.int32)
        self.num_prompt_tokens_cpu = np.array([16, 16], dtype=np.int32)
        self.num_tokens_no_spec = np.array([16, 16], dtype=np.int32)
        self.token_ids_cpu = np.zeros((4, 64), dtype=np.int32)
        self.token_ids_cpu_tensor = self.token_ids_cpu
        self.block_table = _StubMultiGroupBlockTable()
        self.spec_token_ids = [[], []]


class _StubSchedulerOutput:
    def __init__(self, step: int) -> None:
        self.num_scheduled_tokens = {"stub-0": 16, "stub-1": 1}
        self.total_num_scheduled_tokens = 17
        self.scheduled_new_reqs: list = []
        self.scheduled_cached_reqs = _StubCachedReqData()
        self.scheduled_spec_decode_tokens: dict = {}
        self.scheduled_encoder_inputs: dict = {}
        self.num_common_prefix_blocks = [0]
        self.finished_req_ids: set = set()
        self.free_encoder_mm_hashes: list = []
        self.new_block_ids_to_zero = [7] if step % 2 == 0 else None
        self.kv_cache_block_copies = None
        self.num_invalid_spec_tokens = None
        self.preempted_req_ids = None
        self.num_spec_tokens_to_schedule = 0


class _StubCachedReqData:
    def __init__(self) -> None:
        self.req_ids = ["stub-1"]
        self.resumed_req_ids: set = set()
        self.new_token_ids: list = []
        self.all_token_ids: dict = {}
        self.new_block_ids = [([3],)]
        self.num_computed_tokens = [5]
        self.num_output_tokens = [0]


class _StubModelRunner:
    """Mirrors the real flow: execute_model() (which internally runs
    _update_states + _prepare_inputs and returns None) followed by a separate
    sample_tokens() call that produces the sampled ids."""

    def __init__(self) -> None:
        self.input_batch = _StubInputBatch()
        self.requests = {"stub-0": _StubReqState("stub-0", 16), "stub-1": _StubReqState("stub-1", 16)}
        self.calls = 0
        self.last_output: dict | None = None

    def _update_states(self, scheduler_output):
        return None

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        return None, None, int(scheduler_output.total_num_scheduled_tokens)

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self.calls += 1
        self._update_states(scheduler_output)
        self._prepare_inputs(
            scheduler_output, np.array([16, 1][: len(scheduler_output.num_scheduled_tokens)], dtype=np.int32)
        )
        self.last_output = {
            "req_ids": list(scheduler_output.num_scheduled_tokens),
            "req_id_to_index": {
                r: i for i, r in enumerate(scheduler_output.num_scheduled_tokens)
            },
            "sampled_token_ids": [[1] for _ in scheduler_output.num_scheduled_tokens],
        }
        return None  # split-sampling path: sample_tokens() produces the output

    def sample_tokens(self, grammar_output=None):
        return self.last_output


def test_synth_pure() -> list[StepRecord]:
    print("[1] pure synthesis + schema")
    cfg = G.SynthConfig(
        batch=4,
        isl=256,
        osl=8,
        block_size=64,
        chunk_size=256,
        engine="pure",
        max_model_len=1024,
        seed=1,
        prefix_hit_ratio=0.5,
        spec_k=0,
    )
    recs = G.synth_trace(cfg)
    check(len(recs) > 5, f"produced {len(recs)} steps")
    check(all(isinstance(r, StepRecord) for r in recs), "records are StepRecord")
    first = recs[0]
    so = first.scheduler_output
    for key in (
        "num_scheduled_tokens",
        "scheduled_new_reqs",
        "scheduled_cached_reqs",
        "scheduled_spec_decode_tokens",
        "finished_req_ids",
        "new_block_ids_to_zero",
        "kv_cache_block_copies",
        "num_common_prefix_blocks",
        "total_num_scheduled_tokens",
    ):
        check(key in so, f"scheduler_output has {key}")
    check(bool(first.input_batch and first.input_batch["req_ids"]), "input_batch snapshot")
    check(
        isinstance(so["num_scheduled_tokens"], dict)
        and all(isinstance(v, int) for v in so["num_scheduled_tokens"].values()),
        "num_scheduled_tokens is dict[str,int]",
    )
    summ = summarize_trace(recs)
    check(summ["num_steps"] == len(recs), "summarize_trace step count")
    return recs


def test_jsonl(recs: list[StepRecord]) -> None:
    print("[2] jsonl round trip")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.jsonl")
        n = save_jsonl(recs, path)
        back = load_jsonl(path)
        check(n == len(recs) == len(back), "line count")
        check(back[0].scheduler_output == recs[0].scheduler_output, "content identical")


def test_shuffle(recs: list[StepRecord]) -> None:
    print("[3] shuffle_trace: same shape, different values")
    sh = G.shuffle_trace(recs, seed=3)
    same = all(
        shape_signature(a) == shape_signature(b) for a, b in zip(recs, sh)
    )
    check(same, "shape signatures identical")
    tok_a = recs[0].req_states[0]["prompt_token_ids"]
    tok_b = sh[0].req_states[0]["prompt_token_ids"]
    check(bool(tok_a) and tok_a != tok_b, "token ids actually changed")
    ids_a = set(recs[0].scheduler_output["num_scheduled_tokens"])
    ids_b = set(sh[0].scheduler_output["num_scheduled_tokens"])
    check(ids_a and ids_b and not (ids_a & ids_b), "request ids renamed consistently")
    # join integrity: every req id in the scheduler output is present in req_states
    for rec in sh:
        so_ids = set(rec.scheduler_output["num_scheduled_tokens"])
        st_ids = {s["req_id"] for s in rec.req_states}
        if not so_ids <= st_ids:
            check(False, "shuffled ids still join across fields")
            break
    else:
        check(True, "shuffled ids still join across fields")


def test_local_exec(recs: list[StepRecord]) -> None:
    print("[4] local executor")
    ex = LocalPrepareInputExecutor(
        max_num_reqs=16, max_model_len=1024, block_size=64, max_num_batched_tokens=256
    )
    rows = run_trace(recs, executor=ex, max_num_reqs=16, max_model_len=1024,
                     block_size=64, max_num_batched_tokens=256)
    check(len(rows) == len(recs), "one row per step")
    check(all(r["total_us"] > 0 for r in rows), "positive timings")
    check(
        all(r["num_reqs"] > 0 for r in rows if not r["drain"]),
        "num_reqs > 0 on non-drain steps",
    )
    check(rows[-1]["drain"] is True, "trace ends with a drain step")
    stats = AB._percentiles([r["prepare_inputs_us"] for r in rows])
    check(stats["p50"] > 0, f"p50 = {stats['p50']:.1f} us")


def test_capture_hook() -> None:
    print("[5] capture hook (stub runner)")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cap.jsonl")
        cap = Capture(path, max_steps=3, flush_every=2, record_timing=True)
        cap.install(_StubModelRunner)
        mr = _StubModelRunner()
        for i in range(5):
            mr.execute_model(_StubSchedulerOutput(i))
            mr.sample_tokens(None)  # sampling is a separate call in the real engine
        cap.close()
        cap.uninstall()
        recs = load_jsonl(path)
        check(len(recs) == 3, f"max_steps honoured ({len(recs)} records)")
        check(cap.stats.steps_recorded == 3, "stats.steps_recorded")
        rec = recs[0]
        check(
            rec.scheduler_output["num_scheduled_tokens"] == {"stub-0": 16, "stub-1": 1},
            "scheduler output captured",
        )
        check(rec.input_batch is not None and rec.input_batch["num_reqs"] == 2,
              "input_batch snapshot captured")
        check(
            rec.timings is not None
            and "update_states_us" in rec.timings
            and "prepare_inputs_us" in rec.timings,
            "sub-step timings captured",
        )
        check(rec.t_prepare_input_us is not None, "t_prepare_input_us filled")
        check(rec.req_states and rec.req_states[0]["req_id"] == "stub-0",
              "req_states captured in batch order")
        check(all(r.model_output or r.input_batch for r in recs), "records usable")


def test_replay_decode(recs: list[StepRecord]) -> None:
    print("[6] replay decode")
    try:
        import vllm  # noqa: F401
    except Exception:
        print("  skip (vllm not importable here)")
        return
    from pi_harness.trace.replay import decode_scheduler_output_with_notes, decode_model_output

    so, notes = decode_scheduler_output_with_notes(recs[0].scheduler_output)
    check(so.total_num_scheduled_tokens == recs[0].total_num_scheduled_tokens,
          "decoded total tokens")
    check(isinstance(so.finished_req_ids, set), "finished_req_ids is a set")
    check(isinstance(so.scheduled_cached_reqs.resumed_req_ids, set), "resumed ids set")
    check(all(isinstance(b, tuple) for r in so.scheduled_new_reqs for b in [r.block_ids]),
          "block_ids is a tuple of lists")
    mo = decode_model_output(recs[0].model_output)
    check(mo is not None and isinstance(mo.sampled_token_ids, list), "model output decoded")
    check(not notes, f"no decode notes ({notes[:2]})")


def main() -> int:
    print(f"python={sys.version.split()[0]} numpy={np.__version__}")
    recs = test_synth_pure()
    test_jsonl(recs)
    test_shuffle(recs)
    test_local_exec(recs)
    test_capture_hook()
    test_replay_decode(recs)
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
