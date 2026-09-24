"""Cross-check the synthetic worker-side state against the *real* ``InputBatch``.

The synthetic trace carries an emulated ``input_batch`` snapshot
(:mod:`pi_harness.workload.worker_state`).  This script feeds the very same
``SchedulerOutput`` sequence into a real ``NPUModelRunner`` (built by the
replay_harness runner: ``pi_harness.runner.build.build_runner``) and compares the
worker state after ``_update_states``:

* request order in the persistent batch,
* ``num_computed_tokens_cpu`` / ``num_prompt_tokens``,
* block-table row widths,
* the set of requests the worker keeps cached.

Usage (inside the no-card container)::

    python -m pi_harness.workload.crosscheck_worker_state \
        --trace data/harness/trace_synth_<tag>.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys

from pi_harness.trace.replay import decode_scheduler_output
from pi_harness.workload.schema import load_jsonl


def _snapshot_real(runner) -> dict:
    ib = runner.input_batch
    req_ids = list(ib.req_ids)
    widths = []
    tables = getattr(ib.block_table, "block_tables", [])
    for g, tbl in enumerate(tables):
        widths.append(
            [int(tbl.num_blocks_per_row[i]) for i in range(len(req_ids))]
        )
    return {
        "req_ids": req_ids,
        "num_computed_tokens_cpu": [
            int(ib.num_computed_tokens_cpu[i]) for i in range(len(req_ids))
        ],
        "num_prompt_tokens_cpu": [
            int(ib.num_prompt_tokens[i]) for i in range(len(req_ids))
        ],
        "num_blocks_per_row": widths,
        "requests_kept": sorted(getattr(runner, "requests", {}).keys()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-num-reqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--num-spec-tokens", type=int, default=0)
    ap.add_argument("--enable-prefix-caching", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    from pi_harness.runner.build import build_runner
    from pi_harness.runner.config import RunnerConfig

    cfg = RunnerConfig(
        max_num_reqs=args.max_num_reqs,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        max_num_batched_tokens=args.chunk_size,
        num_spec_tokens=args.num_spec_tokens,
        enable_prefix_caching=bool(args.enable_prefix_caching),
    )
    runner, meta = build_runner(cfg)

    recs = load_jsonl(args.trace)
    if args.limit:
        recs = recs[: args.limit]

    results = []
    n_ok = 0
    for rec in recs:
        so = decode_scheduler_output(rec.scheduler_output)
        runner._update_states(so)
        got = _snapshot_real(runner)
        want = rec.input_batch or {}
        cmp = {
            "step_idx": rec.step_idx,
            "req_ids_equal": got["req_ids"] == list(want.get("req_ids") or []),
            "req_ids_real": got["req_ids"],
            "req_ids_emul": list(want.get("req_ids") or []),
            "num_computed_equal": got["num_computed_tokens_cpu"]
            == [int(x) for x in (want.get("num_computed_tokens_cpu") or [])],
            "num_prompt_equal": got["num_prompt_tokens_cpu"]
            == [int(x) for x in (want.get("num_prompt_tokens_cpu") or [])],
            "block_widths_equal": got["num_blocks_per_row"]
            == [[int(v) for v in row] for row in (want.get("num_blocks_per_row") or [])],
            "real_requests": got["requests_kept"],
            "emul_requests": sorted(
                {s["req_id"] for s in (rec.req_states or [])}
            ),
        }
        ok = (
            cmp["req_ids_equal"]
            and cmp["num_computed_equal"]
            and cmp["num_prompt_equal"]
            and cmp["block_widths_equal"]
        )
        n_ok += int(ok)
        results.append(cmp)

    summary = {
        "trace": args.trace,
        "steps": len(results),
        "steps_matching": n_ok,
        "match_ratio": n_ok / len(results) if results else 0.0,
        "first_mismatch": next((r for r in results if not (
            r["req_ids_equal"]
            and r["num_computed_equal"]
            and r["num_prompt_equal"]
            and r["block_widths_equal"]
        )), None),
        "runner_meta": {k: v for k, v in (meta or {}).items() if k != "attr_audit"},
        "per_step": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_step"}, indent=2)[:3000])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.out}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
