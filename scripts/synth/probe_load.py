#!/usr/bin/env python3
"""probe_load.py - controlled *CPython-side* synthetic load.

`prepare_input` in vLLM 0.26.0 is dominated by pure-CPython bookkeeping
(list/dict/attrs traffic, tiny tensor factories, per-request loops), so a
toolchain qualification that only exercises a C loop would mis-state where the
samples land.  This script gives perf a victim whose top frames are the
interpreter ones we care about (`list.append`, `dict.get`, `_PyEval_...`,
`builtin_*`), so the `perf record -g --call-graph dwarf` contract can be
validated against a realistic call tree.

Usage: probe_load.py [--duration-ms N] [--reqs R] [--tokens T] [--numpy]
"""

from __future__ import annotations

import argparse
import time


def emulate_request(i: int, tokens: int) -> dict:
    """Mirrors the per-request record shape prepare_input touches."""
    rec = {
        "req_id": i,
        "num_prompt_tokens": tokens,
        "num_computed_tokens": 0,
        "num_scheduled_tokens": tokens,
        "block_ids": [((i * 7 + b) % 4096) for b in range(tokens // 16 + 1)],
    }
    acc = 0
    for b in rec["block_ids"]:
        acc = (acc * 31 + b) & 0xFFFFFFFF
    rec["hash"] = acc
    return rec


def emulate_prepare(recs: list, sample: list) -> int:
    """list/dict traffic close to _prepare_inputs' inner loops."""
    total = 0
    for rec in recs:
        n = rec["num_scheduled_tokens"]
        total += n
        sample.append(rec["hash"] ^ n)
        if rec["num_computed_tokens"] == 0:
            sample.append(-1)
    return total


def emulate_embedding_gather(ids: list, dim: int, out: list) -> None:
    """Small per-token arithmetic - stands in for the tiny tensor math that
    prepare_input does on the CPU before the device kernels run."""
    for t in ids:
        base = (t * 2654435761) & 0xFFFFFFFF
        for d in range(dim):
            out[d] = (base ^ (d * 40503)) & 0xFFFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-ms", type=int, default=2000)
    ap.add_argument("--reqs", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--numpy", action="store_true",
                    help="add a numpy elementwise pass (needs numpy)")
    args = ap.parse_args()

    np = None
    arr = None
    if args.numpy:
        import numpy as _np  # noqa: PLC0415
        np = _np
        arr = np.zeros(4096, dtype=np.float32)

    deadline = time.monotonic() + args.duration_ms / 1000.0
    iters = 0
    total_tokens = 0
    while time.monotonic() < deadline:
        recs = [emulate_request(i, args.tokens) for i in range(args.reqs)]
        sample: list[int] = []
        total_tokens += emulate_prepare(recs, sample)
        ids = [r["hash"] % 4096 for r in recs]
        out = [0] * args.dim
        emulate_embedding_gather(ids, args.dim, out)
        if np is not None and arr is not None:
            arr += 1.0
            arr *= 0.999
        iters += 1
    print(f"probe_load.py done iters={iters} total_tokens={total_tokens} "
          f"reqs={args.reqs} tokens={args.tokens} numpy={bool(np)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
