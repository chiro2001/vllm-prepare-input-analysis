#!/usr/bin/env python3
"""dispatch_micro_bench.py -- pure opcode-dispatch sensitivity control.

The `pi_sim_bench.py` workload is a *mix* (attribute/IC traffic, small tensor
calls into numpy, allocation).  To bound how much of the measured difference
is attributable to the eval-loop dispatch mechanism itself, this script runs
tight, dispatch-dominated Python loops: many bytecodes per unit of user work,
almost no C-level work, no allocation.

If computed gotos help at all, the effect here must be >= the effect on the
mixed workload; it is the upper bound, not the prediction.

Kernels are fixed-iteration and deterministic; the digests are compared
across arms to prove identical work.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time


class Point:
    __slots__ = ("a", "b", "c")

    def __init__(self, a: int, b: int, c: int):
        self.a = a
        self.b = b
        self.c = c

    def norm(self) -> int:
        return self.a * self.a + self.b * self.b - self.c


def k_int_loop(n: int) -> int:
    """branchy integer loop: LOAD_FAST/COMPARE_JUMP/BINARY_OP heavy"""
    s = 0
    i = 0
    while i < n:
        s += i
        if s & 0xFFFF == 0:
            s -= 7
        i += 1
    return s


def k_attr_loop(n: int, p: Point) -> int:
    s = 0
    for _ in range(n):
        s += p.a + p.b - p.c
    return s


def k_call_loop(n: int, p: Point) -> int:
    s = 0
    for _ in range(n):
        s += p.norm()
    return s


def k_branch_loop(n: int) -> int:
    """data-dependent branches (branch predictor pressure in user code)"""
    s = 0
    for i in range(n):
        if i % 3 == 0:
            s += i
        elif i % 3 == 1:
            s -= i
        else:
            s ^= i
    return s


def k_global_loop(n: int, d: dict) -> int:
    s = 0
    for i in range(n):
        s += d.get(i, 0)
    return s


KERNELS = (
    ("int_loop", lambda n, ctx: k_int_loop(n)),
    ("attr_loop", lambda n, ctx: k_attr_loop(n, ctx["point"])),
    ("call_loop", lambda n, ctx: k_call_loop(n, ctx["point"])),
    ("branch_loop", lambda n, ctx: k_branch_loop(n)),
    ("global_loop", lambda n, ctx: k_global_loop(n, ctx["d"])),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    ctx = {"point": Point(3, 4, 5), "d": {i: i for i in range(1000)}}
    out: dict = {
        "label": args.label,
        "iters": args.iters,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "kernels": {},
    }
    digest = 0
    for name, fn in KERNELS:
        for _ in range(2):  # warmup / specialize
            digest += fn(args.iters, ctx)
        recs = []
        for rnd in range(args.rounds):
            gc.collect()
            t0 = time.perf_counter()
            acc = fn(args.iters, ctx)
            dt = time.perf_counter() - t0
            digest += acc
            recs.append({"round": rnd, "seconds": dt,
                         "ns_per_iter": dt / args.iters * 1e9})
        out["kernels"][name] = {
            "rounds": recs,
            "median_ns_per_iter": statistics.median(r["ns_per_iter"] for r in recs),
            "min_ns_per_iter": min(r["ns_per_iter"] for r in recs),
        }
    out["work_digest"] = digest

    text = json.dumps(out, indent=1, sort_keys=True)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
