#!/usr/bin/env python3
"""跑编译产物（Cython / mypyc）里的某个函数，计时口径与 bench_glue.py 完全一致。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_glue  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--tensor-backend", choices=["torch", "numpy", "none"], default="torch")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    fn = getattr(mod, args.func)
    r = bench_glue.time_variant(fn, args.iters, args.rounds, args.tensor_backend)
    out = {
        "label": args.label or f"{args.module}.{args.func}",
        "variant": f"{args.module}.{args.func}",
        "tensor_backend": args.tensor_backend,
        "iters": args.iters,
        "rounds": args.rounds,
        "median_ns": r["median_ns"],
        "min_ns": r["min_ns"],
        "p10_ns": r["p10_ns"],
        "p90_ns": r["p90_ns"],
        "us_per_iter": r["median_ns"] / 1000.0,
        "checksum": r["checksum"],
        "python": sys.version.split()[0],
        "numpy": bench_glue.np.__version__,
        "torch": (bench_glue.torch.__version__ if bench_glue.torch is not None else None),
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
