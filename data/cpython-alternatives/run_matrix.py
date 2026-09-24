#!/usr/bin/env python3
"""跑微基准矩阵：每个 (解释器, 变体, 张量后端) 组合单独起进程，结果汇总成 CSV。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(python: str, variant: str, backend: str, iters: int, rounds: int,
            label: str, env_extra: dict | None = None) -> dict:
    cmd = [python, os.path.join(HERE, "bench_glue.py"), "--variant", variant,
           "--tensor-backend", backend, "--iters", str(iters), "--rounds", str(rounds),
           "--label", label]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    if p.returncode != 0:
        return {"label": label, "variant": variant, "tensor_backend": backend,
                "error": (p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout) else "failed"}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--python-label", required=True)
    ap.add_argument("--variants", default="plain,slots,manual_dict,manual_slots,namedtuple,dict")
    ap.add_argument("--backends", default="torch,numpy,none")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for variant in args.variants.split(","):
        for backend in args.backends.split(","):
            label = f"{args.python_label}:{variant}:{backend}"
            r = run_one(args.python, variant, backend, args.iters, args.rounds, label)
            r["python_label"] = args.python_label
            r["python_bin"] = args.python
            rows.append(r)
            status = r.get("error") or f"{r.get('us_per_iter', float('nan')):.2f} us/iter"
            print(f"{label:55s} {status}", flush=True)

    fields = ["label", "python_label", "python_bin", "python", "python_impl", "variant",
              "tensor_backend", "iters", "rounds", "median_ns", "min_ns", "p10_ns", "p90_ns",
              "us_per_iter", "checksum", "numpy", "torch", "gil_enabled", "affinity", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["affinity"] = ",".join(str(x) for x in r.get("affinity") or [])
            w.writerow(r)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
