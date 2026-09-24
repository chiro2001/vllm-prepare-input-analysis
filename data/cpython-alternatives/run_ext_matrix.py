#!/usr/bin/env python3
"""跑编译产物（Cython / mypyc）的矩阵：每个 (模块, 函数, 后端) 单独起进程。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = ["label", "python_label", "python_bin", "python", "python_impl", "variant",
          "tensor_backend", "iters", "rounds", "median_ns", "min_ns", "p10_ns", "p90_ns",
          "us_per_iter", "checksum", "numpy", "torch", "gil_enabled", "affinity", "error"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--python-label", required=True)
    ap.add_argument("--targets", required=True,
                    help="逗号分隔的 module:func 列表")
    ap.add_argument("--backends", default="torch,numpy,none")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    rows = []
    for target in args.targets.split(","):
        module, func = target.split(":")
        for backend in args.backends.split(","):
            label = f"{args.python_label}:{module}.{func}:{backend}"
            cmd = [args.python, os.path.join(HERE, "run_ext.py"), "--module", module,
                   "--func", func, "--tensor-backend", backend, "--iters", str(args.iters),
                   "--rounds", str(args.rounds), "--label", label]
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
            if p.returncode != 0:
                r = {"label": label, "tensor_backend": backend,
                     "error": (p.stderr or p.stdout).strip().splitlines()[-1]}
            else:
                r = json.loads(p.stdout.strip().splitlines()[-1])
            r["python_label"] = args.python_label
            r["python_bin"] = args.python
            rows.append(r)
            print(f"{label:60s} {r.get('error') or f'{r.get(chr(117)+chr(115)+chr(95)+chr(112)+chr(101)+chr(114)+chr(95)+chr(105)+chr(116)+chr(101)+chr(114), 0):.2f} us/iter'}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["affinity"] = ",".join(str(x) for x in r.get("affinity") or [])
            w.writerow(r)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
