#!/usr/bin/env python3
"""跨解释器 A/B 交替计时。

背景：这台开发机是 KVM 虚机，同一个解释器跑同一条命令，不同批次之间绝对值可以差 2×
（实测 cp314 纯胶水 1.82 µs vs 3.80 µs，两次都无并发干扰）。因此跨进程/跨二进制的比较
必须**交替多轮**再做统计，否则会把机器状态的漂移误判成版本差异。

用法：python cross_interp_ab.py --backend none --reps 7
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv/bin/python")

# (标签, 解释器, 额外环境变量)
CONFIGS = [
    ("cp312", VENV, {}),
    ("cp312-malloc", VENV, {"PYTHONMALLOC": "malloc"}),
    ("cp313", "/home/chiro/miniforge3/envs/.env313/bin/python", {}),
    ("cp314", "/home/chiro/miniforge3/envs/.env314/bin/python", {}),
    ("cp314-malloc", "/home/chiro/miniforge3/envs/.env314/bin/python", {"PYTHONMALLOC": "malloc"}),
    ("cp314-jit", "/home/chiro/miniforge3/envs/.env314/bin/python", {"PYTHON_JIT": "1"}),
    ("cp314t-gil", "/home/chiro/miniforge3/envs/.env314t/bin/python", {"PYTHON_GIL": "1"}),
    ("cp314t-nogil", "/home/chiro/miniforge3/envs/.env314t/bin/python", {"PYTHON_GIL": "0"}),
]


def one(py: str, env_extra: dict, backend: str, iters: int, rounds: int, arms: str) -> dict:
    cmd = ["taskset", "-c", "4-7", py, os.path.join(HERE, "bench_all_inproc.py"),
           "--backend", backend, "--no-ext", "--iters", str(iters), "--rounds", str(rounds)]
    if arms:
        cmd += ["--python-label", arms]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env.update(env_extra)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("py."):
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="none")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples: dict[str, dict[str, list[float]]] = {c[0]: {} for c in CONFIGS}
    for rep in range(args.reps):
        order = CONFIGS[rep % len(CONFIGS):] + CONFIGS[: rep % len(CONFIGS)]
        for label, py, env_extra in order:
            res = one(py, env_extra, args.backend, args.iters, args.rounds, label)
            for arm, us in res.items():
                samples[label].setdefault(arm, []).append(us)
        print(f"# rep {rep + 1}/{args.reps} done", file=sys.stderr, flush=True)

    print(f"backend={args.backend} reps={args.reps} iters={args.iters}/round {args.rounds} rounds")
    hdr = f"{'config':14s} {'n':>3s} " + " ".join(f"{a:>10s}" for a in
            ["py.plain", "py.slots", "py.manual_slots", "py.namedtuple", "py.dict"])
    print(hdr)
    table = {}
    for label in samples:
        row = {}
        for arm, xs in samples[label].items():
            row[arm] = statistics.median(xs)
        table[label] = row
        n = max((len(v) for v in samples[label].values()), default=0)
        cells = " ".join(f"{row.get(a, float('nan')):10.2f}" for a in
                         ["py.plain", "py.slots", "py.manual_slots", "py.namedtuple", "py.dict"])
        print(f"{label:14s} {n:3d} {cells}")
    print()
    base = table.get("cp312", {}).get("py.plain")
    print("相对 cp312/py.plain 的比值：")
    for label, row in table.items():
        if "py.plain" in row and base:
            print(f"  {label:14s} none-backend {row['py.plain'] / base:.3f}×")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"configs": {k: {a: xs for a, xs in v.items()} for k, v in samples.items()},
                       "medians": table}, f, indent=2)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
