#!/usr/bin/env python3
"""NUMA 局部性与 IPC 测量（无卡，纯 CPU）。

两个子命令：

1. `numa`  —— 内存受限算子（gather / repeat / index_select）在**不同
   (CPU node × 内存 node) 组合**下的耗时。绑定由外部 `numactl` 完成：

       for cpu in 1 5; do for mem in 1 5; do
         numactl --cpunodebind=$cpu --membind=$mem python microbench_numa_ipc.py numa \
           --label "cpu$cpu-mem$mem" --out data/model/microbench-numa.csv
       done; done

2. `ipc`   —— 把 `PrepareInputReplica` 的典型工况循环跑 N 秒，
   用于外部 `perf stat -e cycles,instructions` / libkperfx topdown 采集。

       perf stat -e cycles,instructions,task-clock \
         taskset -c 200-203 python microbench_numa_ipc.py ipc --seconds 5 --B 64 --T 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time

import numpy as np

import microbench_prepare_input as mb

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _timed(fn, reps: int = 30, rounds: int = 7) -> dict:
    return mb.bench(fn, repeats=reps, rounds=rounds)


def cmd_numa(args) -> None:
    T = args.tokens
    B = args.batch
    mult = args.src_mult
    flat = np.arange(T * mult, dtype=np.int64)
    idx = (np.arange(T, dtype=np.int64) * 7919) % (T * mult - 1)
    buf = np.empty(T, dtype=np.int64)
    src32 = np.arange(T * mult, dtype=np.int32)
    rows = []
    rows.append({"label": args.label, "name": f"__setup__(src={T * mult * 8 / 1e6:.0f}MB)",
                 "elements": 0, "unit": "meta", "median_ns": 0.0, "p10_ns": 0.0,
                 "p90_ns": 0.0, "min_ns": 0.0, "median_us": 0.0, "p10_us": 0.0,
                 "p90_us": 0.0, "repeats": 0, "rounds": 0, "ns_per_element": 0.0,
                 "gbytes_per_s": 0.0})
    rows.pop()  # 仅占位：保持 CSV 列序一致，不写出 meta 行

    def add(name, fn, elems, unit="element"):
        st = _timed(fn)
        r = {"label": args.label, "name": name, "elements": elems, "unit": unit, **st}
        r["ns_per_element"] = st["median_ns"] / elems
        r["gbytes_per_s"] = elems * 8 / st["median_ns"]   # 粗估（8B/elem 读写折算）
        rows.append(r)

    add("np.gather(int64[B*T], idx[T])", lambda: flat[idx], T)
    add("np.repeat(arange[T], ones[T]) int64", lambda: np.repeat(np.arange(T, dtype=np.int64), 1), T)
    add("np.subtract(arange[T], 0, out)", lambda: np.subtract(np.arange(T, dtype=np.int64), 0, out=buf), T)
    add("np.take(int32[4T], idx[T]) alloc", lambda: np.take(src32, idx), T)
    if torch is not None:
        tflat = torch.from_numpy(src32)
        tidx = torch.from_numpy(idx)
        tout = torch.empty(T, dtype=torch.int32)
        add("torch.index_select(int32[4T], idx[T], out)",
            lambda: torch.index_select(tflat, 0, tidx, out=tout), T)

    write = not args.no_write
    if write and rows:
        exists = os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            keys = list(rows[0].keys())
            w = csv.writer(f)
            if not exists:
                w.writerow(keys)
            for r in rows:
                w.writerow([f"{r[k]:.3f}" if isinstance(r[k], float) else r[k] for k in keys])
        print(f"appended {len(rows)} rows -> {args.out}")
    for r in rows:
        print(f"[{r['label']}] {r['name'][:42]:<44} median={r['median_us']:9.2f}us "
              f"ns/el={r['ns_per_element']:7.3f} ~{r['gbytes_per_s']:6.2f} GB/s(8B/elem)")


def cmd_ipc(args) -> None:
    """为外部 perf/topdown 提供稳定的 CPU 负载窗口。"""
    rep = mb.PrepareInputReplica(max_num_reqs=args.max_num_reqs,
                                 max_num_tokens=max(args.tokens, 1024),
                                 max_model_len=args.max_model_len,
                                 num_layers=args.num_layers)
    B, T = args.batch, args.tokens
    steps = 0
    t_end = time.perf_counter() + args.seconds
    while time.perf_counter() < t_end:
        for _ in range(200):          # 内层批量减少计时调用开销
            rep.step(B, T)
        steps += 200
    elapsed = time.perf_counter() - (t_end - args.seconds)
    meta = {
        "batch": B, "tokens": T, "steps": steps, "elapsed_s": round(elapsed, 3),
        "us_per_step": round(elapsed / steps * 1e6, 3),
        "affinity": sorted(os.sched_getaffinity(0))[:4],
        "torch": getattr(torch, "__version__", None),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "label": args.label,
    }
    print(json.dumps(meta, ensure_ascii=False))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("numa")
    p1.add_argument("--label", required=True)
    p1.add_argument("--tokens", type=int, default=32768)
    p1.add_argument("--batch", type=int, default=256)
    p1.add_argument("--src-mult", type=int, default=4,
                    help="源数组长度 = tokens × src_mult（调大以强制 DRAM 访问）")
    p1.add_argument("--out", default="data/model/microbench-numa.csv")
    p1.add_argument("--no-write", action="store_true")
    p1.set_defaults(func=cmd_numa)

    p2 = sub.add_parser("ipc")
    p2.add_argument("--seconds", type=float, default=5.0)
    p2.add_argument("--batch", type=int, default=64)
    p2.add_argument("--tokens", type=int, default=64)
    p2.add_argument("--max-num-reqs", type=int, default=256)
    p2.add_argument("--max-model-len", type=int, default=8192)
    p2.add_argument("--num-layers", type=int, default=28)
    p2.add_argument("--label", default="decode64")
    p2.add_argument("--out", default="")
    p2.set_defaults(func=cmd_ipc)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
