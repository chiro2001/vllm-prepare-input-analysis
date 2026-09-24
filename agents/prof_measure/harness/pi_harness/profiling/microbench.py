#!/usr/bin/env python3
"""无卡 CPU 标定 microbenchmark（纯 CPU，固定工作量，可复现）。

用途：作为"无卡 harness vs 真机"对照的锚点。它不依赖 NPU / torch，
只用 CPython + libc，因此任何装了 python3 的机器（含真机容器）都能跑出
同一套 IPC / topdown / 热点函数，用来证明采集链路本身可信。

三个子负载（覆盖 topdown 的不同象限）：

* ``arith``  —— 纯解释器整数运算：frontend/retiring 高，几乎无访存；
* ``memcpy`` —— bytearray 块拷贝：backend memory-bound；
* ``hash``   —— hashlib.sha256（C 实现，>2KB 时释放 GIL）：retiring 高、IPC 稳定，
                多线程时是唯一能真正并行扩核的相位。

固定工作量模式（``--iters``）保证 9 组 topdown pass 做的活完全一样；
``--seconds`` 模式适合快速冒烟。

用法：
    python3 -m pi_harness.profiling.microbench --phase arith --iters 30000000
    python3 -m pi_harness.profiling.microbench --phase all --target-seconds 20 --json out.json
    python3 -m pi_harness.profiling.microbench --phase hash --iters 20000 --threads 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import CALIBER_VERSION

# CPython 解释器里 1 次 arith 迭代 ≈ 6 条 bytecode
# 数值按 a3-22 容器实测标定到「每相位约 4 秒」（python3.12.13，单线程）
DEFAULT_ITERS = {"arith": 20_000_000, "memcpy": 3_000_000, "hash": 350_000}
BYTECODES_PER_ARITH_ITER = 6


def _arith(n: int) -> int:
    acc = 0
    mask = 0x7FFFFFFF
    for i in range(n):
        acc = (acc * 1103515245 + 12345 + i) & mask
        acc ^= (acc >> 7)
    return acc


def _memcpy(n: int, block: int = 1 << 16) -> int:
    src = bytearray((i * 7) & 0xFF for i in range(block))
    dst = bytearray(block)
    for _ in range(n):
        dst[:] = src          # C 层 memcpy，64KiB 超出 L1 落 L2
        dst[0] = (dst[0] + 1) & 0xFF
    return dst[0]


def _hash(n: int, block: int = 1 << 14) -> int:
    buf = bytes((i * 13) & 0xFF for i in range(block))
    h = hashlib.sha256(buf).digest()
    for _ in range(n):
        h = hashlib.sha256(h + buf).digest()   # 16KiB > 2KB，释放 GIL
    return h[0]


PHASES: dict[str, tuple[Callable[..., int], int]] = {
    "arith": (_arith, BYTECODES_PER_ARITH_ITER),
    "memcpy": (_memcpy, 65536 // 8),
    "hash": (_hash, 16384),
}


def _estimate_iters(phase: str, target_seconds: float) -> int:
    """粗测一把，估算达到 target_seconds 需要的迭代数。"""
    fn = PHASES[phase][0]
    probe = max(1000, DEFAULT_ITERS[phase] // 10)
    t0 = time.perf_counter()
    fn(probe)
    dt = time.perf_counter() - t0
    if dt <= 0:
        return DEFAULT_ITERS[phase]
    return max(1, int(probe * target_seconds / dt))


def run_phase(phase: str, iters: int, threads: int = 1) -> dict[str, Any]:
    fn, unit = PHASES[phase]
    per = max(1, iters // max(1, threads))
    t0 = time.perf_counter()
    if threads <= 1:
        checksum = fn(per)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            checksum = sum(ex.map(lambda _: fn(per), range(threads)))
    dt = time.perf_counter() - t0
    ops = per * threads
    return {
        "phase": phase,
        "iters": iters,
        "iters_per_thread": per,
        "threads": threads,
        "elapsed_s": round(dt, 6),
        "ops": ops,
        "ops_per_s": (ops / dt) if dt > 0 else None,
        "units": ops * unit,
        "unit": "bytecodes" if phase == "arith" else "bytes",
        "checksum": int(checksum),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="无卡 CPU 标定 microbenchmark")
    ap.add_argument("--phase", default="all",
                    choices=["all", "arith", "memcpy", "hash"])
    ap.add_argument("--iters", type=int, default=None, help="固定迭代数（确定性模式）")
    ap.add_argument("--target-seconds", type=float, default=None,
                    help="按目标秒数自动估算迭代数（非确定性）")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1, help="每相位预热轮数")
    ap.add_argument("--gate", default=None,
                    help="门控文件：出现后才开始计时（配合 STOP/CONT 采集）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    phases = list(PHASES) if args.phase == "all" else [args.phase]
    if args.gate:
        while not os.path.exists(args.gate):
            time.sleep(0.005)

    results: list[dict[str, Any]] = []
    for phase in phases:
        iters = args.iters
        if iters is None:
            ts = args.target_seconds if args.target_seconds else 5.0
            iters = _estimate_iters(phase, ts)
        for _ in range(max(0, args.warmup)):
            run_phase(phase, max(1, iters // 50), args.threads)
        results.append(run_phase(phase, iters, args.threads))

    out = {
        "caliber_version": CALIBER_VERSION,
        "tool": "microbench",
        "python": sys.version.split()[0],
        "threads": args.threads,
        "iters_mode": "fixed" if args.iters is not None else "auto_seconds",
        "phases": results,
    }
    if args.json:
        from .envcap import write_json
        write_json(args.json, out)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
