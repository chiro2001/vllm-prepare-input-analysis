#!/usr/bin/env python3
"""CPU 忙循环探针：混合算术 + memcpy + sha256，用于把 perf/PMU 链路打通。

用法: python3 busy.py [--seconds 10] [--mode arith|memcpy|hash|mixed]
"""
import argparse
import hashlib
import time


def arith(n: int) -> int:
    acc = 0
    for i in range(n):
        acc = (acc * 1103515245 + 12345 + i) & 0x7FFFFFFF
    return acc


def memcpy(n: int, blk: int = 1 << 16) -> int:
    src = bytearray((i * 7) & 0xFF for i in range(blk))
    dst = bytearray(blk)
    for _ in range(n):
        dst[:] = src
        dst[0] = (dst[0] + 1) & 0xFF
    return dst[0]


def hashing(n: int) -> int:
    h = b"seed"
    for _ in range(n):
        h = hashlib.sha256(h).digest()
    return h[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--mode", default="mixed",
                    choices=["arith", "memcpy", "hash", "mixed"])
    args = ap.parse_args()

    t0 = time.time()
    total = 0
    while time.time() - t0 < args.seconds:
        if args.mode in ("arith", "mixed"):
            total += arith(2_000_000)
        if args.mode in ("memcpy", "mixed"):
            total += memcpy(512)
        if args.mode in ("hash", "mixed"):
            total += hashing(200_000)
    print(f"mode={args.mode} elapsed={time.time() - t0:.2f}s total={total}")


if __name__ == "__main__":
    main()
