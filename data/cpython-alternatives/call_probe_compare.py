#!/usr/bin/env python3
"""同一进程内交替计时：Cython 编译的调用循环 vs CPython 字节码调用循环。"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch

import call_probe_py as py
import cy_call_probe as cy

N = 20000


def timed(fn) -> float:
    per_round = []
    for _ in range(7):
        t0 = time.perf_counter_ns()
        fn(N)
        t1 = time.perf_counter_ns()
        per_round.append((t1 - t0) / N)
    return statistics.median(per_round)


def main() -> None:
    a = np.array([1], dtype=np.int32)
    t = torch.from_numpy(a)
    cases = [
        ("np.cumsum(a)            ", lambda k: py.np_cumsum_fixed(a, k), lambda k: cy.np_cumsum_fixed(a, k)),
        ("np.zeros(1,dtype=)      ", lambda k: py.np_zeros_kw(k), lambda k: cy.np_zeros_kw(k)),
        ("torch.from_numpy(a)     ", lambda k: py.torch_from_numpy_fixed(a, k), lambda k: cy.torch_from_numpy_fixed(a, k)),
        ("tensor.mul(2)           ", lambda k: py.torch_add_mul_fixed(t, k), lambda k: cy.torch_add_mul_fixed(t, k)),
        ("f(a) via indirect arg   ", lambda k: py.call1(np.cumsum, a, k), lambda k: cy.call1(np.cumsum, a, k)),
        ("f(a,dtype=) indirect    ", lambda k: py.call_kw(np.zeros, 1, k), lambda k: cy.call_kw(np.zeros, 1, k)),
    ]
    print(f"N={N}/round, rounds=7, median ns/call")
    print(f"{'case':26s} {'cpython':>10s} {'cython':>10s} {'ratio':>8s}")
    for label, fpy, fcy in cases:
        p = timed(fpy)
        c = timed(fcy)
        print(f"{label:26s} {p:10.1f} {c:10.1f} {c / p:8.2f}x")
    # 反向顺序再测一次，排除顺序/漂移
    print("-- 反向顺序复测 --")
    for label, fpy, fcy in cases:
        c = timed(fcy)
        p = timed(fpy)
        print(f"{label:26s} {p:10.1f} {c:10.1f} {c / p:8.2f}x")


if __name__ == "__main__":
    raise SystemExit(main())
