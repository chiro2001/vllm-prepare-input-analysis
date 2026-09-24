#!/usr/bin/env python3
"""把"构造 / 读字段 / 算子调用"三段分别放在同一个进程里交替计时（消除跨进程噪声）。"""

from __future__ import annotations

import argparse
import statistics
import time

import bench_glue as bg


def timed(fn, n: int, rounds: int) -> float:
    """fn 是已经绑定好参数的零参可调用对象；返回 ns/iter 中位数。"""

    per_round = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        fn(n)
        t1 = time.perf_counter_ns()
        per_round.append((t1 - t0) / n)
    return statistics.median(per_round)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=9)
    args = ap.parse_args()
    I = bg.INPUTS
    n, r = args.iters, args.rounds

    import cy_probe as cp
    import cy_probe_py as pp

    cases = [
        ("ctor_only   py(slots)", lambda k: pp.py_ctor_only(I, k)),
        ("ctor_only   cy(cdef)", lambda k: cp.cy_ctor_only(I, k)),
        ("ctor+read   py(slots)", lambda k: pp.py_ctor_read(I, k)),
        ("ctor+read   cy(cdef)", lambda k: cp.cy_ctor_read(I, k)),
        ("ctor+read   cy(py-attr)", lambda k: cp.cy_ctor_read_pygattr(I, k)),
        ("ops         py", lambda k: pp.py_ops(k, "torch")),
        ("ops         cy(cdef-inline)", lambda k: cp.cy_ops(k, "torch")),
        ("ops         py", lambda k: pp.py_ops(k, "numpy")),
        ("ops         cy(cdef-inline)", lambda k: cp.cy_ops(k, "numpy")),
        ("ops         py", lambda k: pp.py_ops(k, "none")),
        ("ops         cy(cdef-inline)", lambda k: cp.cy_ops(k, "none")),
    ]
    print(f"{'case':32s} {'ns/iter':>12s}")
    for label, fn in cases:
        ns = timed(fn, n, r)
        print(f"{label:32s} {ns:12.1f}")

    print("\n-- 交替重复（检查顺序/漂移）--")
    for _ in range(2):
        print(f"  py ctor+read {timed(lambda k: pp.py_ctor_read(I, k), n, r):8.1f} ns | "
              f"cy ctor+read {timed(lambda k: cp.cy_ctor_read(I, k), n, r):8.1f} ns | "
              f"cy py-attr {timed(lambda k: cp.cy_ctor_read_pygattr(I, k), n, r):8.1f} ns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
