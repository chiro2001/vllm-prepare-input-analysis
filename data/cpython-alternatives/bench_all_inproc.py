#!/usr/bin/env python3
"""主计时脚本：同一个进程里把"纯 Python / Cython(原样编译) / Cython(typed) / mypyc"
所有臂轮转交替计时，避免跨进程的漂移与首次调用效应。

输出 JSON + CSV；`ratio_vs_py_plain` 是相对同进程纯 Python dataclass 臂的比值。

用法：
  python bench_all_inproc.py --backend torch --iters 1500 --rounds 9 --out inproc-cp312-torch.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_glue as bg

OPTIONAL_MODULES = ["bench_glue_cy", "cy_variants", "mypyc_variant"]

VARIANT_NAMES = ["plain", "plain_reuse", "slots", "manual_dict", "manual_slots", "namedtuple", "dict"]
EXT_TARGETS = [
    ("bench_glue_cy", ["run_plain", "run_slots", "run_manual_dict", "run_manual_slots",
                       "run_namedtuple", "run_dict"]),
    ("cy_variants", ["run_cy_naive", "run_cy_cdef", "run_cy_cdef_notensor"]),
    ("mypyc_variant", ["run_mypyc_native", "run_mypyc_dataclass"]),
]

# 少数臂没有 mode 参数（对照臂），这里统一包一层，保证签名一致
WRAPPERS = {
    ("cy_variants", "run_cy_cdef_notensor"): lambda fn: (lambda inp, mode: fn(inp)),
}


def collect_arms(include_ext: bool) -> list[tuple[str, object]]:
    arms: list[tuple[str, object]] = [(f"py.{n}", bg.VARIANTS[n]) for n in VARIANT_NAMES]
    if not include_ext:
        return arms
    for mod_name, funcs in EXT_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:  # 没编译出来的臂直接跳过
            print(f"# skip {mod_name}: {exc}", file=sys.stderr)
            continue
        for fname in funcs:
            fn = getattr(mod, fname, None)
            if fn is not None:
                wrap = WRAPPERS.get((mod_name, fname))
                if wrap is not None:
                    fn = wrap(fn)
                arms.append((f"{mod_name}.{fname}", fn))
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "numpy", "none"], default="torch")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--python-label", default=None)
    ap.add_argument("--no-ext", action="store_true")
    args = ap.parse_args()

    arms = collect_arms(not args.no_ext)
    print(f"backend={args.backend} arms={len(arms)} iters={args.iters} rounds={args.rounds}",
          file=sys.stderr)
    py = sys.version.split()[0]
    label = args.python_label or f"py{py}"

    # 预热（覆盖全部臂）
    for _ in range(max(1, args.warmup // len(arms))):
        for _, fn in arms:
            fn(bg.INPUTS, args.backend)

    samples: dict[str, list[float]] = {name: [] for name, _ in arms}
    checks: dict[str, int] = {name: 0 for name, _ in arms}
    for r in range(args.rounds):
        order = arms[r % len(arms):] + arms[: r % len(arms)]  # 轮转顺序，消除次序偏差
        for name, fn in order:
            t0 = time.perf_counter_ns()
            acc = 0
            for _ in range(args.iters):
                acc += int(fn(bg.INPUTS, args.backend))
            t1 = time.perf_counter_ns()
            samples[name].append((t1 - t0) / args.iters)
            checks[name] += acc

    baseline = statistics.median(samples["py.plain"]) if "py.plain" in samples else None
    rows = []
    for name, _ in arms:
        xs = sorted(samples[name])
        med = statistics.median(xs)
        rows.append({
            "label": label,
            "arm": name,
            "backend": args.backend,
            "python": py,
            "python_impl": platform.python_implementation(),
            "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
            "iters": args.iters,
            "rounds": args.rounds,
            "median_ns": med,
            "min_ns": xs[0],
            "p10_ns": xs[max(0, int(0.1 * len(xs)) - 1)],
            "p90_ns": xs[min(len(xs) - 1, int(0.9 * len(xs)))],
            "us_per_iter": med / 1000.0,
            "ratio_vs_py_plain": (med / baseline) if baseline else None,
            "checksum": checks[name],
        })

    print(f"{'arm':32s} {'us/iter':>9s} {'ratio':>7s} {'min':>9s}")
    for r in rows:
        print(f"{r['arm']:32s} {r['us_per_iter']:9.2f} {r['ratio_vs_py_plain'] or 0:7.3f} "
              f"{r['min_ns'] / 1000:9.2f}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {args.out}")
    print(json.dumps({"python": py, "backend": args.backend,
                      "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
