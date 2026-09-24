#!/usr/bin/env python3
"""宿主机噪声门：采集前确认目标核上没有被别人占着的重负载。

判定口径（只读，不改任何系统配置）：

* 目标核忙度：/proc/stat 逐核两次快照，busy% = 1 - (idle+iowait)/total；
* 整机压力：/proc/loadavg 1 分钟均值 / 本进程可见逻辑核数；
* 抢核进程：ps 按 CPU 排序的现场快照（写进报告，人工可复核）。

用法：
    python3 -m pi_harness.profiling.hostnoise --cpus 200-215 --seconds 5 --json noise.json
    返回码 0 = 静默；1 = 噪声超阈值（仍写 json，调用方决定是否继续）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Sequence

from . import CALIBER_VERSION
from .envcap import (format_cpu_list, now_ts, parse_cpu_list, read_text,
                     top_procs, write_json)


def read_cpu_stat() -> dict[int, tuple[int, int]]:
    """返回 {cpu: (idle_all, total)}， idel_all = idle + iowait。"""
    out: dict[int, tuple[int, int]] = {}
    for line in (read_text("/proc/stat", "") or "").splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        parts = line.split()
        try:
            cpu = int(parts[0][3:])
            vals = [int(x) for x in parts[1:]]
        except (ValueError, IndexError):
            continue
        if len(vals) < 4:
            continue
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        out[cpu] = (idle, sum(vals))
    return out


def sample_busy(cpus: Sequence[int], seconds: float = 5.0) -> dict[int, float]:
    before = read_cpu_stat()
    time.sleep(seconds)
    after = read_cpu_stat()
    busy: dict[int, float] = {}
    for cpu in cpus:
        if cpu not in before or cpu not in after:
            continue
        d_idle = after[cpu][0] - before[cpu][0]
        d_tot = after[cpu][1] - before[cpu][1]
        busy[cpu] = 0.0 if d_tot <= 0 else max(0.0, min(100.0, 100.0 * (1 - d_idle / d_tot)))
    return busy


def gate(cpus: Sequence[int], seconds: float = 5.0, max_busy_pct: float = 5.0,
         max_load_per_cpu: float = 0.5) -> dict[str, Any]:
    loadavg = (read_text("/proc/loadavg", "") or "").split()
    busy = sample_busy(cpus, seconds)
    ncpu_online: int | None = None
    try:
        ncpu_online = len(os.sched_getaffinity(0))
    except Exception:  # pragma: no cover
        pass
    load1 = float(loadavg[0]) if loadavg else 0.0
    load_per_cpu = (load1 / ncpu_online) if ncpu_online else 0.0

    hot = {str(c): round(v, 2) for c, v in busy.items() if v > max_busy_pct}
    report: dict[str, Any] = {
        "caliber_version": CALIBER_VERSION,
        "ts_start": now_ts(),
        "cpus": list(cpus),
        "window_s": seconds,
        "busy_pct": {str(c): round(v, 2) for c, v in busy.items()},
        "busy_pct_max": round(max(busy.values()), 2) if busy else None,
        "busy_threshold_pct": max_busy_pct,
        "hot_cpus": hot,
        "loadavg": loadavg[:3],
        "load1_per_cpu": round(load_per_cpu, 3),
        "load_threshold_per_cpu": max_load_per_cpu,
        "top_procs": top_procs(),
        "ok": (not hot) and load_per_cpu <= max_load_per_cpu,
    }
    report["ts_end"] = now_ts()
    return report


def _siblings(cpu: int) -> list[int]:
    txt = read_text(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", "")
    return parse_cpu_list(txt or str(cpu))


def pick(pool: Sequence[int], n: int, seconds: float = 6.0,
         prefer_distinct_physical: bool = True) -> dict[str, Any]:
    """在给定核池里挑 n 个最安静的核（默认避开同一物理核的 SMT 兄弟）。"""
    busy = sample_busy(pool, seconds)
    ranked = sorted(busy, key=lambda c: (busy[c], c))
    chosen: list[int] = []
    taken_siblings: set[int] = set()
    for cpu in ranked:
        if len(chosen) >= n:
            break
        if prefer_distinct_physical and cpu in taken_siblings:
            continue
        chosen.append(cpu)
        taken_siblings.update(_siblings(cpu))
    chosen.sort()
    return {
        "caliber_version": CALIBER_VERSION,
        "ts": now_ts(),
        "pool": format_cpu_list(pool),
        "n_requested": n,
        "cpus": chosen,
        "cpuset": format_cpu_list(chosen),
        "prefer_distinct_physical": prefer_distinct_physical,
        "picked_busy_pct": {str(c): round(busy[c], 2) for c in chosen},
        "picked_busy_pct_max": round(max((busy[c] for c in chosen), default=0.0), 2),
        "pool_busy_pct": {str(c): round(v, 2) for c, v in sorted(busy.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="采集前噪声门（只读）")
    ap.add_argument("--cpus", default="200-215")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--max-busy-pct", type=float, default=5.0)
    ap.add_argument("--max-load-per-cpu", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-noisy", action="store_true", help="噪声超阈也返回 0")
    ap.add_argument("--pick", type=int, default=None, help="挑 N 个最安静的核并打印 cpuset")
    ap.add_argument("--pool", default="200-239,360-399", help="--pick 的候选核池")
    ap.add_argument("--allow-smt", action="store_true", help="允许挑到同一物理核的兄弟")
    args = ap.parse_args()

    if args.pick:
        rep_pick = pick(parse_cpu_list(args.pool), args.pick, args.seconds,
                        prefer_distinct_physical=not args.allow_smt)
        if args.json:
            write_json(args.json, rep_pick)
        print(json.dumps({k: rep_pick[k] for k in
                          ("cpuset", "picked_busy_pct", "picked_busy_pct_max")},
                         ensure_ascii=False))
        return 0

    rep = gate(parse_cpu_list(args.cpus), args.seconds,
               args.max_busy_pct, args.max_load_per_cpu)
    if args.json:
        write_json(args.json, rep)
    print(json.dumps({k: rep[k] for k in
                      ("ok", "busy_pct_max", "hot_cpus", "load1_per_cpu", "loadavg")},
                     ensure_ascii=False))
    return 0 if (rep["ok"] or args.allow_noisy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
