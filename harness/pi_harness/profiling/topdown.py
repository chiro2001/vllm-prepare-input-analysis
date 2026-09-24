#!/usr/bin/env python3
"""920B topdown 汇总：把 libkperfx 多组 split JSON 合并成四层分量。

口径（与 plan/COORDINATION.md 第 6 节一致）：

* 一次 full topdown = 9 个 preset 组（每组 <=8 个计数器，920B 硬件上限 8），
  每组跑**同一个负载**一遍（libkperfx split 语义），最后按 event config 合并；
* 每个组都记录 time_enabled / time_running / confidence，报告取最小值；
* level1（retiring / bad_spec / frontend_bound / backend_bound）来自第 1 组，
  backend 细分需要第 2 组，前端细分来自第 1 组，OOO stall 细分来自第 4/5 组。

用法：
    python3 -m pi_harness.profiling.topdown --split-dir DIR --out topdown.json
    python3 -m pi_harness.profiling.topdown --files a.json b.json --out topdown.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Iterable

from . import CALIBER_VERSION, ensure_kperfx_importable
from .envcap import write_json

LEVEL1_KEYS = ("retiring", "bad_spec", "frontend_bound", "backend_bound")
FRONTEND_KEYS = ("frontend_latency_bound", "frontend_bandwidth_bound")
BACKEND_KEYS = ("resource_bound", "core_bound", "mem_bound")
MEM_KEYS = ("mem_l1_bound", "mem_l2_bound", "mem_l3_dram_bound",
            "mem_mem_bound", "mem_store_bound")
OOO_KEYS = ("rob_stall_pct", "ptag_stall_pct", "mapq_stall_pct",
            "pcbuf_stall_pct", "other_stall_pct")


def _load(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def merge_counters(paths: Iterable[Path]) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """按 event config 合并多组计数，同时收每组置信度。"""
    merged: dict[int, int] = {}
    groups: list[dict[str, Any]] = []
    for p in paths:
        data = _load(p)
        for entry in data.get("entries", []):
            cfg = entry.get("config")
            if isinstance(cfg, str):
                cfg = int(cfg, 16)
            merged[int(cfg)] = merged.get(int(cfg), 0) + int(entry.get("delta", 0))
        groups.append({
            "file": p.name,
            "preset": data.get("preset"),
            "counter": data.get("counter"),
            "count": data.get("count"),
            "time_enabled": data.get("time_enabled"),
            "time_running": data.get("time_running"),
            "confidence": data.get("confidence"),
            "useronly_forced": data.get("useronly_forced"),
            "pmu": data.get("pmu"),
        })
    return merged, groups


def summarize(counters: dict[int, int], groups: list[dict[str, Any]]) -> dict[str, Any]:
    """计算 topdown 分量 + IPC + 自检信息。"""
    ensure_kperfx_importable()
    import kperfx  # type: ignore

    metrics = kperfx.topdown_920b(counters)
    c = {int(k): float(v) for k, v in counters.items()}
    cycles = c.get(0x0011, 0.0)
    retired = c.get(0x0008, 0.0)
    spec = c.get(0x001B, 0.0)

    level1 = {k: metrics.get(k) for k in LEVEL1_KEYS}
    level1_sum = sum(v for v in level1.values() if v is not None)

    confs = [g["confidence"] for g in groups if g.get("confidence") is not None]
    te = [g["time_enabled"] for g in groups if g.get("time_enabled") is not None]
    tr = [g["time_running"] for g in groups if g.get("time_running") is not None]

    notes: list[str] = []
    if len(groups) < 9:
        notes.append(f"只合并了 {len(groups)}/9 个 preset 组，缺失分量按 0 处理")
    if confs and min(confs) < 0.99:
        notes.append("存在 multiplex：time_running/time_enabled < 0.99，分量按缩放后计数计算")
    if abs(level1_sum - 100.0) > 1.0:
        notes.append(f"level1 四项之和 = {level1_sum:.2f}%，偏离 100%")

    out: dict[str, Any] = {
        "caliber_version": CALIBER_VERSION,
        "pmu": groups[0].get("pmu") if groups else None,
        "issue_width": 6,
        "groups_present": len(groups),
        "level1": level1,
        "level1_sum": level1_sum,
        "level2": {
            "frontend": {k: metrics.get(k) for k in FRONTEND_KEYS},
            "backend": {k: metrics.get(k) for k in BACKEND_KEYS},
        },
        "level3_mem": {k: metrics.get(k) for k in MEM_KEYS},
        "ooo_stall": {k: metrics.get(k) for k in OOO_KEYS},
        "ipc": (retired / cycles) if cycles else None,
        "raw": {
            "cpu_cycles": cycles,
            "inst_retired": retired,
            "inst_spec": spec,
            "bad_spec_inst": spec - retired,
            "dispatch_slots": 6.0 * cycles,
        },
        "counters": {hex(int(k)): int(v) for k, v in counters.items()},
        "confidence": {
            "per_group": groups,
            "min": min(confs) if confs else None,
            "time_enabled_sum": sum(te) if te else None,
            "time_running_sum": sum(tr) if tr else None,
            "ratio": (sum(tr) / sum(te)) if (te and tr and sum(te)) else None,
        },
        "notes": notes,
    }
    return out


def summarize_paths(paths: Iterable[Path]) -> dict[str, Any]:
    counters, groups = merge_counters(paths)
    return summarize(counters, groups)


def main() -> int:
    ap = argparse.ArgumentParser(description="920B topdown 汇总（libkperfx split 结果合并）")
    ap.add_argument("--split-dir", default=None, help="split 输出目录（读 split_*.json）")
    ap.add_argument("--files", nargs="*", default=None, help="显式文件列表")
    ap.add_argument("--out", default=None, help="输出 JSON")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.files:
        paths = [Path(f) for f in args.files]
    elif args.split_dir:
        split_dir = Path(args.split_dir)
        paths = [Path(p) for p in sorted(glob.glob(str(split_dir / "split_*.json")))]
        if not paths:
            print(json.dumps({
                "caliber_version": CALIBER_VERSION,
                "error": f"{split_dir} 里没有 split_*.json（该 pass 的 PMU 会话没起来）",
                "level1": None, "ipc": None,
            }, ensure_ascii=False))
            return 3
    if not paths:
        ap.error("需要 --split-dir 或 --files")

    out = summarize_paths(paths)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        write_json(args.out, out)
    else:
        print(text)
    print(json.dumps({"level1": out["level1"], "ipc": out["ipc"],
                      "confidence_min": out["confidence"]["min"],
                      "notes": out["notes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
