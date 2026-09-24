#!/usr/bin/env python3
"""两份采集结果的热点/topdown 对照（无卡内部 A/B，或与真机历史值对照）。

用法：
    # harness 污染量化：noop vs cpu_fallback
    python3 -m pi_harness.profiling.compare --a noop_summary.json --b fallback_summary.json \\
        --out data/harness/prof_cmp_noop_vs_fallback.json
    # 与真机历史区间对照（IPC / frontend-bound 区间由调用方给定）
    python3 -m pi_harness.profiling.compare --a noop_summary.json \\
        --ref-ipc 0.719:0.890 --ref-frontend 56:65
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import CALIBER_VERSION
from .envcap import write_json


def _ipc(summary: dict[str, Any]) -> float | None:
    """summary 里的 IPC：顶层没有就退回 pmu.ipc（collect.py 只填后者）。"""
    v = summary.get("ipc")
    if v is None:
        v = (summary.get("pmu") or {}).get("ipc")
    if v is None:
        v = (summary.get("topdown") or {}).get("ipc")
    return v


def hotmap(summary: dict[str, Any], n: int = 20) -> dict[str, float]:
    """{符号: overhead%}，取前 n 个（去掉 [.] 前缀方便比对）。"""
    rows = (summary.get("hotspots_top") or [])[:n]
    out: dict[str, float] = {}
    for r in rows:
        sym = str(r.get("symbol", "")).strip()
        for pre in ("[.] ", "[k] ", "[.]", "[k]"):
            if sym.startswith(pre):
                sym = sym[len(pre):]
        out[sym] = float(r.get("overhead_pct", 0.0))
    return out


def compare_hotspots(a: dict[str, Any], b: dict[str, Any], n: int = 20) -> dict[str, Any]:
    ha, hb = hotmap(a, n), hotmap(b, n)
    inter = sorted(set(ha) & set(hb), key=lambda s: -ha[s])
    return {
        "n": n,
        "a_tag": a.get("tag"),
        "b_tag": b.get("tag"),
        "a_top": list(ha),
        "b_top": list(hb),
        "intersection": inter,
        "overlap_ratio": (len(inter) / len(ha)) if ha else None,
        "only_a": [s for s in ha if s not in hb],
        "only_b": [s for s in hb if s not in ha],
        "shared_pct": {s: {"a": round(ha[s], 2), "b": round(hb[s], 2)} for s in inter},
    }


def _fmt_range(lo: float | None, hi: float | None) -> str | None:
    if lo is None or hi is None:
        return None
    return f"{lo}..{hi}"


def check_reference(summary: dict[str, Any], *, ref_ipc: tuple | None,
                    ref_frontend: tuple | None) -> dict[str, Any]:
    """与真机历史区间对照：给出是否落在区间 + 偏离量。"""
    td = summary.get("topdown") or {}
    l1 = td.get("level1") or {}
    ipc = _ipc(summary)
    fe = l1.get("frontend_bound")
    out: dict[str, Any] = {"ref_ipc": _fmt_range(*(ref_ipc or (None, None))),
                           "ref_frontend_bound": _fmt_range(*(ref_frontend or (None, None))),
                           "measured_ipc": ipc, "measured_frontend_bound": fe}
    if ref_ipc and ipc is not None:
        out["ipc_in_range"] = ref_ipc[0] <= ipc <= ref_ipc[1]
        out["ipc_delta"] = round(ipc - (ref_ipc[0] + ref_ipc[1]) / 2, 4)
    if ref_frontend and fe is not None:
        out["frontend_in_range"] = ref_frontend[0] <= fe <= ref_frontend[1]
        out["frontend_delta_pp"] = round(fe - (ref_frontend[0] + ref_frontend[1]) / 2, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="采集结果对照（无卡 A/B 或真机区间）")
    ap.add_argument("--a", required=True, help="A 的 summary.json")
    ap.add_argument("--b", default=None, help="B 的 summary.json（可选）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--ref-ipc", default=None, help="真机 IPC 区间，如 0.719:0.890")
    ap.add_argument("--ref-frontend", default=None, help="真机 frontend-bound %% 区间，如 56:65")
    args = ap.parse_args()

    A = json.loads(Path(args.a).read_text())
    out: dict[str, Any] = {"caliber_version": CALIBER_VERSION, "a": args.a}
    if args.b:
        B = json.loads(Path(args.b).read_text())
        out["b"] = args.b
        out["hotspots"] = compare_hotspots(A, B, args.top)
        out["topdown_level1"] = {"a": (A.get("topdown") or {}).get("level1"),
                                 "b": (B.get("topdown") or {}).get("level1")}
        out["ipc"] = {"a": _ipc(A), "b": _ipc(B)}
    if args.ref_ipc or args.ref_frontend:
        ri = tuple(float(x) for x in args.ref_ipc.split(":")) if args.ref_ipc else None
        rf = tuple(float(x) for x in args.ref_frontend.split(":")) if args.ref_frontend else None
        out["reference_check"] = check_reference(A, ref_ipc=ri, ref_frontend=rf)
    if args.out:
        write_json(args.out, out)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
