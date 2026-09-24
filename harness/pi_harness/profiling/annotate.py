#!/usr/bin/env python3
"""热点函数 / 热点源码行 / 热点指令。

三个动作：

* ``hotspots``：``perf report --stdio --no-children`` 按 DSO+符号给 self overhead；
* ``annotate``：``perf annotate --stdio -l`` 出指令级注释（含源码行号）；
* ``parse``：把上面那份文本解析成结构化 JSON（源码行 top-N + 指令 top-N）。

用法：
    python3 -m pi_harness.profiling.annotate hotspots -i perf.data -o hotspots.json
    python3 -m pi_harness.profiling.annotate annotate -i perf.data --symbol _PyEval_EvalFrameDefault \\
        -o annotate.txt --json hot_instr.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from . import CALIBER_VERSION
from .envcap import write_json

INSTR_RE = re.compile(
    r"^\s*(?P<pct>[\d.]+)\s*:\s+(?P<addr>[0-9a-fA-F]{4,})\s*:\s+(?P<mn>\S+)\s*(?P<rest>.*)$")
SRCLINE_RE = re.compile(r"^\s*(?P<pct>[\d.]*)\s*:\s+(?P<line>\d+)\s+(?P<text>.*)$")
SUMMARY_LOC_RE = re.compile(r"^\s*(?P<pct>[\d.]+)\s+(?P<loc>\S+\.(?:c|h|py|cc|cpp|hpp|S|s):\d+)\s*$")


def _sudo(sudo: bool) -> list[str]:
    if sudo and os.geteuid() != 0:
        return ["sudo", "-n"]
    return []


def run(cmd: list[str], timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def hotspots(perf_data: str | Path, out_json: str | Path, *, top: int = 40,
             perf: str = "perf", sudo: bool = True,
             symfs: str | None = None) -> dict[str, Any]:
    """self overhead 热点函数（--no-children）。"""
    cmd = _sudo(sudo) + [perf, "report", "--force", "--stdio", "--no-children",
                         "--sort", "dso,symbol", "-t", "|",
                         "-i", str(perf_data)]
    if symfs:
        cmd += ["--symfs", str(symfs)]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"perf report 失败: {p.stderr.strip()[:300]}")
    rows: list[dict[str, Any]] = []
    header: dict[str, Any] = {}
    for line in p.stdout.splitlines():
        if line.startswith("#"):
            if "Samples:" in line:
                header["samples"] = line.split(":", 1)[1].strip()
            if "Event count" in line:
                header["event_count"] = line.split(":", 1)[1].strip()
            if "Total Lost Samples" in line:
                header["lost_samples"] = line.split(":", 1)[1].strip()
            continue
        if "|" not in line:
            continue
        parts = [x.strip() for x in line.split("|")]
        if not parts or not parts[0].endswith("%"):
            continue
        try:
            pct = float(parts[0].rstrip("%"))
        except ValueError:
            continue
        dso = parts[1] if len(parts) > 1 else ""
        symbol = parts[2] if len(parts) > 2 else ""
        rows.append({"overhead_pct": pct, "dso": dso, "symbol": symbol})
    rows.sort(key=lambda r: -r["overhead_pct"])
    out = {
        "caliber_version": CALIBER_VERSION,
        "kind": "self_overhead",
        "perfdata": str(perf_data),
        "cmd": cmd,
        "header": header,
        "total_symbols": len(rows),
        "top": rows[:top],
        "sum_of_top_pct": round(sum(r["overhead_pct"] for r in rows[:top]), 2),
        "sum_all_pct": round(sum(r["overhead_pct"] for r in rows), 2),
    }
    write_json(out_json, out)
    return out


def annotate(perf_data: str | Path, symbol: str, out_txt: str | Path, *,
             perf: str = "perf", sudo: bool = True,
             extra: list[str] | None = None,
             symfs: str | None = None) -> dict[str, Any]:
    """perf annotate --stdio -l，原文落盘。"""
    cmd = _sudo(sudo) + [perf, "annotate", "--force", "--stdio", "-l",
                         "-i", str(perf_data), "--symbol", symbol]
    if symfs:
        cmd += ["--symfs", str(symfs)]
    cmd += list(extra or [])
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"perf annotate 失败: {p.stderr.strip()[:300]}")
    Path(out_txt).write_text(p.stdout, encoding="utf-8")
    return {"cmd": cmd, "lines": len(p.stdout.splitlines()), "out": str(out_txt),
            "stderr": p.stderr.strip()[:200]}


def parse_annotate(path: str | Path, out_json: str | Path | None = None, *,
                   top_instr: int = 40, top_src: int = 30) -> dict[str, Any]:
    """解析 perf annotate 文本：源码行热度 + 指令热度（带所属源码行）。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    src_summary: list[dict[str, Any]] = []
    instrs: list[dict[str, Any]] = []
    cur_src: str | None = None
    in_summary = False
    # 先解析文件级 "Sorted summary" 段
    for line in text.splitlines():
        m = SUMMARY_LOC_RE.match(line)
        if m:
            in_summary = True
            src_summary.append({"pct": float(m.group("pct")), "loc": m.group("loc")})
            continue
        if line.startswith(" Percent |"):
            in_summary = False
            continue
        if in_summary and not line.strip():
            continue
    # 再解析反汇编段：区分"指令行"和"源码行"
    for line in text.splitlines():
        mi = INSTR_RE.match(line)
        if mi:
            instrs.append({
                "pct": float(mi.group("pct")),
                "addr": mi.group("addr"),
                "mnemonic": mi.group("mn"),
                "operands": mi.group("rest").strip(),
                "src": cur_src,
            })
            continue
        ms = SRCLINE_RE.match(line)
        if ms:
            txt = ms.group("text").strip()
            cur_src = f"L{ms.group('line')} {txt}"[:120]
    instrs.sort(key=lambda r: -r["pct"])
    out = {
        "caliber_version": CALIBER_VERSION,
        "kind": "annotate",
        "file": str(path),
        "n_instructions": len(instrs),
        "sum_instr_pct": round(sum(r["pct"] for r in instrs), 2),
        "hot_source_lines": src_summary[:top_src],
        "top_instructions": instrs[:top_instr],
    }
    if out_json:
        write_json(out_json, out)
    return out


def annotate_top(perf_data: str | Path, hotspots_json: str | Path,
                 out_dir: str | Path, *, top: int = 3,
                 symfs: str | None = None, sudo: bool = True,
                 skip_kernel: bool = True,
                 merged_json: str | Path | None = None) -> dict[str, Any]:
    """对热点榜 top-N 个函数逐个做指令级注释，并汇总成一个 JSON。

    跳过内核符号（`[k]` / `[kernel.kallsyms]`）与裸地址（没有符号名可传给
    `perf annotate --symbol`）。
    """
    hs = json.loads(Path(hotspots_json).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked: list[dict[str, Any]] = []
    for row in hs.get("top", []):
        if len(picked) >= top:
            break
        raw = str(row.get("symbol", "")).strip()
        dso = str(row.get("dso", ""))
        if skip_kernel and ("kernel" in dso or raw.startswith("[k]")):
            continue
        sym = re.sub(r"^\[[.k]\]\s*", "", raw).strip()
        if not sym or re.fullmatch(r"0x[0-9a-fA-F]+", sym):
            continue                      # 裸地址没法用 --symbol 指定
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sym)[:60]
        txt = out_dir / f"{safe}.txt"
        jso = out_dir / f"{safe}.json"
        info = annotate(perf_data, sym, txt, sudo=sudo, symfs=symfs)
        parsed = parse_annotate(txt, jso)
        picked.append({
            "rank": len(picked) + 1,
            "symbol": sym,
            "symbol_raw": raw,
            "dso": dso,
            "self_overhead_pct": row.get("overhead_pct"),
            "annotate_lines": info["lines"],
            "n_instructions": parsed["n_instructions"],
            "hot_source_lines": parsed["hot_source_lines"],
            "top_instructions": parsed["top_instructions"][:20],
            "txt": str(txt),
            "json": str(jso),
        })
    out: dict[str, Any] = {
        "caliber_version": CALIBER_VERSION,
        "kind": "annotate_top",
        "perfdata": str(perf_data),
        "hotspots": str(hotspots_json),
        "symfs": symfs,
        "requested_top": top,
        "annotated": len(picked),
        "functions": picked,
    }
    if merged_json:
        write_json(merged_json, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="热点函数/源码行/指令")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("hotspots")
    p1.add_argument("-i", "--perfdata", required=True)
    p1.add_argument("-o", "--out", required=True)
    p1.add_argument("--top", type=int, default=40)
    p1.add_argument("--no-sudo", action="store_true")
    p1.add_argument("--symfs", default=None, help="容器 DSO 抽取目录（解析符号）")

    p2 = sub.add_parser("annotate")
    p2.add_argument("-i", "--perfdata", required=True)
    p2.add_argument("--symbol", required=True)
    p2.add_argument("-o", "--out", required=True, help="annotate 原文路径")
    p2.add_argument("--json", default=None, help="解析后的 JSON")
    p2.add_argument("--no-sudo", action="store_true")
    p2.add_argument("--symfs", default=None, help="容器 DSO 抽取目录（解析符号）")

    p3 = sub.add_parser("parse")
    p3.add_argument("txt")
    p3.add_argument("--json", default=None)

    p4 = sub.add_parser("top", help="热点榜 top-N 函数逐个指令级注释")
    p4.add_argument("-i", "--perfdata", required=True)
    p4.add_argument("--hotspots", required=True, help="hotspots.json")
    p4.add_argument("--out-dir", required=True)
    p4.add_argument("--json", default=None, help="汇总 JSON")
    p4.add_argument("--top", type=int, default=3)
    p4.add_argument("--symfs", default=None)
    p4.add_argument("--no-sudo", action="store_true")
    p4.add_argument("--include-kernel", action="store_true")
    args = ap.parse_args()

    if args.cmd == "hotspots":
        out = hotspots(args.perfdata, args.out, top=args.top, sudo=not args.no_sudo,
                       symfs=args.symfs)
        print(json.dumps({"top3": out["top"][:3], "sum_top_pct": out["sum_of_top_pct"]},
                         ensure_ascii=False))
    elif args.cmd == "annotate":
        info = annotate(args.perfdata, args.symbol, args.out, sudo=not args.no_sudo,
                        symfs=args.symfs)
        print(json.dumps(info, ensure_ascii=False))
        if args.json:
            parsed = parse_annotate(args.out, args.json)
            print(json.dumps({"hot_src": parsed["hot_source_lines"][:5],
                              "top_instr": parsed["top_instructions"][:5]},
                             ensure_ascii=False))
    elif args.cmd == "parse":
        parsed = parse_annotate(args.txt, args.json)
        print(json.dumps({"n_instructions": parsed["n_instructions"],
                          "top_instr": parsed["top_instructions"][:5]},
                         ensure_ascii=False))
    else:
        merged = annotate_top(args.perfdata, args.hotspots, args.out_dir,
                              top=args.top, symfs=args.symfs,
                              sudo=not args.no_sudo,
                              skip_kernel=not args.include_kernel,
                              merged_json=args.json)
        print(json.dumps({"annotated": merged["annotated"],
                          "functions": [{"rank": f["rank"], "symbol": f["symbol"],
                                         "pct": f["self_overhead_pct"]}
                                        for f in merged["functions"]]},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
