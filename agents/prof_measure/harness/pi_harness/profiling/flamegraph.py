#!/usr/bin/env python3
"""on-CPU 火焰图：perf.data -> collapsed 栈 -> SVG。

工具优先级（与任务要求一致）：

1. ``flamegraph-rs``（``cargo install flamegraph``，默认 ~/tools/cargo/bin/flamegraph）
   —— 直接读 perf.data，内部调 perf script + inferno；
2. 退化路径：本模块自带的 ``perf script`` 解析 + 纯 Python SVG 渲染（无第三方依赖），
   产物语义相同（on-CPU 采样折叠栈），差别只在着色/交互细节；
3. Python 帧：perf 只能给出 C 层（_PyEval_EvalFrameDefault），需要真正的
   Python 函数名时用 ``py-spy --native``（本模块 ``pyspy`` 子命令）。

用法：
    python3 -m pi_harness.profiling.flamegraph from-perfdata perf.data -o out.svg
    python3 -m pi_harness.profiling.flamegraph from-perfdata perf.data \\
        --collapsed out.folded --tool builtin -o out.svg
    python3 -m pi_harness.profiling.flamegraph pyspy --pid 1234 --duration 10 -o py.svg --native
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

from . import CALIBER_VERSION, DEFAULT_FLAMEGRAPH_RS, DEFAULT_PYSPY

HEADER_RE = re.compile(r"^(?P<comm>.*?)\s+(?P<pid>\d+)(?:/(?P<tid>\d+))?\s+"
                       r"(?P<time>[\d.]+):\s+(?P<rest>.*)$")
FRAME_RE = re.compile(r"^\s+(?P<addr>[0-9a-fA-F]+)\s+(?P<sym>.*?)\s+\((?P<dso>.*)\)\s*$")
KERNEL_FRAME_RE = re.compile(
    r"^\s+(?P<addr>[0-9a-fA-F]+)\s+(?P<sym>.*?)\s+\(\[kernel\.kallsyms\]\)\s*$")
DOS_PATH_RE = re.compile(r"\((/[^)]+)\)\s*$")


def _sudo_prefix(sudo: bool) -> list[str]:
    if sudo and os.geteuid() != 0:
        return ["sudo", "-n"]
    return []


def run(cmd: list[str], timeout: float | None = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def collapsed_from_perf_script(perf_data: str | Path, out_folded: str | Path,
                               perf: str = "perf", sudo: bool = True,
                               extra_args: Iterable[str] = (),
                               symfs: str | None = None,
                               kernel_map: dict[str, str] | None = None) -> dict[str, Any]:
    """把 perf script 的调用栈折叠成 flamegraph 输入格式。"""
    cmd = _sudo_prefix(sudo) + [perf, "script", "--force", "-i", str(perf_data)]
    if symfs:
        cmd += ["--symfs", str(symfs)]
    cmd += list(extra_args)
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"perf script 失败: {p.stderr.strip()[:400]}")
    stacks: dict[str, int] = {}
    cur: list[str] = []
    n_samples = 0
    for line in p.stdout.splitlines():
        if not line.strip():
            if cur:
                stacks[";".join(reversed(cur))] = stacks.get(";".join(reversed(cur)), 0) + 1
                n_samples += 1
                cur = []
            continue
        if line[0] not in " \t":  # 样本头
            if cur:
                stacks[";".join(reversed(cur))] = stacks.get(";".join(reversed(cur)), 0) + 1
                n_samples += 1
                cur = []
            continue
        m = FRAME_RE.match(line)
        if m:
            sym = m.group("sym")
            if sym == "[unknown]" and kernel_map:
                sym = kernel_map.get(m.group("addr"), sym)
            sym = re.sub(r"\+0x[0-9a-fA-F]+$", "", sym) or "[unknown]"
            if "(inlined)" in sym:
                sym = sym.replace(" (inlined)", "").strip()
            cur.append(sym)
    if cur:
        stacks[";".join(reversed(cur))] = stacks.get(";".join(reversed(cur)), 0) + 1
        n_samples += 1
    with open(out_folded, "w", encoding="utf-8") as fh:
        for stack, count in sorted(stacks.items()):
            fh.write(f"{stack} {count}\n")
    return {"collapsed": str(out_folded), "n_samples": n_samples,
            "n_stacks": len(stacks), "perf_script_stderr": p.stderr.strip()[:400]}


def kernel_symbol_map(perf_data: str | Path, perf: str = "perf",
                      sudo: bool = True, timeout: float = 900) -> dict[str, str]:
    """先跑一遍不带 --symfs 的 perf script，拿内核地址 -> 符号名。

    为什么要这个：perf 的 --symfs 会把内核 DSO 也改到 symfs 里找，导致
    [kernel.kallsyms] 全部变成 [unknown]。两次输出里的地址字符串完全一致，
    所以用这份映射把带 symfs 的结果里的内核帧还原成符号名。
    """
    cmd = _sudo_prefix(sudo) + [perf, "script", "--force", "-i", str(perf_data)]
    p = run(cmd, timeout=timeout)
    out: dict[str, str] = {}
    for line in p.stdout.splitlines():
        m = KERNEL_FRAME_RE.match(line)
        if m and m.group("sym") != "[unknown]":
            out[m.group("addr").lower()] = m.group("sym")
    return out


def list_dso_paths(perf_data: str | Path, perf: str = "perf", sudo: bool = True,
                   timeout: float = 1800) -> list[str]:
    """perf.data 里出现过的 DSO 绝对路径。

    走 `perf report -D` 的原始事件流，抓 PERF_RECORD_MMAP 的最后一个字段
    （文件名）。比 `perf script -F dso` 可靠：后者在 6.6 上输出空行。
    """
    cmd = _sudo_prefix(sudo) + [perf, "report", "--force", "-D", "-i", str(perf_data)]
    p = run(cmd, timeout=timeout)
    paths: set[str] = set()
    for line in p.stdout.splitlines():
        if "PERF_RECORD_MMAP" not in line:
            continue
        cand = line.split()[-1]
        cand = cand.removesuffix("(deleted)")
        if cand.startswith("/") and not cand.startswith("//") and cand != "/":
            paths.add(cand)
    return sorted(paths)


def flamegraph_rs(perf_data: str | Path, out_svg: str | Path, *,
                  title: str | None = None, binary: str | Path = DEFAULT_FLAMEGRAPH_RS,
                  sudo: bool = True, symfs: str | None = None,
                  perf_wrapper: str | None = None) -> dict[str, Any]:
    """用 flamegraph-rs 直接从 perf.data 生成 SVG。"""
    if perf_wrapper:
        # rsync 不带可执行位时会静默失败成 "Permission denied"，自己补上
        try:
            p = Path(perf_wrapper)
            if p.is_file() and not os.access(p, os.X_OK):
                p.chmod(p.stat().st_mode | 0o755)
        except OSError:
            pass
    cmd = _sudo_prefix(sudo)
    if symfs and perf_wrapper:
        # 通过 PERF 包装器把 --symfs 传给 flamegraph-rs 内部的 perf script
        cmd += ["env", f"PERF={perf_wrapper}", f"PERF_SYMFS={symfs}"]
    cmd += [str(binary), "--perfdata", str(perf_data), "-o", str(out_svg)]
    if title:
        cmd += ["--title", title]
    p = run(cmd)
    ok = p.returncode == 0 and Path(out_svg).is_file()
    return {"tool": "flamegraph-rs", "ok": ok, "cmd": cmd,
            "stdout": p.stdout.strip()[-400:], "stderr": p.stderr.strip()[-400:]}


# --------------------------------------------------------------------------- #
# 内置渲染器（退化路径，无第三方依赖）
# --------------------------------------------------------------------------- #
def _name_color(name: str) -> str:
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    hue = h % 360
    return f"hsl({hue},70%,68%)"


class _Node:
    __slots__ = ("name", "value", "children")

    def __init__(self, name: str) -> None:
        self.name = name
        self.value = 0
        self.children: dict[str, "_Node"] = {}


def load_collapsed(path: str | Path) -> tuple[_Node, int]:
    root = _Node("all")
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                stack, count_s = line.rsplit(" ", 1)
                count = int(count_s)
            except ValueError:
                continue
            node = root
            node.value += count
            for frame in stack.split(";"):
                node = node.children.setdefault(frame, _Node(frame))
                node.value += count
            total += count
    return root, total


def render_svg(collapsed: str | Path, out_svg: str | Path, *,
               title: str = "on-CPU flamegraph", width: int = 1400,
               frame_h: int = 15, font: int = 11, min_px: float = 0.4) -> dict[str, Any]:
    """把 collapsed 折叠栈渲染成静态 SVG（深度优先，宽度按样本数）。"""
    root, total = load_collapsed(collapsed)
    if total <= 0:
        raise RuntimeError(f"{collapsed} 里没有样本")
    max_depth = 0

    def depth_of(node: _Node, d: int = 0) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, d)
        for ch in node.children.values():
            depth_of(ch, d + 1)

    depth_of(root)
    height = 40 + (max_depth + 1) * frame_h
    rows: list[str] = []
    text: list[str] = []

    def emit(node: _Node, x: float, y: float, w: float, depth: int) -> None:
        if w < min_px or depth > 200:
            return
        name = html.escape(node.name)
        label = ""
        if w > 30:
            approx = max(1, int(w / (font * 0.62)))
            short = node.name if len(node.name) <= approx else node.name[: max(1, approx - 1)] + "…"
            label = html.escape(short)
        rows.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{frame_h - 1}" '
            f'fill="{_name_color(node.name)}" stroke="#333" stroke-width="0.3">'
            f'<title>{name} ({node.value} samples, {100.0 * node.value / total:.2f}%)</title></rect>')
        if label:
            text.append(f'<text x="{x + 3:.2f}" y="{y + frame_h - 4:.2f}" '
                        f'font-size="{font}" font-family="monospace" fill="#111">{label}</text>')
        cursor = x
        # 大孩子先画，保证视觉稳定
        for ch in sorted(node.children.values(), key=lambda n: -n.value):
            cw = w * ch.value / node.value if node.value else 0.0
            emit(ch, cursor, y + frame_h, cw, depth + 1)
            cursor += cw

    emit(root, 0.0, 30.0, float(width), 0)
    svg = (
        f'<?xml version="1.0" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">\n'
        f'<rect width="100%" height="100%" fill="#f7f7f7"/>\n'
        f'<text x="8" y="14" font-size="12">{html.escape(title)} — '
        f'{total} samples, {len(root.children)} root frames, max depth {max_depth}</text>\n'
        f'<text x="8" y="27" font-size="10" fill="#555">prof-v1 / on-CPU (cycles) / '
        f'宽度=样本占比；本图为内置渲染器兜底产物</text>\n'
        + "\n".join(rows) + "\n" + "\n".join(text) + "\n</svg>\n"
    )
    Path(out_svg).write_text(svg, encoding="utf-8")
    return {"tool": "builtin-svg", "ok": True, "samples": total,
            "max_depth": max_depth, "bytes": len(svg)}


def pyspy_flamegraph(pid: int, out_svg: str | Path, *, duration: int = 10,
                     rate: int = 199, native: bool = True, idle: bool = False,
                     sudo: bool = True, binary: str | Path = DEFAULT_PYSPY) -> dict[str, Any]:
    """py-spy 采样：默认只采 running 线程（on-CPU 语义），带 Python 帧。"""
    cmd = _sudo_prefix(sudo) + [str(binary), "record", "--pid", str(pid),
                                "--duration", str(duration), "--rate", str(rate),
                                "--format", "flamegraph", "-o", str(out_svg)]
    if native:
        cmd.append("--native")
    if idle:
        cmd.append("--idle")
    p = run(cmd)
    ok = p.returncode == 0 and Path(out_svg).is_file()
    return {"tool": "py-spy", "ok": ok, "cmd": cmd,
            "stdout": p.stdout.strip()[-600:], "stderr": p.stderr.strip()[-600:]}


def main() -> int:
    ap = argparse.ArgumentParser(description="on-CPU 火焰图（perf.data / py-spy）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("from-perfdata", help="perf.data -> SVG")
    p1.add_argument("perfdata")
    p1.add_argument("-o", "--out", required=True)
    p1.add_argument("--collapsed", default=None, help="同时保存折叠栈")
    p1.add_argument("--tool", default="auto", choices=["auto", "flamegraph-rs", "builtin"])
    p1.add_argument("--title", default=None)
    p1.add_argument("--no-sudo", action="store_true")
    p1.add_argument("--symfs", default=None,
                    help="容器 DSO 抽取目录：perf script 与 flamegraph-rs 都用它解析符号")
    p1.add_argument("--perf-wrapper", default=None,
                    help="包装 perf 的脚本（给 flamegraph-rs 注入 --symfs）")

    p2 = sub.add_parser("from-collapsed", help="折叠栈 -> SVG（内置渲染器）")
    p2.add_argument("collapsed")
    p2.add_argument("-o", "--out", required=True)
    p2.add_argument("--title", default="on-CPU flamegraph")

    p3 = sub.add_parser("pyspy", help="py-spy 采样容器进程（Python 帧）")
    p3.add_argument("--pid", type=int, required=True)
    p3.add_argument("-o", "--out", required=True)
    p3.add_argument("--duration", type=int, default=10)
    p3.add_argument("--rate", type=int, default=199)
    p3.add_argument("--no-native", action="store_true")
    p3.add_argument("--idle", action="store_true")
    p3.add_argument("--no-sudo", action="store_true")

    args = ap.parse_args()
    if args.cmd == "from-perfdata":
        import json

        info: dict[str, Any] = {}
        folded = args.collapsed or str(Path(args.out).with_suffix(".folded"))
        wrapper = args.perf_wrapper or os.environ.get("PI_PERF_WRAPPER") or str(
            Path(__file__).resolve().parents[2] / "scripts" / "prof_perf_wrapper.sh")
        if not Path(wrapper).is_file():
            wrapper = None
        if args.tool in ("auto", "flamegraph-rs"):
            rs_bin = Path(DEFAULT_FLAMEGRAPH_RS)
            if rs_bin.is_file():
                info["flamegraph_rs"] = flamegraph_rs(
                    args.perfdata, args.out, title=args.title,
                    binary=rs_bin, sudo=not args.no_sudo,
                    symfs=args.symfs, perf_wrapper=wrapper)
                if info["flamegraph_rs"]["ok"]:
                    print(json.dumps(info, ensure_ascii=False))
                    return 0
            elif args.tool == "flamegraph-rs":
                raise SystemExit(f"找不到 {rs_bin}")
        kmap = kernel_symbol_map(args.perfdata, sudo=not args.no_sudo) if args.symfs else None
        if kmap:
            info["kernel_symbols"] = len(kmap)
        info["collapsed"] = collapsed_from_perf_script(
            args.perfdata, folded, sudo=not args.no_sudo, symfs=args.symfs,
            kernel_map=dict(kmap) if kmap else None)
        info["builtin"] = render_svg(folded, args.out, title=args.title or "on-CPU flamegraph")
        print(json.dumps(info, ensure_ascii=False))
    elif args.cmd == "from-collapsed":
        print(render_svg(args.collapsed, args.out, title=args.title))
    else:
        print(pyspy_flamegraph(args.pid, args.out, duration=args.duration,
                               rate=args.rate, native=not args.no_native,
                               idle=args.idle, sudo=not args.no_sudo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
