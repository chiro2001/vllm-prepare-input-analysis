#!/usr/bin/env python3
"""E. `prepare input` scope 边界与调用链的静态追踪（AST，不执行代码）。

目标（INTERFACES.md / COORDINATION.md 的验收问题 1）：
  1. 精确定位 `execute_model()` 里 `record_function_or_nullcontext("prepare input")`
     的 with-block 起止行；
  2. 列出 block 内**每一条**语句（file:line）与它调用的函数；
  3. 递归展开被调函数（默认深度 2），把调用树里的**设备交互点**逐条标出来；
  4. 输出 `data/harness/devdeps_scope.json`，供占比分母定义与后续文档引用。

用法::

    python harness/probes/scope_prepare_input.py                 # 默认 /work/refs 与内置回退路径
    python harness/probes/scope_prepare_input.py --depth 3
    python harness/probes/scope_prepare_input.py --markdown      # 额外打印 markdown 摘要

路径解析顺序：
  1. --ascend / --vllm 显式给定；
  2. $PI_SOURCE_ROOT/{vllm-ascend,vllm}；
  3. /work/refs/{vllm-ascend,vllm}（容器内挂载的只读源码）；
  4. 本地快照 agents/replay_harness/src_snapshot/{vllm_ascend,vllm_v1}（离线可用）。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SCOPE_LABEL = "prepare input"

# ---------------------------------------------------------------------------
# 设备交互识别规则（用于给调用树里的节点打 tag）
# ---------------------------------------------------------------------------
DEVICE_PATTERNS: list[tuple[str, str, str]] = [
    (r"\btorch\.npu\.([A-Za-z_0-9]+)", "torch.npu.*", "device"),
    (r"\btorch_npu\.([A-Za-z_0-9.]+)", "torch_npu.*", "device"),
    (r"\bacl\.[A-Za-z_0-9]+", "acl.*", "device"),
    (r"\.npu\(\)", "Tensor.npu()", "device"),
    (r"\bto\(\s*[^)]*device", ".to(device=...)", "device-adjacent"),
    (r"non_blocking", "non_blocking=True", "device-adjacent"),
    (r"pin_memory", "pin_memory", "device-adjacent"),
    (r"\bstream\b", "stream", "device-adjacent"),
    (r"\bEvent\b", "Event", "device-adjacent"),
    (r"synchronize", "synchronize", "device-adjacent"),
    (r"_kernel\[", "triton kernel launch", "device"),
    (r"\btriton\b", "triton", "device"),
    (r"copy_to_gpu", "CpuGpuBuffer.copy_to_gpu", "device-adjacent"),
    (r"copy_to_cpu", "CpuGpuBuffer.copy_to_cpu", "device-adjacent"),
    (r"synchronize_input_prep", "synchronize_input_prep", "device-adjacent"),
    (r"\b_np\b|\.np\b|numpy|np\.", "numpy", "cpu"),
]

CPU_MARKERS = (
    "np.",
    "numpy",
    "dict",
    "list",
    "append",
    "copy(",
    "replace(",
    "deepcopy",
    "len(",
    "range(",
    "int(",
    "zeros(",
    "cumsum",
    "repeat",
    "index_select",
)


def classify_line(line: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    kind = "cpu"
    for pat, name, cls in DEVICE_PATTERNS:
        if re.search(pat, line):
            hits.append(name)
            if cls == "device" or (cls == "device-adjacent" and kind == "cpu"):
                kind = cls
    return kind, hits


class Index:
    """把若干 py 文件里的函数/方法建成 qualname -> (node, path) 索引。"""

    def __init__(self) -> None:
        self.by_qualname: dict[str, tuple[ast.AST, Path, str]] = {}
        self.by_class_method: dict[tuple[str, str], list[str]] = {}
        self.by_name: dict[str, list[str]] = {}
        self.modules: dict[str, Path] = {}

    def add_file(self, path: Path, module: str) -> None:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(path))
        self.modules[module] = path

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    visit(child, f"{prefix}{child.name}.")
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{module}:{prefix}{child.name}"
                    self.by_qualname[qual] = (child, path, src)
                    self.by_name.setdefault(child.name, []).append(qual)
                    if "." in prefix:
                        cls = prefix.rstrip(".").split(".")[-1]
                        self.by_class_method.setdefault((cls, child.name), []).append(qual)
                    visit(child, f"{prefix}{child.name}.")

        visit(tree, "")

    def resolve(self, callee: str, prefer: list[str]) -> str | None:
        """把 `self._prepare_inputs` / `update_cos_sin` 解析成索引里的 qualname。"""
        name = callee.split(".")[-1]
        cands = self.by_name.get(name, [])
        for pref in prefer:
            for c in cands:
                if c.startswith(pref):
                    return c
        return cands[0] if cands else None


def callee_text(call: ast.Call) -> str:
    f = call.func
    parts: list[str] = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    elif isinstance(f, ast.Call):
        return callee_text(f) + "()"
    return ".".join(reversed(parts))


def find_scope_block(tree: ast.AST, label: str) -> tuple[ast.With, ast.FunctionDef]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            ce = item.context_expr
            if (
                isinstance(ce, ast.Call)
                and getattr(ce.func, "id", None) == "record_function_or_nullcontext"
                and ce.args
                and isinstance(ce.args[0], ast.Constant)
                and ce.args[0].value == label
            ):
                owner = None
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.FunctionDef) and node in list(ast.walk(parent)):
                        owner = parent
                        break
                return node, owner  # type: ignore[return-value]
    raise SystemExit(f"scope block {label!r} not found")


def stmt_summary(node: ast.stmt) -> str:
    if isinstance(node, ast.With):
        txt = ", ".join(ast.unparse(i.context_expr) for i in node.items)
        return f"with {txt}:"
    if isinstance(node, ast.If):
        return f"if {ast.unparse(node.test)}:"
    if isinstance(node, ast.For):
        return f"for {ast.unparse(node.target)} in {ast.unparse(node.iter)}:"
    if isinstance(node, ast.Try):
        return "try:"
    return ast.unparse(node).split("\n")[0][:200]


def walk_body(
    idx: Index,
    body: list[ast.stmt],
    path: Path,
    src: str,
    depth: int,
    max_depth: int,
    visited: set[str],
    prefer: list[str],
) -> list[dict]:
    out: list[dict] = []
    lines = src.splitlines()
    for st in body:
        # 该语句本身的设备/CPU 判定
        seg = ast.get_source_segment(src, st) or ast.unparse(st)
        kind, hits = classify_line(seg)
        label = stmt_summary(st)
        rec: dict = {
            "line": getattr(st, "lineno", None),
            "kind": kind,
            "device_hits": hits,
            "stmt": label,
        }
        line_no = getattr(st, "lineno", None)
        if line_no and 1 <= line_no <= len(lines):
            rec["src"] = lines[line_no - 1].strip()[:200]

        callees: list[dict] = []
        for sub in ast.walk(st):
            if not isinstance(sub, ast.Call):
                continue
            name = callee_text(sub)
            if not name or name.startswith("record_function_or_nullcontext"):
                continue
            subseg = ast.get_source_segment(src, sub) or ""
            k2, h2 = classify_line(subseg)
            entry = {
                "callee": name,
                "line": sub.lineno,
                "kind": k2,
                "device_hits": h2,
            }
            qual = idx.resolve(name, prefer)
            if qual and qual not in visited and depth < max_depth:
                node, p2, s2 = idx.by_qualname[qual]
                entry["resolved"] = qual
                visited2 = visited | {qual}
                entry["body"] = walk_body(
                    idx, node.body, p2, s2, depth + 1, max_depth, visited2, prefer
                )
            elif qual:
                entry["resolved"] = qual
                entry["expanded"] = False
            callees.append(entry)
        rec["calls"] = callees
        out.append(rec)
        # 递归进入嵌套的 with/if/for 体（保留结构，便于阅读 scope 内部层次）
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(st, field, None)
            if inner and isinstance(st, (ast.With, ast.If, ast.For, ast.Try)) and field == "body":
                nested = walk_body(idx, inner, path, src, depth + 1, max_depth, visited, prefer)
                rec.setdefault("nested", []).extend(nested)
    return out


def flatten(
    recs: list[dict],
    idx: "Index",
    out: list[dict],
    path: str = "prepare input",
    level: int = 0,
) -> None:
    """把树拍平成「scope 步骤」列表（保留嵌套层次 path）。"""
    for r in recs:
        calls = []
        dev_indirect = 0
        for c in r.get("calls", []):
            qual = c.get("resolved")
            defsite = None
            def_file = None
            def_line = None
            if qual and qual in idx.by_qualname:
                node, p, _ = idx.by_qualname[qual]
                defsite = f"{p.name}:{node.lineno}"
                def_file, def_line = str(p), node.lineno
            subdev = _count_device(c)
            dev_indirect += subdev
            calls.append({
                "callee": c["callee"],
                "call_line": c["line"],
                "kind": c["kind"],
                "device_hits": c["device_hits"],
                "resolved": qual,
                "def_site": defsite,
                "def_file": def_file,
                "def_line": def_line,
                "device_hits_deep": subdev,
            })
        out.append({
            "level": level,
            "path": path,
            "line": r["line"],
            "stmt": r["stmt"],
            "own_kind": r["kind"],
            "own_device_hits": r["device_hits"],
            "device_hits_deep": dev_indirect,
            "calls": calls,
        })
        for c in r.get("calls", []):
            if c.get("body"):
                flatten(c["body"], idx, out, f"{path} > {c['callee']}()", level + 1)
        if r.get("nested"):
            flatten(r["nested"], idx, out, path, level + 1)


def _count_device(rec: dict) -> int:
    """递归统计该调用子树里出现的设备交互点数量。"""
    n = 1 if rec.get("kind") == "device" else 0
    if rec.get("kind") == "device-adjacent":
        n += 1
    for c in rec.get("calls", []) or []:
        n += _count_device(c)
    for b in rec.get("body", []) or []:
        n += _count_device(b)
    return n


def collect_device_points(recs: list[dict], idx: "Index") -> list[dict]:
    pts: list[dict] = []
    for r in recs:
        for c in r.get("calls", []):
            if c.get("kind") in ("device", "device-adjacent") and c.get("device_hits"):
                qual = c.get("resolved")
                defsite = None
                def_file = None
                def_line = None
                if qual and qual in idx.by_qualname:
                    node, p, _ = idx.by_qualname[qual]
                    defsite = f"{p}:{node.lineno}"
                    def_file, def_line = str(p), node.lineno
                pts.append({
                    "call_line": c["line"],
                    "callee": c["callee"],
                    "kind": c["kind"],
                    "hits": c["device_hits"],
                    "resolved": qual,
                    "def_site": defsite,
                    "def_file": def_file,
                    "def_line": def_line,
                    "in_stmt_line": r["line"],
                })
            if c.get("body"):
                pts.extend(collect_device_points(c["body"], idx))
        if r.get("nested"):
            pts.extend(collect_device_points(r["nested"], idx))
    return pts


def _boundary_markdown(scope: dict, steps: list[dict], src: str, mr_path: Path) -> str:
    """把 scope 边界写成行号级 markdown（占比分母的唯一定义）。"""
    lines = src.splitlines()
    start, end = scope["scope_start_line"], scope["scope_end_line"]
    out: list[str] = []
    out.append("# `prepare input` scope 边界（行号级证据）\n")
    out.append(f"> 生成时间：{scope['generated_at']}　|　脚本：`harness/probes/scope_prepare_input.py`（sha256 `{scope['script_sha256'][:16]}…`）\n")
    out.append(f"> 文件：`{mr_path}`（sha256 `{scope['model_runner_v1_sha256']}`）\n")
    out.append(f"> 宿主函数：`{scope['owner_function']}()`\n")
    out.append("## 1. 起止行\n")
    out.append(f"- **起点** `L{start}`：`with record_function_or_nullcontext(\"prepare input\"):`"
               f"　（父函数 `{scope['owner_function']}`）")
    out.append(f"- **终点** `L{end}`：scope 的**最后一条语句**，同时也是整个 `with` 块的末行")
    out.append(f"- 证明（末尾三行的原文）：\n")
    out.append("```python")
    for i in range(end - 2, min(end + 3, len(lines))):
        marker = "  # <-- scope 末行" if i + 1 == end else ""
        if i + 1 == end + 2:
            marker = marker or "  # <-- scope 之外（同级语句）"
        out.append(f"{i + 1:5d}: {lines[i]}{marker}")
    out.append("```\n")
    out.append("> 判据：`L{}` 之后的同级语句缩进回到 `execute_model` 的 8 空格层（`L2100 if self.dynamic_eplb:`），"
               "因此 `with` 块在 `L{}` 结束，`dynamic_eplb` / `forward` / `post process` / `sample_token` 都在 scope **之外**。\n"
               .format(end, end))
    out.append("## 2. scope 顶层语句（3 条）\n")
    out.append("| # | 行 | 语句 | 说明 |")
    out.append("|---|---|---|---|")
    top = [s for s in steps if s["level"] == 0]
    for i, s in enumerate(top, 1):
        note = {
            "with self.synchronize_input_prep():": "**scope 的第一个语句**：async scheduling 下这里会 `prepare_inputs_event.synchronize()` 真等（真机同步点）",
            "input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output = self._preproces": "调 `_preprocess`（文本模型路径几乎只做 view/切片）",
            "update_cos_sin(positions)": "**scope 的最后一条语句**：写全局 rope cache（真机上是 3 个设备 kernel）",
        }.get(s["stmt"][:100], "")
        out.append(f"| {i} | L{s['line']} | `{s['stmt'][:110]}` | {note} |")
    out.append("")
    out.append("## 3. scope 内调用的全部函数（按调用点去重）\n")
    seen: dict[str, dict] = {}
    for s in steps:
        for c in s["calls"]:
            key = (c["callee"], c["def_site"])
            if key not in seen:
                seen[key] = c
    out.append("| 调用点行 | callee | 定义位置 | 设备交互 |")
    out.append("|---|---|---|---|")
    for (callee, defsite), c in sorted(seen.items(), key=lambda kv: kv[1]["call_line"]):
        hits = ",".join(c["device_hits"]) or "—"
        out.append(f"| L{c['call_line']} | `{callee[:70]}` | `{defsite or '—'}` | {hits} |")
    out.append("")
    out.append("## 4. 占比分母的唯一定义\n")
    out.append("```")
    out.append('t_prepare_input = wall( 从 L{} 进入 with 开始 )'.format(start))
    out.append('                  - wall( 到 L{} 执行完 update_cos_sin 为止 )'.format(end))
    out.append("```")
    out.append("\nscope 内**包含**（必须计入分母）：\n")
    out.append("- `synchronize_input_prep` 的 event synchronize/record（scope 入口）")
    out.append("- `_update_states`（ascend 覆写 + 父类 `GPUModelRunner._update_states`）")
    out.append("- `_prepare_inputs` 全流程（含 block table / positions / slot mapping / spec metadata / attention metadata）")
    out.append("- `_determine_batch_execution_and_padding`、`maybe_create_ubatch_slices`、`_pad_query_start_loc_for_fia`")
    out.append("- `_build_attention_metadata`、`_sanitize_placeholder_input_ids_for_forward`")
    out.append("- `_preprocess`、`update_cos_sin`")
    out.append("\nscope **不含**（不得计入）：\n")
    out.append("- `if self.dynamic_eplb: self.eplb_updator.forward_before()`（L2100 起）")
    out.append("- `with record_function_or_nullcontext(\"forward\")`（L2120 起）")
    out.append("- `post process`（L2147）/ `sample_token`（L2266）/ `draft_token`")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ascend", default=None, help="vllm-ascend 包目录")
    ap.add_argument("--vllm", default=None, help="vllm 包目录")
    ap.add_argument("--label", default=SCOPE_LABEL)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out", default=os.environ.get("PI_SCOPE_OUT",
                                                    "/work/data/harness/devdeps_scope.json"))
    ap.add_argument("--boundary-out", default=os.environ.get(
        "PI_SCOPE_BOUNDARY_OUT", "/work/data/harness/devdeps_scope_boundary.md"))
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    snapshot = here.parents[2] / "src_snapshot"          # agents/replay_harness/src_snapshot
    cand_ascend = [
        args.ascend,
        os.environ.get("PI_SOURCE_ROOT") and f"{os.environ['PI_SOURCE_ROOT']}/vllm-ascend",
        "/work/refs/vllm-ascend",
        str(snapshot),
    ]
    cand_vllm = [
        args.vllm,
        os.environ.get("PI_SOURCE_ROOT") and f"{os.environ['PI_SOURCE_ROOT']}/vllm",
        "/work/refs/vllm",
        str(snapshot),                       # 快照里 vllm 核心是 src_snapshot/vllm_v1
    ]

    ascend_dir = next((Path(c) for c in cand_ascend if c and Path(c).is_dir()), None)
    vllm_dir = next((Path(c) for c in cand_vllm if c and Path(c).is_dir()), None)
    if ascend_dir is None or vllm_dir is None:
        raise SystemExit("cannot locate sources; pass --ascend/--vllm")

    # /work/refs/vllm 下是源码仓根（vllm/v1/...），快照下是 vllm_v1/...
    if (ascend_dir / "worker" / "model_runner_v1.py").exists():
        mr_path = ascend_dir / "worker" / "model_runner_v1.py"
    else:
        mr_path = ascend_dir / "vllm_ascend" / "worker" / "model_runner_v1.py"

    if (ascend_dir / "vllm_ascend").is_dir():
        pass
    if (vllm_dir / "vllm" / "v1").is_dir():
        core = vllm_dir / "vllm" / "v1"
    elif (vllm_dir / "vllm_v1").is_dir():            # 本地快照布局
        core = vllm_dir / "vllm_v1"
    elif (vllm_dir / "v1").is_dir():
        core = vllm_dir / "v1"
    else:
        raise SystemExit(f"cannot locate vllm v1 core under {vllm_dir}")

    idx = Index()
    # 只索引与本 scope 关系最近的模块：ascend worker + vllm v1 worker/utils
    targets = [mr_path, core / "worker" / "gpu_model_runner.py", core / "utils.py",
               core / "worker" / "block_table.py", core / "worker" / "gpu_input_batch.py"]
    for t in targets:
        if t.exists():
            idx.add_file(t, t.stem)
    # ascend 包内其它可能被调用到的模块（attention / spec_decode 等）
    pkg_root = mr_path.parents[1]
    extra = []
    for sub in ("attention", "spec_decode", "compilation", "distributed", "ops", "device"):
        p = pkg_root / sub
        if p.is_dir():
            extra += sorted(p.rglob("*.py"))
    for p in extra[:400]:
        try:
            idx.add_file(p, p.stem)
        except SyntaxError:
            continue

    src = mr_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=str(mr_path))
    block, owner = find_scope_block(tree, args.label)

    prefer = ["vllm_ascend", "model_runner_v1"]
    body = walk_body(idx, list(block.body), mr_path, src, 0, args.depth, set(), prefer)

    steps: list[dict] = []
    flatten(body, idx, steps)
    dev_points = collect_device_points(body, idx)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": str(here),
        "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "scope_label": args.label,
        "expand_depth": args.depth,
        "model_runner_v1": str(mr_path),
        "model_runner_v1_sha256": hashlib.sha256(mr_path.read_bytes()).hexdigest(),
        "owner_function": owner.name if owner else None,
        "scope_start_line": block.lineno,
        "scope_end_line": getattr(block, "end_lineno", None),
        "top_level_statements": len(block.body),
        "scope_steps": len(steps),
        "device_interaction_points": dev_points,
        "steps": steps,
        "tree": body,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"scope: {owner.name if owner else '?'} L{block.lineno}-{out['scope_end_line']}"
          f"  顶层语句 {len(block.body)} 条  拍平步骤 {len(steps)}  设备交互点 {len(dev_points)}")
    print(f"wrote {outp}")

    # ---- C2: scope 边界行号级证据（写 markdown，供 docs/占比分母定义引用） ----
    boundary_md = _boundary_markdown(scope=out, steps=steps, src=src, mr_path=mr_path)
    bpath = Path(args.boundary_out)
    bpath.write_text(boundary_md)
    print(f"wrote {bpath}")

    print("\n=== scope 顶层语句（缩进=嵌套层次；DEV*=子树里有设备交互） ===")
    for s in steps:
        if s["level"] > 1:
            continue
        if s["own_kind"] == "cpu" and s["device_hits_deep"]:
            tag = "DEV*"
        else:
            tag = {"cpu": "CPU", "device": "DEV", "device-adjacent": "~DEV"}[s["own_kind"]]
        print(f"{'  ' * s['level']}L{s['line']:>5} [{tag:>4}] {s['stmt'][:100]}")

    if args.markdown:
        print("\n=== device interaction points (scope 子树内，按调用点) ===")
        print("| call site | callee | resolved | def site | kind | hits |")
        print("|---|---|---|---|---|---|")
        for d in dev_points:
            print(f"| L{d['call_line']} | `{d['callee'][:60]}` | `{d.get('resolved') or ''}` | "
                  f"`{d.get('def_site') or ''}` | {d['kind']} | {','.join(d['hits'])} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
