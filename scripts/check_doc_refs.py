#!/usr/bin/env python3
"""检查文档树里的图片引用是否有效。

三类检查：
  A. 引用的图**是否存在**（含 `figures/xxx.svg` 形式与裸文件名）
  B. 花括号简写 `06-sweep-{a,b}.svg` **展开后**每一项是否存在
  C. `figures/` 里是否有**从未被任何文档提及**的孤儿图

用法：
    python3 scripts/check_doc_refs.py            # 报告（有失效则退出码 1）
    python3 scripts/check_doc_refs.py --quiet    # 只输出汇总

为什么需要它：图片引用用的是**相对仓库根**的路径（`figures/...`），
而文档分布在 `docs/` 与 `docs/<子目录>/` 两层；子目录里的作者很容易
漏写前缀或写成自己相对路径。这个脚本在每次改动文档后跑一遍即可。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

# 文档集合：docs/ 全部（含子目录）+ 仓库根的 README
MD_FILES = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]

MD_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")            # ![alt](path)
HTML_IMG = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)
INLINE = re.compile(r"`([^`\s]+\.(?:svg|png))`")          # `path.svg`
BRACE = re.compile(r"\{([^}]+)\}")


def expand_braces(raw: str) -> list[str]:
    """展开 `a/{x,y}.svg` → ['a/x.svg', 'a/y.svg']；无花括号则原样返回。"""
    m = BRACE.search(raw)
    if not m:
        return [raw]
    opts = m.group(1).split(",")
    return [f"{raw[:m.start()]}{o}{raw[m.end():]}" for o in opts]


def main() -> int:
    quiet = "--quiet" in sys.argv
    figures = {p.name for p in FIGDIR.iterdir() if p.is_file()}

    broken: list[tuple[str, int, str]] = []      # (文档, 行, 路径)
    bad_brace: list[tuple[str, int, str]] = []
    mentioned: set[str] = set()
    total = 0

    for md in MD_FILES:
        if not md.exists():
            continue
        rel_md = md.relative_to(ROOT)
        for lineno, line in enumerate(md.read_text(errors="replace").splitlines(), 1):
            for rx in (MD_IMG, HTML_IMG, INLINE):
                for m in rx.finditer(line):
                    raw = m.group(1).split("#")[0].strip()
                    if raw.startswith(("http://", "https://")):
                        continue
                    if any(t in raw for t in ("<", "*")):
                        continue          # 模板占位符，如 figures/<name>.svg
                    total += 1
                    for cand in expand_braces(raw):
                        name = cand.split("/")[-1]
                        mentioned.add(name)
                        if not (ROOT / cand).exists():
                            (bad_brace if "{" in raw else broken).append(
                                (str(rel_md), lineno, cand)
                            )

    orphans = sorted(
        f for f in figures
        if f.endswith(".svg") and f not in mentioned
    )
    prefix_missing = sorted(
        (d, l, c) for d, l, c in broken
        if "/" not in c and c in figures
    )

    if not quiet:
        print(f"扫描 {sum(1 for m in MD_FILES if m.exists())} 个文档，"
              f"提取图片引用 {total} 条\n")

    def dump(title: str, rows, tag: str) -> None:
        if quiet:
            return
        print(f"{tag} {title}")
        for d, l, c in rows:
            print(f"    {d}:{l}  {c}")
        if not rows:
            print("    (无)")
        print()

    dump("A. 引用的图不存在", broken, "❌")
    dump("B. 花括号简写展开后缺失", bad_brace, "❌")
    dump("C. 缺 `figures/` 前缀（图在 figures/ 里但写成了裸名）", prefix_missing, "⚠️ ")
    dump("D. figures/ 里从未被提及的 SVG（孤儿图）",
         [(f, 0, f) for f in orphans], "⚠️ ")

    ok = not (broken or bad_brace or orphans)
    print(f"结果：{'✅ 全部通过' if ok else '❌ 存在问题'}"
          f"（失效 {len(broken)} · 花括号缺失 {len(bad_brace)}"
          f" · 缺前缀 {len(prefix_missing)} · 孤儿 {len(orphans)}）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
