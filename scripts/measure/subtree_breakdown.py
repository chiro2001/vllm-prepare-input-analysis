#!/usr/bin/env python3
"""Per-frame children breakdown from a perf callchain capture.

Why not ``perf report --children``: the flat listing gives each symbol's total
inclusive cost, but not *who calls whom*, and ``--symbol`` is ambiguous in this
perf build (``-S/--symbols`` filters whole entries, dropping the tree).  For the
head-line question -- "what are the children of
``AscendGDNAttentionMetadataBuilder.build``?" -- we need the call tree under one
specific frame.

So this walks ``perf script`` stacks directly.  For every sample whose stack
contains a frame matching ``--match``, it records:

* ``direct``   - the frame immediately *below* the match (its direct callee);
* ``subtree``  - every frame below the match, i.e. the inclusive cost per symbol;
* ``leaf``     - the leaf frame of the sample (where the cycles were actually
  spent);
* ``self_share``- ``100% - sum(direct callees)``, i.e. time attributed to the
  matched frame's own body (Python bytecode / dispatch that never entered a
  callee).

Percentages are reported against three denominators, all printed explicitly:
``all_samples_of_tid``, ``samples_containing_match``, and
``samples_in_the_deepest_match`` (the match may appear several times in a stack;
the deepest occurrence is the one whose subtree is being measured).

Usage (on a3-22, while the target container is alive so the perf map resolves):
    python3 scripts/measure/subtree_breakdown.py --perf-data DIR/perf.data \
        --match "AscendGDNAttentionMetadataBuilder.build" \
        --outdir DIR/subtree [--top 30] [--no-sudo]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scope_attribution as sa  # noqa: E402  (sibling module)


def prettify(frame: str) -> str:
    """`py::build:/path/file.py` -> `build:/file.py` (keeps it readable)."""
    if frame.startswith("py::"):
        body = frame[4:]
        name, _, path = body.partition(":")
        return f"{name}:{os.path.basename(path)}" if path else name
    return frame


def classify_leaf(frame: str) -> str:
    """Bucket a leaf frame so the categories can be compared with the probes."""
    low = frame.lower()
    if "py::" in low:
        return "python_frame"
    if any(k in low for k in ("aclrt", "ascendcl", "libascend", "rt_")):
        return "ascend_driver"
    if "synchron" in low or "_local_scalar_dense" in low or "aten::item" in low:
        return "d2h_sync"
    if any(k in low for k in ("pin_memory", "hostalloc", "malloc", "free", "munmap",
                              "mmap", "je_", "operator new", "operator delete")):
        return "allocator"
    if any(k in low for k in ("copy_", "memcpy", "c10::cuda", "clone")):
        return "copy_h2d"
    if any(k in low for k in ("fill_", "index_select", "gather", "arange", "cat",
                              "sub", "add", "mul", "expand", "to_", "contiguous",
                              "permute", "view", "reshape", "select", "slice")):
        return "aten_op_dispatch"
    if any(k in low for k in ("libpython", "_pyeval", "pyobject", "_pyobject")):
        return "cpython_runtime"
    if "kernel" in low or "schedule" in low or "el0_" in low:
        return "kernel"
    if "libc" in low or "libc.so" in low:
        return "libc"
    return "other"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf-data", required=True)
    ap.add_argument("--match", required=True,
                    help="substring of the frame to analyse (e.g. 'GDNAttentionMetadataBuilder.build')")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tid", type=int, default=None)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--script-file", default=None,
                    help="reuse a previously dumped `perf script` text (gz ok)")
    args = ap.parse_args(argv)

    perf_data = pathlib.Path(args.perf_data)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    script_path = pathlib.Path(args.script_file) if args.script_file else (
        outdir / "perf.script.gz")
    if args.script_file is None:
        proc = sa.sh(["perf", "script", "--ns", "-i", str(perf_data), "--header"],
                     sudo=not args.no_sudo)
        if proc.returncode != 0:
            print(f"ERROR: perf script rc={proc.returncode}: {proc.stderr[-400:]}",
                  file=sys.stderr)
            return 2
        with gzip.open(script_path, "wt", encoding="utf-8") as fh:
            fh.write(proc.stdout)

    samples = sa.parse_perf_script(script_path, sa.boot_epoch())
    tid_counts = Counter(s["tid"] for s in samples)
    tid = args.tid or (tid_counts.most_common(1)[0][0] if tid_counts else None)
    own = [s for s in samples if s["tid"] == tid]

    direct: Counter[str] = Counter()
    subtree: Counter[str] = Counter()
    leaves: Counter[str] = Counter()
    leaf_class: Counter[str] = Counter()
    matched = 0
    multi_match = 0
    self_only = 0

    for s in own:
        idx = None
        for i, frame in enumerate(s["frames"]):
            if args.match in frame:
                idx = i  # deepest occurrence wins
        if idx is None:
            continue
        matched += 1
        if sum(1 for f in s["frames"] if args.match in f) > 1:
            multi_match += 1
        below = s["frames"][idx + 1:]
        if below:
            direct[prettify(below[0])] += 1
            for frame in below:
                subtree[prettify(frame)] += 1
        else:
            self_only += 1
        leaf = s["frames"][-1] if s["frames"] else "[unknown]"
        leaves[prettify(leaf)] += 1
        leaf_class[classify_leaf(leaf)] += 1

    denom_match = matched or 1
    doc = {
        "schema": "pi-subtree-breakdown-v1",
        "perf_data": str(perf_data),
        "match": args.match,
        "target_tid": tid,
        "started_utc": started,
        "samples_total": len(samples),
        "samples_of_tid": len(own),
        "samples_containing_match": matched,
        "samples_with_nested_match": multi_match,
        "samples_where_match_is_leaf": self_only,
        "share_of_tid_samples": (matched / len(own)) if own else None,
        "denominators": {
            "all_samples_of_tid": len(own),
            "samples_containing_match": matched,
        },
        "direct_callees_pct": {k: 100.0 * v / denom_match
                               for k, v in direct.most_common(args.top)},
        "subtree_pct": {k: 100.0 * v / denom_match
                        for k, v in subtree.most_common(args.top)},
        "leaf_pct": {k: 100.0 * v / denom_match
                     for k, v in leaves.most_common(args.top)},
        "leaf_class_pct": {k: 100.0 * v / denom_match
                           for k, v in leaf_class.most_common()},
        "self_pct_estimate": 100.0 * self_only / denom_match,
        "note": ("Percentages are of the samples that contain the matched frame "
                 "(the matched frame's own inclusive cost). 'self_pct_estimate' "
                 "counts samples where the match is the innermost frame; the "
                 "remainder of a frame's self time is Python dispatch that still "
                 "entered a callee, so it is not separately attributable here."),
    }
    (outdir / "summary.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_text_report(outdir / "report.txt", doc)
    print(json.dumps({
        "match": args.match,
        "tid": tid,
        "samples_of_tid": len(own),
        "samples_containing_match": matched,
        "share_of_tid_samples": doc["share_of_tid_samples"],
        "leaf_class_pct": doc["leaf_class_pct"],
        "top_direct": list(doc["direct_callees_pct"].items())[:10],
        "outdir": str(outdir),
    }, indent=2))
    return 0 if matched else 1


def write_text_report(path: pathlib.Path, doc: dict) -> None:
    lines = [
        f"# subtree breakdown: {doc['match']}",
        "",
        f"- perf.data: {doc['perf_data']}",
        f"- thread: {doc['target_tid']}",
        f"- samples of tid: {doc['samples_of_tid']}",
        f"- samples containing the match: {doc['samples_containing_match']} "
        f"({(doc['share_of_tid_samples'] or 0) * 100:.2f}% of the thread's samples)",
        f"- nested matches (recursion/repeats in one stack): "
        f"{doc['samples_with_nested_match']}",
        "",
        "## leaf class (where the cycles were actually spent)",
        "",
        "| class | % of matched samples |",
        "|---|---:|",
    ]
    for key, value in doc["leaf_class_pct"].items():
        lines.append(f"| {key} | {value:.2f} |")
    lines += ["", "## direct callees", "", "| callee | % |", "|---|---:|"]
    for key, value in doc["direct_callees_pct"].items():
        lines.append(f"| {key} | {value:.2f} |")
    lines += ["", "## subtree (inclusive)", "", "| symbol | % |", "|---|---:|"]
    for key, value in doc["subtree_pct"].items():
        lines.append(f"| {key} | {value:.2f} |")
    lines += ["", "## leaves (top)", "", "| leaf | % |", "|---|---:|"]
    for key, value in doc["leaf_pct"].items():
        lines.append(f"| {key} | {value:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
