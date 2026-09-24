#!/usr/bin/env python3
"""Produce the children(inclusive)-overhead breakdown around one call path.

The campaign's head-line question is whether ``AscendAttentionMetadataBuilder.build()``
(called from ``_build_attention_metadata``, 4x per step) is really the single
largest part of the ``prepare input`` scope.  A flat self-overhead report cannot
answer that: the cost is spread across many small callees (pin_memory, tolist,
index_select, tril/masked_fill, ...), each of which looks negligible on its own.

So this script reads an existing ``perf.data`` and emits, for a set of symbol
patterns:

* ``<name>.children.txt``  - ``perf report --stdio --children --sort symbol`` rows
  whose symbol matches the pattern (inclusive % + self %), i.e. the breakdown;
* ``<name>.callers.txt``   - ``perf report --stdio -g graph,0.5,caller`` for the
  same filter, showing who calls into it;
* ``<name>.callees.txt``   - ``perf report --stdio -g graph,0.5,callee``, showing
  what it calls and how much each callee costs;
* ``summary.json``         - the same numbers in machine-readable form, plus a
  ``dso_resolution`` block so a reader can see whether the symbol was found at
  all (a missing symbol must never be reported as "0% cost").

Nothing here re-runs the workload: it is pure post-processing of a capture that
already exists, so it can be repeated as often as needed without touching chip3.

Usage (on a3-22, root or passwordless sudo available):
    python3 scripts/measure/build_breakdown.py --perf-data DIR/perf.data \
        --outdir DIR/breakdown [--symbol build] [--symbol _build_attention_metadata] \
        [--symbol pin_memory] [--top 40]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

DEFAULT_SYMBOLS = (
    # headline suspicion: the metadata builders (ascend + core GDN)
    "AscendGDNAttentionMetadataBuilder",
    "GDNAttentionMetadataBuilder",
    "AscendAttentionMetadataBuilder",
    "_build_attn_group_metadata",
    "_treat_single_token_prefills_with_state_as_decodes",
    "_fold_spec_sized_prefill_chunks_into_spec",
    "mamba_get_block_table_tensor",
    "compute_num_computed_tokens",
    "compute_causal_conv1d_metadata",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "build",
    "_build_attention_metadata",
    "split_decodes_and_prefills",
    # every-request pinned/H2D evidence (docs/01 predicted these as the cost)
    "pin_memory",
    "PinMemoryAllocator",
    "cudaHostAlloc",
    "async_tensor_h2d",
    "np_to_pinned_tensor",
    "tolist",
    "index_select",
    # state churn
    "_update_states",
    "_prepare_inputs",
    "condense",
    "swap_states",
    "_calc_spec_decode_metadata",
    # a D2H sync showing up here would mean the builder blocks on the device
    "aclrtSynchronize",
    "_local_scalar_dense",
    "aten::item",
)

# Two flat-report layouts have to be accepted, because perf prints different
# column sets depending on --sort:
#   --sort dso,symbol : "  12.34%   5.67%  comm  dso  [.] symbol"
#   --sort symbol     : "  12.34%   5.67%  [.] symbol"
# An earlier revision only matched the first, so a `--sort symbol` capture
# parsed to zero rows and every symbol was reported as "not observed".
ROW_RE = re.compile(
    r"^\s*(?P<children>[\d.]+)%\s+(?P<self>[\d.]+)%\s+"
    r"(?:(?P<comm>\S+)\s+(?P<dso>\S+)\s+)?"
    r"\[(?P<symtype>[.kug])\]\s+(?P<sym>.*?)\s*$")
GRAPH_ROW_RE = re.compile(
    r"^\s*(?P<pct>[\d.]+)%\s+(?P<sym>.*?)\s*$")


def sh(cmd: list[str], timeout: float = 1800, sudo: bool = True):
    if sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def perf_version() -> str:
    proc = subprocess.run(["perf", "--version"], capture_output=True, text=True)
    return (proc.stdout or proc.stderr).strip()


def report(perf_data: pathlib.Path, extra: list[str], sudo: bool) -> tuple[int, str, str]:
    cmd = ["perf", "report", "--stdio", "-i", str(perf_data)] + extra
    proc = sh(cmd, sudo=sudo)
    return proc.returncode, proc.stdout, proc.stderr


def parse_flat(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        rows.append({
            "children_pct": float(m.group("children")),
            "self_pct": float(m.group("self")),
            "comm": m.group("comm"),
            "dso": m.group("dso"),
            "symtype": m.group("symtype"),
            "symbol": m.group("sym").strip(),
        })
    return rows


def parse_graph(text: str) -> list[dict]:
    """Parse `-g graph,0.5,callee/caller` output into (indent, pct, symbol)."""
    rows: list[dict] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        m = GRAPH_ROW_RE.match(stripped)
        if not m:
            continue
        indent = len(stripped) - len(stripped.lstrip())
        rows.append({"indent": indent, "pct": float(m.group("pct")),
                     "symbol": m.group("sym").strip()})
    return rows


def match(rows: list[dict], pattern: str) -> list[dict]:
    needle = pattern.lower()
    return [r for r in rows if needle in (r.get("symbol") or "").lower()
            or needle in (r.get("dso") or "").lower()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf-data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--symbol", action="append", default=None,
                    help="symbol substring to break down (repeatable; "
                         "defaults to the prepare-input call path)")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--no-sudo", action="store_true")
    args = ap.parse_args(argv)

    perf_data = pathlib.Path(args.perf_data)
    if not perf_data.is_file():
        print(f"ERROR: {perf_data} not found", file=sys.stderr)
        return 2
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    patterns = args.symbol if args.symbol else list(DEFAULT_SYMBOLS)
    sudo = not args.no_sudo

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary: dict = {
        "perf_data": str(perf_data),
        "perf_data_bytes": perf_data.stat().st_size,
        "perf_version": perf_version(),
        "started_utc": started,
        "symbols": {},
        "notes": [],
    }

    # flat listing with inclusive (children) overhead, sorted by symbol
    rc, flat_txt, flat_err = report(perf_data, ["--children", "--sort", "symbol"], sudo)
    if rc != 0:
        summary["notes"].append(f"flat --children report rc={rc}: {flat_err[-400:]}")
    (outdir / "children-flat.txt").write_text(flat_txt, encoding="utf-8")
    flat_rows = parse_flat(flat_txt)
    summary["n_flat_rows"] = len(flat_rows)

    for pattern in patterns:
        entry: dict = {"pattern": pattern}
        hits = match(flat_rows, pattern)
        entry["flat_hits"] = hits[: args.top]
        entry["found_in_flat"] = bool(hits)
        (outdir / f"{slug(pattern)}.children.txt").write_text(
            "\n".join(f"{r['children_pct']:7.2f}%  self {r['self_pct']:6.2f}%  "
                      f"{r['dso']}  {r['symbol']}" for r in hits[: args.top]) + "\n",
            encoding="utf-8")

        # callee tree, restricted to this symbol when perf supports --symbol
        rc_c, callees, err_c = report(
            perf_data,
            ["--no-children", "-g", "graph,0.5,callee", "--symbol", pattern,
             "--top", str(args.top)],
            sudo)
        if rc_c != 0 or not callees.strip():
            rc_c, callees, err_c = report(
                perf_data,
                ["--no-children", "-g", "graph,0.5,callee", "--sort", "symbol",
                 "--top", str(args.top)],
                sudo)
        (outdir / f"{slug(pattern)}.callees.txt").write_text(callees, encoding="utf-8")
        graph_rows = parse_graph(callees)
        entry["callee_rows"] = graph_rows[: args.top]
        entry["callee_found"] = bool(graph_rows)

        rc_k, callers, err_k = report(
            perf_data,
            ["--no-children", "-g", "graph,0.5,caller", "--symbol", pattern,
             "--top", str(args.top)],
            sudo)
        if rc_k != 0 or not callers.strip():
            rc_k, callers, err_k = report(
                perf_data,
                ["--no-children", "-g", "graph,0.5,caller", "--sort", "symbol",
                 "--top", str(args.top)],
                sudo)
        (outdir / f"{slug(pattern)}.callers.txt").write_text(callers, encoding="utf-8")
        entry["caller_rows"] = parse_graph(callers)[: args.top]
        entry["rc"] = {"callees": rc_c, "callers": rc_k}
        summary["symbols"][pattern] = entry

    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing = [p for p, e in summary["symbols"].items() if not e["found_in_flat"]]
    if missing:
        summary["notes"].append(
            "symbols not present in the flat report (report as 'not observed by "
            f"sampling', never as 0%): {missing}")
    print(json.dumps({
        "outdir": str(outdir),
        "flat_rows": summary["n_flat_rows"],
        "found": [p for p, e in summary["symbols"].items() if e["found_in_flat"]],
        "not_found": missing,
    }, indent=2))
    return 0 if summary["n_flat_rows"] else 1


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sym"


if __name__ == "__main__":
    raise SystemExit(main())
