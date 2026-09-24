#!/usr/bin/env python3
"""summarize_bench.py -- turn raw per-run JSON/logs into CSV + markdown.

Reads the interleaved per-round JSON files produced by run_bench_arms.sh and
the perf stat text files produced by perf_stat_arms.sh, then writes:

  * bench-pi.csv / bench-dispatch.csv : per-round raw numbers
  * perf-summary.csv                  : parsed PMU counters per arm
  * summary.md                        : median + speedup table

Usage:
  python summarize_bench.py --exp DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from glob import glob


def load_pi(files):
    rows = []
    digests = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        arm = d["label"]
        digests.setdefault(arm, set()).add(d.get("work_digest"))
        for m in d["measurements"]:
            rows.append({
                "arm": arm, "round": m["round"], "seconds": m["seconds"],
                "steps": m["steps"], "ns_per_step": m["ns_per_step"],
                "file": os.path.basename(f),
            })
    return rows, digests


def load_dispatch(files):
    rows = []
    digests = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        arm = d["label"]
        digests.setdefault(arm, set()).add(d.get("work_digest"))
        for name, kd in d["kernels"].items():
            for r in kd["rounds"]:
                rows.append({"arm": arm, "kernel": name, "round": r["round"],
                             "seconds": r["seconds"],
                             "ns_per_iter": r["ns_per_iter"]})
    return rows, digests


def med(rows, key, arm, kernel=None):
    vals = [r[key] for r in rows if r["arm"] == arm and
            (kernel is None or r["kernel"] == kernel)]
    return statistics.median(vals) if vals else float("nan")


def best(rows, key, arm, kernel=None):
    vals = [r[key] for r in rows if r["arm"] == arm and
            (kernel is None or r["kernel"] == kernel)]
    return min(vals) if vals else float("nan")


def parse_perf(path):
    """parse `perf stat` text output (values are locale-independent enough)"""
    out = {}
    txt = open(path, errors="replace").read()
    for line in txt.splitlines():
        line = line.strip()
        m = re.match(r"^([\d,]+(?:\.\d+)?)\s+(\S.*?)\s*(?:#.*)?$", line)
        if not m:
            continue
        val = float(m.group(1).replace(",", ""))
        name = m.group(2).strip()
        name = re.sub(r"\s+\([^)]*\)$", "", name).strip()
        if name in ("seconds time elapsed", "seconds user", "seconds sys"):
            continue
        out.setdefault(name, val)
    # derived
    if "cycles" in out and "instructions" in out:
        out["IPC"] = out["instructions"] / out["cycles"]
    if "branches" in out and "branch-misses" in out:
        out["branch-miss-rate-%"] = 100.0 * out["branch-misses"] / out["branches"]
    if "L1-icache-loads" in out and "L1-icache-load-misses" in out:
        out["L1I-miss-rate-%"] = 100.0 * out["L1-icache-load-misses"] / out["L1-icache-loads"]
    if "iTLB-loads" in out and "iTLB-load-misses" in out:
        out["iTLB-miss-rate-%"] = 100.0 * out["iTLB-load-misses"] / out["iTLB-loads"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    exp, out = args.exp, args.out
    os.makedirs(out, exist_ok=True)

    lines = []

    # ------------------------------------------------------------ pi bench
    pi_files = sorted(glob(f"{exp}/data/bench-pi/pi-arm*-r*.json"))
    pi_rows, pi_dig = (load_pi(pi_files) if pi_files else ([], {}))
    if pi_files:
        with open(f"{out}/bench-pi.csv", "w") as fh:
            fh.write("arm,round,ns_per_step,seconds,steps,file\n")
            for r in sorted(pi_rows, key=lambda r: (r["arm"], r["round"])):
                fh.write(f"{r['arm']},{r['round']},{r['ns_per_step']:.1f},"
                         f"{r['seconds']:.6f},{r['steps']},{r['file']}\n")
    # ------------------------------------------------------ dispatch bench
    dp_files = sorted(glob(f"{exp}/data/bench-dispatch/dispatch-arm*-r*.json"))
    dp_rows, dp_dig = (load_dispatch(dp_files) if dp_files else ([], {}))
    if dp_files:
        with open(f"{out}/bench-dispatch.csv", "w") as fh:
            fh.write("arm,kernel,round,ns_per_iter,seconds\n")
            for r in sorted(dp_rows, key=lambda r: (r["arm"], r["kernel"], r["round"])):
                fh.write(f"{r['arm']},{r['kernel']},{r['round']},"
                         f"{r['ns_per_iter']:.2f},{r['seconds']:.6f}\n")

    # ---------------------------------------------------------------- perf
    perf = {}
    perf_files = sorted(glob(f"{exp}/data/perf/perf-arm*-g*.txt"))
    for f in perf_files:
        m = re.search(r"perf-arm([ABC])-g(\d)", os.path.basename(f))
        if m:
            perf.setdefault(m.group(1), {}).update(parse_perf(f))
    if perf:
        keys = ["cycles", "instructions", "IPC", "branches", "branch-misses",
                "branch-miss-rate-%", "L1-icache-loads", "L1-icache-load-misses",
                "L1I-miss-rate-%", "iTLB-loads", "iTLB-miss-misses",
                "iTLB-miss-rate-%", "seconds time elapsed"]
        with open(f"{out}/perf-summary.csv", "w") as fh:
            fh.write("metric," + ",".join(f"arm{a}" for a in "ABC") + "\n")
            for k in keys:
                fh.write(k + "," + ",".join(
                    f"{perf.get(a, {}).get(k, float('nan'))}" for a in "ABC") + "\n")

    # ------------------------------------------------------------ markdown
    lines.append("# CPython build-config benchmark summary\n")
    lines.append(f"raw inputs: `{exp}/data/bench-pi`, `bench-dispatch`, `perf`\n")
    if pi_rows:
        lines.append("\n## interleaved prepare_input-like workload\n")
        lines.append("| arm | median ns/step | best ns/step | round ns/step | speedup vs A (median) |")
        lines.append("|---|---|---|---|---|")
        base = med(pi_rows, "ns_per_step", "A")
        for arm in "ABC":
            vals = ", ".join(f"{r['ns_per_step']:.0f}"
                             for r in sorted(pi_rows, key=lambda r: r["round"])
                             if r["arm"] == arm)
            m_, b_ = med(pi_rows, "ns_per_step", arm), best(pi_rows, "ns_per_step", arm)
            lines.append(f"| {arm} | {m_:.1f} | {b_:.1f} | {vals} | "
                         f"{base / m_:.4f}x |")
        lines.append("\nwork digests (must match across arms): "
                     + " ".join(f"{a}={sorted(v)}" for a, v in sorted(pi_dig.items())))
    if dp_rows:
        lines.append("\n## dispatch-dominated kernels (upper bound)\n")
        kernels = sorted({r["kernel"] for r in dp_rows})
        lines.append("| kernel | A ns/iter | B ns/iter | C ns/iter | B/A | C/A |")
        lines.append("|---|---|---|---|---|---|")
        for k in kernels:
            a = med(dp_rows, "ns_per_iter", "A", k)
            b = med(dp_rows, "ns_per_iter", "B", k)
            c = med(dp_rows, "ns_per_iter", "C", k)
            lines.append(f"| {k} | {a:.2f} | {b:.2f} | {c:.2f} | "
                         f"{b / a:.4f}x | {c / a:.4f}x |")
    if perf:
        lines.append("\n## PMU (`perf stat`, same binary/workload/CPUs)\n")
        lines.append("| metric | arm A | arm B | arm C |")
        lines.append("|---|---|---|---|")
        for k in ["cycles", "instructions", "IPC", "branches", "branch-misses",
                  "branch-miss-rate-%", "L1-icache-load-misses", "L1I-miss-rate-%",
                  "iTLB-miss-rate-%", "seconds time elapsed"]:
            vals = [perf.get(a, {}).get(k) for a in "ABC"]
            if all(v is None for v in vals):
                continue
            fmt = (lambda v: "n/a") if k not in ("IPC",) else (lambda v: f"{v:.3f}")
            lines.append("| " + k + " | " + " | ".join(
                "n/a" if v is None else (f"{v:.3f}" if "rate" in k or k == "IPC"
                                         else f"{v:,.0f}") for v in vals) + " |")
    md = "\n".join(lines) + "\n"
    with open(f"{out}/summary.md", "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
