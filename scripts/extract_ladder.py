#!/usr/bin/env python3
"""Build the definitive A/B/C/D1/D2 ladder table from analyze_lite.py output.

Why this exists instead of reading numbers by eye: with async scheduling a
single engine iteration emits **two** `Step:Model` rows, so `Step:Model` is the
wrong denominator; and the steady-state table can contain several *phase*
columns (`prefill`, `prefill+decode`, `decode`, `idle`) in an order that varies
with the workload.  A regex that grabs "the first number on the n_steps row"
silently reads the wrong phase (this happened for the B=64 scenario, whose
`prefill+decode` column comes first).

This script parses the header row of `analyze_lite.py`'s per-phase table and
indexes columns by **phase name**, then reports the `decode` column explicitly
plus an independent `prepare input` p50 taken from
`parse_phase_timing.py`'s per-step CSV (which is anchored on `prepare input`
rows and therefore immune to the double-row problem).

Usage::

    extract_ladder.py --project-root ~/projects/vllm/prepare-input-phase \
        --scenario s1 [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import subprocess
import sys

# arm -> run-id suffix (the stamp is passed on the command line)
ARMS = {
    "A": "baseline",
    "B": "mounted-off",
    "C": "on",
    "D1": "mounted-off-pystack",
    "D2": "mounted-off-2",
    "C2": "on-attn",
}


def parse_table(text: str) -> dict[str, dict[str, float]]:
    """Parse analyze_lite.py's per-phase table into {scope: {phase: value}}."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("scope") and i + 1 < len(lines) \
                and lines[i + 1].strip().startswith("step_dur"):
            header_idx = i
            break
    if header_idx is None:
        return {}
    header = lines[header_idx].split()
    phases = header[1:]
    out: dict[str, dict[str, float]] = {}
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if not parts or len(parts) < 2:
            break
        if len(parts) <= len(phases):
            # A footnote / legend line rather than a data row.
            break
        # Scope names may contain spaces ("prepare input"), so the last
        # len(phases) tokens are the values and everything before is the name.
        scope = " ".join(parts[: len(parts) - len(phases)])
        values = parts[-len(phases):]
        if scope.startswith("("):
            break
        row: dict[str, float] = {}
        for phase, raw in zip(phases, values):
            try:
                row[phase] = float(raw)
            except ValueError:
                pass
        if row:
            out[scope] = row
    return out


def independent_prepare_p50(project_root: pathlib.Path, run_id: str,
                            phase: str = "decode") -> float | None:
    path = project_root / "runs" / run_id / "lite-profiler" / "phase_timing.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as fh:
        rows = [
            r for r in csv.DictReader(fh)
            if r.get("phase") == phase and r.get("prepare_input")
        ]
    if not rows:
        return None
    return round(statistics.median(float(r["prepare_input"]) for r in rows), 1)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project-root", type=pathlib.Path, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--arms", default="A,B,C,D1,D2")
    ap.add_argument("--phase", default="decode")
    ap.add_argument(
        "--run-id", action="append", default=[], metavar="ARM=RUN_ID",
        help="override the run id for one arm (arms measured in a later "
             "sub-batch have a different stamp)",
    )
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args(argv[1:])

    analyze = args.project_root / "scripts" / "measure" / "analyze_lite.py"
    modes = [m.strip().upper() for m in args.arms.split(",") if m.strip()]
    overrides = {}
    for item in args.run_id:
        arm, _, run_id = item.partition("=")
        overrides[arm.strip().upper()] = run_id
    table: dict[str, dict] = {}
    print(f"scenario {args.scenario} stamp {args.stamp} (phase column: {args.phase})")
    print("%-4s %-24s %10s %10s %10s %9s %7s" % (
        "arm", "run_id", "prep_mean", "step_p50", "prep_p50*", "share%", "n"))
    for arm in modes:
        slug = ARMS.get(arm, arm.lower())
        run_id = overrides.get(arm) or f"sub-{args.scenario}-{slug}-{args.stamp}"
        log = args.project_root / "runs" / run_id / "lite-profiler" / "lite.log"
        if not log.is_file():
            print("%-4s %-24s  MISSING" % (arm, run_id))
            continue
        proc = subprocess.run(
            [sys.executable, str(analyze), str(log),
             "--step-scope", "Step:Schedule", "--window-mode", "next"],
            capture_output=True, text=True,
        )
        parsed = parse_table(proc.stdout)
        prep = parsed.get("prepare input", {}).get(args.phase)
        step = parsed.get("step_dur", {}).get(args.phase)
        fwd = parsed.get("forward", {}).get(args.phase)
        n = parsed.get("n_steps", {}).get(args.phase)
        share = (100.0 * prep / step) if (prep and step) else None
        indep = independent_prepare_p50(args.project_root, run_id, args.phase)
        table[arm] = {
            "run_id": run_id, "prepare_input_mean_us": prep, "step_p50_us": step,
            "forward_mean_us": fwd, "n_steps": n, "share_pct": share,
            "prepare_input_p50_us_independent": indep, "phases": sorted(parsed),
        }
        print("%-4s %-24s %10s %10s %10s %8s%% %7s" % (
            arm, run_id,
            f"{prep:.1f}" if prep else "n/a",
            f"{step:.1f}" if step else "n/a",
            f"{indep:.1f}" if indep else "n/a",
            f"{share:.1f}" if share else "n/a",
            f"{n:.0f}" if n else "n/a"))

    print("\n* prep_p50 is an independent check from parse_phase_timing.py's "
          "per-step CSV (anchored on `prepare input` rows).")
    if "A" in table and "B" in table:
        def d(x, y):
            a = table[x].get("prepare_input_p50_us_independent")
            b = table[y].get("prepare_input_p50_us_independent")
            return (b - a) if (a and b) else None
        print("\nladder (independent p50, prepare_input):")
        for left, right, what in (("A", "B", "mount/import"),
                                  ("B", "C", "probes"),
                                  ("B", "D1", "pystack pollution"),
                                  ("B", "D2", "noise floor"),
                                  ("B", "C2", "probes + attn second level")):
            if left in table and right in table:
                v = d(left, right)
                base = table[left].get("prepare_input_p50_us_independent")
                pct = f" ({100.0*v/base:+.2f}%)" if (v is not None and base) else ""
                print(f"  {right}-{left} {what:<28} {v:+.1f} us{pct}" if v is not None
                      else f"  {right}-{left} {what:<28} n/a")
    if args.json:
        args.json.write_text(json.dumps(
            {"scenario": args.scenario, "stamp": args.stamp, "arms": table},
            indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
