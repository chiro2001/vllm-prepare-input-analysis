#!/usr/bin/env python3
"""Compare the A/B/C instrumentation ladder on one metric: prepare-input time.

For every run ``runs/<run_id>/lite-profiler/phase_timing.csv`` produced by
``scripts/parse_phase_timing.py``, compute the median/p90 of the
``prepare input`` scope wall time and of the total engine step, then print and
persist the ladder::

    A baseline      T_A
    B mounted-off   T_B      (B - A) = cost of mounting + importing the probes
    C on            T_C      (C - B) = cost of the probes themselves
    D on+pystack    T_D      (D - C) = cost of the pystack sampler

The result is written to ``data/subscope/overhead-<scenario>-<stamp>.json`` so
the number quoted in the report always has a machine-readable source.

Usage::

    compare_overhead.py --scenario s1 --stamp 20260924T0200Z [--modes A,B,C,D]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def read_phase_timing(path: pathlib.Path, warmup: int) -> dict | None:
    """Summarise parse_phase_timing.py's per-step CSV (one row per engine step).

    Its shape is wide: ``step,phase,start_us,<scope>...,Step:Model`` where the
    scope names have their spaces replaced by underscores (``prepare_input``).
    """
    if not path.is_file():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    meta = {"step", "phase", "start_us"}
    columns = [c for c in rows[0] if c not in meta]
    out: dict[str, dict] = {}
    for column in columns:
        values: list[float] = []
        for row in rows:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if not values:
            continue
        values = values[warmup:] or values
        out[column] = {
            "n": len(values),
            "p50": round(quantile(values, 0.5), 3),
            "p90": round(quantile(values, 0.9), 3),
            "p99": round(quantile(values, 0.99), 3),
            "mean": round(statistics.fmean(values), 3),
        }
    return out


def find_column(stats: dict, candidates: tuple[str, ...]) -> dict | None:
    """Match a column name ignoring case, underscores and surrounding space."""

    def norm(text: str) -> str:
        return text.strip().lower().replace("_", " ")

    for key in stats:
        for cand in candidates:
            if norm(key) == norm(cand):
                return stats[key]
    for key in stats:
        for cand in candidates:
            if norm(cand) in norm(key):
                return stats[key]
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--project-root", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument(
        "--modes",
        default="A,B,C,D",
        help="comma separated subset of A,B,C,D to compare",
    )
    ap.add_argument("--run-id", action="append", default=[],
                    metavar="LETTER=RUN_ID",
                    help="override the run id for one arm, e.g. C=sub-s1-on-S1a")
    args = ap.parse_args(argv[1:])

    slug = {
        "A": "baseline",
        "B": "mounted-off",
        "C": "on",
        "D": "on-pystack",
    }
    overrides = {}
    for item in args.run_id:
        letter, _, run_id = item.partition("=")
        overrides[letter.strip().upper()] = run_id

    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    ids = {
        mode: overrides.get(mode)
        or f"sub-{args.scenario}-{slug.get(mode, mode.lower())}-{args.stamp}"
        for mode in modes
    }
    ladder: dict[str, dict] = {}
    for label, run_id in ids.items():
        path = args.project_root / "runs" / run_id / "lite-profiler" / "phase_timing.csv"
        stats = read_phase_timing(path, args.warmup_steps)
        ladder[label] = {
            "run_id": run_id,
            "phase_timing_csv": str(path),
            "available": stats is not None,
            "prepare_input": (find_column(stats, ("prepare input",)) if stats else None),
            "step": (find_column(stats, ("step", "step:model", "model step"))
                     if stats else None),
        }

    def p50(label: str, what: str) -> float | None:
        entry = ladder.get(label, {}).get(what)
        return entry["p50"] if entry else None

    deltas: dict[str, dict] = {}
    for what in ("prepare_input", "step"):
        entry: dict = {}
        for mode in modes:
            value = p50(mode, what)
            entry[f"{mode}_us"] = value
        for left, right in zip(modes, modes[1:]):
            a, b = p50(left, what), p50(right, what)
            if a is None or b is None:
                entry[f"{right}_minus_{left}_us"] = None
                entry[f"{right}_vs_{left}_pct"] = None
                continue
            entry[f"{right}_minus_{left}_us"] = round(b - a, 3)
            entry[f"{right}_vs_{left}_pct"] = round(100.0 * (b - a) / a, 2) if a else None
        # The headline number the report needs: cost of the probes alone.
        base_mode = modes[0]
        entry[f"last_vs_first_abs_us"] = (
            round(p50(modes[-1], what) - p50(base_mode, what), 3)
            if p50(modes[-1], what) is not None and p50(base_mode, what) is not None
            else None
        )
        deltas[what] = entry

    doc = {
        "schema": "prepare-input-overhead-ladder/v1",
        "scenario": args.scenario,
        "stamp": args.stamp,
        "warmup_steps_dropped": args.warmup_steps,
        "runs": ids,
        "modes": modes,
        "per_run": ladder,
        "deltas": deltas,
        "interpretation": {
            "B_minus_A": "cost of bind-mounting the instrumented module (import only)",
            "C_minus_B": "cost of the probes themselves (PI_SUBSCOPE=on)",
            "D_minus_C": "cost of the pystack sampler inside the prepare-input window",
            "budget": "task requires C-B <= 5% of prepare input",
        },
    }
    out = args.project_root / "data" / "subscope" / f"overhead-{args.scenario}-{args.stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"scenario {args.scenario}  (warmup steps dropped: {args.warmup_steps})")
    for what in ("prepare_input", "step"):
        d = deltas.get(what)
        if d is None:
            print(f"  {what:<14} unavailable (missing run or column)")
            continue
        parts = [
            f"{m}={d[f'{m}_us']:.1f}us"
            if d.get(f"{m}_us") is not None
            else f"{m}=n/a"
            for m in modes
        ]
        steps = []
        for left, right in zip(modes, modes[1:]):
            up = d.get(f"{right}_minus_{left}_us")
            pct = d.get(f"{right}_vs_{left}_pct")
            steps.append(
                f"{right}-{left}={up:+.1f}us ({pct:+.2f}%)"
                if up is not None
                else f"{right}-{left}=n/a"
            )
        print(f"  {what:<14} " + "  ".join(parts))
        if steps:
            print(f"  {'':<14} " + "   ".join(steps))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
