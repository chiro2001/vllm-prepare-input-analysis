#!/usr/bin/env python3
"""Turn a LiteProfiler log into per-step phase timings.

The log is one row per scope instance::

    <scope name>|<duration us>|<start us>|<tid>|<pid>
    phase: <phase name>|<duration us>|<start us>|<tid>|<pid>

Engine step boundaries are taken from the ``prepare input`` rows of the
selected thread: vLLM v1 calls ``NPUModelRunner.execute_model`` exactly once
per engine step, and the call is wrapped in that scope, so each row is one
step.  Every other scope is assigned to the step whose window contains its
start timestamp, which makes the tool independent of the exact nesting order
the runner happens to use.

Usage:
    parse_phase_timing.py LITE.LOG [--tid TID] [--warmup-steps N]
                          [--json OUT.json] [--csv OUT.csv] [--top N]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import json
import pathlib
import statistics
import sys

STEP_SCOPE = "prepare input"
ANCHOR_SCOPE = "Step:Model"
PHASE_PREFIX = "phase: "
PHASE_ORDER = ("prefill", "prefill+decode", "decode", "idle")


def parse_log(path: pathlib.Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    malformed = 0
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            malformed += 1
            continue
        name, dur, start, tid, pid = parts
        try:
            rows.append(
                {
                    "name": name,
                    "dur_us": float(dur),
                    "start_us": int(start),
                    "tid": int(tid),
                    "pid": int(pid),
                }
            )
        except ValueError:
            malformed += 1
    rows.sort(key=lambda row: (row["tid"], row["start_us"], row["dur_us"]))
    return rows, malformed


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = p * (len(ordered) - 1)
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        frac = rank - low
        return ordered[low] * (1 - frac) + ordered[high] * frac

    return {
        "n": len(ordered),
        "mean_us": statistics.fmean(ordered),
        "p50_us": pct(0.50),
        "p90_us": pct(0.90),
        "p99_us": pct(0.99),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "sum_us": sum(ordered),
    }


def label_for(start_us: int, bands: list[dict]) -> str:
    if not bands:
        return "unknown"
    starts = [band["start_us"] for band in bands]
    position = bisect.bisect_right(starts, start_us) - 1
    for candidate in (position, position + 1):
        if 0 <= candidate < len(bands):
            band = bands[candidate]
            if band["start_us"] <= start_us < band["end_us"]:
                return band["phase"]
    if position >= 0:
        return bands[position]["phase"]
    return bands[0]["phase"]


def assign_window(row_start: int, window_starts: list[int]) -> tuple[int | None, bool]:
    """Index of the step whose window contains ``row_start``, plus an orphan flag.

    The window is a half-open interval starting at the step's *anchor* row, so
    it is the engine step itself rather than the prepare-input call.  An
    enclosing scope that opens a hair before the anchor (a ``Step:*`` wrapper,
    or the very first ``Step:Schedule`` of the collection window) is an orphan
    relative to the first window; those are folded into step 0 and counted, so
    the number is visible instead of silently wrong.
    """
    if not window_starts:
        return None, False
    position = bisect.bisect_right(window_starts, row_start) - 1
    if position < 0:
        return 0, True
    return position, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--tid", type=int, default=None,
                        help="restrict to one OS thread id (default: the thread with the most "
                             "'prepare input' rows)")
    parser.add_argument("--warmup-steps", type=int, default=2,
                        help="leading steps reported separately and excluded from the means")
    parser.add_argument("--anchor", choices=("auto", "step-model", "prepare-input"),
                        default="auto",
                        help="which per-step scope defines the step window "
                             "(default: Step:Model when present, else 'prepare input')")
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)

    rows, malformed = parse_log(args.log)
    if not rows:
        print(f"ERROR: no usable rows in {args.log}", file=sys.stderr)
        return 1

    by_tid: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_tid[row["tid"]].append(row)
    step_counts = {
        tid: sum(1 for row in items if row["name"] == STEP_SCOPE)
        for tid, items in by_tid.items()
    }
    if args.tid is None:
        tid = max(step_counts, key=lambda key: step_counts[key])
        if step_counts[tid] == 0:
            tid = max(by_tid, key=lambda key: len(by_tid[key]))
    else:
        tid = args.tid
    tid_rows = by_tid.get(tid, [])
    if not tid_rows:
        print(f"ERROR: tid {tid} has no rows", file=sys.stderr)
        return 1
    pid = tid_rows[0]["pid"]

    bands = []
    for row in tid_rows:
        if row["name"].startswith(PHASE_PREFIX):
            bands.append(
                {
                    "phase": row["name"][len(PHASE_PREFIX):],
                    "start_us": row["start_us"],
                    "end_us": row["start_us"] + row["dur_us"],
                }
            )

    step_calls = [row for row in tid_rows if row["name"] == STEP_SCOPE]
    if not step_calls:
        print(f"ERROR: no '{STEP_SCOPE}' rows on tid {tid}", file=sys.stderr)
        return 1

    anchors = [row for row in tid_rows if row["name"] == ANCHOR_SCOPE]
    if args.anchor == "step-model":
        anchor_rows, anchor_name = anchors, ANCHOR_SCOPE
    elif args.anchor == "prepare-input":
        anchor_rows, anchor_name = step_calls, STEP_SCOPE
    else:
        anchor_rows = anchors if len(anchors) == len(step_calls) else step_calls
        anchor_name = ANCHOR_SCOPE if len(anchors) == len(step_calls) else STEP_SCOPE
    if not anchor_rows:
        anchor_rows, anchor_name = step_calls, STEP_SCOPE

    step_windows = []
    for index, row in enumerate(anchor_rows):
        end = (
            anchor_rows[index + 1]["start_us"] if index + 1 < len(anchor_rows) else None
        )
        step_windows.append({"index": index, "start_us": row["start_us"], "end_us": end})

    # Assign every row on the thread to the step that contains its start.
    per_step: list[dict] = [
        {"index": window["index"], "start_us": window["start_us"], "scopes": {}}
        for window in step_windows
    ]
    window_starts = [window["start_us"] for window in step_windows]
    orphan_rows = 0
    for row in tid_rows:
        if row["name"].startswith(PHASE_PREFIX):
            continue
        target, orphan = assign_window(row["start_us"], window_starts)
        if target is None:
            continue
        orphan_rows += int(orphan)
        bucket = per_step[target]["scopes"].setdefault(row["name"], 0.0)
        per_step[target]["scopes"][row["name"]] = bucket + row["dur_us"]

    # Per-scope statistics, over all steps and per phase.
    scope_values: dict[str, list[float]] = collections.defaultdict(list)
    scope_by_phase: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    phase_of_step: list[str] = []
    for item in per_step:
        phase = label_for(item["start_us"], bands)
        phase_of_step.append(phase)
        for name, value in item["scopes"].items():
            scope_values[name].append(value)
            scope_by_phase[name][phase].append(value)

    steady = [item for index, item in enumerate(per_step) if index >= args.warmup_steps]
    warmup = [item for index, item in enumerate(per_step) if index < args.warmup_steps]
    steady_phases = [
        phase for index, phase in enumerate(phase_of_step) if index >= args.warmup_steps
    ]

    def mean_of_scope(items: list[dict], name: str) -> float | None:
        values = [item["scopes"][name] for item in items if name in item["scopes"]]
        return statistics.fmean(values) if values else None

    core_scopes = [STEP_SCOPE, "forward", "post process", "sample_token"]
    report = {
        "log": str(args.log),
        "log_bytes": args.log.stat().st_size,
        "total_rows": len(rows),
        "malformed_rows": malformed,
        "distinct_tids": len(by_tid),
        "step_rows_per_tid": {str(key): value for key, value in sorted(step_counts.items())},
        "scope_matrix_by_tid": {
            str(key): {
                name: count
                for name, count in collections.Counter(
                    row["name"] for row in items if not row["name"].startswith(PHASE_PREFIX)
                ).most_common(8)
            }
            for key, items in sorted(by_tid.items())
        },
        "selected": {
            "tid": tid,
            "pid": pid,
            "n_rows": len(tid_rows),
            "n_steps": len(step_calls),
            "n_anchor_rows": len(anchor_rows),
            "anchor_scope": anchor_name,
            "orphan_rows": orphan_rows,
            "selection": "explicit --tid" if args.tid is not None else "auto: most 'prepare input' rows",
        },
        "phase_bands": [
            {
                "phase": band["phase"],
                "start_us": band["start_us"],
                "end_us": band["end_us"],
                "dur_ms": (band["end_us"] - band["start_us"]) / 1000.0,
            }
            for band in bands
        ],
        "warmup_steps": args.warmup_steps,
        "step_mean_us": {
            name: mean_of_scope(steady, name) for name in core_scopes
        },
        "warmup_step_mean_us": {
            name: mean_of_scope(warmup, name) for name in core_scopes
        },
        "phase_step_counts": {phase: steady_phases.count(phase) for phase in PHASE_ORDER
                              if steady_phases.count(phase)},
        "core_scope_by_phase_us": {
            name: {
                phase: statistics.fmean(values) if values else None
                for phase, values in sorted(scope_by_phase[name].items())
            }
            for name in core_scopes
        },
        "scope_inventory": [
            {
                "scope": name,
                "per_step_mean_us": (
                    statistics.fmean(values) if values else None
                ),
                "rows_per_step": len(values) / max(1, len(per_step)),
                **stats(values),
            }
            for name, values in sorted(
                scope_values.items(), key=lambda kv: -sum(kv[1])
            )[: args.top]
        ],
    }

    # Prepare-input share of the engine step, using the mean of per-step ratios
    # (not the ratio of means) so a single long step cannot dominate.
    model_values = [item["scopes"].get("Step:Model") for item in steady]
    prepare_values = [item["scopes"].get(STEP_SCOPE) for item in steady]
    ratios = [
        prepare / model
        for prepare, model in zip(prepare_values, model_values)
        if prepare is not None and model
    ]
    report["prepare_input_share_of_step_model"] = (
        {
            "mean_of_per_step_ratio": statistics.fmean(ratios),
            "n_steps": len(ratios),
        }
        if ratios
        else None
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        columns = ["step", "phase", "start_us"] + [name.replace(" ", "_") for name in core_scopes] + ["Step:Model"]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for index, item in enumerate(per_step):
                writer.writerow(
                    [index, phase_of_step[index], item["start_us"]]
                    + [item["scopes"].get(name, "") for name in core_scopes]
                    + [item["scopes"].get("Step:Model", "")]
                )

    selected = report["selected"]
    print(f"log                     : {report['log']} ({report['log_bytes']} bytes)")
    print(f"rows                    : {report['total_rows']} usable, {malformed} malformed")
    print(f"threads                 : {report['distinct_tids']} writing to this log")
    print(f"selected thread         : tid={selected['tid']} pid={selected['pid']} "
          f"({selected['n_rows']} rows, {selected['n_steps']} prepare-input calls, "
          f"anchor={selected['anchor_scope']} x{selected['n_anchor_rows']})")
    print(f"step rows per tid       : {report['step_rows_per_tid']}")
    print(f"orphan rows folded to 0 : {selected['orphan_rows']}")
    print(f"phase bands             : "
          + ", ".join(f"{b['phase']}({b['dur_ms']:.1f} ms)" for b in report["phase_bands"]))
    print(f"steady phase step count : {report['phase_step_counts']} "
          f"(warmup {args.warmup_steps} steps excluded)")
    print()
    print("core scopes, mean per step (us)")
    print(f"  {'scope':<16}{'warmup':>12}{'steady':>12}{'rows/step':>10}")
    inventory = {item["scope"]: item for item in report["scope_inventory"]}
    for name in core_scopes:
        item = inventory.get(name, {})
        warm = report["warmup_step_mean_us"].get(name)
        steady_mean = report["step_mean_us"].get(name)
        print(
            f"  {name:<16}"
            f"{('-' if warm is None else f'{warm:.1f}'):>12}"
            f"{('-' if steady_mean is None else f'{steady_mean:.1f}'):>12}"
            f"{item.get('rows_per_step', 0):>10.2f}"
        )
    print()
    print("core scopes, mean per step by phase (us)")
    phases = [
        phase
        for phase in PHASE_ORDER
        if report["phase_step_counts"].get(phase)
        or any(phase in report["core_scope_by_phase_us"][name] for name in core_scopes)
    ]
    header = "  " + f"{'scope':<16}" + "".join(f"{phase:>16}" for phase in phases)
    print(header)
    for name in core_scopes:
        row = f"  {name:<16}"
        for phase in phases:
            value = report["core_scope_by_phase_us"][name].get(phase)
            row += f"{('-' if value is None else f'{value:.1f}'):>16}"
        print(row)
    share = report["prepare_input_share_of_step_model"]
    if share:
        print()
        print(f"prepare input / Step:Model = {share['mean_of_per_step_ratio'] * 100:.1f}% "
              f"(mean of per-step ratios over {share['n_steps']} steps)")
    print()
    print("top scopes by total time on the selected thread")
    print(f"  {'scope':<28}{'rows':>7}{'mean us':>11}{'p90 us':>11}{'total ms':>11}")
    for item in report["scope_inventory"]:
        print(
            f"  {item['scope']:<28}{item['n']:>7}{item['mean_us']:>11.1f}"
            f"{item['p90_us']:>11.1f}{item['sum_us'] / 1000.0:>11.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
