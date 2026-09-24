#!/usr/bin/env python3
"""Turn a LiteProfiler log slice into per-step phase timings (pi-lite-summary-v1).

Row format (one line per finished scope instance, appended by any thread of any
process sharing the file)::

    <scope name>|<duration us>|<start time us>|<tid>|<pid>
    phase: <phase name>|<duration us>|<start time us>|<tid>|<pid>

``start_us`` is epoch microseconds (``time.time_ns() // 1000`` at scope entry),
so it can be aligned with wall-clock logs.

Engine steps
------------
The step windows come from the ``--step-scope`` rows (default ``Step:Model``) of
the *selected tid*; one window is ``[start_us, start_us + dur_us)`` and every
other row of that tid is assigned to the window that contains its ``start_us``.
A row that starts after every window's end is attributed to the nearest
preceding step and counted in ``orphans`` (the contract for this analyzer).

Async batch-queue caveat
------------------------
The engine core wraps *two* different code regions in ``Step:Model`` when async
scheduling is on: the dispatch of ``execute_model`` (which contains ``prepare
input`` / ``forward`` / ``post process``) and the later ``batch_queue.pop();
future.result()`` wait (a few tens of microseconds, no child scope).  A raw log
therefore has about twice as many ``Step:Model`` rows as ``prepare input`` rows.
This analyzer keeps only the *primary* anchors -- the ``Step:Model`` rows that
contain at least one ``prepare input`` row -- as step definitions, and reports
the secondary ones in ``warnings``.  Without that filter ``n_steps`` would
double and the step-duration statistics would mix 5.5 ms dispatch steps with
35 us waits.

Usage::

    analyze_lite.py LITE_LOG [--csv OUT.csv] [--json OUT.json] [--txt OUT.txt]
                   [--tid TID] [--step-scope "Step:Model"] [--warmup-steps N]

Standard library only (the a3-22 system python3 has nothing else guaranteed).
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

SCHEMA = "pi-lite-summary-v1"
PHASE_PREFIX = "phase: "
PREPARE_SCOPE = "prepare input"
CANONICAL_PHASES = ("prefill", "prefill+decode", "decode", "idle")

# steps.csv column -> LiteProfiler scope name.  ``schedule_us`` is special: the
# engine-core wrapper is ``Step:Schedule`` and the individual scheduler phases
# are ``schedule: *``; see the fallback in main().
COLUMN_SCOPE = {
    "prepare_input_us": PREPARE_SCOPE,
    "forward_us": "forward",
    "post_process_us": "post process",
    "sample_token_us": "sample_token",
    "draft_token_us": "draft_token",
    "async_state_update_us": "async_state_update",
    "schedule_us": "Step:Schedule",
}
CSV_HEADER = ["step_idx", "tid", "phase", "start_us", "dur_us"] + list(COLUMN_SCOPE)
SCHEDULE_FALLBACK_PREFIX = "schedule: "

# Scopes whose per-step means feed ``coverage`` (steps.csv columns).
COVERAGE_SCOPES = tuple(COLUMN_SCOPE.values())

SCOPE_INVENTORY_TOP = 40


def r3(value: float) -> float:
    """Round a microsecond value to the 3 decimals the schema promises."""
    return round(float(value), 3)


def parse_log(path: pathlib.Path) -> tuple[list[dict], int]:
    """Read the log; return (rows, malformed_count).

    Malformed = a non-empty line that is not ``name|dur|start|tid|pid`` with
    numeric dur/start/tid/pid.  Blank lines are ignored, not counted.
    """
    rows: list[dict] = []
    malformed = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
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
    rows.sort(key=lambda row: (row["start_us"], row["dur_us"]))
    return rows, malformed


def quantile(ordered: list[float], p: float) -> float:
    """Linear-interpolated percentile over an already sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    rank = p * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def scope_stats(values: list[float]) -> dict:
    """The fixed 8-key per-scope statistics block (empty -> n=0 only)."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50_us": r3(quantile(ordered, 0.50)),
        "p90_us": r3(quantile(ordered, 0.90)),
        "p99_us": r3(quantile(ordered, 0.99)),
        "mean_us": r3(statistics.fmean(ordered)),
        "min_us": r3(ordered[0]),
        "max_us": r3(ordered[-1]),
        "sum_us": r3(sum(ordered)),
    }


def step_dur_stats(values: list[float]) -> dict:
    """The fixed 4-key step-duration statistics block."""
    if not values:
        return {"p50_us": None, "p90_us": None, "p99_us": None, "mean_us": None}
    ordered = sorted(values)
    return {
        "p50_us": r3(quantile(ordered, 0.50)),
        "p90_us": r3(quantile(ordered, 0.90)),
        "p99_us": r3(quantile(ordered, 0.99)),
        "mean_us": r3(statistics.fmean(ordered)),
    }


def label_step_phase(start_us: int, bands: list[dict]) -> str:
    """Phase of a step, using the band that contains its start.

    A start in a gap the bands do not cover inherits the nearest previous band,
    and a start before every band inherits the first band -- the same rule the
    sibling ``parse_phase_timing.py`` uses, so both tools label a step
    identically.
    """
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


def select_anchors(
    tid_rows: list[dict], step_scope: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Split the step-scope rows into primary anchors plus warnings.

    Primary = the rows that contain at least one ``prepare input`` row; they are
    one per engine step even on the async batch-queue path (two ``Step:Model``
    scopes per iteration).  When the scope is ``prepare input`` itself, or when
    no row contains a prepare row (plain instrumented build), every row is a
    step.
    """
    warnings: list[str] = []
    anchors = [row for row in tid_rows if row["name"] == step_scope]
    if step_scope == PREPARE_SCOPE:
        return anchors, [], warnings
    prepare_rows = [row for row in tid_rows if row["name"] == PREPARE_SCOPE]
    if not anchors:
        warnings.append(
            f"step scope '{step_scope}' has no rows on the selected tid; "
            f"falling back to '{PREPARE_SCOPE}' anchors"
        )
        return prepare_rows, [], warnings
    if not prepare_rows:
        warnings.append(
            f"no '{PREPARE_SCOPE}' rows on the selected tid; every '{step_scope}' "
            f"row is treated as one step"
        )
        return anchors, [], warnings
    prepare_starts = [row["start_us"] for row in prepare_rows]
    primary: list[dict] = []
    secondary: list[dict] = []
    for row in anchors:
        low = bisect.bisect_left(prepare_starts, row["start_us"])
        high = bisect.bisect_left(prepare_starts, row["start_us"] + row["dur_us"])
        (primary if high > low else secondary).append(row)
    if not primary:
        warnings.append(
            f"no '{step_scope}' row contains a '{PREPARE_SCOPE}' row; every "
            f"'{step_scope}' row is treated as one step"
        )
        return anchors, [], warnings
    if secondary:
        warnings.append(
            f"step_scope={step_scope}: {len(anchors)} anchor rows for "
            f"{len(primary)} steps; {len(secondary)} secondary anchor rows were "
            "folded into the nearest preceding step as orphans (the async "
            "batch-queue path emits two Step:Model scopes per engine-core "
            "iteration: the execute_model dispatch and the batch_queue wait). "
            "Step windows use the dispatch anchors only."
        )
    return primary, secondary, warnings


def build_windows(anchors: list[dict], mode: str) -> tuple[list[int], list[int]]:
    """Window starts/ends for the selected anchors.

    ``scope`` (the contract's rule): ``[start, start + dur)`` of the anchor row
    itself, i.e. one window per scope instance.

    ``next`` (additive, opt-in): ``[start_i, start_{i+1})``, the last one ending
    at its own ``start + dur``.  Useful under async scheduling where the model
    phase scope does not cover the whole engine-loop iteration: combine with
    ``--step-scope Step:Schedule`` to get the iteration wall clock.
    """
    starts = [row["start_us"] for row in anchors]
    if mode == "next":
        ends = starts[1:] + [anchors[-1]["start_us"] + max(anchors[-1]["dur_us"], 0.0)]
    else:
        ends = [row["start_us"] + max(row["dur_us"], 0.0) for row in anchors]
    return starts, ends


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--txt", type=pathlib.Path, default=None)
    parser.add_argument(
        "--tid",
        type=int,
        default=None,
        help="engine thread id (default: the tid with the most 'prepare input' "
        "rows, i.e. the engine-core thread)",
    )
    parser.add_argument(
        "--step-scope",
        default="Step:Model",
        help="scope that defines one engine step (default: 'Step:Model'; use "
        "'prepare input' for builds without the Step:* instrumentation)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="leading steps excluded from the by-phase statistics (they stay in "
        "steps.csv); default 2",
    )
    parser.add_argument(
        "--window-mode",
        choices=("scope", "next"),
        default="scope",
        help="'scope' (default, contract): step window = "
        "[anchor.start, anchor.start+anchor.dur); 'next' (additive): step window "
        "= [anchor.start, next_anchor.start). Use 'next' with --step-scope "
        "Step:Schedule on async builds to measure the whole engine-loop "
        "iteration instead of just the model phase.",
    )
    args = parser.parse_args(argv)

    if not args.log.is_file():
        print(f"ERROR: {args.log} is not a file", file=sys.stderr)
        return 1

    rows, malformed = parse_log(args.log)
    if not rows:
        print(f"ERROR: no usable rows in {args.log}", file=sys.stderr)
        return 1

    warnings: list[str] = []
    if malformed:
        warnings.append(f"{malformed} malformed log line(s) ignored")

    # ---------------------------------------------------------------- threads
    tids: dict[str, int] = collections.Counter(row["tid"] for row in rows)
    by_tid: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_tid[row["tid"]].append(row)
    prepare_per_tid = {
        tid: sum(1 for row in items if row["name"] == PREPARE_SCOPE)
        for tid, items in by_tid.items()
    }
    if args.tid is None:
        selected_tid = max(
            by_tid, key=lambda tid: (prepare_per_tid[tid], len(by_tid[tid]), -tid)
        )
        if prepare_per_tid[selected_tid] == 0:
            warnings.append(
                "no 'prepare input' row in this slice; the selected tid is the "
                "one with the most rows instead"
            )
    else:
        selected_tid = args.tid
        if selected_tid not in by_tid:
            print(
                f"ERROR: tid {selected_tid} has no rows (tids present: "
                f"{sorted(tids)})",
                file=sys.stderr,
            )
            return 1
    tid_rows = by_tid[selected_tid]
    tid_rows.sort(key=lambda row: (row["start_us"], row["dur_us"]))

    # ------------------------------------------------------------ phase bands
    bands = [
        {
            "phase": row["name"][len(PHASE_PREFIX):],
            "start_us": row["start_us"],
            "end_us": int(row["start_us"] + max(row["dur_us"], 0.0)),
        }
        for row in tid_rows
        if row["name"].startswith(PHASE_PREFIX)
    ]
    if not bands:
        warnings.append(
            "no 'phase: ' rows on the selected tid; every step is labelled "
            "'unknown'"
        )

    # ---------------------------------------------------------------- anchors
    anchors, _secondary, anchor_warnings = select_anchors(tid_rows, args.step_scope)
    warnings.extend(anchor_warnings)
    if not anchors:
        print(
            f"ERROR: no '{args.step_scope}' and no '{PREPARE_SCOPE}' rows on tid "
            f"{selected_tid}; cannot define steps",
            file=sys.stderr,
        )
        return 1
    if len(anchors) <= args.warmup_steps:
        warnings.append(
            f"only {len(anchors)} step(s) in this slice while --warmup-steps="
            f"{args.warmup_steps}; the steady-state statistics are empty"
        )

    window_starts, window_ends = build_windows(anchors, args.window_mode)
    per_step: list[dict] = [
        {
            "index": index,
            "start_us": row["start_us"],
            "dur_us": (
                row["dur_us"]
                if args.window_mode == "scope"
                else max(window_ends[index] - row["start_us"], 0.0)
            ),
            "anchor_us": row["dur_us"],
            "tid": selected_tid,
            "scopes": {},
        }
        for index, row in enumerate(anchors)
    ]

    # --------------------------------------------------- assign rows to steps
    orphans = 0
    for row in tid_rows:
        if row["name"].startswith(PHASE_PREFIX):
            continue
        target = bisect.bisect_right(window_starts, row["start_us"]) - 1
        if target < 0:
            target = 0
            orphans += 1
        elif row["start_us"] >= window_ends[target]:
            # starts after this step's end: nearest preceding step, still counted
            orphans += 1
        scopes = per_step[target]["scopes"]
        scopes[row["name"]] = scopes.get(row["name"], 0.0) + row["dur_us"]

    # ``schedule_us``: prefer the engine-core wrapper ``Step:Schedule``; if the
    # build only has the scheduler's internal phases, sum those instead.
    schedule_scope = COLUMN_SCOPE["schedule_us"]
    if not any(schedule_scope in item["scopes"] for item in per_step):
        fallback_names = {
            name
            for item in per_step
            for name in item["scopes"]
            if name.startswith(SCHEDULE_FALLBACK_PREFIX)
        }
        if fallback_names:
            warnings.append(
                f"no '{schedule_scope}' rows; schedule_us sums "
                f"{len(fallback_names)} '{SCHEDULE_FALLBACK_PREFIX}*' scope(s) "
                "instead (not directly comparable with Step:Schedule)"
            )

    for item in per_step:
        scopes = item["scopes"]
        item["phase"] = label_step_phase(item["start_us"], bands)
        for column, scope in COLUMN_SCOPE.items():
            if scope in scopes:
                item[column] = scopes[scope]
            elif scope == schedule_scope:
                value = sum(
                    duration
                    for name, duration in scopes.items()
                    if name.startswith(SCHEDULE_FALLBACK_PREFIX)
                )
                item[column] = value if value else None
            else:
                item[column] = None
        measured = sum(
            item[column] for column in COLUMN_SCOPE if item[column] is not None
        )
        item["other_us"] = item["dur_us"] - measured

    negative_other = sum(1 for item in per_step if item["other_us"] < 0)
    if negative_other:
        warnings.append(
            f"{negative_other} of {len(per_step)} steps have measured scopes "
            "summing to more than the step scope itself (scopes that run outside "
            "the step window are folded in by the orphan rule); other_us is "
            "clamped to 0 for those steps"
        )
    for item in per_step:
        if item["other_us"] < 0:
            item["other_us"] = 0.0

    # ----------------------------------------------------------- statistics
    steady = [item for item in per_step if item["index"] >= args.warmup_steps]

    def scopes_of(items: list[dict]) -> dict[str, list[float]]:
        values: dict[str, list[float]] = collections.defaultdict(list)
        for item in items:
            for name, duration in item["scopes"].items():
                values[name].append(duration)
        return values

    def block(items: list[dict]) -> dict:
        values = scopes_of(items)
        entry: dict = {"n_steps": len(items)}
        entry["step_dur"] = step_dur_stats([item["dur_us"] for item in items])
        for scope in ("prepare input", "forward", "post process", "sample_token",
                      "draft_token"):
            entry[scope] = scope_stats(values.get(scope, []))
        measured = 0.0
        for scope in COVERAGE_SCOPES:
            stats = scope_stats(values.get(scope, []))
            if stats.get("n"):
                measured += stats["mean_us"]
        step_mean = entry["step_dur"]["mean_us"]
        entry["coverage"] = {
            "sum_measured_mean_us": r3(measured),
            "ratio_of_step_dur": r3(measured / step_mean) if step_mean else None,
        }
        return entry

    phases_steady = collections.Counter(item["phase"] for item in steady)
    by_phase: dict[str, dict] = {}
    for phase in CANONICAL_PHASES:
        if phases_steady.get(phase):
            by_phase[phase] = block([i for i in steady if i["phase"] == phase])
    for phase in sorted(phases_steady):
        if phase not in by_phase:
            by_phase[phase] = block([i for i in steady if i["phase"] == phase])
    by_phase["all"] = block(steady)

    def share_of(items: list[dict]) -> dict:
        ratios = [
            item["prepare_input_us"] / item["dur_us"]
            for item in items
            if item["prepare_input_us"] is not None and item["dur_us"]
        ]
        if not ratios:
            return {"prepare_over_step_p50": None, "prepare_over_step_mean": None}
        ordered = sorted(ratios)
        return {
            "prepare_over_step_p50": r3(quantile(ordered, 0.50)),
            "prepare_over_step_mean": r3(statistics.fmean(ordered)),
        }

    share = {
        phase: share_of([i for i in steady if i["phase"] == phase])
        for phase in by_phase
        if phase != "all"
    }
    share["all"] = share_of(steady)

    # scope inventory: over *all* steps (warm-up included) so that the numbers
    # line up with parse_phase_timing.py's scope inventory.
    inventory_values = scopes_of(per_step)
    inventory = [
        {
            "scope": name,
            "n": len(values),
            "mean_us": r3(statistics.fmean(values)),
            "p50_us": r3(quantile(sorted(values), 0.50)),
            "p90_us": r3(quantile(sorted(values), 0.90)),
            "sum_us": r3(sum(values)),
        }
        for name, values in inventory_values.items()
    ]
    inventory.sort(key=lambda item: -item["sum_us"])
    if len(inventory) > SCOPE_INVENTORY_TOP:
        warnings.append(
            f"scope_inventory truncated to the top {SCOPE_INVENTORY_TOP} scopes "
            f"by total time ({len(inventory)} seen); its values are per-step sums "
            "over all steps, warm-up included"
        )
        inventory = inventory[:SCOPE_INVENTORY_TOP]
    if not any(item["scope"] == PREPARE_SCOPE for item in inventory):
        warnings.append(
            f"no '{PREPARE_SCOPE}' row lands inside a step window on tid "
            f"{selected_tid}; nothing to attribute to the prepare phase"
        )

    summary = {
        "schema": SCHEMA,
        "lite_log": str(args.log),
        "rows": len(rows),
        "malformed": malformed,
        "selected_tid": selected_tid,
        "tids": {str(tid): count for tid, count in sorted(tids.items())},
        "orphans": orphans,
        "n_steps": len(per_step),
        "steps_by_phase": {
            phase: phases_steady.get(phase, 0) for phase in CANONICAL_PHASES
        },
        "by_phase": by_phase,
        "share": share,
        "scope_inventory": inventory,
        "warnings": warnings,
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for item in per_step:
                writer.writerow(
                    [
                        item["index"],
                        item["tid"],
                        item["phase"],
                        item["start_us"],
                        r3(item["dur_us"]),
                    ]
                    + [
                        "" if item[column] is None else r3(item[column])
                        for column in COLUMN_SCOPE
                    ]
                    + [r3(item["other_us"])]
                )

    text = render_text(summary, args)
    if args.txt:
        args.txt.parent.mkdir(parents=True, exist_ok=True)
        args.txt.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


def render_text(summary: dict, args) -> str:
    """Human-readable rendering of the summary (also the --txt payload)."""
    out: list[str] = []
    add = out.append
    add(f"lite log                : {summary['lite_log']}")
    add(
        f"rows                    : {summary['rows']} usable, "
        f"{summary['malformed']} malformed"
    )
    add(
        "threads                 : "
        + ", ".join(
            f"tid {tid}={count}" for tid, count in summary["tids"].items()
        )
    )
    add(
        f"selected thread         : tid={summary['selected_tid']} "
        f"({summary['n_steps']} steps from scope '{args.step_scope}')"
    )
    add(f"orphan rows folded      : {summary['orphans']}")
    add(
        "steps by phase (steady) : "
        + ", ".join(f"{k}={v}" for k, v in summary["steps_by_phase"].items())
    )
    add(
        f"warmup steps excluded   : {args.warmup_steps} of "
        f"{summary['n_steps']} steps in the slice"
    )
    add("")
    phases = [key for key in summary["by_phase"] if key != "all"] + ["all"]
    add("per-phase statistics (steady state, us)")
    add(f"  {'scope':<18}" + "".join(f"{phase:>14}" for phase in phases))
    for name in ("step_dur", PREPARE_SCOPE, "forward", "post process",
                 "sample_token", "draft_token"):
        row = f"  {name:<18}"
        for phase in phases:
            entry = summary["by_phase"].get(phase) or {}
            if name == "step_dur":
                value = (entry.get("step_dur") or {}).get("p50_us")
            else:
                value = (entry.get(name) or {}).get("mean_us")
            row += f"{('-' if value is None else f'{value:.1f}'):>14}"
        add(row)
    add(
        f"  {'n_steps':<18}"
        + "".join(
            f"{summary['by_phase'].get(phase, {}).get('n_steps', 0):>14}"
            for phase in phases
        )
    )
    add("  (step_dur prints p50; every other row prints the mean us/step)")
    add("")
    add("prepare input / step_dur (share_step: mean of per-step ratios, p50)")
    for phase, values in summary["share"].items():
        p50 = values["prepare_over_step_p50"]
        mean = values["prepare_over_step_mean"]
        add(
            f"  {phase:<18}"
            f"p50={('-' if p50 is None else f'{p50 * 100:.1f}%'):>8}  "
            f"mean={('-' if mean is None else f'{mean * 100:.1f}%'):>8}"
        )
    add("")
    add("coverage of the step scope by the measured scopes")
    for phase in phases:
        coverage = (summary["by_phase"].get(phase) or {}).get("coverage") or {}
        ratio = coverage.get("ratio_of_step_dur")
        add(
            f"  {phase:<18} measured={coverage.get('sum_measured_mean_us')} us "
            f"({'-' if ratio is None else f'{ratio * 100:.1f}%'} of step_dur)"
        )
    add("")
    add("scope inventory (per-step sums over all steps, warm-up included)")
    add(
        f"  {'scope':<30}{'steps':>7}{'mean us':>11}{'p50 us':>11}{'total ms':>11}"
    )
    for item in summary["scope_inventory"]:
        add(
            f"  {item['scope']:<30}{item['n']:>7}{item['mean_us']:>11.1f}"
            f"{item['p50_us']:>11.1f}{item['sum_us'] / 1000.0:>11.2f}"
        )
    if summary["warnings"]:
        add("")
        add("warnings")
        for warning in summary["warnings"]:
            add(f"  ! {warning}")
    add("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
