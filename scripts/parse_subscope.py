#!/usr/bin/env python3
"""Parse pi_subscope raw logs into per-step and summary tables.

Input
-----
The raw log written by ``instrument/pi_subscope.py`` (one row per
``(step, scope)``)::

    step,phase,scope,self_us,incl_us,max_us,n_calls,wall_us,num_reqs,total_tokens,tid,pid

``wall_us`` is the wall time of the whole ``prepare input`` scope of that
step; ``self_us`` is the exclusive time attributed to the scope.  The
``pi: TOTAL`` row carries the wall time itself.

Outputs
-------
``per_step_subscope.csv``
    ``step,phase,subscope,duration_us,...`` — the long-form per-step table.
``summary.csv``
    one row per sub-scope with mean/median/p90/p99/max, share of the
    ``prepare input`` wall time, plus the canonical ``group`` it belongs to.
``summary_groups.csv``
    the same statistics rolled up into the canonical groups used by
    ``figures/05-subscope-breakdown.svg``.
``meta.json``
    run-level numbers: step counts, wall-time quantiles, the instrumentation
    invariant check (sum of self times + unattributed == TOTAL) and the
    probe-attributable floor.

Usage
-----
    parse_subscope.py RAW.csv --outdir data/subscope/RUN_ID [--json] [--quiet]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

TOTAL = "pi: TOTAL"
UNATTR = "pi: (unattributed)"

# --------------------------------------------------------------------------- #
# Canonical groups: the 8-12 "meaningful" sub-scopes required by the task.
# Every probe name must map to exactly one group.
# --------------------------------------------------------------------------- #
GROUPS: dict[str, tuple[str, ...]] = {
    "synchronize+update_states": (
        "pi: sync_input_prep",
        "pi: update_states",
    ),
    "block_table commit": (
        "pi: in.block_table_commit",
    ),
    "attn_state": (
        "pi: in.attn_state",
    ),
    "positions+token_indices": (
        "pi: in.positions",
        "pi: in.req_indices",
        "pi: in.token_indices_select",
        "pi: in.prev_positions",
        "pi: in.dcp_init",
    ),
    "input_ids/H2D": (
        "pi: in.prepare_input_ids",
        "pi: in.tokens_to_gpu",
        "pi: in.num_computed_tokens",
        "pi: in.seq_lens",
        "pi: in.positions_assembly",
        "pi: in.mrope_positions",
        "pi: in.mrope_drift",
        "pi: in.num_accepted_tokens",
    ),
    "query_start_loc": (
        "pi: in.query_start_loc",
    ),
    "bookkeeping other": (
        # `pi: in.slot_mapping` is a Triton kernel launch: it is bookkeeping in
        # the sense that the *work* happens on the NPU, but its launch cost is
        # real CPU time, so it is reported on its own line inside this group.
        "pi: in.slot_mapping",
        "pi: in.discard_mask",
        "pi: in.optimistic_seq_lens",
        "pi: in.optimistic_correct",
        "pi: in.logits_indices",
        "pi: in.lora",
        "pi: in.lmhead_tp_pad",
        "pi: in.prompt_embeds",
        "pi: in.dcp_rebuild",
        "pi: mamba_preprocess",
        "pi: dsa_positions",
    ),
    "spec decode metadata": (
        "pi: in.spec_decode_metadata",
    ),
    "batch padding decision": (
        "pi: batch_exec_and_padding",
    ),
    "scheduler glue": (
        # Aggregate wrapper whose self time is whatever the `in.*` probes miss.
        "pi: prepare_inputs",
        "pi: tokens_list",
        "pi: phase_classify",
        "pi: ubatch_slices",
        "pi: cascade_attn_prefix_lens",
        "pi: pad_query_start_loc",
        "pi: sanitize_placeholder_ids",
    ),
    "attn metadata": (
        "pi: build_attention_metadata",
        "pi: am.max_seq_len",
        "pi: am.cm_base_pre",
        "pi: am.group_loop",
        "pi: am.builder_build",
        "pi: am.layer_assign",
        # second level: inside AscendAttentionMetadataBuilder.build
        # (attention_v1.py), mounted only when --attn-probes on
        "pi: am.build.split_decodes",
        "pi: am.build.seq_lens_select",
        "pi: am.build.attn_mask",
        "pi: am.build.qsl_h2d",
        "pi: am.build.tolist",
        "pi: am.build.fia_pad",
        "pi: am.build.backend_metadata",
        "pi: am.build.metadata_ctor",
    ),
}

# --------------------------------------------------------------------------- #
# Second level of the "attn metadata" group.  `pi: am.builder_build` is the
# single largest probe in decode, so its children are reported separately as a
# share of the builder itself (parse_subscope.py --attn-breakdown).
# --------------------------------------------------------------------------- #
ATTN_BUILDER = "pi: am.builder_build"
ATTN_BUILDER_CHILDREN = (
    "pi: am.build.split_decodes",
    "pi: am.build.seq_lens_select",
    "pi: am.build.attn_mask",
    "pi: am.build.qsl_h2d",
    "pi: am.build.tolist",
    "pi: am.build.fia_pad",
    "pi: am.build.backend_metadata",
    "pi: am.build.metadata_ctor",
)

GROUP_OF: dict[str, str] = {
    scope: group for group, scopes in GROUPS.items() for scope in scopes
}

# Sub-scopes from the SFA builder belong to their own reporting group so the
# two builder families never get mixed in one bar.  They are *nested* inside
# `pi: am.builder_build`, which is why the attention_v1-only probes showed a
# 92% residual: AscendSFAMetadataBuilder.build() is a different override that
# never calls AscendAttentionMetadataBuilder.build().
GROUPS["attn metadata (SFA)"] = (
    "pi: sfa.build",
    "pi: sfa.slice_inputs",
    "pi: sfa.seq_lens_select",
    "pi: sfa.get_cos_sin",
    "pi: sfa.store_kv_block_metadata",
    "pi: sfa.metadata_ctor",
)
GROUP_OF.update(
    {scope: "attn metadata (SFA)" for scope in GROUPS["attn metadata (SFA)"]}
)

# Sub-scopes from core vLLM's GDN (linear attention) builder.  On a hybrid model
# such as Qwen3.5-0.8B these run in the *same* `pi: am.builder_build` scope as
# the vllm-ascend builders, so they need their own group to be readable.
GROUPS["attn metadata (GDN)"] = (
    "pi: gdn.query_start_loc",
    "pi: gdn.mamba_block_table",
    "pi: gdn.split_decodes",
    "pi: gdn.prefill_metadata",
    # AscendGDNAttentionMetadataBuilder (vllm_ascend/ops/gdn_attn_builder.py) --
    # the class that actually serves Qwen3.5's 18 GDN layers on Ascend.
    "pi: gdnb.treat_single_token",
    "pi: gdnb.compute_num_computed_tokens",
    "pi: gdnb.mamba_block_table",
    "pi: gdnb.split_decodes",
    "pi: gdnb.pad_graph_inputs",
    "pi: gdnb.attach_metadata",
    "pi: gdnb.ctor",
    "pi: gdnb.attach_prefill",
    "pi: gdnb.attach_spec",
    "pi: gdnb.attach_decode",
    "pi: gdnb.build_actual_seq_lengths",
    # `gdnA.*` names used by the first (pre-rename) revision of the
    # AscendGDNAttentionMetadataBuilder probes; kept so an older raw.csv still
    # parses into the same group instead of landing in "unmapped".
    "pi: gdnA.treat_prefills",
    "pi: gdnA.mamba_block_table",
    "pi: gdnA.split_decodes",
    "pi: gdnA.pad_graph_inputs",
    "pi: gdnA.attach_decode",
    "pi: gdnA.ctor",
    "pi: gdnA.prefill_meta",
    "pi: gdnA.tail",
)
GROUP_OF.update(
    {scope: "attn metadata (GDN)" for scope in GROUPS["attn metadata (GDN)"]}
)

# Per-builder-class probes: name pattern `pi: am.builder_build@<ClassName>`.
# They are matched dynamically (not listed) so a new builder class shows up in
# the report without editing this table.
BUILDER_CLASS_PREFIX = "pi: am.builder_build@"


# Display order for the stacked bar chart (inner -> outer).
GROUP_ORDER: list[str] = [
    "synchronize+update_states",
    "block_table commit",
    "attn_state",
    "positions+token_indices",
    "query_start_loc",
    "input_ids/H2D",
    "bookkeeping other",
    "spec decode metadata",
    "batch padding decision",
    "scheduler glue",
    "attn metadata",
    "attn metadata (SFA)",
    "attn metadata (GDN)",
    "attn metadata (builder by class)",
    UNATTR,
]

RAW_FIELDS = [
    "step",
    "phase",
    "scope",
    "self_us",
    "incl_us",
    "max_us",
    "n_calls",
    "wall_us",
    "num_reqs",
    "total_tokens",
    "tid",
    "pid",
]


def quantile(values: list[float], q: float) -> float:
    """Nearest-rank quantile (no numpy dependency, stable for small n)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def read_raw(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in RAW_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        for raw in reader:
            try:
                rows.append(
                    {
                        "step": int(raw["step"]),
                        "phase": raw["phase"],
                        "scope": raw["scope"],
                        "self_us": float(raw["self_us"]),
                        "incl_us": float(raw["incl_us"]),
                        "max_us": float(raw["max_us"]),
                        "n_calls": int(raw["n_calls"]),
                        "wall_us": float(raw["wall_us"]),
                        "num_reqs": int(raw["num_reqs"]),
                        "total_tokens": int(raw["total_tokens"]),
                        "tid": int(raw["tid"]),
                        "pid": int(raw["pid"]),
                    }
                )
            except ValueError:
                continue
    return rows


def select_worker(rows: list[dict], tid: int | None) -> tuple[list[dict], int]:
    """Keep only the thread that owns the most ``pi: TOTAL`` rows."""
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        if row["scope"] == TOTAL:
            counts[row["tid"]] += 1
    if not counts:
        return rows, -1
    if tid is not None:
        return [r for r in rows if r["tid"] == tid], tid
    best = max(counts, key=lambda k: (counts[k], k))
    return [r for r in rows if r["tid"] == best], best


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("raw", type=pathlib.Path, help="raw pi-subscope CSV")
    ap.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=None,
        help="output directory (default: dirname of RAW)",
    )
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="drop the first N steps (graph capture / cache warm-up)",
    )
    ap.add_argument("--tid", type=int, default=None, help="force a worker thread id")
    ap.add_argument(
        "--engine-steps",
        type=pathlib.Path,
        default=None,
        help="optional CSV from parse_phase_timing.py (phase_timing.csv) used "
        "to cross-check the prepare-input wall time",
    )
    ap.add_argument(
        "--probe-floor-us",
        type=float,
        default=None,
        help="measured cost of one probe (from instrument/probe_overhead.py); "
        "written into meta.json to flag scopes dominated by the probe",
    )
    ap.add_argument("--json", action="store_true", help="print meta.json to stdout")
    ap.add_argument(
        "--attn-breakdown",
        action="store_true",
        help="also emit attn_builder_breakdown.csv: the children of "
        "`pi: am.builder_build` as a share of the builder itself (requires the "
        "second-level probes, i.e. --attn-probes on)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv[1:])

    outdir = args.outdir or args.raw.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_raw(args.raw)
    if not rows:
        raise SystemExit(f"{args.raw}: no parseable rows")
    rows, tid = select_worker(rows, args.tid)

    max_step = max(r["step"] for r in rows)
    kept = [r for r in rows if r["step"] > args.warmup_steps]
    dropped = max_step - max((r["step"] for r in kept), default=0)

    per_step_path = outdir / "per_step_subscope.csv"
    with open(per_step_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "step",
                "phase",
                "subscope",
                "duration_us",
                "incl_us",
                "n_calls",
                "wall_us",
                "num_reqs",
                "total_tokens",
            ]
        )
        for row in sorted(kept, key=lambda r: (r["step"], r["scope"])):
            writer.writerow(
                [
                    row["step"],
                    row["phase"],
                    row["scope"],
                    f"{row['self_us']:.3f}",
                    f"{row['incl_us']:.3f}",
                    row["n_calls"],
                    f"{row['wall_us']:.3f}",
                    row["num_reqs"],
                    row["total_tokens"],
                ]
            )

    by_scope: dict[str, list[dict]] = defaultdict(list)
    step_wall: dict[int, float] = {}
    step_phase: dict[int, str] = {}
    for row in kept:
        by_scope[row["scope"]].append(row)
        if row["scope"] == TOTAL:
            step_wall[row["step"]] = row["self_us"]
            step_phase[row["step"]] = row["phase"]

    n_steps = len(step_wall) or len({r["step"] for r in kept})
    walls = list(step_wall.values())
    wall_p50 = quantile(walls, 0.5)

    def summarise(scope_rows: dict[str, list[dict]]) -> list[dict]:
        out = []
        for scope, rows_ in scope_rows.items():
            if scope == TOTAL:
                continue
            self_us = [r["self_us"] for r in rows_]
            incl = [r["incl_us"] for r in rows_]
            calls = [r["n_calls"] for r in rows_]
            out.append(
                {
                    "scope": scope,
                    "group": (
                        "attn metadata (builder by class)"
                        if scope.startswith(BUILDER_CLASS_PREFIX)
                        else GROUP_OF.get(scope, "unmapped")
                    ),
                    "n_steps_present": len(rows_),
                    "steps_present_pct": round(
                        100.0 * len(rows_) / n_steps if n_steps else 0.0, 1
                    ),
                    "self_us_mean": round(statistics.fmean(self_us), 3),
                    "self_us_p50": round(quantile(self_us, 0.5), 3),
                    "self_us_p90": round(quantile(self_us, 0.9), 3),
                    "self_us_p99": round(quantile(self_us, 0.99), 3),
                    "self_us_max": round(max(self_us), 3),
                    "incl_us_p50": round(quantile(incl, 0.5), 3),
                    "n_calls_per_step": round(statistics.fmean(calls), 2),
                    "share_pct_of_prepare_input_p50": round(
                        100.0 * quantile(self_us, 0.5) / wall_p50 if wall_p50 else 0.0,
                        2,
                    ),
                }
            )
        return sorted(out, key=lambda r: -r["self_us_p50"])

    scope_rows = {s: r for s, r in by_scope.items() if s != TOTAL}
    summary = summarise(scope_rows)

    summary_path = outdir / "summary.csv"
    if summary:
        fieldnames = list(summary[0].keys())
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)

    # ---- canonical groups, including the unattributed remainder ----------- #
    # The distribution is built over *whole steps*: a group's per-step total is
    # the sum of every scope in it, and a step where the group is absent
    # contributes 0.0 (that is what "median cost of this group per step" means).
    per_step_group: dict[str, dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    # A `pi: am.builder_build@Foo` probe is a *child* of
    # `pi: am.builder_build`.  Self times are exclusive, so the two never
    # double count: `builder_build.self_us` already excludes the child.
    # They are mapped to their own group so the bar chart stays readable, and
    # summarised per class in builder_by_class.csv.
    for scope, rows_ in scope_rows.items():
        group = (
            "attn metadata (builder by class)"
            if scope.startswith(BUILDER_CLASS_PREFIX)
            else GROUP_OF.get(scope, "unmapped")
        )
        for row in rows_:
            per_step_group[group][row["step"]] += row["self_us"]

    per_step_attributed: dict[int, float] = defaultdict(float)
    for row in kept:
        if row["scope"] != TOTAL:
            per_step_attributed[row["step"]] += row["self_us"]
    group_self: dict[str, list[float]] = {}
    for group in set(per_step_group) | set(GROUP_ORDER):
        if group == UNATTR:
            continue
        group_self[group] = [
            per_step_group[group].get(step, 0.0) for step in step_wall
        ]
    group_self[UNATTR] = [
        step_wall[s] - per_step_attributed.get(s, 0.0) for s in step_wall
    ]

    groups = []
    for group, values in group_self.items():
        n_present = sum(
            1 for step in step_wall if per_step_group.get(group, {}).get(step, 0.0) > 0
        )
        groups.append(
            {
                "group": group,
                "n_steps_present": n_present if group != UNATTR else n_steps,
                "steps_present_pct": round(100.0 * n_present / n_steps, 1)
                if n_steps
                else 0.0,
                "self_us_p50": round(quantile(values, 0.5), 3),
                "self_us_p90": round(quantile(values, 0.9), 3),
                "self_us_p99": round(quantile(values, 0.99), 3),
                "self_us_mean": round(statistics.fmean(values), 3),
                "share_pct_of_prepare_input_p50": round(
                    100.0 * quantile(values, 0.5) / wall_p50 if wall_p50 else 0.0, 2
                ),
            }
        )
    order = {name: i for i, name in enumerate(GROUP_ORDER)}
    groups.sort(key=lambda r: order.get(r["group"], len(order)))
    groups_path = outdir / "summary_groups.csv"
    with open(groups_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(groups[0].keys()))
        writer.writeheader()
        writer.writerows(groups)

    # ---- invariant check -------------------------------------------------- #
    unattr = group_self[UNATTR]
    residuals = [
        abs(per_step_attributed.get(s, 0.0) + unattr[i] - step_wall[s])
        for i, s in enumerate(step_wall)
    ]
    meta = {
        "raw": str(args.raw),
        "worker_tid": tid,
        "steps_total": max_step,
        "steps_analysed": n_steps,
        "steps_dropped_as_warmup": dropped,
        "n_probe_names": len(scope_rows),
        "prepare_input_wall_us": {
            "p50": round(wall_p50, 3),
            "p90": round(quantile(walls, 0.9), 3),
            "p99": round(quantile(walls, 0.99), 3),
            "mean": round(statistics.fmean(walls), 3) if walls else None,
            "min": round(min(walls), 3) if walls else None,
        },
        "phases": sorted({p for p in step_phase.values()}),
        "unattributed_us_p50": round(quantile(unattr, 0.5), 3),
        "unattributed_pct_p50": round(
            100.0 * quantile(unattr, 0.5) / wall_p50 if wall_p50 else 0.0, 2
        ),
        "invariant_max_abs_residual_us": round(max(residuals), 6) if residuals else None,
        "probe_floor_us": args.probe_floor_us,
        "scopes_below_probe_floor": (
            [
                r["scope"]
                for r in summary
                if args.probe_floor_us is not None
                and r["self_us_p50"] < args.probe_floor_us
            ]
            if args.probe_floor_us is not None
            else []
        ),
    }
    if args.engine_steps and args.engine_steps.is_file():
        meta["cross_check_engine_steps"] = cross_check(args.engine_steps, step_wall)

    # Structural notes captured by pi_note() inside the worker (attn group
    # identity, mask path).  Absent for runs that mounted the probes without
    # the second-level attention file.
    notes_path = outdir / "probe_notes.json"
    if notes_path.is_file():
        try:
            meta["probe_notes"] = json.loads(notes_path.read_text())
        except json.JSONDecodeError:
            meta["probe_notes"] = {"error": f"{notes_path} is not valid JSON"}

    # ---- per-builder-class breakdown (the "who is the 940 us?" table) ----- #
    builder_rows: list[dict] = []
    for scope in sorted(step_scopes := {
        r["scope"] for r in kept if r["scope"].startswith(BUILDER_CLASS_PREFIX)
    }):
        rows_ = [r for r in kept if r["scope"] == scope]
        self_us = [r["self_us"] for r in rows_]
        per_step_total: dict[int, float] = defaultdict(float)
        for row in rows_:
            per_step_total[row["step"]] += row["self_us"]
        builder_rows.append(
            {
                "builder_class": scope[len(BUILDER_CLASS_PREFIX):],
                "n_calls_per_step": round(statistics.fmean(r["n_calls"] for r in rows_), 2),
                "self_us_p50": round(quantile(self_us, 0.5), 3),
                "self_us_p90": round(quantile(self_us, 0.9), 3),
                "self_us_p99": round(quantile(self_us, 0.99), 3),
                "self_us_max": round(max(self_us), 3),
                "self_us_per_call_p50": round(
                    quantile(self_us, 0.5)
                    / max(1.0, statistics.fmean(r["n_calls"] for r in rows_)),
                    3,
                ),
                "share_pct_of_prepare_input_p50": round(
                    100.0 * quantile(self_us, 0.5) / wall_p50 if wall_p50 else 0.0, 2
                ),
            }
        )
    if builder_rows:
        builder_rows.sort(key=lambda r: -r["self_us_p50"])
        builder_path = outdir / "builder_by_class.csv"
        with open(builder_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(builder_rows[0].keys()))
            writer.writeheader()
            writer.writerows(builder_rows)
        meta["builder_by_class"] = {
            "csv": str(builder_path),
            "total_us_p50": round(sum(r["self_us_p50"] for r in builder_rows), 3),
            "note": "self_us is exclusive; the sum is the total time spent inside "
                    "`pi: am.builder_build` (its own self_us is ~0 when the "
                    "per-class probes are mounted)",
        }
        if not args.quiet:
            print("builder by class (p50 us/step, exclusive):")
            for row in builder_rows:
                print(
                    f"  {row['builder_class']:<44} {row['self_us_p50']:9.3f} us "
                    f"({row['n_calls_per_step']:.1f} calls/step, "
                    f"{row['self_us_per_call_p50']:7.1f} us/call, "
                    f"{row['share_pct_of_prepare_input_p50']:5.1f}% of prepare)"
                )
            print(f"wrote {builder_path}")

    if args.attn_breakdown:
        builder = [r for r in kept if r["scope"] == ATTN_BUILDER]
        builder_wall = {
            r["step"]: r["incl_us"] for r in builder
        }
        child_rows = []
        for scope in ATTN_BUILDER_CHILDREN:
            values = [
                r["self_us"] for r in kept if r["scope"] == scope
            ]
            if not values:
                continue
            child_rows.append(
                {
                    "subscope": scope,
                    "self_us_p50": round(quantile(values, 0.5), 3),
                    "self_us_p90": round(quantile(values, 0.9), 3),
                    "self_us_p99": round(quantile(values, 0.99), 3),
                    "share_pct_of_builder_p50": round(
                        100.0
                        * quantile(values, 0.5)
                        / quantile(list(builder_wall.values()), 0.5),
                        2,
                    )
                    if builder_wall
                    else None,
                }
            )
        child_rows.sort(key=lambda r: -r["self_us_p50"])
        accounted = sum(r["self_us_p50"] for r in child_rows)
        builder_p50 = quantile(list(builder_wall.values()), 0.5) if builder_wall else 0.0
        child_rows.append(
            {
                "subscope": "pi: am.builder_build (unattributed residual)",
                "self_us_p50": round(max(0.0, builder_p50 - accounted), 3),
                "self_us_p90": None,
                "self_us_p99": None,
                "share_pct_of_builder_p50": round(
                    100.0 * max(0.0, builder_p50 - accounted) / builder_p50, 2
                )
                if builder_p50
                else None,
            }
        )
        attn_path = outdir / "attn_builder_breakdown.csv"
        with open(attn_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(child_rows[0].keys()))
            writer.writeheader()
            writer.writerows(child_rows)
        meta["attn_builder"] = {
            "scope": ATTN_BUILDER,
            "incl_us_p50": round(builder_p50, 3),
            "children_accounted_us_p50": round(accounted, 3),
            "residual_us_p50": round(max(0.0, builder_p50 - accounted), 3),
            "breakdown_csv": str(attn_path),
            "n_steps_with_builder": len(builder_wall),
        }
        if not args.quiet:
            print(f"attn builder p50 : {builder_p50:.1f} us "
                  f"({100.0 * builder_p50 / wall_p50:.1f}% of prepare input)")
            for row in child_rows:
                print(f"  {row['subscope']:<48} {row['self_us_p50']:9.3f} us  "
                      f"{row['share_pct_of_builder_p50']:5.1f}%")
            print(f"wrote {attn_path}")

    (outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if args.json:
        print(json.dumps(meta, indent=2))
    elif not args.quiet:
        print(f"steps analysed      : {n_steps} (tid {tid})")
        print(f"prepare input wall  : p50 {wall_p50:.1f} us   p99 "
              f"{meta['prepare_input_wall_us']['p99']:.1f} us")
        print(f"unattributed        : {meta['unattributed_us_p50']:.1f} us "
              f"({meta['unattributed_pct_p50']:.1f} %)")
        print("top sub-scopes (p50 us, share of prepare input):")
        for row in summary[:12]:
            print(
                f"  {row['scope']:<34} {row['self_us_p50']:9.3f}  "
                f"{row['share_pct_of_prepare_input_p50']:5.1f}%  "
                f"({row['steps_present_pct']:.0f}% of steps)"
            )
        print(f"wrote {per_step_path}")
        print(f"wrote {summary_path}")
        print(f"wrote {groups_path}")
    return 0


def cross_check(engine_steps: pathlib.Path, step_wall: dict[int, float]) -> dict:
    """Compare our per-step wall time with LiteProfiler's prepare-input rows."""
    try:
        with open(engine_steps, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        return {"error": str(exc)}
    field = next(
        (f for f in ("prepare_input_us", "prepare_input", "duration_us") if rows and f in rows[0]),
        None,
    )
    if not rows or field is None:
        return {"error": f"no prepare-input column in {engine_steps}"}
    theirs = [float(r[field]) for r in rows if r.get(field)]
    ours = list(step_wall.values())
    return {
        "source": str(engine_steps),
        "column": field,
        "n_liteprofiler": len(theirs),
        "n_subscope": len(ours),
        "liteprofiler_p50_us": round(quantile(theirs, 0.5), 3),
        "subscope_p50_us": round(quantile(ours, 0.5), 3),
        "delta_pct": round(
            100.0 * (quantile(ours, 0.5) - quantile(theirs, 0.5)) / quantile(theirs, 0.5),
            2,
        )
        if theirs and quantile(theirs, 0.5)
        else None,
    }


if __name__ == "__main__":
    sys.exit(main(sys.argv))
