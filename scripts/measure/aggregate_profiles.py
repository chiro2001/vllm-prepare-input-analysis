#!/usr/bin/env python3
"""Aggregate the per-point profiler outputs of a campaign into campaign-level files.

Input layout (produced on a3-22 by ``point_run.py``):

    data/measure/<run>/points/<tag>/perf/{hotspots.csv,flame.svg,manifest.json,...}
    data/measure/<run>/points/<tag>/pmu/{topdown.json,libkperfx.json,manifest.json,...}

Output (this script):

    data/profiles/<run>/hotspots.csv   - every point's hotspot rows, tagged by point
    data/profiles/<run>/topdown.csv    - one row per point x topdown metric
    data/profiles/<run>/ipc.csv        - one row per point: cycles/instructions/IPC/confidence
    data/profiles/<run>/profile_index.json
    data/profiles/<run>/README.md

Pure stdlib (runs on a3-22 too).

Usage:
    python3 scripts/measure/aggregate_profiles.py --run-dir data/measure/<run> \
        [--out-dir data/profiles/<run>]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics


def read_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv(path: pathlib.Path) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def point_records(run_dir: pathlib.Path) -> list[dict]:
    pts = run_dir / "points.json"
    if pts.is_file():
        try:
            data = json.loads(pts.read_text(encoding="utf-8"))
            return [r.get("point", {}) for r in data if r.get("point")]
        except json.JSONDecodeError:
            pass
    out = []
    for pj in sorted(run_dir.glob("points/*/point.json")):
        doc = read_json(pj)
        if doc:
            out.append(doc)
    return out


FLAT_METRICS = (
    "retiring", "bad_spec", "frontend_bound", "backend_bound",
    "frontend_latency_bound", "frontend_bandwidth_bound",
    "resource_bound", "core_bound", "mem_bound",
    "mem_l1_bound", "mem_l2_bound", "mem_l3_dram_bound", "mem_mem_bound",
    "mem_store_bound",
    "rob_stall_pct", "ptag_stall_pct", "mapq_stall_pct", "pcbuf_stall_pct",
    "other_stall_pct",
)


def flatten_topdown(doc: dict) -> dict:
    """topdown.json is nested (level1 / frontend / backend / mem / ooo)."""
    flat: dict = {}

    def walk(prefix: str, node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}{key}", value)
        elif isinstance(node, (int, float)):
            flat[prefix.rstrip(".")] = node
        else:
            flat[prefix.rstrip(".")] = node

    for section in ("level1", "frontend", "backend", "mem", "ooo", "metrics", "topdown"):
        if section in doc:
            walk("", doc[section])
    for key in FLAT_METRICS:
        if key in doc and key not in flat:
            flat[key] = doc[key]
    return flat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    run_id = run_dir.name
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        run_dir.parent.parent / "profiles" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = point_records(run_dir)
    index: list[dict] = []
    topdown_rows: list[dict] = []
    ipc_rows: list[dict] = []
    hotspot_rows: list[dict] = []

    for rec in records:
        tag = rec.get("tag")
        if not tag:
            continue
        pdir = run_dir / "points" / tag
        entry: dict = {
            "tag": tag,
            "group": rec.get("group"),
            "workload": rec.get("workload"),
            "perf": None,
            "pmu": None,
        }

        perf_dir = pdir / "perf"
        perf_manifest = read_json(perf_dir / "manifest.json")
        if perf_manifest:
            entry["perf"] = {
                "flame_svg": str((perf_dir / "flame.svg").relative_to(run_dir))
                if (perf_dir / "flame.svg").is_file() else None,
                "samples": perf_manifest.get("sample_count"),
                "lost": perf_manifest.get("lost_samples"),
                "freq": perf_manifest.get("freq"),
                "call_graph": perf_manifest.get("call_graph"),
                "symbol_quality": perf_manifest.get("symbol_quality"),
                "annotate": perf_manifest.get("annotate"),
            }
            for row in read_csv(perf_dir / "hotspots.csv"):
                hotspot_rows.append({
                    "point": tag, "group": rec.get("group"),
                    "rank": row.get("rank"), "dso": row.get("dso"),
                    "symbol": row.get("symbol"),
                    "self_pct": row.get("self_pct"), "samples": row.get("samples"),
                })

        pmu_dir = pdir / "pmu"
        pmu_manifest = read_json(pmu_dir / "manifest.json")
        if pmu_manifest:
            td = read_json(pmu_dir / "topdown.json")
            lib = read_json(pmu_dir / "libkperfx.json")
            entry["pmu"] = {
                "manifest": {k: pmu_manifest.get(k) for k in
                             ("groups", "window", "all_confident",
                              "low_confidence_groups", "zero_counting_events",
                              "exit_code")},
                "confidence_min": pmu_manifest.get("min_confidence"),
                "zero_counting_events": pmu_manifest.get("zero_counting_events"),
            }
            flat = flatten_topdown(td)
            row = {"point": tag, "group": rec.get("group"),
                   "confidence_min": pmu_manifest.get("min_confidence")}
            for key in FLAT_METRICS:
                if key in flat:
                    row[key] = flat[key]
            topdown_rows.append(row)

            counters = lib.get("counters") or {}
            derived = lib.get("derived") or {}
            cycles = counters.get("CPU_CYCLES") or counters.get("0x11")
            inst = counters.get("INST_RETIRED") or counters.get("0x8")
            ipc = derived.get("ipc")
            if ipc is None and cycles and inst:
                try:
                    ipc = float(inst) / float(cycles)
                except (TypeError, ZeroDivisionError):
                    ipc = None
            ipc_rows.append({
                "point": tag, "group": rec.get("group"),
                "cycles": cycles, "instructions": inst, "ipc": ipc,
                "confidence_min": pmu_manifest.get("min_confidence"),
                "time_enabled_ns": (lib.get("meta") or {}).get("time_enabled_ns"),
                "time_running_ns": (lib.get("meta") or {}).get("time_running_ns"),
                "useronly_forced": (lib.get("meta") or {}).get("useronly_forced"),
            })

        # phase summary (both denominators)
        phases = (rec.get("phases") or {})
        summ = phases.get("summary") or {}
        loop = phases.get("loop_summary") or {}
        entry["phase_share"] = {
            "step_model": (summ.get("share") or {}),
            "whole_loop": (loop.get("share") or {}),
        }
        index.append(entry)

    write_csv(out_dir / "hotspots.csv", hotspot_rows,
              ["point", "group", "rank", "dso", "symbol", "self_pct", "samples"])
    write_csv(out_dir / "topdown.csv", topdown_rows,
              ["point", "group", "confidence_min", *FLAT_METRICS])
    write_csv(out_dir / "ipc.csv", ipc_rows,
              ["point", "group", "cycles", "instructions", "ipc", "confidence_min",
               "time_enabled_ns", "time_running_ns", "useronly_forced"])
    (out_dir / "profile_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        render_readme(run_id, index, topdown_rows, ipc_rows, hotspot_rows),
        encoding="utf-8")
    print(json.dumps({
        "run_id": run_id, "points": len(records),
        "perf_points": sum(1 for e in index if e["perf"]),
        "pmu_points": sum(1 for e in index if e["pmu"]),
        "hotspot_rows": len(hotspot_rows),
        "out_dir": str(out_dir),
    }, indent=2))
    return 0


def write_csv(path: pathlib.Path, rows: list[dict], header: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_readme(run_id, index, topdown_rows, ipc_rows, hotspot_rows) -> str:
    lines = [f"# profiles/{run_id}", "",
             f"- points with perf capture: {sum(1 for e in index if e['perf'])}",
             f"- points with PMU sweep:    {sum(1 for e in index if e['pmu'])}",
             f"- hotspot rows:             {len(hotspot_rows)}",
             "", "## topdown", ""]
    if topdown_rows:
        lines.append("| point | " + " | ".join(
            ["retiring", "bad_spec", "frontend", "backend", "conf"]) + " |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in topdown_rows:
            lines.append("| {point} | {retiring} | {bad_spec} | {frontend_bound} | "
                         "{backend_bound} | {confidence_min} |".format(
                             **{k: fmt(row.get(k)) for k in
                                ("point", "retiring", "bad_spec", "frontend_bound",
                                 "backend_bound", "confidence_min")}))
    else:
        lines.append("_(no PMU data yet)_")
    lines += ["", "## IPC", ""]
    if ipc_rows:
        ipcs = [float(r["ipc"]) for r in ipc_rows if r.get("ipc") is not None]
        lines.append(f"- IPC range: {min(ipcs):.3f} .. {max(ipcs):.3f} "
                     f"(median {statistics.median(ipcs):.3f}, n={len(ipcs)})"
                     if ipcs else "- no IPC values")
    else:
        lines.append("_(no PMU data yet)_")
    lines += ["", "## files", "",
              "| file | content |", "|---|---|",
              "| `hotspots.csv` | perf self-overhead hotspots, tagged by point |",
              "| `topdown.csv` | 920B topdown buckets + sub-metrics per point |",
              "| `ipc.csv` | cycles/instructions/IPC + confidence per point |",
              "| `profile_index.json` | per-point profiler artifacts + phase shares |",
              "", "Raw `perf.data` stays on a3-22 only (see plan/COORDINATION.md §5)."]
    return "\n".join(lines) + "\n"


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
