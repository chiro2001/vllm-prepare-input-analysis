#!/usr/bin/env python3
"""Turn pi-point-v1 records into the chapter-03 figures and a flat matrix table.

Input is the array of ``point.json`` records that ``matrix_run.sh`` aggregates
verbatim (one record per measured point).  Output is SVG + CSV + README.md, so
the figures can be regenerated from the archived data at any time.

Metric convention
-----------------
Per point the analyzer summary (``pi-lite-summary-v1``) is read: the *decode*
phase block is preferred because that is the steady-state engine behaviour; a
point with fewer than 5 decode steps (prefill-only points such as the ISL scan)
falls back to the ``all`` block.  The block actually used is stated per point in
the README table of the data directory.

All labels are English on purpose: the a3-22 / dev-container matplotlib has no
CJK font, and a missing glyph renders as a box.

Usage::

    plot_measure.py --points-json POINTS.json --outdir figures \\
        --data-outdir data/measure/<run> [--prefix 03] [--title-suffix ""]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

# Accessible palette: no red/green pair anywhere; the two-state bars also differ
# by hatch so the figure survives a grayscale print.
COLORS = {
    "prepare": "#0072B2",  # blue
    "device": "#E69F00",  # orange
    "step": "#4D4D4D",  # grey
    "share": "#CC79A7",  # pink
    "off": "#0072B2",
    "on": "#E69F00",
}
HATCH = {"off": "", "on": "//"}

FIGURES = (
    "prepare-vs-concurrency",
    "prepare-share-vs-concurrency",
    "prepare-vs-device-cross",
    "isl-scan",
    "chunked-prefill",
    "mtp-onoff",
)

MATRIX_COLUMNS = (
    "tag", "group", "concurrency", "isl", "osl", "chunk", "mtp",
    "prepare_p50_us", "prepare_p90_us", "prepare_share_p50",
    "forward_p50_us", "step_p50_us", "ttft_ms", "itl_ms", "tps",
    "on_cpu_ratio",
)

PREFERRED_PHASES = ("decode", "all", "prefill+decode", "prefill")
MIN_DECODE_STEPS = 5


def dig(obj, *keys, default=None):
    """Nested ``dict.get`` that never raises on a missing/ill-typed branch."""
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def fnum(value):
    """Return ``float(value)`` or ``None`` (never a string, never NaN/inf)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def pick_block(summary):
    """(phase_name, block) of the analyzer summary, decode preferred."""
    by_phase = (summary or {}).get("by_phase") or {}
    if not isinstance(by_phase, dict):
        return None, None
    decode = by_phase.get("decode")
    decode_steps = dig(decode, "prepare input", "n") or 0
    if decode_steps >= MIN_DECODE_STEPS:
        return "decode", decode
    for phase in PREFERRED_PHASES:
        block = by_phase.get(phase)
        if dig(block, "prepare input", "n"):
            return phase, block
    return None, None


def mtp_state(point):
    """'on' / 'off' / None for the speculative-decoding axis of a point."""
    server = dig(point, "service", "server", default={}) or {}
    for key in ("mtp", "enable_mtp", "speculative", "spec_decode",
                "enable_spec_decode"):
        value = server.get(key)
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, str) and value.strip().lower() in ("on", "off", "true",
                                                               "false"):
            return "on" if value.strip().lower() in ("on", "true") else "off"
    count = fnum(server.get("num_speculative_tokens"))
    if count is not None:
        return "on" if count > 0 else "off"
    config = server.get("speculative_config")
    if isinstance(config, dict):
        nested = fnum(config.get("num_speculative_tokens"))
        if nested is not None:
            return "on" if nested > 0 else "off"
        return "on" if config else "off"
    text = " ".join(
        str(point.get(key) or "") for key in ("tag", "note", "group")
    ).lower()
    if "mtp-off" in text or "mtp_off" in text or "mtp off" in text:
        return "off"
    if "mtp" in text or "spec" in text:
        return "on"
    return None


def extract(point: dict, index: int, warnings: list[str]):
    """Flatten one point record; returns None when the record is unusable."""
    if not isinstance(point, dict):
        warnings.append(f"record {index}: not an object, skipped")
        return None
    if point.get("schema") not in (None, "pi-point-v1"):
        warnings.append(
            f"record {index}: unexpected schema {point.get('schema')!r}, skipped"
        )
        return None
    tag = point.get("tag") or f"record-{index}"
    phases = point.get("phases") or {}
    summary = phases.get("summary") if isinstance(phases, dict) else None
    if not isinstance(summary, dict):
        warnings.append(f"{tag}: no phases.summary (analyzer not run?), "
                        "performance columns stay empty")
        summary = {}
    phase, block = pick_block(summary)
    if phase is None:
        warnings.append(f"{tag}: analyzer summary has no usable prepare-input "
                        "block; performance columns stay empty")
    rounds = point.get("rounds") or []
    aggregates = [
        entry.get("aggregate") or {}
        for entry in rounds
        if isinstance(entry, dict)
    ]
    ttft = mean([fnum(agg.get("ttft_s_mean")) for agg in aggregates])
    itl = mean([fnum(agg.get("mean_itl_ms")) for agg in aggregates])
    tps = mean([fnum(agg.get("output_tps_aggregate")) for agg in aggregates])
    share = (summary.get("share") or {}).get(phase or "", {}) if summary else {}
    row = {
        "run_id": point.get("run_id"),
        "tag": tag,
        "group": (point.get("group") or "").strip().upper(),
        "note": point.get("note") or "",
        "requests": fnum(dig(point, "workload", "requests")),
        "concurrency": fnum(dig(point, "workload", "concurrency")),
        "isl": fnum(dig(point, "workload", "prompt_tokens_target")),
        "osl": fnum(dig(point, "workload", "max_tokens")),
        "rounds": fnum(dig(point, "workload", "rounds")),
        "chunk": fnum(dig(point, "service", "server", "max_num_batched_tokens")),
        "max_model_len": fnum(dig(point, "service", "server", "max_model_len")),
        "max_num_seqs": fnum(dig(point, "service", "server", "max_num_seqs")),
        "tp": fnum(dig(point, "service", "server", "tensor_parallel_size")),
        "async_scheduling": dig(point, "service", "server", "async_scheduling"),
        "cudagraph_mode": dig(point, "service", "server", "cudagraph_mode"),
        "prefix_caching": dig(point, "service", "server", "prefix_caching"),
        "model": dig(point, "service", "model", "served_name")
        or dig(point, "service", "model", "path"),
        "mtp": mtp_state(point),
        "metrics_phase": phase,
        "prepare_n": dig(block, "prepare input", "n"),
        "prepare_p50_us": fnum(dig(block, "prepare input", "p50_us")),
        "prepare_p90_us": fnum(dig(block, "prepare input", "p90_us")),
        "prepare_mean_us": fnum(dig(block, "prepare input", "mean_us")),
        "prepare_share_p50": fnum(share.get("prepare_over_step_p50")),
        "prepare_share_mean": fnum(share.get("prepare_over_step_mean")),
        "forward_p50_us": fnum(dig(block, "forward", "p50_us")),
        "forward_mean_us": fnum(dig(block, "forward", "mean_us")),
        "step_p50_us": fnum(dig(block, "step_dur", "p50_us")),
        "step_mean_us": fnum(dig(block, "step_dur", "mean_us")),
        "post_process_mean_us": fnum(dig(block, "post process", "mean_us")),
        "sample_token_mean_us": fnum(dig(block, "sample_token", "mean_us")),
        "on_cpu_ratio": fnum(dig(point, "cpu", "on_cpu_ratio")),
        "exec_ns_delta": fnum(dig(point, "cpu", "exec_ns_delta")),
        "ttft_ms": None if ttft is None else ttft * 1000.0,
        "itl_ms": itl,
        "tps": tps,
        "errors": len(point.get("errors") or []),
    }
    return row


def fmt(value, spec=".3f", dash="-"):
    if value is None:
        return dash
    if spec == ".0f":
        return f"{value:.0f}"
    return format(value, spec)


def finish_axes(ax, xlabel, ylabel, title=None, grid_axis="both"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", axis=grid_axis, alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", axis="x", alpha=0.15, linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def series(rows, xkey, ykey, predicate=None):
    """Sorted (x, y) pairs with both fields present."""
    points = [
        (row[xkey], row[ykey])
        for row in rows
        if row.get(xkey) is not None
        and row.get(ykey) is not None
        and (predicate is None or predicate(row))
    ]
    points.sort(key=lambda pair: pair[0])
    return points


def crossover(points_a, points_b):
    """First x where curve a crosses curve b, linearly interpolated.

    ``points_a``/``points_b`` are sorted (x, y) pairs over the same x grid.
    """
    lookup_b = {x: y for x, y in points_b}
    previous = None
    for x, y in points_a:
        other = lookup_b.get(x)
        if other is None:
            continue
        gap = y - other
        if previous is not None and (gap == 0 or (previous[1] < 0 < gap) or
                                     (previous[1] > 0 > gap)):
            x0, gap0 = previous[0], previous[1]
            if gap == gap0:
                crossing = x0
            else:
                ratio = gap0 / (gap0 - gap)
                crossing = x0 + ratio * (x - x0)
            return crossing, y + (other - y) * 0.5
        if gap == 0:
            return x, y
        previous = (x, gap)
    return None


def log2_axis(ax, values):
    """Log2-style x axis when the values are powers of two (sweeps usually are)."""
    if not values:
        return False
    if all(value > 0 for value in values) and max(values) / min(values) >= 8:
        ax.set_xscale("log", base=2)
        return True
    return False


def plot_prepare_vs_concurrency(rows, ax, title_suffix):
    """prepare p50 vs concurrency, with step_dur on a twin axis."""
    points = series(rows, "concurrency", "prepare_p50_us")
    if len(points) < 2:
        return False
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    steps = series(rows, "concurrency", "step_p50_us")
    ax.plot(xs, ys, "-o", color=COLORS["prepare"], linewidth=1.8, markersize=5,
            label=f"prepare input p50 ({len(points)} pts)")
    log2_axis(ax, xs)
    finish_axes(ax, "concurrency (requests in flight, decode)", "prepare input (us/step)",
                "prepare input per engine step vs concurrency" + title_suffix)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    if len(steps) >= 2:
        twin = ax.twinx()
        twin.plot([x for x, _ in steps], [y for _, y in steps], "--s",
                  color=COLORS["step"], linewidth=1.4, markersize=4,
                  label=f"step_dur p50 ({len(steps)} pts)")
        twin.set_ylabel("engine step (us)")
        twin.grid(False)
        twin.spines["top"].set_visible(False)
        twin.legend(loc="lower right", fontsize=8, frameon=False)
    return True


def plot_share_vs_concurrency(rows, ax, title_suffix):
    points = series(rows, "concurrency", "prepare_share_p50")
    if len(points) < 2:
        return False
    xs = [x for x, _ in points]
    ax.plot(xs, [y * 100.0 for _, y in points], "-o", color=COLORS["share"],
            linewidth=1.8, markersize=5,
            label=f"prepare / step_dur, p50 ({len(points)} pts)")
    means = series(rows, "concurrency", "prepare_share_mean")
    if len(means) >= 2:
        ax.plot([x for x, _ in means], [y * 100.0 for _, y in means], ":^",
                color=COLORS["prepare"], linewidth=1.4, markersize=4,
                label=f"prepare / step_dur, mean of per-step ratios "
                      f"({len(means)} pts)")
    log2_axis(ax, xs)
    ax.set_ylim(bottom=0)
    finish_axes(ax, "concurrency (requests in flight, decode)",
                "prepare input share of the engine step (%)",
                "prepare input share vs concurrency" + title_suffix)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return True


def plot_prepare_vs_device_cross(rows, ax, title_suffix):
    prepare = series(rows, "concurrency", "prepare_p50_us")
    device = series(rows, "concurrency", "forward_p50_us")
    if len(prepare) < 2 or len(device) < 2:
        return False
    ax.plot([x for x, _ in prepare], [y for _, y in prepare], "-o",
            color=COLORS["prepare"], linewidth=1.8, markersize=5,
            label=f"prepare input p50 ({len(prepare)} pts)")
    ax.plot([x for x, _ in device], [y for _, y in device], "-s",
            color=COLORS["device"], linewidth=1.8, markersize=5,
            label=f"forward = device proxy ({len(device)} pts)")
    xs = [x for x, _ in prepare]
    log2_axis(ax, xs)
    finish_axes(ax, "concurrency (requests in flight, decode)", "wall time (us/step)",
                "prepare input vs device side (forward)" + title_suffix)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    found = crossover(prepare, device)
    if found:
        x_cross, y_cross = found
        ax.axvline(x_cross, color=COLORS["step"], linestyle="--", linewidth=1.0)
        ax.annotate(f"crossover  B~{x_cross:.1f}\nprepare > device from here",
                    xy=(x_cross, y_cross), xytext=(6, 24),
                    textcoords="offset points", fontsize=8, color=COLORS["step"],
                    arrowprops={"arrowstyle": "->", "color": COLORS["step"],
                                "linewidth": 0.8})
    else:
        ax.annotate("no crossover inside the measured range",
                    xy=(0.02, 0.06), xycoords="axes fraction", fontsize=8,
                    color=COLORS["step"])
    return True


def plot_isl_scan(rows, top, bottom, title_suffix, subset=""):
    points = series(rows, "isl", "prepare_p50_us")
    if len(points) < 2:
        return False
    xs = [x for x, _ in points]
    top.plot(xs, [y for _, y in points], "-o", color=COLORS["prepare"],
             linewidth=1.8, markersize=5,
             label=f"prepare input p50 ({len(points)} pts)")
    log2_axis(top, xs)
    finish_axes(top, "", "prepare input (us/step)",
                "prepare input vs input length" + subset + title_suffix)
    top.legend(loc="upper left", fontsize=8, frameon=False)
    per_token = []
    for row in rows:
        isl, prepare = row.get("isl"), row.get("prepare_p50_us")
        if isl is None or prepare is None or isl <= 0:
            continue
        scheduled = min(isl, row.get("chunk") or isl)
        if scheduled <= 0:
            continue
        per_token.append((isl, prepare / scheduled))
    per_token.sort()
    if len(per_token) >= 2:
        bottom.plot([x for x, _ in per_token], [y for _, y in per_token], "-o",
                    color=COLORS["device"], linewidth=1.8, markersize=5,
                    label=f"per scheduled token ({len(per_token)} pts)")
        log2_axis(bottom, [x for x, _ in per_token])
    finish_axes(bottom, "input length ISL (tokens, log2)",
                "prepare input (us/token)",
                "per-token cost: us / min(ISL, chunk)")
    if len(per_token) >= 2:
        bottom.legend(loc="upper right", fontsize=8, frameon=False)
    return True


def plot_chunked_prefill(rows, ax, title_suffix, subset=""):
    points = series(rows, "chunk", "prepare_p50_us")
    if len(points) < 2:
        return False
    xs = [x for x, _ in points]
    ax.plot(xs, [y for _, y in points], "-o", color=COLORS["prepare"],
            linewidth=1.8, markersize=5,
            label=f"prepare input p50 ({len(points)} pts)")
    log2_axis(ax, xs)
    finish_axes(ax, "max_num_batched_tokens = chunk size (tokens)",
                "prepare input (us/step)",
                "prepare input vs chunked-prefill chunk" + subset + title_suffix)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    constant = [y for _, y in points]
    if constant and min(constant) > 0 and max(constant) / min(constant) <= 1.05:
        ax.annotate("flat: prepare is a per-step constant, not per-token",
                    xy=(0.03, 0.08), xycoords="axes fraction", fontsize=8,
                    color=COLORS["step"])
    return True


def plot_mtp_onoff(rows, ax, title_suffix):
    metrics = (("prepare_p50_us", "prepare input"),
               ("forward_p50_us", "forward"),
               ("step_p50_us", "step_dur"))
    states = ("off", "on")
    # Only compare *matched* configurations: the dedicated MTP group (E) when it
    # exists, otherwise the points that differ only in their MTP state.  Without
    # this the "off" bar would average every non-MTP point of the whole matrix.
    candidates = [row for row in rows if row.get("mtp") in states]
    group_e = [row for row in candidates if row.get("group") == "E"]
    if len({row["mtp"] for row in group_e}) == 2:
        selected = group_e
        subset = "group E points"
    else:
        keys = ("concurrency", "isl", "osl", "chunk", "max_num_seqs")
        buckets: dict[tuple, list[dict]] = {}
        for row in candidates:
            buckets.setdefault(tuple(row.get(key) for key in keys), []).append(row)
        matched = [
            row for bucket in buckets.values()
            if len({row["mtp"] for row in bucket}) == 2
            for row in bucket
        ]
        if matched:
            selected = matched
            subset = "configuration-matched pairs"
        else:
            selected = candidates
            subset = "all points carrying an MTP state"
    data = {
        state: [mean([row.get(key) for row in selected if row.get("mtp") == state])
                for key, _ in metrics]
        for state in states
    }
    counts = {state: sum(1 for row in selected if row.get("mtp") == state)
              for state in states}
    present = [state for state in states if counts[state]]
    if len(present) < 2:
        return False
    if not any(value is not None for state in present for value in data[state]):
        return False
    width = 0.38
    positions = list(range(len(metrics)))
    for offset, state in zip((-width / 2, width / 2), present):
        values = [value or 0.0 for value in data[state]]
        bars = ax.bar([p + offset for p in positions], values, width,
                      color=COLORS[state], hatch=HATCH[state], edgecolor="white",
                      linewidth=0.6,
                      label=f"MTP {state} (n={counts[state]})")
        for bar, value in zip(bars, data[state]):
            if value is not None:
                ax.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2,
                                             value), textcoords="offset points",
                            xytext=(0, 2), ha="center", fontsize=7,
                            color=COLORS["step"])
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in metrics])
    finish_axes(ax, "", "wall time (us/step, mean of the points' p50)",
                "MTP off vs on" + title_suffix, grid_axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
              fontsize=8, frameon=False)
    ax.annotate(subset, xy=(0.02, 0.95), xycoords="axes fraction", fontsize=7,
                color=COLORS["step"])
    return True


def load_points(path: pathlib.Path, warnings: list[str]):
    """Read POINTS.json; tolerate a list or a wrapping object."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        warnings.append(f"cannot read {path}: {exc}")
        return []
    except json.JSONDecodeError as exc:
        warnings.append(f"{path} is not valid JSON: {exc}")
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("points", "records", "items"):
            if isinstance(raw.get(key), list):
                warnings.append(f"{path} is an object; using its '{key}' array")
                return raw[key]
        warnings.append(f"{path} is a single object; treating it as one record")
        return [raw]
    warnings.append(f"{path} has unexpected type {type(raw).__name__}")
    return []


def select_for_concurrency(rows):
    """Prefer the dedicated concurrency groups when they exist."""
    grouped = [row for row in rows if row.get("group") in ("A", "G", "I", "J")]
    distinct = {row.get("concurrency") for row in grouped if
                row.get("concurrency") is not None}
    return grouped if len(distinct) >= 2 else rows


def select_for_axis(rows, group: str, key: str):
    """Rows of a dedicated sweep group, else every row with that axis filled.

    Returns ``(rows, label)``; the label is rendered into the figure title so a
    reader can always tell which points a curve is made of.
    """
    grouped = [row for row in rows if row.get("group") == group]
    distinct = {row.get(key) for row in grouped if row.get(key) is not None}
    if len(distinct) >= 2:
        return grouped, f" ({len(grouped)} group-{group} points)"
    return rows, " (all points)"


def write_matrix(path: pathlib.Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MATRIX_COLUMNS)
        for row in sorted(rows, key=lambda item: (item.get("group") or "",
                                                  item.get("concurrency") or 0,
                                                  item.get("isl") or 0,
                                                  item.get("tag") or "")):
            writer.writerow([
                row.get("tag"), row.get("group"), fmt(row.get("concurrency"), ".0f"),
                fmt(row.get("isl"), ".0f"), fmt(row.get("osl"), ".0f"),
                fmt(row.get("chunk"), ".0f"), row.get("mtp") or "",
                fmt(row.get("prepare_p50_us")), fmt(row.get("prepare_p90_us")),
                fmt(None if row.get("prepare_share_p50") is None
                    else row["prepare_share_p50"] * 100.0),
                fmt(row.get("forward_p50_us")), fmt(row.get("step_p50_us")),
                fmt(row.get("ttft_ms")), fmt(row.get("itl_ms")),
                fmt(row.get("tps"), ".2f"), fmt(row.get("on_cpu_ratio"), ".4f"),
            ])


def render_readme(points_json, rows, produced, missing, warnings, args) -> str:
    out: list[str] = []
    add = out.append
    add("# 03 章图与数据表（plot_measure.py 生成）")
    add("")
    add(f"- 输入：`{points_json}`（{len(rows)} 条可用 point 记录）")
    add(f"- 数据目录：`{args.data_outdir}`；图目录：`{args.outdir}`")
    add(f"- 前缀：`{args.prefix}`；标题后缀：`{args.title_suffix!r}`")
    add("")
    add("## 口径")
    add("")
    add("- 每个点的指标取自 `point.json` → `phases.summary`（`pi-lite-summary-v1`）。")
    add(f"  **默认用 `decode` 波段**（engine 稳态）；decode 的 prepare 步数 < "
        f"{MIN_DECODE_STEPS} 的 prefill-only 点退化为 `all`。每个点实际用的波段见下表。")
    add("- `prepare *_us` / `forward *_us` / `step_p50_us` 都来自该波段的 `p50_us`；")
    add("  `step_p50_us` = `by_phase[phase].step_dur.p50_us`（engine 单步 wall）。")
    add("- `prepare_share_p50` = `share[phase].prepare_over_step_p50`，单位 **%**；")
    add("  分子 = 单步 `prepare input` scope，分母 = 同一步的 step scope（async 下是")
    add("  `Step:Model` dispatch，不覆盖整轮 engine loop；要整轮分母请用")
    add("  `analyze_lite.py --step-scope Step:Schedule --window-mode next`）。")
    add("- `ttft_ms` / `itl_ms` / `tps` = 该点所有 round 的 `ttft_s_mean` /")
    add("  `mean_itl_ms` / `output_tps_aggregate` 的算术平均（单位分别 ms / ms / token/s）。")
    add("- `on_cpu_ratio` 来自 `point.json → cpu.on_cpu_ratio`（engine-core 主线程")
    add("  `/proc/<pid>/task/<tid>/schedstat` 的 on-CPU 占比）。")
    add("- 图表标签一律英文：容器/开发机 matplotlib 无中文字体，中文会渲染成方框。")
    add("")
    add("## 生成的点表")
    add("")
    add("| tag | group | B | ISL | OSL | chunk | MTP | 波段 | prepare n | "
        "prepare p50 us | share p50 % | step p50 us |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda item: (item.get("group") or "",
                                              item.get("concurrency") or 0,
                                              item.get("isl") or 0,
                                              item.get("tag") or "")):
        add(
            "| {tag} | {group} | {conc} | {isl} | {osl} | {chunk} | {mtp} | "
            "{phase} | {n} | {prep} | {share} | {step} |".format(
                tag=row.get("tag"), group=row.get("group") or "-",
                conc=fmt(row.get("concurrency"), ".0f"),
                isl=fmt(row.get("isl"), ".0f"), osl=fmt(row.get("osl"), ".0f"),
                chunk=fmt(row.get("chunk"), ".0f"), mtp=row.get("mtp") or "-",
                phase=row.get("metrics_phase") or "-",
                n=fmt(row.get("prepare_n"), ".0f"),
                prep=fmt(row.get("prepare_p50_us")),
                share=fmt(None if row.get("prepare_share_p50") is None
                          else row["prepare_share_p50"] * 100.0),
                step=fmt(row.get("step_p50_us")),
            )
        )
    add("")
    add("## 图")
    add("")
    for name, path in produced:
        add(f"- `{path}`")
    add("")
    add("## 未生成的图 / 缺失数据")
    add("")
    if missing:
        for name, reason in missing:
            add(f"- `{args.prefix}-{name}.svg`：{reason}")
    else:
        add("- 无：6 张图全部生成。")
    add("")
    add("## 解析告警")
    add("")
    if warnings:
        for warning in warnings:
            add(f"- {warning}")
    else:
        add("- 无。")
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--points-json", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("figures"))
    parser.add_argument("--data-outdir", type=pathlib.Path, required=True)
    parser.add_argument("--prefix", default="03")
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args(argv)

    warnings: list[str] = []
    raw_points = load_points(args.points_json, warnings)
    rows: list[dict] = []
    for index, point in enumerate(raw_points):
        row = extract(point, index, warnings)
        if row is not None:
            rows.append(row)
    if not rows:
        print(f"ERROR: no usable point records in {args.points_json}",
              file=sys.stderr)
        for warning in warnings:
            print(f"  ! {warning}", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.data_outdir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.data_outdir / "matrix.csv", rows)

    concurrency_rows = select_for_concurrency(rows)
    if concurrency_rows is not rows:
        warnings.append(
            f"{len(concurrency_rows)} of {len(rows)} points are in the "
            "concurrency groups (A/G/I/J) used for the three concurrency figures"
        )

    produced: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []

    def emit(name: str, draw) -> None:
        figure_path = args.outdir / f"{args.prefix}-{name}.svg"
        ok = draw(figure_path)
        if ok:
            produced.append((name, str(figure_path)))
        else:
            missing.append((name, "数据不足（该轴上少于 2 个有效点）"))

    def single_axes(name: str, plotter):
        def draw(path: pathlib.Path) -> bool:
            figure, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
            try:
                drawn = plotter(rows if name not in (
                    "prepare-vs-concurrency", "prepare-share-vs-concurrency",
                    "prepare-vs-device-cross") else concurrency_rows,
                    ax, args.title_suffix)
                if not drawn:
                    return False
                figure.tight_layout()
                figure.savefig(path, format="svg", bbox_inches="tight")
                return True
            finally:
                plt.close(figure)
        return draw

    isl_rows, isl_label = select_for_axis(rows, "B", "isl")
    chunk_rows, chunk_label = select_for_axis(rows, "C", "chunk")

    def draw_isl(path: pathlib.Path) -> bool:
        figure, (top, bottom) = plt.subplots(
            2, 1, figsize=(6.4, 6.0), dpi=120, sharex=True
        )
        try:
            if not plot_isl_scan(isl_rows, top, bottom, args.title_suffix,
                                 isl_label):
                return False
            figure.tight_layout()
            figure.savefig(path, format="svg", bbox_inches="tight")
            return True
        finally:
            plt.close(figure)

    def draw_chunk(path: pathlib.Path) -> bool:
        figure, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
        try:
            if not plot_chunked_prefill(chunk_rows, ax, args.title_suffix,
                                        chunk_label):
                return False
            figure.tight_layout()
            figure.savefig(path, format="svg", bbox_inches="tight")
            return True
        finally:
            plt.close(figure)

    emit("prepare-vs-concurrency",
         single_axes("prepare-vs-concurrency",
                     plot_prepare_vs_concurrency))
    emit("prepare-share-vs-concurrency",
         single_axes("prepare-share-vs-concurrency",
                     plot_share_vs_concurrency))
    emit("prepare-vs-device-cross",
         single_axes("prepare-vs-device-cross",
                     plot_prepare_vs_device_cross))
    emit("isl-scan", draw_isl)
    emit("chunked-prefill", draw_chunk)
    emit("mtp-onoff", single_axes("mtp-onoff", plot_mtp_onoff))

    warnings.append(
        f"figure subsets: concurrency figures use {len(concurrency_rows)} "
        f"point(s) from groups A/G/I/J; isl-scan uses {isl_label.strip(' ()')}; "
        f"chunked-prefill uses {chunk_label.strip(' ()')}"
    )
    readme = render_readme(args.points_json, rows, produced, missing, warnings, args)
    (args.data_outdir / "README.md").write_text(readme, encoding="utf-8")

    print(f"points-json : {args.points_json}")
    print(f"records     : {len(raw_points)} read, {len(rows)} usable")
    print(f"figures     : {len(produced)} written to {args.outdir}")
    for name, path in produced:
        print(f"  + {path}")
    for name, reason in missing:
        print(f"  - {args.prefix}-{name}.svg skipped: {reason}")
    print(f"matrix.csv  : {args.data_outdir / 'matrix.csv'}")
    print(f"README.md   : {args.data_outdir / 'README.md'}")
    for warning in warnings:
        print(f"  ! {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
