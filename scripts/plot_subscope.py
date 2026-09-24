#!/usr/bin/env python3
"""Stacked-bar figure of the prepare_input sub-scope breakdown.

Reads one or more ``summary_groups.csv`` files (from
``scripts/parse_subscope.py``) and writes an SVG with two panels:

* left  — share of the ``prepare input`` wall time per canonical sub-scope;
* right — the same data in absolute microseconds per step.

Usage::

    plot_subscope.py --out figures/05-subscope-breakdown.svg \
        --series "B=1  decode" data/subscope/run-b1/summary_groups.csv \
        --series "B=64 decode" data/subscope/run-b64/summary_groups.csv

If matplotlib is unavailable the script falls back to writing the figure with
a small built-in SVG writer (bars only, no text metrics needed), so the
pipeline never depends on an optional package.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

UNATTR = "pi: (unattributed)"

COLORS = {
    "synchronize+update_states": "#1f4e79",
    "block_table commit": "#2e75b6",
    "attn_state": "#5b9bd5",
    "positions+token_indices": "#9dc3e6",
    "query_start_loc": "#ffc000",
    "input_ids/H2D": "#ed7d31",
    "bookkeeping other": "#a5a5a5",
    "spec decode metadata": "#70ad47",
    "batch padding decision": "#c00000",
    "scheduler glue": "#7030a0",
    "attn metadata": "#00b0f0",
    UNATTR: "#d9d9d9",
}


def read_groups(path: pathlib.Path, stat: str = "self_us_mean"):
    """Read a summary_groups.csv.

    Default statistic is the **mean**, not p50, because these values are
    stacked: a percentile of a sum is not the sum of percentiles, so stacking
    p50s makes the bars stop short of 100% (observed: 78% and 62%).  Means are
    additive, so the stack reproduces the total.
    """
    share: dict[str, float] = {}
    absolute: dict[str, float] = {}
    wall = float("nan")
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            group = row["group"]
            absolute[group] = float(row[stat])
            share[group] = 0.0
            if group == UNATTR:
                wall = float(row["self_us_p50"]) / max(
                    float(row["share_pct_of_prepare_input_p50"]), 1e-9) * 100.0
    total = sum(absolute.values()) or 1.0
    for group in absolute:
        share[group] = 100.0 * absolute[group] / total
    return share, absolute, wall


def order_of(groups: list[str], series: list[dict]) -> list[str]:
    seen: list[str] = []
    for data in series:
        for group in data["order"]:
            if group not in seen:
                seen.append(group)
    for group in groups:
        if group not in seen:
            seen.append(group)
    return seen


def render_svg(
    labels: list[str],
    series: list[dict],
    groups: list[str],
    *,
    absolute: bool,
    width: int = 640,
    height: int = 460,
) -> str:
    """Minimal stacked-bar SVG writer (no optional dependencies)."""
    left, top, right, bottom = 62, 28, 22, 118
    plot_w = width - left - right
    plot_h = height - top - bottom
    slot = plot_w / max(len(labels), 1)
    bar_w = min(96.0, slot * 0.62)

    if absolute:
        totals = [
            sum(d["data"].get(g, 0.0) for g in groups) for d in series
        ]
    else:
        totals = [100.0 for _ in series]
    ymax = max(totals) * 1.12 if absolute else 100.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="18" font-size="13" font-weight="bold">'
        f'{"Median µs per step (exclusive)" if absolute else "Share of prepare input wall time (%)"}'
        f"</text>",
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#333" stroke-width="1"/>',
    ]
    for i in range(5):
        value = ymax * i / 4
        y = top + plot_h - plot_h * i / 4
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#eeeeee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" font-size="10" '
            f'text-anchor="end" fill="#444">{value:.0f}</text>'
        )

    for si, data in enumerate(series):
        cx = left + slot * (si + 0.5)
        x = cx - bar_w / 2
        acc = 0.0
        for group in groups:
            value = data["data"].get(group, 0.0)
            if value <= 0:
                continue
            h = plot_h * value / ymax
            y = top + plot_h - plot_h * (acc + value) / ymax
            unit = "µs" if absolute else "%"
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" fill="{COLORS.get(group, "#888")}" '
                f'stroke="#ffffff" stroke-width="0.5"><title>{group}: '
                f"{value:.2f}{unit}</title></rect>"
            )
            acc += value
        parts.append(
            f'<text x="{cx:.1f}" y="{top + plot_h + 16}" font-size="11" '
            f'text-anchor="middle">{labels[si]}</text>'
        )

    legend_x = left
    legend_y = top + plot_h + 38
    for i, group in enumerate(groups):
        row, col = divmod(i, 3)
        gx = legend_x + col * (plot_w / 3)
        gy = legend_y + row * 16
        parts.append(
            f'<rect x="{gx:.1f}" y="{gy - 9:.1f}" width="10" height="10" '
            f'fill="{COLORS.get(group, "#888")}"/>'
            f'<text x="{gx + 14:.1f}" y="{gy:.1f}" font-size="10" '
            f'fill="#222">{group}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--series", nargs=2, action="append", metavar=("LABEL", "SUMMARY_GROUPS_CSV"),
                    required=True)
    ap.add_argument("--order", default="", help="comma separated group order override")
    args = ap.parse_args(argv[1:])

    series = []
    ordered: list[str] = []
    for label, path in args.series:
        share, absolute, _wall = read_groups(pathlib.Path(path))
        ordered.extend(g for g in share if g not in ordered)
        series.append({"label": label, "share": share, "absolute": absolute,
                       "order": list(share)})
    if args.order:
        wanted = [g.strip() for g in args.order.split(",") if g.strip()]
        groups = [g for g in wanted if any(g in d["share"] for d in series)]
    else:
        groups = ordered

    labels = [d["label"] for d in series]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, absolute, key in (("", False, "share"), ("-absolute", True, "absolute")):
        data = [
            {"data": d[key], "order": d["order"]} for d in series
        ]
        svg = render_svg(labels, data, groups, absolute=absolute)
        out = args.out if not suffix else args.out.with_name(
            args.out.stem + suffix + args.out.suffix)
        out.write_text(svg)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
