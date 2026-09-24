#!/usr/bin/env python3
"""Draw the chapter-06 (synthetic-load harness) figures from the harness CSVs.

Inputs (all produced by `harness/pi_harness/runner/{cli,_replay_main}.py`):

* ``--substep-dir``  a directory holding ``substep_*.csv`` + ``summary_*.json``
                     from one long ``--preset realmachine`` / ``stress`` run
                     (per-step DELTAS, unit = seconds)
* ``--mode-csv``     long-form CSV comparing slot-mapping modes
                     (columns: mode, prepare_inputs_p50_us, triton_cpu_us, ...)
* ``--outdir``       where the SVGs go

All labels are English on purpose: the target container's matplotlib has no CJK
font and a missing glyph renders as a box.

Usage::

    python3 scripts/plot_harness_figures.py \\
        --substep-dir data/harness/substep/tmp_bs16b \\
        --realmachine-summary data/harness/substep/tmp_realmachine \\
        --outdir figures --prefix 06
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})


def read_substep(directory: str | pathlib.Path) -> tuple[dict[str, float], int]:
    """Sum per-step deltas from `substep_*.csv` -> {function: total_seconds}."""
    files = sorted(glob.glob(str(pathlib.Path(directory) / "substep_*.csv")))
    if not files:
        raise SystemExit(f"no substep_*.csv under {directory}")
    totals: dict[str, float] = {}
    n = 0
    with open(files[-1]) as fh:
        for row in csv.DictReader(fh):
            n += 1
            for key, val in row.items():
                if not val or key == "step":
                    continue
                try:
                    totals[key] = totals.get(key, 0.0) + float(val)
                except ValueError:
                    pass
    return totals, n


def read_summary(directory: str | pathlib.Path) -> dict:
    files = sorted(glob.glob(str(pathlib.Path(directory) / "summary_*.json")))
    if not files:
        raise SystemExit(f"no summary_*.json under {directory}")
    return json.load(open(files[-1]))


def read_sweep(path_glob: str, xcol: str, ycol: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for f in sorted(glob.glob(path_glob)):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("error"):
                    continue
                try:
                    xs.append(float(row[xcol]))
                    ys.append(float(row[ycol]))
                except (KeyError, ValueError):
                    pass
    order = np.argsort(xs)
    return [xs[i] for i in order], [ys[i] for i in order]


def fig_substep_breakdown(totals: dict[str, float], n_steps: int, out: pathlib.Path,
                          title: str) -> dict:
    """Horizontal bar of per-step cost by wrapped function + the un-wrapped residual."""
    pi = totals.get("NPUModelRunner._prepare_inputs", 0.0)
    us = totals.get("NPUModelRunner._update_states", 0.0)
    scope = pi + us
    if scope <= 0:
        raise SystemExit("scope total is zero; wrong CSV?")

    # 只显示"叶子级"函数：把父函数（会被子函数重复计入）排除掉，避免双计。
    parents = {
        "NPUModelRunner._prepare_inputs",
        "NPUModelRunner._update_states",
        "MultiGroupBlockTable.commit_block_table",
        "MultiGroupBlockTable.compute_slot_mapping",
        "BlockTable.compute_slot_mapping",
        "BlockTable.commit_block_table",
        "GPUModelRunner._update_states",
        "InputBatch._make_sampling_metadata",
    }
    leaves = {k: v for k, v in totals.items() if v > 0 and k not in parents}
    leaf_sum = sum(leaves.values())
    residual = max(pi + us - leaf_sum, 0.0)
    leaves["(inline code in _prepare_inputs / _update_states)"] = residual

    items = sorted(leaves.items(), key=lambda kv: kv[1])[-16:]
    labels = [k.replace("NPUModelRunner.", "").replace("GPUModelRunner.", "")
              .replace("MultiGroupBlockTable.", "").replace("BlockTable.", "")
              .replace("InputBatch.", "") for k, _ in items]
    vals = [v / n_steps * 1e6 for _, v in items]

    fig, ax = plt.subplots(figsize=(8.2, 0.36 * len(items) + 1.6))
    colors = ["#c44e4e" if "inline" in lb else "#4e79a7" for lb in labels]
    ax.barh(labels, vals, color=colors)
    for y, (lb, v) in enumerate(zip(labels, vals)):
        share = (items[y][1] / scope) * 100
        ax.text(v + max(vals) * 0.012, y, f"{v:.1f} µs ({share:.1f}%)",
                va="center", fontsize=7.5)
    ax.set_xlabel("per-step CPU time (µs)")
    ax.set_title(f"{title}\nscope = {scope / n_steps * 1e6:.1f} µs/step over {n_steps:,} steps",
                 fontsize=10)
    ax.set_xlim(0, max(vals) * 1.28)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    return {
        "n_steps": n_steps,
        "scope_us_per_step": scope / n_steps * 1e6,
        "prepare_inputs_us_per_step": pi / n_steps * 1e6,
        "update_states_us_per_step": us / n_steps * 1e6,
        "leaf_sum_us_per_step": leaf_sum / n_steps * 1e6,
        "inline_residual_us_per_step": residual / n_steps * 1e6,
        "inline_residual_share_pct": residual / scope * 100,
        "rows": [
            {"function": k, "us_per_step": v / n_steps * 1e6,
             "share_of_scope_pct": v / scope * 100}
            for k, v in sorted(leaves.items(), key=lambda kv: -kv[1])
        ],
    }


def fig_mode_comparison(rows: list[dict], out: pathlib.Path) -> None:
    """Grouped bars: total vs the harness-only numpy fallback, per slot-mapping mode."""
    modes = [r["mode"] for r in rows]
    total = [r["prepare_inputs_p50_us"] for r in rows]
    fallback = [r.get("triton_cpu_us", 0.0) or 0.0 for r in rows]
    net = [t - f for t, f in zip(total, fallback)]

    x = np.arange(len(modes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w / 2, net, w, label="vLLM code only (net)", color="#4e79a7")
    ax.bar(x - w / 2, fallback, w, bottom=net,
           label="harness numpy fallback (must be subtracted)", color="#e1a44e")
    ax.bar(x + w / 2, total, w, label="measured total", color="#b8c9e0")
    for xi, (t, n) in enumerate(zip(total, net)):
        ax.text(xi + w / 2, t + max(total) * 0.015, f"{t:.0f}", ha="center", fontsize=8)
        ax.text(xi - w / 2, n / 2, f"{n:.0f}", ha="center", va="center", fontsize=8,
                color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("prepare_inputs p50 (µs)")
    ax.set_title("slot-mapping mode: what the harness adds on top of vLLM")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_sweep_panels(panels: list[tuple[str, str, list[float], list[float], str]],
                     out: pathlib.Path) -> None:
    """2xN grid of sweep curves (log-x where the axis is a token/size count)."""
    n = len(panels)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.3 * rows), squeeze=False)
    for i, (title, xlabel, xs, ys, note) in enumerate(panels):
        ax = axes[i // cols][i % cols]
        ax.plot(xs, ys, "o-", color="#4e79a7", lw=1.4, ms=4)
        spread = (max(ys) / min(ys)) if min(ys) > 0 else float("nan")
        ax.set_title(f"{title}  (spread {spread:.2f}×)", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("prepare_inputs p50 (µs)", fontsize=8)
        if note:
            ax.text(0.03, 0.95, note, transform=ax.transAxes, fontsize=7,
                    va="top", color="#555")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--substep-dir", default="data/harness/substep/tmp_bs16b",
                    help="stress-preset long run (substep_*.csv + summary_*.json)")
    ap.add_argument("--realmachine-summary", default="data/harness/substep/tmp_realmachine",
                    help="realmachine-preset long run")
    ap.add_argument("--sweep-glob", default="data/harness/hw64_sweep_*.csv",
                    help="sweep CSVs for the panel figure")
    ap.add_argument("--mode-csv", default=None,
                    help="optional CSV with columns mode,prepare_inputs_p50_us,triton_cpu_us")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--prefix", default="06")
    ap.add_argument("--data-outdir", default=None)
    a = ap.parse_args()

    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_out = pathlib.Path(a.data_outdir) if a.data_outdir else None
    if data_out:
        data_out.mkdir(parents=True, exist_ok=True)

    produced = []

    # --- 1. substep breakdown (realmachine is the headline) -------------------
    if pathlib.Path(a.realmachine_summary).exists():
        totals, n = read_substep(a.realmachine_summary)
        summ = read_summary(a.realmachine_summary)
        stats = fig_substep_breakdown(
            totals, n, outdir / f"{a.prefix}-substep-realmachine.svg",
            "realmachine preset (matches the real launcher: max-model-len 2048, 8 reqs)")
        stats["preset"] = "realmachine"
        stats["summary"] = {k: summ[k] for k in ("steps_run", "prepare_inputs_us",
                                                 "update_states_us") if k in summ}
        produced.append(stats)
        if data_out:
            json.dump(stats, open(data_out / "substep_realmachine.json", "w"),
                      ensure_ascii=False, indent=2)

    # --- 2. substep breakdown (stress, for the slope study) -------------------
    if pathlib.Path(a.substep_dir).exists():
        totals, n = read_substep(a.substep_dir)
        summ = read_summary(a.substep_dir)
        stats = fig_substep_breakdown(
            totals, n, outdir / f"{a.prefix}-substep-stress.svg",
            "stress preset (max-model-len 32768, 64 reqs, 16384 batched tokens)")
        stats["preset"] = "stress"
        stats["summary"] = {k: summ[k] for k in ("steps_run", "prepare_inputs_us",
                                                 "update_states_us") if k in summ}
        produced.append(stats)
        if data_out:
            json.dump(stats, open(data_out / "substep_stress.json", "w"),
                      ensure_ascii=False, indent=2)

    # --- 3. mode comparison ---------------------------------------------------
    if a.mode_csv and pathlib.Path(a.mode_csv).exists():
        with open(a.mode_csv) as fh:
            rows = [r for r in csv.DictReader(fh)]
        for r in rows:
            for k in ("prepare_inputs_p50_us", "triton_cpu_us"):
                r[k] = float(r.get(k) or 0.0)
        fig_mode_comparison(rows, outdir / f"{a.prefix}-slot-mapping-modes.svg")

    # --- 4. sweep panels ------------------------------------------------------
    panels = []
    specs = [
        ("ISL (chunked prefill)", "ISL (tokens)", "isl",
         "flat: chunk_size caps the per-step work"),
        ("Batch", "concurrent requests", "batch", "non-monotonic: host noise"),
        ("chunk_size", "chunk (tokens)", "chunk_size", ""),
        ("block_size", "block size (tokens)", "block_size", ""),
        ("spec_k (MTP)", "draft tokens", "spec_k", ""),
        ("prefix hit ratio", "fraction of requests", "prefix_hit_ratio", ""),
    ]
    for title, xlabel, key, note in specs:
        xs, ys = read_sweep(a.sweep_glob, "value", "pi_p50_us")
        # the glob mixes dimensions; re-read per dimension
        xs, ys = [], []
        for f in sorted(glob.glob(a.sweep_glob)):
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    if row.get("error") or row.get("sweep") != key:
                        continue
                    try:
                        xs.append(float(row["value"]))
                        ys.append(float(row["pi_p50_us"]))
                    except (KeyError, ValueError):
                        pass
        if xs:
            order = np.argsort(xs)
            panels.append((title, xlabel, [xs[i] for i in order],
                           [ys[i] for i in order], note))
    if panels:
        fig_sweep_panels(panels, outdir / f"{a.prefix}-sweep-panels.svg")

    print("wrote:", ", ".join(sorted(p.name for p in outdir.glob(f"{a.prefix}-*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
