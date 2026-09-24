#!/usr/bin/env python3
"""出图：realmachine 2 张（batch/isl）+ stress 6 张 + 1 张 2×3 汇总。

严禁中文标签：开发机 / a3-22 的 matplotlib 都没有 CJK 字体（会渲染成方框）。

用法：python3 agents/sweep_analysis/plot_sweep.py [--data data/harness] [--figdir figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

C = {
    "pi": "#0072B2",      # blue    pi_net（主口径）
    "raw": "#999999",     # grey    pi_raw
    "triton": "#CC79A7",  # pink    triton_cpu
    "us": "#E69F00",      # orange  _update_states
    "scope": "#4D4D4D",   # dark grey scope_net
    "meta": "#7B3294",    # purple  结构元数据
    "fit": "#009E73",     # green   拟合
}


def finish(ax, xlabel, ylabel, title, logx=False):
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    if logx:
        ax.set_xscale("log", base=2)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def box(ax, text, loc="upper left", size=7.5):
    xy = {"upper left": (0.03, 0.97), "upper right": (0.97, 0.97),
          "lower left": (0.03, 0.05), "lower right": (0.97, 0.05)}[loc]
    ax.text(*xy, text, transform=ax.transAxes,
            va="top" if "upper" in loc else "bottom",
            ha="left" if "left" in loc else "right", fontsize=size,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#4D4D4D", alpha=0.9))


def series(points, key):
    pts = [(p["x"], p.get(key)) for p in points
           if p.get(key) is not None and p.get(key) == p.get(key)]
    return [p[0] for p in pts], [p[1] for p in pts]


def fitline(ax, points, key, color=C["fit"]):
    xs, ys = series(points, key)
    if len(xs) < 2:
        return None
    a, c = np.polyfit(xs, ys, 1)
    x0, x1 = min(xs), max(xs)
    ax.plot([x0, x1], [c + a * x0, c + a * x1], "--", color=color, lw=1.4,
            label="linear fit (pi_net)")
    return {"a": float(a), "c": float(c)}


def fittxt(f) -> str:
    if not f or f.get("n_points", 0) < 2 or f.get("r2") != f.get("r2"):
        return "n<2 (no fit)"
    return (f"pi_net: T = {f['c']:.1f} + {f['a']:.4f}*x us\n"
            f"R2={f['r2']:.3f}  n={int(f['n_points'])}")


def draw(ax, entry, xlabel, title, logx, mode="std", extra=""):
    pts = entry["points"]
    xn, yn = series(pts, "pi_net_p50")
    xu, yu = series(pts, "us_p50")
    xt, yt = series(pts, "triton_p50")
    xr, yr = series(pts, "pi_raw_p50")
    xs_, ys_ = series(pts, "scope_net_p50")
    if mode == "scope":
        ax.plot(xs_, ys_, "-D", color=C["scope"], ms=4, lw=1.5, label="scope_net (= us + pi_net)")
        ax.plot(xn, yn, "-o", color=C["pi"], ms=5, lw=1.8, label="pi_net (_prepare_inputs)")
        ax.plot(xu, yu, "-s", color=C["us"], ms=4.5, lw=1.8, label="_update_states")
    else:
        ax.plot(xr, yr, "-o", color=C["raw"], ms=3.5, lw=1.2, label="pi_raw (unsub)")
        ax.plot(xn, yn, "-o", color=C["pi"], ms=5, lw=1.8, label="pi_net = raw - triton")
        ax.plot(xu, yu, "-s", color=C["us"], ms=4, lw=1.6, label="_update_states")
        if max(yt) > 2:
            ax.plot(xt, yt, "-^", color=C["triton"], ms=3.5, lw=1.1, label="triton_cpu (harness-only)")
    fitline(ax, pts, "pi_net_p50")
    txt = fittxt(entry.get("pi_net"))
    u = entry.get("us") or {}
    if u.get("n_points", 0) >= 2 and u.get("r2") == u.get("r2"):
        txt += f"\n_update_states: c={u['c']:.1f} a={u['a']:.4f} R2={u['r2']:.3f}"
    if extra:
        txt += "\n" + extra
    box(ax, txt, "upper left" if mode == "scope" else "upper left")
    finish(ax, xlabel, "us per step (p50)", title, logx)
    # 相对变化很小时，数据线会横穿顶部：把注释框和图例一起压到底部，避免压线
    spread = (max(yn) - min(yn)) / (np.mean(yn) or 1.0)
    if mode != "scope" and spread < 0.12:
        ax.get_children()
        for t in ax.texts:
            t.remove()
        box(ax, txt, "lower left")
        ax.legend(fontsize=7, loc="upper right")
    else:
        ax.legend(fontsize=7, loc="lower right")


REALMACHINE = (
    ("06-sweep-realmachine-batch", "rm_batch_mmr64", "concurrent requests B",
     "REALMACHINE preset (matches real launcher): pi_net vs batch "
     "(MML=2048, max_num_reqs=64, isl=1024)", True, "scope",
     "caliber: pi_net_p50_us (triton_cpu subtracted)\nreal-machine caliber - only for comparing with real HW"),
    ("06-sweep-rm-batch", "rm_batch_mmr64", "concurrent requests B",
     "REALMACHINE: pi_net vs batch (MML=2048, max_num_reqs=64, isl=1024)", True, "scope",
     "real-machine preset: _update_states slope > pi_net slope"),
    ("06-sweep-rm-isl", "rm_isl", "ISL (tokens)",
     "REALMACHINE: pi_net vs ISL (batch=1, chunk=2048, warmup=3 -> decode window)", True, "std",
     "window is decode-only: ISL only widens token_ids_cpu"),
)
STRESS = (
    ("06-sweep-isl", "stress_isl", "ISL (tokens)",
     "STRESS: pi_net vs ISL (batch=32, chunk=16384)", True, "chunked prefill flattens ISL"),
    ("06-sweep-batch", "stress_batch", "concurrent requests B",
     "STRESS: pi_net vs batch (isl=2048)", True, ""),
    ("06-sweep-chunk", "stress_chunk_size", "chunk_size (tokens/step)",
     "STRESS: pi_net vs chunk_size", True, "small chunk -> fewer tokens/step"),
    ("06-sweep-blocksize", "stress_block_size", "block_size (tokens/block)",
     "STRESS: pi_net vs block_size (MML=32768)", True,
     "block_size down -> max_num_blocks_per_req up\n-> commit_block_table bytes up"),
    ("06-sweep-spec", "stress_spec_k", "spec_k (MTP draft tokens)",
     "STRESS: pi_net vs spec_k", False, "MTP: tokens/step = 1+k"),
    ("06-sweep-prefix", "stress_prefix_hit_ratio", "prefix_hit_ratio",
     "STRESS: pi_net vs prefix hit ratio", False, "prefix hits skip prefill"),
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/harness")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args(argv)
    doc = json.loads((Path(args.data) / "sweep_fits.json").read_text())
    fits = doc["fits"]
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    def render(fname, entry, xlabel, title, logx, mode, extra):
        if entry is None or len(entry["points"]) < 2:
            print(f"[plot] SKIP {fname}: no data")
            return
        fig, ax = plt.subplots(figsize=(7.8, 5.1))
        draw(ax, entry, xlabel, title, logx, mode, extra)
        fig.tight_layout()
        base = figdir / fname
        fig.savefig(base.with_suffix(".svg"))
        fig.savefig(base.with_suffix(".png"), dpi=150)
        plt.close(fig)
        made.append(str(base.with_suffix(".svg")))

    for fname, key, xl, title, logx, mode, extra in REALMACHINE:
        render(fname, fits["realmachine"].get(key), xl, title, logx, mode, extra)
    for fname, key, xl, title, logx, extra in STRESS:
        render(fname, fits["stress"].get(key), xl, title, logx, "std", extra)

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.8))
    for ax, (fname, key, xl, title, logx, extra) in zip(axes.ravel(), STRESS):
        e = fits["stress"].get(key)
        if e is None or len(e["points"]) < 2:
            ax.text(0.5, 0.5, f"no data: {key}", ha="center", va="center", transform=ax.transAxes)
            finish(ax, xl, "us per step", title, logx)
            continue
        draw(ax, e, xl, title, logx, "std", extra)
    fig.suptitle("STRESS preset sweep summary -- pi_net = prepare_inputs_us - triton_cpu_us (p50 per step, "
                 "MML=32768 / 64 reqs / 16384 tokens / slot_mapping=noop, qwen35-0.8b, a3-22 no-card harness)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = figdir / "06-sweep-summary"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=150)
    plt.close(fig)
    made.append(str(base.with_suffix(".svg")))

    print("[plot] wrote:")
    for p in made:
        print("   " + p)

    # ---- 06-slot-mapping-modes：交错对照 + 跨窗口对照（methodology 图）
    modes = doc.get("mode_interleaved_realmachine", {})
    keys = [k for k in ("noop", "cpu_fallback", "inject:15") if isinstance(modes.get(k), dict)]
    if keys:
        net = [modes[k]["pi_net_p50_median_us"] for k in keys]
        tri = [modes[k]["triton_p50_median_us"] for k in keys]
        raw = [modes[k]["pi_raw_p50_median_us"] for k in keys]
        xs = np.arange(len(keys))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                                       gridspec_kw={"width_ratios": [1.25, 1]})
        ax1.bar(xs, net, 0.55, color=C["pi"], label="pi_net = raw − triton (net CPU work)")
        ax1.bar(xs, tri, 0.55, bottom=net, color=C["triton"],
                label="triton_cpu (harness slot-mapping fallback + injected launch)")
        for i, (n, t) in enumerate(zip(net, tri)):
            ax1.text(i, n / 2, f"{n:.1f}", ha="center", va="center", fontsize=8, color="white")
            ax1.text(i, n + t / 2, f"{t:.1f}", ha="center", va="center", fontsize=8, color="white")
            ax1.text(i, n + t + 8, f"raw {raw[i]:.1f}", ha="center", fontsize=8)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([f"{k}\n(triton launches/step = 1)" for k in keys], fontsize=8.5)
        ax1.set_ylabel("us per step (p50)")
        ax1.set_title("REALMACHINE preset, interleaved rounds (same window) — "
                      "batch=1, isl=128, osl=64, steps=200", fontsize=10)
        ax1.legend(fontsize=7.5, loc="upper left")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.spines["top"].set_visible(False)

        paired = modes.get("paired_within_round", {})
        labels = []
        vals = []
        errs = []
        for k, base in (("cpu_fallback_minus_noop", "cpu vs noop"),
                        ("inject:15_minus_noop", "inject15 vs noop"),
                        ("inject:15_minus_cpu_fallback", "inject15 vs cpu")):
            p = paired.get(k)
            if p:
                labels.append(f"{base}\n(net, paired)")
                vals.append(p["median_us"])
                errs.append([[p["median_us"] - p["min_us"]], [p["max_us"] - p["median_us"]]])
        if vals:
            ax2.bar(np.arange(len(vals)), vals, 0.5, color=C["fit"], yerr=np.array(errs).reshape(2, -1),
                    capsize=5, error_kw=dict(lw=1.2, ecolor="#333333"))
            for i, v in enumerate(vals):
                ax2.text(i, v + 0.6, f"{v:+.1f} us", ha="center", fontsize=8.5)
            ax2.axhline(0, color="#333333", lw=0.9)
            ax2.set_xticks(np.arange(len(vals)))
            ax2.set_xticklabels(labels, fontsize=8.5)
            ax2.set_ylabel("delta pi_net (us per step)")
            ax2.set_title("Paired within-round deltas — the only trustworthy\n"
                          "mode-to-mode comparison (bars = per-round min/max)", fontsize=10)
            ax2.grid(True, axis="y", alpha=0.25)
            ax2.spines["top"].set_visible(False)
            ax2.text(0.03, 0.97, "inject15 − cpu = +0.8 us →\nthe 15 us/launch spin is 100%\n"
                                 "accounted in triton_cpu, zero bleed",
                     transform=ax2.transAxes, va="top", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#4D4D4D", alpha=0.9))
        fig.suptitle("Slot-mapping mode comparison (preset=realmachine, caliber=pi_net_p50_us; "
                     "every step = exactly 1 triton launch)", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        base = figdir / "06-slot-mapping-modes"
        fig.savefig(base.with_suffix(".svg"))
        fig.savefig(base.with_suffix(".png"), dpi=150)
        plt.close(fig)
        print("   " + str(base.with_suffix(".svg")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
