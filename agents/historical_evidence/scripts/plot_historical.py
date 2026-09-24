#!/usr/bin/env python3
"""历史数据的三张图（供 docs/03 与 agents/historical_evidence/REPORT.md 引用）。

输出到 figures/：
  hist-01-phase-decomposition.svg/.png     每步 phase 分解（堆叠柱）
  hist-02-prepare-vs-step-period.svg/.png  prepare input 与 step 周期的关系
  hist-03-topdown-ipc.svg/.png             worker 主线程 top-down 四桶 + IPC
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

HIST = "/home/chiro/projects/vllm/preparing-input-phase/data/historical"
FIG = "/home/chiro/projects/vllm/preparing-input-phase/figures"

PHASES = [
    ("prepare_input_ms", "prepare input", "#e4572e"),
    ("forward_ms", "forward", "#3d7ea6"),
    ("draft_token_ms", "draft_token", "#7fb069"),
    ("sample_token_ms", "sample_token", "#f2c14e"),
    ("Step_Schedule_ms", "Step:Schedule", "#9b8ec4"),
    ("post_process_ms", "post process", "#c9c9c9"),
    ("Step_Output_ms", "Step:Output", "#8d8d8d"),
    ("async_state_update_ms", "async_state_update", "#5c5c5c"),
]


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fig1() -> None:
    rows = [r for r in read(os.path.join(HIST, "phase_per_step_configs.csv")) if r["rank"] == "0"]
    rows.sort(key=lambda r: (fnum(r["step_period_us"])))
    labels = [f"{r['model']}\nTP{r['tp']} {r['mode']} MTP-{r['mtp']}" for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    bottom = [0.0] * len(rows)
    for col, name, color in PHASES:
        vals = [fnum(r.get(col)) / 1000.0 for r in rows]
        ax.bar(range(len(rows)), vals, bottom=bottom, label=name, color=color, width=0.72)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.plot(range(len(rows)), [fnum(r["step_period_us"]) / 1000.0 for r in rows],
            "o-", color="black", ms=4, lw=1.2, label="step 周期（墙钟）")
    ax.set_ylabel("ms / decode step")
    ax.set_title("历史实测：每 decode step 的 phase 分解（lite trace，5×25 s 窗口，rank0；阶梯=step 周期）")
    ax.legend(ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    shares = [fnum(r["prepare_input_share_pct"]) for r in rows]
    colors = ["#e4572e" if r["mtp"] == "on" else "#3d7ea6" for r in rows]
    ax2.bar(range(len(rows)), shares, color=colors, width=0.72)
    for i, s in enumerate(shares):
        ax2.text(i, s + 1, f"{s:.0f}%", ha="center", fontsize=8)
    ax2.set_ylabel("prepare input 占比 (%)")
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=7.5,
                        rotation=40, ha="right")
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_ylim(0, max(shares) * 1.2)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIG, f"hist-01-phase-decomposition.{ext}"), dpi=150)
    print("wrote hist-01-phase-decomposition.svg/.png")


def fig2() -> None:
    rows = [r for r in read(os.path.join(HIST, "phase_per_step_by_tag.csv")) if r["rank"] == "0"]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    style = {
        ("graph", "off"): ("#3d7ea6", "o", "graph · MTP-off"),
        ("graph", "on"): ("#e4572e", "s", "graph · MTP-on"),
        ("eager", "off"): ("#7fb069", "^", "eager · MTP-off"),
        ("eager", "on"): ("#8d5b4c", "v", "eager · MTP-on"),
    }
    seen = set()
    for r in rows:
        key = (r["mode"], r["mtp"])
        c, m, lab = style[key]
        x = fnum(r["step_period_us"]) / 1000.0
        y = fnum(r["prepare_input_ms"])
        ax.scatter(x, y, color=c, marker=m, s=52, label=lab if key not in seen else None,
                   edgecolor="white", linewidth=0.6, zorder=3)
        seen.add(key)
        # 只标注「27B graph + MTP-on」这一簇（本图的关键异常）
        if r["mode"] == "graph" and r["mtp"] == "on" and "27b" in r["model"]:
            ax.annotate(f"{r['model']} [{r['batch_tag']}]", (x, y), fontsize=8,
                        xytext=(6, 4), textcoords="offset points", color="#a3341a")
    ax.axhspan(2.0, 8.0, color="#cfcfcf", alpha=0.22, zorder=0)
    ax.text(4.7, 3.4, "所有其它臂：prepare input 2–8 ms/步\n（与模型规模/TP 关系很弱）",
            fontsize=9, color="#444444")
    ax.annotate("graph + MTP-on 的 27B：\nprepare input 29–35 ms/步\n（自旋等待，不是 CPU 工作量）",
                xy=(42, 31), xytext=(9.5, 14), fontsize=9, color="#a3341a",
                arrowprops=dict(arrowstyle="->", color="#a3341a", lw=1.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step 周期（ms/步，log）")
    ax.set_ylabel("prepare input 墙钟（ms/步，log）")
    ax.set_title("prepare input 不是常量：graph+MTP-on 的 27B 臂把它抬到 30–35 ms\n"
                 "（同一批配置里其它臂只有 2–7 ms）")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIG, f"hist-02-prepare-vs-step-period.{ext}"), dpi=150)
    print("wrote hist-02-prepare-vs-step-period.svg/.png")


def fig3() -> None:
    rows = read(os.path.join(HIST, "worker_thread_topdown_configs.csv"))
    rows = [r for r in rows if r["rank"] == "0" and r["mode"] == "graph"]
    rows.sort(key=lambda r: (r["model"], r["tp"], r["mtp"]))
    labels = [f"{r['model']} {r['tp']} MTP-{r['mtp']}" for r in rows]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    bottom = [0.0] * len(rows)
    for col, name, color in [
        ("frontend_bound_mean_pct", "frontend-bound", "#e4572e"),
        ("bad_spec_mean_pct", "bad-spec", "#f2c14e"),
        ("retiring_mean_pct", "retiring", "#7fb069"),
        ("backend_bound_mean_pct", "backend-bound", "#3d7ea6"),
    ]:
        vals = [fnum(r[col]) for r in rows]
        ax.bar(range(len(rows)), vals, bottom=bottom, label=name, color=color, width=0.72)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("top-down L1 (%)")
    ax.set_title("worker forward 主线程 top-down L1（graph 模式，perf stat -t <main TID>，"
                 "percent_enabled=100.00）")
    ax.legend(ncols=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ipcs = [fnum(r["ipc_mean"]) for r in rows]
    ax2.bar(range(len(rows)), ipcs, color="#5c5c5c", width=0.72)
    for i, s in enumerate(ipcs):
        ax2.text(i, s + 0.015, f"{s:.2f}", ha="center", fontsize=8)
    ax2.set_ylabel("IPC（inst_retired / cycles）")
    ax2.set_ylim(0, max(ipcs) * 1.25)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIG, f"hist-03-topdown-ipc.{ext}"), dpi=150)
    print("wrote hist-03-topdown-ipc.svg/.png")


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    main()
