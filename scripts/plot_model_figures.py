#!/usr/bin/env python3
"""从 data/model/*.csv|jsonl 生成 docs/07 用的两张图（SVG+PNG）。

输出：
    figures/07-replica-cost-curve.svg|png   合成负载 µs/step vs B（decode/chunked/MTP）
    figures/07-ipc-profile.svg|png          IPC 与分支失败率（负载特征指纹）
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

OUT = "figures"
DATA = "data/model"


def load_composite() -> dict:
    rows = list(csv.DictReader(open(f"{DATA}/microbench-composite.csv")))
    d: dict[str, dict[int, float]] = {"decode": {}, "spec_mtp2": {}}
    for r in rows:
        grp, B, m = r["group"], int(r["B"]), int(r["tokens_per_req"])
        v = float(r["median_us"])
        if grp == "decode":
            d["decode"][B] = v
        elif grp == "spec_mtp2":
            d["spec_mtp2"][B] = v
        else:
            d.setdefault(f"m={m}", {})[B] = v
    return d


def fig_cost_curve(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    Bs = sorted(d["decode"])
    ax.plot(Bs, [d["decode"][b] for b in Bs], "o-", lw=2, label="decode (m=1)")
    ax.plot(sorted(d["spec_mtp2"]), [d["spec_mtp2"][b] for b in sorted(d["spec_mtp2"])],
            "s--", label="decode + MTP n=2")
    for key, style in [("m=4", ":"), ("m=16", "-."), ("m=64", "--"), ("m=256", "-")]:
        if key in d:
            bs = sorted(d[key])
            ax.plot(bs, [d[key][b] for b in bs], style, alpha=0.85,
                    label=f"chunked {key} (T=B·{key[2:]})")
    # 拟合线
    x = np.array(Bs, dtype=float)
    y = np.array([d["decode"][b] for b in Bs])
    A = np.vstack([np.ones_like(x), x]).T
    (c0, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    xs = np.linspace(1, max(Bs), 50)
    ax.plot(xs, c0 + a * xs, "k--", lw=1, alpha=0.6,
            label=f"decode 拟合: {c0:.1f} + {a:.3f}·B µs")
    ax.set_xlabel("B (num_reqs)")
    ax.set_ylabel("replica 单步耗时 (µs, CPU-only)")
    ax.set_title("prepare_input CPU-only 合成负载：单步耗时 vs B\n(920B 核200-203, 单线程; 不含真 H2D)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT}/07-replica-cost-curve.{ext}", dpi=150)
    plt.close(fig)
    print(f"figures/07-replica-cost-curve.svg  (c0={c0:.2f} us, a={a:.4f} us/req, R2={_r2(x,y,c0,a):.5f})")


def _r2(x, y, c0, a) -> float:
    pred = c0 + a * x
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fig_ipc() -> None:
    rows = [json.loads(l) for l in open(f"{DATA}/ipc-summary.jsonl")]
    labels = [r["label"] for r in rows]
    us = [r["us_per_step"] for r in rows]
    # IPC/分支失败率来自 perf stat（见 docs/07 §6.1b），此处按同一顺序硬编码记录值
    ipc = [1.57, 1.95, 2.39, 2.93]
    bmiss = [3.73, 2.51, 1.51, 0.82]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, ipc, width=0.4, label="IPC", color="#3b7dd8")
    ax.set_ylabel("IPC (instructions/cycle)", color="#3b7dd8")
    ax.set_ylim(0, 3.4)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, bmiss, width=0.4, label="分支失败率 %", color="#d8723b")
    ax2.set_ylabel("分支失败率 (%)", color="#d8723b")
    ax2.set_ylim(0, 5)
    for i, u in enumerate(us):
        ax.text(i, max(ipc[i], bmiss[i] * 0.68) + 0.15, f"{u:.0f} µs/step",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("合成负载的 CPU 特征指纹：小 batch 解释器主导（低 IPC/高分支失败）\n→ 大 token 内存/向量主导")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT}/07-ipc-profile.{ext}", dpi=150)
    plt.close(fig)
    print("figures/07-ipc-profile.svg")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    d = load_composite()
    fig_cost_curve(d)
    if os.path.exists(f"{DATA}/ipc-summary.jsonl"):
        fig_ipc()


if __name__ == "__main__":
    main()
