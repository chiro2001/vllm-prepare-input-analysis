#!/usr/bin/env python3
"""sweep 分析与拟合（preset 分口径）→ data/harness/sweep_fits.json + sweep_points.csv。

统一口径：
    pi_net_us      = prepare_inputs_us - triton_cpu_us      # ★ 所有结论用这个
    scope_total_us = update_states_us + prepare_inputs_us   # harness 默认口径
    scope_net_us   = scope_total_us - triton_cpu_us

**preset 是本报告的一等公民**：同一维度在不同 preset 下的 c/a 不可混池
（max_model_len 决定 token_ids_cpu_tensor 大小，max_num_reqs 决定 Python 循环长度）。

数据来源：
  A `sweep_<tag>_<preset>_*.csv`            preset_grid.py 产出（含 raw/net/triton/meta）
  B `sweep_steps_<preset>_<tag>.csv`        逐 step 池（整体建模用）
  C `sweep_modecmp_rm_*.csv`                交错三模式对照
  D `sweep_noise_*.json`                    hostnoise gate
  E `hw64_sweep_LEGACY_*.csv`               旧口径原始 sweep（只作历史引用）

用法：python3 agents/sweep_analysis/analyze.py [--data data/harness]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sweep_lib as sl

STRESS_DIMS = ["isl", "batch", "chunk_size", "block_size", "spec_k", "prefix_hit_ratio"]
RM_DIMS = ["batch", "isl"]


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def isna(v):
    return v is None or v != v


# --------------------------------------------------------------------------- loaders
def load_preset_runs(data_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(data_dir.glob("sweep_*_realmachine_*.csv")) + \
            sorted(data_dir.glob("sweep_*_stress_*.csv")):
        if f.name.startswith("sweep_steps"):
            continue
        for i, r in enumerate(read_csv(f), start=2):
            if r.get("error") or not r.get("dim"):
                continue
            r["_file"] = f.name
            r["_row"] = i
            rows.append(r)
    return rows


def load_steps(data_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(data_dir.glob("sweep_steps_*.csv")):
        for i, r in enumerate(read_csv(f), start=2):
            rec = {k: fnum(v) for k, v in r.items()
                   if k not in ("dim", "phase", "preset", "slot_mapping_mode")}
            rec["dim"] = r["dim"]
            rec["phase"] = r["phase"]
            rec["preset"] = r.get("preset", "")
            rec["slot_mapping_mode"] = r.get("slot_mapping_mode", "")
            rec["_file"] = f.name
            rows.append(rec)
    return rows


def load_modes(data_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(data_dir.glob("sweep_modecmp_rm_*.csv")):
        # 只取网格文件（`<tag>_<ts>.csv`）；排除 `<tag>_r<k>_<mode>_per_step_<ts>.csv`
        if "_per_step_" in f.name or "_substep_" in f.name or "_summary_" in f.name:
            continue
        for i, r in enumerate(read_csv(f), start=2):
            if r.get("error") or not r.get("mode"):
                continue
            rec = {k: fnum(v) for k, v in r.items() if k != "mode"}
            rec["mode"] = r["mode"]
            rec["_file"] = f.name
            rec["_row"] = i
            rows.append(rec)
    return rows


def load_noise(data_dir: Path) -> dict:
    out: dict = {"gates": {}, "slice": "200-215"}
    for f in sorted(data_dir.glob("sweep_noise_*.json")):
        d = json.loads(f.read_text())
        cpu = d.get("cpu", {})
        out["gates"][f.stem.replace("sweep_noise_", "")] = {
            "file": f.name,
            "mean_busy_pct": cpu.get("mean_busy_pct"),
            "max_busy_pct": cpu.get("max_busy_pct"),
            "hot_cpus": cpu.get("hot_cpus"),
            "n_offenders_on_slice": cpu.get("n_offenders_on_slice"),
            "slice_quiet": cpu.get("slice_quiet"),
            "top_offender_cpu_pct": [o.get("cpu_pct") for o in (cpu.get("offenders_on_slice") or [])[:5]],
        }
    means = [g["mean_busy_pct"] for g in out["gates"].values() if g.get("mean_busy_pct") is not None]
    if means:
        out["mean_busy_pct_across_gates"] = float(np.mean(means))
        out["mean_busy_pct_min"] = float(np.min(means))
        out["mean_busy_pct_max"] = float(np.max(means))
        out["any_gate_quiet"] = any(g.get("slice_quiet") for g in out["gates"].values())
    return out


# --------------------------------------------------------------------------- aggregation
def agg_by_value(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r["value"]), []).append(r)
    out = {}
    for val, rs in by.items():
        def m(key):
            vs = [fnum(r[key]) for r in rs if not isna(fnum(r.get(key)))]
            return float(np.median(vs)) if vs else float("nan")

        def rng(key):
            vs = [fnum(r[key]) for r in rs if not isna(fnum(r.get(key)))]
            return (float(np.min(vs)), float(np.max(vs))) if vs else (float("nan"), float("nan"))

        rec = {"x": fnum(val), "x_label": val, "n_runs": len(rs),
               "files": sorted({r["_file"] for r in rs})}
        for key, short in (("pi_raw_p50_us", "pi_raw_p50"), ("pi_net_p50_us", "pi_net_p50"),
                           ("pi_net_p90_us", "pi_net_p90"),
                           ("pi_net_p50_decode_us", "pi_net_p50_decode"),
                           ("pi_net_p50_prefill_us", "pi_net_p50_prefill"),
                           ("triton_p50_us", "triton_p50"), ("us_p50_us", "us_p50"),
                           ("scope_raw_p50_us", "scope_raw_p50"),
                           ("scope_net_p50_us", "scope_net_p50"),
                           ("tokens_per_step_p50", "tokens_per_step"),
                           ("tokens_per_step_decode_p50", "tokens_per_step_decode"),
                           ("tokens_per_step_prefill_p50", "tokens_per_step_prefill"),
                           ("logits_len_p50", "logits_len_p50"),
                           ("reqs_per_step_p50", "reqs_per_step"),
                           ("n_prefill_steps", "n_prefill_steps"),
                           ("n_decode_steps", "n_decode_steps"),
                           ("steps_run", "steps_run"),
                           ("max_num_blocks_per_req", "max_num_blocks_per_req"),
                           ("max_model_len", "max_model_len"),
                           ("max_num_reqs_cfg", "max_num_reqs"),
                           ("token_ids_cpu_MB", "token_ids_cpu_MB")):
            rec[short] = m(key)
            if short in ("pi_net_p50", "pi_raw_p50", "triton_p50", "us_p50", "scope_net_p50"):
                lo, hi = rng(key)
                rec[short + "_min"], rec[short + "_max"] = lo, hi
                rec[short + "_range"] = hi - lo
        rec["has_spec_decode"] = any(r.get("has_spec_decode") in ("True", True) for r in rs)
        rec["launches_set"] = sorted({r.get("triton_launches_set", "") for r in rs})
        rec["slot_mapping_mode"] = sorted({r.get("slot_mapping_mode", "") for r in rs})
        rec["async_scheduling"] = sorted({r.get("async_scheduling", "") for r in rs})
        rec["prefix_caching"] = sorted({r.get("prefix_caching", "") for r in rs})
        out[val] = rec
    return out


def fit_dim(points: list[dict], ykey: str) -> dict:
    sel = [p for p in points if not isna(p.get(ykey))]
    if len(sel) < 2:
        return {"c": float("nan"), "a": float("nan"), "r2": float("nan"), "n_points": len(sel)}
    xs = [p["x"] for p in sel]
    ys = [p[ykey] for p in sel]
    f = sl.fit_line(xs, ys)
    f["x_min"], f["x_max"] = min(xs), max(xs)
    f["y_at_x_min"] = f["c"] + f["a"] * f["x_min"]
    f["y_at_x_max"] = f["c"] + f["a"] * f["x_max"]
    f["constant_share_at_x_min"] = sl.constant_share(f["c"], f["a"], f["x_min"])
    f["constant_share_at_x_max"] = sl.constant_share(f["c"], f["a"], f["x_max"])
    f["per_unit_ns"] = f["a"] * 1000.0
    # 分区间判断：前两点 / 后两点各自斜率，看哪一段是斜率主导
    if len(sel) >= 4:
        lo = sl.fit_line(xs[:2], ys[:2])
        hi = sl.fit_line(xs[-2:], ys[-2:])
        f["slope_first_2"] = lo["a"]
        f["slope_last_2"] = hi["a"]
        f["slope_ratio_last_over_first"] = (
            hi["a"] / lo["a"] if lo["a"] not in (0.0,) and not isna(lo["a"]) else float("nan"))
    return f


def fit_blocks(pts_by_dim: dict, dims: list[str], group: dict) -> dict:
    out = {}
    for dim in dims:
        if dim not in pts_by_dim:
            continue
        points = list(pts_by_dim[dim].values())
        points.sort(key=lambda p: p["x"])
        ranges = [p["pi_raw_p50_range"] for p in points if not isna(p.get("pi_raw_p50_range"))]
        entry = {
            "preset": group["preset"],
            "dim": dim,
            "x": "ratio" if dim == "prefix_hit_ratio" else (
                "count" if dim in ("batch", "block_size", "spec_k") else "tokens"),
            "primary_metric": "pi_net_us p50（每步；已扣 triton_cpu_us）",
            "pi_net": fit_dim(points, "pi_net_p50"),
            "us": fit_dim(points, "us_p50"),
            "scope_net": fit_dim(points, "scope_net_p50"),
            "pi_raw": fit_dim(points, "pi_raw_p50"),
            "triton": fit_dim(points, "triton_p50"),
            "noise_band": {
                "median_pi_raw_p50_range_us": float(np.median(ranges)) if ranges else None,
                "max_pi_raw_p50_range_us": float(max(ranges)) if ranges else None,
                "note": "同配置重复轮之间 pi_raw p50 的极差（raw 口径），仅本维度点数",
            },
            "n_points": len(points),
            "points": points,
        }
        out[dim] = entry
    return out


def pooled_model(steps: list[dict], preset: str, tag_filter: str | None = None) -> dict:
    rows = [s for s in steps if s["preset"] == preset and s["phase"] == "decode"
            and (tag_filter is None or tag_filter in s["_file"])]
    if len(rows) < 5:
        return {"n_steps": len(rows)}
    out = {"n_steps": len(rows), "sources": sorted({r["_file"] for r in rows})}
    for label, key in (("vs_num_reqs", "num_reqs"), ("vs_tokens_per_step", "total_scheduled_tokens")):
        xs = [r[key] for r in rows]
        for ykey, yname in (("pi_net_us", "pi_net"), ("update_states_us", "us")):
            ys = [r[ykey] for r in rows]
            f = sl.fit_line(xs, ys)
            f["per_unit_ns"] = f["a"] * 1000.0
            out[f"{yname}_{label}"] = f
    B = np.array([[1.0, r["num_reqs"], r["total_scheduled_tokens"]] for r in rows])
    for yname, ykey in (("pi_net", "pi_net_us"), ("us", "update_states_us")):
        y = np.array([r[ykey] for r in rows])
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
        pred = B @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[f"{yname}_two_var"] = {
            "c": float(coef[0]), "a_num_reqs": float(coef[1]),
            "a_tokens_per_step": float(coef[2]),
            "a_num_reqs_ns": float(coef[1]) * 1000.0,
            "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }
    return out


def mode_summary(modes: list[dict]) -> dict:
    out: dict = {}
    for mode in sorted({m["mode"] for m in modes}):
        sel = [m for m in modes if m["mode"] == mode]
        out[mode] = {
            "n_rounds": len(sel),
            "pi_raw_p50_median_us": float(np.median([m["pi_raw_p50_us"] for m in sel])),
            "pi_raw_p50_min_us": float(np.min([m["pi_raw_p50_us"] for m in sel])),
            "pi_raw_p50_max_us": float(np.max([m["pi_raw_p50_us"] for m in sel])),
            "triton_p50_median_us": float(np.median([m["triton_p50_us"] for m in sel])),
            "pi_net_p50_median_us": float(np.median([m["pi_net_p50_us"] for m in sel])),
            "us_p50_median_us": float(np.median([m["us_p50_us"] for m in sel])),
            "scope_net_p50_median_us": float(np.median([m["scope_net_p50_us"] for m in sel])),
            "steps_run": int(np.median([m["steps_run"] for m in sel])),
        }
    # ★ 同 round 配对差值（唯一可信的"模式间"差值；跨窗口比较会被漂移污染）
    by_round: dict[tuple[int, str], float] = {
        (int(m["round"]), m["mode"]): m["pi_net_p50_us"] for m in modes}
    if "noop" in out:
        base = out["noop"]
        for k in ("cpu_fallback", "inject:15"):
            if k in out:
                out[k]["delta_pi_raw_vs_noop_us"] = base and (
                    out[k]["pi_raw_p50_median_us"] - base["pi_raw_p50_median_us"])
                out[k]["delta_triton_vs_noop_us"] = (
                    out[k]["triton_p50_median_us"] - base["triton_p50_median_us"])
                out[k]["delta_pi_net_vs_noop_us"] = (
                    out[k]["pi_net_p50_median_us"] - base["pi_net_p50_median_us"])
        out["device_interaction_share_of_noop_pi_net_pct"] = {
            k: (out[k]["triton_p50_median_us"] / base["pi_net_p50_median_us"] * 100.0)
            for k in list(out)
            if k in ("cpu_fallback", "inject:15")
        }
        out["device_interaction_share_of_noop_pi_raw_pct"] = {
            k: (out[k]["triton_p50_median_us"] / base["pi_raw_p50_median_us"] * 100.0)
            for k in list(out)
            if k in ("cpu_fallback", "inject:15")
        }
    paired = {}
    for label, base in (("cpu_fallback", "noop"), ("inject:15", "noop"),
                        ("inject:15", "cpu_fallback")):
        rounds = sorted({r for (r, m) in by_round if m == label}
                        & {r for (r, m) in by_round if m == base})
        ds = [by_round[(r, label)] - by_round[(r, base)] for r in rounds]
        if ds:
            paired[f"{label}_minus_{base}"] = {
                "metric": "pi_net_p50_us (同 round 配对)",
                "per_round_us": [float(d) for d in ds],
                "median_us": float(np.median(ds)),
                "mean_us": float(np.mean(ds)),
                "min_us": float(np.min(ds)),
                "max_us": float(np.max(ds)),
                "n_rounds": len(ds),
                "source": sorted({m["_file"] for m in modes}),
            }
    out["paired_within_round"] = paired
    return out


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/harness")
    args = ap.parse_args(argv)
    data = Path(args.data)

    runs = load_preset_runs(data)
    steps = load_steps(data)
    modes = load_modes(data)
    noise = load_noise(data)

    # 同一 preset 的**不同窗口**批次必须分开（stress 原网格 vs stresssub 复测）；
    # 否则跨窗口漂移会污染同一个拟合。
    def stress_sub(r):  # stresssub 批次（block_size 复测, 2 轮）
        return r["_file"].startswith(("sweep_stresssub_stress_",))

    def stress_main(r):
        return (r["_file"].startswith("sweep_stress_stress_")) and not stress_sub(r)

    by_preset: dict[str, list[dict]] = {}
    for r in runs:
        bucket = "stresssub" if stress_sub(r) else r["preset"]
        by_preset.setdefault(bucket, []).append(r)

    # 同 preset 同 dim 可能有多份（rm 与 rm64、rm warmup3 与 warmup0）：用 file 前缀区分
    groups = {
        "realmachine": [
            ("rm_batch_mmr8", "batch", lambda r: r["_file"].startswith("sweep_rm_realmachine_")
             and r["dim"] == "batch"),
            ("rm_isl", "isl", lambda r: r["_file"].startswith("sweep_rm_realmachine_")
             and r["dim"] == "isl"),
            ("rm_isl_warmup0", "isl", lambda r: r["_file"].startswith("sweep_rmw0_")),
            ("rm_batch_mmr64", "batch", lambda r: r["_file"].startswith("sweep_rm64_")),
            ("rm_steady12s", "batch", lambda r: r["_file"].startswith("sweep_rmsteady_")),
        ],
        "stress": [(f"stress_{d}", d, (lambda dd: (lambda r: r["dim"] == dd and
                                                   r["_file"].startswith("sweep_stress_stress_")))(d))
                   for d in STRESS_DIMS],
        "stresssub": [("stresssub_block_size", "block_size",
                       lambda r: r["dim"] == "block_size")],
    }

    fits: dict = {}
    for preset, spec_list in groups.items():
        rows = by_preset.get(preset, [])
        fits[preset] = {}
        for name, dim, pred in spec_list:
            sel = [r for r in rows if pred(r)]
            if not sel:
                continue
            agg = agg_by_value(sel)
            pts = {dim: agg}
            fits[preset][name] = fit_blocks(pts, [dim], {"preset": preset})[dim]
            fits[preset][name]["notes"] = name

    # 主图用的"规范"入口
    canonical = {
        "realmachine": {
            "batch_mmr8": "rm_batch_mmr8", "batch_mmr64": "rm_batch_mmr64",
            "isl": "rm_isl", "isl_warmup0": "rm_isl_warmup0", "steady": "rm_steady12s",
        },
        "stress": {d: f"stress_{d}" for d in STRESS_DIMS},
    }

    # ---------------------------------------------------------------- ① 请求的正式 schema
    #  {preset, dim, x_values, y_values, c_us, a_us_per_unit, r2, n_points,
    #   noise_band_pct, caliber}
    def schema_entry(preset: str, name: str, fit: dict, ykey: str = "pi_net") -> dict:
        f = fit[ykey]
        pts = fit["points"]
        xs = [p["x"] for p in pts if not isna(p.get(f"{ykey}_p50"))]
        ys = [p[f"{ykey}_p50"] for p in pts if not isna(p.get(f"{ykey}_p50"))]
        # 噪声带用 **百分比**：同配置重复轮的 (max-min)/median
        pcts = []
        for p in pts:
            med = p.get("pi_raw_p50")
            rng = p.get("pi_raw_p50_range")
            if med and not isna(med) and not isna(rng) and med > 0:
                pcts.append(rng / med * 100.0)
        xs_all = [p["x"] for p in pts if not isna(p.get("pi_net_p50"))]
        return {
            "preset": preset,
            "dim": fit["dim"],
            "entry": name,
            "x_values": xs,
            "y_values": ys,
            "caliber": "pi_net_p50_us" if ykey == "pi_net" else f"{ykey}_p50_us",
            "c_us": f["c"],
            "a_us_per_unit": f["a"],
            "a_unit": "us per " + fit["x"],
            "r2": f["r2"],
            "n_points": f["n_points"],
            "noise_band_pct": float(np.median(pcts)) if pcts else None,
            "noise_band_pct_max": float(max(pcts)) if pcts else None,
            "x_range": [min(xs_all), max(xs_all)] if xs_all else None,
            "source_files": sorted({f_ for p in pts for f_ in p["files"]}),
        }

    formal: list[dict] = []
    for preset, spec_list in groups.items():
        for name, dim, pred in spec_list:
            if name not in fits[preset]:
                continue
            fit = fits[preset][name]
            formal.append(schema_entry(preset, name, fit, "pi_net"))
            formal.append(schema_entry(preset, name, fit, "us"))
            formal.append(schema_entry(preset, name, fit, "scope_net"))

    out = {
        "schema": "sweep-fits-v3-preset",
        "formal_fits": formal,
        "ipc_preset_sensitivity": {
            "source": "docs/06-synthetic-load.md §9.2（PMU/libkperfx 920B preset, 9 组 confidence=1.0）",
            "caliber": "IPC（同一 prepare_input 代码，只换 preset）",
            "rows": [
                {"preset": "realmachine", "token_ids_cpu_shape": "(8, 2048)",
                 "working_set": "64 KB", "ipc": 0.949,
                 "note": "与真机历史区间 0.719–0.890 同区间"},
                {"preset": "stress", "token_ids_cpu_shape": "(64, 32768)",
                 "working_set": "8 MB", "ipc": 1.553,
                 "note": "远离真机，不能用于一致性对照"},
            ],
            "delta_pct": 63.6,
            "conclusion": "同一个 prepare_input 代码，只换引擎启动参数，IPC 差 64%",
        },
        "metric_convention": {
            "pi_net_us": "prepare_inputs_us - triton_cpu_us",
            "scope_total_us": "update_states_us + prepare_inputs_us",
            "scope_net_us": "scope_total_us - triton_cpu_us",
            "stat": "per-step p50 unless stated; 跨轮取中位数",
            "preset_note": "同一维度的 c/a 只在同一 preset 内可比；preset 决定 max_model_len "
                           "→ token_ids_cpu_tensor 形状 → cache 行为",
        },
        "presets": {
            "realmachine": "max_model_len=2048 / max_num_reqs=8 / max_num_batched_tokens=2048 / "
                           "no-prefix-caching / async=on / slot_mapping=noop（真机 launcher 口径）",
            "stress": "max_model_len=32768 / 64 reqs / 16384 tokens / prefix-caching=on / "
                      "async=off / slot_mapping=noop（放斜率用）",
        },
        "canonical_entries": canonical,
        "noise": noise,
        "fits": fits,
        "pooled_step_models": {
            "realmachine": pooled_model(steps, "realmachine", "sweep_steps_realmachine_rm"),
            "realmachine_mmr64": pooled_model(steps, "realmachine", "sweep_steps_realmachine_rm64"),
            "stress": pooled_model(steps, "stress", "sweep_steps_stress_stress"),
        },
        "mode_interleaved_realmachine": mode_summary(modes),
        "raw_rows": {
            "preset_runs": {r["_file"]: None for r in runs},
        },
    }
    (data / "sweep_fits.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = ["preset,entry,dim,x,source,pi_raw_p50,pi_net_p50,triton_p50,us_p50,scope_net_p50,"
             "n_runs,range_pi_raw"]
    for preset, spec_list in groups.items():
        for name, dim, pred in spec_list:
            if name not in fits[preset]:
                continue
            for p in fits[preset][name]["points"]:
                lines.append(f"{preset},{name},{dim},{p['x']},{'|'.join(p['files'])},"
                             f"{p['pi_raw_p50']},{p['pi_net_p50']},{p['triton_p50']},{p['us_p50']},"
                             f"{p['scope_net_p50']},{p['n_runs']},{p['pi_raw_p50_range']}")
    (data / "sweep_points.csv").write_text("\n".join(lines) + "\n")

    print(f"[analyze] wrote {data/'sweep_fits.json'} and {data/'sweep_points.csv'}")
    for preset, entries in fits.items():
        print(f"\n== preset={preset}")
        for name, f in entries.items():
            n = f["pi_net"]
            u = f["us"]
            print(f"   {name:18s} dim={f['dim']:17s} pi_net: c={n['c']:7.2f} a={n['a']:9.4f} "
                  f"r2={n['r2']:6.3f} n={n['n_points']:2d} | us: a={u['a']:8.4f} r2={u['r2']:6.3f}")
    print("\n== 交错三模式（realmachine）")
    for k, v in out["mode_interleaved_realmachine"].items():
        if isinstance(v, dict):
            print(f"   {k:16s} raw={v.get('pi_raw_p50_median_us', float('nan')):7.1f} "
                  f"triton={v.get('triton_p50_median_us', float('nan')):7.1f} "
                  f"net={v.get('pi_net_p50_median_us', float('nan')):7.1f} "
                  f"Δnet_vs_noop={v.get('delta_pi_net_vs_noop_us', 0.0):+6.1f}")
    print(f"\n== 噪声门: {noise.get('mean_busy_pct_min')}–{noise.get('mean_busy_pct_max')}% "
          f"(gates={list(noise['gates'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
