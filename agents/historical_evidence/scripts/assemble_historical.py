#!/usr/bin/env python3
"""把 HIST_PROJECT 的历史实测数据汇编成 prepare_input 可引用的整洁表。

只读来源（全部在 /home/chiro/projects/vllm/HIST_PROJECT/ 下）：
  reports/lite-traces/manifest.json                      （lite trace 索引）
  reports/topdown-campaign/{topdown-l1-windows,topdown-l1,per-config-rank}.csv
  reports/topdown-campaign/windows.jsonl
  reports/mode-campaign-815/per-config.csv               （graph vs eager，含吞吐）
  reports/spin-attribution/windows.csv                   （五臂自旋归因）
  reports/mtp-attribution/{attribution,tokens-per-step}.csv
  vllm-slice-insn-opt/results/SUMMARY.{csv,json}          （ROI 指令/周期）
  vllm-slice-insn-opt/results/perf-prof-08b-mtp-*.flat.txt（带 IPC 列的剖面）
agents/historical_evidence/raw/                          （本 agent 从 a3-21 只读复制的原始件）
  a321_perf_analysis/record-rank0.stacks.json
  a321_perfstat/*.stat-{ipc,software}.csv / *.normalized.json

输出：data/historical/*.csv + *.meta.json（除 phase_* 由 extract_lite_phases.py 产出）
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys

MC = "/home/chiro/projects/vllm/HIST_PROJECT"
OUT = "/home/chiro/projects/vllm/preparing-input-phase/data/historical"
RAW = "/home/chiro/projects/vllm/preparing-input-phase/agents/historical_evidence/raw"


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


def fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# 1) phase：把「每窗口 × 每作用域」聚成「每配置」
# --------------------------------------------------------------------------
PHASE_ORDER = [
    "Step:Schedule",
    "prepare input",
    "forward",
    "post process",
    "sample_token",
    "draft_token",
    "async_state_update",
    "Step:Output",
    "model: mtp: propose",
    "model: mtp: first_pass",
    "model: mtp: draft_forward",
]

# 只有这些是「每步一次、彼此不嵌套」的顶层作用域；求和算覆盖率时必须排除
# 嵌套子作用域（schedule:* 在 Step:Schedule 内，output:* 在 Step:Output 内，
# model: mtp:* 在 draft_token 内），否则会重复计数。
TOP_LEVEL_FOR_SUM = [
    "Step:Schedule",
    "prepare input",
    "forward",
    "post process",
    "sample_token",
    "draft_token",
    "async_state_update",
    "Step:Output",
]


def build_phase_configs() -> None:
    rows = read_csv(os.path.join(OUT, "phase_per_step_windows.csv"))
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], r["tp"], r["mode"], r["mtp"], r["rank"])
        g = groups.setdefault(
            key,
            {
                "windows": 0,
                "tags": set(),
                "runs": set(),
                "step_period": [],
                "anchors": set(),
                "scope": {},
            },
        )
        g["windows"] += 1
        g["tags"].add(r["batch_tag"])
        g["runs"].add(r["run"])
        g["anchors"].add(r["step_anchor"])
        if r["step_period_us"]:
            g["step_period"].append(float(r["step_period_us"]))
        g["scope"].setdefault(r["scope"], []).append(float(r["mean_us"]))

    out: list[dict] = []
    def sort_key(kv):
        k = kv[0]
        return (k[0] or "", int(k[1] or 0), k[2] or "", k[3] or "", int(k[4] or 0))

    for key, g in sorted(groups.items(), key=sort_key):
        period = statistics.fmean(g["step_period"]) if g["step_period"] else float("nan")
        row = {
            "model": key[0],
            "tp": key[1],
            "mode": key[2],
            "mtp": key[3],
            "rank": key[4],
            "n_windows": g["windows"],
            "n_runs": len(g["runs"]),
            "batch_tags": ";".join(sorted(g["tags"])),
            "step_anchor": ";".join(sorted(g["anchors"])),
            "step_period_us": round(period, 1),
            "step_per_second": round(1e6 / period, 2) if period else "",
        }
        named = 0.0
        nested_double_count = 0.0
        for name in PHASE_ORDER:
            vals = g["scope"].get(name)
            col = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
            if vals:
                m = statistics.fmean(vals)
                row[f"{col}_ms"] = round(m / 1000.0, 4)
                row[f"{col}_share_pct"] = round(100.0 * m / period, 2) if period else ""
                if name in TOP_LEVEL_FOR_SUM:
                    named += m
                else:
                    nested_double_count += m
            else:
                row[f"{col}_ms"] = ""
                row[f"{col}_share_pct"] = ""
        row["top_level_scopes_sum_ms"] = round(named / 1000.0, 4)
        row["top_level_coverage_pct"] = round(100.0 * named / period, 2) if period else ""
        row["nested_scopes_sum_ms"] = round(nested_double_count / 1000.0, 4)
        row["unaccounted_ms"] = round((period - named) / 1000.0, 4) if period else ""
        out.append(row)

    fields = list(out[0].keys())
    write_csv(os.path.join(OUT, "phase_per_step_configs.csv"), out, fields)

    # 同一配置在不同波次（tag）之间的差异也要能直接看到：w* 波次是第一轮，
    # e*/m* 波次是后来的复现轮。
    tag_groups: dict[tuple, dict] = {}
    tag_pool = {}
    for r in read_csv(os.path.join(MC, "reports/mode-campaign-815/per-config.csv")):
        tp_norm = r["tp"].lower().replace("tp", "")
        tag_pool[(r["model"], tp_norm, r["mode"], r["mtp"], r["tag"])] = (
            r["pool"] + f" | n_windows={r['windows']} | ranks={r['ranks']}"
        )
    for r in rows:
        key = (r["model"], r["tp"], r["mode"], r["mtp"], r["batch_tag"], r["rank"])
        g = tag_groups.setdefault(
            key, {"period": [], "scope": {}, "runs": set()}
        )
        g["runs"].add(r["run"])
        if r["step_period_us"]:
            g["period"].append(float(r["step_period_us"]))
        g["scope"].setdefault(r["scope"], []).append(float(r["mean_us"]))
    tag_out = []
    for key, g in sorted(tag_groups.items()):
        period = statistics.fmean(g["period"])
        row = {
            "model": key[0],
            "tp": key[1],
            "mode": key[2],
            "mtp": key[3],
            "batch_tag": key[4],
            "rank": key[5],
            "pool": tag_pool.get(key[:5], ""),
            "n_windows": len(g["period"]),
            "run_id": ";".join(sorted(g["runs"])),
            "step_period_us": round(period, 1),
            "step_per_second": round(1e6 / period, 2),
        }
        for name in TOP_LEVEL_FOR_SUM:
            vals = g["scope"].get(name)
            col = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
            row[f"{col}_ms"] = round(statistics.fmean(vals) / 1000.0, 4) if vals else ""
        tag_out.append(row)
    write_csv(
        os.path.join(OUT, "phase_per_step_by_tag.csv"),
        tag_out,
        list(tag_out[0].keys()),
    )
    with open(os.path.join(OUT, "phase_per_step_configs.meta.json"), "w") as fh:
        json.dump(
            {
                "description": "每个 (模型,TP,模式,MTP,rank) 的 decode 每步 phase 均值（毫秒/步）与占 step 周期百分比。",
                "derived_from": "data/historical/phase_per_step_windows.csv（本 agent 生成）",
                "upstream": "HIST_PROJECT/reports/lite-traces/<run>/lite.trace.json.gz",
                "tier": "historical-measured",
                "caveats": [
                    "w*/m4/v4b 波次用带 schedule 插桩的镜像；e1x/m1x 没有 Step:Schedule，步数退化为 prepare input 计数。",
                    "这些波次没有 wait scope：prepare input / forward 的墙钟包含 device 等待，"
                    "graph+MTP-on 臂尤其不能读成 CPU 侧耗时。",
                    "step_period_us = 窗口墙钟 / 窗口内步数，含未插桩的空隙；"
                    "named_coverage_pct 给出已插桩作用域对 step 周期的覆盖率。",
                ],
            },
            fh,
            indent=1,
        )


# --------------------------------------------------------------------------
# 2) worker 线程 top-down / IPC
# --------------------------------------------------------------------------
def build_topdown() -> None:
    src_w = os.path.join(MC, "reports/topdown-campaign/topdown-l1-windows.csv")
    rows = read_csv(src_w)
    out = []
    for r in rows:
        out.append(
            {
                "config_id": r["config_id"],
                "model": r["model"],
                "tp": r["tp"],
                "mode": r["execution_mode"],
                "mtp": r["mtp"],
                "rank": r["rank"],
                "replay": r["replay"],
                "run_id": r["run_id"],
                "worker_main_tid": r["worker_main_tid"],
                "frontend_bound_pct": r["frontend_bound"],
                "bad_spec_pct": r["bad_spec"],
                "retiring_pct": r["retiring"],
                "backend_bound_pct": r["backend_bound"],
                "buckets_sum_pct": r["buckets_sum"],
                "ipc": r["ipc"],
                "cycles": r["cycles"],
                "instructions": r["instructions"],
                "fetch_bubble": r["fetch_bubble"],
                "bad_spec_count": r["bad_spec_count"],
                "created_at": r["created_at"],
                "ok": r["ok"],
                "source_file": "HIST_PROJECT/reports/topdown-campaign/topdown-l1-windows.csv",
                "scope": "exact_worker_main_tid",
                "percent_enabled": "100.00 (verified on a3-22 raw.csv, 724/724 rows)",
            }
        )
    write_csv(
        os.path.join(OUT, "worker_thread_topdown_windows.csv"),
        out,
        list(out[0].keys()),
    )

    cfg = read_csv(os.path.join(MC, "reports/topdown-campaign/per-config-rank.csv"))
    cfg_out = []
    for r in cfg:
        model, tp, mode, mtp = r["config_id"].split(".")
        cfg_out.append(
            {
                "model": model,
                "tp": tp,
                "mode": {"FULL_DECODE_ONLY": "graph"}.get(mode, mode),
                "mtp": mtp.replace("mtp-", ""),
                "rank": r["rank"],
                "n_windows": r["windows"],
                "frontend_bound_mean_pct": r["frontend_mean"],
                "frontend_spread_pct": r["frontend_spread"],
                "bad_spec_mean_pct": r["bad_spec_mean"],
                "retiring_mean_pct": r["retiring_mean"],
                "backend_bound_mean_pct": r["backend_mean"],
                "backend_spread_pct": r["backend_spread"],
                "ipc_mean": r["ipc_mean"],
                "n_runs": r["runs"],
                "source_file": "HIST_PROJECT/reports/topdown-campaign/per-config-rank.csv",
                "instrument": "perf stat -t <worker main TID> -e r08,r2011,r11,r1b",
            }
        )
    write_csv(
        os.path.join(OUT, "worker_thread_topdown_configs.csv"),
        cfg_out,
        list(cfg_out[0].keys()),
    )

    # graph vs eager 对照（同一批 run 的两套模式）
    mc = read_csv(os.path.join(MC, "reports/mode-campaign-815/per-config.csv"))
    mc_out = []
    for r in mc:
        mc_out.append(
            {
                "config_id": r["config_id"],
                "model": r["model"],
                "tp": r["tp"],
                "mode": r["mode"],
                "mtp": r["mtp"],
                "rank_count": r["ranks"],
                "n_windows": r["windows"],
                "frontend_bound_mean_pct": r["frontend_mean"],
                "bad_spec_mean_pct": r["bad_spec_mean"],
                "retiring_mean_pct": r["retiring_mean"],
                "backend_bound_mean_pct": r["backend_bound_mean"],
                "ipc_mean": r["ipc_mean"],
                "throughput_tok_s": r["throughput_mean"],
                "throughput_spread": r["throughput_spread"],
                "lite_log_lines": r["lite_log_lines"],
                "source_file": "HIST_PROJECT/reports/mode-campaign-815/per-config.csv",
            }
        )
    write_csv(
        os.path.join(OUT, "worker_thread_mode_eager_vs_graph.csv"),
        mc_out,
        list(mc_out[0].keys()),
    )

    with open(os.path.join(OUT, "worker_thread_topdown.meta.json"), "w") as fh:
        json.dump(
            {
                "description": "worker forward producer 主线程的 top-down L1 四桶 + IPC。",
                "tier": "historical-measured",
                "scope": "exact_worker_main_tid（perf stat -t <worker main TID>）",
                "definition": "920B_topdown.txt: frontend=fetch_bubble/(6*cycles); bad_spec=(inst_spec-inst_retired)/(6*cycles); retiring=inst_retired/(6*cycle); backend=100-其余三桶",
                "confidence": {
                    "multiplexing": "无。a3-22 上 182 个 topdown-l1 raw.csv（43 个 run、724 行事件）全部 percent_enabled=100.00；"
                    "perf stat ipc/software 的 3 个有效 run 同样 100.00。",
                    "time_enabled_s": "topdown 窗口 4.48–24.65 s（均值 19.9 s）",
                },
                "sources": [
                    "reports/topdown-campaign/topdown-l1-windows.csv（362 个窗口）",
                    "reports/topdown-campaign/per-config-rank.csv",
                    "reports/mode-campaign-815/per-config.csv",
                ],
                "caveats": [
                    "graph+MTP-on 臂的四桶被下游流 event 的忙轮询污染（见 MTP_PREPARE_INPUT_SPIN.md §8.2/8.3），"
                    "不能直接读作 CPU 工作画像。",
                    "全部为 graph(FULL_DECODE_ONLY) 与 eager 两种执行模式，无 chunked-prefill/高并发维度的 topdown。",
                ],
            },
            fh,
            indent=1,
        )


# --------------------------------------------------------------------------
# 3) perf stat（cycles/instructions/task-clock/context-switch）
# --------------------------------------------------------------------------
def build_perfstat() -> None:
    out: list[dict] = []
    for fn in sorted(os.listdir(os.path.join(RAW, "a321_perfstat"))):
        if not fn.endswith(".stat-ipc.normalized.json"):
            continue
        run = fn.split(".stat-ipc")[0]
        with open(os.path.join(RAW, "a321_perfstat", fn)) as fh:
            d = json.load(fh)
        for rank in d["ranks"]:
            ev = rank["events"]
            row = {
                "run_id": run,
                "event_set": d["event_set"],
                "rank": rank["rank"],
                "chip": rank["chip"],
                "worker_main_tid": rank["tid"],
                "decode_after_release_seconds": d["decode_after_release_seconds"],
                "total_output_tokens": d["total_output_tokens"],
                "measured_output_tokens": d["measured_output_tokens"],
                "cycles": ev["cycles"]["value"],
                "instructions": ev["instructions"]["value"],
                "ipc": rank["derived"]["ipc"],
                "cycles_time_enabled_s": round(ev["cycles"]["time_enabled_ns"] / 1e9, 4),
                "cycles_time_running_percent": ev["cycles"]["time_running_percent"],
                "instructions_time_running_percent": ev["instructions"]["time_running_percent"],
                "cycles_per_decode_second": round(ev["cycles"]["per_decode_second"], 1),
                "instructions_per_decode_second": round(ev["instructions"]["per_decode_second"], 1),
                "cycles_per_output_token": round(ev["cycles"]["per_measured_output_token"], 1),
                "instructions_per_output_token": round(ev["instructions"]["per_measured_output_token"], 1),
                "scope": d["scope"],
                "source_file": f"agents/historical_evidence/raw/a321_perfstat/{fn}",
                "model": "qwen35-08b",
                "tp": "TP1",
                "mode": "graph",
                "mtp": "off",
            }
            out.append(row)

    # software event set 补进同一张表（task-clock / context-switches / page-faults）
    soft: dict[str, dict] = {}
    for fn in sorted(os.listdir(os.path.join(RAW, "a321_perfstat"))):
        if not fn.endswith(".stat-software.csv"):
            continue
        run = fn.split(".stat-software")[0]
        vals = {}
        for line in open(os.path.join(RAW, "a321_perfstat", fn)):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            flds = [x.strip() for x in line.split(",")]
            if len(flds) < 6:
                continue
            # 布局：<task-prefix>,value,unit,event,time_enabled,percent[,metric,comment]
            vals[flds[3]] = {
                "value": flds[1],
                "unit": flds[2],
                "time_enabled_ns": flds[4],
                "percent_enabled": flds[5],
            }
        soft[run] = vals
    for row in out:
        v = soft.get(row["run_id"], {})
        for name, key in [
            ("task-clock", "task_clock"),
            ("context-switches", "context_switches"),
            ("cpu-migrations", "cpu_migrations"),
            ("page-faults", "page_faults"),
        ]:
            row[f"{key}_value"] = v.get(name, {}).get("value", "")
            row[f"{key}_unit"] = v.get(name, {}).get("unit", "")
            row[f"{key}_percent_enabled"] = v.get(name, {}).get("percent_enabled", "")
    write_csv(os.path.join(OUT, "worker_thread_perfstat.csv"), out, list(out[0].keys()))

    with open(os.path.join(OUT, "worker_thread_perfstat.meta.json"), "w") as fh:
        json.dump(
            {
                "description": "worker 主线程的 perf stat：cycles/instructions/IPC 与 task-clock/上下文切换/缺页。",
                "tier": "historical-measured",
                "scope": "exact_worker_main_tid",
                "host": "a3-21",
                "config": "qwen35-08b / TP1 / graph(FULL_DECODE_ONLY) / MTP-off / chip15 / ISL128 OSL7168 C1",
                "runs": "runs/2026091{4}T09xxZ-qwen35-08b-tp1-mtp-off-long-decode-b1-perf-stat-r1",
                "excluded": "20260914T0946Z 一轮为 <not counted>（time_enabled=0），未进入本表",
                "confidence": "所有事件的 time_running_percent = 100.0（无 multiplexing）",
                "note": "per_decode_second 的分母是 decode_after_release_seconds；"
                "per_measured_output_token 的分母是 measured_output_tokens。",
            },
            fh,
            indent=1,
        )


# --------------------------------------------------------------------------
# 4) 热点函数
# --------------------------------------------------------------------------
def build_hotspots() -> None:
    p = os.path.join(RAW, "a321_perf_analysis/record-rank0.stacks.json")
    with open(p) as fh:
        d = json.load(fh)
    out = []
    for s in d["symbols"]:
        out.append(
            {
                "rank": s["rank"],
                "symbol": s["symbol"],
                "dso": s["dso"],
                "category": s["category"],
                "overhead_percent": s["overhead_percent"],
                "samples": s["samples"],
                "period": s["period"],
                "tags": ";".join(s.get("tags") or []),
            }
        )
    write_csv(os.path.join(OUT, "hotspots_perf_record_top.csv"), out, list(out[0].keys()))

    # 按 tag 归并（一个 tag 可覆盖多个符号）
    tot = sum(s["period"] for s in d["symbols"])
    full = d["totals"]["period"]
    agg: dict[str, dict] = {}
    for s in d["symbols"]:
        for t in (s.get("tags") or ["(untagged)"]):
            a = agg.setdefault(t, {"period": 0, "samples": 0, "symbols": 0})
            a["period"] += s["period"]
            a["samples"] += s["samples"]
            a["symbols"] += 1
    tag_rows = []
    for t, a in sorted(agg.items(), key=lambda kv: -kv[1]["period"]):
        tag_rows.append(
            {
                "tag": t,
                "period": a["period"],
                "share_of_top30_percent": round(100.0 * a["period"] / tot, 3),
                "share_of_full_profile_percent": round(100.0 * a["period"] / full, 3),
                "samples": a["samples"],
                "n_symbols": a["symbols"],
                "profile_total_period": full,
                "basis": (
                    "share_of_top30_percent 的分母是 top-30 之和；"
                    "share_of_full_profile_percent 的分母是整份 perf record 的 period（86023367614）。"
                    "注意 tags 只覆盖 top-30 符号，不是全部热点。"
                ),
            }
        )
    write_csv(
        os.path.join(OUT, "hotspots_perf_record_tags.csv"),
        tag_rows,
        list(tag_rows[0].keys()),
    )
    with open(os.path.join(OUT, "hotspots_perf_record_top.meta.json"), "w") as fh:
        json.dump(
            {
                "description": "worker 主线程 perf record 的自采样热点（period 加权，top 30；原始 top 表更长）。",
                "tier": "historical-measured",
                "scope": "exact_worker_main_tid",
                "source": "a3-21:HIST_PROJECT/perf-analysis/20260914T1014Z-perf-record-r1/record-rank0.stacks.json",
                "raw_perf_data": "runs/20260914T1014Z-...-perf-record-r1/perf/record-rank0.perf.data（32.2 MB，未上传）",
                "capture": d["capture"],
                "totals": d["totals"],
                "categories": d["categories"],
                "config": "qwen35-08b / TP1 / graph / MTP-off / chip15 / 4000 Hz / 31.9 s 解码窗口",
                "caveats": [
                    "symbols 列表被截断（symbols_truncated=true），只保留 top 30。",
                    "未解析（strip DSO）的长尾约占 period 的 6.8%（94.75% user + 5.25% kernel 中已归类的部分）。",
                ],
            },
            fh,
            indent=1,
        )

    # slice-insn-opt 的两份带 IPC 列的 perf report
    src_dir = os.path.join(MC, "vllm-slice-insn-opt/results")
    for arm in ["off", "on"]:
        fp = os.path.join(src_dir, f"perf-prof-08b-mtp-{arm}.flat.txt")
        if not os.path.exists(fp):
            continue
        rows = []
        for line in open(fp):
            m = re.match(
                r"^\s*([\d.]+)%\s+(\S+)\s+(\[.\]\s+.*?)\s{2,}(\S+)\s+(\S+)\s*$", line
            )
            if not m:
                continue
            rows.append(
                {
                    "overhead_percent": m.group(1),
                    "shared_object": m.group(2),
                    "symbol": m.group(3).strip(),
                    "ipc": m.group(4),
                    "ipc_coverage": m.group(5),
                }
            )
        if rows:
            write_csv(
                os.path.join(OUT, f"hotspots_perf_record_slice_insn_mtp_{arm}.csv"),
                rows,
                list(rows[0].keys()),
            )


# --------------------------------------------------------------------------
# 5) 吞吐 / tokens-per-step 基线
# --------------------------------------------------------------------------
def build_throughput() -> None:
    mc = read_csv(os.path.join(MC, "reports/mode-campaign-815/per-config.csv"))
    tps = read_csv(os.path.join(MC, "reports/mtp-attribution/tokens-per-step.csv"))
    tps_by_run: dict[str, list[dict]] = {}
    for r in tps:
        tps_by_run.setdefault(r["run_id"], []).append(r)

    out = []
    for r in mc:
        out.append(
            {
                "model": r["model"],
                "tp": r["tp"],
                "mode": r["mode"],
                "mtp": r["mtp"],
                "pool": r["pool"],
                "batch_tag": r["tag"],
                "n_windows": r["windows"],
                "ranks": r["ranks"],
                "throughput_tok_s": r["throughput_mean"],
                "throughput_spread_tok_s": r["throughput_spread"],
                "throughput_windows": r["throughput_windows"],
                "ipc_mean": r["ipc_mean"],
                "frontend_bound_mean_pct": r["frontend_mean"],
                "backend_bound_mean_pct": r["backend_bound_mean"],
                "ttft_ms": "",
                "tpot_ms": "",
                "source_file": "HIST_PROJECT/reports/mode-campaign-815/per-config.csv",
                "throughput_definition": "窗口内步数 × tokens/步 ÷ 窗口秒数（由 lite.log 派生，非 e2e 基准）",
            }
        )
    write_csv(os.path.join(OUT, "throughput_baseline.csv"), out, list(out[0].keys()))

    tps_out = []
    for r in tps:
        tps_out.append(dict(r))
    write_csv(
        os.path.join(OUT, "tokens_per_step.csv"),
        tps_out,
        list(tps_out[0].keys()),
    )

    with open(os.path.join(OUT, "throughput_baseline.meta.json"), "w") as fh:
        json.dump(
            {
                "description": "历史波次的吞吐与每步 token 数基线。",
                "tier": "historical-measured",
                "sources": [
                    "HIST_PROJECT/reports/mode-campaign-815/per-config.csv",
                    "HIST_PROJECT/reports/mtp-attribution/tokens-per-step.csv",
                    "HIST_PROJECT/reports/mtp-attribution/attribution.csv",
                ],
                "gaps": [
                    "没有 service_ttft / service_tpot 的真机实测（历史 run 里缺 benchmark 结果；"
                    "reports/dataset/long-table.csv 只有 7 行且来自 2026-09-14 的早期 run）。",
                    "吞吐口径是 lite.log 派生的单请求长 decode，不是吞吐基准。",
                ],
            },
            fh,
            indent=1,
        )

    # MTP 归因表（含 K、接受长度、四桶偏移分解）
    at = read_csv(os.path.join(MC, "reports/mtp-attribution/attribution.csv"))
    write_csv(
        os.path.join(OUT, "mtp_attribution.csv"),
        at,
        list(at[0].keys()),
    )


# --------------------------------------------------------------------------
# 6) 自旋五臂归因
# --------------------------------------------------------------------------
def build_spin() -> None:
    rows = read_csv(os.path.join(MC, "reports/spin-attribution/windows.csv"))
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["arm_key"], r["arm"]), []).append(r)
    out = []
    for (model, arm_key, arm), g in sorted(groups.items()):
        def mean(k):
            v = [fnum(x[k]) for x in g if fnum(x[k]) is not None]
            return round(statistics.fmean(v), 4) if v else ""

        out.append(
            {
                "model": model,
                "arm_key": arm_key,
                "arm": arm,
                "n_windows": len(g),
                "run_ids": ";".join(sorted({x["run_id"] for x in g})),
                "cycles_mean": round(statistics.fmean([fnum(x["cycles"]) for x in g]), 1),
                "instructions_mean": round(statistics.fmean([fnum(x["instructions"]) for x in g]), 1),
                "ipc_mean": mean("ipc"),
                "frontend_bound_mean_pct": mean("frontend_bound"),
                "bad_spec_mean_pct": mean("bad_spec"),
                "retiring_mean_pct": mean("retiring"),
                "backend_bound_mean_pct": mean("backend_bound"),
                "source_file": "HIST_PROJECT/reports/spin-attribution/windows.csv",
            }
        )
    write_csv(os.path.join(OUT, "spin_attribution_arms.csv"), out, list(out[0].keys()))
    with open(os.path.join(OUT, "spin_attribution.meta.json"), "w") as fh:
        json.dump(
            {
                "description": "graph+MTP-on 自旋归因五臂：只差「spec-decode event 记录在哪个流」。",
                "tier": "historical-measured",
                "arms": {
                    "A": "MTP-off 基线（无自旋参照）",
                    "B": "MTP-on 上游（自旋）",
                    "C1": "只改 valid_sampled_token_count_event",
                    "C2": "只改 num_accepted_tokens_event",
                    "C": "两处都改",
                },
                "source": "HIST_PROJECT/reports/spin-attribution/{README.md,windows.csv,figures/}",
                "conclusion": "把等待改回默认流只消掉约 50% 的额外周期；残余约 9 ms/步未归因。",
            },
            fh,
            indent=1,
        )


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    build_phase_configs()
    build_topdown()
    build_perfstat()
    build_hotspots()
    build_throughput()
    build_spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
