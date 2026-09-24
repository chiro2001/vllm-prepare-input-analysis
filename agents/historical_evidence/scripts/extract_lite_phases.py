#!/usr/bin/env python3
"""把 HIST_PROJECT/reports/lite-traces 的 52 份 Chrome trace 抽成「每步 phase 表」。

来源（只读）：
  /home/chiro/projects/vllm/HIST_PROJECT/reports/lite-traces/<run>/lite.trace.json.gz
  /home/chiro/projects/vllm/HIST_PROJECT/reports/lite-traces/<run>/trace.meta.json

口径：
  * 只取 worker main 泳道（trace.meta.json 的 ``worker_main``，形如 "131:131"）；
  * 窗口 = 同一泳道上 ``cat=="phase"`` 的 ``phase: decode`` 事件区间（每份 trace 5 个，
    各 ~25.1 s，窗口之间隔 ~15 s 未采样）；
  * 事件按 **起始 ts** 归入窗口；
  * 每窗口一步的 `step_period_us = 窗口墙钟 / 窗口内步数`，步数取
    ``Step:Schedule`` 的实例数（wait scope 版本文档确认 ``Step:Schedule`` 内无等待）；
  * 输出**不扣减设备等待**：w1/m1a/e1a 三波用的是无 wait scope 的镜像，
    ``prepare input`` / ``forward`` 里可能包含 device 等待（见 MTP_PREPARE_INPUT_SPIN.md）。

用法：
  python3 extract_lite_phases.py --traces-root <root> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import statistics
import sys

RUN_RE = re.compile(
    r"^(?P<prefix>[a-z0-9_]+?)_"
    r"(?P<model>qwen[0-9a-z\-\.]*?)"
    r"_tp(?P<tp>[0-9]+)"
    r"_(?P<mode>graph-draft-eager|graph|eager)"
    r"_mtp-(?P<mtp>on|off)"
    # batch tag 可能是 `m4`、`v4b`、`w1`，也可能是 `m4_qwen35-08b_graph_off` 这种
    # 「tag + 被复用的 run 名后缀」；两种都用非贪婪匹配到时间戳之前。
    r"_(?P<batch_tag>[A-Za-z0-9_\-]+?)"
    r"_(?P<stamp>[0-9]{8}T[0-9]{6}Z)"
    r"_(?P<rank>[0-9]+)$"
)

# 每步的兄弟作用域（顺序执行、互不嵌套，来自 vllm-ascend/worker/model_runner_v1.py
# 与 gpu_model_runner.py 的插桩名）。
PER_STEP_SCOPES = [
    "Step:Schedule",
    "schedule: allocate_slots",
    "schedule: get_num_common_prefix_blocks",
    "schedule: make_cached_request_data",
    "schedule: update_after_schedule",
    "prepare input",
    "forward",
    "post process",
    "sample_token",
    "draft_token",
    "async_state_update",
    "Step:Output",
    "output: process_outputs",
]

# 只有 MTP-on 才出现的 scope
MTP_SCOPES = [
    "model: mtp: propose",
    "model: mtp: first_pass",
    "model: mtp: draft_forward",
]

# 步数锚点优先级：`Step:Schedule` 只在带 schedule 插桩的波次（w*/m4/v4b）里有；
# `prepare input` / `forward` 每步各一次，是 e1x/m1x 波次唯一可用的锚。
STEP_ANCHORS = ["Step:Schedule", "prepare input", "forward"]


def parse_run_name(name: str) -> dict:
    m = RUN_RE.match(name)
    if not m:
        return {
            "parse_ok": False,
            "model": None,
            "tp": None,
            "mode": None,
            "mtp": None,
            "batch_tag": None,
            "stamp": None,
            "rank": None,
        }
    d = m.groupdict()
    d["parse_ok"] = True
    d["tp"] = int(d["tp"])
    return d


def pct(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数（numpy 默认口径），输入必须已升序。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


RANK_IN_LABEL = re.compile(r"rank(\d+)")


def lane_rank(lane_label: str, fallback: int) -> int:
    m = RANK_IN_LABEL.search(lane_label or "")
    return int(m.group(1)) if m else fallback


def process_trace(trace_dir: str, want_scopes: set[str]) -> tuple[list[dict], dict]:
    meta_path = os.path.join(trace_dir, "trace.meta.json")
    trace_path = os.path.join(trace_dir, "lite.trace.json.gz")
    run = os.path.basename(trace_dir.rstrip("/"))
    cfg = parse_run_name(run)

    with open(meta_path) as fh:
        meta = json.load(fh)

    worker_main = meta.get("worker_main") or []
    lanes: list[tuple[int, int]] = []
    for lane in worker_main:
        pid_s, tid_s = lane.split(":")
        lanes.append((int(pid_s), int(tid_s)))

    with gzip.open(trace_path) as fh:
        trace = json.load(fh)
    events = trace["traceEvents"]

    # phase 泳道名形如 "<pid>:<tid|1073741xxx>"，用 lane_names 反查所属 rank。
    lane_names = meta.get("lane_names") or {}
    # 每个 worker main lane 一个独立的 (rank, windows, events) 桶。
    per_lane: dict[tuple[int, int], dict] = {
        lane: {"rank": lane_rank(lane_names.get(f"{lane[0]}:{lane[1]}", ""), i),
               "windows": [], "scopes": {}}
        for i, lane in enumerate(lanes)
    }
    # phase 泳道 -> 它属于哪个 worker main：按 pid 匹配（同进程内 phase 泳道
    # 的 label 里同样带 rank）。
    pid_to_lane: dict[int, tuple[int, int]] = {}
    for lane in lanes:
        pid_to_lane.setdefault(lane[0], lane)
    for key, label in lane_names.items():
        pid_s, tid_s = key.split(":")
        if "phase" not in label:
            continue
        lane = pid_to_lane.get(int(pid_s))
        if lane is None:
            continue
        per_lane[lane]["phase_lane"] = (int(pid_s), int(tid_s))

    for ev in events:
        if ev.get("ph") != "X":
            continue
        key = (ev.get("pid"), ev.get("tid"))
        lane = per_lane.get(key)
        if lane is not None:
            if ev.get("cat") == "phase":
                lane["windows"].append(
                    (float(ev["ts"]), float(ev["ts"]) + float(ev["dur"]))
                )
            else:
                lane["scopes"].setdefault(ev.get("name", ""), []).append(
                    (float(ev["ts"]), float(ev["dur"]))
                )
            continue
        # phase 泳道单独一条（tid 带 1073741xxx 位）
        for lane2 in per_lane.values():
            if lane2.get("phase_lane") == key:
                lane2["windows"].append(
                    (float(ev["ts"]), float(ev["ts"]) + float(ev["dur"]))
                )
                break

    rows: list[dict] = []
    lane_summary: list[dict] = []
    for lane_key, lane in per_lane.items():
        rank = lane["rank"]
        windows = sorted(lane["windows"])
        scopes: dict[str, list[tuple[float, float]]] = lane["scopes"]
        if not windows:
            lane_summary.append(
                {"lane": f"{lane_key[0]}:{lane_key[1]}", "rank": rank, "windows": 0}
            )
            continue
        # 按起始 ts 归窗
        per_win: dict[int, dict[str, list[float]]] = {i: {} for i in range(len(windows))}
        unattributed: dict[str, int] = {}
        for name, insts in scopes.items():
            if name not in want_scopes:
                continue
            for ts, dur in insts:
                idx = None
                for i, (w0, w1) in enumerate(windows):
                    if w0 <= ts < w1:
                        idx = i
                        break
                if idx is None:
                    unattributed[name] = unattributed.get(name, 0) + 1
                    continue
                per_win[idx].setdefault(name, []).append(dur)
        lane_summary.append(
            {
                "lane": f"{lane_key[0]}:{lane_key[1]}",
                "rank": rank,
                "windows": len(windows),
                "unattributed": unattributed,
            }
        )
        rows.extend(
            build_rows(cfg, run, rank, windows, per_win)
        )
    info = {
        "run": run,
        "trace_dir": trace_dir,
        "worker_main_lanes": worker_main,
        "lanes": lane_summary,
        "trace_sha256": meta.get("trace_sha256"),
        "source_logs": meta.get("source_logs"),
        "window_seconds": meta.get("window_seconds"),
        "meta_warnings": meta.get("warnings"),
        "meta_clean": meta.get("clean"),
    }
    return rows, info


def build_rows(
    cfg: dict,
    run: str,
    rank: int,
    windows: list[tuple[float, float]],
    per_win: dict[int, dict[str, list[float]]],
) -> list[dict]:
    rows: list[dict] = []
    for i, (w0, w1) in enumerate(windows):
        scope_vals = per_win[i]
        anchor_counts = {
            name: len(scope_vals.get(name, [])) for name in STEP_ANCHORS
        }
        step_anchor = ""
        n_steps = 0
        for name in STEP_ANCHORS:
            if anchor_counts[name]:
                step_anchor = name
                n_steps = anchor_counts[name]
                break
        win_dur = w1 - w0
        step_period = win_dur / n_steps if n_steps else float("nan")
        base = {
            "run": run,
            "model": cfg["model"],
            "tp": cfg["tp"],
            "mode": cfg["mode"],
            "mtp": cfg["mtp"],
            "batch_tag": cfg["batch_tag"],
            "stamp": cfg["stamp"],
            "rank": rank,
            "rank_from_name": cfg["rank"],
            "parse_ok": cfg["parse_ok"],
            "window_index": i,
            "window_start_us": round(w0, 1),
            "window_dur_us": round(win_dur, 1),
            "step_anchor": step_anchor,
            "n_steps": n_steps,
            "n_steps_schedule": anchor_counts["Step:Schedule"],
            "n_steps_prepare_input": anchor_counts["prepare input"],
            "n_steps_forward": anchor_counts["forward"],
            "step_period_us": round(step_period, 3) if n_steps else "",
        }
        for name in sorted(scope_vals):
            vals = scope_vals.get(name, [])
            if not vals:
                continue
            vals_sorted = sorted(vals)
            row = dict(base)
            row.update(
                {
                    "scope": name,
                    "count": len(vals),
                    "per_step": round(len(vals) / n_steps, 4) if n_steps else "",
                    "mean_us": round(statistics.fmean(vals), 3),
                    "sum_us": round(sum(vals), 1),
                    "p50_us": round(pct(vals_sorted, 0.50), 3),
                    "p90_us": round(pct(vals_sorted, 0.90), 3),
                    "p99_us": round(pct(vals_sorted, 0.99), 3),
                    "min_us": round(vals_sorted[0], 3),
                    "max_us": round(vals_sorted[-1], 3),
                    "share_of_step_pct": round(
                        100.0 * statistics.fmean(vals) / step_period, 3
                    )
                    if n_steps
                    else "",
                }
            )
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-traces", type=int, default=0)
    args = ap.parse_args()

    want = set(PER_STEP_SCOPES) | set(MTP_SCOPES)
    dirs = sorted(
        os.path.join(args.traces_root, d)
        for d in os.listdir(args.traces_root)
        if os.path.isdir(os.path.join(args.traces_root, d))
        and os.path.exists(os.path.join(args.traces_root, d, "lite.trace.json.gz"))
    )
    if args.max_traces:
        dirs = dirs[: args.max_traces]

    all_rows: list[dict] = []
    infos: list[dict] = []
    failures: list[dict] = []
    fieldnames: list[str] = []
    for d in dirs:
        try:
            rows, info = process_trace(d, want)
        except Exception as exc:  # noqa: BLE001 - 采集脚本：失败要留证而不是中断
            failures.append({"trace_dir": d, "error": repr(exc)})
            print(f"[FAIL] {d}: {exc!r}", file=sys.stderr)
            continue
        all_rows.extend(rows)
        infos.append(info)
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        n_win = sum(l.get("windows", 0) for l in info["lanes"])
        print(
            f"[OK] {info['run']}: {len(info['lanes'])} lane(s), "
            f"{n_win} windows, {len(rows)} scope rows"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "phase_per_step_windows.csv")
    ordered = [
        "run",
        "model",
        "tp",
        "mode",
        "mtp",
        "batch_tag",
        "rank",
        "rank_from_name",
        "window_index",
        "window_dur_us",
        "step_anchor",
        "n_steps",
        "n_steps_schedule",
        "n_steps_prepare_input",
        "n_steps_forward",
        "step_period_us",
        "scope",
        "count",
        "per_step",
        "mean_us",
        "p50_us",
        "p90_us",
        "p99_us",
        "min_us",
        "max_us",
        "sum_us",
        "share_of_step_pct",
        "window_start_us",
        "stamp",
        "parse_ok",
    ]
    extra = [f for f in fieldnames if f not in ordered]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ordered + extra, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    meta_path = os.path.join(args.out_dir, "phase_per_step_windows.meta.json")
    with open(meta_path, "w") as fh:
        json.dump(
            {
                "description": (
                    "每个 lite trace 的每个 phase 窗口 × 每个 scope 的耗时统计；"
                    "单位微秒（除 share_of_step_pct）。"
                ),
                "source": f"{args.traces_root}/<run>/lite.trace.json.gz",
                "tier": "historical-measured",
                "caveats": [
                    "w1/m1a/e1a 三波的镜像没有 wait scope：prepare input / forward 内含 device 等待，"
                    "不能直接当作 CPU 侧耗时；MTP-on 的 graph 臂尤其明显。",
                    "窗口之间不连续（每份 trace 5 个 ~25 s 窗口，间隔 ~15 s 未采样）。",
                    "步数由 Step:Schedule 实例数确定。",
                ],
                "traces": infos,
                "failures": failures,
            },
            fh,
            indent=1,
        )
    print(f"\nwrote {csv_path} ({len(all_rows)} rows)")
    print(f"wrote {meta_path}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
