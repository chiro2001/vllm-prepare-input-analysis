"""sweep_analysis 数据层：解析 harness 输出的三种 CSV/JSON，统一成 per-step 表。

口径（见 agents/sweep_analysis/REPORT.md §1）：
    scope_total_us = update_states_us + prepare_inputs_us   （harness 默认口径）
    pi_net_us      = prepare_inputs_us - triton_cpu_us       （本次所有结论使用的口径）
`triton_cpu_us` 是 harness 独有开销（无卡环境里 slot-mapping kernel 的 numpy 等价实现 +
注入的 launch 开销），必须从上报数字里扣掉。所有函数只读解析，不依赖 pandas。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PER_STEP_HEADER = [
    "step", "phase", "update_states_us", "prepare_inputs_us", "deferred_fixup_us",
    "total_us", "num_reqs", "total_scheduled_tokens", "logits_indices_len",
    "has_spec_decode", "triton_cpu_us", "triton_launches",
]


# --------------------------------------------------------------------------- parsing
def _split_phase_row(line: str) -> list[str]:
    """per_step CSV 的 `phase` 列由 `json.dumps(dict)` 写成且未加引号，
    因此该行会多出逗号。按花括号配平把 phase 重新合成一个字段。"""
    parts = line.rstrip("\n").split(",")
    if len(parts) <= 1 or not parts[1].startswith("{"):
        return parts
    depth = 0
    buf: list[str] = []
    j = 1
    while j < len(parts):
        buf.append(parts[j])
        depth += parts[j].count("{") - parts[j].count("}")
        j += 1
        if depth <= 0:
            break
    return [parts[0], ",".join(buf)] + parts[j:]


def load_per_step(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split(",")
        assert header == PER_STEP_HEADER, f"unexpected per_step header: {header}"
        for line in fh:
            if not line.strip():
                continue
            fields = _split_phase_row(line)
            assert len(fields) == len(header), (len(fields), line[:120])
            row = {
                "step": int(fields[0]),
                "phase": json.loads(fields[1]),
                "update_states_us": float(fields[2]),
                "prepare_inputs_us": float(fields[3]),
                "deferred_fixup_us": float(fields[4]),
                "total_us": float(fields[5]),
                "num_reqs": int(fields[6]),
                "total_scheduled_tokens": int(fields[7]),
                "logits_indices_len": int(fields[8]),
                "has_spec_decode": fields[9] == "True",
                "triton_cpu_us": float(fields[10]),
                "triton_launches": int(fields[11]),
            }
            row["pi_net_us"] = row["prepare_inputs_us"] - row["triton_cpu_us"]
            row["scope_total_us"] = row["prepare_inputs_us"] + row["update_states_us"]
            row["scope_net_us"] = row["scope_total_us"] - row["triton_cpu_us"]
            row["n_prefill"] = int(row["phase"].get("n_prefill", 0))
            row["n_decode"] = int(row["phase"].get("n_decode", 0))
            rows.append(row)
    return rows


def load_substeps(path: str | Path) -> tuple[list[str], list[dict]]:
    """substep CSV 是**宽表 + 累计值**（timer.reset() 只在 run() 开头调用一次），
    因此这里转换成逐 step 的增量表。"""
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        labels = [c for c in rdr.fieldnames if c not in ("step", "calls")]
        raw = []
        for row in rdr:
            rec = {"step": int(row["step"])}
            for lab in labels:
                rec[lab] = float(row[lab]) if row.get(lab) else 0.0
            raw.append(rec)
    prev = {lab: 0.0 for lab in labels}
    out = []
    for row in raw:
        delta = {"step": row["step"]}
        for lab in labels:
            delta[lab] = row[lab] - prev[lab]
            prev[lab] = row[lab]
        out.append(delta)
    return labels, out


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def pct(values, q: float) -> float:
    values = [v for v in values if v == v]
    return float(np.percentile(values, q)) if len(values) else float("nan")


def fit_line(xs, ys) -> dict:
    """一阶拟合 T ≈ c + a·x。"""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2:
        return {"a": float("nan"), "c": float("nan"), "r2": float("nan"),
                "n_points": int(len(x))}
    a, c = np.polyfit(x, y, 1)
    pred = a * x + c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"a": float(a), "c": float(c), "r2": float(r2), "n_points": int(len(x))}


def constant_share(c: float, a: float, x: float) -> float:
    """该点上"常数项占总时间"的比例（1=纯底噪主导，0=纯斜率主导）。"""
    tot = c + a * x
    return float(c / tot) if tot else float("nan")
