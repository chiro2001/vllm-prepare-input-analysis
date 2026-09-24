#!/usr/bin/env python3
"""preset 感知的参数扫描驱动（分析侧）。

为什么需要：`--sweep` 走 CLI，而 CLI 目前（02:55 修订）有 `a.num_spec_tokens` 回归会直接
AttributeError；而且 CLI 的长表**不含 `triton_cpu_us`**，无法用 `pi_net` 口径出结论。
本驱动直接构造 `RunnerConfig`，把 preset 的语义显式写进结果，并落
  * 每配置一行（含 raw / net / triton / 结构 meta）
  * 逐 step 池（用于 `pi_net vs tokens/step` / `vs num_reqs` 的整体建模）

**口径（关键）**：
  realmachine  MML=2048 / max_num_reqs=8 / max_num_batched_tokens=2048 /
               no-prefix-caching / async=on / has_gdn / slot_mapping=noop
  stress       MML=32768 / 64 reqs / 16384 tokens（只用于看斜率，不与真机对照）
  modelmax     MML=模型自身值（只用于研究 max_model_len 维度）

用法（无卡容器内）:
  cd /work/harness && PYTHONPATH=/work/harness:/work/agents/sweep_analysis \\
    PI_MODEL_TMP=/tmp/pi_models python /work/agents/sweep_analysis/preset_grid.py \\
    --preset realmachine --dims batch,isl --repeat 3 --steps 200 \\
    --out /work/data/harness --tag rm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PI_HARNESS_ROOT", "/work/harness"))

from pi_harness.runner.config import RunnerConfig, PRESETS   # noqa: E402
from pi_harness.runner.replay import PrepareInputReplay       # noqa: E402
from pi_harness.runner.timer import percentile                # noqa: E402

GRIDS = {
    "isl": [64, 128, 256, 512, 1024, 2048],
    "batch": [1, 2, 4, 8, 16, 32, 64],
    "chunk_size": [128, 256, 512, 1024, 2048, 8192],
    "block_size": [32, 64, 128, 256],
    "spec_k": [0, 1, 2, 3],
    "prefix_hit_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
}

_PIN_DONE = False


def install_pin_memory_workaround() -> None:
    """`Tensor.pin_memory()` → no-op（等价于 shim 既有的 pin_memory=False 策略）。

    spec 路径会走到 `model_runner_v1.py:1412` 的 `Tensor.pin_memory()`，shim 只拦了
    关键字形式，没拦方法形式 → 无卡容器 aclInit 507008 崩。此处是分析侧 workaround，
    harness 保持原样（建议补丁见 REPORT §3）。
    """
    global _PIN_DONE
    if _PIN_DONE:
        return
    import torch

    torch.Tensor.pin_memory = lambda self, device=None: self
    _PIN_DONE = True


def make_cfg(preset: str, dim: str, value, args) -> RunnerConfig:
    cfg = RunnerConfig(preset=preset)
    cfg.apply_preset()
    # 显式覆盖（preset 语义 + 本次扫描的自由度）
    if args.max_model_len is not None:
        cfg.max_model_len = args.max_model_len
    if args.max_num_reqs is not None:
        cfg.max_num_reqs = args.max_num_reqs
    if args.max_num_batched_tokens is not None:
        cfg.max_num_batched_tokens = args.max_num_batched_tokens
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.no_prefix_caching:
        cfg.enable_prefix_caching = False
    if args.slot_mapping_mode:
        cfg.slot_mapping_mode = args.slot_mapping_mode
    cfg.batch = args.batch
    cfg.isl = args.isl
    cfg.osl = args.osl
    cfg.steps = args.steps

    chunk = None
    if dim == "spec_k":
        cfg.num_spec_tokens = int(value)
    elif dim == "chunk_size":
        chunk = int(value)
    else:
        setattr(cfg, dim, value)
    cfg.extra["chunk_size"] = chunk if chunk is not None else cfg.max_num_batched_tokens
    return cfg


STEP_POOL_HEADER = [
    "preset", "dim", "value", "repeat", "step", "phase", "num_reqs",
    "total_scheduled_tokens", "n_prefill", "n_decode", "logits_indices_len",
    "has_spec_decode", "update_states_us", "prepare_inputs_us", "triton_cpu_us",
    "triton_launches", "pi_net_us", "scope_total_us", "block_size",
    "max_model_len", "max_num_reqs", "max_num_batched_tokens", "slot_mapping_mode",
    "use_async", "chunk_size",
]


def run_one(cfg: RunnerConfig, warmup: int, steady_seconds: float | None):
    rl = PrepareInputReplay(cfg, timing=True)
    rl.build()
    summary = rl.run(steps=cfg.steps, warmup=warmup, steady_seconds=steady_seconds)
    return summary, rl


def summarize(cfg, summary, rl) -> dict:
    rows = rl.rows
    dec = [r for r in rows if r["phase"]["n_decode"] > 0]
    pre = [r for r in rows if r["phase"]["n_prefill"] > 0 and r["phase"]["n_decode"] == 0]
    tri = [r["triton_cpu_us"] for r in rows]
    net = [r["prepare_inputs_us"] - r["triton_cpu_us"] for r in rows]
    meta = summary.get("meta", {})
    rec = {
        "preset": cfg.preset,
        "steps_run": summary["steps_run"],
        "n_prefill_steps": len(pre), "n_decode_steps": len(dec),
        "pi_raw_p50_us": summary["prepare_inputs_us"]["p50"],
        "pi_raw_p90_us": summary["prepare_inputs_us"]["p90"],
        "pi_raw_mean_us": summary["prepare_inputs_us"]["mean"],
        "pi_net_p50_us": percentile(net, 50),
        "pi_net_p90_us": percentile(net, 90),
        "pi_net_mean_us": float(np.mean(net)) if net else float("nan"),
        "pi_net_p50_decode_us": percentile(
            [r["prepare_inputs_us"] - r["triton_cpu_us"] for r in dec], 50),
        "pi_net_p50_prefill_us": percentile(
            [r["prepare_inputs_us"] - r["triton_cpu_us"] for r in pre], 50),
        "triton_p50_us": percentile(tri, 50),
        "triton_mean_us": float(np.mean(tri)) if tri else float("nan"),
        "triton_launches_set": ",".join(str(v) for v in sorted({r["triton_launches"] for r in rows})),
        "us_p50_us": summary["update_states_us"]["p50"],
        "us_mean_us": summary["update_states_us"]["mean"],
        "scope_raw_p50_us": summary["scope_total_us"]["p50"],
        "scope_net_p50_us": summary["scope_total_us"]["p50"] - percentile(tri, 50),
        "tokens_per_step_p50": percentile([r["total_scheduled_tokens"] for r in rows], 50),
        "tokens_per_step_decode_p50": percentile([r["total_scheduled_tokens"] for r in dec], 50),
        "reqs_per_step_p50": percentile([r["num_reqs"] for r in rows], 50),
        "logits_len_p50": percentile([r["logits_indices_len"] for r in rows], 50),
        "has_spec_decode": bool(any(r["has_spec_decode"] for r in rows)),
        "max_model_len": meta.get("max_model_len"),
        "max_num_blocks_per_req": meta.get("max_num_blocks_per_req"),
        "max_num_reqs_cfg": cfg.max_num_reqs,
        "max_num_batched_tokens_cfg": cfg.max_num_batched_tokens,
        "block_size_cfg": cfg.block_size,
        "prefix_caching": cfg.enable_prefix_caching,
        "async_scheduling": cfg.async_scheduling,
        "slot_mapping_mode": cfg.slot_mapping_mode,
        "token_ids_cpu_MB": (cfg.max_num_reqs * (meta.get("max_model_len") or 0) * 4 / 1e6),
    }
    if rows:
        rec["wall_s_per_step_us"] = float(np.mean([r["total_us"] for r in rows]))
    # substep 均值（每步）。注意 harness 的 substep CSV 是"自 reset 起的累计值"，
    # 但 rl.substep_rows 已是replay 内部按步增量（见 replay.py::_snapshot_delta）。
    if rl.substep_rows:
        labels = [k for k in rl.substep_rows[0] if k != "step"]
        ns = len(rl.substep_rows)
        rec["substeps_us_per_step"] = {
            lab: float(np.mean([r.get(lab, 0.0) for r in rl.substep_rows])) * 1e6 * ns / max(len(rows), 1)
            for lab in labels
        }
    return rec


def dump_steps(path: Path, cfg, dim: str, value, rep: int, rl, warmup: int) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(STEP_POOL_HEADER)
        for r in rl.rows:
            ph = r["phase"]
            w.writerow([
                cfg.preset, dim, value, rep, r["step"],
                "prefill" if ph["n_prefill"] else "decode", r["num_reqs"],
                r["total_scheduled_tokens"], ph["n_prefill"], ph["n_decode"],
                r["logits_indices_len"], int(bool(r["has_spec_decode"])),
                f"{r['update_states_us']:.3f}", f"{r['prepare_inputs_us']:.3f}",
                f"{r['triton_cpu_us']:.3f}", r["triton_launches"],
                f"{r['prepare_inputs_us'] - r['triton_cpu_us']:.3f}",
                f"{r['total_us']:.3f}", cfg.block_size, cfg.max_model_len,
                cfg.max_num_reqs, cfg.max_num_batched_tokens, cfg.slot_mapping_mode,
                int(cfg.async_scheduling), cfg.extra.get("chunk_size"),
            ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="realmachine", choices=list(PRESETS))
    ap.add_argument("--dims", default="batch,isl")
    ap.add_argument("--values", default=None, help="覆盖网格（逗号分隔，对所有 dim 生效）")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--steady-seconds", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--isl", type=int, default=None)
    ap.add_argument("--osl", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-reqs", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--slot-mapping-mode", default=None)
    ap.add_argument("--out", default="/work/data/harness")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--dump-steps", action="store_true")
    args = ap.parse_args(argv)

    install_pin_memory_workaround()

    base_preset = PRESETS[args.preset]
    args.batch = args.batch if args.batch is not None else base_preset.get("batch", 1)
    args.isl = args.isl if args.isl is not None else base_preset.get("isl", 128)
    args.osl = args.osl if args.osl is not None else base_preset.get("osl", 64)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for dim in [d.strip() for d in args.dims.split(",") if d.strip()]:
        values = ([float(v) if "." in v else int(v) for v in args.values.split(",")]
                  if args.values else GRIDS[dim])
        for value in values:
            for rep in range(args.repeat):
                cfg = make_cfg(args.preset, dim, value, args)
                rec = {"dim": dim, "value": value, "repeat": rep,
                       "preset": args.preset, "batch": cfg.batch, "isl": cfg.isl,
                       "osl": cfg.osl, "steps_req": args.steps}
                t0 = time.time()
                try:
                    summary, rl = run_one(cfg, args.warmup, args.steady_seconds)
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[pgrid] {args.preset} {dim}={value} rep={rep} FAILED: {exc}", flush=True)
                    if os.environ.get("SWEEP_TRACEBACK"):
                        import traceback
                        traceback.print_exc()
                    rows.append(rec)
                    continue
                rec.update(summarize(cfg, summary, rl))
                rec["wall_s"] = round(time.time() - t0, 1)
                rows.append(rec)
                if args.dump_steps:
                    dump_steps(outdir / f"sweep_steps_{args.preset}_{args.tag}.csv",
                               cfg, dim, value, rep, rl, args.warmup)
                print(f"[pgrid] {args.preset} {dim}={value} rep={rep} "
                      f"pi_net_p50={rec['pi_net_p50_us']:7.1f} (raw {rec['pi_raw_p50_us']:7.1f}, "
                      f"triton {rec['triton_p50_us']:6.1f}) us_p50={rec['us_p50_us']:6.1f} "
                      f"scope_net={rec['scope_net_p50_us']:7.1f} steps={rec['steps_run']} "
                      f"pref={rec['n_prefill_steps']}/dec={rec['n_decode_steps']} "
                      f"MML={rec['max_model_len']} blk/req={rec['max_num_blocks_per_req']} "
                      f"tok/step={rec['tokens_per_step_p50']:.0f} wall={rec['wall_s']}s", flush=True)

    keys = sorted({k for r in rows for k in r})
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = outdir / f"sweep_{args.tag}_{args.preset}_{ts}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v)
                        for k, v in r.items()})
    print(f"[pgrid] wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
