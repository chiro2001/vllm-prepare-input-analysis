#!/usr/bin/env python3
"""分析侧网格驱动：一次 python 进程跑完某个维度的全部取值，逐配置落**含 triton 列**的记录。

为什么要自己写而不是直接用 `--sweep`：
  1) `cli.py::run_sweep` 的长表**不含 `triton_cpu_us`**，而本次所有结论都必须用
     `pi_net = prepare_inputs_us - triton_cpu_us`；用它只能拿到 raw 口径。
  2) 它也不落 substep / per-step，无法做"哪个子步骤变大"的归因。
  3) `--sweep spec_k` 有 bug：`field = {"model": "model_profile"}.get(a.sweep, a.sweep)`
     缺 `"spec_k": "num_spec_tokens"`，于是 4 行其实是同配置重复（`spec_k` 列全 0）。
  4) spec 路径会走到 `model_runner_v1.py:1412` 的 `Tensor.pin_memory()`，
     shim 只拦了 `pin_memory=` 关键字、没拦方法形式 → 无卡容器里真实 aclInit 507008 崩。

按任务约束（不改 harness），本驱动：
  * 用正确的 `num_spec_tokens` 扫描；
  * 本地补 `Tensor.pin_memory()` no-op（与 shim 既有 pin_memory→False 策略等价）；
  * 输出 raw + net + triton + 结构元数据 + 每步子步骤均值。
这两处 gap 的复现与建议补丁写在 agents/sweep_analysis/REPORT.md。

用法（无卡容器内）:
  cd /work/harness && PYTHONPATH=/work/harness:/work/agents/sweep_analysis \\
    PI_MODEL_TMP=/tmp/pi_models python /work/agents/sweep_analysis/sweep_grid.py \\
    --dims isl,batch,chunk_size,block_size,spec_k,prefix_hit_ratio --steps 80 --repeat 1 \\
    --out /work/data/harness --tag sweep_grid
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

from pi_harness.runner.config import RunnerConfig             # noqa: E402
from pi_harness.runner.replay import PrepareInputReplay        # noqa: E402
from pi_harness.runner.timer import percentile                 # noqa: E402

GRIDS = {
    "isl": [128, 512, 1024, 2048, 4096, 8192],
    "batch": [1, 2, 4, 8, 16, 32, 64],
    "chunk_size": [128, 256, 512, 1024, 2048, 8192],
    "block_size": [32, 64, 128, 256],
    "spec_k": [0, 1, 2, 3],
    "prefix_hit_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
}

_PIN_DONE = False


def install_pin_memory_workaround() -> None:
    """`Tensor.pin_memory()` → no-op（等价于 shim 的 pin_memory=False 策略）。"""
    global _PIN_DONE
    if _PIN_DONE:
        return
    import torch

    def _pin_memory(self, device=None):
        return self

    torch.Tensor.pin_memory = _pin_memory
    _PIN_DONE = True


def substep_means(rl: PrepareInputReplay, n_steps: int) -> dict:
    """把 substep（累计值宽表）转成每步增量均值。"""
    if not rl.substep_rows:
        return {}
    labels = [k for k in rl.substep_rows[0] if k != "step"]
    out = {lab: 0.0 for lab in labels}
    prev = {lab: 0.0 for lab in labels}
    for row in rl.substep_rows:
        for lab in labels:
            cur = row.get(lab, 0.0)
            out[lab] += cur - prev[lab]
            prev[lab] = cur
    return {lab: v / max(n_steps, 1) * 1e6 for lab, v in out.items()}


def run_config(base_kwargs: dict, dim: str, value, steps: int, warmup: int):
    kwargs = dict(base_kwargs)
    chunk = None
    if dim == "spec_k":
        kwargs["num_spec_tokens"] = int(value)
    elif dim == "chunk_size":
        # RunnerConfig 没有 chunk_size 字段：chunk 走 cfg.extra（synth.SynthScheduler 读它）
        chunk = int(value)
    else:
        kwargs[dim] = value
    cfg = RunnerConfig(**kwargs)
    cfg.extra["chunk_size"] = chunk if chunk is not None else cfg.max_num_batched_tokens

    t0 = time.time()
    rl = PrepareInputReplay(cfg, timing=True)
    rl.build()
    summary = rl.run(steps=steps, warmup=warmup)
    wall = time.time() - t0

    rows = rl.rows
    dec = [r for r in rows if r["phase"]["n_decode"] > 0]
    pre = [r for r in rows if r["phase"]["n_prefill"] > 0 and r["phase"]["n_decode"] == 0]
    net = [r["prepare_inputs_us"] - r["triton_cpu_us"] for r in rows]
    tri = [r["triton_cpu_us"] for r in rows]
    meta = summary.get("meta", {})
    rec = {
        "dim": dim, "value": value, "wall_s": round(wall, 1),
        "steps_run": summary["steps_run"],
        "n_prefill_steps": len(pre), "n_decode_steps": len(dec),
        "pi_raw_p50_us": summary["prepare_inputs_us"]["p50"],
        "pi_raw_p90_us": summary["prepare_inputs_us"]["p90"],
        "pi_raw_mean_us": summary["prepare_inputs_us"]["mean"],
        "pi_net_p50_us": percentile(net, 50),
        "pi_net_p90_us": percentile(net, 90),
        "pi_net_mean_us": float(np.mean(net)) if net else float("nan"),
        "pi_net_p50_decode_us": percentile([r["prepare_inputs_us"] - r["triton_cpu_us"] for r in dec], 50),
        "pi_net_p50_prefill_us": percentile([r["prepare_inputs_us"] - r["triton_cpu_us"] for r in pre], 50),
        "triton_p50_us": percentile(tri, 50),
        "triton_p90_us": percentile(tri, 90),
        "triton_mean_us": float(np.mean(tri)) if tri else float("nan"),
        "triton_launches_set": ",".join(str(v) for v in sorted({r["triton_launches"] for r in rows})),
        "us_p50_us": summary["update_states_us"]["p50"],
        "us_mean_us": summary["update_states_us"]["mean"],
        "scope_raw_p50_us": summary["scope_total_us"]["p50"],
        "scope_net_p50_us": summary["scope_total_us"]["p50"] - percentile(tri, 50),
        "tokens_per_step_p50": percentile([r["total_scheduled_tokens"] for r in rows], 50),
        "tokens_per_step_decode_p50": percentile([r["total_scheduled_tokens"] for r in dec], 50),
        "tokens_per_step_prefill_p50": percentile([r["total_scheduled_tokens"] for r in pre], 50),
        "reqs_per_step_p50": percentile([r["num_reqs"] for r in rows], 50),
        "logits_len_p50": percentile([r["logits_indices_len"] for r in rows], 50),
        "has_spec_decode": bool(any(r["has_spec_decode"] for r in rows)),
        "max_num_blocks_per_req": meta.get("max_num_blocks_per_req"),
        "meta_block_size": meta.get("block_size"),
        "max_model_len": meta.get("max_model_len"),
        "attr_audit_empty": not bool(meta.get("attr_audit")),
        "readonly_attrs_skipped": len(meta.get("readonly_attrs_skipped") or []),
    }
    for lab, v in substep_means(rl, summary["steps_run"]).items():
        rec[f"sub::{lab.split('.')[-1]}"] = v
    return rec, rl


STEP_POOL_HEADER = [
    "dim", "value", "repeat", "step", "phase", "num_reqs", "total_scheduled_tokens",
    "n_prefill", "n_decode", "logits_indices_len", "has_spec_decode",
    "update_states_us", "prepare_inputs_us", "triton_cpu_us", "triton_launches",
    "pi_net_us", "scope_total_us", "block_size", "max_model_batched_tokens", "warmup",
]


def dump_steps(path: Path, dim: str, value, rep: int, rl: PrepareInputReplay,
               warmup: int = 3) -> None:
    """把所有配置的逐步数据池化到一个长表，用于 `pi_net vs tokens/step` 整体建模。"""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(STEP_POOL_HEADER)
        for r in rl.rows:
            ph = r["phase"]
            w.writerow([
                dim, value, rep, r["step"], "prefill" if ph["n_prefill"] else "decode",
                r["num_reqs"], r["total_scheduled_tokens"], ph["n_prefill"], ph["n_decode"],
                r["logits_indices_len"], int(bool(r["has_spec_decode"])),
                f"{r['update_states_us']:.3f}", f"{r['prepare_inputs_us']:.3f}",
                f"{r['triton_cpu_us']:.3f}", r["triton_launches"],
                f"{r['prepare_inputs_us'] - r['triton_cpu_us']:.3f}", f"{r['total_us']:.3f}",
                rl.cfg.block_size, rl.cfg.max_num_batched_tokens, warmup,
            ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="isl,batch,chunk_size,block_size,spec_k,prefix_hit_ratio")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--max-num-reqs", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--slot-mapping-mode", default="cpu_fallback")
    ap.add_argument("--out", default="/work/data/harness")
    ap.add_argument("--tag", default="sweep_grid")
    ap.add_argument("--only-values", default=None,
                    help="逗号分隔，只跑这些取值（用于补点）")
    ap.add_argument("--dump-steps", action="store_true",
                    help="把逐 step 数据追加到 sweep_steps_pool.csv")
    ap.add_argument("--pool-name", default="sweep_steps_pool.csv")
    args = ap.parse_args(argv)

    install_pin_memory_workaround()

    base_kwargs = dict(
        batch=args.batch, isl=args.isl, osl=args.osl, block_size=args.block_size,
        max_num_batched_tokens=args.max_num_batched_tokens, max_num_reqs=args.max_num_reqs,
        has_gdn=True, slot_mapping_mode=args.slot_mapping_mode,
    )
    if args.chunk_size is not None:
        base_kwargs["chunk_size"] = args.chunk_size

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    only = [v.strip() for v in args.only_values.split(",")] if args.only_values else None

    for dim in [d.strip() for d in args.dims.split(",") if d.strip()]:
        for value in GRIDS[dim]:
            if only is not None and str(value) not in only and f"{value:g}" not in only:
                continue
            for rep in range(args.repeat):
                try:
                    rec, rl = run_config(base_kwargs, dim, value, args.steps, args.warmup)
                except Exception as exc:
                    rec = {"dim": dim, "value": value, "error": f"{type(exc).__name__}: {exc}"}
                    print(f"[grid] {dim}={value} rep={rep} FAILED: {exc}", flush=True)
                    rl = None
                rec["warmup"] = args.warmup
                rec["repeat"] = rep
                rows.append(rec)
                if rl is not None and args.dump_steps:
                    dump_steps(outdir / args.pool_name, dim, value, rep, rl, args.warmup)
                if not rec.get("error"):
                    print(f"[grid] {dim}={value} rep={rep} "
                          f"pi_net_p50={rec['pi_net_p50_us']:.1f} (raw {rec['pi_raw_p50_us']:.1f}, "
                          f"triton {rec['triton_p50_us']:.1f}) us_p50={rec['us_p50_us']:.1f} "
                          f"pref={rec['n_prefill_steps']}/dec={rec['n_decode_steps']} "
                          f"blk/req={rec['max_num_blocks_per_req']} wall={rec['wall_s']}s", flush=True)

    keys = sorted({k for r in rows for k in r})
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = outdir / f"{args.tag}_{ts}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[grid] wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
