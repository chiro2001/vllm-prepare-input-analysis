#!/usr/bin/env python3
"""三模式对照（cpu_fallback / noop / inject:<us>）——分析侧驱动，一次进程内**交错**跑。

为什么不用 CLI：`--slot-mapping-mode` 走 CLI 的路径在 02:55 的 harness 修订里
被 `cli.py:121`（`a.num_spec_tokens` 不存在，argparse 的 dest 是 `spec_k`）打断，
任何 CLI 调用都会 AttributeError。本脚本直接构造 RunnerConfig，绕开该回归。

交错（round-robin + 每轮轮换顺序）是为了消除"整段噪声漂移"：
实测同一配置在不同时间窗的 pi_raw p50 会整体漂移 1.2–1.9×（见 REPORT §2）。

用法（无卡容器内）:
  cd /work/harness && PYTHONPATH=/work/harness:/work/agents/sweep_analysis \\
    PI_MODEL_TMP=/tmp/pi_models python /work/agents/sweep_analysis/mode_compare.py \\
    --rounds 3 --steps 200 --batch 16 --isl 1024 --out /work/data/harness --tag sweep_modecmp
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

MODES = ["noop", "cpu_fallback", "inject:15"]


def run_mode(cfg: RunnerConfig, steps: int, warmup: int):
    rl = PrepareInputReplay(cfg, timing=True)
    rl.build()
    return rl.run(steps=steps, warmup=warmup), rl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--inject-us", type=float, default=15.0)
    ap.add_argument("--preset", default="realmachine",
                    choices=["realmachine", "stress", "modelmax"])
    ap.add_argument("--max-num-reqs", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--out", default="/work/data/harness")
    ap.add_argument("--tag", default="sweep_modecmp")
    args = ap.parse_args(argv)

    modes = [m if not m.startswith("inject") else f"inject:{args.inject_us:g}" for m in MODES]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ts = time.strftime("%Y%m%d-%H%M%S")

    for rnd in range(args.rounds):
        order = modes[rnd % len(modes):] + modes[: rnd % len(modes)]
        for mode in order:
            cfg = RunnerConfig(preset=args.preset)
            cfg.apply_preset()
            cfg.batch = args.batch
            cfg.isl = args.isl
            cfg.osl = args.osl
            cfg.steps = args.steps
            cfg.slot_mapping_mode = mode
            if args.max_num_reqs is not None:
                cfg.max_num_reqs = args.max_num_reqs
            if args.max_model_len is not None:
                cfg.max_model_len = args.max_model_len
            if args.max_num_batched_tokens is not None:
                cfg.max_num_batched_tokens = args.max_num_batched_tokens
            cfg.extra["chunk_size"] = cfg.max_num_batched_tokens
            cfg.extra["preset"] = args.preset
            t0 = time.time()
            try:
                summary, rl = run_mode(cfg, args.steps, args.warmup)
            except Exception as exc:
                rows.append({"round": rnd, "mode": mode, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[mode] r{rnd} {mode} FAILED: {exc}", flush=True)
                continue
            wall = time.time() - t0
            rws = rl.rows
            tri = [r["triton_cpu_us"] for r in rws]
            net = [r["prepare_inputs_us"] - r["triton_cpu_us"] for r in rws]
            steps = summary["steps_run"]
            rows.append({
                "round": rnd, "mode": mode, "preset": args.preset, "error": "",
                "max_model_len": cfg.max_model_len, "max_num_reqs": cfg.max_num_reqs,
                "max_num_batched_tokens": cfg.max_num_batched_tokens,
                "wall_s": round(wall, 1), "steps_run": steps,
                "pi_raw_p50_us": summary["prepare_inputs_us"]["p50"],
                "pi_raw_mean_us": summary["prepare_inputs_us"]["mean"],
                "pi_raw_p90_us": summary["prepare_inputs_us"]["p90"],
                "pi_net_p50_us": percentile(net, 50),
                "pi_net_mean_us": float(np.mean(net)),
                "triton_p50_us": percentile(tri, 50),
                "triton_mean_us": float(np.mean(tri)),
                "triton_launches_per_step": sorted({r["triton_launches"] for r in rws})[0]
                if len({r["triton_launches"] for r in rws}) == 1 else -1,
                "triton_launches_total": int(sum(r["triton_launches"] for r in rws)),
                "us_p50_us": summary["update_states_us"]["p50"],
                "scope_raw_p50_us": summary["scope_total_us"]["p50"],
                "scope_net_p50_us": summary["scope_total_us"]["p50"] - percentile(tri, 50),
                "tokens_per_step_p50": percentile([r["total_scheduled_tokens"] for r in rws], 50),
                "n_prefill_steps": sum(1 for r in rws if r["phase"]["n_prefill"] > 0),
            })
            print(f"[mode] r{rnd} {mode:14s} pi_raw_p50={rows[-1]['pi_raw_p50_us']:7.1f} "
                  f"triton_p50={rows[-1]['triton_p50_us']:7.1f} "
                  f"pi_net_p50={rows[-1]['pi_net_p50_us']:7.1f} us_p50={rows[-1]['us_p50_us']:6.1f} "
                  f"steps={steps} wall={wall:.0f}s", flush=True)
            prefix = f"{args.tag}_r{rnd}_{mode.replace(':', '')}"
            per_step, summ, subs = rl.dump(outdir)
            for p in (per_step, summ, subs):
                if p.exists():
                    os.replace(p, p.with_name(prefix + "_" + p.name))

    keys = sorted({k for r in rows for k in r})
    path = outdir / f"{args.tag}_{ts}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[mode] wrote {path} ({len(rows)} rows)")

    # 汇总：每模式跨轮中位数
    print("[mode] per-mode medians across rounds:")
    for mode in modes:
        sel = [r for r in rows if r.get("mode") == mode and not r.get("error")]
        if not sel:
            continue
        print(f"   {mode:14s} pi_raw_p50={np.median([r['pi_raw_p50_us'] for r in sel]):7.1f} "
              f"triton_p50={np.median([r['triton_p50_us'] for r in sel]):7.1f} "
              f"pi_net_p50={np.median([r['pi_net_p50_us'] for r in sel]):7.1f} "
              f"us_p50={np.median([r['us_p50_us'] for r in sel]):6.1f}  n={len(sel)}")
    json.dump(rows, open(outdir / f"{args.tag}_{ts}.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
