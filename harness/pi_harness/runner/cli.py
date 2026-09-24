"""`pi-replay`：无卡 prepare_input 负载生成 + 计时 + 参数化 sweep。

示例::

    python -m pi_harness.runner.cli --help
    python -m pi_harness.runner.cli --batch 32 --isl 4096 --osl 128 --steps 200 \
        --block-size 128 --out /work/data/harness
    python -m pi_harness.runner.cli --sweep isl --out /work/data/harness
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .config import RunnerConfig
from .replay import PrepareInputReplay


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pi-replay",
        description="无卡复刻 vLLM 0.26.0 + vllm-ascend 0.26.0rc1 的 prepare_input CPU 侧负载",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 模型/引擎 ---
    p.add_argument("--preset", default="realmachine",
                   choices=["realmachine", "stress", "modelmax"],
                   help="realmachine=真机 launcher 默认口径；stress=放大并发/预算；modelmax=模型自身 max_model_len")
    p.add_argument("--model-profile", default="qwen35-0.8b",
                   choices=["qwen35-0.8b", "qwen35-2b", "qwen3-1.7b", "synth"])
    p.add_argument("--hf-config-path", default=None,
                   help="覆盖 profile 使用的真机 config.json 路径（只读参考）")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--layers", type=int, default=None, dest="num_hidden_layers")
    p.add_argument("--heads", type=int, default=None, dest="num_attention_heads")
    p.add_argument("--kv-heads", type=int, default=None, dest="num_key_value_heads")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-reqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--spec-k", type=int, default=None, help="MTP 草稿数（>0 走 spec decode 路径）")
    p.add_argument("--async-scheduling", action="store_true")
    p.add_argument("--has-gdn", action="store_true",
                   help="模型含线性注意力层（Qwen3.5 为 True）→ 启用 gdn_query_start_loc 路径")
    # --- 负载 ---
    p.add_argument("--batch", type=int, default=None, help="并发请求数")
    p.add_argument("--isl", type=int, default=None, help="输入长度")
    p.add_argument("--osl", type=int, default=None, help="输出长度")
    p.add_argument("--chunk-size", type=int, default=None, help="chunked prefill 的 chunk")
    p.add_argument("--prefix-hit-ratio", type=float, default=0.0)
    p.add_argument("--arrival", default="simultaneous",
                   choices=["simultaneous", "stair", "poisson"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-values", action="store_true",
                   help="A/B：同形状但打乱 token id 取值")
    # --- 保真度注入 ---
    p.add_argument("--triton-launch-us", type=float, default=0.0,
                   help="每次 Ascend Triton launch 的 Python 侧开销（由 profiling 实测填入）")
    p.add_argument("--slot-mapping-mode", default="cpu_fallback",
                   metavar="{cpu_fallback,noop,inject:<us>}",
                   help="cpu_fallback=数值正确(有 numpy 兜底开销)；noop=最小 CPU(PMU/火焰图用)；"
                        "inject:<us>=noop+自旋注入 launch 开销")
    p.add_argument("--trace", default=None, help="用 record 出来的 JSONL trace 而不是合成负载")
    # --- 输出 ---
    p.add_argument("--out", default=None, help="输出目录（写 per_step/substep/summary）")
    p.add_argument("--tag", default="", help="输出文件名前缀")
    p.add_argument("--no-timing", action="store_true", help="关闭子步骤计时（用于纯 PMU 采集）")
    p.add_argument("--warmup", type=int, default=3)
    # --- sweep ---
    p.add_argument("--sweep", default=None,
                   choices=["isl", "batch", "osl", "block_size", "spec_k",
                            "prefix_hit_ratio", "chunk_size", "model"],
                   help="对某个维度做参数扫描，输出长表 CSV")
    p.add_argument("--sweep-values", default=None,
                   help="逗号分隔的扫描取值（默认用内置网格）")
    p.add_argument("--repeat", type=int, default=1, help="每个配置重复次数")
    return p


def cfg_from_args(a: argparse.Namespace, **overrides) -> RunnerConfig:
    cfg = RunnerConfig(preset=a.preset)
    cfg.apply_preset()
    # 显式传入的参数覆盖 preset
    # 只有**显式给出**的参数才覆盖 preset（None = 未给出）
    explicit = dict(
        model_profile=a.model_profile,
        hf_config_path=a.hf_config_path,
        hidden_size=a.hidden_size,
        num_hidden_layers=a.num_hidden_layers,
        num_attention_heads=a.num_attention_heads,
        num_key_value_heads=a.num_key_value_heads,
        max_model_len=a.max_model_len,
        max_num_reqs=a.max_num_reqs,
        max_num_batched_tokens=a.max_num_batched_tokens,
        batch=a.batch,
        isl=a.isl,
        osl=a.osl,
        prefix_hit_ratio=a.prefix_hit_ratio,
        steps=a.steps,
        seed=a.seed,
        triton_launch_us=a.triton_launch_us,
        slot_mapping_mode=a.slot_mapping_mode,
        trace_path=a.trace,
    )
    for k, v in explicit.items():
        if v is not None:
            setattr(cfg, k, v)
    if a.block_size is not None:
        cfg.block_size = a.block_size
    if a.dtype:
        cfg.dtype = a.dtype
    if a.spec_k is not None:
        cfg.num_spec_tokens = a.spec_k
    cfg.enable_chunked_prefill = not a.no_chunked_prefill
    cfg.enable_prefix_caching = not a.no_prefix_caching
    if a.async_scheduling:
        cfg.async_scheduling = True
    if a.has_gdn:
        cfg.has_gdn = True
    cfg.extra["chunk_size"] = a.chunk_size or cfg.max_num_batched_tokens
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_one(cfg: RunnerConfig, timing: bool, warmup: int = 3) -> dict:
    rl = PrepareInputReplay(cfg, timing=timing)
    rl.build()
    return rl.run(warmup=warmup), rl


SWEEP_GRIDS = {
    "isl": [128, 512, 1024, 2048, 4096, 8192],
    "batch": [1, 2, 4, 8, 16, 32, 64],
    "osl": [16, 64, 256, 1024],
    "block_size": [32, 64, 128, 256],
    "spec_k": [0, 1, 2, 3],
    "prefix_hit_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    "chunk_size": [128, 256, 512, 1024, 2048, 8192],
    "model": ["qwen35-0.8b", "qwen35-2b", "qwen3-1.7b"],
}


def run_sweep(a: argparse.Namespace) -> int:
    values = (
        [v.strip() for v in a.sweep_values.split(",")]
        if a.sweep_values
        else [str(v) for v in SWEEP_GRIDS[a.sweep]]
    )
    field = {"model": "model_profile"}.get(a.sweep, a.sweep)
    numeric = field not in ("model_profile",)
    rows = []
    outdir = Path(a.out or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    for raw in values:
        val = float(raw) if numeric and "." in raw else (int(raw) if numeric else raw)
        for rep in range(a.repeat):
            cfg = cfg_from_args(a, **{field: val})
            if field == "chunk_size":
                cfg.extra["chunk_size"] = int(val)
            try:
                summary, _ = run_one(cfg, timing=True, warmup=a.warmup)
            except Exception as exc:  # 记录失败而不是中断整个 sweep
                rows.append({"sweep": a.sweep, "value": raw, "repeat": rep,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"[sweep] {a.sweep}={raw} rep={rep} FAILED: {exc}", flush=True)
                continue
            pi = summary["prepare_inputs_us"]
            us = summary["update_states_us"]
            rows.append({
                "sweep": a.sweep,
                "value": raw,
                "repeat": rep,
                "steps_run": summary["steps_run"],
                "pi_p50_us": pi["p50"], "pi_p90_us": pi["p90"], "pi_p99_us": pi["p99"],
                "pi_mean_us": pi["mean"],
                "us_p50_us": us["p50"],
                "scope_p50_us": summary["scope_total_us"]["p50"],
                "model": cfg.model_profile, "batch": cfg.batch, "isl": cfg.isl,
                "osl": cfg.osl, "block_size": cfg.block_size, "spec_k": cfg.num_spec_tokens,
                "prefix_hit_ratio": cfg.prefix_hit_ratio,
                "chunk_size": cfg.extra.get("chunk_size"),
                "max_num_batched_tokens": cfg.max_num_batched_tokens,
                "max_num_reqs": cfg.max_num_reqs,
                "triton_launch_us": cfg.triton_launch_us,
            })
            print(f"[sweep] {a.sweep}={raw} rep={rep} pi_p50={pi['p50']:.1f}us "
                  f"us_p50={us['p50']:.1f}us steps={summary['steps_run']}", flush=True)

    keys = sorted({k for r in rows for k in r})
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{a.tag + '_' if a.tag else ''}sweep_{a.sweep}_{ts}.csv"
    path = outdir / name
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(json.dumps(r.get(k, "")) for k in keys) + "\n")
    print(f"[sweep] wrote {path} ({len(rows)} rows)")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if not (a.slot_mapping_mode in ("cpu_fallback", "noop")
            or a.slot_mapping_mode.startswith("inject:")):
        print(f"error: --slot-mapping-mode 取值非法: {a.slot_mapping_mode!r}"
              "（应为 cpu_fallback | noop | inject:<us>）", file=sys.stderr)
        return 2
    if a.sweep:
        return run_sweep(a)

    cfg = cfg_from_args(a)
    errs = cfg.validate()
    if errs:
        print("error: 配置不合法（真机引擎也会拒绝这些配置）：", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2
    summary, rl = run_one(cfg, timing=not a.no_timing, warmup=a.warmup)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if a.out:
        outdir = Path(a.out)
        ts = time.strftime("%Y%m%d-%H%M%S")
        prefix = f"{a.tag + '_' if a.tag else ''}"
        per_step, summ, subs = rl.dump(outdir)
        for p in (per_step, summ, subs):
            if p.exists():
                target = p.with_name(prefix + p.name)
                os.replace(p, target)
                print(f"[cli] wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
