"""`python -m pi_harness.runner.replay`：供 prof_harness.sh 直接喂 trace 或合成负载。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import RunnerConfig
from .replay import PrepareInputReplay


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pi_harness.runner.replay")
    p.add_argument("--trace", default=None, help="JSONL trace（P2 record 产物）")
    p.add_argument("--model-profile", default="qwen35-0.8b")
    p.add_argument("--preset", default="realmachine",
                   choices=["realmachine", "stress", "modelmax"],
                   help="realmachine=真机 launcher 默认口径（做一致性对照用）；"
                        "stress=放大并发与 token 预算；modelmax=模型自身 max_model_len")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="不填则用 preset 的值（realmachine=2048）")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--isl", type=int, default=1024)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-num-reqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--spec-k", type=int, default=0)
    p.add_argument("--prefix-hit-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--has-gdn", action="store_true")
    p.add_argument("--no-gdn", action="store_true")
    p.add_argument("--async-scheduling", action="store_true")
    p.add_argument("--no-async-scheduling", action="store_true")
    p.add_argument("--slot-mapping-mode", default="noop",
                   help="默认 noop：为 PMU/火焰图准备的最小 CPU 口径")
    p.add_argument("--triton-launch-us", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steady-seconds", type=float, default=None,
                   help="按墙钟跑够 N 秒（PMU 采集用；负载会循环重灌以保持 CPU 满载）")
    p.add_argument("--out", default=None)
    p.add_argument("--no-timing", action="store_true")
    a = p.parse_args(argv)

    cfg = RunnerConfig(preset=a.preset)
    cfg.apply_preset()
    cfg.model_profile = a.model_profile
    cfg.batch = a.batch
    cfg.isl = a.isl
    cfg.osl = a.osl
    cfg.steps = a.steps
    cfg.block_size = a.block_size
    if a.max_model_len is not None:
        cfg.max_model_len = a.max_model_len
    if a.max_num_reqs is not None:
        cfg.max_num_reqs = a.max_num_reqs
    if a.max_num_batched_tokens is not None:
        cfg.max_num_batched_tokens = a.max_num_batched_tokens
    cfg.num_spec_tokens = a.spec_k
    cfg.prefix_hit_ratio = a.prefix_hit_ratio
    cfg.seed = a.seed
    if a.has_gdn:
        cfg.has_gdn = True
    if a.no_gdn:
        cfg.has_gdn = False
    if a.async_scheduling:
        cfg.async_scheduling = True
    if a.no_async_scheduling:
        cfg.async_scheduling = False
    cfg.slot_mapping_mode = a.slot_mapping_mode
    cfg.triton_launch_us = a.triton_launch_us
    cfg.trace_path = a.trace
    cfg.extra["chunk_size"] = a.chunk_size or a.max_num_batched_tokens

    rl = PrepareInputReplay(cfg, timing=not a.no_timing)
    rl.build()
    s = rl.run(warmup=a.warmup, steady_seconds=a.steady_seconds)
    if a.out:
        rl.dump(a.out)
    print(json.dumps({k: s[k] for k in ("steps_run", "prepare_inputs_us", "update_states_us")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
