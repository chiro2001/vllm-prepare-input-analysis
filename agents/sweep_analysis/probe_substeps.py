#!/usr/bin/env python3
"""补充计时探针：在**不修改 harness 代码**的前提下，运行时追加 wrap 候选调用点，
用来定位 `NPUModelRunner._prepare_inputs` 里没被 harness `SubStepTimer` 覆盖的时间。

为什么需要：harness 的 substep CSV 只覆盖少数方法；在 batch=16 decode 步里
`prepare_inputs_us ≈ 535 µs`，而 substep 之和只有 ~215 µs，剩下 ~320 µs 无归属。

两种模式：
  --mode wrap   : 额外 wrap 若干调用点（copy_to_gpu / numpy / torch 工厂函数），~0.5 µs/调用
  --mode ccall  : sys.setprofile 的 c_call/c_return 归因（高开销，用来看相对分布）

输出：/work/data/harness/sweep_probe_<mode>_<label>.csv / .json（本 agent 的写入范围）

用法（在无卡容器里）:
  cd /work/harness && PYTHONPATH=/work/harness:/work/agents/sweep_analysis \\
    PI_MODEL_TMP=/tmp/pi_models python /work/agents/sweep_analysis/probe_substeps.py \\
    --mode wrap --batch 16 --isl 1024 --steps 40 --out /work/data/harness --label batch16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/harness")

from pi_harness.runner.config import RunnerConfig           # noqa: E402
from pi_harness.runner.replay import PrepareInputReplay      # noqa: E402
from pi_harness.runner.timer import percentile                # noqa: E402


class Accum:
    """轻量累加器：label -> (总秒数, 调用次数)。"""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def add(self, label: str, dt: float) -> None:
        self.total[label] = self.total.get(label, 0.0) + dt
        self.calls[label] = self.calls.get(label, 0) + 1

    def reset(self) -> None:
        self.total.clear()
        self.calls.clear()


ACC = Accum()
_patched: list[tuple[object, str, object]] = []


def patch(target, name: str, label: str | None = None) -> bool:
    label = label or f"{getattr(target, '__name__', type(target).__name__)}.{name}"
    if not hasattr(target, name):
        return False
    original = getattr(target, name)
    if getattr(original, "_sweep_probe_wrapped", False):
        return True

    def wrapper(*args, __orig=original, __label=label, **kwargs):
        t0 = time.perf_counter()
        try:
            return __orig(*args, **kwargs)
        finally:
            ACC.add(__label, time.perf_counter() - t0)

    wrapper._sweep_probe_wrapped = True  # type: ignore[attr-defined]
    try:
        setattr(target, name, wrapper)
    except (TypeError, AttributeError):
        return False
    _patched.append((target, name, original))
    return True


def patch_module_callable(mod, attr: str, label: str | None = None) -> bool:
    """wrap 模块级可调用对象（如 np.repeat / torch.index_select）。"""
    if not hasattr(mod, attr):
        return False
    original = getattr(mod, attr)
    if getattr(original, "_sweep_probe_wrapped", False):
        return True
    label = label or f"{mod.__name__}.{attr}"

    def wrapper(*args, __orig=original, __label=label, **kwargs):
        t0 = time.perf_counter()
        try:
            return __orig(*args, **kwargs)
        finally:
            ACC.add(__label, time.perf_counter() - t0)

    wrapper._sweep_probe_wrapped = True  # type: ignore[attr-defined]
    setattr(mod, attr, wrapper)
    _patched.append((mod, attr, original))
    return True


CANDIDATE_MODULE_CALLS = [
    ("numpy", "repeat"), ("numpy", "add"), ("numpy", "subtract"), ("numpy", "cumsum"),
    ("numpy", "nonzero"), ("numpy", "array"), ("numpy", "ones"), ("numpy", "zeros"),
    ("numpy", "full"), ("numpy", "arange"),
    ("torch", "index_select"), ("torch", "from_numpy"), ("torch", "add"),
    ("torch", "arange"), ("torch", "zeros"), ("torch", "ones"),
]


def install_wraps() -> dict:
    import torch
    import vllm.v1.utils as v1utils

    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm_ascend.worker.block_table import BlockTable, MultiGroupBlockTable
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    report = {}
    report["CpuGpuBuffer.copy_to_gpu"] = patch(v1utils.CpuGpuBuffer, "copy_to_gpu")
    report["CpuGpuBuffer.copy_to_cpu"] = patch(v1utils.CpuGpuBuffer, "copy_to_cpu")
    report["Tensor.copy_"] = patch(torch.Tensor, "copy_", "Tensor.copy_")
    for attr in ("zero_", "fill_", "numpy"):
        report[f"Tensor.{attr}"] = patch(torch.Tensor, attr, f"Tensor.{attr}")
    for attr in ("fill", "reshape", "ravel"):
        report[f"ndarray.{attr}"] = patch(np.ndarray, attr, f"ndarray.{attr}")
    for modname, attr in CANDIDATE_MODULE_CALLS:
        mod = np if modname == "numpy" else torch
        report[f"{modname}.{attr}"] = patch_module_callable(mod, attr)
    for cls, name in (
        (NPUModelRunner, "_prepare_inputs"),
        (GPUModelRunner, "_get_cumsum_and_arange"),
        (GPUModelRunner, "_prepare_input_ids"),
        (GPUModelRunner, "_may_reorder_batch"),
        (InputBatch, "refresh_metadata"),
        (InputBatch, "condense"),
        (InputBatch, "_make_sampling_metadata"),
        (InputBatch, "update_req_spec_token_ids"),
        (MultiGroupBlockTable, "commit_block_table"),
        (MultiGroupBlockTable, "compute_slot_mapping"),
        (BlockTable, "commit_block_table"),
        (BlockTable, "compute_slot_mapping"),
        (BlockTable, "append_row"),
        (BlockTable, "add_row"),
        (BlockTable, "move_row"),
        (BlockTable, "clear_row"),
        # Ascend 内部辅助
        (NPUModelRunner, "_build_attn_state"),
        (NPUModelRunner, "_pad_query_start_loc_for_fia"),
        (NPUModelRunner, "_sanitize_placeholder_input_ids_for_forward"),
    ):
        report[f"{cls.__name__}.{name}"] = patch(cls, name)
    return report


def run_ccall(replay: PrepareInputReplay, steps: int) -> dict:
    """c_call/c_return 级归因（高开销；用于看相对分布）。"""
    py_total: dict[str, float] = {}
    c_total: dict[str, float] = {}
    c_calls: dict[str, int] = {}
    py_calls: dict[str, int] = {}
    stack: list[tuple[str, float]] = []

    def prof(frame, event, arg):
        if event == "c_call":
            stack.append((arg.__qualname__ if hasattr(arg, "__qualname__") else str(arg),
                          time.perf_counter()))
        elif event == "c_return" or event == "c_exception":
            if stack:
                name, t0 = stack.pop()
                dt = time.perf_counter() - t0
                c_total[name] = c_total.get(name, 0.0) + dt
                c_calls[name] = c_calls.get(name, 0) + 1
        elif event == "call":
            code = frame.f_code.co_name
            stack.append(("py:" + code, time.perf_counter()))
        elif event == "return":
            if stack and stack[-1][0].startswith("py:"):
                name, t0 = stack.pop()
                dt = time.perf_counter() - t0
                py_total[name] = py_total.get(name, 0.0) + dt
                py_calls[name] = py_calls.get(name, 0) + 1
        return prof

    for i in range(3):
        replay._one_step(-1 - i)
    ACC.reset()
    c_total.clear(); c_calls.clear(); py_total.clear(); py_calls.clear()
    sys.setprofile(prof)
    try:
        for i in range(steps):
            replay._one_step(i)
    finally:
        sys.setprofile(None)
    return {
        "c_calls_per_step": {k: v / steps for k, v in sorted(c_calls.items(), key=lambda kv: -kv[1])},
        "c_us_per_step": {k: v / steps * 1e6 for k, v in sorted(c_total.items(), key=lambda kv: -kv[1])},
        "py_calls_per_step": {k: v / steps for k, v in sorted(py_calls.items(), key=lambda kv: -kv[1])},
        "py_us_per_step": {k: v / steps * 1e6 for k, v in sorted(py_total.items(), key=lambda kv: -kv[1])},
        "steps": steps,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="wrap", choices=["wrap", "ccall"])
    ap.add_argument("--label", default="probe")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--max-num-reqs", type=int, default=64)
    ap.add_argument("--spec-k", type=int, default=0)
    ap.add_argument("--slot-mapping-mode", default="cpu_fallback")
    ap.add_argument("--out", default="/work/data/harness")
    args = ap.parse_args(argv)

    cfg = RunnerConfig(
        batch=args.batch, isl=args.isl, osl=args.osl, steps=args.steps,
        block_size=args.block_size, max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_reqs=args.max_num_reqs, num_spec_tokens=args.spec_k,
        has_gdn=True, slot_mapping_mode=args.slot_mapping_mode,
    )
    cfg.extra["chunk_size"] = args.chunk_size or args.max_num_batched_tokens

    replay = PrepareInputReplay(cfg, timing=True)
    replay.build()
    wrapped = install_wraps() if args.mode == "wrap" else {}

    if args.mode == "ccall":
        doc = run_ccall(replay, args.steps)
        doc["wrap_report"] = wrapped
    else:
        for i in range(3):
            replay._one_step(-1 - i)
        ACC.reset(); replay.timer.reset()
        rows = []
        call_rows = []
        for i in range(args.steps):
            snap = dict(ACC.total)
            snap_calls = dict(ACC.calls)
            row = replay._one_step(i)
            if row is None:
                break
            delta = {k: (v - snap.get(k, 0.0)) * 1e6 for k, v in ACC.total.items()}
            delta_calls = {k: v - snap_calls.get(k, 0) for k, v in ACC.calls.items()}
            rows.append({
                "step": i,
                **{f"acc::{k}": delta[k] for k in sorted(delta)},
                "harness_timer::total_from_wrapped": sum(replay.timer.inclusive.values()) * 1e6,
                "pi_raw_us": row["prepare_inputs_us"],
                "update_states_us": row["update_states_us"],
                "triton_cpu_us": row["triton_cpu_us"],
                "triton_launches": row["triton_launches"],
                "n_decode": row["phase"]["n_decode"],
                "n_prefill": row["phase"]["n_prefill"],
                "total_scheduled_tokens": row["total_scheduled_tokens"],
            })
            call_rows.append(delta_calls)
            ACC.reset(); replay.timer.reset()
        doc = {"rows": rows, "wrapped": wrapped, "steps": len(rows)}
        keys = sorted({k for r in rows for k in r if k.startswith("acc::")})
        doc["per_step_mean_us"] = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
        doc["calls_per_step"] = {
            k: float(np.mean([c.get(k, 0) for c in call_rows]))
            for k in sorted({k for c in call_rows for k in c})
        }
        doc["pi_raw_p50"] = percentile([r["pi_raw_us"] for r in rows], 50)
        doc["triton_p50"] = percentile([r["triton_cpu_us"] for r in rows], 50)
        doc["n_decode_steps"] = sum(1 for r in rows if r["n_decode"] > 0)
        doc["n_prefill_steps"] = sum(1 for r in rows if r["n_prefill"] > 0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"sweep_probe_{args.mode}_{args.label}_{ts}.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[probe] wrote {path}")

    if args.mode == "wrap":
        print(f"[probe] steps={doc['steps']} decode={doc['n_decode_steps']} "
              f"prefill={doc['n_prefill_steps']} pi_raw_p50={doc['pi_raw_p50']:.1f}us "
              f"triton_p50={doc['triton_p50']:.1f}us")
        for k, v in sorted(doc["per_step_mean_us"].items(), key=lambda kv: -kv[1])[:24]:
            print(f"   {v:10.2f} us/step  calls/step={doc['calls_per_step'].get(k, 0):8.1f}  {k}")
    else:
        print("[probe] top C calls (us/step):")
        for k, v in list(doc["c_us_per_step"].items())[:20]:
            print(f"   {v:10.2f}  calls/step={doc['c_calls_per_step'][k]:8.1f}  {k}")
        print("[probe] top python funcs (us/step):")
        for k, v in list(doc["py_us_per_step"].items())[:15]:
            print(f"   {v:10.2f}  calls/step={doc['py_calls_per_step'][k]:8.1f}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
