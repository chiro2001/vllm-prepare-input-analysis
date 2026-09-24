#!/usr/bin/env python3
"""C（动态部分）. 给 `torch.npu.*` / `torch_npu.*` 套一层**计数代理**，然后在分阶段
（import vllm → import vllm_ascend.ops → import vllm_ascend.worker.model_runner_v1 →
构造 CpuGpuBuffer / BlockTable 级别的对象）记录每个 API 的真实调用次数。

与 shim 的关系：计数层是**独立**的（不依赖 shim 的计数器），所以即使 shim 把 API 变成
no-op，我们仍能看到真实调用次数。shim 的 `report()` 也会一并落盘做交叉验证。

用法::

    python harness/probes/probe_dynamic_hits.py                # shim + device 重映射补丁
    python harness/probes/probe_dynamic_hits.py --no-patch     # 只装 shim（多数用例会崩）
    python harness/probes/probe_dynamic_hits.py --cosmetic-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

REPO = Path(os.environ.get("PI_REPO_ROOT", "/work"))
OUT = Path(os.environ.get("PI_DEVD_DEPS_OUT", REPO / "data" / "harness"))

PHASE = {"name": "startup"}
COUNTS: dict[str, Counter] = {}
ERRORS: list[dict] = []


def phase(name: str):
    PHASE["name"] = name
    COUNTS.setdefault(name, Counter())


def _record(api: str) -> None:
    COUNTS.setdefault(PHASE["name"], Counter())[api] += 1


def _stub_dist_groups() -> None:
    """`BlockTable.__init__` 读 `get_dcp_group().world_size`，`_update_states` 读 `get_pp_group()`。

    无卡 harness 里这些 group 没初始化（replay_harness 已确认），这里就地打一个
    world_size=1 的桩，**仅用于本探针**，把"没有 group 就根本构造不出对象"这件事
    变成可复现的最小实验。
    """
    import vllm_ascend.worker.block_table as bt_mod

    class _Group:
        world_size = 1
        rank_in_group = 0
        rank = 0
        is_last_rank = True
        is_first_rank = True

    fake = _Group()
    bt_mod.get_dcp_group = lambda *a, **k: fake
    try:
        import vllm.distributed.parallel_state as ps
        ps.get_dcp_group = lambda *a, **k: fake
        ps.get_pp_group = lambda *a, **k: fake
    except Exception:
        pass


def instrument_torch_npu() -> None:
    """把 torch.npu / torch_npu / torch.Tensor 上的设备 API 换成计数代理。"""
    import torch
    import torch_npu

    npu = torch.npu
    for name in dir(npu):
        if name.startswith("_"):
            continue
        try:
            value = getattr(npu, name)
        except Exception:
            continue
        if callable(value) and not isinstance(value, type):
            def make(orig, name):
                def wrapper(*a, **k):
                    _record(f"torch.npu.{name}")
                    return orig(*a, **k)
                return wrapper
            try:
                setattr(npu, name, make(value, name))
            except Exception:
                pass
        elif isinstance(value, type):
            # Stream / Event / NPUGraph 之类的类：包一层计数构造器
            def make_cls(orig, name):
                class _Counted(orig):    # type: ignore[misc, valid-type]
                    def __init__(self, *a, **k):
                        _record(f"torch.npu.{name}()")
                        super().__init__(*a, **k)
                _Counted.__name__ = f"Counted{name}"
                return _Counted
            try:
                setattr(npu, name, make_cls(value, name))
            except Exception:
                pass

    # torch.accelerator.*（vLLM 0.26 的新门面）
    acc = getattr(torch, "accelerator", None)
    if acc is not None:
        for name in ("synchronize", "current_device_index", "device_count", "is_available",
                     "empty_cache", "memory_stats", "get_memory_info", "set_device_index",
                     "current_stream", "current_accelerator", "reset_peak_memory_stats"):
            value = getattr(acc, name, None)
            if not callable(value):
                continue

            def make(orig, name):
                def wrapper(*a, **k):
                    _record(f"torch.accelerator.{name}")
                    return orig(*a, **k)
                return wrapper
            try:
                setattr(acc, name, make(value, name))
            except Exception:
                pass

    # torch_npu 顶层函数
    for name in dir(torch_npu):
        if name.startswith("_") or name in ("npu",):
            continue
        try:
            value = getattr(torch_npu, name)
        except Exception:
            continue
        if callable(value) and not isinstance(value, type):
            def make(orig, name):
                def wrapper(*a, **k):
                    _record(f"torch_npu.{name}")
                    return orig(*a, **k)
                return wrapper
            try:
                setattr(torch_npu, name, make(value, name))
            except Exception:
                pass


def _triton_stats(no_triton: bool) -> dict:
    if no_triton:
        return {}
    try:
        from pi_harness.runner import triton_cpu
        return triton_cpu.stats()
    except Exception as e:                        # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-patch", action="store_true",
                    help="不装 device 重映射补丁（预计在 CpuGpuBuffer 处失败，用于对照）")
    ap.add_argument("--no-triton", action="store_true",
                    help="不装 runner/triton_cpu 的 slot-mapping CPU 替换（对照：应在 launch 处崩）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from pi_harness import shim
    phase("install_shim")
    shim.install(device="cpu")

    patch_report = None
    if not args.no_patch:
        from probes.shim_patch_proposal import install_device_remap
        phase("install_patch")
        patch_report = install_device_remap()

    step("import_torch_npu", lambda: __import__("torch_npu"))
    step("import_vllm", lambda: __import__("vllm"))
    step("import_vllm_ascend_ops", lambda: __import__("vllm_ascend.ops"))
    step("import_vllm_ascend_platform", lambda: __import__("vllm_ascend.platform"))

    phase("import_model_runner_v1")
    instrument_torch_npu()          # 计数代理要在目标模块 import 之前装好
    step("import_model_runner_v1",
         lambda: __import__("vllm_ascend.worker.model_runner_v1", fromlist=["x"]),
         keep_phase=True)

    phase("build_CpuGpuBuffer")
    def _cgp():
        import torch
        from vllm.v1.utils import CpuGpuBuffer
        b = CpuGpuBuffer(1024, dtype=torch.int32, device=torch.device("npu"))
        b.np[:8] = 5
        b.copy_to_gpu(8)
        b.copy_to_cpu(8)
        return str(b.gpu.device)
    step("build_CpuGpuBuffer", _cgp)

    phase("build_block_table")
    def _bt():
        import torch
        _stub_dist_groups()               # BlockTable.__init__ 读 get_dcp_group().world_size
        if not args.no_triton:
            from pi_harness.runner import triton_cpu
            triton_cpu.install()          # _compute_slot_mapping_kernel → numpy 等价实现
        from vllm_ascend.worker.block_table import BlockTable
        bt = BlockTable(
            block_size=128,
            max_num_reqs=32,
            max_num_blocks_per_req=64,
            max_num_batched_tokens=2048,
            pin_memory=False,
            device=torch.device("npu"),
            kernel_sizes=[128],
            cp_kv_cache_interleave_size=1,
            kv_cache_group=None,
        )
        bt.block_table.np[:2, :3] = 1
        bt.commit_block_table(2)
        # compute_slot_mapping 会 launch triton kernel（本轮最重要的设备交互点之一）
        qsl = torch.tensor([0, 3], dtype=torch.int32)
        pos = torch.tensor([0, 1, 2], dtype=torch.int64)
        bt.compute_slot_mapping(1, qsl, pos)
        out = bt.slot_mapping.cpu[:3].tolist()
        extra = ""
        if not args.no_triton:
            from pi_harness.runner import triton_cpu
            extra = f" triton_cpu.stats={triton_cpu.stats()}"
        return f"slot_mapping[:3]={out}{extra}"
    step("build_block_table", _bt)

    phase("call_torch_npu_apis")
    def _apis():
        import torch
        torch.zeros(4, device="npu")
        torch.zeros(4).npu()
        torch.zeros(4).to("npu")
        torch.zeros(4, pin_memory=True)
        torch.npu.current_stream()
        torch.npu.Event()
        torch.npu.synchronize()
        torch.npu.mem_get_info()
        torch.npu.empty_cache()
        torch.accelerator.synchronize()
        with torch.npu.stream(torch.npu.Stream()):
            pass
        return "ok"
    step("call_torch_npu_apis", _apis)

    phase("done")
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "patch_applied": not args.no_patch,
        "triton_cpu_applied": not args.no_triton,
        "patch_report": patch_report,
        "shim_report": shim.report(),
        "triton_cpu_stats": _triton_stats(args.no_triton),
        "phases": {k: dict(v) for k, v in COUNTS.items()},
        "totals": dict(sum(COUNTS.values(), Counter())),
        "errors": ERRORS,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_nopatch" if args.no_patch else ""
    (out / f"devdeps_dynamic_hits{suffix}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")

    print("\n=== 动态命中总计 ===")
    for k, v in sorted(result["totals"].items(), key=lambda kv: -kv[1]):
        print(f"{v:5d}  {k}")
    if ERRORS:
        print("\n=== 失败步骤 ===")
        for e in ERRORS:
            print(f"- {e['step']}: {e['exc'][:200]}")
    print(f"\nwrote {out}/devdeps_dynamic_hits{suffix}.json")
    return 0


def step(name: str, fn, keep_phase: bool = False) -> None:
    """跑一步；失败也继续（记录到 ERRORS）。count 归到当前 phase。"""
    if not keep_phase:
        phase(name)
    else:
        PHASE["name"] = name
        COUNTS.setdefault(name, Counter())
    try:
        v = fn()
        print(f"OK   {name}: {repr(v)[:90]}", flush=True)
    except BaseException as e:                      # noqa: BLE001
        ERRORS.append({"step": name, "exc": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[:4000]})
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
