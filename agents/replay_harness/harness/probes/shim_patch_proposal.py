#!/usr/bin/env python3
"""tensor/device 层 npu→cpu 重映射的**参考实现**（提案，不在 owner 的 shim 目录里）。

背景（实测，见 data/harness/devdeps_stacks/A8_device_runtime_probes_devapi.txt）：
`pi_harness/shim` 目前只替换 `torch.npu.*` 命名空间，**没有**碰 device 字符串本身，
因此下面这些调用会穿透到真实的 privateuse1 后端：

    torch.zeros(4, device="npu")        -> aclInit 507008 / UNSHIMMED_DEVICE_OP
    torch.zeros(4).npu()                -> 同上
    x.to("npu") / x.to(torch.device("npu:0"))  -> 同上
    torch.zeros(4, pin_memory=True)     -> RuntimeError: no pinned memory allocator
    CpuGpuBuffer(..., device=torch.device("npu"))  -> torch.zeros_like(..., device=npu)

本模块给出最小可用的 Python 层拦截表。用法::

    python harness/probes/shim_patch_proposal.py --selftest     # 与 pi_harness.shim 联合自测
    python harness/probes/shim_patch_proposal.py --dump         # 打印将 patch 的对象清单

集成方式（replay_harness 决定）::

    from probes.shim_patch_proposal import install_device_remap
    install_device_remap()          # 在 shim.install() 之后调用
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

_CALLS: Counter = Counter()
_ORIGINALS: dict[str, object] = {}

# 构造型 API：device= 与 pin_memory= 都要处理
_FACTORIES = (
    "zeros", "ones", "empty", "full", "tensor", "arange", "linspace",
    "rand", "randn", "randint", "randperm", "eye", "scalar_tensor",
    "as_tensor", "asarray", "empty_strided",
)
_FACTORIES_LIKE = ("zeros_like", "ones_like", "empty_like", "full_like", "rand_like", "randn_like")


def _is_npu_device(x) -> bool:
    import torch
    if isinstance(x, str):
        return x == "npu" or x.startswith("npu:")
    if isinstance(x, torch.device):
        return x.type == "npu"
    return False


def _remap(x):
    """把 npu device（字符串或 torch.device）换成 cpu。"""
    import torch
    if _is_npu_device(x):
        _CALLS["device_remapped"] += 1
        return torch.device("cpu")
    return x


def install_device_remap(keep_pin_memory: bool = False, verbose: bool = False) -> dict:
    """打补丁。**必须**在 import torch 之后、构造任何 npu tensor 之前调用。

    keep_pin_memory=True 时保留 pin_memory=True（容器里没有 pinned allocator，会抛异常，
    仅用于验证"真的没有 pinned 内存"这一结论）。
    """
    import torch
    import torch.nn as nn

    if _ORIGINALS:
        return report()                        # 幂等

    def _fix_kwargs(kwargs: dict) -> dict:
        if "device" in kwargs:
            kwargs["device"] = _remap(kwargs["device"])
        if kwargs.get("pin_memory") and not keep_pin_memory:
            _CALLS["pin_memory_downgraded"] += 1
            kwargs["pin_memory"] = False
        return kwargs

    # --- 1. 构造型 API ---------------------------------------------------------------
    for name in _FACTORIES:
        orig = getattr(torch, name, None)
        if orig is None:
            continue
        _ORIGINALS[f"torch.{name}"] = orig

        def make(orig, name):
            @functools.wraps(orig)
            def wrapper(*args, **kwargs):
                args = tuple(_remap(a) for a in args)
                _fix_kwargs(kwargs)
                _CALLS[name] += 1
                return orig(*args, **kwargs)
            return wrapper

        setattr(torch, name, make(orig, name))

    for name in _FACTORIES_LIKE:
        orig = getattr(torch, name, None)
        if orig is None:
            continue
        _ORIGINALS[f"torch.{name}"] = orig

        def make(orig, name):
            @functools.wraps(orig)
            def wrapper(*args, **kwargs):
                args = tuple(_remap(a) for a in args)
                _fix_kwargs(kwargs)
                _CALLS[name] += 1
                return orig(*args, **kwargs)
            return wrapper

        setattr(torch, name, make(orig, name))

    # --- 2. Tensor 方法 ---------------------------------------------------------------
    _ORIGINALS["Tensor.to"] = torch.Tensor.to

    @functools.wraps(_ORIGINALS["Tensor.to"])
    def _to(self, *args, **kwargs):
        args = tuple(_remap(a) for a in args)
        if "device" in kwargs:
            kwargs["device"] = _remap(kwargs["device"])
        _CALLS["Tensor.to"] += 1
        return _ORIGINALS["Tensor.to"](self, *args, **kwargs)

    torch.Tensor.to = _to

    _ORIGINALS["Tensor.npu"] = getattr(torch.Tensor, "npu", None)

    def _npu(self, *args, **kwargs):
        kwargs.pop("non_blocking", None)
        _CALLS["Tensor.npu"] += 1
        return self.to("cpu")

    torch.Tensor.npu = _npu
    if hasattr(torch.Tensor, "npu_"):
        _ORIGINALS["Tensor.npu_"] = torch.Tensor.npu_

        def _npu_(self, *args, **kwargs):
            _CALLS["Tensor.npu_"] += 1
            return self

        torch.Tensor.npu_ = _npu_

    _ORIGINALS["Tensor.pin_memory"] = getattr(torch.Tensor, "pin_memory", None)

    def _pin_memory(self, device=None):
        _CALLS["pin_memory_downgraded"] += 1
        return self

    torch.Tensor.pin_memory = _pin_memory

    # --- 3. nn.Module ----------------------------------------------------------------
    _ORIGINALS["Module.to"] = nn.Module.to

    @functools.wraps(_ORIGINALS["Module.to"])
    def _m_to(self, *args, **kwargs):
        args = tuple(_remap(a) for a in args)
        if "device" in kwargs:
            kwargs["device"] = _remap(kwargs["device"])
        _CALLS["Module.to"] += 1
        return _ORIGINALS["Module.to"](self, *args, **kwargs)

    nn.Module.to = _m_to
    nn.Module.npu = lambda self, *a, **k: self.to("cpu")      # type: ignore[assignment]

    # --- 4. torch.accelerator（vLLM 0.26 新 API，prepare_input 链上有 _sync_device） ----
    acc = getattr(torch, "accelerator", None)
    if acc is not None:
        def _wrap(name, value):
            _ORIGINALS[f"torch.accelerator.{name}"] = value
            if callable(value):
                @functools.wraps(value)
                def wrapper(*a, **k):
                    _CALLS[f"accelerator.{name}"] += 1
                    return value(*a, **k)
                return wrapper
            return value

        acc.synchronize = _wrap("synchronize", lambda *a, **k: None)
        acc.current_device_index = _wrap("current_device_index", lambda *a, **k: 0)
        acc.set_device_index = _wrap("set_device_index", lambda *a, **k: None)
        acc.device_count = _wrap("device_count", lambda *a, **k: 1)
        acc.is_available = _wrap("is_available", lambda *a, **k: True)
        acc.current_accelerator = _wrap("current_accelerator", lambda *a, **k: torch.device("cpu"))
        acc.empty_cache = _wrap("empty_cache", lambda *a, **k: None)
        acc.memory_allocated = _wrap("memory_allocated", lambda *a, **k: 0)
        acc.memory_reserved = _wrap("memory_reserved", lambda *a, **k: 0)
        acc.memory_stats = _wrap("memory_stats", lambda *a, **k: {})
        acc.reset_peak_memory_stats = _wrap("reset_peak_memory_stats", lambda *a, **k: None)
        acc.get_memory_info = _wrap("get_memory_info", lambda *a, **k: (1 << 40, 1 << 40))
        acc.set_stream = _wrap("set_stream", lambda *a, **k: None)
        acc.current_stream = _wrap("current_stream", lambda *a, **k: _CpuStream())
        acc.default_stream = _wrap("default_stream", lambda *a, **k: _CpuStream())
        acc.stream = _wrap("stream", lambda *a, **k: _nullctx())
        acc.Event = type("CpuShimAcceleratorEvent", (), {
            "__init__": lambda self, *a, **k: None,
            "record": lambda self, *a, **k: None,
            "wait": lambda self, *a, **k: None,
            "synchronize": lambda self, *a, **k: None,
            "query": lambda self, *a, **k: True,
        })

    if verbose:
        print(f"patched {len(_ORIGINALS)} objects")
    return report()


def _nullctx():
    import contextlib
    return contextlib.nullcontext()


class _CpuStream:
    def __init__(self, *a, **k):
        pass

    def synchronize(self, *a, **k):
        return None

    def wait_stream(self, *a, **k):
        return None

    def record_event(self, *a, **k):
        return None

    def query(self, *a, **k):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __repr__(self):
        return "<CpuShimAcceleratorStream>"


def report() -> dict:
    return {"patched": sorted(_ORIGINALS), "calls": dict(_CALLS),
            "pin_memory_supported": False}


# ---------------------------------------------------------------------------
# 自测：shim.install() + install_device_remap() 之后重跑冒烟用例
# ---------------------------------------------------------------------------
CASES = [
    ("C6", "CpuGpuBuffer(1024,int32,device=npu)+copy_to_gpu", "cgp_npu"),
    ("C7", "CpuGpuBuffer(device=npu:0, pin_memory=True)", "cgp_npu0_pin"),
    ("C8", "Tensor.npu()", "t_npu"),
    ("C9", "Tensor.to('npu')", "t_to_npu"),
    ("C10", "Tensor.to(device('npu:0'))", "t_to_npu0"),
    ("C11", "torch.zeros(device='npu')", "z_npu"),
    ("C12", "torch.zeros(device='npu:0')", "z_npu0"),
    ("C13", "torch.zeros(pin_memory=True)", "pin"),
    ("C14", "with torch.device('npu'): torch.zeros(2)", "devctx"),
    ("C16", "torch.npu.config.allow_internal_format 读写", "cfg"),
    ("C17", "torch.accelerator.* 基本调用", "acc"),
    ("C18", "torch.zeros_like(cpu_t, device='npu')", "like"),
    ("C19", "torch.empty(4, device='npu') / torch.tensor([1], device='npu')", "factory"),
]


def _run(kind: str):
    import torch
    if kind == "cgp_npu":
        from vllm.v1.utils import CpuGpuBuffer
        b = CpuGpuBuffer(1024, dtype=torch.int32, device=torch.device("npu"))
        b.np[:8] = 7
        b.copy_to_gpu()
        assert torch.equal(b.gpu[:8], b.cpu[:8])
        return f"cpu={b.cpu.device} gpu={b.gpu.device}"
    if kind == "cgp_npu0_pin":
        from vllm.v1.utils import CpuGpuBuffer
        b = CpuGpuBuffer(64, dtype=torch.int32, device=torch.device("npu:0"), pin_memory=True)
        b.copy_to_gpu()
        return f"cpu={b.cpu.device} gpu={b.gpu.device}"
    if kind == "t_npu":
        return str(torch.zeros(4).npu().device)
    if kind == "t_to_npu":
        return str(torch.zeros(4).to("npu").device)
    if kind == "t_to_npu0":
        return str(torch.zeros(4).to(torch.device("npu:0")).device)
    if kind == "z_npu":
        return str(torch.zeros(4, device="npu").device)
    if kind == "z_npu0":
        return str(torch.zeros(4, device="npu:0").device)
    if kind == "pin":
        return str(tuple(torch.zeros(4, pin_memory=True).shape))
    if kind == "devctx":
        with torch.device("npu"):
            t = torch.zeros(2)
        return str(t.device)
    if kind == "cfg":
        torch.npu.config.allow_internal_format = True
        return getattr(torch.npu.config, "allow_internal_format", "<read FAILED>")
    if kind == "acc":
        return {
            "synchronize": torch.accelerator.synchronize(),
            "current_device_index": torch.accelerator.current_device_index(),
            "device_count": torch.accelerator.device_count(),
            "is_available": torch.accelerator.is_available(),
            "current_accelerator": str(torch.accelerator.current_accelerator()),
            "empty_cache": torch.accelerator.empty_cache(),
        }
    if kind == "like":
        t = torch.zeros(4, dtype=torch.int32)
        return str(torch.zeros_like(t, device="npu").device)
    if kind == "factory":
        return [str(torch.empty(4, device="npu").device),
                str(torch.tensor([1], device="npu").device),
                str(torch.arange(4, device="npu").device),
                str(torch.ones(2, 2, device=torch.device("npu:0")).device)]
    raise KeyError(kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=os.environ.get("PI_DEVD_DEPS_OUT", "/work/data/harness"))
    args = ap.parse_args()

    from pi_harness import shim
    shim.install(device="cpu")
    rep = install_device_remap()

    if args.dump:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0

    if not args.selftest:
        print("nothing to do; pass --selftest or --dump")
        return 0

    cases = CASES if not args.only else [c for c in CASES if c[0] in args.only]
    results = []
    out = Path(args.out)
    stacks = out / "devdeps_stacks"
    stacks.mkdir(parents=True, exist_ok=True)
    for cid, title, kind in cases:
        rec = {"id": cid, "title": title}
        t0 = time.perf_counter()
        try:
            rec.update(ok=True, value=repr(_run(kind))[:300])
        except BaseException as e:                   # noqa: BLE001
            rec.update(ok=False, exc=f"{type(e).__name__}: {e}"[:400],
                       traceback=traceback.format_exc()[-2500:])
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        results.append(rec)
        (stacks / f"shim_patch_{cid}.txt").write_text(
            f"# {cid} {title}\n# ok={rec['ok']}\n# value={rec.get('value')}\n"
            f"# exc={rec.get('exc')}\n# tb\n{rec.get('traceback','')}\n")
        print(f"{'OK  ' if rec['ok'] else 'FAIL'} {cid:4s} {title[:55]:57s} "
              f"{(rec.get('value') or rec.get('exc',''))[:90]}")

    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "shim_report": shim.report(), "patch_report": install_device_remap(),
               "cases": results,
               "passed": sum(1 for r in results if r["ok"]),
               "failed": sum(1 for r in results if not r["ok"])}
    (out / "devdeps_shim_patch_selftest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"\n{summary['passed']} passed / {summary['failed']} failed")
    print("patch calls:", json.dumps(summary["patch_report"]["calls"], ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
