#!/usr/bin/env python3
"""D. shim 冒烟验证（无卡容器内跑；被测对象 = harness/pi_harness/shim）。

覆盖 INTERFACES.md 第 2 节的 4 项要求：
  1. `install(device="cpu")` 幂等、`import torch_npu` 命中、`torch.npu.*` 可用；
  2. `from vllm_ascend.worker.model_runner_v1 import NPUModelRunner` 成功；
  3. `CpuGpuBuffer(1024, dtype=torch.int32, device=torch.device("npu"))` 构造 + `copy_to_gpu()`；
  4. `Tensor.npu()` / `.to("npu")` / `torch.zeros(device="npu")` / `pin_memory=True` 可用。

用法::

    python harness/probes/smoke_shim.py                  # 全部用例
    python harness/probes/smoke_shim.py --list
    python harness/probes/smoke_shim.py --only C1 C2
    python harness/probes/smoke_shim.py --out /work/data/harness

每个用例独立 try/except，完整栈写进 `data/harness/devdeps_stacks/shim_smoke_<id>.txt`；
汇总写 `data/harness/devdeps_shim_smoke.json`，shim 的 `report()` 另存
`data/harness/devdeps_shim_report.json`（这一份是「API 运行时命中计数」的证据）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(os.environ.get("PI_REPO_ROOT", "/work"))
OUT = Path(os.environ.get("PI_DEVD_DEPS_OUT", REPO / "data" / "harness"))
STACKS = OUT / "devdeps_stacks"


def _run_case(cid: str, title: str, fn) -> dict:
    rec = {"id": cid, "title": title}
    t0 = time.perf_counter()
    try:
        value = fn()
        rec.update(ok=True, value=repr(value)[:400])
    except BaseException as e:                      # noqa: BLE001
        rec.update(ok=False, exc=f"{type(e).__name__}: {e}"[:600],
                   traceback=traceback.format_exc()[-4000:])
    rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    STACKS.mkdir(parents=True, exist_ok=True)
    (STACKS / f"shim_smoke_{cid}.txt").write_text(
        f"# {cid}: {title}\n# ok={rec['ok']}  {rec['duration_ms']} ms\n"
        f"# value: {rec.get('value')}\n# exc: {rec.get('exc')}\n"
        f"# ---- traceback ----\n{rec.get('traceback', '')}\n"
    )
    return rec


# ---------------------------------------------------------------------------
# 用例定义（顺序敏感：模块级用例按顺序执行）
# ---------------------------------------------------------------------------

def build_cases():
    cases: list[tuple[str, str, object]] = []

    def c0_install_idempotent():
        from pi_harness import shim
        shim.install(device="cpu")
        shim.install(device="cpu")           # 幂等
        assert "torch_npu" in sys.modules, "import torch_npu 未命中"
        import torch_npu                     # noqa: F401
        import torch
        assert hasattr(torch, "npu"), "torch.npu 不存在"
        return {"installed": True, "torch_npu": torch_npu.__file__}

    def c1_torch_npu_importable():
        import torch_npu
        return torch_npu.__file__

    def c2_torch_npu_namespace():
        import torch
        probes = {
            "npu.is_available": torch.npu.is_available(),
            "npu.device_count": torch.npu.device_count(),
            "npu.current_device": torch.npu.current_device(),
            "npu.current_stream": str(torch.npu.current_stream()),
            "npu.synchronize": torch.npu.synchronize(),
            "npu.Event": str(torch.npu.Event()),
            "npu.Stream": str(torch.npu.Stream()),
            "npu.empty_cache": torch.npu.empty_cache(),
            "npu.mem_get_info": torch.npu.mem_get_info(),
        }
        return probes

    def c3_import_vllm():
        import vllm
        return vllm.__version__

    def c4_import_ascend_ops():
        import vllm_ascend.ops
        return vllm_ascend.ops.__file__

    def c5_import_npu_model_runner():
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        return {"cls": str(NPUModelRunner),
                "mro": [c.__name__ for c in NPUModelRunner.__mro__]}

    def c6_cpugpubuffer_npu_device():
        import torch
        from vllm.v1.utils import CpuGpuBuffer
        buf = CpuGpuBuffer(1024, dtype=torch.int32, device=torch.device("npu"))
        out = {"cpu.device": str(buf.cpu.device), "gpu.device": str(buf.gpu.device)}
        buf.np[:8] = 7
        ret = buf.copy_to_gpu()
        out["copy_to_gpu"] = str(ret.device)
        out["gpu[0:3]"] = buf.gpu[:3].tolist()
        import torch as t
        assert t.equal(buf.gpu[:8], buf.cpu[:8]), "CPU→CPU memcpy 结果不一致"
        return out

    def c7_cpugpubuffer_npu0_and_pin():
        import torch
        from vllm.v1.utils import CpuGpuBuffer
        buf = CpuGpuBuffer(64, dtype=torch.int32, device=torch.device("npu:0"),
                           pin_memory=True)
        ret = {"cpu.device": str(buf.cpu.device), "gpu.device": str(buf.gpu.device),
               "cpu.is_pinned": bool(buf.cpu.is_pinned()) if hasattr(buf.cpu, "is_pinned") else None}
        buf.copy_to_gpu()
        return ret

    def c8_tensor_npu_method():
        import torch
        a = torch.zeros(4)
        b = a.npu()
        return {"device": str(b.device), "type": type(b).__name__}

    def c9_tensor_to_npu():
        import torch
        a = torch.zeros(4)
        return str(a.to("npu").device)

    def c10_tensor_to_npu0():
        import torch
        a = torch.zeros(4)
        return str(a.to(torch.device("npu:0")).device)

    def c11_zeros_device_npu():
        import torch
        return str(torch.zeros(4, device="npu").device)

    def c12_zeros_device_npu0():
        import torch
        return str(torch.zeros(4, device="npu:0").device)

    def c13_pin_memory_true():
        import torch
        t = torch.zeros(4, pin_memory=True)
        return {"shape": tuple(t.shape), "str": str(t)[:60]}

    def c14_device_context_npu():
        import torch
        with torch.device("npu"):
            t = torch.zeros(2)
        return str(t.device)

    def c15_npu_stream_event_semantics():
        import torch
        s = torch.npu.Stream()
        e = torch.npu.Event()
        with torch.npu.stream(s):
            e.record()
        e.synchronize()
        torch.npu.current_stream().synchronize()
        return "ok"

    def c16_module_import_chain_attrs():
        """模块 import 期访问的属性（torch.npu.config.allow_internal_format）。"""
        import torch
        torch.npu.config.allow_internal_format = True
        return torch.npu.config.allow_internal_format

    cases = [
        ("C0", "install(device='cpu') 幂等 + import torch_npu 命中", c0_install_idempotent),
        ("C1", "import torch_npu 可用", c1_torch_npu_importable),
        ("C2", "torch.npu.* 基本命名空间（is_available/current_stream/Event/Stream/synchronize/mem_get_info）", c2_torch_npu_namespace),
        ("C3", "import vllm", c3_import_vllm),
        ("C4", "import vllm_ascend.ops", c4_import_ascend_ops),
        ("C5", "from vllm_ascend.worker.model_runner_v1 import NPUModelRunner", c5_import_npu_model_runner),
        ("C6", "CpuGpuBuffer(1024, int32, device='npu') + copy_to_gpu()", c6_cpugpubuffer_npu_device),
        ("C7", "CpuGpuBuffer(device='npu:0', pin_memory=True)", c7_cpugpubuffer_npu0_and_pin),
        ("C8", "Tensor.npu()", c8_tensor_npu_method),
        ("C9", "Tensor.to('npu')", c9_tensor_to_npu),
        ("C10", "Tensor.to(device('npu:0'))", c10_tensor_to_npu0),
        ("C11", "torch.zeros(device='npu')", c11_zeros_device_npu),
        ("C12", "torch.zeros(device='npu:0')", c12_zeros_device_npu0),
        ("C13", "torch.zeros(pin_memory=True)", c13_pin_memory_true),
        ("C14", "with torch.device('npu'): torch.zeros(2)", c14_device_context_npu),
        ("C15", "torch.npu Stream/Event 语义（record/synchronize/stream ctx）", c15_npu_stream_event_semantics),
        ("C16", "torch.npu.config.allow_internal_format 读写", c16_module_import_chain_attrs),
    ]
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cases = build_cases()
    if args.list:
        for cid, title, _ in cases:
            print(f"{cid:4s} {title}")
        return 0
    if args.only:
        cases = [c for c in cases if c[0] in args.only]

    out = Path(args.out)
    global STACKS
    STACKS = out / "devdeps_stacks"

    results = []
    for cid, title, fn in cases:
        print(f"=== {cid} {title}", flush=True)
        rec = _run_case(cid, title, fn)
        results.append(rec)
        print(f"    {'OK  ' if rec['ok'] else 'FAIL'} {rec.get('value') or rec.get('exc')}",
              flush=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "cases": results,
        "passed": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "devdeps_shim_smoke.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    try:
        from pi_harness import shim as _shim
        rep = _shim.report()
        (out / "devdeps_shim_report.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False, default=str) + "\n")
        print("\n=== shim.report() ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False, default=str)[:4000])
    except BaseException as e:                       # noqa: BLE001
        print("shim.report() failed:", e)

    print(f"\n{summary['passed']} passed / {summary['failed']} failed  -> {out}/devdeps_shim_smoke.json")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
