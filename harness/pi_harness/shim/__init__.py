"""无卡 CPU shim。

设计依据（实测）
----------------
* `import torch_npu` 在没有 NPU 设备的容器里**可以成功**（只有 warning），
  `torch` / `vllm` / `vllm_ascend` 也能 import。所以本 shim **不伪造 torch_npu 模块**。
* 真正会炸的是**设备操作**：任何 `torch.zeros(device="npu")` / `torch.npu.current_stream()`
  / `torch.npu.synchronize()` 都会走 `torch_npu.npu._lazy_init()` → `aclInit` 失败
  (`RuntimeError ... error code is 507008`)。
* 因此 shim 的策略是：**把 device 统一成 CPU，把所有流/事件/显存类 API 换成 no-op**，
  并保留一个「未拦截的设备操作」探针（`_lazy_init` guard），用于审计还有哪些路径没被覆盖。

保真度声明：被替换的 API 在真机上的 CPU 侧成本（launch / event sync / 显存查询）
在无卡 harness 里默认为 0，可用 `PI_OP_COST_*` 环境变量注入经验成本模型。详见
`docs/06-synthetic-load.md` 的「已知偏差」表。
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections import Counter

import torch

_state: dict[str, object] = {
    "installed": False,
    "device": "cpu",
    "calls": Counter(),
    "warnings": [],
    "degraded_copy_bytes": Counter(),
}
_lock = threading.Lock()


def _record(api: str) -> None:
    with _lock:
        _state["calls"][api] += 1  # type: ignore[index]


def _record_bytes(bucket: str, n: int) -> None:
    with _lock:
        _state["degraded_copy_bytes"][bucket] += int(n)  # type: ignore[index]


# --------------------------------------------------------------------------------------
# no-op 流 / 事件
# --------------------------------------------------------------------------------------
class CpuShimStream:
    """替代 torch.npu.Stream。所有同步语义都是 no-op（无卡时没有异步在飞）。"""

    def __init__(self, device=None, priority: int = 0, **kwargs):
        self.device = device if device is not None else torch.device("cpu")
        self.priority = priority

    def synchronize(self) -> None:
        _record("Stream.synchronize")

    def wait_stream(self, other) -> None:
        _record("Stream.wait_stream")

    def wait_event(self, event) -> None:
        _record("Stream.wait_event")

    def record_event(self, event=None):
        _record("Stream.record_event")
        return event if event is not None else CpuShimEvent()

    def query(self) -> bool:
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<CpuShimStream>"


class CpuShimEvent:
    """替代 torch.npu.Event。"""

    def __init__(self, enable_timing: bool = False, blocking: bool = False,
                 interprocess: bool = False, external: bool = False, **kwargs):
        self.enable_timing = enable_timing
        self.blocking = blocking
        self._recorded = False

    def record(self, stream=None) -> None:
        _record("Event.record")
        self._recorded = True

    def wait(self, stream=None) -> None:
        _record("Event.wait")

    def synchronize(self) -> None:
        _record("Event.synchronize")

    def query(self) -> bool:
        return True

    def elapsed_time(self, other) -> float:
        return 0.0

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<CpuShimEvent>"


def _stream_ctx(stream=None, device=None):
    """`with torch.npu.stream(s):` 的替身。"""
    _record("npu.stream")
    return contextlib.nullcontext()


class _DummyDeviceProps:
    def __init__(self, name: str = "Ascend910B(shim)"):
        self.name = name
        self.major = 0
        self.minor = 0
        self.total_memory = 64 * (1 << 30)
        self.multi_processor_count = 0
        self.L2_cache_size = 0
        self.max_threads_per_multi_processor = 0


def _noop(name: str):
    def _fn(*args, **kwargs):
        _record(name)
        return None
    return _fn


def _install_lazy_init_guard(torch_npu) -> None:
    """把 torch_npu 的 `_lazy_init` 换成会**报出调用点**的探针。

    目的：如果还有没被 shim 覆盖的路径试图真正访问设备，我们要在文档里有据可查，
    而不是拿到一条含义不明的 aclInit 报错。
    """
    npu_mod = getattr(torch_npu, "npu", None)
    if npu_mod is None or not hasattr(npu_mod, "_lazy_init"):
        _state["warnings"].append("torch_npu.npu._lazy_init not found; guard not installed")
        return
    original = npu_mod._lazy_init

    def _guarded_lazy_init(*args, **kwargs):
        if os.environ.get("PI_SHIM_LAZY_INIT_STRICT", "1") == "1":
            import traceback

            stack = "".join(traceback.format_stack()[-6:-1])
            _record("UNSHIMMED_DEVICE_OP")
            _state["warnings"].append(f"unshimmed device op:\n{stack}")
            raise RuntimeError(
                "PI_SHIM: 无卡 harness 拦截到未 shim 的设备操作（torch_npu._lazy_init）。\n"
                "说明该路径仍会真正访问 NPU；请把它加进 pi_harness/shim 的替换表，"
                "或确认它不在 prepare_input 路径上。调用栈：\n" + stack
            )
        return original(*args, **kwargs)

    npu_mod._lazy_init = _guarded_lazy_init


class _CpuTarget:
    backend = "cpu"
    arch = "cpu"
    warp_size = 1

    def __str__(self) -> str:  # pragma: no cover
        return "cpu-shim"

    def __repr__(self) -> str:  # pragma: no cover
        return "<Target cpu-shim>"


class _ActiveDriverProxy:
    """triton-ascend 的 driver 会在 import 期查询 NPU arch（无卡时 SystemError）。"""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def get_current_target(self, *args, **kwargs):
        _record("triton.get_current_target")
        return _CpuTarget()

    def get_current_device(self, *args, **kwargs):
        try:
            return object.__getattribute__(self, "_inner").get_current_device(*args, **kwargs)
        except Exception:
            return 0

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)


def _patch_triton() -> None:
    try:
        import triton  # noqa: F401
        from triton.runtime import driver as _triton_driver
    except Exception as exc:  # pragma: no cover
        _state["warnings"].append(f"triton shim skipped: {exc!r}")
        return

    # (1) 根因修复：triton-ascend 的 NPUUtils.get_arch() 会去加载设备扩展；
    #     无卡时 `_load_mod().get_arch()` 直接 SystemError。
    try:
        from triton.backends.ascend import driver as _ascend_driver

        _utils_cls = getattr(_ascend_driver, "NPUUtils", None)
        if _utils_cls is not None:
            def _get_arch(self):  # noqa: ANN001
                _record("triton.NPUUtils.get_arch")
                return _state.get("soc_version", "Ascend910B3")

            def _get_aicore_num(self):  # noqa: ANN001
                _record("triton.NPUUtils.get_aicore_num")
                return 24

            def _get_device_properties(self, device=None):  # noqa: ANN001
                return {"max_shared_mem": 1, "num_aicore": 24, "num_vectorcore": 48}

            _utils_cls.get_arch = _get_arch
            _utils_cls.get_aicore_num = _get_aicore_num
            _utils_cls.get_device_properties = _get_device_properties
    except Exception as exc:  # pragma: no cover
        _state["warnings"].append(f"triton NPUUtils shim skipped: {exc!r}")

    # (2) 兜底：任何走 `triton.runtime.driver.active.get_current_target()` 的代码路径
    active = getattr(_triton_driver, "active", None)
    if active is None or isinstance(active, _ActiveDriverProxy):
        return
    _triton_driver.active = _ActiveDriverProxy(active)
    try:
        triton.runtime.driver.active = _triton_driver.active
    except Exception:  # pragma: no cover
        pass


def _is_npu_like(value) -> bool:
    if isinstance(value, torch.device):
        return value.type in ("npu", "cuda", "privateuseone")
    if isinstance(value, str):
        low = value.lower()
        return low.startswith(("npu", "cuda", "privateuseone"))
    return False


def _normalize_device(value):
    """把任何 npu 设备说明符映射到 CPU。"""
    if value is None:
        return None
    if isinstance(value, torch.device):
        if value.type in ("npu", "cuda", "privateuseone"):
            _record("device.map_npu_to_cpu")
            return torch.device("cpu")
        return value
    if isinstance(value, str):
        low = value.lower()
        if low.startswith("npu") or low.startswith("cuda") or low.startswith("privateuseone"):
            _record("device.map_npu_to_cpu_str")
            return "cpu"
        return value
    return value


def _patch_tensor_and_factories() -> None:
    """第二层 shim：tensor/device 层的 npu→cpu 映射。

    这一层是**必需的**：`CpuGpuBuffer` 会调 `torch.zeros(..., device=<npu>)`，
    任何 `.npu()` / `.to("npu")` 都会走 privateuse1 dispatch 触发真实 aclInit。
    """
    T = torch.Tensor

    def _tensor_to(self, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if target is not None and _is_npu_like(target):
            _record("Tensor.to(npu)")
            if self.is_cpu:
                # 真机是 H2D/D2H 拷贝；无卡下退化为 no-op（同设备）。按字节记账，
                # 供 docs/06 用实测 DMA 带宽把"缺失的拷贝成本"补回来。
                _record_bytes("to_npu", self.numel() * self.element_size())
            args = tuple(a for a in args if a is not target) if args and args[0] is target else args
            if args:
                args = (_normalize_device(args[0]),) + tuple(args[1:])
            if "device" in kwargs:
                kwargs = dict(kwargs)
                kwargs["device"] = torch.device("cpu")
            return _original_to(self, *args, **kwargs)
        if args:
            args = (_normalize_device(args[0]),) + tuple(args[1:])
        if "device" in kwargs:
            kwargs = dict(kwargs)
            kwargs["device"] = _normalize_device(kwargs["device"])
        return _original_to(self, *args, **kwargs)

    _original_to = T.to
    T.to = _tensor_to

    def _tensor_npu(self, *args, **kwargs):
        _record("Tensor.npu")
        return self

    def _tensor_npu_(self, *args, **kwargs):
        _record("Tensor.npu_")
        return self

    def _tensor_cuda(self, *args, **kwargs):
        _record("Tensor.cuda")
        return self

    T.npu = _tensor_npu
    T.npu_ = _tensor_npu_
    T.cuda = _tensor_cuda

    # --- factories ---------------------------------------------------------------
    def _wrap_factory(fn, name: str):
        def _wrapped(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _normalize_device(kwargs["device"])
            if kwargs.get("pin_memory"):
                _record(f"pin_memory->False:{name}")
                kwargs["pin_memory"] = False
            return fn(*args, **kwargs)

        return _wrapped

    for name in (
        "zeros", "ones", "empty", "full", "tensor", "arange", "randn", "rand",
        "randint", "zeros_like", "ones_like", "empty_like", "full_like",
        "scatter", "linspace", "eye",
    ):
        if hasattr(torch, name):
            setattr(torch, name, _wrap_factory(getattr(torch, name), name))

    # --- nn.Module ---------------------------------------------------------------
    try:
        import torch.nn as nn

        def _module_npu(self, *args, **kwargs):
            _record("Module.npu")
            return self

        def _module_to(self, *args, **kwargs):
            if args:
                args = (_normalize_device(args[0]),) + tuple(args[1:])
            if "device" in kwargs:
                kwargs["device"] = _normalize_device(kwargs["device"])
            return _original_module_to(self, *args, **kwargs)

        _original_module_to = nn.Module.to
        nn.Module.to = _module_to
        nn.Module.npu = _module_npu
    except Exception as exc:  # pragma: no cover
        _state["warnings"].append(f"nn.Module shim skipped: {exc!r}")

    # --- Generator / pin_memory ---------------------------------------------------
    # 注意：必须保留"类"的语义（`torch.Generator | None` 这种注解在 transformers
    # 里会在 import 期求值），所以用子类而不是函数替换。
    _base_generator = torch.Generator

    class _CpuGenerator(_base_generator):  # type: ignore[misc, valid-type]
        def __init__(self, device: str = "cpu"):
            if _is_npu_like(device):
                _record("torch.Generator(npu)")
            super().__init__(device="cpu")

    torch.Generator = _CpuGenerator

    def _is_pinned(self):  # pragma: no cover
        return False

    T.is_pinned = _is_pinned


def pin_memory_available() -> bool:
    try:
        t = torch.empty(8, dtype=torch.int32, pin_memory=True)
        del t
        return True
    except Exception:
        return False


def _wrap_copy_to_gpu() -> None:
    """审计 `CpuGpuBuffer.copy_to_gpu()` 的真实搬运量。

    真机上这是 **H2D async DMA**；无卡 harness 里退化成 CPU→CPU memcpy（同 buffer
    时甚至就是 `torch.Tensor.copy_` 的浅拷贝路径）。我们用字节数记账，
    这样文档里可以用实测 DMA 带宽把"缺失的设备拷贝成本"补进来。
    """
    from vllm.v1.utils import CpuGpuBuffer

    original = CpuGpuBuffer.copy_to_gpu
    if getattr(original, "_pi_shim_wrapped", False):
        return

    def copy_to_gpu(self, n: int | None = None):
        _record("CpuGpuBuffer.copy_to_gpu")
        slice_ = self.cpu if n is None else self.cpu[:n]
        _record_bytes("copy_to_gpu", slice_.numel() * slice_.element_size())
        return original(self, n)

    copy_to_gpu._pi_shim_wrapped = True  # type: ignore[attr-defined]
    CpuGpuBuffer.copy_to_gpu = copy_to_gpu

    original_c2c = CpuGpuBuffer.copy_to_cpu

    def copy_to_cpu(self, n: int | None = None):
        _record("CpuGpuBuffer.copy_to_cpu")
        slice_ = self.gpu if n is None else self.gpu[:n]
        _record_bytes("copy_to_cpu", slice_.numel() * slice_.element_size())
        return original_c2c(self, n)

    CpuGpuBuffer.copy_to_cpu = copy_to_cpu


def install(device: str = "cpu") -> None:
    """在 import vllm / vllm_ascend 之前调用。幂等。"""
    if _state["installed"]:
        return
    _state["device"] = device

    import torch_npu  # noqa: F401  (真实模块；无卡下 import 是安全的)

    _state["pin_memory_available"] = pin_memory_available()
    _patch_tensor_and_factories()

    npu = torch.npu

    # --- 流 / 事件 ---------------------------------------------------------------------
    npu.Stream = CpuShimStream
    npu.Event = CpuShimEvent
    npu.ExternalEvent = CpuShimEvent
    _default_stream = CpuShimStream()
    npu.current_stream = lambda device=None: _default_stream
    npu.default_stream = lambda device=None: _default_stream
    npu.stream = _stream_ctx
    npu.set_stream = _noop("npu.set_stream")

    # --- 同步 / 设备 -------------------------------------------------------------------
    npu.synchronize = _noop("npu.synchronize")
    npu.current_device = lambda: 0
    npu.set_device = _noop("npu.set_device")
    npu.device_count = lambda: 1
    npu.is_available = lambda: True
    npu.is_initialized = lambda: True
    npu.get_device_name = lambda *a, **k: "Ascend910B(shim)"
    npu.get_device_properties = lambda *a, **k: _DummyDeviceProps()

    # --- 显存 ---------------------------------------------------------------------------
    npu.mem_get_info = lambda *a, **k: (1 << 40, 1 << 40)
    npu.memory_allocated = lambda *a, **k: 0
    npu.max_memory_allocated = lambda *a, **k: 0
    npu.memory_reserved = lambda *a, **k: 0
    npu.reset_peak_memory_stats = _noop("npu.reset_peak_memory_stats")
    npu.memory_stats = lambda *a, **k: {}
    npu.empty_cache = _noop("npu.empty_cache")

    # --- graph / 编译 -------------------------------------------------------------------
    npu.is_current_stream_capturing = lambda: False
    npu.graph_pool_handle = lambda: 0
    npu.set_compile_mode = _noop("npu.set_compile_mode")
    npu.graph = lambda *a, **k: contextlib.nullcontext()
    npu.NPUGraph = type("CpuShimNPUGraph", (), {})

    # `torch.npu.config.allow_internal_format = True`（model_runner_v1 模块级语句）
    if not hasattr(npu, "config"):
        npu.config = type("_ShimConfig", (), {})()
    if not hasattr(npu.config, "allow_internal_format"):
        npu.config.allow_internal_format = False

    _install_lazy_init_guard(torch_npu)
    _patch_triton()
    _wrap_copy_to_gpu()
    _state["installed"] = True
    _state["torch_npu"] = torch_npu
    return None


def report() -> dict:
    """返回被 shim 命中的 API 计数，用于保真度文档。"""
    with _lock:
        return {
            "installed": _state["installed"],
            "device": _state["device"],
            "pin_memory_available": _state.get("pin_memory_available"),
            "calls": dict(_state["calls"]),
            "degraded_copy_bytes": dict(_state["degraded_copy_bytes"]),  # type: ignore[arg-type]
            "warnings": list(_state["warnings"]),
        }


def reset_report() -> None:
    with _lock:
        _state["calls"] = Counter()
        _state["warnings"] = []
        _state["degraded_copy_bytes"] = Counter()
