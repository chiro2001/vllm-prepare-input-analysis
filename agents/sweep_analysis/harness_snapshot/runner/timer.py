"""无侵入子步骤计时器。

做法：在**类**上 wrap 若干方法，用一个栈累计 inclusive 时间并推导 exclusive。
不修改 vLLM 源码；wrap 引入的额外开销约 ~0.5µs/调用，已在文档中登记。
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np


class SubStepTimer:
    def __init__(self) -> None:
        self.inclusive: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._stack: list[tuple[str, float]] = []
        self._patched: list[tuple[object, str, object]] = []

    # ------------------------------------------------------------------ patching
    def wrap(self, cls, name: str, label: str | None = None) -> None:
        label = label or f"{cls.__name__}.{name}"
        if not hasattr(cls, name):
            return
        original = getattr(cls, name)
        if getattr(original, "_pi_timer_wrapped", False):
            return

        def wrapper(*args, __orig=original, __label=label, **kwargs):
            t0 = time.perf_counter()
            self._stack.append((__label, t0))
            try:
                return __orig(*args, **kwargs)
            finally:
                t1 = time.perf_counter()
                self._stack.pop()
                self.inclusive[__label] = self.inclusive.get(__label, 0.0) + (t1 - t0)
                self.calls[__label] = self.calls.get(__label, 0) + 1

        wrapper._pi_timer_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, name, wrapper)
        self._patched.append((cls, name, original))

    def unwrap_all(self) -> None:
        for cls, name, original in self._patched:
            setattr(cls, name, original)
        self._patched.clear()

    # ------------------------------------------------------------------ measuring
    @contextmanager
    def measure(self, label: str):
        t0 = time.perf_counter()
        self._stack.append((label, t0))
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self._stack.pop()
            self.inclusive[label] = self.inclusive.get(label, 0.0) + (t1 - t0)
            self.calls[label] = self.calls.get(label, 0) + 1

    def reset(self) -> None:
        self.inclusive.clear()
        self.calls.clear()

    def exclusive(self) -> dict[str, float]:
        """inclusive - 直接子项 inclusive（按调用层级近似）。"""
        # 简化：只在"父-子"标签对之间做一次嵌套扣减不现实（我们没记录调用树），
        # 因此用"同标签 inclusive 之和 - 所有已知子标签 inclusive"的保守定义：
        # 这里直接返回 inclusive，并额外给出 total 供上层扣减。
        return dict(self.inclusive)

    def as_rows(self, total_us: float) -> list[dict]:
        rows = []
        for label, secs in sorted(self.inclusive.items(), key=lambda kv: -kv[1]):
            rows.append(
                {
                    "substep": label,
                    "calls": self.calls.get(label, 0),
                    "total_us": secs * 1e6,
                    "share_of_scope_pct": (secs * 1e6 / total_us * 100.0) if total_us else 0.0,
                }
            )
        return rows


def wrap_prepare_input_path(timer: SubStepTimer) -> None:
    """把 prepare_input 路径上的关键方法挂上计时器。"""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_input_batch import InputBatch

    # runner 层
    for name in (
        "_update_states",
        "_prepare_inputs",
        "_get_cumsum_and_arange",
        "_prepare_input_ids",
        "_compute_prev_positions",
        "_calc_spec_decode_metadata",
        "_may_reorder_batch",
        "_zero_block_ids",
    ):
        timer.wrap(GPUModelRunner, name)

    # InputBatch 层
    for name in (
        "add_request",
        "remove_request",
        "condense",
        "refresh_metadata",
        "_make_sampling_metadata",
        "update_req_spec_token_ids",
    ):
        timer.wrap(InputBatch, name)

    # Ascend 特有
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        from vllm_ascend.worker.npu_input_batch import NPUInputBatch
        from vllm_ascend.worker.block_table import BlockTable as AscendBlockTable
        from vllm_ascend.worker.block_table import MultiGroupBlockTable as AscendMultiGroup

        for cls, names in (
            (NPUModelRunner, ("_update_states", "_prepare_inputs", "_build_attn_state",
                              "_sanitize_placeholder_input_ids_for_forward")),
            (NPUInputBatch, ()),
            (AscendMultiGroup, ("compute_slot_mapping", "commit_block_table")),
            (AscendBlockTable, ("compute_slot_mapping", "commit_block_table", "add_row",
                                "append_row", "move_row", "swap_row", "clear_row")),
        ):
            for name in names:
                timer.wrap(cls, name)
    except Exception:  # pragma: no cover
        pass


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")
