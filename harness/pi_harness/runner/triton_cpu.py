"""把 prepare_input 路径上的 Ascend Triton kernel 换成 CPU 等价实现。

为什么这是必需的
----------------
`vllm_ascend/worker/block_table.py::BlockTable.compute_slot_mapping()` 每步都会
**从 Python 发起一次 Triton kernel launch**：

    _compute_slot_mapping_kernel[(num_reqs + 1,)](num_tokens, max_num_tokens,
        query_start_loc, positions, block_table.gpu, stride, block_size,
        slot_mapping.gpu, KV_CACHE_BLOCK_SIZE=..., BLOCKS_PER_KV_BLOCK=...,
        TOTAL_CP_WORLD_SIZE=..., TOTAL_CP_RANK=..., CP_KV_CACHE_INTERLEAVE_SIZE=...,
        PAD_ID=..., TILE_BLOCK_SIZE=1024, BLOCK_TABLE_WINDOW_SIZE=...)

无卡容器里这个 launch 会走到 triton-ascend 的 NPU Launcher 并因无设备失败。
**注意：Python 侧的 launch 开销（参数绑定 / 特化 / cache 查找 / launcher 调度）
本身就是 prepare_input CPU 负载的重要组成**，因此本 harness：
  * 用 numpy 复刻 kernel 的**数值语义**（保证后续步骤吃到正确的 slot_mapping）
  * 用 `launch_cost_us` 显式注入"Python launch 开销"（默认 0；由 profiling 侧
    实测值填充，见 docs/06 的偏差表）

数值语义（照抄 kernel 源码 `vllm_ascend/ops/triton/compute_slot_mapping.py`）
  * 对每个 request r：[start, end) = query_start_loc[r], query_start_loc[r+1]
    slot[i] = block_table[r, pos[i] // block_size] * KV_CACHE_BLOCK_SIZE
              + (pos[i] % block_size)  （BLOCKS_PER_KV_BLOCK=1 时）
  * 对 [num_tokens, max_num_tokens) 的 padding 填 PAD_SLOT_ID

本文件**只**替换这一个 kernel；其它 kernel 若被触发会走 shim 的 lazy-init 探针
并抛错，从而让"未覆盖路径"显式暴露（不允许静默跳过）。
"""

from __future__ import annotations

import time

import numpy as np
import torch

_state = {
    "launches": 0,
    "launch_ns": 0,
    "launch_cost_us": 0.0,
    "fallback_ms": 0.0,
    "mode": "cpu_fallback",
}

# padding 区（[num_tokens, max_num_tokens)）内容恒定，缓存后只写 token 区，
# 避免每步 O(max_num_tokens) 的 memset 污染测量。
_pad_cache: dict[tuple[int, int], np.ndarray] = {}


def set_launch_cost_us(us: float) -> None:
    _state["launch_cost_us"] = float(us)


def set_mode(mode: str) -> None:
    """`cpu_fallback`（默认，数值正确） | `noop`（最小 CPU，PMU/火焰图用） |
    `inject:<us>`（no-op + 自旋注入 launch 开销）。"""
    _state["mode"] = mode
    if mode.startswith("inject:"):
        set_launch_cost_us(float(mode.split(":", 1)[1]))
        _state["mode"] = "inject"


def stats() -> dict:
    return dict(_state)


def _slot_mapping_cpu(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    pad_id: int,
    blocks_per_kv_block: int = 1,
    kv_cache_block_size: int | None = None,
) -> None:
    kv_cache_block_size = kv_cache_block_size if kv_cache_block_size is not None else block_size
    def _np(t):
        return t.numpy() if t.device.type == "cpu" else t.detach().cpu().numpy()

    qsl = _np(query_start_loc)
    pos = _np(positions)[:num_tokens]
    bt = _np(block_table)
    num_reqs = len(qsl) - 1

    # 全向量化（与 kernel 的 per-request 语义等价）：
    #   token_pos -> 该 token 在 positions 里的下标（全局展平）
    #   req_of_token -> 该 token 属于哪个 request
    lens = np.clip(qsl[1 : num_reqs + 1] - qsl[:num_reqs], 0, None)
    lens = np.minimum(lens, np.maximum(num_tokens, 0)).astype(np.int64, copy=False)
    cum = np.cumsum(lens)
    total = int(cum[-1]) if len(cum) else 0
    if total <= 0:
        return
    req_of_token = np.repeat(np.arange(num_reqs), lens)
    token_pos = np.arange(total, dtype=np.int64) - np.repeat(cum - lens, lens)
    p = pos[token_pos]
    blk_idx = (p // blocks_per_kv_block) // block_size
    vals = bt[req_of_token, blk_idx].astype(np.int64) * kv_cache_block_size + (p % block_size)
    key = (max_num_tokens, pad_id)
    out = _pad_cache.get(key)
    if out is None:
        out = np.full(max_num_tokens, pad_id, dtype=np.int32)
        _pad_cache[key] = out
    out[token_pos] = vals
    slot_mapping[:num_tokens].copy_(torch.from_numpy(out[:num_tokens]))


class _CpuKernel:
    """复刻 `@triton.jit` 装饰后对象的调用语法：`kernel[grid](args..., **constexpr)`。"""

    def __init__(self, name: str = "_compute_slot_mapping_kernel"):
        self.name = name
        self._cpu_impl = _slot_mapping_cpu
        self.calls = 0

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            t0 = time.perf_counter()

            # 模拟 Python launch 开销（默认 0；由 profiling 实测值填充）
            cost = _state["launch_cost_us"]
            if cost:
                # 用 spin 而不是 sleep：真机的 launch 开销是纯 CPU 忙时间
                deadline = time.perf_counter() + cost * 1e-6
                while time.perf_counter() < deadline:
                    pass

            mode = _state["mode"]
            if mode != "noop":
                num_tokens, max_num_tokens = args[0], args[1]
                (query_start_loc, positions, block_table, _stride, block_size,
                 slot_mapping) = args[2:8]
                self._cpu_impl(
                    int(num_tokens),
                    int(max_num_tokens),
                    query_start_loc,
                    positions,
                    block_table,
                    int(block_size),
                    slot_mapping,
                    pad_id=int(kwargs.get("PAD_ID", -1)),
                    blocks_per_kv_block=int(kwargs.get("BLOCKS_PER_KV_BLOCK", 1)),
                    kv_cache_block_size=int(kwargs.get("KV_CACHE_BLOCK_SIZE", block_size)),
                )
            dt = time.perf_counter() - t0
            _state["launches"] += 1
            _state["launch_ns"] += int(dt * 1e9)
            _state["fallback_ms"] += dt * 1e3
            self.calls += 1

        return _launch

    # 让静态扫描/调试能看出它替代了什么
    def __repr__(self) -> str:  # pragma: no cover
        return f"<CpuKernel shim for {self.name}>"


def install() -> None:
    """把 vllm_ascend 的 slot-mapping kernel 换成 CPU 版本（幂等）。"""
    import vllm_ascend.ops.triton.compute_slot_mapping as mod

    if getattr(mod._compute_slot_mapping_kernel, "_pi_cpu_shim", False):
        return
    cpu = _CpuKernel("_compute_slot_mapping_kernel")
    cpu._pi_cpu_shim = True  # type: ignore[attr-defined]
    mod._compute_slot_mapping_kernel = cpu

    # BlockTable 在 import 期以 `from ... import _compute_slot_mapping_kernel`
    # 把名字绑进自己的模块命名空间，因此必须同时替换该模块里的引用。
    import vllm_ascend.worker.block_table as bt

    if hasattr(bt, "_compute_slot_mapping_kernel") and not getattr(
        bt._compute_slot_mapping_kernel, "_pi_cpu_shim", False
    ):
        bt._compute_slot_mapping_kernel = cpu
