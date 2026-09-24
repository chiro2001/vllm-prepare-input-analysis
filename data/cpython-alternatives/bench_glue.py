#!/usr/bin/env python3
"""微基准：模拟 vLLM `prepare_input` 的"每步构造 22 字段 attention metadata +
十几次小 numpy/torch CPU 算子 + 属性查找"负载。

设计约束（对齐真机负载特征，见 docs/00-INDEX.md §0）：
  * 每步数据规模小：B=1、几十~几百个元素；
  * 单次小算子固定开销 2-9 µs（920B 实测），所以"调用条数"是成本主项；
  * 负载是"短函数调用 + 属性查找 + 小张量"，没有数值热点。

用法：
    python bench_glue.py --variant plain --iters 5000 --rounds 7
输出：JSON（stdout），字段 median_ns / p10_ns / p90_ns / us_per_iter / checksum。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import numpy as np

try:
    import torch

    torch.set_num_threads(1)
except ImportError:  # 允许在没有 torch 的解释器上跑 numpy / none 后端
    torch = None  # type: ignore[assignment]

N_FIELDS = 22
B = 1

# --------------------------------------------------------------------------------------
# 输入（在计时循环外构造，模拟每个 engine step 从上游拿到的状态）
# --------------------------------------------------------------------------------------


def make_inputs() -> dict[str, Any]:
    seq_lens_np = np.array([1] * B, dtype=np.int32)
    seq_lens_obj = torch.from_numpy(seq_lens_np.copy()) if torch is not None else seq_lens_np.copy()
    return {
        "num_reqs": B,
        "num_actual_tokens": B,
        "max_query_len": 1,
        "query_start_loc": np.arange(B + 1, dtype=np.int32),
        "seq_lens": seq_lens_obj,
        "block_table": np.arange(B * 4, dtype=np.int32).reshape(B, 4),
        "slot_mapping": np.arange(B, dtype=np.int32),
        "spec_decode": False,
        "num_decode_tokens": B,
        "num_prefill_tokens": 0,
        "num_computed_tokens": np.array([7] * B, dtype=np.int32),
        "seq_lens_np": seq_lens_np,
        "max_seq_len": 8,
        "kv_cache_dtype": "auto",
        "block_size": 128,
        "num_kv_heads": 8,
        "head_size": 128,
        "causal": True,
        "sliding_window": -1,
        "token_ids": np.array([42] * B, dtype=np.int32),
        "positions": torch.arange(B, dtype=torch.int64) if torch is not None else np.arange(B, dtype=np.int64),
        "workspace": np.zeros(64, dtype=np.float32),
    }


INPUTS = make_inputs()


# --------------------------------------------------------------------------------------
# 小张量算子块（15 次调用，全部是 B=1 规模）
# --------------------------------------------------------------------------------------


def small_tensor_ops(inp: dict[str, Any], mode: str = "torch") -> tuple:
    """15 次小算子调用。mode=torch/numpy/none（none 是"剥掉算子"的对照臂）。"""
    if mode == "none":
        return (0, 0, 0, 0, 0, 0, True, 0)
    seq = inp["seq_lens_np"]
    qsl = np.cumsum(seq)                       # 1
    delta = seq[1:] - seq[:-1]                 # 2
    zeros = np.zeros(B, dtype=np.int64)        # 3
    smap = np.arange(B, dtype=np.int32)        # 4
    last = int(seq[-1])                        # 5
    if mode == "numpy":
        t = np.asarray(seq)                    # 6
        t2 = t * 2                             # 7
        s = float(t2.sum())                    # 8
        ar = np.arange(B, dtype=np.int32)      # 9
        cl = t.copy()                          # 10
        mx = int(cl.max())                     # 11
        any_pos = bool((t > 0).any())          # 12
        cs = np.cumsum(t, axis=0)              # 13
        fz = np.zeros(B, dtype=np.float32)     # 14
        ws = inp["workspace"][: 2 * B]         # 15
        return (int(qsl[-1]), int(delta.sum()) if delta.size else 0, float(zeros.sum()),
                int(smap[-1]), float(last), s, any_pos,
                int(cs[-1]) + int(ar[-1]) + int(mx) + int(fz.shape[0]) + int(ws.shape[0]))
    t = torch.from_numpy(seq)                  # 6
    t2 = t.mul(2)                              # 7
    s = float(t2.sum())                        # 8
    ar = torch.arange(B, dtype=torch.int32)    # 9
    cl = t.clone()                             # 10
    mx = int(cl.max())                         # 11
    any_pos = bool((t > 0).any())              # 12
    cs = torch.cumsum(t, 0)                    # 13
    fz = torch.zeros(B, dtype=torch.float32)   # 14
    ws = inp["workspace"][: 2 * B]             # 15
    return (int(qsl[-1]), int(delta.sum()) if delta.size else 0, int(zeros.sum()),
            int(smap[-1]), float(last), s, any_pos, int(cs[-1]) + int(ar[-1]) + int(mx)
            + int(fz.shape[0]) + int(ws.shape[0]))


# --------------------------------------------------------------------------------------
# 6 个变体：区别只在"22 字段容器"的实现方式
# --------------------------------------------------------------------------------------


@dataclass
class MetaPlain:
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: Any
    seq_lens: Any
    block_table: Any
    slot_mapping: Any
    spec_decode: bool
    num_decode_tokens: int
    num_prefill_tokens: int
    num_computed_tokens: Any
    seq_lens_np: Any
    max_seq_len: int
    kv_cache_dtype: str
    block_size: int
    num_kv_heads: int
    head_size: int
    causal: bool
    sliding_window: int
    token_ids: Any
    positions: Any
    workspace: Any


@dataclass(slots=True)
class MetaSlots:
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: Any
    seq_lens: Any
    block_table: Any
    slot_mapping: Any
    spec_decode: bool
    num_decode_tokens: int
    num_prefill_tokens: int
    num_computed_tokens: Any
    seq_lens_np: Any
    max_seq_len: int
    kv_cache_dtype: str
    block_size: int
    num_kv_heads: int
    head_size: int
    causal: bool
    sliding_window: int
    token_ids: Any
    positions: Any
    workspace: Any


class MetaManualDict:
    def __init__(self, num_reqs, num_actual_tokens, max_query_len, query_start_loc,
                 seq_lens, block_table, slot_mapping, spec_decode, num_decode_tokens,
                 num_prefill_tokens, num_computed_tokens, seq_lens_np, max_seq_len,
                 kv_cache_dtype, block_size, num_kv_heads, head_size, causal,
                 sliding_window, token_ids, positions, workspace):
        self.num_reqs = num_reqs
        self.num_actual_tokens = num_actual_tokens
        self.max_query_len = max_query_len
        self.query_start_loc = query_start_loc
        self.seq_lens = seq_lens
        self.block_table = block_table
        self.slot_mapping = slot_mapping
        self.spec_decode = spec_decode
        self.num_decode_tokens = num_decode_tokens
        self.num_prefill_tokens = num_prefill_tokens
        self.num_computed_tokens = num_computed_tokens
        self.seq_lens_np = seq_lens_np
        self.max_seq_len = max_seq_len
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.causal = causal
        self.sliding_window = sliding_window
        self.token_ids = token_ids
        self.positions = positions
        self.workspace = workspace


class MetaManualSlots:
    __slots__ = ("num_reqs", "num_actual_tokens", "max_query_len", "query_start_loc",
                 "seq_lens", "block_table", "slot_mapping", "spec_decode",
                 "num_decode_tokens", "num_prefill_tokens", "num_computed_tokens",
                 "seq_lens_np", "max_seq_len", "kv_cache_dtype", "block_size",
                 "num_kv_heads", "head_size", "causal", "sliding_window",
                 "token_ids", "positions", "workspace")

    def __init__(self, num_reqs, num_actual_tokens, max_query_len, query_start_loc,
                 seq_lens, block_table, slot_mapping, spec_decode, num_decode_tokens,
                 num_prefill_tokens, num_computed_tokens, seq_lens_np, max_seq_len,
                 kv_cache_dtype, block_size, num_kv_heads, head_size, causal,
                 sliding_window, token_ids, positions, workspace):
        self.num_reqs = num_reqs
        self.num_actual_tokens = num_actual_tokens
        self.max_query_len = max_query_len
        self.query_start_loc = query_start_loc
        self.seq_lens = seq_lens
        self.block_table = block_table
        self.slot_mapping = slot_mapping
        self.spec_decode = spec_decode
        self.num_decode_tokens = num_decode_tokens
        self.num_prefill_tokens = num_prefill_tokens
        self.num_computed_tokens = num_computed_tokens
        self.seq_lens_np = seq_lens_np
        self.max_seq_len = max_seq_len
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.causal = causal
        self.sliding_window = sliding_window
        self.token_ids = token_ids
        self.positions = positions
        self.workspace = workspace


class MetaNamedTuple(NamedTuple):
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: Any
    seq_lens: Any
    block_table: Any
    slot_mapping: Any
    spec_decode: bool
    num_decode_tokens: int
    num_prefill_tokens: int
    num_computed_tokens: Any
    seq_lens_np: Any
    max_seq_len: int
    kv_cache_dtype: str
    block_size: int
    num_kv_heads: int
    head_size: int
    causal: bool
    sliding_window: int
    token_ids: Any
    positions: Any
    workspace: Any


FIELD_ORDER = [
    "num_reqs", "num_actual_tokens", "max_query_len", "query_start_loc", "seq_lens",
    "block_table", "slot_mapping", "spec_decode", "num_decode_tokens",
    "num_prefill_tokens", "num_computed_tokens", "seq_lens_np", "max_seq_len",
    "kv_cache_dtype", "block_size", "num_kv_heads", "head_size", "causal",
    "sliding_window", "token_ids", "positions", "workspace",
]


def _kwargs(inp: dict[str, Any]) -> list:
    return [inp[name] for name in FIELD_ORDER]


def read_back(m) -> int:
    """22 次属性查找 ×2 遍 + 少量算术（模拟 builder 里反复读取字段）。"""
    acc = m.num_reqs + m.num_actual_tokens + m.max_query_len + m.num_decode_tokens
    acc += m.num_prefill_tokens + m.max_seq_len + m.block_size + m.num_kv_heads
    acc += m.head_size + m.sliding_window + int(m.spec_decode) + int(m.causal)
    acc += len(m.kv_cache_dtype) + int(m.query_start_loc[-1]) + int(m.slot_mapping[-1])
    acc += int(m.token_ids[-1]) + int(m.workspace.shape[0])
    acc += m.num_reqs + m.num_actual_tokens + m.max_query_len + m.seq_lens_np[-1]
    acc += int(m.block_table.shape[1]) + int(m.positions[-1]) + int(m.num_computed_tokens[-1])
    acc += int(m.seq_lens[-1]) if hasattr(m.seq_lens, "__getitem__") else 0
    return acc




def run_plain(inp, mode="torch"):
    m = MetaPlain(*_kwargs(inp))
    ops = small_tensor_ops(inp, mode)
    return read_back(m) + int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)


def run_slots(inp, mode="torch"):
    m = MetaSlots(*_kwargs(inp))
    ops = small_tensor_ops(inp, mode)
    return read_back(m) + int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)


def run_manual_dict(inp, mode="torch"):
    m = MetaManualDict(*_kwargs(inp))
    ops = small_tensor_ops(inp, mode)
    return read_back(m) + int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)


def run_manual_slots(inp, mode="torch"):
    m = MetaManualSlots(*_kwargs(inp))
    ops = small_tensor_ops(inp, mode)
    return read_back(m) + int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)


def run_namedtuple(inp, mode="torch"):
    m = MetaNamedTuple(*_kwargs(inp))
    ops = small_tensor_ops(inp, mode)
    return read_back(m) + int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)


def run_dict(inp, mode="torch"):
    m = dict(zip(FIELD_ORDER, _kwargs(inp)))
    ops = small_tensor_ops(inp, mode)
    acc = (m["num_reqs"] + m["num_actual_tokens"] + m["max_query_len"] + m["num_decode_tokens"]
           + m["num_prefill_tokens"] + m["max_seq_len"] + m["block_size"] + m["num_kv_heads"]
           + m["head_size"] + m["sliding_window"] + int(m["spec_decode"]) + int(m["causal"])
           + len(m["kv_cache_dtype"]) + int(m["query_start_loc"][-1]) + int(m["slot_mapping"][-1])
           + int(m["token_ids"][-1]) + int(m["workspace"].shape[0])
           + m["num_reqs"] + m["num_actual_tokens"] + m["max_query_len"] + m["seq_lens_np"][-1]
           + int(m["block_table"].shape[1]) + int(m["positions"][-1])
           + int(m["num_computed_tokens"][-1]) + int(m["seq_lens"][-1]))
    return acc + int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)


VARIANTS = {
    "plain": run_plain,
    "slots": run_slots,
    "manual_dict": run_manual_dict,
    "manual_slots": run_manual_slots,
    "namedtuple": run_namedtuple,
    "dict": run_dict,
}

# 对照臂：复用同一个对象、只改字段（零对象分配），用于验证"分配器"假设
_REUSE_BOX = [None]


def run_plain_reuse(inp, mode="torch"):
    m = _REUSE_BOX[0]
    if m is None:
        m = _REUSE_BOX[0] = MetaPlain(*_kwargs(inp))
    m.num_reqs = 1
    m.num_actual_tokens = 1
    m.max_query_len = 1
    m.query_start_loc = inp["query_start_loc"]
    m.seq_lens = inp["seq_lens"]
    m.block_table = inp["block_table"]
    m.slot_mapping = inp["slot_mapping"]
    m.spec_decode = False
    m.num_decode_tokens = 1
    m.num_prefill_tokens = 0
    m.num_computed_tokens = inp["num_computed_tokens"]
    m.seq_lens_np = inp["seq_lens_np"]
    m.max_seq_len = 8
    m.kv_cache_dtype = "auto"
    m.block_size = 128
    m.num_kv_heads = 8
    m.head_size = 128
    m.causal = True
    m.sliding_window = -1
    m.token_ids = inp["token_ids"]
    m.positions = inp["positions"]
    m.workspace = inp["workspace"]
    ops = small_tensor_ops(inp, mode)
    return read_back(m) + int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)


VARIANTS["plain_reuse"] = run_plain_reuse


# --------------------------------------------------------------------------------------
# 计时驱动
# --------------------------------------------------------------------------------------


def time_variant(fn, iters: int, rounds: int, mode: str = "torch") -> dict[str, float]:
    for _ in range(min(500, iters)):
        fn(INPUTS, mode)
    per_round: list[float] = []
    checksum = 0
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            checksum += int(fn(INPUTS, mode))
        t1 = time.perf_counter_ns()
        per_round.append((t1 - t0) / iters)
    per_round.sort()
    return {
        "median_ns": statistics.median(per_round),
        "min_ns": per_round[0],
        "p10_ns": per_round[max(0, int(0.1 * len(per_round)) - 1)],
        "p90_ns": per_round[min(len(per_round) - 1, int(0.9 * len(per_round)))],
        "checksum": checksum,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--tensor-backend", choices=["torch", "numpy", "none"], default="torch")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.variant not in VARIANTS:
        print(f"unknown variant {args.variant}", file=sys.stderr)
        return 2
    r = time_variant(VARIANTS[args.variant], args.iters, args.rounds, args.tensor_backend)
    out = {
        "label": args.label or args.variant,
        "variant": args.variant,
        "iters": args.iters,
        "rounds": args.rounds,
        "tensor_backend": args.tensor_backend,
        "median_ns": r["median_ns"],
        "min_ns": r["min_ns"],
        "p10_ns": r["p10_ns"],
        "p90_ns": r["p90_ns"],
        "us_per_iter": r["median_ns"] / 1000.0,
        "checksum": r["checksum"],
        "python": sys.version.split()[0],
        "python_impl": getattr(sys, "implementation", None).name if hasattr(sys, "implementation") else "?",
        "numpy": np.__version__,
        "torch": (torch.__version__ if torch is not None else None),
        "gil_enabled": (lambda: getattr(sys, "_is_gil_enabled", lambda: True)())(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
