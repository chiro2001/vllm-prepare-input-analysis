"""mypyc 臂：同一个负载的 mypyc 写法。

两个版本：
  * run_mypyc_native    —— mypyc native class（标量字段走 C 结构体）
  * run_mypyc_dataclass —— @dataclass（mypyc 对 dataclass 只有 partial native support）

注意：numpy/torch 对象在 mypyc 里是"外部类型"，只能按 erased(Any) 处理，
调用它们仍然是普通 Python 调用 —— 这正是要测的东西。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

FIELDS = [
    "num_reqs", "num_actual_tokens", "max_query_len", "query_start_loc", "seq_lens",
    "block_table", "slot_mapping", "spec_decode", "num_decode_tokens",
    "num_prefill_tokens", "num_computed_tokens", "seq_lens_np", "max_seq_len",
    "kv_cache_dtype", "block_size", "num_kv_heads", "head_size", "causal",
    "sliding_window", "token_ids", "positions", "workspace",
]


class MetaNative:
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

    def __init__(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
                 query_start_loc: Any, seq_lens: Any, block_table: Any, slot_mapping: Any,
                 spec_decode: bool, num_decode_tokens: int, num_prefill_tokens: int,
                 num_computed_tokens: Any, seq_lens_np: Any, max_seq_len: int,
                 kv_cache_dtype: str, block_size: int, num_kv_heads: int, head_size: int,
                 causal: bool, sliding_window: int, token_ids: Any, positions: Any,
                 workspace: Any) -> None:
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


@dataclass
class MetaDataclass:
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


def _kwargs(inp: dict[str, Any]) -> list[Any]:
    return [inp[n] for n in FIELDS]


def small_ops(inp: dict[str, Any], mode: str = "torch") -> tuple[Any, ...]:
    if mode == "none":
        return (0, 0, 0, 0, 0, 0, True, 0)
    seq: Any = inp["seq_lens_np"]
    qsl: Any = np.cumsum(seq)
    delta: Any = seq[1:] - seq[:-1]
    zeros: Any = np.zeros(1, dtype=np.int64)
    smap: Any = np.arange(1, dtype=np.int32)
    last: float = float(seq[-1])
    if mode == "numpy":
        t: Any = np.asarray(seq)
        t2: Any = t * 2
        s: float = float(t2.sum())
        ar: Any = np.arange(1, dtype=np.int32)
        cl: Any = t.copy()
        cs: Any = np.cumsum(t, axis=0)
        fz: Any = np.zeros(1, dtype=np.float32)
        ws: Any = inp["workspace"][:2]
        return (int(qsl[-1]), int(delta.sum()), int(zeros.sum()), int(smap[-1]), last, s,
                bool((t > 0).any()),
                int(cs[-1]) + int(ar[-1]) + int(cl.max()) + int(fz.shape[0]) + int(ws.shape[0]))
    t = torch.from_numpy(seq)
    t2 = t.mul(2)
    s = float(t2.sum())
    ar = torch.arange(1, dtype=torch.int32)
    cl = t.clone()
    cs = torch.cumsum(t, 0)
    fz = torch.zeros(1, dtype=torch.float32)
    ws = inp["workspace"][:2]
    return (int(qsl[-1]), int(delta.sum()), int(zeros.sum()), int(smap[-1]), last, s,
            bool((t > 0).any()),
            int(cs[-1]) + int(ar[-1]) + int(cl.max()) + int(fz.shape[0]) + int(ws.shape[0]))


def _read_scalars(m: Any) -> int:
    acc: int = (m.num_reqs + m.num_actual_tokens + m.max_query_len + m.num_decode_tokens
                + m.num_prefill_tokens + m.max_seq_len + m.block_size + m.num_kv_heads
                + m.head_size + m.sliding_window + int(m.spec_decode) + int(m.causal))
    return acc


def _read_containers(m: Any) -> int:
    acc: int = len(m.kv_cache_dtype)
    acc += int(m.query_start_loc[-1])
    acc += int(m.slot_mapping[-1])
    acc += int(m.token_ids[-1])
    acc += int(m.workspace.shape[0])
    acc += m.num_reqs + m.num_actual_tokens + m.max_query_len
    acc += int(m.seq_lens_np[-1])
    acc += int(m.block_table.shape[1])
    acc += int(m.positions[-1])
    acc += int(m.num_computed_tokens[-1])
    acc += int(m.seq_lens[-1])
    return acc


def run_mypyc_native(inp: dict[str, Any], mode: str = "torch") -> int:
    m: MetaNative = MetaNative(*_kwargs(inp))
    ops: tuple[Any, ...] = small_ops(inp, mode)
    acc: int = _read_scalars(m) + _read_containers(m)
    acc += int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)
    return acc


def run_mypyc_dataclass(inp: dict[str, Any], mode: str = "torch") -> int:
    m: MetaDataclass = MetaDataclass(*_kwargs(inp))
    ops: tuple[Any, ...] = small_ops(inp, mode)
    acc: int = _read_scalars(m) + _read_containers(m)
    acc += int(ops[0]) + int(ops[1]) + int(ops[2]) + int(ops[3]) + int(ops[4]) + len(ops)
    return acc
