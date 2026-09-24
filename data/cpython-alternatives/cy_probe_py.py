"""cy_probe.pyx 的纯 Python 对照（同样的三段隔离逻辑）。"""

from __future__ import annotations

import numpy as np

from bench_glue import FIELD_ORDER as FIELDS, MetaManualSlots, small_tensor_ops


def py_ctor_only(inp, n):
    acc = 0
    k = [inp[x] for x in FIELDS]
    for _ in range(n):
        m = MetaManualSlots(*k)
        acc += m.num_reqs
    return acc


def py_ctor_read(inp, n):
    acc = 0
    k = [inp[x] for x in FIELDS]
    for _ in range(n):
        m = MetaManualSlots(*k)
        acc += (m.num_reqs + m.num_actual_tokens + m.max_query_len + m.num_decode_tokens
                + m.num_prefill_tokens + m.max_seq_len + m.block_size + m.num_kv_heads
                + m.head_size + m.sliding_window + int(m.spec_decode) + int(m.causal)
                + len(m.kv_cache_dtype) + int(m.query_start_loc[-1]) + int(m.slot_mapping[-1])
                + int(m.token_ids[-1]) + int(m.workspace.shape[0]) + int(m.seq_lens_np[-1])
                + int(m.block_table.shape[1]) + int(m.positions[-1])
                + int(m.num_computed_tokens[-1]) + int(m.seq_lens[-1]))
    return acc


def py_ops(n, mode, seq=None):
    if seq is None:
        seq = np.array([1], dtype=np.int32)
    acc = 0
    for _ in range(n):
        ops = small_tensor_ops({"seq_lens_np": seq, "workspace": np.zeros(2, dtype=np.float32)},
                               mode)
        acc += int(ops[0]) + int(ops[1]) + int(ops[3]) + int(ops[4])
    return acc
