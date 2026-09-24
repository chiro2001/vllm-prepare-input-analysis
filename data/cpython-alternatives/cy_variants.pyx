# cython: language_level=3, boundscheck=False, cdivision=True
"""Cython 臂：同一个"22 字段 metadata + 15 次小算子 + 属性查找"负载的三种写法。

文件头只关掉 boundscheck（这是我们想量的优化之一）；wraparound 保持默认 True ——
因为 `arr[-1]` 在 wraparound=False 下是未定义行为，Cython 会直接给出警告，
真改造时必须把负索引改写掉（见 docs/10-cpython-directions/04-runtime-alternatives.md）。

  * run_cy_naive —— 最小改造：逻辑照抄，容器还是普通 Python 类
  * run_cy_cdef  —— cdef class：标量字段变成 C int/bint（无 refcount、无 dict 查找）
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None

FIELDS = [
    "num_reqs", "num_actual_tokens", "max_query_len", "query_start_loc", "seq_lens",
    "block_table", "slot_mapping", "spec_decode", "num_decode_tokens",
    "num_prefill_tokens", "num_computed_tokens", "seq_lens_np", "max_seq_len",
    "kv_cache_dtype", "block_size", "num_kv_heads", "head_size", "causal",
    "sliding_window", "token_ids", "positions", "workspace",
]


class MetaPython:
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


cdef class MetaCdef:
    """所有标量字段是 C int/bint：无 Python 对象、无 refcount、无 dict 查找。"""
    cdef int num_reqs
    cdef int num_actual_tokens
    cdef int max_query_len
    cdef int num_decode_tokens
    cdef int num_prefill_tokens
    cdef int max_seq_len
    cdef int block_size
    cdef int num_kv_heads
    cdef int head_size
    cdef int sliding_window
    cdef bint spec_decode
    cdef bint causal
    cdef object query_start_loc
    cdef object seq_lens
    cdef object block_table
    cdef object slot_mapping
    cdef object num_computed_tokens
    cdef object seq_lens_np
    cdef object kv_cache_dtype
    cdef object token_ids
    cdef object positions
    cdef object workspace

    def __init__(self, int num_reqs, int num_actual_tokens, int max_query_len,
                 object query_start_loc, object seq_lens, object block_table,
                 object slot_mapping, bint spec_decode, int num_decode_tokens,
                 int num_prefill_tokens, object num_computed_tokens, object seq_lens_np,
                 int max_seq_len, object kv_cache_dtype, int block_size,
                 int num_kv_heads, int head_size, bint causal, int sliding_window,
                 object token_ids, object positions, object workspace):
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

    cdef long read(self):
        return (self.num_reqs + self.num_actual_tokens + self.max_query_len
                + self.num_decode_tokens + self.num_prefill_tokens + self.max_seq_len
                + self.block_size + self.num_kv_heads + self.head_size + self.sliding_window
                + <long>self.spec_decode + <long>self.causal + len(self.kv_cache_dtype)
                + self.query_start_loc[-1] + self.slot_mapping[-1] + self.token_ids[-1]
                + self.workspace.shape[0] + self.num_reqs + self.num_actual_tokens
                + self.max_query_len + self.seq_lens_np[-1] + self.block_table.shape[1]
                + self.positions[-1] + self.num_computed_tokens[-1] + self.seq_lens[-1])


cdef inline tuple small_ops(object inp, str mode):
    cdef object seq, t, t2, ar, cl, cs, fz, ws
    cdef object qsl, delta, zeros, smap
    cdef long result0, result3
    cdef double last, s
    cdef bint any_pos
    cdef long tail
    if mode == "none":
        return (0, 0, 0, 0, 0, 0, True, 0)
    seq = inp["seq_lens_np"]
    qsl = np.cumsum(seq)
    delta = seq[1:] - seq[:-1]
    zeros = np.zeros(1, dtype=np.int64)
    smap = np.arange(1, dtype=np.int32)
    last = <double>seq[-1]
    if mode == "numpy":
        t = np.asarray(seq)
        t2 = t * 2
        s = float(t2.sum())
        ar = np.arange(1, dtype=np.int32)
        cl = t.copy()
        any_pos = bool((t > 0).any())
        cs = np.cumsum(t, axis=0)
        fz = np.zeros(1, dtype=np.float32)
        ws = inp["workspace"][:2]
        tail = <long>cs[-1] + <long>ar[-1] + <long>cl.max() + fz.shape[0] + ws.shape[0]
        return (<long>qsl[-1], <long>delta.sum(), <long>zeros.sum(), <long>smap[-1],
                last, s, any_pos, tail)
    t = torch.from_numpy(seq)
    t2 = t.mul(2)
    s = float(t2.sum())
    ar = torch.arange(1, dtype=torch.int32)
    cl = t.clone()
    any_pos = bool((t > 0).any())
    cs = torch.cumsum(t, 0)
    fz = torch.zeros(1, dtype=torch.float32)
    ws = inp["workspace"][:2]
    tail = int(cs[-1]) + int(ar[-1]) + int(cl.max()) + int(fz.shape[0]) + int(ws.shape[0])
    return (int(qsl[-1]), int(delta.sum()), int(zeros.sum()), int(smap[-1]),
            <double>last, s, any_pos, tail)


cdef list _kwargs(object inp):
    return [inp[n] for n in FIELDS]


cdef inline long agg(tuple ops):
    """所有臂共用的聚合表达式：ops[0]+[1]+[2]+[3]+[4] + len(ops)。"""
    return (<long>ops[0] + <long>ops[1] + <long>ops[2] + <long>ops[3] + <long>ops[4]
            + len(ops))


def run_cy_naive(object inp, str mode="torch"):
    """最小改造：容器还是 Python 类，只有函数体被编译。"""
    cdef object m = MetaPython(*_kwargs(inp))
    cdef tuple ops = small_ops(inp, mode)
    cdef object acc = (m.num_reqs + m.num_actual_tokens + m.max_query_len + m.num_decode_tokens
                       + m.num_prefill_tokens + m.max_seq_len + m.block_size + m.num_kv_heads
                       + m.head_size + m.sliding_window + int(m.spec_decode) + int(m.causal)
                       + len(m.kv_cache_dtype) + int(m.query_start_loc[-1])
                       + int(m.slot_mapping[-1]) + int(m.token_ids[-1])
                       + int(m.workspace.shape[0]) + m.num_reqs + m.num_actual_tokens
                       + m.max_query_len + int(m.seq_lens_np[-1]) + int(m.block_table.shape[1])
                       + int(m.positions[-1]) + int(m.num_computed_tokens[-1])
                       + int(m.seq_lens[-1]))
    return <long>acc + agg(ops)


def run_cy_cdef(object inp, str mode="torch"):
    """类型化改造：标量进 C 结构体，属性读取在 C 层完成。"""
    cdef MetaCdef m = MetaCdef(*_kwargs(inp))
    cdef tuple ops = small_ops(inp, mode)
    return m.read() + agg(ops)


def run_cy_cdef_notensor(object inp):
    """对照臂：只构造 + 读字段，不做任何张量/数组算子。"""
    cdef MetaCdef m = MetaCdef(*_kwargs(inp))
    return m.read()
