# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""隔离探针：把"构造 / 读字段 / 算子调用"三段分开计时，定位 Cython 慢在哪。"""

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


cdef class MetaCdef:
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

    cdef long _read(self):
        return (self.num_reqs + self.num_actual_tokens + self.max_query_len
                + self.num_decode_tokens + self.num_prefill_tokens + self.max_seq_len
                + self.block_size + self.num_kv_heads + self.head_size + self.sliding_window
                + <long>self.spec_decode + <long>self.causal + len(self.kv_cache_dtype)
                + last_int(self.query_start_loc) + last_int(self.slot_mapping)
                + last_int(self.token_ids) + self.workspace.shape[0] + last_int(self.seq_lens_np)
                + self.block_table.shape[1] + last_int(self.positions)
                + last_int(self.num_computed_tokens) + last_int(self.seq_lens))


cdef inline long last_int(object arr):
    """wraparound=False 下不能用负索引，用 len()-1 代替（真实改造时必须注意的坑）。"""
    return <long>arr[len(arr) - 1]


cdef list _kwargs(object inp):
    return [inp[n] for n in FIELDS]


def cy_ctor_only(object inp, int n):
    cdef int i
    cdef long acc = 0
    for i in range(n):
        acc += (<MetaCdef>MetaCdef(*_kwargs(inp))).num_reqs
    return acc


def cy_ctor_read(object inp, int n):
    cdef int i
    cdef long acc = 0
    cdef MetaCdef m
    for i in range(n):
        m = MetaCdef(*_kwargs(inp))
        acc += m._read()
    return acc


def cy_ctor_read_pygattr(object inp, int n):
    """cdef 构造 + 用 Python 语义读（模拟"没改属性访问"的编译）。"""
    cdef int i
    cdef long acc = 0
    cdef MetaCdef m
    for i in range(n):
        m = MetaCdef(*_kwargs(inp))
        acc += (m.num_reqs + m.num_actual_tokens + m.max_query_len + m.num_decode_tokens
                + m.num_prefill_tokens + m.max_seq_len + m.block_size + m.num_kv_heads
                + m.head_size + m.sliding_window + int(m.spec_decode) + int(m.causal)
                + len(m.kv_cache_dtype) + last_int(m.query_start_loc) + last_int(m.slot_mapping)
                + last_int(m.token_ids) + int(m.workspace.shape[0]) + last_int(m.seq_lens_np)
                + int(m.block_table.shape[1]) + last_int(m.positions)
                + last_int(m.num_computed_tokens) + last_int(m.seq_lens))
    return acc


def cy_ops(int n, str mode):
    cdef int i
    cdef tuple ops
    cdef long acc = 0
    cdef object seq = np.array([1], dtype=np.int32)
    for i in range(n):
        ops = small_ops(seq, mode)
        acc += int(ops[0]) + int(ops[1]) + int(ops[3]) + int(ops[4])
    return acc


cdef inline tuple small_ops(object seq, str mode):
    cdef object t, t2, ar, cl, cs, fz, ws
    cdef object qsl, delta, zeros, smap
    if mode == "none":
        return (0, 0, 0.0, 0, 0.0, 0.0, True, 0)
    qsl = np.cumsum(seq)
    delta = seq[1:] - seq[:-1]
    zeros = np.zeros(1, dtype=np.int64)
    smap = np.arange(1, dtype=np.int32)
    if mode == "numpy":
        t = np.asarray(seq)
        t2 = t * 2
        ar = np.arange(1, dtype=np.int32)
        cl = t.copy()
        return (last_int(qsl), int(delta.sum()), float(zeros.sum()), last_int(smap),
                float(last_int(seq)), float(t2.sum()), bool((t > 0).any()), 0)
    t = torch.from_numpy(np.asarray(seq))
    t2 = t.mul(2)
    ar = torch.arange(1, dtype=torch.int32)
    cl = t.clone()
    cs = torch.cumsum(t, 0)
    fz = torch.zeros(1, dtype=torch.float32)
    ws = np.zeros(2, dtype=np.float32)
    return (last_int(qsl), int(delta.sum()), float(zeros.sum()), last_int(smap),
            float(last_int(seq)), float(t2.sum()), bool((t > 0).any()),
            last_int(cs) + last_int(ar) + int(cl.max()) + int(fz.shape[0]) + int(ws.shape[0]))
