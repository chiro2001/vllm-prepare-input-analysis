# cython: language_level=3, boundscheck=False, cdivision=True
"""调用开销探针：Cython 编译出来的代码 vs CPython 字节码，调同一个 C 扩展函数/算子的成本。

动机：bench_glue_cy（原样 Cython 编译）整体比纯 Python 慢 ~2x，需要确认是
"编译后调 Python/C 函数变慢"还是别的原因。
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def call1(object f, object a, int n):
    """n 次单参调用 f(a)。"""
    cdef int i
    cdef object r
    for i in range(n):
        r = f(a)
    return r


def call_kw(object f, object a, int n):
    """n 次带关键字调用 f(a, dtype=...)：每个 C 扩展小算子都是这个形状。"""
    cdef int i
    cdef object r
    for i in range(n):
        r = f(a, dtype=np.int32)
    return r


def np_cumsum_fixed(object a, int n):
    """模块级 numpy 全局 + 固定调用点（与真实代码同形）。"""
    cdef int i
    cdef object r
    for i in range(n):
        r = np.cumsum(a)
    return r


def np_zeros_kw(int n):
    cdef int i
    cdef object r
    for i in range(n):
        r = np.zeros(1, dtype=np.int64)
    return r


def torch_from_numpy_fixed(object a, int n):
    cdef int i
    cdef object r
    for i in range(n):
        r = torch.from_numpy(a)
    return r


def torch_add_mul_fixed(object t, int n):
    cdef int i
    cdef object r
    for i in range(n):
        r = t.mul(2)
    return r
