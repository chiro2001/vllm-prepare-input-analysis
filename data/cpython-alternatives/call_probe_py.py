"""cy_call_probe.pyx 的纯 Python 对照（同形循环）。"""

import numpy as np
import torch


def call1(f, a, n):
    r = None
    for _ in range(n):
        r = f(a)
    return r


def call_kw(f, a, n):
    r = None
    for _ in range(n):
        r = f(a, dtype=np.int32)
    return r


def np_cumsum_fixed(a, n):
    r = None
    for _ in range(n):
        r = np.cumsum(a)
    return r


def np_zeros_kw(n):
    r = None
    for _ in range(n):
        r = np.zeros(1, dtype=np.int64)
    return r


def torch_from_numpy_fixed(a, n):
    r = None
    for _ in range(n):
        r = torch.from_numpy(a)
    return r


def torch_add_mul_fixed(t, n):
    r = None
    for _ in range(n):
        r = t.mul(2)
    return r
