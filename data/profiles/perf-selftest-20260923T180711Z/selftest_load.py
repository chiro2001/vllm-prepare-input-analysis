#!/usr/bin/env python3
"""perf_capture self-test load: single-thread CPU-bound, CPython+numpy+torch mix.

Mimics the *shape* of prepare_input's CPU work: lots of small python objects
(dict / attribute / string), a few array ops, a few torch calls -- all on ONE
thread so a per-TID perf capture sees exactly one CPU sink.
"""
import os
import sys
import time

import numpy as np
import torch

DUR = float(os.environ.get("LOAD_SECONDS", "20"))
torch.set_num_threads(1)


class Rec:
    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a, b, c, d

    def score(self):
        return self.a * 1.7 + self.b + len(self.c) + len(self.d)


def build(n):
    d = {}
    for i in range(n):
        d[i] = Rec(i, i * 1.5, (i, i + 1), "s%d" % i)
    return d


def scan(d):
    s = 0.0
    for k, v in d.items():
        s += v.score() + (k & 7)
    return s


def numpy_part():
    a = np.arange(2048, dtype=np.float32)
    acc = 0.0
    for _ in range(120):
        b = a * 1.000001 + 0.5
        acc += float(b.sum())
    return acc


def torch_part():
    x = torch.randn(128, 128, dtype=torch.float32)
    acc = 0.0
    for _ in range(30):
        y = torch.mm(x, x)
        acc += float(y.sum())
    return acc


def main():
    t0 = time.time()
    it = 0
    while time.time() - t0 < DUR:
        scan(build(6000))
        numpy_part()
        torch_part()
        it += 1
    print("iterations=%d wall=%.2fs" % (it, time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.exit(main())
