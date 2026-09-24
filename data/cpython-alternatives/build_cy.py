#!/usr/bin/env python3
"""用 setuptools + Cython 把 cy_variants.pyx 编成 inplace .so（同名：cy_variants）。"""

import os
import shutil
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

MODULES = sys.argv[1:] or ["cy_variants"]

# "原样编译"臂：bench_glue_cy 是 bench_glue.py 的逐字拷贝（不改一行代码），
# 用于量化"把同一份 Python 代码直接交给 Cython 编译"能得到多少。
COPIES = {"bench_glue_cy": "bench_glue.py"}
for _m, _src in COPIES.items():
    if not os.path.exists(f"{_m}.py") or open(_m + ".py").read() != open(_src).read():
        shutil.copyfile(_src, f"{_m}.py")

exts = [
    Extension(
        m,
        [f"{m}.pyx" if os.path.exists(f"{m}.pyx") else f"{m}.py"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
    for m in MODULES
]

setup(
    name="cy_variants",
    ext_modules=cythonize(
        exts,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "cdivision": True,
        },
        force=True,  # 指令变化不会被 cythonize 的增量检查捕获（会被静默跳过），一律强制重编
        quiet=True,
    ),
    script_args=["build_ext", "--inplace"],
)
