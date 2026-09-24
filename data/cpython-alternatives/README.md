# cpython-alternatives 数据与脚本

本目录是 `docs/10-cpython-directions/04-runtime-alternatives.md` 的配套产物。两套正式口径：

* **`cross-ab-*.json`（跨配置比较，正式）**：由 `cross_interp_ab.py` 产生，**交替多轮取中位数**，
  用于比较不同解释器 / JIT / GIL / 分配器。这台开发机是 KVM 虚机，同一命令跨批次绝对值可差 2×
  （同一个 cp314：1.82 µs vs 3.80 µs），所以跨二进制比较**必须**用这个口径。
* **`inproc-*.csv`（同一进程内的实现比较，正式）**：由 `bench_all_inproc.py` 产生，
  用于比较容器写法与 Cython/mypyc 编译臂；绝对值只在同一会话内可比。
* `s2-*.csv` / `matrix-*.csv` / `compiled-cp312.csv`：早期单批次或子进程口径，**只作对照，不建议引用**。

## 环境（2026-09-24，开发机 server-mini / x86_64 / Arch）

| 用途 | 路径 | 版本 |
|---|---|---|
| 主环境 | `.venv`（`--system-site-packages`，基于 conda-forge python 3.12.14） | Cython 3.3.0、mypy 2.3.1、numpy 2.5.3、torch 2.13.0+cpu |
| 版本对照 | `~/miniforge3/envs/.env313` / `.env314` / `.env314t` | CPython 3.13.15 / 3.14.7 / 3.14.7t，numpy 均为 2.5.3 |
| 编译器 | 系统 `gcc`（Cython/mypyc 均用 `-O3`） | — |

四个解释器都是 conda-forge 构建：`--enable-optimizations --with-lto=full --with-computed-gotos`；
`.env314` 还带 `--enable-experimental-jit=yes-off`（用 `PYTHON_JIT=1` 打开 JIT）；
`.env314t` 是 free-threaded 构建（`--disable-gil`，强制 mimalloc），用 `PYTHON_GIL=1|0` 切换 GIL（同一份二进制，A/B 最干净）。
这些 conda env 是本次调研临时建的，可用 `mamba env remove -p <prefix>` 清理。

## 负载定义

每轮 = 构造 22 字段元数据对象（dataclass / `slots=True` / 手写普通类 / 手写 `__slots__` / NamedTuple / dict，
外加一个**复用同一对象、只改字段**的零分配对照臂 `plain_reuse`）
+ 15 次 B=1 的小算子调用 + 25 次属性读取。三种张量后端：`torch`（真实形状）、`numpy`（同 15 步换 numpy）、
`none`（剥掉算子，只留胶水）。

## 一键复现

```bash
# 臂内（同一会话内比较容器写法与编译臂）
for be in torch numpy none; do it=1200; [ $be != torch ] && it=5000; \
  taskset -c 4-7 .venv/bin/python bench_all_inproc.py --backend $be --iters $it --rounds 9 \
    --python-label cp312 --out inproc-cp312-$be.csv; done
.venv/bin/python build_cy.py cy_variants cy_probe bench_glue_cy
.venv/bin/python -m mypyc mypyc_variant.py
# 跨解释器 / JIT / GIL / 分配器（交替多轮，跨二进制唯一可信口径）
taskset -c 4-7 .venv/bin/python cross_interp_ab.py --backend none  --reps 9 --out cross-ab-none.json
taskset -c 4-7 .venv/bin/python cross_interp_ab.py --backend numpy --reps 5 --out cross-ab-numpy.json
taskset -c 4-7 .venv/bin/python cross_interp_ab.py --backend torch --reps 3 --iters 800 --out cross-ab-torch.json
taskset -c 4-7 .venv/bin/python call_probe_compare.py     # 单次调用开销对照
taskset -c 4-7 .venv/bin/python probe_compare.py          # 构造/读/算子三段隔离
```
