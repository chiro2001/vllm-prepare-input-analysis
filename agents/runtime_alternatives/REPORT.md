# runtime_alternatives —— Python 运行时替代方案与编译方案调研（工作记录）

> 交付物：`docs/10-cpython-directions/04-runtime-alternatives.md`（正式文档）
> 数据/脚本：`data/cpython-alternatives/`
> 时间：2026-09-24。**没有登录 a3-22/a3-21、没有碰 chip3、没有起容器、没有跑 NPU；所有实验都在开发机 `server-mini`（x86_64, 12 核）本地完成。**

## 1. 做了什么

1. **文献/网络调研**（exa + 官方文档）：Cython、mypyc、Nuitka、PyO3/C++（含 vLLM/Dynamo/TRT-LLM/FlashInfer-SGLang 先例）、
   free-threading（PEP 703/779）、subinterpreter（PEP 684/734）、PyPy/GraalPy、CPython 3.13/3.14/3.15 JIT（含 PEP 836）。
2. **本地微基准**（可复现，10 分钟量级）：合成"22 字段 metadata + 15 次小 torch/numpy 算子 + 25 次属性读"负载，
   对比 6 种容器实现 × 3 种张量后端 × 8 个"编译/解释器/构建"配置；
   臂内用同进程轮转，**跨解释器用 `cross_interp_ab.py` 交替多轮取中位**。
3. **`cython.inline` 实测**：验证其缓存目录、编译行为与调用开销（结论：不可用于热路径）。
4. **两个追加实验**：3.14 的 JIT（`PYTHON_JIT=1`）、`PYTHONMALLOC=malloc`、以及"零分配复用容器"对照臂。

## 2. 关键结论（带数字）

| 结论 | 数字 | 证据 |
|---|---|---|
| 成本结构：算子在整段里占绝对多数 | torch 后端 30.95 µs/iter，其中 15 次小算子 **27.2 µs（88%）**，纯胶水 3.76 µs | `inproc-cp312-torch.csv` / `inproc-cp312-none.csv` |
| 编译只作用于胶水，且胶水已接近 C 速度 | Cython 原样编译：纯胶水 **-12%**（3.76→3.32）；完整 torch 负载 **+8.9%**（变慢） | 同上 |
| mypyc 同上 | native class 纯胶水 -14%、完整负载 +17%；`@dataclass` 纯胶水 -8.5%、完整负载 +17% | 同上 |
| 语法级改造是噪声 | 6 种容器在同后端内差异 ≤±5%；完整负载 <1.5% | `inproc-cp312-*.csv` |
| 换 numpy 比编译有用得多 | 同样 15 步：torch 27.2 µs vs numpy 9.6 µs（**-65%**） | 同上 |
| 升级 CPython 收益很小 | 交替多轮中位数：3.12→3.14 纯胶水 **-3.7%**、numpy 负载 -2.9%、torch 负载 -7.2%；3.13 与 3.12 打平 | `cross-ab-*.json` |
| 3.14 JIT 对本负载 ≈0 | `PYTHON_JIT=1` vs 不开：胶水 **+2.7%（变慢）**、numpy -1.5%、torch -0.2% | `cross-ab-none/numpy/torch.json` |
| 关 GIL 对单线程零收益 | 同一份 3.14.7t：`PYTHON_GIL=1` vs `=0` → 2.19 vs 2.20 µs（胶水）、11.72 vs 11.82（numpy） | 同上 |
| **未解观察：FT 构建快 1.75×** | 3.14t 构建（2.19）vs 普通 3.14（3.68）；与 GIL 开关无关，零分配对照臂里优势仍有 2.26× ⇒ **不是分配器**，根因未定位 | `cross-ab-none.json`、`cross-ab-none-reuse.json` |
| 每步"别新建对象"值 12–15% | 零分配复用容器臂：cp312 -12%、3.14 -14%（FT 构建 -34%） | `cross-ab-none-reuse.json` |
| pymalloc 好于 glibc malloc | `PYTHONMALLOC=malloc` 让胶水慢 5.8%（cp312）/ 9.8%（cp314） | `cross-ab-none.json` |

## 3. 踩到的坑（后续同学请务必看）

1. **跨进程计时不可信，而且这台机器是 KVM 虚机、跨批次能差 2×**：同一个解释器、同一条命令，
   11:32 批次纯胶水 1.82 µs、11:44 批次 3.80 µs（都无并发）。我们因此把跨解释器比较改成
   **`cross_interp_ab.py` 交替多轮取中位**，并且**放弃了一个原本会写进报告的错误结论**
   （"3.12→3.13 纯胶水快 2.0×"——那是批次漂移，交替口径下只有 1.6%）。
2. **Cython `wraparound=False` + 负索引 = 未定义行为**：同一份 `.so`，一次运行正常、一次在 `seq[-1]` 抛
   `IndexError: index -2 is out of bounds for axis 0 with size 1`。要开 `wraparound=False` 就必须把负索引改写成 `len()-1`。
3. **`cythonize` 的增量检查不跟踪 `compiler_directives`**：改完指令它静默跳过重编（`.so` 不变），必须 `force=True`。
4. **`cython.inline` 的调用开销是 3049 ns/call**（同进程纯 Python `def` 为 24 ns/call）：缓存命中后仍然很贵，
   而且会在 `~/.cython/inline/` 落地 `.pyx/.c/.so`、需要运行时 C 编译器、无法直接读调用者局部变量。
5. **mypyc 首次编译 93 s**（含 `mypyc/lib-rt`）、产物 436 KB；Cython 单模块 build 3.7 s、产物 176 KB（`.c` 736 KB）。
6. `cdef class` 把 22 个字段做成 C 结构体，在"字段是外部对象（torch/np）"的场景**反而更慢**（构造时逐参数
   类型转换 + 丢特化），实测完整负载 +37%、纯胶水 +102%。**"类型化改造不是免费的"。**

## 4. 环境

- 开发机 `server-mini`：x86_64、AMD Eng Sample（KVM）、12 逻辑核、Arch Linux、内核 7.1.5；实验绑定 `taskset -c 4-7`。
- 解释器：CPython 3.12.14 / 3.13.15 / 3.14.7 / 3.14.7t（**均为 conda-forge 同源构建**，numpy 全为 2.5.3）。
  构建参数：`--enable-optimizations --with-lto=full --with-computed-gotos`（PGO+LTO+computed goto 全开）；
  3.14 另有 `--enable-experimental-jit=yes-off`（JIT 编入、默认关，`PYTHON_JIT=1` 可开）；
  3.14t 为 `--disable-gil`（强制 mimalloc，`WITH_MIMALLOC=1`）。
  **注意**：生产镜像里的 Python 是默认参数构建（`USE_COMPUTED_GOTOS=0`、无 PGO，见 `docs/10-cpython-directions/00-FINDING-build-config.md`），
  所以本报告测的是"构建修好之后还剩什么"。
  - 主 venv：`data/cpython-alternatives/.venv`（`--system-site-packages`，Cython 3.3.0、mypy 2.3.1、torch 2.13.0+cpu）
  - 版本对照 env：`~/miniforge3/envs/.env313|.env314|.env314t`（用 `mamba create -p … python=3.13/3.14` 与
    `python-freethreading=3.14` 建的；`.env314t` 里额外装了 `torch==2.13.0+cpu`）。
    > 这三个 env 是本次调研新建的临时环境，可随时用 `mamba env remove -p <prefix>` 清掉。
- 与真机差异：真机是 aarch64 920B + torch 2.9.0+cpu(Ascend) + numpy 2.4.6。**绝对数不可外推**，只用比值与机制。

## 5. 文件清单（`data/cpython-alternatives/`）

| 文件 | 说明 |
|---|---|
| `bench_glue.py` | 负载定义（7 种容器/写法 × 3 种张量后端，含零分配复用臂 `plain_reuse`） |
| `bench_all_inproc.py` | 臂内计时脚本（同进程轮转，输出 CSV） |
| `cross_interp_ab.py` | **跨解释器/JIT/GIL/分配器计时脚本**（交替多轮取中位，唯一可信的跨二进制口径） |
| `cy_variants.pyx` / `build_cy.py` | Cython 臂（原样编译 / 最小改造 / `cdef` 类型化），`force=True` 重编 |
| `mypyc_variant.py` | mypyc 臂（native class + dataclass） |
| `probe_compare.py` / `cy_probe.pyx` | 隔离探针（构造 / 读字段 / 算子三段分开） |
| `call_probe_compare.py` / `cy_call_probe.pyx` | 单次调用开销对照（证明"单次持平、整段变慢"） |
| `run_matrix.py` / `run_ext_matrix.py` | 子进程矩阵驱动（早期方法，保留作对照） |
| `cross-ab-*.json` | **正式数据**（跨配置交替多轮；含 none/numpy/torch 与零分配对照） |
| `inproc-*.csv` | 同会话臂内数据（表 1/表 2 的来源；绝对值不可跨会话比较） |
| `s2-*.csv` / `matrix-*.csv` / `compiled-cp312.csv` | 单批次/子进程口径（保留作对照，**不建议引用**） |

## 6. 复现命令

```bash
cd data/cpython-alternatives
for be in torch numpy none; do it=1200; [ $be != torch ] && it=5000; \
  taskset -c 4-7 .venv/bin/python bench_all_inproc.py --backend $be --iters $it --rounds 9 \
    --python-label cp312 --out inproc-cp312-$be.csv; done
.venv/bin/python build_cy.py cy_variants cy_probe bench_glue_cy   # 约 15 s
.venv/bin/python -m mypyc mypyc_variant.py                       # 首次约 93 s
taskset -c 4-7 .venv/bin/python cross_interp_ab.py --backend none  --reps 9 --out cross-ab-none.json
taskset -c 4-7 .venv/bin/python cross_interp_ab.py --backend numpy --reps 5 --out cross-ab-numpy.json
taskset -c 4-7 .venv/bin/python call_probe_compare.py
```

## 7. 未解 / 待补

1. **FT 构建快 1.75× 的根因**：排除了 GIL（开关打平）与分配器（零分配臂里仍有 2.26×），
   剩下"FT 构建的对象布局/inline values"或"二进制布局"两种可能，都没验证。建议单独开 10 分钟实验：
   把"22 字段对象构造 / 只读 / 只写"拆成三个纯 Python 微基准，在 3.14 与 3.14t 上交替跑。
2. **GIL 构建能否单独启用 mimalloc**：官方只说 `--without-mimalloc` 不能与 `--disable-gil` 共存，
   没验证 `--with-mimalloc` 能否用于普通构建（若能，可能是一条构建级"免费"收益）。
3. **本机微基准跨批次差 2×（1.82 vs 3.80 µs）**：没定位是 vCPU 放置/邻居还是二进制布局，
   建议后续用 `perf stat`（instructions/cycle）而不是墙钟复核。
4. 没有做 aarch64 上的同类微基准（开发机是 x86_64；真机复测需要一次 10 分钟级的 A/B，建议排在 L0 优化之后）。
5. 没有做"Cython 最优写法"的上限探索（把字段降级为局部变量 / 换 tuple / 显式管理张量生命周期），
   因此本报告只能断言"原样/浅改造的编译没有净收益"。
6. 920B 上 numpy vs torch 的**逐算子**替代收益表缺失（本仓只有零散点：`np.cumsum` 2.6 µs、torch ATen 1.6 µs）。
7. vLLM 侧把 per-step metadata 搬到 C++/Rust 的公开先例没找到（同生态里 FlashInfer/SGLang、Dynamo、TRT-LLM 都有）。
