# CPython 运行时替代方案与编译方案调研

> 调研对象：vLLM 0.26.0 + vllm-ascend 0.26.0rc1 的 `prepare_input` 阶段（Kunpeng 920B / aarch64 / openEuler + Python 3.12.13）
> 结论口径：**不换解释器骨架**（必须留下 CPython 生态：torch / torch_npu / vLLM Python API；生产镜像不能改解释器本身）
> 文档版本：2026-09-24（Asia/Shanghai）；所有链接访问日期均为 2026-09-24
> 配套数据与脚本：`../../data/cpython-alternatives/`（含原始 CSV、基准脚本、编译脚本）

---

## 0. 一页结论

1. **编译类方案（Cython / mypyc / Nuitka）对这条负载的净收益 ≈ 0，实测甚至是负的。**
   原因不是"编译器不行"，而是**成本结构**：本负载的 **88%** 花在 15 次小张量算子的 **派发**上（本机实测 27.2 µs / 31.0 µs），
   编译器碰不到；剩下的 12%（22 字段构造 + 25 次属性读）在 CPython 3.12 上已经被 specializing interpreter 压到接近 C 的速度
   （同样的读字段逻辑，Cython 编译后只快 12–19%，但整段负载反而慢 8–17%）。
2. **语法级改造（`dataclass(slots=True)` / 手写 `__slots__` / NamedTuple / dict）在本负载上是噪声级**：
   同一后端内 6 种容器实现差异只有 **±5%**；换成整段负载（含 torch 派发）差异 **<1.5%**。
3. **真正能动收益的是"减少/替换调用条数"，不是"编译现有代码"**：同样 15 步小算子，
   torch 后端 27.2 µs、numpy 后端 9.6 µs（**-65%**），差距全在派发而非计算。
4. **升级 CPython 小版本（3.12→3.13/3.14）在这条负载上收益很小**：同一 conda-forge 构建链、交替多轮取中位后，
   纯胶水 3.12→3.14 只快 **3.7%**，numpy 算子负载 **2.9%**，含 torch 派发的完整负载 **7%（3 轮，噪声较大）**。
   3.13 与 3.12 基本持平。**"升级 Python 能省三分之一"是错的**——它只对"纯 Python 逻辑占绝对多数"的负载成立。
5. **3.14 的 JIT（`PYTHON_JIT=1`，本机 conda-forge 构建已编入）实测对这条负载 ≈0**：
   纯胶水 +1%（变慢）、numpy 负载 -1.5%、torch 负载 -0.2%，全部落在 ±2% 噪声内。
6. **PEP 703 free-threading 的"关 GIL"本身对单线程没有任何收益**：同一份 3.14.7t 二进制，`PYTHON_GIL=1` 与 `=0`
   在纯胶水上 **2.19 vs 2.20 µs**（打平）、numpy 负载 **11.72 vs 11.82 µs**（打平）。
   ⚠️ 但注意一个**未解释的观察**：这个 free-threaded **构建**在我们的纯 Python 负载上比普通构建快 **1.75×**，
   而且**不是**分配器或 GIL 造成的（详见 §2.6）——这条线索值得单独追，不能当成"换 FT 构建就能提速"的结论。
7. **零分配对照臂**（复用同一个容器、只改字段）在 cp312 上省 **12%**、3.14 上省 **14%**、FT 构建上省 **34%**：
   "每步别新建对象"是一个可落地的代码级杠杆（§2.6）。
8. **先看构建参数，再谈换运行时**：本仓 `00-FINDING-build-config.md` 已核实**生产镜像里的 Python 是用默认参数编的
   （`USE_COMPUTED_GOTOS=0`，无 PGO/LTO）**；本报告所有开发机解释器都是 PGO+LTO+computed-goto 的"最优构建"，
   也就是说本报告测的是"**把构建参数修好之后**还剩下什么给编译/换运行时"——答案是几乎没有。
9. **PEP 684/734 subinterpreter 不适用**：它的收益只在多核并行；隔离解释器要求 C 扩展走多阶段初始化（PEP 489），
   torch 不满足，且每个解释器都要重新 import。
10. **PyPy / GraalPy 明确不能用**：PyTorch 从未支持 PyPy（issue #17835 开放至今，PyPy 核心开发者明说 `torch.compile`/TorchScript 依赖 PEP 523 不太可能支持），
   PyPy 的 cpyext 本身"often much slower"，numpy 已于 2025-12 宣布 drop PyPy；GraalPy 必须用它自带的 pip 从源码重建 C 扩展，
   而我们链路上的 `torch_npu` + CANN（闭源 `.so`）无法重编。
11. **CPython 3.15 的 JIT 仍是实验性/opt-in**（官方 Windows/macOS 二进制"编进去但默认关闭"，`PYTHON_JIT=1` 手工开），
   官方口径 x86-64 Linux **8–9%**、AArch64 macOS **12–13%**（相对 tail-calling 解释器，pyperformance 几何均值）；
   对"以 C 扩展调用为主"的负载收益预期进一步打折（我们已在 3.14 JIT 上实测出 ≈0，见第 5 条）。
   结论：**不要为 JIT 而升级**；升级只值它带来的常规解释器改进（本负载 3–7%，见第 4 条）。
12. **把整段逻辑搬到 C++/Rust 是生态里已被验证的路径**，而且**我们这条链上已经有先例**：
   vLLM 0.26.0 自带 PyO3 扩展（`vllm._rust_tool_parser`，`abi3-py38` + limited API）、
   vllm-ascend 自带 CMake 构建的 C++ 扩展（`csrc/`）。但请注意：搬过去只能消掉"胶水时间"（≈12%），
   搬不掉的 torch/ACL 派发成本原样存在。

---

## 1. 为什么"编译"在这条负载上天生天花板低

### 1.1 负载画像（引用本仓实测）

| 事实 | 数值 | 来源 |
|---|---|---|
| `prepare_input` 占单步 | 55.3%（2757.6 µs / 4915 µs） | `../00-INDEX.md` §0 |
| topdown | frontend_bound 66.01%、IPC 0.771 | 同上 |
| 火焰图 | top-1 `_PyEval_EvalFrameDefault` 10.35%，top-10 合计 22.83% | 同上 |
| 指令级 | refcount/GC 页开销最大（`object.h:646` 16.01%、`pycore_gc.h:60` 54.15%） | 同上 |
| 920B 单次小操作固定开销 | **2–9 µs/次**（torch ATen per-call 1.61 µs、`zeros` 分配 2.30 µs、`index_select` 1.02 µs） | `data/model/microbench-torch.csv` |
| 最大单一函数 | `AscendGDNAttentionMetadataBuilder.build` 908 µs/步（29.3%） | `../00-INDEX.md` §0 |

### 1.2 本机（x86_64 开发机）复现出来的成本切分

用等价的合成负载（22 字段 metadata + 15 次小算子 + 25 次属性读，B=1）在同进程轮转计时：

| 成分 | 耗时 | 占比 |
|---|---|---|
| 15 次小算子（torch 后端） | 27.2 µs | **88%** |
| 22 字段构造 + 25 次属性读（"纯胶水"） | 3.8 µs | 12% |
| 15 次小算子（numpy 后端，同样语义） | 9.6 µs | — |

**推论：任何只作用于 Python 字节码层的手段（Cython / mypyc / JIT / 语法糖）最多能碰 12% 的地板；
而"把 torch 小算子换成 numpy 或减少调用条数"能动 65%。**

> ⚠️ 该切分是本机 x86_64 的比例，真实 920B 上算子/胶水的比例可能不同；920B 自带微基准显示
> `np.cumsum` 单次也要 2.6 µs（`data/model/microbench-numpy.csv`），所以"numpy 一定更快"不成立，必须逐算子验证。

---

## 2. 微基准（我们做的实验）

### 2.1 方法与口径

- 脚本：`data/cpython-alternatives/bench_glue.py`（负载定义）、`bench_all_inproc.py`（主计时）、
  `build_cy.py` + `cy_variants.pyx`（Cython 臂）、`mypyc_variant.py`（mypyc 臂）、`run_matrix.py`（子进程矩阵）。
- 负载：每轮 = 构造 22 字段元数据对象（1 个 dataclass / slots / 手写类 / NamedTuple / dict，5 种写法）
  + 15 次小 numpy/torch 算子调用（B=1，几十个元素）+ 同一对象的 25 次属性读取（读两遍防优化）。
- 三种"张量后端"：`torch`（真实形状）、`numpy`（同样 15 步换成 numpy）、`none`（剥掉算子，只留胶水）。
  三者的**唯一差异**是算子实现与容器实现，构造/读取逻辑逐字相同。
- 计时（臂内比较）：**同进程、轮转顺序、9 轮取中位数**，`taskset -c 4-7`，`OMP_NUM_THREADS=1`，`torch.set_num_threads(1)`。
- 计时（跨解释器比较，`cross_interp_ab.py`）：**交替多轮**（A/B/C… 轮转 5~9 轮）后取中位数。
  原因是这台开发机是 KVM 虚机，**同一个解释器跑同一条命令，不同批次的绝对值可以差 2×**：
  我们实测 cp314 的纯胶水负载 1.82 µs（11:32 批次）与 3.80 µs（11:44 批次）——两次都无并发干扰。
  因此：
  - **绝对 µs 只在同一次运行内可比**；跨批次只能看"交替多轮中位数"的比值；
  - 任何"某解释器/某方案快 2×"的结论都必须先在交替口径下复现（我们已有一次教训：3.12→3.13 的"2×"就是批次漂移的假象）。

### 2.2 环境清单

| 项 | 值 |
|---|---|
| 机器 | 开发机 `server-mini`（x86_64, AMD EPYC Eng Sample, 1 socket × 12 core = 12 逻辑核, 29 GB RAM） |
| 内核 / 发行版 | Linux 7.1.5-arch1-2 / Arch |
| CPU 绑定 | `taskset -c 4-7`（4 核），未占用任何 a3-22 资源 |
| 编译 / 计时解释器 | CPython 3.12.14 (conda-forge, GCC 15.3.0)，`data/cpython-alternatives/.venv`（`--system-site-packages`） |
| 其它解释器 | CPython 3.13.15 / 3.14.7 / 3.14.7t（**同一 conda-forge 构建链**，`python-freethreading` 包），prefix = `~/miniforge3/envs/.env313`、`.env314`、`.env314t` |
| 解释器构建参数 | conda-forge 构建：`--enable-optimizations --with-lto=full --with-computed-gotos`（PGO+LTO+computed goto 全开）；3.14 还带 `--enable-experimental-jit=yes-off`（JIT 编入但默认关，可用 `PYTHON_JIT=1` 打开）；3.14t 为 `--disable-gil`，因此强制使用 **mimalloc**（`WITH_MIMALLOC=1`，`PYTHONMALLOC` 不可改） |
| 与生产构建的关系 | 生产镜像里的 Python 是**默认参数构建**（`USE_COMPUTED_GOTOS=0`、无 PGO/LTO，见 `00-FINDING-build-config.md`）。**本报告所有数字都是在"构建参数修好之后"的解释器上测的**，这正是我们想问的问题："构建修好之后，编译/换运行时还能带来什么？" |
| numpy | 2.5.3（**四个解释器完全一致**，避免版本混淆） |
| torch | 2.13.0+cpu（`--index-url https://download.pytorch.org/whl/cpu`） |
| Cython / mypy | Cython 3.3.0（`gcc -O3`） / mypy 2.3.1 |
| 与真机差异 | 真机是 aarch64 920B + torch 2.9.0+cpu(Ascend) + numpy 2.4.6。**本机绝对数不能外推**，只用相对比值与机制结论。 |

### 2.3 表 1：容器实现方式（cp312，同进程轮转，µs/iter，中位数）

| 容器实现 | torch 后端 | numpy 后端 | 无算子（纯胶水） |
|---|---|---|---|
| `@dataclass`（基线） | 30.95 (1.000) | 13.33 (1.000) | 3.76 (1.000) |
| `@dataclass(slots=True)` | 31.31 (1.012) | 13.30 (0.998) | 3.69 (0.981) |
| 手写普通类（有 `__dict__`） | 30.89 (0.998) | 13.11 (0.984) | 3.70 (0.983) |
| 手写 `__slots__` 类 | 30.66 (0.991) | 13.35 (1.002) | 3.66 (0.975) |
| `NamedTuple` | 31.28 (1.011) | 14.14 (1.061) | 3.92 (1.041) |
| `dict` | 30.79 (0.995) | 13.58 (1.019) | 4.08 (1.085) |

**结论：语法级容器改造在本负载上是 ±5% 量级（整体负载上 <1.5%）。**
`slots` 在纯胶水上稳定小胜（-2%），但在有算子的负载上被淹没；
NamedTuple/dict 反而更慢（构造时要建 tuple/dict + 额外拷贝）。

> 原始数据：`data/cpython-alternatives/inproc-cp312-{torch,numpy,none}.csv`

### 2.4 表 2：编译臂（cp312，同进程轮转，µs/iter）

| 方案 | 实现 | torch 后端 | numpy 后端 | 纯胶水 |
|---|---|---|---|---|
| 纯 Python | 基线 `plain` | 30.95 (1.000) | 13.33 (1.000) | 3.76 (1.000) |
| **Cython 原样编译** | 逐字拷贝 `bench_glue.py` → `cythonize`（无类型标注） | 33.72 (1.089) | 13.09 (0.982) | 3.32 (0.883) |
| Cython 最小改造 | `.pyx` 手抄，仍是 Python 类 | 34.85 (1.126) | 13.26 (0.995) | 3.04 (0.808) |
| Cython 类型化 | `cdef class`（标量进 C 结构体） | 42.39 (1.370) | 21.90 (1.643) | 7.60 (2.020) |
| **mypyc native class** | 类属性全部标注为 native 类型 | 36.08 (1.166) | 13.49 (1.012) | 3.23 (0.860) |
| mypyc `@dataclass` | mypyc 2.3.1 编译通过，无 crash | 36.16 (1.168) | 13.59 (1.020) | 3.44 (0.915) |
| 对照：Cython cdef 只做构造+读 | 剥掉所有算子 | — | — | 7.81 |

**读法（重要）**：
- 编译**确实**能让"纯胶水"变快：Cython 原样编译 -12%，手抄 .pyx -19%，mypyc -8~-14%。
- 但一旦把 15 次 torch 小算子放回来，**编译臂全部变慢 8–37%**：编译后的代码调用 C 扩展时丢掉了 CPython 3.12
  调用点特化（CALL/LOAD_ATTR inline cache）的收益，同时 cdef class 的 22 个类型化构造参数要逐参数做 `PyLong→C`/`bint` 转换。
- 我们单独验证过"单次调用"这一层：Cython 编译的循环 vs CPython 字节码循环调用 `np.cumsum` / `torch.from_numpy` /
  `tensor.mul(2)`，比值 0.94–1.00×（`call_probe_compare.py`）——**单次调用是持平的，慢的是整段混排后的净效应**。
- `cdef class` 臂是我们所有写里最慢的一条（+37%~+102%），说明"类型化改造不是免费的"：当数据本身是外部对象
  （torch.Tensor / np.ndarray）时，把 22 个字段塞进 C 结构体没有回报，反而每次构造都要做类型检查与转换。

> 原始数据：`data/cpython-alternatives/inproc-cp312-{torch,numpy,none}.csv`、`compiled-cp312.csv`、`probe_compare.py`

> ⚠️ 表 1/表 2 的绝对值来自**同一次会话**（11:36，含编译臂），与 §2.5 交替口径表的绝对值**不可直接比较**
> （同一解释器跨会话差 1.8~4.0 µs）。两张表各自的**比值**都是同运行内算出来的，可以放心用。

### 2.5 表 3：解释器版本 / JIT / GIL / 分配器（`cross_interp_ab.py` 交替多轮中位数，µs/iter）

| 配置（同一 conda-forge 构建链，numpy 2.5.3） | 纯胶水（7~9 轮） | 相对 cp312 | numpy 算子负载（5 轮） | 相对 cp312 | torch 负载（3 轮，噪声大） |
|---|---|---|---|---|---|
| CPython 3.12.14 | 3.82 | 1.000× | 13.77 | 1.000× | 33.80 |
| CPython 3.12.14 + `PYTHONMALLOC=malloc` | 4.04 | 1.058× | 14.00 | 1.017× | 33.08 |
| CPython 3.13.15 | 3.76 | **0.984×** | 14.07 | 1.022× | 31.84 |
| CPython 3.14.7 | 3.68 | **0.963×** | 13.37 | 0.971× | 31.38 |
| CPython 3.14.7 + `PYTHON_JIT=1` | 3.78 | **0.990×** | 13.17 | 0.956× | 31.33 |
| CPython 3.14.7 + `PYTHONMALLOC=malloc` | 4.04 | 1.058× | 13.60 | 0.988× | 31.88 |
| **CPython 3.14.7t（GIL 开）** | **2.19** | **0.573×** | **11.72** | **0.851×** | — |
| **CPython 3.14.7t（GIL 关，默认）** | **2.20** | **0.576×** | **11.82** | **0.858×** | — |

**结论（修正后，全部基于交替多轮口径）**：
- **升级版本几乎没有收益**：3.12→3.14 纯胶水 **-3.7%**、numpy 负载 **-2.9%**、torch 负载 **-7.2%**；
  3.13 与 3.12 在胶水上打平（0.984×）。**"每个小版本都快很多"在这条负载上不成立。**
- **JIT 实测 ≈0**：3.14 + `PYTHON_JIT=1` 相对同版本不开 JIT，纯胶水 **+2.7%（变慢）**、numpy 负载 -1.5%、torch -0.2%。
  这和官方"pyperformance 几何均值 8–9%（3.15）/ 3.14 期基本没有"的口径一致：**JIT 帮不到"短函数 + C 扩展调用"型代码**。
- **关 GIL 本身无收益**：同一份 3.14.7t 二进制 `PYTHON_GIL=1` vs `=0` → 2.19 vs 2.20（胶水）、11.72 vs 11.82（numpy），
  差异在噪声内。**free-threading 不是单线程延迟的优化手段。**
- **`PYTHONMALLOC=malloc`（glibc）比默认 pymalloc 慢 5.8%（cp312）/ 9.8%（cp314）**：分配器确实是个变量，但方向是"pymalloc/mimalloc > glibc malloc"。
- ⚠️ **一个未解释的观察**：free-threaded **构建**在大约 20 次独立运行里稳定比普通 3.14 构建快 **1.75×**（胶水）、1.16×（numpy 负载），
  且与 GIL 开关无关、也不能用分配器解释（见 §2.6 的零分配对照）。我们**不把它当作结论**，只当作待追线索（§10）。

> 原始数据：`data/cpython-alternatives/cross-ab-{none,numpy,torch}.json`、`cross-ab-none-reuse.json`；
> 单批次口径（**不可用于跨解释器比较**，仅留档）：`inproc-*.csv`、`s2-*.csv`

### 2.6 零分配对照臂：每步"别新建对象"值多少

同一负载再加一个对照臂 `plain_reuse`：**复用同一个 metadata 对象、只改 22 个字段**（零对象分配），其余不变。

| 配置 | `plain`（每轮新建对象） | `plain_reuse`（零分配） | 节省 |
|---|---|---|---|
| CPython 3.12.14 | 3.79 | 3.32 | **12%** |
| CPython 3.13.15 | 3.70 | 3.13 | 15% |
| CPython 3.14.7 | 3.79 | 3.26 | **14%** |
| CPython 3.14.7t（GIL 关） | 2.37 | 1.73 | 27% |
| CPython 3.14.7t（GIL 开） | 2.17 | 1.44 | **34%** |

**两个结论**：
1. 在**普通构建**上，"复用容器、不每步新建对象"能省 **12–15% 的胶水时间**——这是一个纯代码级（L0）杠杆，
   与编译无关，也不需要换运行时。对应到真机：`prepare_input` 里每步新建的 metadata 容器、临时 list/dict/set 都是候选。
2. 反过来说，`plain_reuse` 臂里几乎没有分配了，**FT 构建仍然比普通 3.14 快 2.26×（1.44 vs 3.26）**，
   所以 §2.5 里那个 1.75× 的优势**不是分配器造成的**，最可能来自 free-threaded 构建在对象布局/属性访问实现上的差异
   （3.13 起 FT 构建启用 managed dict / inline values 一类改动）或二进制布局差异。
   我们**没有定位到根因**，因此这条线索只列入"待查"（§10）。

### 2.7 这些实测**不能**证明什么

- 不能外推到真机：本机 x86_64、torch 2.13 vs 真机 aarch64、torch 2.9.0+cpu(Ascend)。
  真机单次小算子固定开销 2–9 µs 量级与这里的 ~1.8 µs/op 是同一量级，**方向可信、数值不可信**。
- 不能代表"最优 Cython 写法"：我们只做了 3 种写法，没有做第二轮手工调优（例如把 22 个字段拆成
  `cdef` 局部变量、把 metadata 对象换成 tuple、把 tensor 生命周期显式管理）。因此表 2 只能用来证明
  **"原样/浅改造的编译没有净收益"**，不能用来证明"Cython 的上限"。
- `numpy 后端` 不是"可直接替换"的方案：只有不参与图捕获/不下发的元数据才适合 numpy 化。

---

## 3. Cython

**机制**：把 `.py`/`.pyx` 编译成 C 扩展。收益来自"早绑定"：`cdef class` 的字段存在 C 结构体里（无 `__dict__`、无 refcount 遍历）、
`cdef`/`cpdef` 方法可直接 C 调用、局部变量可声明为 C 类型、`boundscheck=False`/`wraparound=False`/`cdivision=True` 去掉检查。
调用外部 C 扩展（torch）仍是普通 Python 调用，编译器不改变其派发成本。

**公开实测收益**

| 来源 | 结论 |
|---|---|
| Stefan Behnel 复跑 Raymond Hettinger 的属性/容器微基准（CPython 3.8 时代） | "大多数操作快 30–50%"：实例属性读 20.8 ns vs 31.7 ns、`__slots__` 属性读 15.3 ns vs 25.8 ns、dict 读 16.5 ns vs 28.7 ns、实例属性写 28.6 ns vs 49.1 ns |
| Cython 官方文档（early binding） | 用 `cdef class` 消除方法调用/属性查找的 Python 开销，"差距可以非常大"，但**前提是调用方也用 C 类型早绑定** |
| Cython 官方 troubleshooting 文档 | `cdef` 属性的查找会**静默回退**到 Python 语义（不报错、只变慢），是常见的"编译了但没变快"的原因 |
| arXiv 2505.02346（2025，7 个基准 × 8 个工具） | Cython/mypyc/Nuitka 在他们的基准上"改进很小或可忽略"（Codon/PyPy/Numba 才是大幅改进者） |
| 本调研实测（§2.4） | 原样编译纯胶水 -12%；但含 torch 小算子的完整负载 +8.9%（变慢） |

**`cython.inline` / `pyximport`（专门验证过）**

- 官方文档明确：`pyximport` "rather experimental, will not work at all for some `.py` files and packages, and will heavily slow down your imports"，
  且"**不推荐**让终端用户走 pyximport，正确做法是提供预编译 wheel"；`cython.inline` 同源，同样需要**运行时存在 C 编译器**。
- 本机实测：`cython.inline("return a+b", b=3)` 可用，但会写入 `~/.cython/inline/`（缓存目录里出现 `.pyx/.c/.so`），
  **缓存命中后的单次调用也要 3049 ns/call**（同一进程里纯 Python `def plus3` 只需 24 ns/call）——比我们真机上"单次小算子 2–9 µs"是同一量级，
  **在 per-step 热路径上完全不可用**；此外它无法直接引用调用者的局部变量（我们用 `a` 写 typed 循环时直接 `CompileError`）。

**收益/代价（本负载）**

| 项 | 实测/事实 |
|---|---|
| 收益 | 纯胶水 -12~-19%；完整负载 -0~+37%（变慢） |
| 编译产物体积 | `cy_variants.c` 736 KB → `.so` 176 KB；`bench_glue_cy.c` 939 KB → `.so` 250 KB |
| 编译时间 | 本机单模块 3.7 s（`gcc -O3`，12 核机器） |
| 调试难度 | traceback 帧不再是纯 Python 帧；`cdef` 属性 miss 静默回退；`wraparound=False` 下负索引是**未定义行为** |
| 与 torch C++ 扩展的互操作 | 不需要特殊处理：Cython 模块可以直接 import torch / 调用其 Python API；必要时可用 `cdef extern` 调 `libtorch`，但 vLLM/ascend 里没有这类先例 |
| 分发成本 | 每个 CPython 小版本 + 每个平台一个 `.so`（除非走 limited API）；要进 CI 与镜像构建 |

**我们踩到的坑（真实发生，建议写进团队 wiki）**

1. **`wraparound=False` + 负索引**：同一份 `.pyx` 用 `wraparound=False` 编译后，第一次运行正常，
   第二次在 `seq[-1]` 上抛 `IndexError: index -2 is out of bounds for axis 0 with size 1`。这就是文档里"未定义行为"的现场表现。
   真改造时必须把 `arr[-1]` 改写成 `arr[len(arr)-1]`，或保留 `wraparound=True`。
2. **`cythonize` 的增量检查不跟踪 compiler directives**：改完 `compiler_directives` 后它**静默跳过**重编（`.so` 时间戳不变、行为照旧），
   必须 `force=True` 或删缓存。这会直接导致"我改了指令却没生效"的假结论。

**可行性判定：有条件可用，但对本负载 ROI 为负。**
只有在"纯 Python 逻辑占比高、且该段不再调用大量 C 扩展"的代码上才值得；本负载恰好相反。
**证据强度：强**（官方文档 + 我们的同进程实测 + 两个独立来源的公开数据）。

---

## 4. mypyc

**机制**：复用 mypy 的类型信息，把**整个模块**编译成 C 扩展；标注为 native 的类/函数会被改成 C 结构体 + 直接 C 调用，
未标注/外部类型（`Any`、"erased"类型、来自其它模块的类如 `torch.Tensor`）退化为**与解释器等价的通用对象操作**。

**公开实测收益**

| 来源 | 结论 |
|---|---|
| mypyc 官方 introduction | "已有类型标注的代码通常快 **1.5–5×**；为 mypyc 调优过的代码可以快 **5–10×**" |
| mypyc 官方 using-type-annotations | "如果只用 erased 类型，相比 CPython 的显著收益只有去掉解释器开销和一点早绑定，**通常只有轻微提升**" ← 这条直接命中我们的场景 |
| Black（首个大规模生产案例，Richard Si 报告） | 格式化 **1.93×**、解析 1.81×、`import black` 1.16×；单文件最好 2.38×；mypy 自身约 4× |
| h11（2026-09-19 的实践复盘） | 改造成本中等：吞吐 **+62%**（作者原话：不是标题党式的 2×，想要 2× 还得继续手动调优） |
| mypyc issue #886 集成反馈（Black） | 崩溃与兼容问题"大部分来自 **dataclass**"；产物体积 120 KB → 3.3 MB（加 `CFLAGS=-g0` 后 ~1 MB）；**docstring 被剥离**，会打破依赖 docstring 的 CLI（对 vLLM 的 `envs` 文档生成是实打实的风险） |
| mypyc 官方 native classes / issue #671 | dataclass、attrs 只有"partial native support"，**当前是按非扩展类编译的**（不追求快） |
| CPython issue #140704（2025-10） | mypyc + dataclass 在扩展模块导入路径上还会踩 `sys.modules`/`KeyError` 一类坑（社区仍在修） |
| 本调研实测 | native class 纯胶水 **-14%**，dataclass 纯胶水 -8.5%；含 torch 算子的完整负载 **+17%**（变慢） |

**对 `torch.Tensor` 这类外部类型的处理**：明确回答——**会退化成等价的 Python 调用**。
官方文档原话是：只用 erased 类型时"除了去掉解释器开销和一点早绑定之外没有显著收益"。
我们的实测印证了这一点：mypyc 臂在 numpy 后端几乎打平（1.012~1.020×），在 torch 后端反而变慢。

**代价**：编译时间（本机首次单模块 **93 s**，含 mypyc 运行时）；产物 **436 KB**（单个小模块；Black 的实测是整包 1–3.3 MB）；
需要"能通过 mypy 的干净类型"（我们只做了一个小模块，vLLM 全仓库要过 mypy 是另一项工程）；
每个 CPython 小版本/平台重建；与 `torch.compile`/动态属性/`__getattr__` 的兼容性未验证。

**可行性判定：有条件可用（对"纯 Python 真实逻辑多、C 扩展调用少"的模块），对本负载不划算。**
**证据强度：强**（官方文档口径 + Black/h11 生产数据 + 我们的实测）。

---

## 5. Nuitka / PyO3(Rust) / C++ 扩展

### 5.1 Nuitka

**机制**：把整个包 AOT 编译成 C/C++（可 standalone/onefile），可选 PGO/LTO；不是"局部加速某个函数"的工具。

**公开实测**

| 来源 | 结论 |
|---|---|
| Nuitka 官方 performance 页 | 自家 pystone 数据 3.35×（LTO）/ 3.72×（PGO）——**这是 Python 2.7 时代的数据**，且 benchmark 单一 |
| Nuitka speedcenter | 逐语言构件（construct）对比，多数构件接近 CPython，部分更慢 |
| arXiv 2505.02346（独立复现，7 基准） | Nuitka 平均收益"小或可忽略"，且 LLC miss 率最高（58.73%），在 `n_body` 等基准上比 CPython 更慢 |

**判定：不推荐。** 我们把逻辑限定在"engine 循环里每步跑一次的函数"，AOT 整包既解决不了 C 扩展派发成本，
又带来全量重编译与分发成本。**证据强度：中**（官方数据陈旧 + 一篇独立学术复现）。

### 5.2 PyO3 / Rust 与 C++ 扩展：生态先例（这一节是"可行性证据"的重点）

| 先例 | 事实 | 链接 |
|---|---|---|
| **vLLM 0.26.0 自身** | `tools/build_rust.py` 用 setuptools-rust 构建两个 Rust 目标：可执行文件 `vllm-rs` 与 **PyO3 扩展 `vllm._rust_tool_parser`（`pyo3/abi3-py38` + `py_limited_api=True`）**；`vllm/tool_parsers/rust_tool_parser.py` 在缺失该扩展时会明确报错 | 本地快照 `refs/vllm/tools/build_rust.py`、`build_rust.sh`、`vllm/tool_parsers/rust_tool_parser.py` |
| **vllm-ascend 自身** | `csrc/` 已是 CMake 工程（`setup.py` 里 `CMakeExtension` + `cmake_build_ext`、`csrc/build_aclnn.sh`），其中 `aclnn_torch_adapter/` 与 `camem_allocator.cpp` 就是**宿主侧 C++** | 本地 `vllm-ascend/csrc/` |
| **NVIDIA Dynamo** | 官方定位"Built in Rust for performance, Python for extensibility"；核心 runtime 在 Rust，Python 侧经 PyO3 桥接；v1.5.0 甚至把 TensorRT-LLM decode worker 的出向路径改成"Python handler 把响应推给 Rust 的 `ResponseSender`（tokio channel）"以把 tokio 线程从 Python 里摘出来 | github.com/ai-dynamo/dynamo；Dynamo v1.5.0 release notes |
| **TensorRT-LLM** | 有 C++ 高层 API（`Executor`）与 Python bindings（`tensorrt_llm.bindings.executor`）；2026-03 专门加了 **PyExecutor 宿主侧性能回归测试套件**（scheduler/sampler/KV cache manager 的 CPU 开销、BS1–256），说明"Python 宿主开销"被当成一等公民问题 | TRT-LLM Executor API 文档；PR #12148 |
| **FlashInfer / SGLang** | attention 的 `plan()`（host 侧构造元数据）在每步 decode 里会做 D2H 同步、卡在 CPU 关键路径；SGLang 用 `fast_decode_plan`（传 `global_override_indptr_cpu`）绕开每步 D2H，并在向"GPU-based planning"迁移；SGLang #3987 也是同类修复（cuda graph replay 期间避免 indptr 张量来回拷） | FlashInfer attention 文档；sglang PR #10760 / #3987、issue #23500 |

**收益的量级（把整段搬到 C++/Rust 能拿到什么）**

- 只能消掉"Python 解释器执行这几十行胶水"的时间。以本机口径算：**≈12% 的该段耗时**；
  以真机 `AscendGDNAttentionMetadataBuilder.build` = 908 µs/步算，乐观估计能省 **几十~150 µs/步**（≈2–5% of `prepare_input`），
  前提是这段逻辑里**没有**大量 torch 调用——事实上它有（每步 3 次 build、内部多次 torch 算子），那部分搬到 C++ 后**还要再付一次 pybind/PyO3 边界成本**。
- 生态里真正让这类优化见效的做法是**顺着"减少同步/减少调用条数"走**（FlashInfer/SGLang 的 fast_decode_plan 之所以有效，是因为它去掉的是**每步 D2H 同步**，不是把 Python 换成 C++）。
- 反例同样存在：TensorRT-LLM 的 PyTorch 后端与 C++ 引擎在首请求上差 5×，但社区归因于**一次性 autotune/graph capture**，并非稳态 5×——说明"用 C++ 重写"不等于自动变快，必须用稳态 per-step 数字验收。

**维护代价**

- 双语言代码库（构建、调试、review、单测都要两套）；ABI/工具链问题（TRT-LLM 官方文档就提到 GCC CXX11 ABI 与 torch 2.7 前后的坑）；
- 每个平台都要重编（aarch64 openEuler + CANN 的组合更小众）；CI 需要交叉编译或原生 builder；
- 好处：**用 `abi3` 可以把"Python 小版本升级"的成本一次性买断**（vLLM 的 PyO3 扩展就是 `abi3-py38`）。
- 我们这条链上"新增一个 C++/Rust 源文件"的**边际构建成本很低**——`csrc/` 与 CMake 已经存在，Rust 侧也有现成 `build_rust.sh`。

**可行性判定：可行（推荐级别：中）。** 建议只对"每步都跑、逻辑稳定、纯胶水占比高"的段做（例如 attention metadata 的字段拼装/切分），
并用稳态 per-step 延迟做验收；不要把"换成 C++"当成对 torch 派发成本的解药。**证据强度：强**（上游同仓库/同生态的落地事实）。

---

## 6. free-threading（PEP 703）与 subinterpreter（PEP 684/734）

### 6.1 free-threading：**关 GIL 对单线程零收益，不作为手段**

**公开证据**

| 来源 | 结论 |
|---|---|
| Python 官方 free-threading 文档（3.14） | 单线程性能损失"现在大约 **5–10%**，取决于平台与 C 编译器"；pyperformance 平均开销 **macOS aarch64 ~1% ~ x86-64 Linux ~8%** |
| 3.13 时期（PEP 703 实验阶段） | 约 **40%** 单线程损失（社区广泛引用） |
| PEP 779 | 3.14 起 free-threaded 官方支持（Phase II，仍为可选构建） |
| arXiv 2603.04782（2026-03，3.14.2 自建 no-GIL） | 顺序纯 Python 负载 **慢 13–43%**；高竞争共享对象负载最多慢 **12.18×**；并行数值负载最多 4× 加速 |
| 第三方实践（Medium/Blog，2026） | "单线程脚本 5–10% 更慢，无收益"；且**导入未声明支持的 C 扩展会静默把 GIL 重新打开** |
| 官方文档（分配器口径） | free-threaded 构建**不用 pymalloc、全部走 mimalloc**，且 `--disable-gil` 强制要求 mimalloc（`--without-mimalloc` 不能与它共存）；mimalloc 的内存开销/行为与 pymalloc 不同 → 说明"FT 构建 vs 普通构建"的差异本来就**不只来自 GIL** |
| **本调研实测（交替多轮中位数，§2.5）** | 同一份 3.14.7t 二进制切 `PYTHON_GIL`：纯胶水 **2.19 vs 2.20 µs**（打平）、numpy 负载 **11.72 vs 11.82 µs**（打平）⇒**关 GIL 本身没有任何收益**；官方说的 5–10% 单线程损失在本负载上没有出现（本负载偏向属性访问/小对象，与本机 FT 构建的差异方向相反，见 §2.6 的未解观察） |

**为什么不该选**：真机火焰图的指令级热点正是 **refcount 写（`object.h:646` 16%）与 GC 相关代码**，
free-threading 要把这些 refcount 变成**原子操作**（biased-referencing / per-object lock），正好加在最贵的那条路径上；
而且它只买"多核吞吐"，不买"单步延迟"。再加上工程面：`torch` / `torch_npu` 没有官方的 free-threaded 支持声明，
**导入未声明支持的 C 扩展会让解释器静默把 GIL 重新打开**（那还不如直接用普通构建），CANN 侧更是无从验证。

**判定：不能用（对本目标）。** 关 GIL 无收益（同一二进制 A/B 打平），收益只出现在多核吞吐；
我们确实观测到 FT **构建**在本负载上快 1.75×，但那不是 GIL 的功劳，且原因未定位（§2.6/§10），**不能据此选型**。
**证据强度：强**（官方文档 + 我们的同二进制交替 A/B + 独立论文）。

### 6.2 subinterpreter（PEP 684 每解释器 GIL / PEP 734 `concurrent.interpreters`）：**不适用**

**机制与约束**（官方文档口径）

- 收益本质是"多核并行"，与单步延迟无关；解释器之间**不能共享可变对象**，通信要经跨解释器队列（拷贝）。
- 隔离解释器只允许导入**多阶段初始化（PEP 489）**的 C 扩展，否则 `ImportError`；`PyGILState_*` 类扩展（`ctypes` 等）在 subinterpreter 下"probably be broken"；
  `os.fork()`/`multiprocessing` 被禁。
- 每个解释器有独立的 `sys.modules`：**等于每个解释器都要把 import 成本再付一遍**（torch/CANN 的 import 是百毫秒级）。

**判定：不能用。** 我们的负载是单线程、单步延迟，且强依赖 `torch`/`torch_npu` 这类大概率不是多阶段初始化的扩展。
**证据强度：强**（PEP 684/734 + CPython C-API 文档的明文约束）。

---

## 7. PyPy / GraalPy：**明确不能用**

### 7.1 PyPy

| 事实 | 来源 |
|---|---|
| PyPy **v8.0.0（2026-09-19）** 才第一次发布 **Python 3.12（beta 质量）**，此前长期停在 3.11 | doc.pypy.org/release-v8.0.0.html |
| PyPy 的 C-API 兼容层 cpyext "**often much slower** than in CPython due to the need to emulate refcounting"，官方建议改用 CFFI | PyPy FAQ |
| **numpy 于 2025-12 提出并关闭 "drop support for PyPy"**（理由：PyPy 不再活跃开发、长期没有 3.12、缺少 limited-API 等） | numpy issue #30416 |
| **PyTorch 至今不支持 PyPy**：issue #17835 自 2019 年开放；PyTorch 侧明确"low priority…not clear that it supports the full C API we need"；PyPy 核心开发者原话：**`torchscript` 与 `torch.compile` 依赖 CPython 内部实现（PEP 523 frame evaluation），不太可能支持** | pytorch/pytorch#17835 |

**判定：不能用。** 我们链路的每个关键组件（torch、torch_npu、vLLM 的 Python API、numpy）都在 PyPy 上不可用或严重退化。
**证据强度：强。**

### 7.2 GraalPy

| 事实 | 来源 |
|---|---|
| GraalPy 25.x = **Python 3.12.8** 兼容运行时（最新 25.2.4，2026-07-28）；官方把 PyTorch/SciPy 列为"primary goal" | GraalVM Python 文档 / CHANGELOG |
| 原生扩展只有**API 级**兼容、**没有 ABI 兼容**：CPython 的 wheel 不能用，必须用 GraalPy 自带的 `pip` 从源码（并打官方补丁）重建；文档明确"do not update pip or use alternative tools such as uv" | GraalVM Native Extensions 文档 |
| 官方补丁列表里能看到的 torch 版本是 **Torch 2.7.0**（25.1.0），有限 | GraalPython CHANGELOG |
| 性能：以 LLVM bitcode 方式跑的扩展"通常最多只有 CPython 的一半性能"；纯 Python 预热后可快 3–4×（对老版本 CPython） | GraalVM Python FAQ |

**判定：不能用（对我们）。** 决定性障碍不是语言版本，而是 **CANN / `torch_npu` 的闭源原生库无法为 GraalPy 重编**（GraalPy 只支持从源码重建扩展，
且其 C-API 兼容层是实验性的）。即使只想跑 torch CPU 模型，也要为每个依赖走"官方补丁 + 源码重建"的私有流程，维护成本远超收益。
**证据强度：强。**

---

## 8. CPython 3.13 / 3.14 / 3.15 的 JIT 与解释器改进

### 8.1 现状（官方口径）

| 版本 | JIT 状态 | 实测收益 |
|---|---|---|
| 3.13 | 实验性、**编译期 opt-in**（`--enable-experimental-jit`），运行期 `PYTHON_JIT=1` | 当时"基本没有收益、常常比解释器还慢"（JIT 核心开发者复盘） |
| 3.14 | 新增 **tail-calling interpreter**（`--with-tail-call-interp`，**仅 Clang 19+ 的 x86-64/AArch64**，推荐配 PGO）；JIT 仍实验性/opt-in | tail-call 相对普通解释器 **3–5%**（几何均值；最初宣称的 9–15% 被证实是 LLVM 19 bug 造成的基线失真）；JIT 在 3.14 仍无实质收益 |
| 3.15（2026-08 RC1） | **JIT 大幅升级**（新 tracing 前端、基本寄存器分配、LLVM 21 生成模板），但**仍是实验性、仍 opt-in**：官方 Windows/macOS 二进制"编进去但默认关闭"，Fedora/Gentoo 类似 | 官方：x86-64 Linux **8–9%**、AArch64 macOS **12–13%**（相对 tail-calling 解释器，pyperformance 几何均值）；PEP 836 汇总的 Tier-1 机器表：M3 Pro 1.126×、**AmpereOne aarch64 1.073×**、i5-8400 1.069×、Ryzen Windows 1.047×；波动范围"约 -20% 到 +100%" |
| 长期路线 | PEP 836：目标是 **3.17 首个 beta 前** 在 free-threading 构建上做到 ≥20%，达标才考虑去实验化/默认开启 | 目前**不满足"默认开"的条件** |

### 8.2 对我们的预期收益

- JIT/特化只能加速"**Python 字节码本身**"，而我们的负载里这部分（胶水）只占 ~12%（cp312 实测），且其中大头是"构造小对象 + 属性读写"。
- **本机实测（交替多轮，§2.5）**：3.12→3.14 纯胶水 -3.7%、numpy 负载 -2.9%、torch 负载 -7.2%；
  3.14 打开 JIT 相对同版本不开 JIT：**≈0（胶水还变慢 2.7%）**。
  ⇒ 与"官方 pyperformance 8–9%"的差距说明：**我们的负载太"薄"，吃不到 pyperformance 里那些能在 3.13/3.14 变快的模式**；
  而 JIT 对"短函数 + C 扩展调用"更是天然无效。
- 因此：**"升级到 3.13/3.14" 的预期收益 ≈ 3–7%，"等 3.15 的 JIT" 的边际收益 ≈ 0（本负载）**，
  且 3.15 尚未 GA（RC1）。**不要为了 JIT 升级**；升级只值"常规解释器改进 + 生态支持窗口"。
- 若升级，真正要注意的是运维面：openEuler 上的 Python 从 3.12.13 升到 3.13/3.14 需要 CANN / `torch_npu` / vllm-ascend 的 wheel 支持，
  以及镜像内的其它 Python 依赖（`numpy 2.x` 等）同步重建。**这是"版本升级"成本，不是"JIT"成本。**
- 一个额外观察：官方 3.15 数字里 **macOS AArch64 的 12–13% 是相对 tail-calling 解释器**，而 tail-call 本身在 Linux+AArch64 需要 Clang 19+ 且要开 PGO。
  生产镜像如果不是 Clang+PGO 构建，能拿到的会打折扣。
- **优先级提醒**：与本条相比，`00-FINDING-build-config.md` 指出的"生产解释器没开 computed goto / 没有 PGO"是**同一类杠杆里更值钱的那个**
  （构建参数 → 解释器分派本身），且不需要改代码、不需要换运行时。

**可行性判定：可以用（低风险、收益有限 3–7%、需要走镜像升级流程）；JIT 部分明确"不要为它升级"。证据强度：中**
（官方数字强，但**没有** aarch64 Linux 生产负载的第三方实测；我们本机测的是 x86_64）。

---

## 9. 按可行性排序的落地路径

> 收益口径：以真机 `prepare_input` = 2757.6 µs/步 为分母；未特别说明时收益都指**这一段**的改善。

| 级别 | 动作 | 预期收益 | 一次性成本 | 长期维护风险 | 证据强度 |
|---|---|---|---|---|---|
| **L0 改代码** | ① 去掉重复计算（`compute_num_computed_tokens` 缓存，本仓已定位，≈130 µs/步） | **≈4–5%** | 0.5–1 人日 | 低 | 强（本仓真机实测） |
| **L0** | ② 把元数据构造里的小 torch 调用换成 numpy/标量运算（同 15 步：torch 27.2 µs vs numpy 9.6 µs 本机） | **10–30%**（视可替换比例） | 2–5 人日 | 低–中（要逐算子验证真机不更慢：920B 上 `np.cumsum` 2.6 µs） | 中（本机比例 + 920B 单算子数据） |
| **L0** | ③ 容器语法改造（`slots=True` / 手写 `__slots__`） | **<1%**（胶水 -2%，整体淹没） | 1 人日 | 极低 | 强（本调研实测） |
| **L0** | ④ 减少每步对象/调用条数（合并 builder 调用、复用 workspace、避免每步新建容器） | **5–15%**（本机零分配对照臂：胶水 -12~-14%） | 3–10 人日 | 中（改的是热路径逻辑，需回归） | 中–强（本调研实测 + 与 ① 同源） |
| **L1 编译现有代码** | Cython / mypyc 编译热点模块 | **≈0 甚至负**（纯胶水 -12~-19%，完整负载 +8~+17%） | 1–2 周（构建链 + wheel + CI） | 中（调试难、dataclass/负索引等坑、每 Python 版本重建） | 强（本调研实测） |
| **L1.5 修解释器构建参数** | 不改代码、不改运行时骨架，只把生产 Python 重新按 `--enable-optimizations --with-lto --with-computed-gotos` 构建（见 `00-FINDING-build-config.md`） | **待实测**（同仓另一篇在 a3-22 上跑 A/B；这是"解释器分派"级杠杆，理论上比 L1 编译方案更对症） | 1 次镜像重建 | 低（构建链固定后无长期负担） | 中（事实链已核实：`USE_COMPUTED_GOTOS=0`；收益待 A/B 落地） |
| **L2 局部搬到 C++/Rust** | 把"每步都把同一段纯胶水逻辑跑一遍"的函数（如 attention metadata 拼装）搬进 C++/Rust，用 pybind11/PyO3 暴露 | **2–5%**（乐观：该段胶水的 100%；但该段里 torch 调用搬不走） | 1–3 周/段（构建链已存在，边际成本低） | 中–高（双语言、ABI、跨平台构建、单测双份） | 中（生态先例强，但 vLLM/aarch64 场景无公开先例） |
| **L3 换运行时/版本** | 升 CPython 3.12 → 3.13/3.14（不改代码） | **3–7%**（本机交替多轮实测：胶水 -3.7%、numpy 负载 -2.9%、torch 负载 -7.2%） | 镜像/工具链重建（CANN/torch_npu 需配套） | 中（生态 wheel 支持） | 中（本机实测 + 官方 pyperformance 方向一致） |
| **L3** | 升到 3.15 + `PYTHON_JIT=1` | **≈0（本负载）**：3.14 JIT 实测 ±2%；JIT 对 C 扩展调用密集代码天然无效，且仍 opt-in | 同上 + 自建 JIT/PGO 构建 | 中–高 | 中（本机实测 + 官方口径） |
| **L3** | free-threading / subinterpreter / PyPy / GraalPy | **负收益或不可用** | — | — | 强（判定见 §6/§7） |

**推荐组合（按性价比）**：先做 L0（①②④）把"调用条数/重复计算/每步分配"打掉，**同时把 L1.5（解释器构建参数）作为最便宜的"解释器级"杠杆推进**；
只有当某一段被证明"每步都跑 + 纯胶水占比高 + 逻辑稳定"时，才考虑 L2 把它整体搬到 C++/Rust（并顺手用 `abi3` 买断 Python 版本升级成本）；
L1 编译方案在本负载上不推荐；L3 版本升级作为常规演进（3–7%），不要指望它救场，**更不要为 JIT 而升级**。

---

## 10. 明确"查不到的 / 不能用的"

**做不到 / 不适用（判定）**

1. **PyPy**：不能用（torch 无支持、numpy 已 drop、cpyext 慢）——证据强。
2. **GraalPy**：不能用（CANN/torch_npu 闭源原生库无法重建；原生扩展无 ABI 兼容）——证据强。
3. **free-threading 的"关 GIL"**：对单线程零收益（同一二进制交替 A/B 打平：2.19 vs 2.20 µs）——证据强。
   （注意：FT **构建**在本负载上反而快 1.75×，但那是构建差异、不是 GIL，原因未定位，见下"查不到"第 7 条。）
4. **subinterpreter（PEP 684/734）**：与单步延迟无关，且要求 C 扩展多阶段初始化——证据强。
5. **`cython.inline` / `pyximport` 进生产热路径**：官方不推荐 + 实测 3049 ns/call 的调用开销——证据强。
6. **Nuitka 整包 AOT**：对 per-step 热路径无收益，独立研究显示"小或可忽略"——证据中。
7. **编译方案（Cython/mypyc）作为本负载的主手段**：实测净收益 ≤0——证据强。

**查不到（诚实的空白，后续要补的）**

1. **Cython/mypyc 针对"属性查找 + 小张量派发"这类负载的公开实测**：没有第三方基准直接覆盖，
   官方与社区数据都是"数值循环/字符串处理/打印格式化"型（Black、h11、pystone）。本报告的实测是这一空白的补位，但只在 x86_64。
2. **mypyc 在 vLLM 规模代码库上的真实加速比与崩溃率**：只有 Black/h11/mypy 自身三个案例。
3. **CPython JIT 在 aarch64 Linux 上的第三方真实负载实测**：只有 pyperformance 与官方机器表（AmpereOne 1.073×），
   没有"serving 引擎每步 host 逻辑"这种负载的公开数据。
4. **Ascend/NPU 场景下 free-threading 的任何公开数据**：没有。
5. **vLLM/vllm-ascend 社区把 per-step metadata 构造搬到 C++/Rust 的公开 PR 与收益数字**：
   同生态里有（FlashInfer/SGLang 的 plan 优化、Dynamo/TRT-LLM 的宿主侧改造），但 vLLM 侧没有可引用的先例。
6. **920B 上 numpy vs torch 单算子成本的系统对比**：本仓有零散点（`data/model/microbench-{numpy,torch}.csv`），
  但不构成"逐算子替代收益表"，需要补一次 A/B。
7. **free-threaded 构建在本负载上快 1.75× 的根因**：我们排除了 GIL 开关（打平）、也排除了分配器
   （零分配对照臂里优势仍有 2.26×），剩下"FT 构建的对象布局/inline values/managed dict 实现差异"或
   "二进制布局差异"两种可能，都没验证。**这条值得单独开一个 10 分钟实验**（例如把一个 22 字段对象的
   构造/读/写拆成三个微基准，分别在 3.14 与 3.14t 上跑；并在纯 Python 对象上关掉 numpy 干扰）。
8. **能不能在 GIL 构建上单独启用 mimalloc**：官方文档只说 `--without-mimalloc` 不能与 `--disable-gil` 共存，
   我们**没验证** `--with-mimalloc` 能否用于普通构建（如果能，那可能是一条"免费"的构建级收益；
   本机 pymalloc→glibc malloc 反而慢 5.8~9.8%，所以方向不明确）。
9. **微基准的跨二进制可比性**：本机同一命令跨批次差 2×（1.82 vs 3.80 µs），我们没能定位是"vCPU 放置/邻居"
   还是"解释器二进制布局"。这直接影响所有微基准结论的置信区间，值得用 `perf stat`（instructions/cycle）
   而不是墙钟时间来复核。

---

## 11. 复现方式

```bash
# 目录：/home/chiro/projects/vllm/preparing-input-phase/data/cpython-alternatives
# 1) 建环境（复用已有 conda env 的 site-packages，避免重复下载 torch）
python3.12 -m venv --system-site-packages .venv
.venv/bin/pip install cython mypy numpy==2.5.3
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
# 2) 纯 Python 表（同进程轮转）
for be in torch numpy none; do it=1200; [ $be != torch ] && it=5000; \
  taskset -c 4-7 .venv/bin/python bench_all_inproc.py --backend $be --iters $it --rounds 9 \
    --python-label cp312 --out inproc-cp312-$be.csv; done
# 3) Cython / mypyc 臂
.venv/bin/python build_cy.py cy_variants cy_probe bench_glue_cy   # 约 15 s
.venv/bin/python -m mypyc mypyc_variant.py                        # 首次约 93 s
# 4) 跨解释器（同一 conda-forge 构建链；3.14t 用 PYTHON_GIL 切 GIL）
mamba create -y -p ~/miniforge3/envs/.env314 -c conda-forge python=3.14 numpy
# 5) 跨解释器/跨 GIL/JIT/分配器的**交替多轮**对照（唯一可信的跨二进制口径）
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0   # 装进 .env313/.env314
taskset -c 4-7 .venv/bin/python cross_interp_ab.py --backend none  --reps 9 --out cross-ab-none.json
taskset -c 4-7 .venv/bin/python cross_interp_ab.py --backend numpy --reps 5 --out cross-ab-numpy.json
# 6) 隔离调用开销探针（证明"单次调用持平、整段变慢"）
taskset -c 4-7 .venv/bin/python call_probe_compare.py
```

**产物清单**：`data/cpython-alternatives/` 下 6 个 `.pyx`/`.py` 源码、5 个 `.so`（CPython 3.12）、
结果 CSV/JSON（`inproc-*.csv`、`s2-*.csv`、`cross-ab-*.json`）、`build_cy.py`/`run_matrix.py`/`run_ext_matrix.py`（构建与矩阵驱动）、
`bench_all_inproc.py`/`cross_interp_ab.py`（两种计时口径）、`probe_compare.py`/`call_probe_compare.py`（隔离探针）。

---

## 12. 引用来源（访问日期均为 2026-09-24）

**Cython**
1. Cython docs, *Source Files and Compilation*（pyximport / cython.inline）— https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html
2. Cython docs, *Early binding for speed* — https://cython.readthedocs.io/en/latest/src/userguide/early_binding_for_speed.html
3. Cython docs, *cdef classes / extension types* — https://docs.cython.org/en/stable/src/tutorial/cdef_classes.html
4. Cython docs, *Troubleshooting*（cdef 属性静默回退）— https://docs.cython.org/en/latest/src/userguide/troubleshooting.html
5. S. Behnel, *Speeding up basic object operations in Cython*（Hettinger 微基准复跑）— http://blog.behnel.de/posts/tuning-basic-object-operations-in-cython/

**mypyc**
6. mypyc docs, *Introduction*（1.5–5× / 5–10×）— https://mypyc.readthedocs.io/en/latest/introduction.html
7. mypyc docs, *Using type annotations*（erased 类型收益有限）— https://mypyc.readthedocs.io/en/stable/using_type_annotations.html
8. mypyc docs, *Native classes*（dataclass/attrs partial native support）— https://mypyc.readthedocs.io/en/latest/native_classes.html
9. mypyc issue #671, *Make dataclasses efficient* — https://github.com/mypyc/mypyc/issues/671
10. mypyc issue #886, *Integration feedback report: psf/black* — https://github.com/mypyc/mypyc/issues/886
11. Ichard26, *Benchmark results from the mypyc integration work for psf/black* — https://gist.github.com/ichard26/b996ccf410422b44fcd80fb158e05b0d
12. D. Fimov, *Speeding up h11 with mypyc*（2026-09-19）— https://danfimov.com/posts/mypyc-h11/
13. CPython issue #140704, *Error compiling modules using dataclasses with mypyc* — https://github.com/python/cpython/issues/140704
14. V. Stoico et al., *An Empirical Study on the Performance and Energy Usage of Compiled Python Code*, arXiv:2505.02346 — https://arxiv.org/abs/2505.02346

**Nuitka**
15. Nuitka, *Performance* — https://nuitka.net/user-documentation/performance.html
16. Nuitka Speedcenter — https://speedcenter.nuitka.net/

**Rust/PyO3/C++ 先例**
17. vLLM 0.26.0 源码（本地快照）`tools/build_rust.py` / `build_rust.sh` / `vllm/tool_parsers/rust_tool_parser.py`
18. NVIDIA Dynamo — https://github.com/ai-dynamo/dynamo ；Dynamo v1.5.0 release notes — https://docs.nvidia.com/dynamo/reference/releases/v1-5-0
19. TensorRT-LLM, *Executor API* — https://nvidia.github.io/TensorRT-LLM/advanced/executor.html
20. TensorRT-LLM, *Architecture Overview*（CUDA Graph 与 host 侧瓶颈）— https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html
21. TensorRT-LLM PR #12148, *Add host performance regression test suite for PyExecutor* — https://github.com/NVIDIA/TensorRT-LLM/pull/12148
22. FlashInfer, *Attention kernels API*（`plan()` 不能用于 CUDA Graph；fast_decode_plan）— https://docs.flashinfer.ai/api/attention.html
23. SGLang PR #10760 / #3987, issue #23500（plan 的每步 D2H 同步）— https://github.com/sgl-project/sglang/pull/10760
24. vllm-ascend 源码（本地）`csrc/CMakeLists.txt`、`csrc/aclnn_torch_adapter/`、`csrc/camem_allocator.cpp`、`setup.py` 的 `CMakeExtension`

**free-threading / subinterpreter**
25. Python docs, *Free-threaded CPython*（5–10%；pyperformance 1%~8%；**mimalloc vs pymalloc** 一节）— https://docs.python.org/3/howto/free-threading-python.html
26. Python docs, *Configure Python*（`--disable-gil` 与 `--without-mimalloc` 互斥、mimalloc 默认启用）— https://docs.python.org/3/using/configure.html
27. PEP 779, *Free-threaded Python is officially supported* — https://peps.python.org/pep-0779/
28. arXiv:2603.04782, *Unlocking Python's Cores: Hardware Usage and Energy Implications of Removing the GIL* — https://www.alphaxiv.org/overview/2603.04782
29. PEP 684 *A Per-Interpreter GIL* — https://peps.python.org/pep-0684/ ；PEP 734 — https://peps.python.org/pep-0734/
30. Python docs, *concurrent.interpreters* / *Sub-interpreters (C-API)* — https://docs.python.org/3/library/concurrent.interpreters.html

**PyPy / GraalPy**
31. PyPy v8.0.0 release notes（2026-09-19，3.12 beta）— https://doc.pypy.org/release-v8.0.0.html
32. PyPy FAQ（cpyext 慢、建议 CFFI）— https://doc.pypy.org/faq.html
33. numpy issue #30416, *MAINT: drop support for PyPy* — https://github.com/numpy/numpy/issues/30416
34. PyTorch issue #17835, *PyPy support* — https://github.com/pytorch/pytorch/issues/17835
35. GraalVM docs, *Python Version Compatibility* — https://www.graalvm.org/python/python-developers/docs/
36. GraalVM docs, *Native Extensions*（无 ABI 兼容、必须用自带 pip 重建）— https://www.graalvm.org/jdk23/reference-manual/python/Native-Extensions/
37. GraalPython CHANGELOG（25.1.0：Python 3.12.8、Torch 2.7.0 补丁）— https://github.com/graalvm/graalpython/blob/master/CHANGELOG.md

**CPython JIT / 版本**
38. What's New In Python 3.14, *tail-calling interpreter*（opt-in、Clang 19+、3–5%）— https://docs.python.org/3/whatsnew/3.14.html
39. K. Jin, *I'm Sorry for Python's tail-calling Interpreter's Results*（LLVM 19 bug）— https://fidget-spinner.github.io/posts/apology-tail-call.html
40. CPython `Doc/whatsnew/3.15.rst`（JIT 8–9% x86-64 / 12–13% AArch64 macOS）— https://github.com/python/cpython/blob/main/Doc/whatsnew/3.15.rst
41. PEP 836, *JIT Go Brrr: The Path to a Supported JIT Compiler for CPython*（4–12%、默认关闭、20% 目标）— https://peps.python.org/pep-0836/
42. Python Insider, *Python 3.15.0 candidate 1 is here!*（2026-08-04）— https://blog.python.org/2026/08/python-3150-rc1/
43. Python Insider / K. Jin, *Python 3.15's JIT is now back on track*（2026-03）— https://blog.python.org/2026/03/jit-on-track/

**本仓数据与同批文档**
44. `data/model/microbench-torch.csv`、`microbench-numpy.csv`（920B 单算子成本）
45. `data/cpython-alternatives/inproc-*.csv`、`s2-*.csv`、`cross-ab-*.json`、`compiled-cp312.csv`（本报告全部实测原始数据）
46. `docs/00-INDEX.md`、`docs/07-bottleneck-analysis.md`（真机 topdown / 热点 / 占比口径）
47. `docs/10-cpython-directions/00-FINDING-build-config.md`（生产镜像 Python 的构建参数事实链：
    `USE_COMPUTED_GOTOS=0`、无 PGO/LTO）与本目录 `01-interpreter-core.md`、`02-object-model.md`、`03-torch-dispatch.md`、`05-graph-dispatch-host-side.md`（同批调研）
