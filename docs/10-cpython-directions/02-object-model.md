# 10-02 对象模型 / 引用计数方向：哪些能真正降低 IPC 0.77 的派发密集负载成本

> 状态：**文献 + 源码 + 真机既有 perf 数据复核完成**（2026-09-24，Asia/Shanghai）。
> 本文不做新采集、不跑 NPU、不登录远端主机；所有"我们的负载"数字来自
> `data/profiles/real-b1/fp-scope/`（2026-09-23 20:31Z 真机采集，19 564 样本 / 0 lost）
> 与同目录下的源码行归因，另有少量**本机 x86_64 微基准**（CPython 3.12.10）用于机制验证。
> 文献类来源全部给 URL + 标题 + 时间，访问日期均为 **2026-09-24**。
>
> **本文的一条纪律**：公开数字只回答"理论上能省多少"，**不自动等于我们这份负载能省多少**。
> 每条技术最后都落到"对本负载适用性 + 证据强度"，并在 §3 给出排序。
>
> 分工（同目录姊妹篇）：[`01-interpreter-core.md`](01-interpreter-core.md) 管解释器核心
> （PEP 659 特化 / Tier 2 / JIT），[`03-torch-dispatch.md`](03-torch-dispatch.md) 管
> torch/LAL 层，[`05-graph-dispatch-host-side.md`](05-graph-dispatch-host-side.md) 管图下发；
> **本篇只管"对象模型侧"**：refcount 流量（immortal / deferred / BIC）、
> 分配器与 freelist、`__slots__`/dataclass、type cache 与属性查找、以及 GC 的取舍。
> 与 01 的那一点重叠（属性查找）在这里只写"对象模型角度"：
> **缓存被什么打穿、代码怎么写才命中**，特化机制本身不重复展开。

---

## 0. 结论速览

### 0.1 七条方向的判决

| # | 方向 | 在我们这套栈（CPython **3.12.13** / GIL 构建 / aarch64）能否落地 | 对本负载的预期收益 | 证据强度 |
|---:|---|---|---|---|
| 1 | Immortal objects（PEP 683） | **已内置，不可再挖掘**（3.12 起；无公开 API） | ≈0（我们画像里 INCREF 的 store 已只占 `_PyEval` self 的 ~11%） | 强（源码 + 本机验证） |
| 2 | Deferred / biased reference counting | **不能**（只在 free-threaded 构建；GIL 构建上是 no-op） | 0 | 强 |
| 3 | Free-threading（PEP 703） | **不能**（需 `cp313t/cp314t` 轮子；单线程自带 5–40% 税） | 负收益，且**不对症**（见 §1.3） | 强 |
| 4 | 分配器（pymalloc / mimalloc / freelist） | 3.13 默认构建**仍是 pymalloc**；换 mimalloc 需换解释器 | 上限 ~7%（分配/释放总占比），但换不来 | 中 |
| 5 | `__slots__` / `slots=True` / NamedTuple | **能**（纯代码改动） | **<0.05%/步** —— 不值得当性能手段 | 强（公开基准 + 本机实测） |
| 6 | 属性查找（type cache / LOAD_ATTR inline cache） | **能**（写代码方式 + 缓存友好） | **2–5%**（本负载最可操作的一项） | 中（真机 + 源码；缺 vLLM 级公开 benchmark） |
| 7 | GC（`freeze` / `disable` / 阈值） | 能，但 vLLM **已经做了** `gc.freeze()` | **≈0**（真机窗口内 GC 周期权重 = 0） | 强（真机 + 本机 + PR 证据） |

### 0.2 三条硬结论

1. **真机 `prepare_input` 窗口里，GC 的贡献是 0，不是 12% 也不是 54%。**
   对 20 s 真机采集的 55 442 828 817 周期（≈19.1 s × 2.9 GHz）逐个样本求和，
   含 `gc_collect_main` / `visit_decref` / `visit_reachable` / `gcmodule` 的样本
   **周期权重合计 = 0**。vLLM 的 `freeze_gc_heap()`（`EngineCore.__init__`，在
   `model_executor` 构造之后调用）已经把它解决掉了。
   → **GC 方向不要再投入**（除非换负载/换执行器）。

2. **源码行归属要修正两处**（详见 §1.3）：
   - `object.h:642 / :646`（真机占比 10.24% / 11.27% of `_PyEval_EvalFrameDefault`）
     = `Py_INCREF` 的 **add / store**；
   - `pycore_gc.h:59 / :60`（自检负载里 32.6% / 53.6% of `gc_collect_main`）
     = `_PyGCHead_NEXT()` 里的 `gc->_gc_next` 读取与返回，
     即 **GC 世代链表的指针追逐**，**不是**"引用计数字段访问"。
     该数据来自 perf 工具自检负载（`selftest_load.py`）而不是真机 vLLM。

3. **能落到 vLLM `prepare_input` 代码上的，只有"少造对象 / 少查属性 / 少派发算子"三类。**
   解释器级的四条（immortal / deferred RC / free-threading / 换分配器）在我们这套
   Ascend + CPython 3.12.13 上要么已生效、要么不可用，**把它们当优化项会浪费人力**。

---

## 1. 口径、换算与两处需要修正的既有结论

### 1.1 真机数据怎么用

| 项 | 值 | 说明 |
|---|---|---|
| 采集 | `data/profiles/real-b1/fp-scope/`（2026-09-23 20:31–20:33Z） | `perf record -e cycles -F 999 -g --call-graph fp -t <EngineCore TID>`，19 564 样本 / 0 lost |
| 线程内占比 | `hotspots.csv` 的 `self_pct` | 分母 = 该 TID 的全部样本（与 `docs/05` §2.2 同口径） |
| **折算到 `prepare_input`** | **线程占比 ÷ 0.55** | 采样归因显示 55.00% 的样本落在 `prepare_input` 窗口内（`docs/05` §1.1） |
| 源码行占比 | `hotspots_srcline.csv` 的 `pct` | 分母 = **该符号自身**的样本；不能跨符号比较 |
| 调用者归因 | 本文 §1.4 / 附录 B | 对 `folded.txt.gz` 按 perf period（周期）加权，取"最近的已符号化祖先帧" |

> ⚠️ 本采集开了 perf trampoline（栈里到处是 `py_trampoline_evaluator`），
> 它是**测量装置**而不是被测量对象：栈里 0.61%（线程口径）的
> `PyUnstable_Code_GetExtra` 就 100% 由 trampoline 调用，属于探针代价；
> 解释器侧符号的 **self% 也被 trampoline 抬高**（`docs/05` 已用 PMU/探针口径规避）。
> 因此本文只用它做**相对排序与调用链归因**，不引用其绝对值作为"优化后能省多少"。
>
> **两种分母不要混**：`docs/05` / `docs/10-cpython-directions/01-*.md` 常直接引用
> **线程口径**（符号占该 TID 全部样本的 %，如 `_PyType_Lookup` 1.67%），
> 本文 §1.2 与 §3 用的是**折算到 `prepare_input` 窗口内**的口径（÷0.55）。
> 两者不矛盾，差的是 1.82 倍；**谈"这段代码给 `prepare_input` 贡献多少"时必须用后者**。

### 1.2 `prepare_input` 窗口内的成本表（真机，按与对象模型相关的部分）

| 符号 | 线程 self%（样本） | ≈ `prepare_input` 内占比 | 属于哪一类 |
|---|---:|---:|---|
| `_PyEval_EvalFrameDefault` | 9.25（1 804） | 16.8% | 解释器派发 + refcount |
| ├ `object.h:642`（INCREF add） | —（占该符号 10.24%） | ≈1.7% | **refcount 流量** |
| ├ `object.h:646`（INCREF store） | —（占该符号 11.27%） | ≈1.9% | **refcount 流量** |
| `eventfd_write`（kernel） | 2.02（396） | 3.7% | CANN 算子提交（见 §1.3） |
| `_PyType_Lookup` | 1.61（315） | 2.9% | **属性查找** |
| `pthread_mutex_lock` | 1.55（302） | 2.8% | CANN 内部锁 + GIL drop/acquire |
| `_PyObject_Malloc` | 1.53（299） | 2.8% | **小对象分配** |
| `unicodekeys_lookup_unicode` | 1.41（277） | 2.6% | dict(kwargs/属性 dict) 查找 |
| `_PyObject_GenericGetAttrWithDict` | 1.28（252） | 2.3% | **未特化的属性查找** |
| `_PyObject_Free` + `malloc` + `cfree` | 1.01 + 1.00 + 0.45 | 4.5% | **分配/释放** |
| `initialize_locals` | 0.80（156） | 1.5% | **函数调用建帧（每次调用都要 INCREF 参数）** |
| `_Py_dict_lookup` | 0.78（153） | 1.4% | dict 查找 |
| `PyObject_Vectorcall` + `_PyFunction_Vectorcall` | 0.62 + 0.71 | 2.4% | **调用派发** |
| `_PyObject_GetMethod` | 0.53（104） | 1.0% | **方法查找** |
| `PyObject_GetAttr` | 0.48（93） | 0.9% | C 侧属性查找 |
| `tupledealloc` / `tuple_alloc` | 0.51 / 0.37 | 1.6% | tuple churn（含 kwargs） |
| `PyType_IsSubtype` | 0.39（77） | 0.7% | `isinstance`/类型检查 |
| `gc_collect_main` | **0** | **≈0** | GC（freeze 已生效） |

### 1.3 两处需要修正的既有结论

#### 修正 1：`pycore_gc.h:60` 不是"引用计数访问"，那份数据也不来自真机 vLLM

CPython **3.12.13** 源码（`Include/internal/pycore_gc.h`）：

```c
58: static inline PyGC_Head* _PyGCHead_NEXT(PyGC_Head *gc) {
59:     uintptr_t next = gc->_gc_next;          // ← 自检数据里 32.6%
60:     return _Py_CAST(PyGC_Head*, next);      // ← 自检数据里 53.6%
61: }
```

该函数在 `Modules/gcmodule.c` 里以 `#define GC_NEXT _PyGCHead_NEXT` 被
`update_refs()` / `move_unreachable()` / `gc_list_merge()` 等**每一个链表遍历**使用。
所以 `gc_collect_main` 的热点指令（`ldr x4, [x3]` 一类）是
**遍历 GC 世代链表时的指针追逐（cache miss）**，不是引用计数读写。

数据出处也要更正：`pycore_gc.h:60 = 53.60%`、`:59 = 32.60%` 来自
`data/profiles/perf-selftest-20260923T180711Z/fp_notramp/hotspots_srcline.csv`，
其负载是 **perf 工具自检脚本** `selftest_load.py`（单线程 dict/属性/字符串 churn，
没有调用 `gc.freeze()`），不是真机 vLLM 的 `prepare_input`。
真机 `real-b1/fp/hotspots_srcline.csv` 里根本**没有** `gcmodule.c`/`pycore_gc.h` 行。

> 结论不变（"少造对象"仍然对），但机理不同：
> GC 成本 ∝ **收集时刻存活的可跟踪对象数**（链表长度）+ 内存局部性，
> 而不是 ∝ 对象上发生的 refcount 次数。这条差别决定了 §3 的排序里 GC 排最后。

#### 修正 2：`eventfd_write` 与 `pthread_mutex_lock` 不是"GIL/进程通信"，主要是 CANN 算子开销

对真机 `folded.txt.gz` 按周期加权，取最近的已符号化祖先帧：

| 符号 | 调用者（周期加权） |
|---|---|
| `eventfd_write` | `at_npu::native::OpCommand::RunOpApiV2` **63.1%**；torch 分发器上 `copy_`（ADInplaceOrView）**17.8%**；`c10_npu::NPUEvent::record` **13.2%**；`NPUEvent::~NPUEvent` 2.6%；`OpCommand::Run` 2.1% |
| `pthread_mutex_lock` | `take_gil` **16.5%** + `drop_gil` **10.0%**（GIL 释放/重取）；其余 ≈63% 是 `c10_npu::MaybeSetDevice` 14.8%、`op::internal::GetOpProfilingRecordArgFlag` 8.9%、`OpCommand::RunOpApiV2` 5.1%、`OpPreparation::apply_tensor_without_format` 4.6%、`Adx::DumpManager::IsEnableDump` 4.0%、CANN runtime 若干 |

也就是说：

* `eventfd_write` ≈ **每提交一个 NPU 算子就要 write 一次 eventfd**（CANN 的 op 命令/事件机制）；
* `pthread_mutex_lock` 里约 1/4 是 **GIL 的 drop/acquire**（栈上表现为
  `THPVariable_dealloc → THPVariable_clear → PyEval_SaveThread → drop_gil`
  以及 ndarray 路径），另 3/4 是 CANN runtime 内部锁。

**这两条都不是"Python 多线程被 GIL 卡住"，而是"Python 侧算子条数太多"的副产品。**
把 GIL 拿掉（PEP 703）不会让 CANN 的锁和 eventfd 消失，
反而要额外付 free-threaded 构建的单线程税 —— 所以 §3 把"减少算子条数"排在前面。

### 1.4 本机微基准（能证明什么、不能证明什么）

| 项 | 值 |
|---|---|
| 环境 | 开发机 x86_64、CPython **3.12.10**（conda-forge），**不是** Kunpeng 920B、不是镜像里的 3.12.13 |
| 用途 | 只验证**机制与相对关系**（slots 省什么、GC 链表写在哪、refcount 是否可观测） |
| 不用于 | 绝对百分比、ARM 微架构结论 |

方法：`python3 - <<'PY'` 内联脚本（不落盘）；计时取 5 次最优；
GC 计时用 `gc.callbacks` 累计每次 collection 的 wall。

---

## 2. 逐条技术调研

### 2.1 Immortal objects（PEP 683，3.12+）

**名称**：Immortal Objects, Using a Fixed Refcount（PEP 683）

**来源**

| 类型 | 链接 / 标题 | 时间 |
|---|---|---|
| 规范 | https://peps.python.org/pep-0683/ 《PEP 683 – Immortal Objects, Using a Fixed Refcount》 | 提案 2022-02-10，Final（3.12） |
| 实现 | https://github.com/python/cpython/pull/19474 《gh-84436: Implement Immortal Objects》 | 2022–2023；PR 讨论内含 pyperformance 数据 |
| 更新 | https://discuss.python.org/t/pep-683-immortal-objects-updates/23382 《PEP 683: Immortal Objects: Updates》 | 2023-01-31 |
| 工程收益 | https://engineering.fb.com/2023/08/15/developer-tools/immortal-objects-for-python-instagram-meta/ 《Introducing Immortal Objects for Python》 | 2023-08-15 |
| 源码 | CPython v3.12.13 `Include/object.h`（L110/125 `_Py_IMMORTAL_REFCNT`、L239–247 `_Py_IsImmortal`、L624–659 `Py_INCREF`、L696– `Py_DECREF`）、`Include/internal/pycore_object.h`（L76 `_Py_SetImmortal`） | v3.12.13 |

**核心机制**（3.12.13 原文）

* 64 位平台上 `_Py_IMMORTAL_REFCNT = UINT_MAX`，`_Py_IsImmortal(op)` 就是判断
  `(int32)op->ob_refcnt < 0`（即低 32 位是 `0xFFFFFFFF`）。
* `Py_INCREF` 在 64 位上做的是**饱和 32 位加法**：
  `cur = op->ob_refcnt_split[...]; new = cur + 1; if (new == 0) return; 写回;`
  —— 对象 immortal 时在"进位为 0"处返回，**省掉一次 store**。
* `Py_DECREF` 先 `if (_Py_IsImmortal(op)) return;`，再 `--refcnt`（省掉 store 和 dealloc 判定）。
* 启动即 immortal 的对象（3.12.13）：静态类型（`PyObject_HEAD_INIT` 直接用 immortal 值）、
  `None/True/False/NotImplemented/Ellipsis`、小整数 `-5..256`、
  被 intern 的标识符字符串（`Objects/codeobject.c:145` 对 code 的字符串常量调
  `_PyUnicode_InternImmortal`）、以及若干 C 侧静态对象。
* **3.12 没有把"启动时的整个堆"immortal 化**（那个激进版本在评审中被去掉）；
  3.13t 里"临时 immortal 化"只发生在 free-threaded 构建（见 §2.2）。

**公开实测收益**

* PEP 683 正文：naive 实现让 CPython **慢 4–6%**（Python 3.12 语言峰会定的可接受上限是 2%）；
  加上缓解措施后目标"性能中性"。
* PR #19474 讨论：2023-04 用 GCC 11.1 测得 **pyperformance 几何平均 1.02–1.03x slower**；
  早期版本最差到 **1.04x slower**（`unpack_sequence` 1.42x slower 等）。
* 真正被兑现的收益是 **内存/COW**：Meta 的结论是私有内存下降、共享内存上升
  （pre-fork 场景），并明确说"核心实现要在最热的两条路径上加检查，这必然带来性能退化"。

**对本负载适用性**

* 我们跑的就是 3.12.13 → **immortal 机制已经在生效**，没有"再打开"的开关。
* 本机验证（3.12.10）：`sys.getrefcount(None/True/3/Ellipsis/NotImplemented/'a'/intern('abc'))`
  全部 = **4294967295**；`[]` = 3；`sys` 模块 = 63。
  也就是说：**代码里写死的属性名、常量、小整数已经在省 store**。
* **没有用户可用的 immortal API**：`_Py_SetImmortal()` 在 `Include/internal/pycore_object.h`
  里（需要 `Py_BUILD_CORE`），不属于公开 C-API，纯 Python 侧更不可能；
  3.14 给扩展作者的是 `PyUnstable_Object_EnableDeferredRefcount`（见 §2.2），**不是** immortal。
* 收益上限估计：真机 `_PyEval_EvalFrameDefault` 的 self 样本里，INCREF 的 add+store 合计
  ~21.5%（→ ≈3.6% of `prepare_input`）。即使把"所有还能 immortal 的对象"都 immortal 化，
  也只能吃掉其中一部分，而且**做不到**。

**证据强度：强**（源码逐行 + 本机可复现的 `getrefcount` 观测 + 官方 PEP/PR 数字）。

---

### 2.2 Deferred / 延迟引用计数（含 biased reference counting）

**名称**：biased reference counting（BRC）/ deferred reference counting（DRC）/ per-thread refcount

**来源**

| 类型 | 链接 / 标题 | 时间 |
|---|---|---|
| 规范 | https://peps.python.org/pep-0703/ 《PEP 703 – Making the Global Interpreter Lock Optional in CPython》 | 2023-01-09 |
| BRC 实现 | https://github.com/python/cpython/issues/110481 《Implement biased reference counting in --disable-gil builds》+ https://github.com/python/cpython/pull/110764 | 2023-10 |
| 线程间队列 | https://github.com/python/cpython/issues/114824 《gh-110481: Implement inter-thread queue for biased reference counting》 | 2023-11 |
| DRC 实现 | https://github.com/python/cpython/issues/117376 《Implement deferred reference counting in free-threaded builds》 | 2024-03-29 |
| 默认构建提案 | https://github.com/python/cpython/issues/120024 《Deferred reference counts》（Mark Shannon） | 2024-06 |
| 早期设计 | https://github.com/faster-cpython/ideas/issues/677 《Deferred reference counts.》 | — |
| 3.13t 权宜 | https://github.com/python/cpython/issues/117783 《Temporarily immortalize objects that use deferred reference counting》 | 2024-04-11 |
| 3.13t 副作用 | https://discuss.python.org/t/reference-leaks-in-free-threaded-3-13-version/61467 《Reference leaks in free-threaded 3.13 version》 | 2024-08-21 |
| API | https://docs.python.org/3.14/c-api/object.html `PyUnstable_Object_EnableDeferredRefcount` | 3.14（2024-11 合入） |
| 现状综述 | https://vstinner.github.io/free-threading-deferred-reference-counting.html 《Free Threading internals: deferred reference counting》 | 2026-06-04 |
| 类型清单 | https://docs.python.org/3/howto/free-threading-python.html 《Python support for free threading》 | 3.14 文档 |

**核心机制**

* **BRC**：每个对象有 `ob_tid`（属主线程）+ `ob_ref_local`（属主线程非原子计数）
  + `ob_ref_shared`（其他线程原子计数）；仅属主线程用快路径，跨线程走慢路径/队列。
* **DRC**：给对象打标记后，**解释器栈上的引用不再计数**（帧的 `localsplus[]` 里存 deferred 引用），
  对象改由 GC 回收；PEP 703 举的类型是模块、顶层函数、方法、code object、descriptor、heap type。
  3.14 文档补充 `threading.local`，并要求对象必须被 GC 跟踪。
* **per-thread refcount**：少数共享热点类型（heap type/type 对象等）用"每线程一份计数数组"，
  真值求和才回收。
* 3.13t 因为 DRC 没赶上特性冻结，**先把这些对象临时 immortal 化**，代价是
  "动态建类/建函数会泄漏内存"（3.14 用 DRC 取代）。

**公开实测收益**

* PEP 703 的设计表（对标"3.12 immortal 分支"）：**单线程开销 6%（Intel Skylake）/ 5%（AMD Zen 3），
  多线程 8% / 7%** —— 表里报的是**开销**，收益只在多线程扩展性上。
* 3.14：https://docs.python.org/3/whatsnew/3.14.html 说 PEP 703 的实现在 3.14 完成，
  free-threaded 单线程税降到 **~5–10%**。
* 官方文档明确：`PyUnstable_Object_EnableDeferredRefcount()`
  "In the free-threaded build, this allows the interpreter to avoid reference count adjustments…
  **It does nothing on builds with the GIL enabled, which do not support deferred reference counting.**"
* 重要边界（nanobind 维护者与 CPython 开发者的问答，https://github.com/wjakob/nanobind/issues/1009 ，2025-04-11）：
  DRC **只绕过解释器栈上的 INCREF/DECREF**；C 扩展里的 `Py_INCREF/Py_DECREF` 一个都不省。

**对本负载适用性**

* 我们的解释器是 **GIL 构建的 3.12.13**：BRC/DRC/per-thread 计数**全部不存在**；
  `PyUnstable_Object_EnableDeferredRefcount` 在这个构建上是 no-op。
* 即使未来迁到 free-threaded 3.14t：我们画像里最热的 refcount 流量在
  `_PyEval_EvalFrameDefault`（会受益）**和** torch/torch_npu 的 C 扩展（不受益），
  并且要额外承担 5–10% 单线程税 + CANN 侧不兼容风险。
* Mark Shannon 的 gh-120024 想把 DRC 带到**默认构建**（"≈80% 的 refcount 操作发生在解释器里"），
  但截至 2026-09 查不到它进入任何已发布版本；3.14 的 DRC 仍然只服务 free-threading。

**证据强度：强**（官方文档 + 官方 issue/PR + 3.14 文档的逐字表述）。
**判决：不可用，不作为方向。**

---

### 2.3 Free-threading / 去掉 GIL（PEP 703，3.13+）

**来源**

| 类型 | 链接 / 标题 | 时间 |
|---|---|---|
| 规范 | https://peps.python.org/pep-0703/ 《PEP 703》 | 2023-01-09 |
| 3.13 文档 | https://docs.python.org/3.13/howto/free-threading-python.html 《Python experimental support for free threading》 | 3.13 |
| 3.14 文档 | https://docs.python.org/3/whatsnew/3.14.html 《What's New In Python 3.14》 | 3.14.0（2025-10） |
| 3.14 howto | https://docs.python.org/3/howto/free-threading-python.html | 3.14 文档 |
| 生态成本 | https://labs.quansight.org/blog/free-threaded-python-rollout 《Free-threaded CPython is ready to experiment with!》 | 2024-07-12 |
| ABI/轮子 | https://pypi.org/pypi/torch/json（2026-09-24 查询：2.14.0 有 cp313/cp314 的 aarch64 轮子，**没有 cp313t/cp314t**） | 查询于 2026-09-24 |

**核心机制**：GIL 移除后，`Py_INCREF/DECREF` 的互斥由 BRC/原子操作/per-object lock 承担；
容器读操作走"乐观访问 + 延迟回收 mimalloc 页"等方案；解释器栈引用走 DRC；
另外 stop-the-world、GC、QSBR 之类机制都会被引入。

**公开实测收益（单线程）**

* 3.13：官方文档写 **"In 3.13, this overhead is about 40% on the pyperformance suite"**。
* 3.14：What's New 写 **"roughly 5–10%, depending on the platform and C compiler"**；
  howto 写 **"about 1% on macOS aarch64 to 8% on x86-64 Linux systems"**。
  → 我们关心的 aarch64 **Linux**（Kunpeng）落在哪个点，**没有任何公开数字**（见 §4）。
* PEP 703 表格（较早基线）：单线程 +5~6%，多线程 +7~8% 开销，收益只在多线程扩展性。

**对本负载适用性**

* 症状对不上：我们窗口里的锁/事件是 **CANN 的 per-op 开销**（§1.3），
  不是"多个 Python 线程抢 GIL"。free-threading 会让 `eventfd_write`
  （63% 来自 `OpCommand::RunOpApiV2`）和 `c10_npu::MaybeSetDevice` 等锁**原样保留**。
* 依赖链对不上：vLLM 的 Ascend 栈要求 torch + torch_npu + CANN 的特定组合；
  PyPI 上 torch 2.14.0 连 `cp313t` 都没发（更不用说 torch_npu），
  而 free-threaded 构建与默认构建 **ABI 不兼容**（Quansight 文中已说明要单独出轮子）。
* 结构对不上：vLLM v1 本来就是 **多进程**（EngineCore / Worker），
  真正需要的是"少发指令"，不是"同进程多线程并行跑 Python"。

**证据强度：强（"不适用"这个判断）**；
**aarch64 Linux 上具体能省/亏多少：查不到（§4）**。

---

### 2.4 分配器：pymalloc / mimalloc / freelist

**来源**

| 类型 | 链接 / 标题 | 时间 |
|---|---|---|
| 任务书里的说法需要更正 | https://docs.python.org/3.14/c-api/memory.html 《Memory Management》：默认（GIL）构建 `PyObject_Malloc = pymalloc`；free-threaded 构建才默认为 mimalloc；默认构建可用 `PYTHONMALLOC=mimalloc` 选择 | 3.14 文档 |
| 3.13 发布说明 | https://docs.python.org/3/whatsnew/3.13.html "CPython now bundles the mimalloc library by default." | 3.13 |
| 3.13 源码 | `Objects/obmalloc.c`：`#if defined(Py_GIL_DISABLED) … PYOBJ_ALLOC = MIMALLOC_OBJALLOC; #elif defined(WITH_PYMALLOC) … PYOBJ_ALLOC = PYMALLOC_ALLOC` | v3.13.7 |
| 原型 benchmark | https://bugs.python.org/issue46657 《Issue 46657: Add mimalloc memory allocator》（含 2022-02 的 pyperformance 对照：带 freelist 几何平均 **1.01–1.02x faster**） | 2022-02 |
| 集成 PR | https://github.com/python/cpython/pull/109914 《gh-90815: Add mimalloc memory allocator》 | 2023-09-26 |
| 内存代价 | https://github.com/python/cpython/issues/135153 《Increased memory usage with mimalloc》：tomllib 场景 RSS 131 MB(pymalloc) → 197 MB(mimalloc)，`MIMALLOC_PURGE_DELAY=0` 可拉到 72 MB 但更慢 | 2025 |
| 3.12 行为 | `Objects/obmalloc.c`（pymalloc：1 MiB arena / 512 B 阈值）、`Include/internal/pycore_freelist.h` 等 | v3.12.13 |

**核心机制**

* **3.12**：对象域（`PyObject_Malloc`）走 pymalloc（≤512 B 分池 + size class freelist），
  大块回落到 glibc `malloc`；此外 float/tuple/dict/list 等有自己的 **object freelist**
  （`_PyFreeListState`），所以"分配/释放一个短命小对象"通常只是**弹一次链表**。
* **3.13+ 默认（GIL）构建仍然是 pymalloc**；mimalloc 被 vendored 且默认编入，但只是
  `PYTHONMALLOC=mimalloc` 可选。**"pymalloc 被 mimalloc 取代"只在 free-threaded 构建成立。**
* mimalloc 相对 pymalloc 的差异：per-thread heap、段更大、**不急于把内存还给 OS**，
  以及（在 FT 构建里）QSBR 延迟回收 → 内存放大是已知代价。

**公开实测收益**：默认构建下 mimalloc 的公开数字只有 2022 年的原型数据（**几何平均 ~1–2%**，
个别项 +9%、个别项 -8%），此后没有官方"默认构建启用 mimalloc"的性能声明；
反而有 RSS 放大 1.5x 的记录（更早的原型 gh-46657 里也有 `pickle_dict` 1.08x slower 之类）。

**对本负载适用性**

* 真机窗口里分配/释放合计 ≈ **7.3% of `prepare_input`**
  （`_PyObject_Malloc` 2.8% + `_PyObject_Free` 1.8% + `malloc` 1.8% + `cfree` 0.8%）。
  这是"能碰到的天花板"，但**换分配器够不到它**：
  换 3.13/3.14 默认构建仍然是 pymalloc；手动 `PYTHONMALLOC=mimalloc` 在我们这套
  Ascend 镜像里没有验证过，且引入 RSS 放大风险。
* 本机微基准（3.12.10）：`object.__new__` 34.9 ns、`{}` 15.6 ns、`(1,2,3)` 7.6 ns、`[]` 17.3 ns、
  `str(i)` 34.2 ns —— 分配本身便宜，**成本在"条数"**：一条路径上少造 3 个对象，
  省的是 3×(分配 + 头部初始化 + 之后的 refcount + 可能的 GC 跟踪)。
* **系统性手法（可落地部分）**：
  1. **结构化复用**：per-step 只改变化字段/用预分配 buffer，而不是每步重建整对象
     （`AscendGDNAttentionMetadataBuilder.__init__` 已经预分配了 5 个容量缓冲，
     真机 perf 却没看到 `torch.empty_like` 的热点 —— 说明"预分配"这条**已经被做了**，
     剩下的是 metadata 对象本身的 churn）；
  2. **array-of-structs / 纯 int 数组**：把 CPU 侧的小张量算术换成 Python int 或
     `array`/numpy 视图（`docs/07` 已定位 `compute_num_computed_tokens` 的 4 次张量相减）；
  3. **让对象不被 GC 跟踪**（见 §2.7）：只装原子值（int/str/bool）的 dict/tuple
     会被 CPython 主动 untrack，不进入 gen0 计数。

**证据强度：中**（版本/默认值有官方文档和源码；"换分配器对我们能省多少"无任何公开数据）。

---

### 2.5 `__slots__` / `dataclass(slots=True)` / `NamedTuple`

**来源**

| 类型 | 链接 / 标题 | 时间 |
|---|---|---|
| 覆盖面最广的对照 | https://semolex.online/post/python-optimize-object-creation/ 《Python: Optimize Dynamic Object Creation》（pytest-benchmark 表） | — |
| 内存 | https://maurodec.com/blog/classes-namedtuples-slots/ 《Python Classes, namedtuples and __slots__》 | 2018-04-29 |
| 内存（含 3.10 debug 对照） | https://chezsoi.org/lucas/blog/slots-memory-optimizations-in-python.html 《__slots__ memory optimization in Python》 | 2023-03-28 |
| 创建/内存矩阵 | https://github.com/PokkaKiyo/python-dataclasses-comparison | 2024-12-22 |
| 1M 实例计时 | https://dev.to/jneeee/dataclasses-have-better-performace-than-namedtuple-1bo2 | 2025-04-04 |

**公开实测收益（摘录）**

| 对象类型 | 属性访问 ns | 创建 ns | 内存/实例 |
|---|---:|---:|---:|
| NamedTuple | 141.6 | 291.6 | 168 B |
| `__slots__` 类 | 149.0 | 288.0 | 160 B |
| `@dataclass(slots=True)` | 149.6 | 356.7 | 160 B |
| `@dataclass` | 166.1 | 419.3 | 592 B |
| 普通类 | 166.6 | 440.1 | 592 B |

（前四列来自 semolex；内存列来自 PokkaKiyo 的对照表；maurodec 的 100 万实例 RSS 是
dataclass 26.7 MiB vs slots 17.2 MiB，**-35%**；chezsoi 是 28.3 → 22.0 MiB，**-22%**。）

**本机实测（x86_64 / CPython 3.12.10 / 24 字段 dataclass）**

| 指标 | `@dataclass` | `@dataclass(slots=True)` | 比值 |
|---|---:|---:|---:|
| `sys.getsizeof(instance)` | 48 B（+ `__dict__` 296 B = **344 B**） | **224 B** | -35% |
| 创建 | 466.2 ns | **250.5 ns** | 1.86x |
| 属性 ×8 | 114.6 ns | **60.9 ns** | 1.88x |
| `dataclasses.replace(o, f0=1)` | 2885.0 ns | **2558.5 ns** | 1.13x |
| 生成的 `__init__` `co_code` 长度 | 340 B | 340 B | **相同** |

**诚实结论（这是任务书特别要求的一点）**

* `slots=True` **不减少引用计数操作的条数**：每个字段仍然是一次 `LOAD_FAST` + 一次 `STORE_ATTR`，
  每次 `LOAD_ATTR` 仍然要对取出的值 INCREF。省下来的是
  **实例 `__dict__` 的分配 + 每次属性写入的 dict 插入 + 每次属性读取的 dict 查找**
  （以及由此带来的 `unicodekeys_lookup_unicode` / `_PyObject_Malloc` 份额），
  外加 35% 的内存。
* 因此对"每步新建 3 个 22 字段数据类"这种模式：
  **收益量级 = 3 × ~0.2 µs ≈ 0.6 µs/步 ≈ 0.01–0.02% of `prepare_input`**（真机步长 4915 µs）。
  这属于"免费顺手做"，**不是性能手段**。
* `NamedTuple` 在我们这里**不适用**：`GDNAttentionMetadata` 的字段在构造后被逐条赋值
  （`vllm_ascend/ops/gdn_attn_builder.py` 里的
  `attn_metadata.non_spec_prefill_metadata = …`、`_attach_*_metadata()` 都在改对象），
  tuple 子类做不到。
* `frozen=True` 同样**会直接破坏现有代码**（构造后赋值会抛 `FrozenInstanceError`），
  除非同时把 builder 改成"先算齐所有字段再一次构造"。而 frozen 本身的读路径
  与普通 dataclass 没差别（读 slot/dict 一样 INCREF），所以收益也是 0。

**证据强度：强**（公开基准 + 本机可复现实验）；**但"对 IPC/refcount 流量无帮助"这个否定结论
同样强**。

---

### 2.6 属性查找：`_PyType_Lookup` type cache 与 `LOAD_ATTR` inline cache

**来源**

| 类型 | 链接 / 标题 | 时间 |
|---|---|---|
| 官方文档 | https://docs.python.org/3/library/dis.html 与 https://peps.python.org/pep-0659/ 《PEP 659 – Specializing Adaptive Interpreter》 | 3.11+ |
| 源码（3.12.13） | `Objects/typeobject.c` L4731–4775（`type_cache` 命中判定/失效/写入）、L43–49（`MCACHE_HASH`、`MCACHE_HASH_METHOD`、`MCACHE_CACHEABLE_NAME`）、`Include/internal/pycore_typeobject.h` L29–39（`MCACHE_SIZE_EXP 12` → 4 096 项）、`Python/specialize.c` L760–1130（LOAD_ATTR 各特化路径与失败原因） | v3.12.13 |
| 命中率证据 | https://github.com/python/cpython/pull/93988 《gh-93911: Specialize LOAD_ATTR for custom __getattr__ and __getattribute__》：`test_typing` 上 LOAD_ATTR 命中率 **+15%**（该模块几乎所有类都有 `__getattr__`） | 2022-06-18 |
| miss 代价 | https://github.com/python/cpython/issues/92216 《Performance of attribute lookup for type objects》：`hasattr(Class, '__array_ufunc__')` 255 ns vs `hasattr(Class,'all')` 38 ns | 3.12 |
| 真机证据 | `data/profiles/real-b1/fp-scope/hotspots.csv` + `hotspots_srcline.csv` | 2026-09-23 |

**核心机制（3.12.13，逐行读源码得到）**

1. **两条独立缓存**：
   * **特化（inline cache）**：字节码 `LOAD_ATTR` 被就地改写成
     `LOAD_ATTR_INSTANCE_VALUE / WITH_HINT / SLOT / MODULE / CLASS / PROPERTY /
     GETATTRIBUTE_OVERRIDDEN / METHOD_*`，缓存里存 `tp_version_tag`（+ 实例 dict 的 keys version / 偏移）。
     守卫是"type version 不变"，**不命中就 deopt 回通用路径**。
   * **type cache（`_PyType_Lookup`）**：per-interpreter 哈希表，4 096 项，
     `h = (tp_version_tag ^ (name 指针 >> 3)) & 0xFFF`；
     命中条件是 **`entry->version == type->tp_version_tag && entry->name == name`**，
     其中 `entry->name == name` 是**指针相等**；只有"精确 str、已 ready、长度 ≤ 100"的名字才缓存；
     一个桶只放一项，冲突即淘汰。
2. **最容易踩的两个坑**：
   * **名字对象不稳定** = 缓存必然 miss。`getattr(o, f"field_{i}")`、每次新建同值字符串、
     `functools.partial` 之后拼名字，都会让 `name` 指针变化 → 换桶 → miss → 走
     `find_name_in_mro()`（MRO 逐类 dict 查找）。
   * **运行时改类** = 版本标签失效。`setattr(cls, ...)`、动态建类、monkeypatch、
     `__set_name__` 等都会 bump/失效 `tp_version_tag`，让该类型的缓存项全部作废，
     并让已经特化的 `LOAD_ATTR` **deopt**。
3. **方法查找与 dunder 查找是独立成本**：`_PyObject_GetMethod`（`obj.m(...)` 走 METHOD 家族）、
   `_PyObject_LookupSpecial`（`__enter__/__exit__/__len__/__add__/__iter__` 等每次都要查类型）
   都会落到 `_PyType_Lookup`。
4. **`__getattr__` / `__getattribute__` 的类**（例如 `torch.nn.Module`）走
   `_Py_slot_tp_getattr_hook` / `LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN`，
   丢属性时最终回到 `_PyObject_GenericGetAttrWithDict` —— 这正是真机里那个 1.28% 符号。

**真机证据（本文最有用的一组）**

* `_PyType_Lookup`：线程 self **1.61%（315 样本）→ ≈2.9% of `prepare_input`**；
  其**符号内**样本分布：`typeobject.c:4734`（缓存命中判定与 entry 读取）**64.37%**、
  `:4726`（函数入口）9.25%、`:4739`（返回 entry->value）6.67%、
  `:4747`（miss → `find_name_in_mro`）4.73%、`:4772/:4775`（写缓存/返回）4.18% + 5.69%。
  → **热点是"查缓存表本身"（表 4 096 × 24 B ≈ 98 KB，超出 L1），不是走 MRO。**
* 调用者（周期加权）：`_PyObject_GenericGetAttrWithDict` 最大，其后是 `_Py_dict_lookup`、
  `find_name_in_mro`、`_PyObject_GetMethod`、`_PyDict_GetItem_KnownHash`、
  `_PyObject_GenericSetAttrWithDict`、`_PyObject_LookupSpecial`、`_Py_type_getattro`、`slot_tp_init`。
  → 也就是说：**成本主要来自"通用属性路径 + 方法查找 + dunder 查找 + 类属性查找"，
  而不是"某个 Python 类属性没加 slots"**。
* 与它配对的还有 `_PyObject_GenericGetAttrWithDict` **1.28% → 2.3%**、
  `PyObject_GetAttr` 0.48% → 0.9%、`_PyObject_GetMethod` 0.53% → 1.0%。
  三者合计 ≈ **4.2% of `prepare_input`** 花在"属性/方法查找"上。

**怎么写代码能让缓存更容易命中（可操作清单）**

1. **把名字写成字面量**（进 `co_names`，启动即 intern 且 3.12 里是 immortal）——
   不要在热循环里用 f-string 或拼接生成属性名；确实要动态查，先 `sys.intern()` 一次并复用。
2. **不要运行时改类**：热路径上别 `cls.attr = ...`、别在循环里建类/加方法；
   需要缓存就用模块级 dict 或实例属性。
3. **把属性查找提到循环外**：`m = self._compute` 一次，循环里用 `m(...)`；
   `torch.foo` → 模块级 `from torch import foo`；`self.cfg.x` → 局部变量。
   这同时减少 `LOAD_ATTR`/`LOAD_GLOBAL` 条数与其 refcount 流量。
4. **少用 `getattr`/`hasattr`/`dunder` 反射**：`_PyObject_LookupSpecial` 每次都要查类型；
   `hasattr` 在 miss 时（例如 `Class.__array_ufunc__`）公开基准显示会慢 **6.7 倍**（255 vs 38 ns）。
5. **避免"通用对象"**：`nn.Module` 子类属性访问要走 `__getattr__` hook；
   能提前把需要的值算成普通本地变量/`__slots__` 对象，就别在每步里访问 module 属性。
6. **注意 kwargs**：真机里 `unicodekeys_lookup_unicode` 1.41% → 2.6%，
   与 `kwargs` 传递/解包相关（`**kwargs` 建 dict + 每次查名字），
   热路径函数尽量用位置参数或已绑定的少量关键字。

**证据强度：中**（3.12.13 源码逐行 + 真机符号/调用者归因都是强的；
但**查不到**"vLLM 级别"或"22 字段 dataclass 每步新建"这类公开 benchmark 数字 —— 见 §4）。

---

### 2.7 GC：`gc.freeze()` / `gc.disable()` / 阈值

**来源**

| 类型 | 链接 / 标题 | 时间 |
|---|---|---|
| 官方文档 | https://docs.python.org/3/library/gc.html 《gc — Garbage Collector interface》（`freeze`/`unfreeze`/`get_freeze_count`；fork 场景说明） | 3.14 文档 |
| 设计动机 | https://bugs.python.org/msg302780（Łukasz Langa 提议 `gc.freeze()`） | 2017 |
| 生产数据 | https://medium.com/instagram-engineering/copy-on-write-friendly-python-garbage-collection-ad6ed5233ddf 《Copy-on-write friendly Python garbage collection》（先 `gc.disable()` 换来 **+10% 容量**，但 3 000 请求内存 +600 MB 且线性增长；最终改用 freeze） | 2017-12-20 |
| 生产数据 | https://making.close.com/posts/taming-the-python-gc/ 《Taming Python GC for Faster Web Requests》：`gc.collect(2)+freeze()` 后 GC 仍占 3%，再把 `threshold0` 700 → 50 000 后降到 **0.5%**，p95 -80~100 ms | 2024-10-23 |
| 生产数据 | https://mkennedy.codes/posts/python-gc-settings-change-this-and-make-your-app-go-20pc-faster/ 《20% Faster Python with a Single GC Tweak》 | 2022-11-17 |
| vLLM 已做 | https://github.com/vllm-project/vllm/pull/24008 《[Perf] Freeze core engine proc heap after init》（2025-09-04 合入）；https://github.com/vllm-project/vllm/commit/b30372c（#27896，把 freeze 从 `EngineCoreProc` 移到 `EngineCore`） | 2025 |
| vLLM 源码 | `refs/vllm/vllm/utils/gc_utils.py::freeze_gc_heap()`；`refs/vllm/vllm/v1/engine/core.py` L125（`model_executor` 构造）→ L235（`freeze_gc_heap()`） | 0.26.0 |
| 3.14 变化 | https://docs.python.org/3.14/whatsnew/3.14.html：3.14.0–3.14.4 曾上线**增量 GC**，因生产内存压力报告在 **3.14.5 回退**回分代 GC | 3.14.5 |

**核心机制**

* 三代分代：阈值默认 `(700, 10, 10)`；**gen0 计数 = 分配数 − 释放数**，
  所以"创建后马上销毁"的短命对象**不会**填满 gen0；只有"跨过一次分配检查仍存活"的对象才算。
* **full（gen2）收集是启发式的**：`Modules/gcmodule.c` 注释明确写着
  "只有 `long_lived_pending / long_lived_total > 25%` 才触发 full collection"，
  因为它的成本 ∝ **长期存活对象总数**。
* `gc.freeze()` 把当前所有被跟踪对象移到 **permanent generation**，永远不再被扫
  （`gc.unfreeze()` 可撤销；`gc.get_freeze_count()` 可观测）。
* **只装原子值的 dict / tuple 会被 CPython 主动 untrack**：
  `Objects/dictobject.c:1130 _PyDict_MaybeUntrack()`（3.12.13 里在同一目录还有
  `_PyTuple_MaybeUntrack`）。本机验证：`gc.is_tracked({"i": 1}) is False`、
  `gc.is_tracked([1]) is True`、`gc.is_tracked(C()) is True`。
  → **GC 成本 ∝ 存活的"可跟踪"对象数**，把 tensor/对象换成 int/数组能顺带减小 gen0 规模。

**公开实测收益 / 本机复现**

| 场景 | 结果 |
|---|---|
| 本机稳态（500 000 个静态对象 + 400 轮 × 1 000 短命对象，**不触发 gen2**） | freeze 前后 **无差别**：GC 4.4 ms vs 4.3 ms / 400 次 gen0+gen1（wall 32.4 vs 31.7 ms） |
| 本机"批量晋升触发 gen2"（400 000 静态对象 + 120 轮 × 2 000 个半数存活对象） | **gen2 收集 13.5 ms → 2.5 ms（-81%）**，GC 总时长 19.3 → 8.0 ms，wall 38.1 → 26.0 ms（-32%），GC 占比 50.6% → 31% |
| 真机 vLLM（engine core，20 s） | 含 GC 符号的周期权重 **0 / 55 442 828 817** |

**对本负载适用性**

* **已经吃满**：vLLM 在 `EngineCore.__init__` 里（`model_executor` 构造之后）
  调 `freeze_gc_heap()` = `gc.collect(0/1/2)` + `gc.freeze()`；
  真机 perf 里 GC 周期为 0 就是它的直接效果。
* `gc.disable()`：不建议。Instagram 的曲线说明它换来 10% 容量但内存线性增长；
  vLLM 自己在 `shutdown()` 里还要 `gc.unfreeze()`、在 CUDA graph 场景有
  `VLLM_ENABLE_CUDAGRAPH_GC` 开关，禁用 GC 会把这些路径暴露在"循环垃圾不回收"的风险下。
* 阈值调大（`700 → 50 000`）：**只有在能观测到 gen0 频繁收集时才有意义**。
  vLLM 自带观测手段：`VLLM_GC_DEBUG='{"top_objects":5}'` 会打印每次 GC 的
  generation / 耗时 / 被收集对象类型（`vllm/utils/gc_utils.py::GCDebugger`）。
  建议先量，再决定 —— 就当前证据（真机 GC = 0）而言，预期收益 ≈ 0，属于"可做但没必要"。
* 升级到 3.14 换增量 GC：**3.14.5 已经回退**，不要作为理由。

**证据强度：强**（官方文档 + vLLM PR + 真机 0 权重 + 本机可复现的两组对照）。

---

## 3. 给我们这个负载的排序建议

> 收益口径：`prepare_input` p50 = 2 757.6 µs（0.8B/TP1/B=1/ISL=128/decode），
> 单步 p50 = 4 915 µs（`docs/00` §0）；下文"≈x%"一律指 **`prepare_input` 的 x%**。
> 所有收益都是**上界估计**，且 P0 与 P1 之间有重叠（同一段代码，不要相加）。

### 3.1 排序总表

| 优先级 | 具体改法 | 落点 | 预期收益 | 风险 | 证据 |
|---|---|---|---|---|---|
| **P0** | 让 `CommonAttentionMetadata.replace()` 不再击穿 `_num_computed_tokens_cache`（3 个 GDN builder 各重算一次） | `vllm/v1/attention/backend.py`（`compute_num_computed_tokens`）+ `_treat_single_token_prefills_with_state_as_decodes` | **≈130 µs/步 ≈ 4%**（真机插桩 129.5 µs，子树 67.5% 在 `PyNumber_Subtract`） | 中（缓存语义/正确性） | 真机 perf + 插桩（`docs/07`、`agents/measurement_profiling`） |
| **P0** | 消除每步 3 次重复的 GDN metadata 构建（group 级共享，而不是每个 group 重建） | `vllm_ascend/ops/gdn_attn_builder.py::AscendGDNAttentionMetadataBuilder.build` | 这是最大单点：`build` self **20.10%**线程（≈36% of PI），`_build_attn_group_metadata` inclusive 23.26% | 中（group 语义/正确性，必须逐项对拍） | 真机 perf children 分解 |
| **P1** | 减少每步 **torch 算子条数**：`out=`/预分配缓冲、合并 slice、CPU 侧小算术改 Python int / numpy | 各 metadata builder 的 `torch.*` 调用点 | **1–3%**（直接打掉 `eventfd_write` 63% 的 `OpCommand::RunOpApiV2`、`NPUEvent::record` 13%、`MaybeSetDevice`/op-prep 锁 ≈30%、以及一部分 GIL drop/acquire 26.5%） | 低–中（数值一致性） | 真机符号 + 调用者归因（本文 §1.3） |
| **P1** | 热路径**属性查找搬到循环外**：`self.x`→局部变量、`module.f`→模块级名字、动态名字改 `sys.intern` 一次复用、别在循环里 `getattr/hasattr` 反射 | vLLM/vllm-ascend 的 `prepare_input` 热路径 | **1–2%**（属性/方法查找合计 ≈4.2% of PI，见 §1.2/§2.6） | 低（可读性/维护性） | 真机 `_PyType_Lookup`/`_PyObject_GenericGetAttrWithDict` + 3.12 源码 |
| **P2** | per-step metadata 类加 `slots=True`（**不要 `frozen=True`**） | `GDNAttentionMetadata` 等 | **<0.05%**（3 次/步 × ~0.2 µs ≈ 0.6 µs） | 极低（但 `frozen=True` 会直接报错：builder 构造后还要赋值） | 公开基准 + 本机 24 字段实测 |
| **P2** | 名字/类型稳定性：不要运行时改类、不要每步生成新属性名字符串 | 全局 | 小但免费（避免 type-cache/特化失效） | 极低 | 3.12 `_PyType_Lookup` / `specialize.c` 源码 |
| **P2** | 用"只装原子值的 dict/tuple"或直接复用实例，减少 **GC 可跟踪对象**数 | metadata 构建 | 当前 GC=0 → **≈0**；但在别的工况（更大 batch、更多 JSON/字符串）可能有意义 | 低 | `_PyDict_MaybeUntrack` + 本机 `gc.is_tracked` 验证 |
| **P3** | `gc.freeze()` / `gc.disable()` / 阈值调优 | — | **≈0**（已 freeze；真机 GC 周期 = 0） | 中偏高（disable 有内存风险） | 真机 0 权重 + vLLM PR #24008 + Instagram 曲线 |
| **不做** | immortal 化用户常量/单例 | — | 0（3.12 无 API；内建常量已 immortal） | — | PEP 683 + 源码 |
| **不做** | deferred/biased RC | — | 0（GIL 构建无实现） | — | 3.14 文档逐字："does nothing on builds with the GIL enabled" |
| **不做** | 升 free-threaded（PEP 703） | — | 负（单线程 5–40% 税 + 生态 ABI 阻塞），且不对症 | 高 | 3.13/3.14 官方文档 + PyPI 轮子现状 |
| **不做** | 换 mimalloc / 升 3.13 图性能 | — | 默认构建仍是 pymalloc；公开数字仅 1–2% 量级且有 RSS 放大 | 中 | 3.14 memory 文档 + gh-135153 |

### 3.2 逐条落地说明（含"值不值得"的直答）

**① `GDNAttentionMetadata` 改成 `slots=True` 值不值得？**
**作为性能手段不值得；作为顺手的一致性改动可以。**
本机 24 字段实测：创建 466 → 250 ns、属性访问 ×8 114.6 → 60.9 ns、内存 344 → 224 B。
每步 3 个实例 → 合计约 **0.6 µs/步 = 0.01% of `prepare_input`**。
而 `docs/00` 里"3 个 GDN builder 各重算一次"那一条是 **130 µs/步**，差 200 倍。
注意两点：(a) `slots=True` **不会**减少 refcount 操作条数（见 §2.5 的诚实说明）；
(b) **`frozen=True` 会直接报错** —— 现有 builder 在构造后还要
`attn_metadata.non_spec_prefill_metadata = ...`、`_attach_*_metadata()` 里就地修改。

**② 真正该花时间的是"每步重建几次 / 重建时做多少事"，不是"每个对象建得多快"**
把 `build()` 从每步 3 次降到 1 次（共享 group 级结果）、把 `.replace()` 的缓存击穿修掉，
量级是 **100 µs 级**；而任何对象层面的微优化（slots/元组/kwargs 改位置参数）都是 **µs 级**。

**③ 减少算子条数为什么会同时改善 `eventfd_write` 和 `pthread_mutex_lock`**
真机归因（§1.3）：`eventfd_write` 的 63% 来自 `at_npu::native::OpCommand::RunOpApiV2`、
13% 来自 `c10_npu::NPUEvent::record`；`pthread_mutex_lock` 有 26.5% 来自
`take_gil`/`drop_gil`（张量销毁/数组运算释放 GIL 的路径）。**这两条都是"算子/对象条数"的函数。**
最能直接受益的候选：`compute_num_computed_tokens` 里那 4 次张量相减（`docs/07` 已定位）、
`torch.empty_like/zeros`、`index_select`、`slice_Tensor`。
落地时要检查"用 numpy/int 替代是否改变 dtype/溢出/设备语义"，这是唯一的风险点。

**④ 属性查找怎么改最划算**
真机 `_PyType_Lookup` 的 64% 样本花在**缓存表命中判定**上（表 ~98 KB > L1），
所以"减少调用次数"比"让单次更快"更有效：
把 `self.attr` / `module.attr` / `obj.method` 在**每步**（而不是每次调用）解析成局部变量；
动态属性名一次性 `sys.intern` 并复用；
热路径别用 `getattr/hasattr/setattr` 反射，也别在循环里改类。

**⑤ 关于 `gc` 的最终建议**
保持现状。要动就先用 vLLM 自带开关量：`VLLM_GC_DEBUG='{"top_objects":5}'`
打印稳态下的 collection 次数与耗时；只有在"GC 明显出现"的工况
（例如换成大 batch / 大量 JSON 结构化输出）才考虑 `threshold0` 调大，
**不要 `gc.disable()`**。

### 3.3 建议的下一步测量（本文查不到、但很快能补上的）

1. **用 3.12 的 `--enable-pystats` 构建跑一次 harness**，导出 `LOAD_ATTR` 的
   hit/miss/deopt 与 failure_kinds（`Tools/scripts/summarize_stats.py`）。
   这能把"属性查找 4.2% of PI"进一步拆成"特化命中/未特化/miss"，
   直接量化 §3.1 的 P1 能拿多少。
2. **给 `_PyType_Lookup` 打点**（或在 3.13+ 用 `sys.monitoring`）统计
   调用次数 × 平均 MRO 深度，确认"热点是查表还是走 MRO"。
3. **`VLLM_GC_DEBUG` 三个工况**（B=1 / B=64 / chunked prefill）各跑一次，
   确认 GC 在各工况下都是 0；若某个工况出现 GC，再谈 §2.7 的阈值调优。
4. `perf record` 复采时**关掉 trampoline**（`PYTHONPERFSUPPORT` 只对需要 Python 帧的
   采集开启），可以顺便去掉 `PyUnstable_Code_GetExtra` 那 1% 的探针代价。

---

## 4. 明确列出"查不到的"

1. **没有任何公开数据能回答"aarch64 Linux（Kunpeng 920 / ARMv8）上 free-threaded CPython
   的单线程税是多少"**：官方只有 macOS aarch64 的 ~1% 和 x86-64 Linux 的 ~8%（3.14），
   以及 3.13 的整体 ~40%。
2. **`__slots__` / `dataclass(slots=True)` 没有"22–24 字段、每步新建、在 aarch64 服务器上"
   的公开 benchmark**；能找到的都是通用微基准（x86/笔记本、2–10 字段）。本文用本机
   24 字段实测补了机制量级，但**不是** 920B 数字。
3. **没有公开数据说明 vLLM（或任何 LLM 推理引擎）`prepare_input` 里
   type cache / LOAD_ATTR inline cache 的命中率与失效原因**；本文只能用符号 self%
   与源码行分布间接推断。
4. **没有公开的"mimalloc 对小型单线程服务负载"的净收益**（默认 GIL 构建）。
   仅有 2022 年原型 pyperformance 的 1–2% 几何平均与若干负向项；
   也没有人公布过在 Ascend/CANN 环境下 `PYTHONMALLOC=mimalloc` 的实测。
5. **没有公开的"3.12 上把用户对象 immortal 化"的收益数字**：3.12 没有公开 API，
   PEP/PR 里的数字都是**性能回归**（1.02–1.03x slower）与 pre-fork 内存收益，
   没有人测过"给 refcount 密集负载开一个 immortal 白名单"能省多少。
6. **没有 torch/torch_npu 的 free-threaded（`cp313t`/`cp314t`）aarch64 轮子**
   （2026-09-24 查 PyPI torch 2.14.0：cp313/cp314 有，带 `t` 的没有）；
   torch_npu/CANN 对 free-threading 的支持状况**查不到任何公开声明**。
7. **没有公开数据说明 `gc.freeze()` 在 LLM 推理服务里的量化收益**：
   vLLM 的 PR #24008 只有定性描述（"避免 static 对象被 full collection 遍历"），
   没有数字；本文给的本机对照（gen2 13.5 → 2.5 ms）是**合成负载**，不是 vLLM 实测。
8. **查不到"vLLM 每步属性查找次数"的官方/社区统计**；只能从 `PyUnstable_Code_GetExtra`
   这类边缘符号反推测量装置本身的影响。

---

## 附录 A：本机微基准原始输出（CPython 3.12.10 / x86_64）

```text
# 24 字段 dataclass 对照
sizeof(obj)   non-slots: 48  slots: 224
sizeof(__dict__) non-slots: 296  slots: n/a
__init__ co_code: NoSlots 340B / Slotted 340B（相同）
create    non-slots 466.2 ns/inst   slots 250.5 ns/inst
attr x8   non-slots 114.6 ns/loop   slots  60.9 ns/loop
replace   non-slots 2885.0 ns/call  slots 2558.5 ns/call

# GC：稳态（不触发 gen2）
freeze=False gen0_runs=400 gc_time= 4.4 ms wall=32.4 ms
freeze=True  gen0_runs=400 gc_time= 4.3 ms wall=31.7 ms

# GC：批量晋升触发 gen2（400k 静态对象 + 120 轮 × 2000，半数存活）
freeze=False runs=241 gc_total=19.3 ms wall=38.1 ms  gen0=3.3 gen1=2.5 gen2=13.5 (ms)
freeze=True  runs=241 gc_total= 8.0 ms wall=26.0 ms  gen0=3.1 gen1=2.5 gen2= 2.5 (ms)

# 小对象成本
object.__new__ 34.9 ns | {} 15.6 ns | (1,2,3) 7.6 ns | [] 17.3 ns | str(i) 34.2 ns

# immortal 观测（3.12.10）
None/True/3/Ellipsis/NotImplemented/'a'/intern('abc') refcnt = 4294967295
[] = 3 ; sys = 63
gc.is_tracked({"i":1}) = False ; gc.is_tracked([1]) = True ; gc.is_tracked(C()) = True
```

## 附录 B：真机调用者归因方法（可复现）

```bash
# 1) 符号调用者（周期加权，跳过未符号化的 [lib*.so] 与 kernel 帧）
zcat data/profiles/real-b1/fp-scope/folded.txt.gz \
| grep -F ";pthread_mutex_lock" \
| awk '{n=split($0,a,";"); w=a[n]; sub(/.* /,"",w);
        f=l=""; for(i=1;i<=n;i++){f=a[i]; sub(/ [0-9]+$/,"",f); frames[i]=f}
        for(i=2;i<=n;i++){ if(frames[i]=="pthread_mutex_lock"){ j=i-1;
          while(j>=1 && (substr(frames[j],1,1)=="[" || frames[j] ~ /^(write|vfs_write|ksys_write|el0)/ )) j--;
          if(j>=1){call[frames[j]]+=w; tot+=w} break } } }
     END {for (k in call) printf "%7.1f%%  %s\n", 100*call[k]/tot, k}' | sort -rn

# 2) 已内联符号的源码行份额（分母 = 该符号自身样本）
column -s, -t data/profiles/real-b1/fp-scope/hotspots_srcline.csv | head -20
# 3) 真机是否发生 GC（答案是 0）
zcat data/profiles/real-b1/fp-scope/folded.txt.gz \
| grep -E "gc_collect|visit_decref|visit_reachable|gcmodule" | awk '{n=split($0,a," "); s+=a[n]} END{print s+0}'
```

## 附录 C：本文引用的 CPython 源码行（v3.12.13，除注明外）

| 文件:行 | 内容 | 本文用途 |
|---|---|---|
| `Include/object.h:110/125` | `_Py_IMMORTAL_REFCNT`（`UINT_MAX`） | §2.1 |
| `Include/object.h:239–247` | `_Py_IsImmortal()`（`(int32)ob_refcnt < 0`） | §2.1 |
| `Include/object.h:624–659` | `Py_INCREF` 饱和加法；**L642 add / L646 store** | §1.3 / §2.1 |
| `Include/object.h:696–706` | `Py_DECREF` 先查 immortal 再递减 | §2.1 |
| `Include/internal/pycore_object.h:76–89` | `_Py_SetImmortal`（internal-only） | §2.1 |
| `Include/internal/pycore_gc.h:58–64` | **`_PyGCHead_NEXT`（L59 load / L60 return）** | §1.3 修正 1 |
| `Modules/gcmodule.c:47` | `#define GC_NEXT _PyGCHead_NEXT` | §1.3 修正 1 |
| `Modules/gcmodule.c:1450–1461` | full GC 只在 `long_lived_pending/long_lived_total > 25%` 时触发 | §2.7 |
| `Objects/dictobject.c:1130–1171` | `_PyDict_MaybeUntrack()` | §2.7 |
| `Objects/typeobject.c:43–49` | `MCACHE_HASH` / `MCACHE_HASH_METHOD`（用 **name 指针**）/ `MCACHE_CACHEABLE_NAME` | §2.6 |
| `Objects/typeobject.c:4725–4775` | `_PyType_Lookup`（**L4734 命中判定**、L4747 miss→`find_name_in_mro`、L4772 写缓存） | §2.6 |
| `Include/internal/pycore_typeobject.h:29–39` | `MCACHE_SIZE_EXP 12` → 4 096 项 | §2.6 |
| `Python/specialize.c:760–1130` | LOAD_ATTR 特化家族/失败原因（`__getattr__`、property、slot、managed dict…） | §2.6 |
| `Objects/obmalloc.c`（3.13.7） | `PYOBJ_ALLOC = PYMALLOC_ALLOC`（GIL）/ `MIMALLOC_OBJALLOC`（`Py_GIL_DISABLED`） | §2.4 |
