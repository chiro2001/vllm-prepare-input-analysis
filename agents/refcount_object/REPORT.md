# 对象模型 / 引用计数方向调研 —— 交接报告

**Agent**：`/root/refcount_object`
**状态**：完成（纯文献 + 源码 + 既有真机数据复核；未登录远端、未跑 NPU、未起容器）
**交付物**：

- `docs/10-cpython-directions/02-object-model.md`（正式文档，中文，含 7 个方向逐条 + 排序建议 + 查不到清单）
- 本文件（交接/过程记录）

**时间**：2026-09-24（Asia/Shanghai）

---

## 0. 三句话结论

1. 我们要优化的运行时是 **CPython 3.12.13（GIL 构建，非 free-threaded）**，
   这直接决定了本方向里 4 条"解释器级"技术（immortal objects / deferred RC /
   free-threading / mimalloc）**要么已生效、要么完全不可用**，只剩"少造对象、少查属性、
   少派发算子"三类代码级动作可以落地。
2. **GC 方向可以结案**：真机 20 s 采集里含 GC 符号的周期权重 = **0 / 55 442 828 817**，
   vLLM 的 `freeze_gc_heap()`（PR #24008）已经吃满收益；任务书里提到的
   "`gc_collect_main` 占 54.15%" 来自 **perf 工具自检负载**，不是真机 vLLM。
3. **属性查找是本方向里最值得做的一项**：真机 `_PyType_Lookup` ≈2.9% + 
   `_PyObject_GenericGetAttrWithDict` ≈2.3% + `PyObject_GetAttr`/`_PyObject_GetMethod`
   ≈1.9%（均为折算到 `prepare_input` 的估计），合计 ≈4–5%，且完全靠写代码改善。

---

## 1. 我做了什么

| 步骤 | 内容 |
|---|---|
| 读项目约定 | `plan/COORDINATION.md`、`docs/00-INDEX.md`、`docs/05-hotspots.md`、`agents/measurement_profiling/PERF_SYMBOLS.md` |
| 复核真机数据 | `data/profiles/real-b1/fp-scope/{hotspots.csv,hotspots_srcline.csv,folded.txt.gz,manifest.json}`；自写 awk 做"周期加权调用者归因"（方法见正式文档附录 B） |
| 复核既有结论的来源 | 发现 `pycore_gc.h:59/60` 的两条最热行来自 `data/profiles/perf-selftest-.../fp_notramp`，其负载是 `selftest_load.py` |
| 逐行读 CPython 源码 | v3.12.13 的 `Include/object.h`、`Include/internal/pycore_object.h`、`Include/internal/pycore_gc.h`、`Objects/dictobject.c`、`Objects/typeobject.c`、`Modules/gcmodule.c`、`Python/specialize.c`、`Include/internal/pycore_typeobject.h`；对照 3.13.7 的 `Objects/obmalloc.c` |
| 读 vLLM 源码 | `refs/vllm` 的 `vllm/utils/gc_utils.py`、`vllm/v1/engine/core.py`（freeze 调用顺序）、`vllm/v1/attention/backends/gdn_attn.py`；只读参考 `~/projects/vllm/HIST_PROJECT/vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py` |
| 本机微基准 | 内联 `python3 - <<PY`（不落盘）：24 字段 dataclass 对照、GC freeze 对照（稳态 / 触发 gen2 两种）、小对象分配成本、immortal 观测、`gc.is_tracked` 语义 |
| 联网调研 | PEP 683/703、CPython PR/issue（19474、110481、110764、117376、117783、120024、93988、109914、135153、92216）、3.13/3.14 官方文档、Meta/Instagram、close.com、Talk Python、Quansight、vLLM PR #24008/#27896、PyPI torch 轮子实况 |

---

## 2. 对既有交付的三处修正（建议 docs 负责人回填）

### 修正 1（重要）：`object.h:646` vs `pycore_gc.h:60` 的机理不同，且数据源不同

* `object.h:642 / :646`（`_PyEval_EvalFrameDefault` 符号内 10.24% / 11.27%）确实是
  **`Py_INCREF`**：L642 是 `new_refcnt = cur_refcnt + 1`，L646 是把新值 **store** 回
  `ob_refcnt_split`。这条与 `docs/05` 的描述一致。
* `pycore_gc.h:59 / :60`（`gc_collect_main` 符号内 32.60% / 53.60%）**不是**引用计数访问：
  这两行属于 `_PyGCHead_NEXT()`（L59 `uintptr_t next = gc->_gc_next;`、
  L60 `return _Py_CAST(...)`），在 `gcmodule.c` 里以 `#define GC_NEXT _PyGCHead_NEXT`
  被 `update_refs/move_unreachable/gc_list_merge` 等**链表遍历**使用。
  即真机/自检里 GC 的主要指令级成本是 **指针追逐（cache miss）**，不是 refcount 读写。
* 该 54.15% 数字的出处是 `data/profiles/perf-selftest-20260923T180711Z/fp_notramp/`，
  负载 = `selftest_load.py`（perf 工具自检），**无 `gc.freeze()`**。
  真机 `real-b1` 的 `hotspots_srcline.csv` 里没有 `gcmodule.c`/`pycore_gc.h` 任何一行。

→ 结论方向不变（"少造对象"仍然对），但机理描述与"这是真机 vLLM 的头号热点"这个隐含语气需要改。

### 修正 2：`eventfd_write` / `pthread_mutex_lock` 的来源不是 vLLM 的 IPC / GIL 争抢

按周期加权取"最近的已符号化祖先帧"后：

* `eventfd_write`：`at_npu::native::OpCommand::RunOpApiV2` **63.1%**、
  torch `copy_` 分发器 17.8%、`c10_npu::NPUEvent::record` 13.2% →
  **每个 NPU 算子提交都要 write 一次 eventfd**（CANN 的 op 命令/事件机制）。
* `pthread_mutex_lock`：`take_gil` 16.5% + `drop_gil` 10.0%（≈26.5% 是 GIL drop/acquire，
  栈为 `THPVariable_dealloc → PyEval_SaveThread → drop_gil` 一类），
  其余 ≈63% 是 `c10_npu::MaybeSetDevice`、CANN op profiling、op preparation、Adx dump 等内部锁。

→ 这两条是"**算子条数**"的副产品，不是"线程并行度"问题；
  `docs/05` §2.2 的注释（"vLLM v1 的 EngineCore↔Worker 事件通知"）建议改为 CANN op 提交。

### 修正 3：任务书里"3.13 起把 pymalloc 换成基于 mimalloc 的实现"不准确

* 3.13/3.14 **默认（GIL）构建的对象分配仍然是 pymalloc**：
  `Objects/obmalloc.c` 里 `#elif defined(WITH_PYMALLOC) → PYOBJ_ALLOC = PYMALLOC_ALLOC`；
  官方 3.14 文档 `c-api/memory.html` 表格同义：free-threaded 构建才是 mimalloc 默认，
  默认构建需要 `PYTHONMALLOC=mimalloc` 手动选择。
* mimalloc 在 3.13 被 **vendored 且默认编入**（不改变默认分配器），
  公开收益只有 2022 年原型数据（pyperformance 几何平均 1.01–1.02x faster），
  且有 RSS 放大记录（gh-135153：131 MB → 197 MB）。

---

## 3. 关键量化（详见正式文档 §1.2 / §2）

真机 `real-b1/fp-scope`（19 564 样本；55.00% 落在 `prepare_input` 窗口 → 
"PI 内占比 ≈ 线程 self% ÷ 0.55"）：

| 类别 | 代表符号（线程 self% → PI 内占比） |
|---|---|
| refcount 流量 | `_PyEval_EvalFrameDefault` 9.25% → 16.8%（其中 INCREF add+store ≈21.5% of 该符号 → ≈3.6% of PI） |
| 属性/方法查找 | `_PyType_Lookup` 1.61% → 2.9%；`_PyObject_GenericGetAttrWithDict` 1.28% → 2.3%；`PyObject_GetAttr` 0.9%；`_PyObject_GetMethod` 1.0% |
| 分配/释放 | `_PyObject_Malloc` 2.8% + `_PyObject_Free` 1.8% + `malloc` 1.8% + `cfree` 0.8% ≈ **7.2%** |
| 调用派发 | `initialize_locals` 1.5% + vectorcall 1.3%+1.1% |
| CANN op 开销 | `eventfd_write` 3.7% + `pthread_mutex_lock` 2.8% |
| **GC** | **0** |

本机（x86_64 / 3.12.10）：

* 24 字段 dataclass：创建 466 → 250 ns（slots）、属性×8 115 → 61 ns、内存 344 → 224 B、
  生成的 `__init__` 字节码**等长**（说明 refcount 操作条数不变）。
* GC：稳态 freeze 前后无差别（4.4 vs 4.3 ms / 400 次 gen0-1）；
  批量晋升触发 gen2 时 gen2 13.5 → 2.5 ms（-81%）、wall -32%。
* `gc.is_tracked({"i":1}) is False`（只装原子值的 dict 不被 GC 跟踪）。

---

## 4. 排序建议（正式文档 §3 的浓缩版）

| 优先级 | 改法 | 预期 |
|---|---|---|
| P0 | `CommonAttentionMetadata.replace()` 保住 `_num_computed_tokens_cache`（3 个 GDN builder 各重算一次） | ≈130 µs/步 ≈4% of PI |
| P0 | 消除每步 3 次重复的 GDN metadata 构建（最大单点：`build` self 20.10% 线程） | 100 µs 级 |
| P1 | 减少 torch 算子条数（`out=`、预分配、CPU 侧小算术换 int/numpy） | 1–3%（同时打掉 eventfd/mutex 的一部分） |
| P1 | 热路径属性/方法提升为局部变量、动态名字 `sys.intern` 复用、少用 `getattr/hasattr` | 1–2% |
| P2 | metadata 类加 `slots=True`（**不要 `frozen=True`**，会破坏"构造后赋值"） | <0.05% |
| P3 | GC 调优（`VLLM_GC_DEBUG` 先观测；不要 `gc.disable()`） | ≈0 |
| 不做 | immortal 用户常量 / deferred RC / free-threading / 换 mimalloc | 0 或负 |

**直接回答"`GDNAttentionMetadata` 改 `slots=True` 值不值得"**：作为性能手段不值得
（3 实例/步 × ~0.2 µs ≈ 0.6 µs，是 P0 那 130 µs 的 1/200），
作为一致性/内存改动的风险极低，可以顺手做；`frozen=True` 则**会直接报错**。

---

## 5. 查不到的（正式文档 §4 全文，这里留摘要）

1. aarch64 **Linux**（Kunpeng）上 free-threaded CPython 的单线程税（官方只有 macOS aarch64 ~1%、x86-64 Linux ~8%）。
2. 22–24 字段 dataclass 在 aarch64 服务器、每步新建场景的公开 benchmark。
3. vLLM `prepare_input` 的 type cache / LOAD_ATTR hit-miss-deopt 统计。
4. 默认 GIL 构建启用 mimalloc 的净收益（只有 2022 原型 1–2% 与 RSS 放大记录）。
5. "给 refcount 密集负载开 immortal 白名单"的收益（3.12 无公开 API，PEP 只有回归数字）。
6. torch/torch_npu 的 `cp313t`/`cp314t` aarch64 轮子（PyPI 上 torch 2.14.0 **没有** `t` 轮子）。
7. `gc.freeze()` 在 LLM 推理服务里的量化收益（vLLM PR #24008 只有定性描述）。

---

## 6. 我没有做的事（边界）

* 没有做任何新的真机采集 / NPU 操作 / 容器操作（按任务约束）。
* 没有修改 `docs/` 下的既有文档（修正 1–3 只写在本报告与正式文档里，等 docs 负责人回填）。
* 没有运行 harness（`docs/06` 的合成负载）——本文的"我们负载"数字全部复用既有真机数据。
* 本机微基准只在开发机 x86_64/3.12.10 上做，**不能**外推成 920B 的绝对数字
  （正式文档中已逐处标注）。

## 7. 建议的后续动作（若还要继续投入）

1. 用 `--enable-pystats` 的 3.12 构建跑一次 harness，导出 LOAD_ATTR 的
   hit/miss/deopt 与 failure_kinds —— 把"属性查找 4–5%"拆成可优化的具体占比。
2. `VLLM_GC_DEBUG='{"top_objects":5}'` 在 B=1 / B=64 / chunked prefill 三工况各跑一次，
   把"GC = 0"从 B=1 单工况推广到全工况。
3. 复采 perf 时关掉 `PYTHONPERFSUPPORT`（可去掉 trampoline 的
   `PyUnstable_Code_GetExtra` ≈1% 探针代价），并把 `eventfd_write`/`pthread_mutex_lock`
   的调用者归因纳入常规报告模板（脚本片段见正式文档附录 B）。
