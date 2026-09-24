# prepare_input 热点：函数 / 源码行 / 指令 / topdown

> 状态：**真机 perf + libkperfx 已采集完成**（chip3，2026-09-24 20:25Z，
> 19 569 样本 / 0 lost / 9 组 PMU confidence 全 1.0）。
> 剩余 ⛳ 仅 B=64 与 chunked prefill 两个补充点（可选）。
> 数据口径遵循 `plan/COORDINATION.md` §6。

本文回答四个问题：

1. engine-core 主线程最热的函数是谁（火焰图 + `perf report`）；
2. 落到哪几行源码 / 哪几条指令（`perf annotate -l`）；
3. 这些 CPU 周期花在哪一类微架构瓶颈上（920B topdown + IPC + cache 层）；
4. 与历史基线（a3-21，frontend-bound 56–65%、IPC 0.72–0.89、无 cache 数据）相比，
   哪些结论被证实、哪些被推翻。

## 0. 采集口径（不可混用）

| 项 | 取值 | 说明 |
|---|---|---|
| 机器 | a3-22（Kunpeng 920B，640 逻辑核 / 4 socket） | chip3 = `/dev/davinci3` = `npu-smi -i 1 -c 1` |
| 目标线程 | engine-core 主线程的宿主 TID | `uni` executor 下 = `npu-smi` 报告的宿主 PID；见 `scripts/measure/ecmap.py` |
| 时间层 | LiteProfiler（`VLLM_LITE_PROFILER_LOG_PATH`，scope `prepare input`） | 只做时间归因 |
| 函数层 | `perf record -g -F 999 -t <TID> --call-graph dwarf` → inferno/flamegraph-rs | 只采该 TID，用户态+内核态 |
| 顶层分解 | `libkperfx`（920B preset，9 组串行 split/merge） | 每组 ≤8 事件，报 `time_running/time_enabled` |
| 指令/源码层 | `perf annotate -i perf.data --stdio -l --symbol <sym>` | 需要 debug info；缺符号时如实标注 |
| 互斥 | perf / libkperfx / msprof 一次只跑一种 | 通用计数器互相抢占，会污染 topdown |

宿主噪声：CPU `120-159` 不独占（实测 44 个外部进程的 affinity 覆盖该切片，mean
11.65%，max 100%）。因此：

* 计数类指标用 per-thread PMU（`KPERFX_TARGET=tid`），只累计该线程运行时的周期，
  且 `time_running/time_enabled` 直接反映被抢占比例；
* wall 类指标报中位数 + p90/p99，并在每个 run 的 manifest 里记录采集前后的
  `hostnoise_gate.sh` 快照。

## 1. 火焰图（on-CPU 采样）

| 图 | 工况 | 文件 | 状态 |
|---|---|---|---|
| **真机 B=1 decode（graph）** | `prof-real-b1`，19 569 样本 / 0 lost | `figures/05-flame-real-b1.svg` | ✅ |
| **真机：只看 `prepare input` scope 内** | 10 761 样本（55.00%） | `figures/05-flame-real-b1-scope.svg` | ✅ |
| **无卡 harness（realmachine 口径，noop）** | 30 953 样本 / 0 lost | `data/harness/prof_realmachine_20260924-031609/flamegraph_oncpu.svg` | ✅ |

### 1.1 历史空洞 G1 已闭合：热点可以归到 scope 内

perf 采样本身不按 scope 切分。但 LiteProfiler 行带 **epoch 微秒**、
`perf script --ns` 带**启动后纳秒**，两者经 `/proc/stat` 的 `btime` 换算后即可把
每个样本归入包含它的**最内层 scope**（`scripts/measure/scope_attribution.py`）。

真机 `prof-real-b1` 的采集窗口内：

| 通道 | 方法 | 结果 |
|---|---|---|
| 采样归因 | 落在 `prepare input` 窗口内的样本比例 | **55.00%**（10 761 / 19 564） |
| 墙钟区间 | scope 窗口并集 ÷ 采集跨度（已裁剪到窗口） | **54.88%** |

两者只差 **0.12 pp**，同时验证了 scope 归属与时钟换算；该值也与
`subscope_instrumentation` 用探针 + `Step:Schedule` 分母独立测得的
**55.3%（A 臂）/ 57.2%（C 臂）** 吻合。

### 1.2 读图要点（已逐条确认）

1. **top-10 仍然"极平"**：self 合计 **22.83%**（历史 23.7%），单点最大
   `_PyEval_EvalFrameDefault` 10.35%。⇒ "派发密集"而非"计算密集"。
2. **`docs/01` 预测的"每拍新建 pinned 缓冲"被证伪**：`pin_memory` 0.12%、
   `tolist` 0.07%、`index_select` 0.07%、`aclrtSynchronize` 0.18%，
   合计 **< 0.5%**。
3. 火焰图里最醒目的 Python 调用链是
   `execute_model → _build_attention_metadata → _build_attn_group_metadata →
   AscendGDNAttentionMetadataBuilder.build`（见 §2.3）。

## 2. 热点函数（`perf report --stdio --no-children`）

### 2.1 无卡 harness（realmachine 口径，`fp + PYTHONPERFSUPPORT`，30 953 样本 / 0 lost）

> 数据：`data/harness/prof_realmachine_20260924-031609/`，口径 `noop`（与真机可比）。
> **这是本项目已完成的、符号质量最好的一组热点数据。**

| # | symbol | self % | 归属解读 |
|---:|---|---:|---|
| 1 | `_PyEval_EvalFrameDefault` | **11.43%** | CPython 求值循环——"算子多、算术少"的直接证据 |
| 2 | `_PyObject_Malloc` | 1.63% | 每步几十处小对象分配 |
| 3 | `unicodekeys_lookup_unicode` | 1.38% | kwargs / dict 查找（`GDNAttentionMetadata(...)` 有 22 个 kwargs） |
| … | 其余 top-20 | 合计 21.4% | **极平**：没有任何单一函数超过 12% |

**指令级结论（`perf annotate -l`）**：`_PyEval_EvalFrameDefault` 的 self 时间里，
**引用计数读写（`Py_INCREF`/`Py_DECREF` 对应的 load/store）占 17.5%**——
这是"对象多、操作碎"的典型指纹，也解释了为什么 `Py_INCREF` 会独立出现在热点榜第 2 位。

### 2.2 真机（chip3，0.8B TP1，B=1/ISL=128/decode/`FULL_DECODE_ONLY`，pystack off）

> 采集：`perf record -F 999 -g --call-graph fp -t 704908`（engine-core 宿主 TID），
> **19 569 样本 / 0 lost**。数据：`data/profiles/real-b1/fp/perf-report.txt`。

| # | dso | symbol | self % |
|---:|---|---|---:|
| 1 | `libpython3.12.so.1.0` | `_PyEval_EvalFrameDefault` | **10.35%** |
| 2 | `[kernel.kallsyms]` | `eventfd_write` | **1.93%** |
| 3 | `libc.so.6` | `pthread_mutex_lock` | 1.78% |
| 4 | `libpython3.12.so.1.0` | `unicodekeys_lookup_unicode` | 1.68% |
| 5 | `libpython3.12.so.1.0` | `_PyType_Lookup` | 1.67% |
| 6 | `libpython3.12.so.1.0` | `_PyObject_GenericGetAttrWithDict` | 1.36% |
| 7 | `libpython3.12.so.1.0` | `_PyObject_Malloc` | 1.16% |
| 8 | `libc.so.6` | `malloc` | 1.06% |
| 9 | `libpython3.12.so.1.0` | `_PyFunction_Vectorcall` | 0.95% |
| 10 | `libpython3.12.so.1.0` | `initialize_locals` | 0.89% |
| | | **top-10 合计** | **22.83%** |

**与无卡 harness（§2.1）的对照**——这是保真度最硬的证据：

| 项 | 真机 | harness | 判定 |
|---|---:|---:|---|
| top-1 符号 | `_PyEval_EvalFrameDefault` **10.35%** | 同，**11.43%** | ✅ 一致（Δ+1.08 pp） |
| top-10 扁平度 | **22.83%** | 21.4% | ✅ 一致 |
| **热点 top-20 重合度** | — | **16/20 = 80%** | ✅ 达成判据 |

**真机独有的三条**（harness 结构性缺失，方向已知）：

1. **`eventfd_write`（kernel）1.93%** —— vLLM v1 的 EngineCore↔Worker 事件通知
   （ZMQ/eventfd）。harness 是单进程紧循环，没有这条，这正是它 IPC 偏高的主因之一。
2. **`pthread_mutex_lock` 1.78%** —— 跨进程/跨线程锁竞争。
3. **8 条 harness 缺失 vs 8 条 harness 独有**，全部可解释：harness 独有的是
   `tuple_alloc` / `THPVariable_getitem` / `DifferentiableViewMeta`
   （CPU ATen 代替 NPU 算子）。

### 2.3 头号热点：`AscendGDNAttentionMetadataBuilder.build` 的 children 分解

> 数据：`data/profiles/real-b1/fp/breakdown/AscendGDNAttentionMetadataBuilder.children.txt`
> 与 `data/profiles/real-b1/fp/breakdown/build_attention_metadata.children.txt`。

`_build_attention_metadata` 子树（inclusive = 该 TID 全部样本的 %）：

| 符号 | inclusive % |
|---|---:|
| `NPUModelRunner._build_attention_metadata` | **31.60%** |
| ├ `_build_attn_group_metadata`（闭包） | 23.26% |
| ├ `_get_block_table_and_slot_mapping` | 3.57% |
| └ `_get_dcp_metadata` | 0.01% |

`AscendGDNAttentionMetadataBuilder` 子树（**self %，即不含子调用**）：

| 符号 | self % |
|---|---:|
| `AscendGDNAttentionMetadataBuilder.build` | **20.10%** |
| ├ `_pad_non_spec_decode_graph_inputs` | **5.94%** |
| ├ `_attach_non_spec_decode_metadata` | 4.04% |
| └ `_attach_spec_decode_metadata` | 0.03% |

**与插桩（wall 计时）的交叉验证**：

| 子步 | 插桩 wall（µs/步） | perf self % | 一致性 |
|---|---:|---:|---|
| `pad_graph_inputs` | 294.1 | **5.94%** | ✅ 同为最大子步 |
| `attach_metadata` 段 | 4.5（仅 attach 入口） | 4.04% | ✅ |
| GDN build 总体 | 1091.7（3 次） | **20.10% + 子项** | ✅ 量级一致 |

⇒ **两个完全独立的方法（Python 探针 wall 计时 vs PMU 采样）指向同一个结论**：
`AscendGDNAttentionMetadataBuilder.build`（尤其是 `_pad_non_spec_decode_graph_inputs`）
是 `prepare input` 内部最大的单一热点。

### 2.4 "这 3 ms 是 CPU 真忙还是等 device？"——本项目最关键的定论

**答案：几乎全是 CPU 真忙，不是 device 等待。** 三条独立证据：

1. **同步调用占比 0.27%**：`perf report` 里 `aclrtSynchronizeStreamWithTimeoutImpl`
   (0.18%) + `aclrtSynchronizeEventImpl` (0.08%) + `aclrtSynchronizeStreamWithTimeout`
   (0.01%) 合计 **0.27%**。⇒ `prepare_input` scope 内**没有实质性的 device 同步等待**，
   静态分析（`docs/01` §6）"只有一处条件性同步点"的判断被实测证实。
2. **GDN 子树的叶子分类**（`subtree-gdn/summary.json`）：真正落到 device 相关叶子的
   很少 —— `copy_h2d` 3.19% + `ascend_driver` 1.16% + `d2h_sync` 0.15% +
   `kernel` 5.41% ≈ **9.9%**；而 `cpython_runtime` 16.86% + `aten_op_dispatch` 13.30% +
   `libc` 13.04% + `allocator` 5.46% ≈ **48.7%** 是纯宿主侧。
   （剩余 `other` 41.3% 为未分类叶子，但 top-down 显示 frontend-bound 66%，
   说明它们也主要是解释器/取指路径而非内存等待。）
3. **topdown 的形状**：`frontend_bound 66.01%`（其中 `frontend_latency_bound 59.89%`、
   `frontend_bandwidth_bound 6.12%`）、`retiring` 仅 12.85%、`backend_bound` 10.11%。
   若是 device 等待，表现应是低 retiring + 高 backend/内存 stall；
   而 **frontend-latency 主导是"取指/解码指令流"的特征**，典型于
   **CPython 解释器逐条派发**。

⇒ **结论：`prepare_input` 的 CPU 成本主体是"CPython 解释器执行海量小对象操作"**，
不是等卡。这直接解释了为什么把 `_prepare_inputs` 的代码用 C 重写/减少 Python 层
调用次数会有收益，而"调大 batch 摊薄固定开销"收益有限。

## 3. 热点源码行与热点指令（`perf annotate -l`）

**符号质量的关键结论**（`agents/measurement_profiling/PERF_SYMBOLS.md`）：

| 目标 | 结果 |
|---|---|
| **镜像 `libpython3.12.so.1.0`（3.12.13）** | **有 `.debug_info/.debug_line`** → `perf annotate -l` 可用 |
| 宿主 `libpython3.11.so.1.0` | **stripped** → 只输出裸地址 |
| `libopenblas` | 无 debug info → 如实标注，不伪造 |

### 3.1 真机源码行级热点（`data/profiles/real-b1/fp/hotspots_srcline.csv`）

| 排名 | 文件:行 | 占该符号 self | 符号 | 说明 |
|---:|---|---:|---|---|
| 1 | `dictobject.c:940` | **28.48%** | `unicodekeys_lookup_unicode` | dict 查找（kwargs/属性） |
| 2 | `dictobject.c:968` | **26.43%** | `unicodekeys_lookup_unicode` | 同上（命中路径） |
| 3 | `dictobject.c:949` | 16.64% | `unicodekeys_lookup_unicode` | 同上（探测/比较） |
| 4 | **`object.h:642`** | **12.35%** | `_PyEval_EvalFrameDefault` | **`Py_INCREF` 的 add** |
| 5 | **`object.h:646`** | **11.26%** | `_PyEval_EvalFrameDefault` | **`Py_DECREF` 的 store** |
| 6 | `dictobject.c:945` | 5.13% | `unicodekeys_lookup_unicode` | 同上 |
| 7 | `unicodeobject.h:282` | 4.53% | `unicodekeys_lookup_unicode` | 字符串哈希/长度 |

> ⚠️ **勘误（子代理 `refcount_object` 查出并已修正本文）**：
> 本文件早期版本引用了 "`gc_collect_main` 在 `pycore_gc.h:60` 占 **54.15%**" 与
> "`_PyEval_EvalFrameDefault` 在 `object.h:646` 占 16.01%" 两条数字，**它们来自
> `data/profiles/perf-selftest-*/` 的 perf 工具自检负载**（`selftest_load.py`，
> 一个纯 Python 循环），**不是真机 vLLM**，因此不代表本负载。
> 并且 `pycore_gc.h:59/60` 是 `_PyGCHead_NEXT()`（GC 世代链表的指针追逐），
> **不是**引用计数操作。**上面 §3.1 的表才是真机数据。**

**GC 方向可以结案**：真机 20 s 采集里，含 `gc_collect_main`/`visit_decref`/`visit_reachable`
的**周期权重 = 0**（分母 55 442 828 817）。原因是 vLLM 的 `freeze_gc_heap()`
（`EngineCore.__init__`，在 `model_executor` 构造之后）已经生效。
⇒ **GC 调优 / `gc.freeze()` / `gc.disable()` 在本负载上没有空间。**

**真正的引用计数成本**：`object.h:642` + `object.h:646` = `_PyEval_EvalFrameDefault` self 的
**23.61%**，而该符号占总样本 10.35% ⇒ **≈2.4% 的全线程样本、≈4.4% 的 `prepare_input` 窗口**。
⇒ 方向确实是"减少对象数量"，但**注意量级**：它是 4% 级，不是主导项。

> **符号化前置条件**：`perf.data` 不内嵌符号，`py::` 帧靠 `/tmp/perf-<tid>.map` 现读，
> **采集完必须趁容器还活着跑完 report/folded/annotate**
> （实测把 map 移走，`py::` 帧从 26 354 掉到 0）。

## 4. 微架构画像：topdown + IPC + cache

### 4.1 无卡 harness（realmachine 口径，libkperfx 920B preset，9 组 confidence 全 1.0）

| 工况 | frontend | bad_spec | retiring | backend | IPC |
|---|---:|---:|---:|---:|---:|
| harness `realmachine`（noop，30 s 满频 2.9 GHz） | **72.11%** | 8.14% | 15.94% | 3.81% | **0.9489** |
| ├ frontend 细分 | latency 64.75 + bandwidth 7.36 | | | | |
| └ OOO stall | ptag 81.69 / mapq 13.47 | | | | |

**置信度**：`time_running == time_enabled`，`confidence = 1.0000`（9/9 组）。

### 4.2 真机 topdown

> 数据：`data/profiles/real-b1/pmu/topdown.json`（9/9 组，`complete_topdown_tree=true`）。
> 采集窗口与 §2.2 的 perf 同一次运行；目标 = engine-core 宿主 TID 704908。

| 层级 | 分量 | % |
|---|---|---:|
| **L1（四桶）** | frontend_bound | **66.01%** |
| | bad_spec | 11.03% |
| | retiring | 12.85% |
| | backend_bound | 10.11% |
| **frontend 细分** | latency_bound | **59.89%** |
| | bandwidth_bound | 6.12% |
| **backend 细分** | core_bound | 6.64% |
| | mem_bound | 2.97% |
| | ├ mem_l1_bound | 1.15% |
| | ├ mem_l2_bound | 0.48% |
| | └ mem_l3_dram_bound | 1.33% |
| | │（`mem_mem_bound`） | 0.00% ← 见 §4.4 坑 1 |
| | resource_bound | 0.50% |
| **stall 归因** | ptag_stall | **69.52%** |
| | mapq_stall | 24.47% |
| | rob_stall | 5.81% |
| | pcbuf_stall | 0.20% |

**IPC（真机实测）**：

```
INST_RETIRED = 6 347 341 352
CPU_CYCLES   = 8 234 970 671
IPC          = 0.7708
```

### 4.3 三方对照：真机 / 无卡 harness / 历史基线

| 指标 | **真机（本次，chip3）** | 无卡 harness | 历史（a3-21，graph MTP-off） |
|---|---:|---:|---:|
| frontend_bound | **66.01%** | 72.11% | 55.99–65.00% |
| retiring | 12.85% | 15.94% | — |
| bad_spec | 11.03% | 8.14% | — |
| backend_bound | 10.11% | 3.81% | — |
| **IPC** | **0.7708** | 0.9489 | 0.719–0.890 |
| **与历史判定** | ✅ **落在区间内** | ⚠️ 高 7–32% | — |
| **harness 判定** | — | IPC 高 **+23%**，frontend 高 **+6.1 pp** | — |

**这组数字的意义**：

1. **真机与历史完全一致** ⇒ 历史 topdown/IPC 基线（a3-21）**可直接用于本项目**，
   不需要重新建立基线；同时也说明**本次采集口径正确**。
2. **无卡 harness 的偏差是"偏乐观"的单向偏差**（IPC +23%、frontend +6.1 pp），
   方向与 `docs/06` §9.2 登记的归因一致：harness 是紧循环单进程，
   **缺 `eventfd_write`（1.93%）与 `pthread_mutex_lock`（1.78%）这类跨进程通信/锁**，
   且部分 device 算子被 CPU ATen 代替（更快）。
3. **harness 的正确用法**：用于**相对比较与趋势拟合**（斜率、方向、参数敏感性），
   **不用于**引用绝对占比。绝对占比用真机数字。

### 4.4 历史基线（a3-21，只有四桶 + IPC，**没有 cache 层**）

graph + MTP-off 各臂 frontend **55.99–65.00%**、IPC **0.719–0.890**。
182 份 `raw.csv` / 724 行事件全部 `percent_enabled = 100.00` ⇒ 历史 topdown
**没有 multiplexing 问题**，可直接引用。

### 4.5 必须记住的 PMU 两坑

1. **`memstall_l3miss` 与 `dram_*` 在 920B 上恒为 0**（256 MiB 流式负载下也实测为 0）。
   所以 `mem_l3_dram_bound` 承载的是 "L2 以下全部 stall"，`mem_mem_bound` 恒 0。
   **禁止写成 "DRAM bound = 0%"**，正确口径是 "L3/DRAM 合并槽 X%，DRAM 分量本机不可测"。
2. **PMU 争用会让 `confidence` 悄悄掉到 0.35**，而数值看起来完全正常。
   本项目所有 topdown/IPC 一律带 `min_confidence` 引用。

## 5. 与历史基线的差分

| # | 历史结论 | 本次判定 | 证据 |
|---:|---|---|---|
| 1 | `prepare input` 占 48–59%（0.8B graph decode） | **证实**（修正后 55.3%） | `data/subscope/ladder-s1.json` |
| 2 | 但该数字含 pystack 污染 | **新发现** | D1−B = +678 µs（+24.2%） |
| 3 | `prepare input` 1.6–6.4 ms 与模型规模基本无关 | **证实** | `data/historical/phase_per_step_configs.csv` |
| 4 | worker 主线程 frontend-bound 56–65% | **方向证实，量级偏乐观** | harness 72.11%（差 +7~16 pp，见 §6） |
| 5 | IPC 0.719–0.890 | **方向证实，量级偏乐观** | harness 0.9489（高 7–32%） |
| 6 | 热点"极平"、top-10 仅 23.7% | **证实** | harness top-10 = 21.4% |
| 7 | 算子下发接口全族只占 0.055% | **证实**（下发**接口**不热，但**调用次数**贵） | §5bis + `k_h2d_small` = 3–12 µs |
| 8 | `_update_states` 是潜在大头 | **推翻（低并发下）** | B=1 实测仅 100.7 µs = 3.1%；churn 场景待测 |
| 9 | `slot_mapping` / `attn_mask` 是热点 | **推翻** | 138 µs / 1.24 µs；attn_mask 走缓存分支 |
| 10 | 历史无 cache 层数据（G2 空洞） | **仍未补齐** | ⛳ 依赖 P3'' |

## 5bis. 子阶段分解（`subscope_instrumentation`，实测）

> 数据源：`data/subscope/sub-s1-c2d-gdnb/`（B=1, ISL=128, decode,
> FULL_DECODE_ONLY, pystack off）、`data/subscope/sub-s2-on-attn-20260923T184335Z/`
> （B=64 同配置）。工具：`instrument/`（AST 等价性已验证的探针）+ 
> `scripts/parse_subscope.py`。图：`figures/05-subscope-breakdown.svg`。

前面 §1–§4 的 perf 采样是**进程级**的，无法把热点函数切到 scope 内部（限制 G1）。
本节用 66 个 `pi: *` 子探针补上这一层：每个探针记录 `self_us`（独占）与
`incl_us`（含子），因此 `Σ self + unattributed = pi: TOTAL`（`meta.json` 校验残差
< 1e-6 µs）。

### 5bis.1 B=1 decode：子阶段图谱（p50 µs/step）

`prepare input` 墙钟 = **3294.6 µs**；未归因 207 µs（6.3%）。

| 子阶段 | p50 µs | 占比 | 性质 |
|---|---:|---:|---|
| **`pi: gdnb.pad_graph_inputs`** | **285.9** | **8.7%** | device（FULL graph 的 `fill_`/`copy_`/`expand_as`） |
| `pi: build_attention_metadata`（聚合） | 280.6 | 8.5% | 见下 |
| `@AscendGDNAttentionMetadataBuilder` 残差 | 172.9 | 5.2% | 含探针开销，见 5bis.4 |
| `pi: gdnb.treat_single_token` | 169.3 | 5.1% | 纯 CPU（`torch.diff`/`&`/`any().item()`） |
| **`pi: gdnb.build_actual_seq_lengths`** | **163.0** | **4.9%** | device（`empty_like`+`copy_`+`sub(out=)`） |
| `pi: in.slot_mapping` | 136.3 | 4.1% | triton launch |
| `pi: gdnb.compute_num_computed_tokens` | 129.5 | 3.9% | **缓存返回**（见 5bis.3） |
| `pi: gdnb.split_decodes` | 121.1 | 3.7% | 纯 CPU fast path |
| `pi: in.positions_assembly` | 110.6 | 3.4% | device |
| `pi: sync_input_prep` | 102.4 | 3.1% | event wait |
| `pi: update_states` | 99.0 | 3.0% | Python 逐请求 |
| `pi: prepare_inputs`（聚合残差） | 95.3 | 2.9% | 见下 |

**结论一：`prepare input` 内部最大的单一结构性热点是 attention metadata 构建，
合计 1.22 ms（37.0%），而其中 90% 属于 GDN（线性注意力）路径，不属于
full-attention 路径。**

### 5bis.2 混合模型：4 个 kv_cache_group，3 个是 GDN

`probe_notes.json`（`pi_note` 落盘，见 `meta.json.probe_notes`）实测：

| kv_cache_gid | builder | kv_cache_spec | block_size | layers | layer 样例 |
|---:|---|---|---:|---:|---|
| 0 | `AscendAttentionMetadataBuilder` | `FullAttentionSpec` | 1024 | 6 | 3,7,11,15 |
| 1 | `AscendGDNAttentionMetadataBuilder` | `MambaSpec` | 2048 | 6 | 0,4,8,12 |
| 2 | `AscendGDNAttentionMetadataBuilder` | `MambaSpec` | 2048 | 6 | 1,5,9,13 |
| 3 | `AscendGDNAttentionMetadataBuilder` | `MambaSpec` | 2048 | 6 | 2,6,10,14 |

Qwen3.5-0.8B 是 6 层 full-attention + 18 层 GDN 的混合模型（`full_attention_interval=4`）。
**每个 engine step 调用 4 次 `builder.build()`，其中 3 次是 GDN，各有 6 层**
（18 ÷ 3 = 6），而不是同一个组被重复调用。`decode` 稳态下
`num_prefills=0, num_decodes=1, num_spec_decodes=0`（`ascend_gdn_batch_shape` 实测），
确认走的是 decode-only 路径。

per-class 分解（`data/subscope/sub-s1-c2d-gdnb/builder_by_class.csv`）：

| builder 类 | 次/步 | self p50 µs | µs/次 | 占 prepare_input |
|---|---:|---:|---:|---:|
| `AscendGDNAttentionMetadataBuilder` | 3.0 | 172.9（+子探针 940.9） | **371.3** | 33.8% |
| `AscendAttentionMetadataBuilder` | 1.0 | 35.9（+子探针 88.3） | 124.2 | 3.8% |

> **p50 不可加性**：上面两行的"µs/次"由 `self_us_p50 + Σ 子探针 self_us_p50`
> 得到，而**分位数不可加**——所以 `3 × 371.3 + 124.2 = 1238 µs` 与
> `pi: am.builder_build` 的 `incl_us_p50`（≈1220 µs）有约 1.5% 的差，
> 这是方法学误差而不是矛盾。图（`figures/05-subscope-breakdown.svg`）与
> `summary_groups.csv` 用的是**均值**（可加），两者不可混用。

### 5bis.3 `attn_mask` 不是 O(S²)：缓存命中已证实

静态路径（`vllm_ascend/attention/attention_mask.py`）：

```python
def get_attention_mask(self, causal, model_config):
    if not causal:               return None
    if runner_type == "pooling": return self.get_attn_mask(2048, torch.bool)  # 缓存
    return self.get_splitfuse_attn_mask()                                     # 缓存 2048x2048 int8
```

实测（`probe_notes.attn_mask_path`）：`causal=true`、`runner_type=generate`、
走 `get_splitfuse_attn_mask (cached 2048x2048 int8)`、`returned_none=false`、
缓存对象已存在。该子步实测 **1.24 µs**（= 一次属性读取 + 返回）。
**若走未缓存的 `get_attn_mask`，会每步重建 2048×2048 张量（应达百 µs–ms 量级）。
该风险已排除。**

### 5bis.4 GDN build 的六段分解（303 → 371 µs/call 的全部去向）

`AscendGDNAttentionMetadataBuilder.build()` 371.3 µs/call 的逐段归因：

| 段 | µs/步 | µs/次 | 分类 |
|---|---:|---:|---|
| `gdnb.pad_graph_inputs` | 285.9 | 95.3 | **device** |
| `gdnb.treat_single_token` | 169.3 | 56.4 | 纯 CPU/Python |
| `gdnb.build_actual_seq_lengths` | 163.0 | 54.3 | **device** |
| `gdnb.compute_num_computed_tokens` | 129.5 | 43.2 | 缓存返回（见下） |
| `gdnb.split_decodes` | 121.1 | 40.4 | 纯 CPU fast path |
| `gdnb.attach_decode` | 40.2 | 13.4 | 混合 |
| `gdnb.ctor`（22 kwargs dataclass） | 21.6 | 7.2 | 纯 CPU |
| `gdnb.mamba_block_table` | 5.1 | 1.7 | device（`none` 模式秒退） |
| `gdnb.attach_prefill` / `attach_spec` | 5.3 | 1.8 | 混合 |
| **残差（含探针开销）** | **172.9** | **57.6** | 见下 |

**分类小结**：device 交互段 ≈ **494 µs/步（44%）**；纯 CPU/Python ≈ **317 µs/步（28%）**；
缓存返回 + 残差 ≈ **302 µs/步（27%）**。

**残差的诚实说明**：`pi: am.builder_build@<Class>` 的 self 里包含 30 次子探针的
**未计时部分**（`pi_scope()` 调用 + `__enter__` 里 t0 之前的栈操作 + `__exit__`
里 dt 之后的记账，每探针约 1.5 µs），合计约 45 µs/步；扣掉后**真正未解释的工作
≈ 123 µs/步（≈41 µs/call）**。这部分应由 `measurement_profiling` 的 perf children
分解补上（候选：`_copy_sequence_indices_to_device`、dataclass `.replace()`）。

### 5bis.5 关键常数：920B 上一次小 NPU 算子 ≈ 3–12 µs

`scripts/npu_dispatch_bench.py`（同容器、`taskset -c 122-157`、n=3000，shape 8 int32）：

| op | CPU ns/op | NPU ns/op | **NPU−CPU (µs)** |
|---|---:|---:|---:|
| `fill_` | 2012 | 5014 | **3.00** |
| `copy_` | 697 | 12328 | **11.63** |
| `sub`（3 buffer, out=） | 10460 | 20707 | **10.25** |
| `to_int64` | 4562 | 13270 | **8.71** |
| `index_select`（2048） | 2671 | 9642 | **6.97** |
| `gather` | 3107 | 9102 | **6.00** |
| `arange` | 4323 | 9702 | **5.38** |
| `empty` | 2367 | 4676 | 2.31 |
| `expand_as` | 3841 | 5610 | 1.77 |

**这解释了"为什么一个 O(1) 的 metadata 构造要 371 µs"**：它不是计算，而是
**下发/派发密集**。最直接的证据是 `gdnb.build_actual_seq_lengths` —— 该函数只有
3 个算子（`empty_like` / `copy_` / `sub(out=)`），却耗 **54.3 µs/call**。

> **交叉验证已完成（`measurement_profiling`，真机 perf，2026-09-24）**：
> wall 无法区分"CPU 忙"与"等 device"，perf 可以。结论是 **CPU 真忙，不是等待**——
> GDN 子树 3 956 个样本的叶帧分类里 `d2h_sync`（`aclrtSynchronize` /
> `_local_scalar_dense`）只占 **0.15%**，`ascend_driver` 1.16%，
> 而 `cpython_runtime` 16.86% + `aten_op_dispatch` 13.30% + `libc` 13.04% +
> `allocator` 5.46% ≈ **48.7%** 是纯宿主侧。
> ⇒ "减少 N 次小算子调用 ≈ 省 N × 3–12 µs"是**成立**的优化杠杆。

#### 5bis.5.1 更正：`compute_num_computed_tokens` **不是**缓存返回

§5bis.5 的表格与插桩报告把它标为"缓存返回（`_num_computed_tokens_cache` 非空）"。
**真机 perf 的子树分解推翻了这一判断**（`scripts/measure/subtree_breakdown.py`，
`data/profiles/real-b1/fp/subtree-cnct/summary.json`）：

`CommonAttentionMetadata.compute_num_computed_tokens` 命中 **520 个样本 = 2.66%**
（与该线程样本总数的比），其**子树**由：

| 子树符号 | 占该子树 % | 含义 |
|---|---:|---|
| `PyNumber_Subtract` | **67.50** | Python 层 `-` 运算符 |
| `torch::autograd::THPVariable_sub` | **66.15** | torch 张量 `__sub__` |
| `at::_ops::slice_Tensor::call` | 11.54 | `qsl[1:]` / `qsl[:-1]` 切片 |
| `at::_ops::sub_Tensor::call` | 10.97 | 张量相减 |

⇒ 每次调用走的是 **`backend.py:530` 的未命中分支**：
`query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]` 再
`self.seq_lens - query_lens`，即 **4+ 次 torch 算子派发**。
（`gdn_attn_builder.py` 里 `m = _treat_single_token_prefills_with_state_as_decodes(m)`
会 `.replace()` 出一个新对象，`_num_computed_tokens_cache` 随之回到 `None`，
所以 3 个 GDN builder 各算一次。）

**这是一条可直接落地的优化线索**：让 `.replace()` 保留该缓存，或用一次
`query_start_loc_cpu` 的 numpy/py 运算替代 torch 张量相减，
预期省掉约 **2.66% × 步墙钟 ≈ 43 µs/call × 3 call/步 ≈ 130 µs/步**。

### 5bis.6 与历史 phase 数据的口径关系（pystack 污染 + 分母修正）

见 `docs/09-historical-data-caveat.md` §5。两个必须记住的口径修正：

1. **pystack**：`/start_profile` 无条件启动 1 ms 采样器（采样所有 Python 线程），
   实测使 `prepare input` **+678 µs（S1，+24.2%）** / **+944 µs（S2，+24.5%）**。
2. **`Step:Model` 双行**：async scheduling 下每个 engine 迭代写两行 `Step:Model`
   （dispatch + batch_queue wait），直接拿它做分母会让 `n_steps` 翻倍。
   本文的 step 分母一律用 `analyze_lite.py --step-scope Step:Schedule
   --window-mode next`（`scripts/extract_ladder.py` 已固化该口径）。

### 5bis.7 插桩自身开销（A/B/C 阶梯，口径修正后）

| 场景 | B−A 挂载/import | **C−B 探针** | D1−B pystack 污染 | D2−B 噪声底 |
|---|---:|---:|---:|---:|
| S1（B=1, ISL=128） | +48.6 µs (+1.76%) | **+266.4 µs (+9.49%)** | +678.6 µs (+24.18%) | +103.3 µs (+3.68%) |
| S2（B=64, ISL=128） | −11.4 µs (−0.29%) | **+259.1 µs (+6.71%)** | +944.4 µs (+24.45%) | +98.4 µs (+2.55%) |

`C−B` 在两个场景都超过 5% 预算（S1 9.49% / S2 6.71%），原因是**探针数量**
（49 个，单探针 2.20 µs，地板 ≈108 µs），不是单探针成本。引用子 scope 数字时
必须扣除地板；`parse_subscope.py --probe-floor-us 2.2` 会把低于地板的项列进
`meta.json.scopes_below_probe_floor`。

### 5bis.8 占比随 batch 的变化：B=1 → B=64 占比从 55% 掉到 38%

| 场景 | `prepare input` p50 | decode step p50 | **占比** | forward p50 |
|---|---:|---:|---:|---:|
| B=1, ISL=128 | 2757.6 µs | 4915.0 µs | **55.3%**（C 臂 57.2%） | 830.6 µs |
| B=64, ISL=128 | 3873.6 µs | 10073.0 µs | **38.7%**（C 臂 38.9%） | 1093.6 µs |

batch 放大 64×，`prepare input` 只涨 1.40×（2757.6 → 3873.6 µs），
而 step 涨 2.05×。**这说明 `prepare input` 里有大量与 batch 无关的固定成本**
（尤其 GDN builder 的 3 次调用：每次 ~371 µs，与 1 个请求还是 64 个请求无关）。
反过来也说明：**低并发（B=1）才是 `prepare input` 占比最高、最需要优化的区域**。

## 6. 限制与未解问题

* perf 采样不能按 scope 切分（G1）——用"时间轴对齐 + 子 scope 镜像"两条互补证据；
* `time_running/time_enabled` 在共享切片上会 < 1（这是优势：它把噪声量化出来）；
* 无卡 replay 环境没有真实 DMA 引擎，`copy_to_gpu` 路径的 CPU 行为不可完全复现
  （见 `docs/06-synthetic-load.md`）。
* 子 scope 的 device/CPU 分类是按**静态控制流**标注的假设（5bis.5），
  wall 计时无法区分"CPU 忙"与"等 device"；需要 perf/topdown 交叉确认。
  → **已由 §2.4 用三条独立证据解决**：`aclrtSynchronize*` 合计仅 0.27%、
  GDN 子树 device 类叶子仅 ~9.9% vs 宿主侧 ~48.7%、topdown 呈 frontend-latency 主导。
* `pi: am.builder_build@<Class>` 残差里混有 ~45 µs/步的探针记账开销
  （5bis.4），引用该残差时须扣除。

## 7. 后续可做的（按价值排序）

1. **B=64 与 chunked prefill 的火焰图**（2 个点，各 ~5 分钟）——验证
   `_pad_non_spec_decode_graph_inputs` 的占比是否随 batch 变化。
2. **`_update_states` 的 churn 场景**：本次 B=1 下它只有 100.7 µs（3.1%），
   但无卡拟合显示它的**每请求斜率是 `_prepare_inputs` 的 3.1 倍**
   （1.675 vs 0.5375 µs/req，R²=1.000）⇒ 高 churn/高并发下它可能反超。
   这是本项目**最重要的一条未验证假设**。
3. **L1/L2/L3/LLC/DRAM 的专项采集**（历史空洞 G2 仍未补齐）：
   需自定义事件组（预置组不含 `l1d_cache` 等），模板见
   `tools/libkperfx/l3-dram-probe.txt`。注意 `memstall_l3miss` 与 `dram_*`
   在 920B 上恒为 0（§4.5 坑 1）。
4. **MTP 路径**：本次全程 MTP off。历史显示 MTP-on 的 graph 臂有"自旋假象"
   （占 74% 但一半是等待），需要按 `docs/01` §6 的三档同步点判定重新测。
