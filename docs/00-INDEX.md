# vLLM 0.26.0 `prepare_input` 阶段 CPU 侧负载分析

> 分析对象：**vLLM 0.26.0**（commit `568afb3a1`）+ **vllm-ascend 0.26.0rc1**（commit `f2f74a16c`）
> 目标硬件：Kunpeng 920B（4×80 core / 8 NUMA）+ Ascend 910（A3），physical chip 3
> 采集时间：2026-09-24（Asia/Shanghai）
> 本目录 = 交付包根目录

![scope boundary](figures/01-scope-boundary.svg)

---

## 0. 一分钟结论

1. **`prepare_input` 在低负载 decode 下确实是宿主瓶颈，且占比可达 55–70%。**
   实测（0.8B / TP1 / B=1 / ISL=128 / decode / `FULL_DECODE_ONLY` / pystack 关闭）：
   `prepare input` p50 = **2757.6 µs**，单步 p50 = **4915 µs** → **占比 55.3%**
   （`Step:Schedule` 口径；若用 `Step:Model` 口径则为 69.5%，见 §2.1 的口径警告）。
   并发升到 B=64 时占比反降到 **38.7%**（`prepare input` 只涨 ×1.40）
   ⇒ **低并发才是最该被优化的区域**。

2. **它的成本结构不是"算得多"，而是"CPython 解释器执行海量小对象操作"。**
   真机 topdown：`frontend_bound 66.01%`（其中 `frontend_latency 59.89%`）、
   `retiring 12.85%`、`IPC 0.771`；**同步调用合计仅 0.27%**
   （`aclrtSynchronize*` 全族）⇒ **不是等卡**。火焰图 top-10 极平（合 22.83%，
   top-1 `_PyEval_EvalFrameDefault` 仅 10.35%），而 `eventfd_write`(1.93%)、
   `pthread_mutex_lock`(1.78%)、`_PyType_Lookup`(1.67%)、`_PyObject_GenericGetAttrWithDict`(1.36%)
   共同刻画了"**小对象多、属性查找多、逐条派发**"的指纹。
   ⇒ 优化方向是**减少 Python 层对象与调用条数**，不是减少数据量。

   > **两条来自子代理的精确化修正（很重要，避免误判优化方向）**：
   > - **`eventfd_write` / `pthread_mutex_lock` 不是 GIL，也不是 vLLM 的 IPC**，
   >   而是 **CANN 的 per-op 开销**：`eventfd_write` 的 63.1% 来自
   >   `at_npu::native::OpCommand::RunOpApiV2`、13.2% 来自 `c10_npu::NPUEvent::record`；
   >   `pthread_mutex_lock` 只有 26.5% 来自 `take_gil`+`drop_gil`。
   >   ⇒ **free-threading（PEP 703）完全不对症**，且本栈没有 `cp313t/cp314t` 轮子。
   > - **GC 方向已结案**：真机 20 s 采集里含 `gc_collect_main`/`visit_decref`/`visit_reachable`
   >   的周期权重 = **0**（vLLM 的 `freeze_gc_heap()` 已生效）。
   > - 真正的引用计数成本（`object.h:642/646` 两行）≈ `_PyEval` self 的 23.6%，
   >   折算 **≈4.4% 的 `prepare_input`** —— 方向对，但**量级是 4% 级，不是主导项**。

3. **头号热点已定位到具体函数、具体调用点。**
   `vllm_ascend/ops/gdn_attn_builder.py::AscendGDNAttentionMetadataBuilder.build`
   —— 每步 **3 次调用 × 303 µs = 908 µs，占 `prepare input` 29.3%**；perf 独立确认其
   self = **20.10%**、最大子项 `_pad_non_spec_decode_graph_inputs` = **5.94%**。
   同期 full-attention 的 `AscendAttentionMetadataBuilder.build` 只有 1 次 × 36 µs（1.2%）。
   这是 **Qwen3.5 混合架构**（6 层 full attention + 18 层 linear attention/GDN）特有的路径。

   **可直接落地的优化线索**：`CommonAttentionMetadata.compute_num_computed_tokens`
   （`vllm/v1/attention/backend.py:530`）本可命中缓存，但
   `_treat_single_token_prefills_with_state_as_decodes()` 的 `.replace()` 会新建对象
   并把 `_num_computed_tokens_cache` 清空 ⇒ **3 个 GDN builder 各重算一次**，
   每次走 4+ 次 torch 算子派发（实测子树 67.5% 在 `PyNumber_Subtract`）。
   修法：让 `.replace()` 保留缓存，或用 numpy 运算替代 torch 张量相减。
   **预期收益 ≈ 130 µs/步（约 4% 的 `prepare input`）。**

   **被证伪的假设**（避免后人重走）：`docs/01` 预测的"每拍新建 pinned 缓冲"
   不成立——`pin_memory` 0.12% + `tolist` 0.07% + `index_select` 0.07% +
   `aclrtSynchronize` 0.18% 合计 **< 0.5%**。

4. **无卡 harness 的定位（重要）**：
   真机 `prepare input` scope = 2806 µs，其中 **908 µs（32%）是 GDN attention metadata
   builder**（`_build_attention_metadata` 内），属于 harness P1 未覆盖范围；harness 覆盖的
   `_update_states + _prepare_inputs` 子集（**同启动参数**：`max_model_len=2048`、
   `max_num_reqs=8`、`max_num_batched_tokens=2048`）实测 **379 µs/步**。
   两者的差额**不能全部归因于 shim**：其中 **≥944 µs（39%）是 P1 范围外的代码**
   （GDN builder + full-attn builder + `_preprocess`/`update_cos_sin`/同步），
   其余才是 H2D/DMA、Triton launch、acl device op 等 shim 缺口。
   920B 上**单次小操作固定开销 2–9 µs**，而 `prepare_input` 一步里有**几十次**这种小操作
   → 优化对象是"调用条数"与"重复 3 次的 GDN builder"，不是数据量。
   完整归因表见 `06-synthetic-load.md` §6.2。

   harness 与真机的**偏差方向是单向"偏乐观"**：IPC 0.949 vs 0.771（+23%）、
   frontend 72.1% vs 66.0%（+6.1 pp），因为 harness 缺 `eventfd_write`/`pthread_mutex_lock`
   这类跨进程通信与锁。⇒ **harness 用于相对比较与趋势拟合，绝对占比一律用真机数字。**
   一致性判据结算：热点 **top-20 重合 16/20 = 80% ✅**、top-1 同符号 ✅、
   扁平度同形 ✅（top-10 21.4% vs 22.83%）；topdown 分量与 IPC 未达 ±2~3 pp / ±10%
   （原因已逐条归因，见 `06-synthetic-load.md` §9）。

   > ⚠️ **保真度第一约束**：harness 的 `--preset` 必须逐字匹配真机启动参数。
   > 实测 IPC：realmachine 口径 **0.949** vs 错误口径（`max_model_len` 用模型自身的
   > 262144）**1.553**。原因：`token_ids_cpu_tensor` 形状 `(max_num_reqs, max_model_len)`
   > 决定它是 L1/L2 常驻（64 KB）还是每步 DRAM 流量（67 MB）。
   > 详见 `06-synthetic-load.md` §1.5 / §5。

5. **历史公开数字需要修正。** 既有项目记录的"prepare input 占 59.6%"来自
   LiteProfiler `POST /start_profile` 采集，而该接口会**无条件**启动 1 ms 间隔的
   Python 栈采样器；实测该采样器让 `prepare input` 自身 **+678 µs（+24.2%）**、
   单步 **+919 µs**。修正方法与修正后数值见 `09-historical-data-caveat.md`。

6. **口径纪律（三条，踩过就会得出错误结论）**
   - **占比分母**：async 下每轮 engine 循环写**两行** `Step:Model`，用它当分母会得到
     69.5%，正确值 55.3%。一律用 `--step-scope Step:Schedule --window-mode next`。
   - **pystack**：占比测量必须 `--pystack-interval-us 0`，否则 +24%。
   - **探针代价**：49 个探针有 2.2 µs/个的地板（合计 +9.5%），引用子 scope 绝对值时
     必须扣除；p50 < 2.2 µs 的子项不可引用。

7. **`prepare_input` scope 归属已被独立验证**：perf 采样落在 `prepare input` 窗口内的
   比例 **55.00%**（10 761/19 564）与墙钟并集比 **54.88%** 只差 0.12 pp，
   并与探针独立测得的 55.3%（A 臂）吻合。见 `05-hotspots.md` §1.1。

---

## 1. 文档地图

| 文档 | 内容 | 状态 |
|---|---|---|
| **`01-prepare-input-code-logic.md`** | 边界定义、26 个子步骤逐条逻辑、数据结构 IO 语义、Ascend 特有分支、调用图 | ✅ 完成 |
| **`02-complexity-and-factors.md`** | 一阶复杂度模型、24 项影响因素矩阵、prefill/decode 差异 | ✅ 完成 |
| `03-share-in-inference.md` | 在推理中的占比（历史基线 + 新实测）、判据公式 | 🔄 实测回填中 |
| `04-profiling-methodology.md` | perf / flamegraph-rs / libkperfx / conda 隔离 / 噪声门 | ✅ 完成 |
| `05-hotspots.md` | 热点函数 / 子阶段分解 / 热点源码 / 热点指令 / 火焰图 | 🔄 回填中 |
| `06-synthetic-load.md` | 无卡负载构造、保真度证据、D1–D10 偏差表 | ✅ 完成 |
| `07-bottleneck-analysis.md` | 成本模型、瓶颈判据、放大/掩盖机制、优化清单 | ✅ 模型完成，实测待回填 |
| `08-reproduction-manual.md` | 从零复现全流程 | 🔄 |
| `09-historical-data-caveat.md` | pystack 采样器污染：事实链、影响、修正 | ✅ 完成（数值回填中） |

### 追加分析（2026-09-24 第二轮）

| 文档 | 内容 | 状态 |
|---|---|---|
| **`10-prepare-input-graphification.md`** | **`prepare_input` 能否也走向"图下发"**：三条前提、已有雏形、TPOT→1ms 预算推演 | ✅ |
| **`10-cpython-directions/`** | **CPython 优化方向**（7 篇，含构建配置量化验收基线） | ✅ |
| **`11-ge-whole-graph/`** | **GE 整图下发调研**（4 篇 + 导航）：能力边界、为何未使能、对 `prepare_input` 的适用性 | ✅ |

其中三篇最值得先读：

| 文档 | 一句话 |
|---|---|
| `10-cpython-directions/06-build-config-experiment.md` | 镜像的 libpython **已经是** computed gotos；**剩余构建杠杆是 PGO+LTO**：微基准 −22.6%、真实 `prepare_input` 路径 **−15.6%** |
| `10-cpython-directions/03-torch-dispatch.md` | **上游已有两个 open PR 可直接抄**（GDN metadata 3 组去重，260–600 µs/步） |
| `11-ge-whole-graph/README.md` | **GE 曾经使能过、在 v0.12 被主动移除**；且"GE 完全抹除 host 开销"在官方文档里没有依据 |

**方向子系列** `10-cpython-directions/`（每条一个优化方向，独立成文）：

| 文档 | 内容 | 状态 |
|---|---|---|
| `10-cpython-directions/00-runtime-build-config.md` | 运行时**构建配置**：量化基线、验收判据、剩余杠杆（**已按真机实验修正两次**） | ✅ 完成 |
| **`10-cpython-directions/06-build-config-experiment.md`** | **构建配置真机实验**：三臂 A/B/C，静态+行为证据、微基准、PMU 归因、真实 `prepare_input_us`；**结论：镜像已经是 computed gotos，剩余杠杆是 PGO+LTO（−15.6%）** | ✅ 完成 |
| `10-cpython-directions/01-interpreter-core.md` | 解释器核心（PEP 659 / computed goto / JIT）文献调研 | ✅ 完成 |
| `10-cpython-directions/02-object-model.md` | 对象模型方向 | ✅ 完成 |
| `10-cpython-directions/03-torch-dispatch.md` | torch 派发方向 | ✅ 完成 |
| `10-cpython-directions/05-graph-dispatch-host-side.md` | host 侧图化/派发方向 | ✅ 完成 |

数据与图：

| 目录 | 内容 |
|---|---|
| `data/static/` | 57 步机器可读分解 `prepare-input-steps.json` |
| `data/historical/` | 从既有项目挖出的 15 个 CSV（2424 行 phase 窗口 / 362 窗口 topdown / 60 吞吐点） |
| `data/measure/` | 本次真机占比矩阵 |
| `data/subscope/` | 子阶段分解（含 pystack 污染的 A/B/C/D 四臂阶梯） |
| `data/profiles/` | perf / libkperfx 原始采集与汇总（不含大体积 perf.data） |
| `data/harness/` | 无卡 harness 的扫描与 A/B 数据 |
| `data/model/` | 复杂度模型 + 920B 微基准 |
| `figures/` | 全部图（SVG + PNG） |

---

## 2. 关键数字速查

### 2.1 真机占比（0.8B / TP1 / decode / graph / pystack off）

| 阶段 | p50 | 占单步 |
|---|---|---|
| `prepare input` | **2757.6 µs**（基线臂 A）/ 2806.2 µs（挂载臂 B） | **55.3% / 55.6%** |
| `forward` | 827.6 / 837.1 µs | 16.8% / 16.8% |
| 单步（`Step:Schedule` 口径） | 4915 / 4979 µs | 100% |

> ⚠️ **占比分母口径**：async scheduling 下每轮 engine 循环写两行 `Step:Model`
> （dispatch + batch_queue wait），用它当分母会得到 69.5% 这个**偏大**的数。
> 本表用 `Step:Schedule` 口径（每轮一行），正确值是 **55.3%**。
> 两个口径都报，但**引用时必须写明用的是哪个**。

### 2.2 `prepare input` 内部构成（同一配置）

> 分母 = `prepare input` p50 ≈ **3294 µs**（C2c 臂，含 61 个探针 ≈134 µs 地板；
> 扣掉后约 3160 µs）。子项按 *self time* 计，不重复计数。

| 子 scope | 每步 p50 | 占 `prepare input` |
|---|---|---|
| **`AscendGDNAttentionMetadataBuilder.build` × 3** | **1091.7 µs** | **33.1%** |
| ├ `gdn.pad_graph_inputs` | 294.1 µs | 8.9% |
| ├ ctor + 未归因残差 | 373.6 µs | 11.3% |
| ├ `gdn.treat_single_token` | 167.7 µs | 5.1% |
| ├ `gdn.compute_num_computed_tokens` | 129.2 µs | 3.9% |
| ├ `gdn.split_decodes` | 117.8 µs | 3.6% |
| └ `gdn.mamba_block_table` + `attach_metadata` | 9.2 µs | 0.3% |
| `AscendAttentionMetadataBuilder.build` × 1 | 124.0 µs | 3.8% |
| **两条 builder 合计** | **1215.7 µs** | **36.9%** |
| `in.slot_mapping` | 138.2 µs | 4.2% |
| `in.positions_assembly` | 113.7 µs | 3.5% |
| `sync_input_prep` | 102.1 µs | 3.1% |
| `update_states` | 100.7 µs | 3.1% |
| `in.block_table_commit` | 88.8 µs | 2.7% |
| `am.cm_base_pre` | 78.0 µs | 2.4% |
| 其余 ~28 项 | 合计约 1.2 ms | 约 36% |

> 交叉验证：`pi: am.builder_build` 的 inclusive = 1219.6 µs ≈ 1215.7 µs ✓ 闭合。

### 2.3 插桩/采样的代价（必读，用于校正任何引用）

| 项 | 差值 | 说明 |
|---|---|---|
| 挂载代价 B−A | **+48.7 µs（+1.77%）** | 多 import 一个模块 |
| **探针代价 C−B** | **+266.4 µs（+9.49%）** | 49 个探针 × 2.2 µs 地板 |
| **pystack 污染 D1−B** | **+678.5 µs（+24.18%）** | 历史数据就带着这个 |

---

## 3. 环境与复现入口

```bash
# 真机服务（chip3 = /dev/davinci3，CPU 120-159，NUMA 1）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash scripts/launch_subscope_service.sh --mode on --serve-only ...'

# 无卡 harness（不需要任何 NPU 设备）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash harness/scripts/pi-docker.sh --preset realmachine --batch 1 --isl 128'

# PMU / 火焰图
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash scripts/flamegraph.sh record --tid <TID> --seconds 20'
```

完整流程见 `08-reproduction-manual.md`。
