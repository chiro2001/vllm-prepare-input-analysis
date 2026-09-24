# 为什么 vllm-ascend 现在没有使能 GE 整图下发

> 调研日期：**2026-09-24**（所有 URL 的"访问日期"均为该日，下文不再逐条重复）。
> 方式：**纯源码 + 网络文献**调研。未登录任何远程主机、未运行 NPU、未起容器、未修改任何被调研仓库。
>
> **本文与 [00-source-evidence.md](/home/chiro/projects/vllm/preparing-input-phase/docs/11-ge-whole-graph/00-source-evidence.md) 的关系**：`00` 记录了 vllm-ascend 0.26.0rc1
> 的**现状源码证据**（compiler_interface.py 的 `force_eager` / `reduce-overhead`、graph_mode.md 的两条图路径、
> ACL_Graph.md 的 host 侧参数更新）。本文**不重复这些内容**，只做 `00` 未覆盖的五件事：
> **A 历史脉络、B 公开 issue/PR、C 阻塞点清单、D 逐条技术核查、E MindIE 等对照组、F 上游 vLLM 视角、
> G 公开计划**，并明确列出**查不到的**。

---

## 0. 对问题前提的两点修正（先说结论）

用户问题隐含两个前提，**其中一个不成立，另一个需要精确化**。

### 0.1 「vllm-ascend 曾经用过 GE 吗？」——**用过，而且不是"短暂试用"，是正式特性**

**vllm-ascend 从 v0.9.x 到 v0.11.0，正式提供过 GE 图模式路径，叫 `TorchAirGraph`，官方文档原话是
"This is the GE graph mode"**，并在 **v0.12.0rc1 被移除**，理由是 "aclgraph is stable and fast now"。
（用 `v0.9.x` 而非 `v0.9.1rc1` 是保守写法：旧 `graph_mode.md` 只说 "In v0.9.1rc1, only DeepSeek series
models are supported"，而 PR [#789](https://github.com/vllm-project/vllm-ascend/pull/789)
`feat: support torchair graph mode in v1 engine` 出现在 2025-05-08（v0.9.0 周期）。
**v0.9.1 之前是否已有该路径，本次未逐一核实。**）

这不是推测，是 `docs/source/user_guide/feature_guide/graph_mode.md` 在 PR #4814 里被**删除的原文**：

```diff
-There are three kinds for graph mode supported by vLLM Ascend:
+There are two kinds for graph mode supported by vLLM Ascend:
 - **ACLGraph**: This is the default graph mode supported by vLLM Ascend. In v0.9.1rc1, Qwen and Deepseek series models are well tested.
-**TorchAirGraph**: This is the GE graph mode. In v0.9.1rc1, only DeepSeek series models are supported.
 - **XliteGraph**: This is the euler xlite graph mode. In v0.11.0, only Llama and Qwen dense serise models are supported.
...
-# Using TorchAirGraph
-
-If you want to run DeepSeek series models with the graph mode, you should use [TorchAirGraph](...).
-# TorchAirGraph only works without chunked-prefill now
```
来源：<https://github.com/vllm-project/vllm-ascend/pull/4814/files>（`docs/source/user_guide/feature_guide/graph_mode.md` 的 patch）

**所以"为什么没使能"这个问题应该改写为**：
> 「GE 整图**曾经**是 vllm-ascend 的正式图路径之一，**为什么在 v0.12 被移除？移除后有没有再回来的计划？**」

### 0.2 「源码里没有 torchair」——**不准确；准确的说法是"没有 GE / Ascend IR 的执行路径"**

0.26.0rc1 源码里**有** `import torchair`，位置在
[patch_npugraph_ex_triton.py:31](/home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend/vllm_ascend/patch/worker/patch_npugraph_ex_triton.py:31)：

```python
try:
    import npugraph_ex as nge          # 新路径
except ImportError:
    import torchair as nge             # ← 向后兼容回退
```

但在 [compiler_interface.py:133](/home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend/vllm_ascend/compilation/compiler_interface.py:133)
两条分支都被**显式配置为不做 IR 变换**：

```python
# torchair 分支
# mode="reduce-overhead": use aclgraph mode, avoid fx graph to Ascend IR transformation.
config.mode = "reduce-overhead"
```

**精确表述**：vllm-ascend 保留了 torchair/npugraph_ex 这个**软件包**（因为 FX 优化 pass 依赖它），
但**从未在 0.26.0rc1 里走 torchair → Ascend IR → GE 编译**这条路。这与 §0.1 的历史并不矛盾：
GE 路径存在于 v0.9.1–v0.11 的 `vllm_ascend/torchair/` 目录，该目录已被删除。

---

## 1. A. 历史脉络：GE 整图的完整生命周期

### 1.1 三种"图"必须先分清

调研中最大的混淆源是"图"这个词。本文严格区分三者：

| 名称 | 机制 | 是否经过 GE | 是否需要静态 shape |
|---|---|---|---|
| **Eager** | 逐算子下发 | 否 | 否 |
| **ACLGraph / npugraph_ex** | `aclmdlRICapture*` **Capture & Replay** | **否** | 是（capture 时的 shape） |
| **GE / Ascend IR（TorchAirGraph）** | FX → Ascend IR → **GE 编译** → 下沉调度 | **是** | **是，且编译期穷举** |

来源（权威）：TorchAir 官方文档 `overview.md`「目前图执行分为两种模式」——
> **基于 npugraph_ex 后端的图模式（aclgraph）**：……采用 Capture&Replay 方式……该模式一般通过 Runtime 提供的 **aclmdlRICaptureXxx 系列接口**实现。
> **基于 GE 的图模式（Ascend IR）**：通过设置 TorchAir 的 CompilerConfig 实例属性 **mode="max-autotune"** 开启，其将 PyTorch 的 FX 计算图转换为昇腾中间表示（IR），即 Ascend IR 计算图，**并通过 GE（Graph Engine，图引擎）实现计算图的编译和执行**。

来源：TorchAir 文档仓库 `docs/zh/overview.md`，本地副本
`/home/chiro/projects/vllm/HIST_PROJECT/.research/torchair-docs/docs/zh/overview.md`
（该 sparse clone HEAD = `d29ef3c3f8fd685ceedc921fbe0bad585272b29f`，2026-09-14）；
线上等价入口 <https://gitcode.com/Ascend/torchair/tree/master/docs/zh/overview.md>。

**"整图下发"的权威定义**（GE 官方）：
> 对于输入 tensor shape **固定不变**的静态 shape 的模型，在编译时即可确定所有算子的输入输出 shape……
> 因此，GE 提供了**静态图下沉调度模式**，让模型中的算子在**加载阶段提前以整图的形式下发到 Device 上**，
> 在执行时，只需在 Host 侧下发一个**模型执行的 Task** 即可触发模型在 Device 上调度执行。
> 相比于 Host 调度模式，下沉调度模式可**大大降低 Host 侧调度开销**。

来源：`ge/docs/zh/user_guides/graph_dev/overview/concepts_and_principles.md`
（<https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/graph_dev/overview/concepts_and_principles.md>）

**注意这句话里的前提**：整图下沉调度的成立条件是**静态 shape**。这是后面 §3(a) 一切硬阻塞的根源。

### 1.2 时间线（每条均有编号或文件行号）

| 时间 | 事件 | 图路径 | 来源 |
|---|---|---|---|
| 2025-06-28 | `[1/N][Feat] Implement primal full graph with limited scenario` | ACLGraph（雏形） | PR [#1503](https://github.com/vllm-project/vllm-ascend/pull/1503) |
| 2025-07-07 | `[RFC]: Support Full Graph with multiple attention kernels` | ACLGraph | RFC [#1649](https://github.com/vllm-project/vllm-ascend/issues/1649) |
| 2025-07-31 | `[Feat][Graph] Support FULL_DECODE_ONLY mode for GQA/MHA models` | ACLGraph | PR [#2128](https://github.com/vllm-project/vllm-ascend/pull/2128) |
| 2025-08-06 | `[V1] MTP supports torchair` | **GE** | PR [#2145](https://github.com/vllm-project/vllm-ascend/pull/2145) |
| 2025-08-21 | 新增 `torchair_graph_config.mode` 配置项 | **GE / ACLGraph 二选一** | PR [#2461](https://github.com/vllm-project/vllm-ascend/pull/2461) |
| 2025-09-03 | `Torchair graph mode works with tp > 4 now` | **GE** | release notes `v0.9.1`，`docs/source/user_guide/release_notes.md:1869` |
| 2025-10-27 | `e2e test, use GE graph mode for torchair` / `test ge graph for torchair` / `add torchair ge test` | **GE** | PR [#3780](https://github.com/vllm-project/vllm-ascend/pull/3780)、[#3783](https://github.com/vllm-project/vllm-ascend/pull/3783)、[#3788](https://github.com/vllm-project/vllm-ascend/pull/3788) |
| 2025-11-10 | `Torchair is deprecated. We'll remove it once the performance of ACL Graph is good enough. The deadline is Q1 2026.` | — | release notes `v0.11.0rc1`，`release_notes.md:1643` |
| 2025-12-01 | `[Bug]: Torchair doesn't worker with v1 scheduler`（3/8 e2e 用例失败） | **GE** | issue [#4581](https://github.com/vllm-project/vllm-ascend/issues/4581) |
| 2025-12-05 | `Reduce the cost of torchair` / `Support multi graphs for torchair`（**移除前最后的优化，但两者都未合并**） | **GE** | PR [#4756](https://github.com/vllm-project/vllm-ascend/pull/4756)、[#4757](https://github.com/vllm-project/vllm-ascend/pull/4757)（均 `merged_at = null`） |
| 2025-12-10 | `Drop torchair` 合并 | **GE 移除** | PR [#4814](https://github.com/vllm-project/vllm-ascend/pull/4814) |
| 2025-12-10 | `cleanup useless torchair logic` | — | PR [#4856](https://github.com/vllm-project/vllm-ascend/pull/4856) |
| 2025-12-13 | `Torchair graph mode is removed. --additional-config {"torchair_graph_config":{"enabled":true}} doesn't work anymore. Please use aclgraph instead.` | — | release notes `v0.12.0rc1`，`release_notes.md:1547` |
| 2025-12-04 | `[Feature] Support npugraph_ex backend` 合并 | npugraph_ex（FX 优化，非 GE） | PR [#4700](https://github.com/vllm-project/vllm-ascend/pull/4700)、RFC [#4715](https://github.com/vllm-project/vllm-ascend/issues/4715) |
| 2026-02-10 | `[npugraph_ex] enable npugraph_ex by default` | npugraph_ex | PR [#6664](https://github.com/vllm-project/vllm-ascend/pull/6664) |
| 2026-02-06 | `Torchair has been dropped. #4814`（正式 release note 确认） | — | `release_notes.md:1288`（v0.13.0） |
| 2026-05-28 | `[Refactor] migrate compilation backend from torchair to npugraph_ex`（连 `import torchair` 都改成 `import npugraph_ex`） | — | PR [#9201](https://github.com/vllm-project/vllm-ascend/pull/9201) |

### 1.3 曾经的 GE 路径**确实是 GE**：`mode` 默认值就是 `max-autotune`

这是本节最关键的一条推理链，三个环节都有出处：

**环节 1**——`torchair_graph_config.mode` 默认值是**空字符串**，代表"不设置 config.mode"：
```python
# vllm_ascend/ascend_config.py（v0.11.0rc3，raw 源码）
class TorchairGraphConfig:
    def __init__(self, torchair_graph_config, vllm_config, additional_config):
        self.enabled = torchair_graph_config.get("enabled", False)
        self.mode = torchair_graph_config.get("mode", '')      # ← 默认 ''
```
来源：<https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.11.0rc3/vllm_ascend/ascend_config.py>（`class TorchairGraphConfig`，第 134–141 行）

**环节 2**——不设置时就不覆盖 `config.mode`，沿用 TorchAir 的**默认模式**：
```python
# vllm_ascend/torchair/torchair_model_runner.py（v0.11.0rc3）
config = torchair.CompilerConfig()
if self.ascend_config.torchair_graph_config.mode:
    config.mode = self.ascend_config.torchair_graph_config.mode     # ← 空串则不赋值
```
来源：<https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.11.0rc3/vllm_ascend/torchair/torchair_model_runner.py>（`_get_torchair_lazy_compiled_model`）

**环节 3**——TorchAir 的**默认 mode 就是 `max-autotune`，即 GE 图模式**：
> GE 图模式一般通过 TorchAir 的 CompilerConfig 属性 **mode="max-autotune"** 开启（**该模式是系统默认模式**），
> 其将 FX 图转换为 Ascend IR 图，并通过 GE 图引擎实现图编译和执行。

来源：TorchAir 文档 `docs/zh/ascend_ir/quick_start.md`（本地副本
`/home/chiro/projects/vllm/HIST_PROJECT/.research/torchair-docs/docs/zh/ascend_ir/quick_start.md`）

⇒ **结论：v0.9.1–v0.11 的 `torchair_graph_config.enabled=True` 默认走的就是 GE（Ascend IR）整图路径。**
只有用户显式写 `mode="reduce-overhead"` 时才退化成 ACLGraph。这解释了为什么官方文档把 `TorchAirGraph` 直接叫做
"the GE graph mode"。**证据强度：强**（三个环节分别是 vllm-ascend 源码、vllm-ascend 源码、TorchAir 官方文档）。

### 1.4 为什么换掉：可查到的理由

**官方给的理由只有一句话**（PR #4814 正文）：
> `aclgraph is stable and fast now. Let's drop torchair graph mode now.`
> `TODO: some logic to adapt torchair should be cleaned up as well. We'll do it in the following PR.`

来源：<https://github.com/vllm-project/vllm-ascend/pull/4814>（created 2025-12-09，merged 2025-12-10）

官方从未发布"GE vs ACLGraph 性能对比"文档。但**可查到的可佐证事实**有 6 条：

1. **与 V1 scheduler 组合时有确定性失败**：issue #4581 的 CI 显示
   `test_e2e_deepseekv3_with_torchair_v1scheduler` 等 3 个用例失败（`3 failed, 5 passed, 1 skipped`）。
   维护者的关闭评论是：**"closing as torchair is dropped now"**（2025-12-10，MengqingCao）。
   来源：<https://github.com/vllm-project/vllm-ascend/issues/4581>
   > 注意：该 issue 的堆栈是 `torchair_mla.py:1052` 的 `torch.cat` OOM，**不是**"调度器语义不兼容"的直接证据。
   > 它证明的是"TorchAir 路径在 V1 引擎下稳定性不足"。证据强度：中。

2. **与 ACLGraph 互斥**：启用 torchair 时强制把 ACLGraph 关掉
   ```python
   # vllm_ascend/platform.py（v0.11.0rc3）
   if ascend_config.torchair_graph_config.enabled:
       logger.info("Torchair compilation enabled on NPU. Setting CUDAGraphMode to NONE")
       compilation_config.cudagraph_mode = CUDAGraphMode.NONE
   ```
   来源：<https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.11.0rc3/vllm_ascend/platform.py>；
   引入该行为的 PR 是 [#2154](https://github.com/vllm-project/vllm-ascend/pull/2154)
   "Turn off aclgraph when enabling TorchAir"（见 `release_notes.md:1957`）。

3. **图缓存不稳定，默认必须删除**（同一段 v0.11 源码）：
   > `We delete the torchair cache folder here to prevent runtime issues caused by dimension mismatches or
   > configuration inconsistencies when users reuse cached computation graphs. Though this will increase
   > graph compilation duration, it significantly enhances robustness and decreases graph launching time during inference.`
   ⇒ 也就是说**默认每次启动都要重新编图**，官方建议用户开 `use_cached_graph` + `use_cached_kv_cache_bytes` 才能减少编译时间。

4. **模型覆盖窄**：旧版 `graph_mode.md` 原文——`In v0.9.1rc1, only DeepSeek series models are supported`；
   并且 `TorchAirGraph only works without chunked-prefill now`。

5. **上游 TorchAir 自己弃用了 `reduce-overhead`**（注意：这是 ACLGraph 那条分支，不是 GE 那条）：
   > 从 TorchNPU 7.3.0 之后的版本开始，**原 reduce-overhead 模式（aclgraph）通过 config.mode 配置图编译后端的方式将不再演进**，
   > 也不再推荐使用，请您切换 **npugraph_ex** 后端以启用 aclgraph 模式。
   来源：TorchAir `docs/zh/overview.md`「兼容性说明」
   ⇒ vllm-ascend 后来（PR #9201）跟着迁移到 npugraph_ex，与这条上游策略一致。

6. **维护成本不对称**：GE 路径需要一整套**平行的模型实现**。
   v0.11.0rc3 的 `vllm_ascend/torchair/` 目录含
   `models/`、`ops/`、`quantization/`、`torchair_model_runner.py`、`torchair_worker.py`、`torchair_attention.py`、
   `torchair_mla.py`、`torchair_sfa.py`——即**为 GE 单独维护的一份模型/算子/量化实现**。
   来源：<https://github.com/vllm-project/vllm-ascend/tree/v0.11.0rc3/vllm_ascend/torchair>
   对比之下 ACLGraph 复用的是上游 vLLM 的 `CUDAGraphMode` / `BatchDescriptor` / `CUDAGraphWrapper` 抽象
   （vllm-ascend 只需提供 `ACLGraphWrapper` 这一层平台适配，见 `00-source-evidence.md`）。**证据强度：强。**

### 1.5 历史脉络小结

> **GE 整图不是"没做"，而是"做了、只覆盖 DeepSeek、与 V1 引擎/A​CLGraph 组合时问题多、
> 每档 shape 要重编图、且需要一整套平行模型实现；当 ACLGraph 达到'stable and fast'后，
> 投入产出比不成立，于是在 v0.12.0rc1 被移除。"**

---

## 2. B. 公开 issue / PR（编号 + 状态 + 时间）

### 2.1 与 GE / TorchAir 直接相关

| 编号 | 类型 | 标题 | 状态 | 创建 | 关闭/合并 |
|---|---|---|---|---|---|
| [#2145](https://github.com/vllm-project/vllm-ascend/pull/2145) | PR | `[V1] MTP supports torchair` | closed | 2025-07-31 | 2025-08-06 |
| [#2154](https://github.com/vllm-project/vllm-ascend/pull/2154) | PR | `Turn off aclgraph when enabling TorchAir` | closed | — | — |
| [#2461](https://github.com/vllm-project/vllm-ascend/pull/2461) | PR | `refact runner model v1`（引入 `torchair_graph_config.mode`） | closed | — | 2025-08-21 |
| [#3780](https://github.com/vllm-project/vllm-ascend/pull/3780) / [#3783](https://github.com/vllm-project/vllm-ascend/pull/3783) / [#3788](https://github.com/vllm-project/vllm-ascend/pull/3788) | PR | `e2e test, use GE graph mode for torchair` / `test ge graph for torchair` / `add torchair ge test` | closed（**未合并**，`merged=null`） | 2025-10-27 | — |
| [#4581](https://github.com/vllm-project/vllm-ascend/issues/4581) | ISSUE | `[Bug]: Torchair doesn't worker with v1 scheduler` | closed | 2025-12-01 | 2025-12-10（"closing as torchair is dropped now"） |
| [#4756](https://github.com/vllm-project/vllm-ascend/pull/4756) | PR | `[Feature] Reduce the cost of torchair` | closed（**未合并**） | 2025-12-05 | `merged_at=null` |
| [#4757](https://github.com/vllm-project/vllm-ascend/pull/4757) | PR | `[Feature] Support multi graphs for torchair` | closed（**未合并**） | 2025-12-05 | `merged_at=null` |
| [#4814](https://github.com/vllm-project/vllm-ascend/pull/4814) | PR | **`Drop torchair`** | **merged** | 2025-12-09 | **2025-12-10** |
| [#4856](https://github.com/vllm-project/vllm-ascend/pull/4856) | PR | `cleanup useless torchair logic` | **merged** | 2025-12-10 | 2025-12-11 |
| [#4927](https://github.com/vllm-project/vllm-ascend/pull/4927) | PR | `[MoE][TorchAir] Remove FusedMoEState` | **merged** | 2025-12-11 | 2025-12-12 |
| [#3881](https://github.com/vllm-project/vllm-ascend/issues/3881) | ISSUE | `[Usage]: deepseek torchair混部多TP场景下，HCCL_OP_EXPANSION_MODE="AIV"环境变量引起编图卡死` | **open** | 2025-10-29 | — |

> **注意：以下 5 个 PR 全部 `merged_at = null`（closed 但未合并）**：
> `#3780`（GE 图模式 e2e 测试）、`#3783`（GE 图测试）、`#3788`（torchair GE 测试）、
> `#4756`（`Reduce the cost of torchair`）、`#4757`（`Support multi graphs for torchair`）。
>
> 这条组合信号值得注意：**"给 GE 路径补测试"和"降低 GE 路径成本"这两类工作**都没有落到主干上，
> 而同期"移除 torchair"（#4814）以及其后的清理（#4856、#4927）则**全部合并了**。
> 也就是说，移除前最后一段时间，社区实际推进的方向已经是"拆"而不是"补"。
> 证据强度：中（能证明这 5 个 PR 未合并，不能证明它们未合并的原因；也不能证明仓库里完全没有 GE 测试 —— 见 §8 第 7 条）。

### 2.2 npugraph_ex（ACLGraph 的编译期优化层，非 GE）

| 编号 | 类型 | 标题 | 状态 | 创建 | 关闭 |
|---|---|---|---|---|---|
| [#4715](https://github.com/vllm-project/vllm-ascend/issues/4715) | **RFC** | `[RFC]: npugraph_ex backend` | **open（至今）** | 2025-12-04 | — |
| [#4700](https://github.com/vllm-project/vllm-ascend/pull/4700) | PR | `[Feature] Support npugraph_ex backend` | merged | 2025-12-04 | 2025-12-10 |
| [#6214](https://github.com/vllm-project/vllm-ascend/issues/6214) | RFC | `[RFC]: make npugraph_ex become default compile backend` | closed | 2026-01-24 | 2026-03-03 |
| [#6664](https://github.com/vllm-project/vllm-ascend/pull/6664) | PR | `[npugraph_ex] enable npugraph_ex by default` | merged | 2026-02-10 | 2026-02-12 |
| [#9201](https://github.com/vllm-project/vllm-ascend/pull/9201) | PR | `[Refactor] migrate compilation backend from torchair to npugraph_ex` | merged | 2026-05-15 | 2026-05-28 |

RFC #4715 的开篇原文（**决定了"为什么是 npugraph_ex 而不是 GE"**）：
> In order to achieve better performance when using aclgraph, we introduced the **npugraph_ex backend**,
> which is a simple, easy-to-boundary, and **accuracy-concern-free fullgraph aclgraph acceleration solution**.
> ... All the work of npugraph_ex is **based on fx.graph to enhance the experience of aclgraph**.

并且 RFC 的"Any Other Things"第 1 条明确：
> The purpose of adding this backend is to achieve optimal performance, so **it is recommended to use it in fullgraph mode**.

⇒ **vllm-ascend 选择的"整图"是 ACLGraph 意义上的 fullgraph（capture 整个 forward），
不是 GE 意义上的 Ascend IR 下沉图。** 证据强度：强。

RFC #4715 里维护者（ChenCangtao，2026-01-21）给出的实测收益：
> 离线场景约 **TPOT 提升 5–8%**（Qwen 与 Deepseek 模型）。
（该评论引用 <https://mp.weixin.qq.com/s/SZgGlvASIq0T6p_H_yK_xw>，本次调研未打开微信链接核对原文。）

### 2.3 关于「GE / 整图 / max-autotune」的检索结论

在 vllm-ascend 仓库用 GitHub search（`repo:vllm-project/vllm-ascend`）检索以下关键词：

- `torchair`：608 条结果，全部落在 2025-02 至 2025-12 区间（移除前后）；
- `max-autotune`：**0 条命中 issue/PR**；
- `GE graph` / `GE 整图`：命中的都是不相关结果（"GE" 被当成交换机型号、或子串匹配）。

**⇒ 自 2025-12-10（#4814）之后，vllm-ascend 仓库中没有任何关于 GE 图模式的新 issue 或 PR。**
（这是"检索范围内无命中"，不等于"绝对没有讨论"。）

---

## 3. C. 阻塞点清单

> 分类标准：
> **(a) 硬阻塞** = 在现行技术前提下做不到，或必须推翻现有架构；
> **(b) 成本阻塞** = 技术上可行，但工程量/启动时间/内存/维护代价不划算；
> **(c) 历史/组织原因** = 与当下技术无关，是路径依赖或协作结构造成的；
> **(d) 已解决或正在解决** = 曾经的阻塞点现在已有对策。

### (a) 硬阻塞（技术上做不到）

| # | 阻塞点 | 证据 | 强度 |
|---|---|---|---|
| **a-1** | **GE 整图下沉调度的前提是"编译期已知的静态 shape"**。LLM decode 的 batch/token 数每步变化，而 GE 的编译期与执行期职责严格分离：`未命中任何档位的输入 shape 会导致执行失败` | GE 官方：`ge/docs/zh/user_guides/graph_dev/overview/concepts_and_principles.md`（"对于输入 tensor shape **固定不变**的静态 shape 的模型……"）；GE `atc_shape_configuration_guide.md`（"**未命中任何档位的输入 shape 会导致执行失败**"） | **强** |
| **a-2** | **GE 的"动态分档"无法覆盖 batch=1**：档位值**不能包含 0 或 1**，因为 Dynamo 对 dim∈{0,1} 不做符号化，会**重新成图** | TorchAir `ascend_ir/api/inference/set_dim_gears.md` 约束原文 | **强** |
| **a-3** | **GE 的"动态分档"不接受私有格式张量**（`FRACTAL_NZ`、`NC1HWC0`），而 vllm-ascend 的 torchair 路径恰恰把权重转成 `ACL_FORMAT_FRACTAL_NZ` | TorchAir `ascend_ir/api/inference/set_dim_gears.md` + `ascend_ir/features/advanced/dynamic_gears_merge_policy.md`（"本功能要求网络中参与分档的 Tensor 不能传入私有格式"）；vllm-ascend v0.11 `torchair_model_runner.py` 中 `converting_weight_acl_format(self.model, ACL_FORMAT_FRACTAL_NZ)` | **中强**（两条都在仓库外/历史 tag，未在 0.26 现状中复现） |
| **a-4** | **自定义算子进 GE 需要额外交付件**：除 PyTorch Schema、Ascend C 实现、OpPlugin 适配、Meta 符号化之外，还要实现 **Ascend Converter**（PyTorch 算子 → Ascend IR）。**没有 Converter 的算子进不了 max-autotune 图** | TorchAir `custom_op_graph/overview.md`：「GE 图模式（mode=max-autotune）……**该模式需要实现 Ascend Converter 交付件**，完成 PyTorch 算子转换为 Ascend IR」；场景 1 需完成步骤 1–6 | **强** |
| **a-5** | **Triton 算子难以进 GE**：vllm-ascend 的图里含 `triton_kernel_wrapper` 调用，其 kernel 索引（`kernel_side_table`）是**进程内注册、无法跨进程序列化**；vllm-ascend 因此只能"跳过缓存、当场重编"，无法把含 Triton 的图当作可复用的整图产物 | 0.26.0rc1 `vllm_ascend/compilation/compiler_interface.py`（Triton 缓存跳过逻辑）；PR [#9874](https://github.com/vllm-project/vllm-ascend/pull/9874) `[BugFix] Skip npugraph_ex cache for graphs containing Triton kernels` | **中**（这是针对 npugraph_ex 缓存机制的证据；"Triton 无法进 GE"是从"GE 需 Ascend IR Converter"**推断**的，未查到 TorchAir 对 Triton 算子的专门说明） |

> **对 a-1/a-2 的重要限定**：a-1/a-2 是"**GE 动态分档**"这条路的硬阻塞，**不是"GE 整图"整体的硬阻塞**。
> 完全可以为每个 batch 档位编译一张**独立静态图**（这正是 vllm-ascend 那个时代 `graph_batch_sizes` 做的事，
> 见 v0.11 `torchair_model_runner._get_torchair_lazy_compiled_model(batch_size)`）。
> 所以严格说：**"动态 shape" 本身没有把 GE 整图判死，把它推入成本阻塞的是"每档一套图的编译代价"（见 b-1）。**

### (b) 成本阻塞（能做，但不划算）

| # | 阻塞点 | 证据 | 强度 |
|---|---|---|---|
| **b-1** | **每档 shape 都要编译，编译时间以十分钟计**。官方文档给出的量级是"startup 增加 **数分钟到数十分钟**"（这是 static kernel 的数字，同属 GE 侧编译行为），并且"编译阶段耗时较长，但通常只需执行一次" | vllm-ascend `docs/source/user_guide/feature_guide/graph_mode.md:184`（"...may add **several minutes to tens of minutes** to the startup time..."）；GE 官方 `atc_shape_configuration_guide.md` | **强**（对"编译慢"这一事实）；**中**（对"GE 整图在 vllm-ascend 上的具体启动时间"——无实测） |
| **b-2** | **图编译缓存不稳定，默认被删除**：v0.11 平台代码在每个启动周期删除 torchair cache，官方注释说这是为了避免"维度不匹配 / 配置不一致"导致运行时问题，并承认这会**增加图编译耗时** | v0.11.0rc3 `vllm_ascend/platform.py`（见 §1.4 第 3 条引用） | **强** |
| **b-3** | **社区已上报编译开销过高**：`[Performance]: The overhead of cache_compile is too high.` | issue [#588](https://github.com/vllm-project/vllm-ascend/issues/588)（2025-04-21，closed） | **中** |
| **b-4** | **算子覆盖需要逐个补 Converter**：vllm-ascend 的算子面很宽——`vllm_ascend/ops/`、`vllm_ascend/_cann_ops_custom/`、`csrc/` 自定义 AscendC 算子、外加 triton-ascend kernel。每个不在 TorchAir ATen 清单里的算子都要写 Converter | TorchAir `appendix/aten_api.md` 开头原文："如果自定义模型用到的 ATen API 不在表 1，说明对应的 API 能力可能不完备，**用户需根据实际情况进行 Converter 适配实现算子入图**" | **强**（对机制）；**中**（对"vllm-ascend 具体有多少算子缺 Converter"——未逐个数） |
| **b-5** | **收益上限不明确**：官方给出的 npugraph_ex（非 GE）实测收益是 "TPOT 提升 5–8%"，且其成本远低于 GE；没有公开数据说明"GE 整图会比 ACLGraph fullgraph 再多省多少 host 时间" | RFC [#4715](https://github.com/vllm-project/vllm-ascend/issues/4715) 评论（2026-01-21） | **中**（收益侧证据缺失） |

### (c) 历史/组织原因

| # | 阻塞点 | 证据 | 强度 |
|---|---|---|---|
| **c-1** | **TorchAir 路径与 V1 引擎的组合稳定性不足**，且最终以"torchair 已被移除"为理由关闭 bug | issue [#4581](https://github.com/vllm-project/vllm-ascend/issues/4581)（3 failed / 5 passed） | 中 |
| **c-2** | **GE 路径与 ACLGraph 互斥**，用户只能在两者间二选一，无法叠加 | v0.11 `platform.py`（`cudagraph_mode = CUDAGraphMode.NONE`）；PR [#2154](https://github.com/vllm-project/vllm-ascend/pull/2154) | 强 |
| **c-3** | **GE 路径只覆盖 DeepSeek 系列**，模型面窄 | 旧版 `graph_mode.md`（PR #4814 删除的原文） | 强 |
| **c-4** | **上游 TorchAir 的战略是抛弃 `reduce-overhead`、主推 npugraph_ex**，vllm-ascend 顺势而为 | TorchAir `docs/zh/overview.md`「兼容性说明」；vllm-ascend PR [#9201](https://github.com/vllm-project/vllm-ascend/pull/9201) | 强 |
| **c-5** | **维护成本不对称**：GE 路径要求一套平行模型/算子/量化实现（`vllm_ascend/torchair/{models,ops,quantization,...}`），而 ACLGraph 复用上游 vLLM 抽象 | v0.11.0rc3 目录树；0.26.0rc1 `platform.py` 的 `get_static_graph_wrapper_cls()` | 强 |
| **c-6** | **组织目标是"跟上 vLLM 主线"（main2main），而非维护第二条图后端**。repo 有专门的 `[Misc]feat: adapt to vLLM main (...)` 系列 PR 和 `main2main` 流程 | 例如 [#14691](https://github.com/vllm-project/vllm-ascend/pull/14691)、[#14750](https://github.com/vllm-project/vllm-ascend/pull/14750)（均 2026-08） | 中 |

### (d) 已解决或正在解决

| # | 阻塞点 | 现在的对策 | 证据 |
|---|---|---|---|
| **d-1** | 动态 batch/shape | ACLGraph 用 **capture size 分档 + padding** 解决：`1,2,4` + `8` 的倍数 + `16` 的倍数直到 `max_cudagraph_capture_size`；超出最大档位则回退 eager | vllm-ascend `docs/source/developer_guide/Design_Documents/ACL_Graph.md`「Capture Sizes and Bucketing」 |
| **d-2** | attention 算子每步元数据 | `update_full_graph_params()` + `torch.npu.graph_task_update_begin/end` + `ExternalEvent` 排序 | `ACL_Graph.md`「Host-side attention parameter update for full graph replay」；源码 `attention_v1.py:476`、`525`、`619`、`809` |
| **d-3** | 图捕获的 stream 资源上限 | 新的 `FULL_AND_PIECEWISE` 默认模式 + HDK 25.5.1+/CANN 8.5.0+ 解除旧 stream 预算限制，可捕获约 **32K（A3）/ 64K（Ascend 950）** 张图 | PR [#9572](https://github.com/vllm-project/vllm-ascend/pull/9572)、[#9962](https://github.com/vllm-project/vllm-ascend/pull/9962)（release notes `:415`、`:424`） |
| **d-4** | 算子融合/下沉的一部分收益 | **npugraph_ex** 以 FX pass 的形式提供（add+rms_norm→npu_add_rms_norm、MatmulAllReduceAddRMSNorm 等），并用 static kernel 预编译固定 shape 的算子二进制 | `graph_mode.md`「Using Npugraph_ex」；`npugraph_ex.md`；RFC [#4715](https://github.com/vllm-project/vllm-ascend/issues/4715) |
| **d-5** | 多步 draft 的 host 开销 | ACLGraph **merged graph**：把多步 draft 捕获成一张图 | release notes `:628`（PR [#5553](https://github.com/vllm-project/vllm-ascend/pull/5553)、[#5940](https://github.com/vllm-project/vllm-ascend/pull/5940)）、`:771`（PR [#6860](https://github.com/vllm-project/vllm-ascend/pull/6860)） |
| **d-6** | 310P（Atlas 300I DUO）不支持 npugraph_ex | 自动检测并禁用，并在文档中给出配置方法 | `graph_mode.md`「Atlas inference products」注；PR [#10874](https://github.com/vllm-project/vllm-ascend/pull/10874) |

---

## 4. D. 逐条技术核查

对用户点名的 6 个候选阻塞点逐条核查。**结论先行：其中只有 2 条是"真阻塞"，2 条是"工程取舍"，
1 条已被现有机制化解，1 条（生态/维护）是移除的真实主因。**

### D-1 动态 batch/shape

**判定：对"GE 动态分档"是真阻塞；对"每档一张静态图"是成本阻塞。**

- GE 下沉调度的前提是静态 shape（§3 a-1）。
- GE 确实提供"动态 shape 图分档执行"，但约束苛刻（§3 a-2、a-3）：
  - 档位值**不能含 0 或 1**——因为 "动态 FX graph 中 dim 值符号化的最大表示范围是 [2, ∞)，因此当 dim 为 0 或 1 时，不会命中动态的 FX graph，**需要重新成图**"。这直接命中 decode **batch=1** 的场景。
  - 总档位数 **≤ 100**。
  - 参与分档的 tensor **不能是 `FRACTAL_NZ`/`NC1HWC0`** 私有格式。
  - **未命中档位 → 编译或执行报错**（不是回退，是失败）。
  - 需与 `torch.compile(dynamic=True)` 搭配，且 `set_dim_gears` 只在首次执行时设置。
  来源：TorchAir `ascend_ir/api/inference/set_dim_gears.md`、`ascend_ir/features/advanced/dynamic_gears_merge_policy.md`
- 历史实现走的是"**每个 batch 档位一张图**"的路线（v0.11 `graph_batch_sizes`），
  并且官方承认这会带来缓存维度不匹配问题（§1.4 第 3 条）。

**为什么这不构成"GE 整图做不到"的证明**：vllm-ascend 现在用的 ACLGraph 也是靠分档 + padding，
分档本身不是新问题。差异在于 **ACLGraph 的分档是"运行时捕获 N 张图"，GE 的分档是"编译期编译 N 张图"**，
后者的启动代价高一个量级（§4 D-2）。

### D-2 未知 shape 的图编译时间

**判定：成本阻塞，且有官方量化。**

- 官方对 GE 侧编译的表述：**"编译阶段耗时较长，但通常只需执行一次"**（GE `atc_shape_configuration_guide.md`）。
- vllm-ascend 官方对同类编译（static kernel，本质是"为固定 shape 预编译算子二进制"）的表述是
  **"may add several minutes to tens of minutes to the startup time"**，
  并说明"Once completed, subsequent request processing is not affected"。
  来源：`docs/source/user_guide/feature_guide/graph_mode.md:184`
- 社区侧诉求：issue [#588](https://github.com/vllm-project/vllm-ascend/issues/588) `[Performance]: The overhead of cache_compile is too high.`
- 内存侧：编译后的图是**常驻资源**。vllm-ascend 有过专门的"图捕获 OOM / KV cache 记账"修复
  （PR [#8111](https://github.com/vllm-project/vllm-ascend/pull/8111) `Fix the graph capturing OOM in model_runner_v2`、
  [#9865](https://github.com/vllm-project/vllm-ascend/pull/9865) `Added ACL graph memory estimation before KV cache allocation`，
  见 `release_notes.md:441` 与 `:857`）。
  ⇒ 分档数 × 每档图内存 直接挤压 KV cache 预算。这条对 GE 与 ACLGraph 同样成立，但 GE 编译产物更大。

**证据强度：强**（"编译慢"是确定的）；**中**（"GE 整图在 vllm-ascend 上具体多少分钟"没有公开实测）。

### D-3 算子覆盖（FIA / MoE / csrc 自定义算子 / Triton）

**判定：成本阻塞为主；Triton 接近硬阻塞。**

- **FIA**：`npu_fused_infer_attention_score` 本来就是 CANN 内置算子，历史上在 GE 路径下也被使用
  （`attention_v1.py` 与旧 `torchair_attention.py` 都调它）。**不是阻塞。**
- **MoE**：历史上 GE 路径有专门的 `vllm_ascend/torchair/models/`（如 `torchair_deepseek_v2.py`，
  见 issue #4581 的堆栈）和 `[refactor] Refactoring AscendFusedMoE`（PR [#2438](https://github.com/vllm-project/vllm-ascend/pull/2438)）。
  也就是说 **MoE 进 GE 需要一份平行实现**，这属于成本而非不可能。**成本阻塞。**
- **自定义 AscendC 算子（`csrc/`、`vllm_ascend/ops/`、`vllm_ascend/_cann_ops_custom/`）**：
  每个都要实现 Ascend Converter 才能进 GE（§3 a-4）。**成本阻塞，且是长尾成本。**
- **Triton（triton-ascend）**：这是最接近硬阻塞的一项。
  0.26.0rc1 明确记录：含 `triton_kernel_wrapper` 的图**不能做跨进程缓存**，原因是
  "Triton kernel indices (`kernel_side_table`) are registered **in-process** at compile time and are
  **not serializable across process boundaries**"，否则新进程加载会触发
  `kernel_side_table.get_kernel()` 的 `AssertionError`。
  来源：`vllm_ascend/compilation/compiler_interface.py`（`patched_get_compiled_gm`）；
  PR [#9874](https://github.com/vllm-project/vllm-ascend/pull/9874)
  ⇒ Triton kernel 是**动态注册的进程内句柄**，要变成 GE 可编译的静态图节点，需要 triton-ascend 侧提供
  Ascend IR Converter，**本次调研未找到该能力的任何证据**。
- **反证**：PR [#2113](https://github.com/vllm-project/vllm-ascend/pull/2113) `[core] Support capture custom ops into aclgraph`
  说明"自定义算子入图"这个问题在 **ACLGraph 侧**已经解决过一次。这佐证了"vllm-ascend 选择在 ACLGraph 上解决算子入图，
  而不是在 GE 上再解决一遍"。

**证据强度：中强。**

### D-4 投机解码 / MTP 的 draft 多步循环

**判定：已不是阻塞；历史上 GE 下也跑通过。**

- GE 时代就有 MTP 支持：PR [#2145](https://github.com/vllm-project/vllm-ascend/pull/2145) `[V1] MTP supports torchair`（2025-08-06，merged）、
  PR [#1090](https://github.com/vllm-project/vllm-ascend/pull/1090) `Adapting torchchair for mtp`、
  PR [#1244](https://github.com/vllm-project/vllm-ascend/pull/1244) `[Draft] Add MTP dummy_run and Adapt torchair graph mode`、
  PR [#1294](https://github.com/vllm-project/vllm-ascend/pull/1294)。
  release notes `:1870`「MTP support torchair graph mode now #2145」。
  ⇒ **draft 多步循环进 GE 图在历史上是可用的，不构成硬阻塞。**
- ACLGraph 时代进一步做了 **merged graph**（把多步 draft 捕成一张图）：
  release notes `:628`「ACLGraph now support capturing a single merged graph for multi-step drafts,
  which greatly reduce host bound in multi-step spec decoding case!」→ PR [#5553](https://github.com/vllm-project/vllm-ascend/pull/5553)、[#5940](https://github.com/vllm-project/vllm-ascend/pull/5940)；
  MTP 版本见 PR [#6860](https://github.com/vllm-project/vllm-ascend/pull/6860)（release notes `:771`）。
- 仍有活跃缺陷（说明这条路径**复杂但可行**，不是被否决）：
  issue [#13639](https://github.com/vllm-project/vllm-ascend/issues/13639) `Speculative decoding hangs in FullGraph mode (MRV2)`（closed）；
  issue [#42271](https://github.com/vllm-project/vllm/issues/42271)（上游 vLLM）`MTP + FULL_AND_PIECEWISE cudagraph deadlocks ...`。

**证据强度：强。**

### D-5 调度与 CPU 重叠（async scheduling）

**判定：部分真实，但不是"GE 做不到"，而是"两个方向互相挤压"。分为三条：**

**(1) capture/replay 与 host 侧参数更新的顺序问题——真实存在，且有官方文档描述。**
> `ACLGraphWrapper` synchronizes the current stream before replay in the common path to ensure that
> host-side parameter updates stay aligned with the graph execution that will consume them.
> **This is especially relevant in asynchronous scheduling or multi-threaded execution.**
> If ordering is not preserved, ... the attention operator may run with mismatched runtime metadata,
> which can cause incorrect results, precision issues, or even hangs.

来源：`docs/source/developer_guide/Design_Documents/ACL_Graph.md`「Replay ordering and synchronization」
⇒ 这是**现有 ACLGraph 设计**为兼容 async scheduling 付出的复杂度。它说明"图 + 异步调度"需要额外的排序保证。

**(2) 历史上 TorchAir 与 V1 引擎的组合出现过确定性失败。**
issue [#4581](https://github.com/vllm-project/vllm-ascend/issues/4581) 的 CI 里
`test_e2e_deepseekv3_with_torchair_v1scheduler` 失败，最终以"torchair is dropped"关闭。
**但该 issue 的报错是 `torchair_mla.py` 的 `torch.cat` OOM，不是调度语义冲突**，所以它不能证明
"async scheduling 与 GE 不兼容"，只能证明"TorchAir 在 V1 引擎下稳定性不足"。**证据强度：中（且不宜过度解读）。**

**(3) 上游 `max_concurrent_batches=2` 的流水并未被图破坏——恰恰相反。**
vLLM 上游用 `max_concurrent_batches`（`vllm/config/vllm.py:512`）作为 batch 队列深度，
而 decode 的"批间流水"靠的是"候选批次查表 + 在调度器继续跑的同时 worker 正在执行上一批"。
**图捕获/回放只替换了"一批内部如何下发"，不改变"批次之间如何排队"**，
所以整图（无论是 ACLGraph fullgraph 还是 GE 整图）在原理上**不破坏**这个流水。
这一点从 vllm-ascend 大量"async scheduling + 图模式"的现存代码也可反证
（`vllm_ascend/worker/model_runner_v1.py` 中 `use_async_scheduling` 与 graph 路径并存）。
**证据强度：中**（架构层面的推断 + 代码共存事实）。

**(4) 一个方向性的观察**：GE 整图的价值主张就是"**把 host 侧下发降到 1 个 task**"，
而 async scheduling 的价值主张是"**让 host 侧调度与 device 计算重叠**"。两者目标一致（都是削减 host 瓶颈），
但**实现手段互斥**：GE 要求编译期知道一切，async scheduling 要求运行时动态决定下一批。
这是"取舍"而非"不可能"。**证据强度：中**（分析性结论，无直接文献）。

### D-6 生态与维护：TorchAir × PyTorch 版本耦合、与 torch.compile 的关系

**判定：真实阻塞，而且是移除的**主因**。分三条：**

**(1) TorchAir 与 torch_npu / CANN 强绑定，无独立包。**
> 目前 TorchAir **暂未提供独立软件包**，而是作为 TorchNPU 的三方库，随着 TorchNPU 包一起发布。
> 需要注意的是：当安装的 TorchNPU 版本为 7.3.0 及之后版本，均可正常使用 TorchAir……
> 为确保正常使用 TorchAir 功能，**PyTorch 建议使用 2.6.0 及以上版本**。

来源：TorchAir `docs/zh/overview.md`「安装」
⇒ vllm-ascend 的 `Dependencies` 在每个版本都要锁
`PyTorch / torch_npu / CANN / Triton Ascend` 四件套（见 0.26.0rc1 release notes「Dependencies」节）。
TorchAir 的 GE 行为随之浮动，**升级 torch_npu 就可能改变图编译结果**。

**(2) 上游已经把 `reduce-overhead` 判了"不再演进"，主推 npugraph_ex。**
见 §1.4 第 5 条引用。vllm-ascend 的应对就是 PR #9201 的"零残留 torchair 迁移"。

**(3) 与 `torch.compile` 的关系：不是竞争，是"同一管道的不同落点"。**
TorchAir 官方对两者的定位很明确：
> TorchAir 在 TorchNPU 中的位置如图 1 所示，图中左侧为**单算子执行模式（Eager）**，
> 右侧为 **torch.compile 图执行模式（Graph）**。

来源：TorchAir `docs/zh/overview.md`「概述」
而 vllm-ascend 通过 vLLM 的 Adaptor 机制在 torch.compile **内部**挂后端
（`AscendCompiler(CompilerInterface)`，见 `compiler_interface.py`）。
⇒ **ACLGraph 与 GE 是同一个 torch.compile 管道下游的两个分叉**，
切换它们**不需要换掉 torch.compile**，只需要换 `CompilerConfig`。
这也解释了为什么"用 GE 还是用 ACLGraph"在工程上是一个**可逆的配置级决定**（v0.11 的 `torchair_graph_config.mode`
就是这个开关），而不是架构级分裂。

---

## 5. E. MindIE 等对照组（本节是回答"能不能照搬"的关键）

### 5.1 MindIE 确实有"整图"后端，但它不是 TorchAir/GE-on-torch.compile

**MindIE LLM 官方架构文档**（`docs/zh/developer_guide/architecture_design/architecture_overview.md`）原文：

> **Modeling**：推理引擎后端，专注模型运行时的性能优化。通过 CustomLayer 形式，提供高效的算子编排、下发、执行接口，
> **支持 ACLGraph 和 ATBGraph 两种图模式后端**。
> - Layer：模型通用内置模块，包括 Attention、Embedding、ColumnLinear、RowLinear、MLP、MoE 等。
> - **Compilation：图引擎后端，将模型从 eager mode 转换为 graph mode，完成整图下发执行**，进而提升推理性能。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/developer_guide/architecture_design/architecture_overview.md>
（镜像：<https://github.com/verylucky01/MindIE-LLM/blob/f032cd3f/docs/zh/developer_guide/architecture_design/architecture_overview.md>）

目录结构里 `mindie_llm/modeling/model_wrapper/atb` 是「**ATBGraph 后端抽象**」，
`examples/atb_models/atb_framework` 是「ATBGraph 运行框架」。

> **重要限定**：MindIE 的"整图"是 **ATB 图**（ATB = Ascend Transformer Boost，C++ 加速库，
> 见 <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/acce/ascendtb/ascendtb_0001.html>）。
> **本次调研未找到权威说明证明 ATB 的 `GraphOperation` 字面上经过 GE 编译**——
> ATB 的图执行接口是 `op->Execute(variantPack, workspace, workspaceSize, context)`，
> 这是 ATB 自己的执行接口，不是 `ge::RunGraphWithStreamAsync`。
> 所以：**"MindIE 用了 GE 整图"这个说法本次未能证实；能证实的是"MindIE 用了 ATB 整图下沉"。**
> 二者在"整图下发"这个效果上等价，在"走不走 GE"上不等价。**这一点必须如实标注。**

### 5.2 MindIE 怎么组图：**手写显式图，不是 trace**

MindIE-LLM 的模型构建流程（以 Llama 为例，来自华为云社区对 MindIE 源码的解析）：

```python
def init_graph(self):
    """Initialze weight, prefill graph and decode graph."""
    self.weight = self.get_weights()
    self.prefill_graph = AtbGraph(f"{self.name}_prefill_graph")
    self.build_graph(self.prefill_graph, is_prefill=True)
    self.decode_graph = AtbGraph(f"{self.name}_decode_graph")
    self.build_graph(self.decode_graph, is_prefill=False)
```

```python
def build_graph(self, graph, is_prefill):
    kv_cache_names = []
    for i in range(self.config.num_hidden_layers):
        kv_cache_names.extend([f"layer_{i}_k_cache", f"layer_{i}_v_cache"])
    graph.add_input_output(
        input=list(self.weight.keys()) + kv_cache_names + self.get_in_tensor_names(is_prefill),
        output=self.get_out_tensor_names())
    self.model.build_graph(graph, is_prefill)
    self.build_lm_head(graph, is_prefill)
    graph.execute_as_single = False
    graph.build()
```

来源：<https://bbs.huaweicloud.com/blogs/455084>（MindIE-LLM ATB 模型推理全流程解析）；
等价代码在 <https://github.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/atb_llm/models/mllama/flash_causal_mllama_atb.py>
等文件中可核对（`build_graph` / `graph.add_input_output` / `graph.add_operation` / `graph.build()`）。

**关键特征**：
1. **每个模型的输入/输出张量是被"声明"出来的**（`get_in_tensor_names` 返回 `['input_ids', 'position_ids',
   'slots_mapping', 'seq_len', 'block_tables', ...]`），不是被 trace 出来的。
2. **权重也是图的输入**（`list(self.weight.keys())`），这与 `frozen_parameter` 类比。
3. **`execute_as_single = False`** 表示允许子图分块执行（`atb_llm/nn/network.py` 的 `AtbGraph` 支持
   `cut_point_idx` 切图 + `sub_ops` 组装，即"自动切图、组图和图融合"）。

### 5.3 MindIE 的 decode 每步元数据怎么处理？——**当成图的输入张量，每步 host 更新**

这是**最关键的一段**，直接回答了"如果 MindIE 用了整图，它怎么处理每步变化的元数据"。
MindIE-LLM 的 `flash_causal_mllama_atb.py` 源码：

```python
def prepare_inputs(self, input_ids, position_ids, is_prefill, kv_cache, block_tables,
                   slots, input_lengths, max_seq_len, lm_head_indices, is_multimodal, **kwargs):
    ...
    target_key = PREFILL if is_prefill else DECODE
    self.graph_inputs[target_key].update({          # ← 每步刷新图输入张量
        "input_ids": input_ids,
        "position_ids": position_ids.to(torch.int64),
        "slots_mapping": slots.to(torch.int32),
        "seq_len": input_lengths.to(torch.int32),
        "block_tables": block_tables.to(torch.int32),
    })
    ...
    # 准备 bind tensor
    self.graph_param[target_key]['seq_len'] = input_lengths.cpu().to(torch.int32)   # ← host 侧参数
    ...

def forward(self, input_ids, position_ids, is_prefill, kv_cache, block_tables, slots,
            input_lengths, max_seq_len, lm_head_indices=None, **kwargs):
    ...
    if is_prefill:
        atb_model_out = prefill_graph.forward(self.graph_inputs[PREFILL],
                                              self.graph_outputs[PREFILL],
                                              self.graph_param[PREFILL])
    else:
        atb_model_out = decode_graph.forward(self.graph_inputs[DECODE],
                                             self.graph_outputs[DECODE],
                                             self.graph_param[DECODE])
```

来源：<https://github.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/atb_llm/models/mllama/flash_causal_mllama_atb.py>

`ATBGraphManager.select_and_execute(context, inputs, runtime_param, **kwargs)` 的 docstring 也印证：
> `runtime_param (str): A json str to indicate cpp graph's **host tensors**.`

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/atb_llm/models/base/graph_manager/graph_manager.py>

**⇒ 结论：MindIE 的"整图"并不是"把一切编进静态图"，而是
「图的**结构**静态 + 图里保留一批**输入张量**和一批 **host 侧参数**，每步由 host 刷新」。**

**这与 vllm-ascend 的 `update_full_graph_params()` 是同一类解法**（见 `00-source-evidence.md` §4：
"some attention operators need runtime metadata updates even when the overall graph is static"）。
`graph_manager.py` 甚至还有一个 `ATBGraphManager` 用于在**多个图变体之间挑选**
（`PrefillGraphWrapper` / `DecodeGraphWrapper` / `SingleLoraGraphWrapper` / `MultiLoraGraphWrapper` /
`SpeculateGraphWrapper` / `SplitFuseGraphWrapper` / `MemPoolGraphWrapper` / `LayerwiseDecodeGraphWrapper` …），
以及一个 `COMPATIBLE_MATRIX` 描述哪些特性可以**叠加成一张图**（`_generate_combinations`）。

⇒ **"每步元数据"不是整图的技术障碍——MindIE 的做法证明它可以被"声明为图输入"来解决。**

### 5.4 那"能不能照搬"？——**不能，卡在算子与模型生态，不在"每步元数据"**

| 维度 | MindIE + ATBGraph | vllm-ascend + ACLGraph（现状） | 能否照搬 |
|---|---|---|---|
| **算子集合** | ATB（C++）自有算子库，为 Transformer 专门设计 | 任意 PyTorch 算子 + `vllm_ascend/ops` + `_cann_ops_custom` + AscendC `csrc/` + triton-ascend | ❌ 换 ATB 等于重写整个算子层 |
| **组图方式** | **手写 `build_graph`**：显式声明输入/输出张量、逐个 `add_operation` | `torch.compile` trace 任意 Python forward + vLLM 的 piecewise 切分 | ❌ trace 出来的图远大于手写图的规模与可控性 |
| **模型实现** | `examples/atb_models/atb_llm/models/<model>/` **每个模型一份专用实现**，且有官方迁移指南（`torcklike_model_migration_guide.md`，要求 `@torch_to_mindie_graph()` 装饰器） | 通用 vLLM modeling，一套代码支持所有模型 | ❌ 模型数量 × 每模型重写 = 不可承受 |
| **每步元数据** | 声明为图输入 + host 参数（§5.3） | `update_full_graph_params()` + `graph_task_update_*`（`00` §4） | ✅ **已经是同一解法，无需照搬** |
| **图变体管理** | `ATBGraphManager` + `COMPATIBLE_MATRIX` 挑选/组合图 | vLLM `CUDAGraphMode` + `BatchDescriptor` 分发 | ✅ 概念等价，各自已实现 |
| **是否需要对齐外部主线** | 不需要（MindIE 自成体系） | **需要**：main2main 跟 vLLM 上游 | ❌ vllm-ascend 无法承受第二条平行栈 |
| **多后端支持** | **自己也支持 ACLGraph**（架构文档原文） | FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE / PIECEWISE 多档 | ⚠️ 说明"整图"并非 MindIE 的唯一选择 |

**可迁移的只有一条**：MindIE 证明了
「**静态图结构 + 动态图输入张量 + host 侧 runtime_param**」这套"半静态"模式足以支撑一个生产级 LLM 服务。
vllm-ascend 的 ACLGraph 已经**独立地到达了同一个设计点**（`00` §4）。
**所以"照搬 MindIE"的净收益 ≈ 0，净成本 ≈ 重写整个算子与模型层。**

### 5.5 其他昇腾生态框架

| 框架 | 图模式 | 证据 | 备注 |
|---|---|---|---|
| **MindIE LLM** | ATBGraph（整图）+ ACLGraph 双后端 | 架构文档（§5.1） | 见上 |
| **MindSpore Serving / MindSpore 图模式** | 静态图（Graph Mode）+ 算子融合 | 昇腾社区/CSDN 案例：Pynative 520 ms → Graph 180 ms → Graph+融合 Cell 125 ms（Atlas 200I DK A2 上的 DeepLabV3+） | **非 LLM** 场景；MindSpore 是图优先框架，"静态图"是原生模式而非后期加成 |
| **SGLang-ascend** | **本次未检索到任何公开的图模式资料** | — | **查不到，见 §8** |
| **XliteGraph（vllm-ascend 内）** | 预配置图路径，需单独安装 `xlite` | `graph_mode.md`「Using XliteGraph」；v0.12.0rc1 release notes「Other」（`release_notes.md` 中"a new graph mode `xlite` is introduced"） | 也是 ACLGraph 族（`vllm_ascend/xlite/xlite.py` 中通过 `ACLGraphWrapper.unwrap` 处理），非 GE |

---

## 6. F. vLLM 上游视角：**上游是"有意解耦"，不是"有意只做分段图"**

### 6.1 结论：用户假设需要修正

**vLLM 主线不是"只做分段图"——它明确有 FULL（整图）模式，而且 FULL 是默认模式的一部分。**
vLLM 官方设计文档（`design/cuda_graphs`）原文：

| 模式 | 官方描述原文 |
|---|---|
| `NONE` | turn CUDA Graphs off. Good for debugging. |
| `PIECEWISE` | a single-mode strategy ... attention or other CUDA Graphs-incompatible operations stay eager, everything else goes into CUDA Graphs. |
| `FULL` | a single-mode strategy, which **only captures full CUDA Graphs** for non-uniform batches, then uniform-decode batches reuse the CUDA Graph of non-uniform batch of the same batch_size |
| `FULL_DECODE_ONLY` | **full CUDA Graph for uniform decode**, no cudagraph for prefill/mixed etc. |
| `FULL_AND_PIECEWISE` | **(default mode)** full CUDA Graph for uniform decode, piecewise CUDA Graphs for others |

来源：<https://docs.vllm.ai/en/latest/design/cuda_graphs/>（vLLM 官方设计文档）
本地等价物：`refs/vllm/vllm/config/compilation.py:53`（`class CUDAGraphMode`）。

⇒ **用户问题里说的 `FULL_DECODE_ONLY` / `FULL`，在 vLLM 的语义里就是"整图"**。
只是这个"整图"= **整模型 forward 被 capture 进一张 CUDA/ACL graph**，
**不是** "整模型被编译成 GE/Ascend IR 下沉模型"。

### 6.2 上游是**刻意**把"图捕获"与"编译"解耦的

官方 design doc 的 Motivation 明确列出了设计目标，其中两条直接相关：
> - **Made full CUDA Graphs support orthogonal to compilation**
> - Separate CUDAGraph capture logic from compilation (as much as feasible) for feature orthogonality, which suggest:
>   - Capturing piecewise and full cudagraphs using the same compiled graph, and
>   - **Full cudagraph capture without compilation.**

并解释了动机：
> ... this tight coupling between compilation and cudagraph capture led to an all-or-nothing experience with little flexibility.

来源：<https://docs.vllm.ai/en/latest/design/cuda_graphs/>

**⇒ 上游的"有意"不是"只做分段图"，而是「把整图做成**与编译器无关的捕获**」。
这直接解释了 vllm-ascend 的实现方式**恰好借用了一句关键：`Full cudagraph capture without compilation`**
—— npugraph_ex 的 `force_eager=True`（不做 IR 变换）+ vLLM 自己的 `ACLGraphWrapper`（做 capture/replay）
正是这条原则的 Ascend 版本。证据：`compiler_interface.py:114-117` 与 `00-source-evidence.md` §2。

### 6.3 vLLM 有没有讨论过"编译式整图"？

**在 vLLM 上游仓库内检索，未找到关于 GE / Ascend IR / 编译期整图下沉的讨论。**
检索 `repo:vllm-project/vllm` 的 `ascend` + `graph` 标题组合，**命中 0 条**；
`full cudagraph` 相关的 52 条 issue/PR **全部是 CUDA/ROCm 侧**的捕获/分发问题
（例如 [#46523](https://github.com/vllm-project/vllm/pull/46523) `Prewarm full cudagraph capture forward pass`、
[#45258](https://github.com/vllm-project/vllm/issues/45258) `[RFC]: FULL cudagraph support for spec-decode drafter chain steps`）。

⇒ **无证据表明 vLLM 上游考虑过"编译期整图下沉"这个方向。**
（这是"检索范围内无命中"，不是"上游从未讨论"。）

---

## 7. G. 公开计划与时间线

### 7.1 明确结论

> **未找到任何"重新引入 GE 整图 / TorchAirGraph / max-autotune"的公开计划。**
> 截至 2026-09-24（本文成稿日），vllm-ascend 仓库中：
> - 最后一条关于 GE 图模式的动作是 **PR #4814 `Drop torchair`（2025-12-10 合并）**；
> - 之后所有图相关的 issue/PR 与 roadmap 都**只涉及 ACLGraph / npugraph_ex / XliteGraph**；
> - **未找到**任何形式的 "GE 整图回归" RFC、issue 或 roadmap 条目。

### 7.2 现行（可查到的）图相关计划

| 计划 | 出处 | 状态 |
|---|---|---|
| npugraph_ex 成为默认编译后端 | RFC [#6214](https://github.com/vllm-project/vllm-ascend/issues/6214)（2026-01-24 → 2026-03-03 closed） | **已完成**（PR [#6664](https://github.com/vllm-project/vllm-ascend/pull/6664)） |
| 彻底去掉 torchair 依赖（迁移到 npugraph_ex） | PR [#9201](https://github.com/vllm-project/vllm-ascend/pull/9201)（2026-05-28 merged） | **已完成** |
| `npugraph_ex support superkernel to reduce TPOT` | vLLM Ascend Roadmap **Q3 2026**，issue [#15067](https://github.com/vllm-project/vllm-ascend/issues/15067) | **未完成（`[ ]`）** |
| npugraph_ex 高级能力的后续项：内存复用、Kernel 性能优化、冗余 kernel 消除、多流、Dynamo cache、权重预取、权重 NZ 预转换 | RFC [#4715](https://github.com/vllm-project/vllm-ascend/issues/4715)「Feature Plan」 | RFC **仍 open** |
| RFC #4715 的遗留问题：`Move the torchair functionality entirely into torch_npu, and no longer explicitly import torchair`（来自 RFC #6214 的 Legacy issue） | RFC [#6214](https://github.com/vllm-project/vllm-ascend/issues/6214) | 部分完成（#9201 已改为 `import npugraph_ex` 优先、torchair 回退） |
| `aclgraph Full mode support`（历史 roadmap） | vLLM Ascend Roadmap **Q1 2026**，issue [#5318](https://github.com/vllm-project/vllm-ascend/issues/5318) | **已完成（`[x]`）** |
| Roadmap Q3 2026 的 `Code rectification` 提到要重构 `graph` 等多个模块 | issue [#15067](https://github.com/vllm-project/vllm-ascend/issues/15067) | 进行中 |

### 7.3 时间线（图表化）

```
2025-06  ACLGraph 雏形 (#1503)          ─┐
2025-07  FULL_DECODE_ONLY (#2128)        │  ACLGraph 与 GE 并行演进
2025-08  TorchAirGraph(GE) 成体系        │  MTP 进 GE (#2145)
         aclgraph 与 torchair 互斥 (#2154)│
2025-10  GE 图模式 e2e 测试 PR (#3780等)─┘  ← 但未被合并
2025-11-10  ★ Torchair 正式弃用（deadline Q1 2026）
2025-12-01  Torchair × V1 scheduler issue (#4581)
2025-12-04  npugraph_ex RFC (#4715) + 首个 PR (#4700)   ← 新方向确立
2025-12-05  最后一批 torchair 优化（#4756/#4757）
2025-12-10  ★ Drop torchair (#4814) 合并 —— GE 整图路径终止
2025-12-13  v0.12.0rc1: "Torchair graph mode is removed"
2026-02-12  npugraph_ex 默认开 (#6664)
2026-05-28  编译后端迁移完成，零残留 torchair (#9201)
2026-09-24  现状：ACLGraph + npugraph_ex（FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE）
            GE 整图：无代码、无测试、无计划
```

---

## 8. H. 明确查不到的（**不要把"没查到"写成"没有"**）

以下事项本次调研**未能查证**，它们在逻辑上可能成立、也可能不成立，**不应被当作结论引用**：

1. **没有找到任何一份"完整解释为什么不选 GE"的设计文档或 RFC。**
   目前最强的官方表述只有两句：PR #4814 的 `aclgraph is stable and fast now. Let's drop torchair graph mode now.`
   与 `compiler_interface.py:133` 的注释 `avoid fx graph to Ascend IR transformation`。
   **"为什么刻意避免 Ascend IR 变换"的完整理由，vllm-ascend 官方从未公开解释过。**
2. **没有找到 GE 路径与 ACLGraph 路径的对照性能数据。**
   仓库内、release notes、roadmap 中均**无**"同模型同负载下 GE vs ACLGraph 的 TPOT/吞吐对比"。
   因此本文无法回答"GE 整图到底能比 ACLGraph 多省多少 host 开销"。
3. **没有找到移除前官方对"shape 分档数 / 编译时间 / 内存"的量化评估。**
   `graph_batch_sizes` 具体设多少个档、每档编译多久、总内存多少 —— 均无公开数据。
4. **未能证实 ATB 的 `GraphOperation` 字面上经过 GE 编译。**
   已证实的是 MindIE 用 ATB 做整图下发（架构文档 + 社区源码解析），
   但 ATB 的执行接口是 `op->Execute(variantPack, workspace, size, context)`，
   与 GE 的 `ge::RunGraphWithStreamAsync` 不是同一个 API。
   **"MindIE 用了 GE 整图"这一说法需要一个独立证据才能成立。**
5. **没有找到 vllm-ascend 关于"SGLang-ascend 图模式"的任何资料**（本次未检索该仓库）。
6. **没有找到"TorchAir/GE × async scheduling"的专门 issue。**
   `use_async_scheduling` 与图模式在 vllm-ascend 源码中共存，但**未找到**任何专门讨论
   "async scheduling 与 GE 整图是否兼容"的公开记录。
7. **没有找到 TorchAir 对 Triton 算子入 GE 的官方说明。**
   §3 a-5 的结论（Triton 难进 GE）中，"Triton kernel 句柄不可跨进程序列化"有直接证据，
   但"因此无法进 GE"是**从 a-4 推断**的。
8. **没有找到 #3780/#3783/#3788/#4756/#4757 未合并的原因。**
   只能确认这 5 个 PR 的 `merged_at = null`（closed 但未合并）。
9. **没有找到 `vllm_ascend/torchair/` 被删除后，社区是否有人提出过恢复请求。**
10. **微信文章 <https://mp.weixin.qq.com/s/SZgGlvASIq0T6p_H_yK_xw> 未打开核对**，
    因此"npugraph_ex 带来 TPOT 5–8% 提升"这个数字**只有二手引用（RFC #4715 评论）**。

---

## 9. 一页总结

**问：为什么 vllm-ascend 没有使能 GE 整图？**

**答（分三层）**：

1. **它使能过。** v0.9.x–v0.11.0 期间，`torchair_graph_config.enabled=True` 默认就走
   TorchAir `max-autotune`（= GE / Ascend IR）路径，官方文档称其为 "the GE graph mode"，
   但只覆盖 DeepSeek 系列。**v0.12.0rc1（PR #4814，2025-12-10）把它移除了**，
   官方理由一句话：`aclgraph is stable and fast now`。

2. **移除的原因是"成本阻塞 + 组织原因"，不是"技术上做不到"。**
   - **硬阻塞只有一条真的**：GE 的动态分档不接受 `FRACTAL_NZ` 私有格式、不接受档位值 0/1（batch=1 无法符号化）、
     未命中档位直接报错；加上自定义算子必须补 Ascend Converter。**但"每档一张静态图"可以绕开前三条。**
   - **成本阻塞是主因**：编译时间以"数分钟到数十分钟"计；图缓存不稳定、默认每次启动都要重编；
     GE 路径需要一套平行的模型/算子/量化实现（`vllm_ascend/torchair/*`），而 ACLGraph 复用上游 vLLM 抽象。
   - **组织原因是决策的扳机**：上游 TorchAir 自己弃用 `reduce-overhead`、主推 npugraph_ex，
     而 vllm-ascend 的核心约束是跟住 vLLM 主线（main2main）。维护第二条图后端不划算。

3. **对照组不构成"能做"的证明。** MindIE 的整图是 **ATB 图**（手写 `build_graph`、
   为每个模型单独实现、算子集合是 ATB 而非任意 PyTorch），
   而且 **MindIE 自己也同时支持 ACLGraph**。
   更重要的是：**MindIE 处理"每步元数据"的方式（图输入张量 + host 侧 `runtime_param`）
   与 vllm-ascend 现有的 `update_full_graph_params()` 是同一类解法**
   —— 所以"每步元数据"从来不是障碍，**障碍是算子与模型生态不可照搬**。

**问：后续有使能计划吗？**

**答：未找到任何公开计划。** 自 2025-12-10 之后，vllm-ascend 仓库中与图模式相关的一切
（issue、PR、roadmap）都指向 **ACLGraph / npugraph_ex / XliteGraph**。
现行可查到的唯一下一步是 Q3 2026 roadmap 里的
`npugraph_ex support superkernel to reduce TPOT`（issue #15067，未完成）。
