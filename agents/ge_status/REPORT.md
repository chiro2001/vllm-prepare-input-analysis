# ge_status 调研报告（GE 整图为什么没有在 vllm-ascend 使能）

任务：回答"为什么 vllm-ascend 没有使能 GE 整图下发"与"后续是否有计划"。
方式：**纯源码 + 网络文献**；未登录远程主机、未运行 NPU、未动 chip3、未起容器、未修改任何被调研仓库。
日期：**2026-09-24**。

## 产出

| 文件 | 内容 |
|---|---|
| [docs/11-ge-whole-graph/02-why-not-enabled.md](/home/chiro/projects/vllm/preparing-input-phase/docs/11-ge-whole-graph/02-why-not-enabled.md) | 主文档（A–G + 阻塞点清单 + MindIE 对照 + 查不到的） |
| `docs/11-ge-whole-graph/00-source-evidence.md` | 由 root 撰写，本文不重复其内容 |
| 本文件 | 执行摘要 + 方法 + 覆盖度自评 |

---

## 1. 最重要的三个发现（对原问题前提的修正）

### 发现 1：**GE 整图曾经被使能过，不是"从来没做"**

v0.9.x – v0.11.0 期间，vllm-ascend 正式提供 `TorchAirGraph` 路径，官方文档原文
`TorchAirGraph: This is the GE graph mode. In v0.9.1rc1, only DeepSeek series models are supported.`
（该行在 PR #4814 中被删除，见其 file patch）。

**且它的默认配置就走 GE**，三环节证据链：
1. `torchair_graph_config.mode` 默认 `''`（v0.11.0rc3 `vllm_ascend/ascend_config.py:139-141`）
2. `if ...mode: config.mode = ...` —— 空串则不覆盖（v0.11.0rc3 `torchair/torchair_model_runner.py`）
3. TorchAir 文档：`mode="max-autotune"` 是 **GE 图模式**且是**系统默认模式**（`docs/zh/ascend_ir/quick_start.md`）

**移除**：PR [#4814](https://github.com/vllm-project/vllm-ascend/pull/4814) `Drop torchair`（2025-12-10 合并），
理由原话：`aclgraph is stable and fast now. Let's drop torchair graph mode now.`
release note 确认：v0.12.0rc1「Torchair graph mode is removed」。

**一条额外的组合信号**：以下 5 个 PR **全部 closed 但未合并（`merged_at = null`）**——
`#3780` / `#3783` / `#3788`（三者都是"给 GE 路径补 e2e 测试"）、
`#4756`（`Reduce the cost of torchair`）、`#4757`（`Support multi graphs for torchair`）。
而同期"移除 torchair"的 #4814 及其清理 PR #4856（2025-12-11）、#4927（2025-12-12）**全部合并**。
⇒ 移除前最后一段时间，实际推进方向已经是"拆"而不是"补"。

### 发现 2：**"每步元数据"不是障碍；障碍是算子与模型生态**

对照组 MindIE 的 ATBGraph 证明了静态整图可以处理每步变化：
`input_ids / position_ids / slots_mapping / seq_len / block_tables` 全部是**图的输入张量**，
每步由 host `self.graph_inputs[target_key].update({...})` 刷新，另有 `graph_param`（host 侧 tensor）。
**这与 vllm-ascend 现有的 `update_full_graph_params()` 是同一类解法。**

不可照搬的是：ATB 是**自有 C++ 算子库**、组图是**手写 `build_graph` + 显式声明 I/O**、
每个模型都有**独立实现**（`examples/atb_models/atb_llm/models/<model>/` + 官方迁移指南）。
而 vllm-ascend 跑的是任意 PyTorch 算子 + csrc + triton-ascend，靠 `torch.compile` trace。

### 发现 3：**上游 vLLM 不是"只做分段图"**

vLLM `CUDAGraphMode` 有 `FULL`（整图），且 `FULL_AND_PIECEWISE` 是默认模式。
上游 design doc 明确写 `Made full CUDA Graphs support orthogonal to compilation` /
`Full cudagraph capture without compilation`。
**上游"整图" = 整 forward 的 capture/replay，不是编译期 IR 下沉。**
vllm-ascend 的 `FULL`/`FULL_DECODE_ONLY` 就是 vLLM 语义下的"整图"。

---

## 2. 阻塞点判定（结论）

| 类别 | 内容 |
|---|---|
| **(a) 硬阻塞** | 仅 1 条真的：GE **动态分档**不接受 `FRACTAL_NZ` 私有格式、档位值不能含 0/1（batch=1 无法符号化）、未命中档位直接失败；自定义算子进 GE 必须补 **Ascend Converter**。**但"每档一张静态图"可绕开前三者**，所以这些是"分档路径"的硬阻塞，不是"整图"的硬阻塞。 |
| **(b) 成本阻塞（主因）** | 编译时间"数分钟到数十分钟"；**图缓存不稳定、默认每次启动删除重编**（v0.11 `platform.py` 原文注释）；算子长尾 Converter；收益侧无公开对照数据。 |
| **(c) 历史/组织原因（扳机）** | TorchAir × V1 scheduler 稳定性不足（#4581，以"torchair is dropped"关闭）；GE 与 ACLGraph **互斥**（#2154）；只覆盖 DeepSeek；**上游 TorchAir 自己弃用 reduce-overhead 转 npugraph_ex**；GE 路径需要**一整套平行模型/算子实现**，而 ACLGraph 复用上游 vLLM 抽象；vllm-ascend 的硬约束是跟住 vLLM 主线（main2main）。 |
| **(d) 已解决** | 动态 batch（capture size 分档 + padding）；attention 每步元数据（`update_full_graph_params` + `graph_task_update_*` + `ExternalEvent`）；stream 预算（FULL_AND_PIECEWISE + HDK 25.5.1+/CANN 8.5.0+，32K/64K 图）；多步 draft（merged graph #5553/#5940/#6860）；一部分融合收益（npugraph_ex + static kernel）。 |

**逐条核对用户点名的 6 个候选阻塞点**（详见主文档 §4）：

| 候选阻塞点 | 判定 | 关键证据 |
|---|---|---|
| 动态 batch/shape | **真阻塞**（对 GE 动态分档）；成本阻塞（对每档一图） | `set_dim_gears` 约束：档位不能含 0/1、≤100 档、不得用 `FRACTAL_NZ`、未命中即报错 |
| 图编译时间 | **成本阻塞**（有量化） | `graph_mode.md:184` "several minutes to tens of minutes"；GE 官方"编译阶段耗时较长"；issue #588 |
| 算子覆盖（FIA/MoE/csrc/Triton） | **成本阻塞**；Triton 接近硬阻塞 | TorchAir `custom_op_graph/overview.md`：max-autotune 必须实现 Ascend Converter；`compiler_interface.py` 记录 Triton kernel 句柄不可跨进程序列化 |
| 投机解码/MTP | **不是阻塞** | GE 时代 MTP 已跑通（#2145 等）；ACLGraph 有 merged graph（#5940/#6860） |
| 与 async scheduling 的兼容 | **部分真实，但非硬阻塞** | `ACL_Graph.md`「Replay ordering and synchronization」明确提到 async scheduling 下的排序风险；#4581 证明 TorchAir 在 V1 引擎下不稳定（但其报错是 OOM，不能过度解读）；原理上整图不破坏 `max_concurrent_batches` 批间流水 |
| 生态与维护（TorchAir × PyTorch、与 torch.compile 关系） | **真实，是移除主因** | TorchAir 随 torch_npu 发布、无独立包；上游弃用 reduce-overhead；两者是**同一个 torch.compile 管道的两个分叉**（可配置级切换，非架构级分裂） |

---

## 3. 公开计划

**明确结论：未找到任何"重新引入 GE 整图"的公开计划。**

自 2025-12-10（#4814 合并）之后，vllm-ascend 仓库中一切图模式相关的 issue/PR/roadmap
都只涉及 **ACLGraph / npugraph_ex / XliteGraph**。检索 `max-autotune` 命中 0 条 issue/PR。

现行可查到的下一步：vLLM Ascend Roadmap **Q3 2026**（issue [#15067](https://github.com/vllm-project/vllm-ascend/issues/15067)）
中的 `npugraph_ex support superkernel to reduce TPOT`（未完成）。
RFC [#4715](https://github.com/vllm-project/vllm-ascend/issues/4715)（npugraph_ex）**至今仍 open**。

---

## 4. 查不到的（关键，勿误读为"没有"）

1. **没有**任何一份完整解释"为什么刻意避免 Ascend IR 变换"的设计文档/RFC。
   最强证据只有两句话：PR #4814 的 `aclgraph is stable and fast now.` 与
   `compiler_interface.py:133` 的注释 `avoid fx graph to Ascend IR transformation`。
2. **没有** GE vs ACLGraph 的同模型同负载性能对照数据 → 无法量化"GE 整图能多省多少 host 开销"。
3. **没有**移除前对"档位数/编译时间/内存"的量化评估。
4. **未能证实** ATB 的 `GraphOperation` 字面上经过 GE 编译（ATB 用的是 `op->Execute(variantPack,...)`，
   不是 `ge::RunGraphWithStreamAsync`）。"MindIE 用了 GE 整图"需要独立证据。
5. **没有** TorchAir 对 Triton 算子入 GE 的官方说明（"Triton 难进 GE"是从 Ascend Converter 要求**推断**的）。
6. **没有** TorchAir/GE × async scheduling 的专门 issue。
7. **没有** #3780/#3783/#3788 未合并的原因（只能确认 `merged_at = null`）。
8. **没有** SGLang-ascend 的图模式资料（本次未检索该仓库）。
9. RFC #4715 引用的微信文章未打开核对，"TPOT 提升 5–8%"目前是二手引用。

---

## 5. 方法与覆盖度

### 5.1 做了什么

- **本地只读源码**：`HIST_PROJECT/vllm-ascend`（0.26.0rc1，commit `f2f74a16c`）、
  `preparing-input-phase/refs/vllm`、`HIST_PROJECT/.research/torchair-docs`（TorchAir 官方文档 sparse clone，
  HEAD `d29ef3c3f8`，2026-09-14）、`HIST_PROJECT/cannbot-skills`。
- **GitHub API**（`gh`，已登录）：vllm-ascend 的 PR/issue/commit/files 检索与正文抓取；
  vllm 上游的 search。
- **历史版本源码**：通过 `raw.githubusercontent.com` 读 v0.11.0rc3 / v0.10.2rc1 的
  `ascend_config.py`、`torchair_model_runner.py`、`platform.py`、`torchair/utils.py`
  （本地 clone 是浅克隆，只有 1 个 commit，历史必须走网络 —— 已按 root 的勘误执行）。
- **网络文献**：Exa 检索 + 华为昇腾官方文档（hiascend.com）、gitcode（torchair / ge / MindIE-LLM）、
  GitHub 镜像、华为云社区博客。

### 5.2 没有做（边界）

- 未在真实 NPU 上跑任何 GE / ACLGraph 对照实验（任务禁止）。
- 未打开微信文章核对二手数字。
- 未检索 SGLang-ascend 仓库。
- 未逐个统计 vllm-ascend 的算子中"缺 Ascend Converter"的具体数量。
- **本地 vllm-ascend clone 为浅克隆（1 commit），所有历史结论均来自网络 API，未用本地 git history。**

### 5.3 对用户原前提的两处修正（需回传）

1. 「vllm-ascend 源码里没有出现 torchair」→ **不准确**。
   `vllm_ascend/patch/worker/patch_npugraph_ex_triton.py:31` 有 `import torchair as nge`（npugraph_ex 的回退）。
   准确说法：**没有 GE / Ascend IR 的执行路径**。
2. 「为什么没有使能 GE 整图」→ **应改写为**「GE 整图曾经使能（v0.9.1–v0.11，仅 DeepSeek），
   为什么在 v0.12 被移除，以及有没有回归计划」。
