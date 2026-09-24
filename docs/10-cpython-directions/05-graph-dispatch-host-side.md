# 05 · host 侧元数据构造能否"图化"？—— 文献 + 本地源码调研

> **调研问题**：未来 TPOT 进一步降到 1 ms 时，decode 的**算子下发**已经从 eager 走向
> 图下发（CUDA Graph / ACLGraph 整图 replay）。那 `prepare_input` 阶段（本项目实测占单步
> **55%**、p50 **2 757.6 µs**）是否也可能出现类似的"图下发"技术？
>
> **一句话结论**：**能被图捕获的只有"设备任务"；host 侧的 Python 求值与元数据构造在结构上
> 无法被图捕获。** 但可以让它逼近"近图"形态 —— **固定地址的持久缓冲 + 参数就地更新 +
> 参数 tensor 化 + 把纯算术下沉成 device kernel**。在 Ascend 上，
> "图化能省 host"这一点**不成立**：官方文档明说 task update 比单独下发**更耗时**（§5.2-A）。
>
> **本文定位**：外部（官方文档 / 上游代码 / 上游 PR / issue）证据的调研与交叉核对，
> 不引入新的实测数字。本文所有"我们这边"的数字都直接引用本项目已交付文档
> （`docs/00-INDEX.md`、`docs/05-hotspots.md`、`docs/01-prepare-input-code-logic.md`、`docs/10-prepare-input-graphification.md`），
> 并标注口径。兄弟文档 `docs/10-prepare-input-graphification.md` 是同一问题的"实测 + 判断"版，两者互补：
> 该文给结论，本文给**来源、机制、能力边界、证据强度**。
>
> 写作日期 / 所有 URL 访问日期：**2026-09-24（Asia/Shanghai）**。
> 本地只读源码：`refs/vllm`（vLLM **0.26.0**，commit `568afb3a1`）、
> `/home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend`（**0.26.0rc1**，commit `f2f74a16c`）。

---

## 0. 结论速览（TL;DR）

1. **"图下发"和"元数据图化"是两个不同量级的问题。**
   图捕获的对象是**Stream 上的 device task**（ACL Graph 官方原话："所有在指定 Stream 上下发的
   任务不会立即执行，而是被暂存在模型的运行实例中…减少 Host 侧的任务下发开销"）。
   host 侧 Python 的求值发生在"下发之前"，**根本不在 Stream 上**，因此**没有**任何图捕获机制能
   把 `_build_attention_metadata` 这样的 Python 函数"捕获"进去。`[官方文档]`

2. **在 CUDA Graph 里，"只更新不重捕获"的合法范围很窄**：拓扑、节点类型、依赖顺序、memcpy 的
   字节数都不能变；可以变的是 kernel 的**参数值/指针/launch 配置**。主流实现
   （PyTorch `cudagraph_trees`、vLLM、FlashInfer、TensorRT-LLM）**几乎都不用** node-param update，
   而是统一走"**静态输入地址 + `copy_` 就地写**"这条路。`[官方文档][上游代码]`

3. **Ascend 已经有一套成熟的"图 + 每步参数更新"机制，而 vllm-ascend 就在用它**：
   `torch.npu.graph_task_group_begin/end`（捕获时打 task group 标记）+
   `torch.npu.graph_task_update_begin/end`（更新时**在同一 update stream 上把同一个算子再用新实参
   下发一次**）。`attention_v1.py:476/525/619/992` 是本项目已经核对过的调用点。`[上游代码]`

4. **关键问题的答案是「能传 list，但不是『覆盖 list』」**：更新阶段的语义是"**重新下发同一个算子**"，
   vllm-ascend 更新 FIA 时传的正是 Python list（`seq_lens_list`、`actual_seq_lengths_q`）；
   也就是说 **list 可以作为重下发的实参**，但**构造这份 list 的 host 成本一分钱都没省**。
   更要命的是华为官方文档对 `aclmdlRICaptureTaskUpdate*` 的定性结论：
   **"更新任务比单独下发任务更耗时"**。`[官方文档][上游代码]`

5. **"下沉到 device"才是真正能省 host 的那条路，而且 vLLM 已经在走**：
   `slot_mapping` 早已是 Triton kernel；spec-decode 的输入准备已有 5 个专用 kernel；
   **Model Runner V2** 更明确："MRV2 uses Triton kernels to prepare inputs such as `input_ids`,
   `positions`, `query_start_loc`, and `seq_lens`"，理由写着 "Lower CPU overhead: input prep is
   very cheap on GPU and avoids Python bottlenecks"。`[官方文档][上游代码]`

6. **上游已经有人把"prepare_input 的一块"塞进 decode cudagraph 并验证成功，但 e2e 收益为 0**：
   vLLM PR #47924（B.1）把 `compute_slot_mapping` 折进 FULL decode cudagraph，字节级一致、
   无回归，但 eager 路径省下的 ~130 µs/step **被 async scheduling 完全掩盖**；
   作者自己的结论是"**真正的奖金（~3 ms host Python）在 B.2：把 `_prepare_inputs` /
   `_build_attention_metadata` tensor 化**"。这条是本调研最重要的**反向证据**。`[上游 PR，草稿并已关闭]`

7. **Ascend 的 `enable_enpu` 与"图能力"无关**：`ENPU_ENABLE` 是 **vCANN-RT NPU 软切分虚拟化**
   （openEuler ubs-virt，时间片轮转）启动成功后设置的进程级环境变量；vllm-ascend 读它只是为了
   在虚拟化层下把"图参数更新"排到"模型前向"**之前**（"record first, wait later"，否则可能卡死）。
   `[上游代码][上游 PR][社区文档]`

8. **对 TPOT→1 ms 的判断**：把 `prepare_input` 从 2.76 ms 拉进预算，**没有任何单一技术做得到**；
   它需要"**去冗余（3× GDN → 1×）+ 静态缓冲/去每步分配 + 纯算术下沉 device + 剩余账本编译化/C++ 化**"
   四件套，而"图下发"本身只覆盖其中最小的那一块（下发/launch 开销）。§8 给必要条件清单。

---

## 1. 先厘清粒度：图捕获的是"设备任务"，不是"host 代码"

### 1.1 ACL Graph 的官方定义

- **来源**：CANN「ACL Graph 简介」<https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/acldevg/runtime_doc_dev_0045.html>（访问 2026-09-24）；
  「Building a Model Running Instance Based on the Capture Mode」<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/appdevg/acldevg/aclcppdevg_000519.html>（访问 2026-09-24）
- **机制（原文）**："在 AI 处理器上可以将相关任务下沉到 Device 上并执行，从而减少 Host 的开销…
  在 `aclmdlRICaptureBegin` 和 `aclmdlRICaptureEnd` 接口之间，**所有在指定 Stream 上下发的任务**
  不会立即执行，而是被暂存在模型的运行实例中，只有在调用 `aclmdlRIExecuteAsync` 接口执行模型时
  这些任务才会被真正执行。"
- **能力边界**：捕获单元 = **Stream task**。任何"还没变成 Stream task"的东西（Python 循环、
  numpy、dict 记账、`.tolist()`）都不在图里，**也不在重放时会重新执行**。
- **证据强度**：官方文档（强）。

### 1.2 一条把边界钉死的报错

- **来源**：vLLM issue #40807「[Bug]: TurboQuant KV + spec-decode + chunked-prefill crashes CUDA
  graph capture at `query_start_loc.tolist()`」<https://github.com/vllm-project/vllm/issues/40807>（2026-04-24，访问 2026-09-24）
- **机制**：在 CUDA graph capture 区间内执行 device→host 的 `.tolist()`，直接报
  `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`。
  该 issue 提出的两个修法方向之一就是"**在 metadata builder 里、进入任何被捕获区间之前，先把 CPU 侧
  `list[int]` 算好**"。
- **能力边界**：**host 需要读 device 值**（哪怕是 `query_start_loc` 这种极小的向量）这一动作
  与"捕获"天然互斥；host 侧 list 只能在图外构造。
- **证据强度**：上游 issue（报错原文，强）；修法方向为作者建议（中）。

### 1.3 由此得到本文的主判据

> **图化省的是"下发"，不省"构造"。**
> `prepare_input` 的成本主体是"构造"（本项目实测 topdown `frontend_bound 66.01%`、`IPC 0.771`、
> `aclrtSynchronize*` 全族 0.27% ⇒ 不是等卡、而是解释器/小对象执行）。
> 因此"给 prepare_input 找一种图下发技术"这个提法本身要拆成两件事：
> **(a) 把构造结果变成固定地址上的就地更新（近图化）；(b) 把构造过程本身搬去 device 或搬出 Python。**

---

## 2. 谱系 A：图捕获 + 参数就地更新

### A1. CUDA：`cudaGraphExecKernelNodeSetParams` / `cudaGraphExecUpdate` 的能力边界

- **来源**：NVIDIA CUDA C Programming Guide §4.2 "CUDA Graphs"
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>（访问 2026-09-24）；
  CUDA Runtime API「Graph Management」
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html>（访问 2026-09-24）
- **机制**：
  - **整图更新** `cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo)`：要求"更新图"与"被实例化图"
    **拓扑完全一致**（节点、依赖、依赖顺序、sink 节点顺序都要一致），只允许节点参数不同。
  - **单节点更新** `cudaGraphExecKernelNodeSetParams(...)`：官方明确"当需要更新的节点相对全图很少时更划算，
    因为它跳过未变节点的拓扑检查与比较"。
  - 官方给出的可更新节点类型表：kernel / memcpy / memset / host / child graph / event record / event wait /
    external semaphore（signal & wait）。`cudaGraphExec*NodeSetParamsFromSymbol` 之类还有额外限制。
- **能力边界（逐轴）**：
  - **拓扑 / 节点类型 / 依赖**：❌ 必须一致（"Major changes to graph structure (such as topology or node types)
    require re-instantiation"）。
  - **kernel 函数的约束**：设备（context）不能变；"原本不用 CDP 的节点不能更新成用 CDP 的函数"；
    "原本不做 device-side update 的节点不能更新成做 device-side update 的函数"；
    device-launch 图有额外限制。**即：能不能"换函数"取决于两边函数的属性，而不是随便换。**
  - **memcpy 的字节数**：❌ 不可变（字节数是实例化期确定的结构量）。
    —— CUDA C Programming Guide 正文对该条表述不完整；此处同时参考了 `cudaGraphExecKernelNodeSetParams`
    的 API 说明与第三方对 `cudaGraphExecUpdate` 契约的转述
    <https://docs.rs/oxicuda-graph/latest/src/oxicuda_graph/exec.rs.html>（访问 2026-09-24），
    **标注为"官方主文档 + 二次源交叉"，证据强度中。**
  - **kernel 参数值 / 指针 / launch 配置**：✅ 可更新（"All nodeParams fields may change"，但受上面 func 类约束）。
- **证据强度**：官方文档（强）；memcpy 字节数不可变一条（中）。
- **对我们的意义**：CUDA 侧确实存在"参数就地更新"这条路，但它解决的是**"参数从哪来"的下发环节**，
  与"参数在 host 上怎么算出来"无关。且下面 A2 会看到：**工业界很少用它**。

### A2. CUDA 生态的实际选择：静态输入 + `copy_`，而不是 node-param update

- **来源**：
  - vLLM `ACL_Graph.md`（vllm-ascend 设计文档，讲的是同一套语义在昇腾的落地）
    <https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html>（访问 2026-09-24）
    以及本地副本 `vllm-ascend/docs/source/developer_guide/Design_Documents/ACL_Graph.md`
  - torch_npu NPUGraph 开发指南（英文版）
    <https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/pytorch_npugraph_desc.md>（访问 2026-09-24）
  - FlashInfer API 文档（plan 与 CUDA Graph 的关系）
    <https://docs.flashinfer.ai/api/attention.html>（访问 2026-09-24）
- **机制（torch_npu 原文）**："Memory management during replay is the key point: **the memory addresses of
  tensors allocated during the capture phase remain unchanged during replay**. If you need to use new data
  between steps, you must write the new data to the memory addresses occupied during capture through `copy_()`.
  Do not directly reassign the tensor…"
- **能力边界**：
  - **地址必须固定**：捕获期分配的内存，重放时地址不变 ⇒ 每步变化的数据必须用 `copy_` 写进**同一个地址**。
    这条同时解释了为什么 `prepare_input` 里那些"每步 `torch.empty` / 每步新建 dataclass"是图化的**头号障碍**。
    —— 同一约束在本项目可核对的源码里也有独立表述：`vllm_ascend/compilation/acl_graph.py` 的
    `ACLGraphWrapper` 文档字符串写明"**tracing and checking the input addresses to be consistent during replay
    is guaranteed when `VLLM_LOGGING_LEVEL == DEBUG`**"（即：只做输入**地址一致性**检查，
    因为地址一变就必须重新捕获）。`[本地源码]`
  - **形状必须固定在捕获粒度上**：vLLM 的做法是"把运行时 batch 归到最近的 capture size 并 padding"
    （`docs/00-INDEX.md` §2 与 `docs/01-...md` §4.4 都记录了我们这条路径上的 FIA padding 逻辑）。
  - **`torch.compile`/`torch.cuda.graph` 的 `static input` 模式**（`make_graphed_callables`、
    `graph_pool_handle`）：语义同上 —— `graph_pool_handle` 只是"多张图共享同一内存池"的 token，
    **它本身不提供任何参数更新能力**；`make_graphed_callables` 的做法是：检测输入张量的 `data_ptr`
    与捕获时不同时，自动 `copy_` 进捕获期的那块地址。这正是"**参数不更新，只就地写**"的通用实现。
- **证据强度**：官方文档 + 上游设计文档（强）。

### A3. Ascend：ACLGraph 的三层机制（本项目已核对源码）

| 层 | API | 作用 | 本项目证据 |
|---|---|---|---|
| 1. 捕获 / 重放 | `aclmdlRICaptureBegin/End` → `aclmdlRIExecuteAsync`；torch_npu 侧 `torch.npu.NPUGraph` / `torch.npu.graph` | 把 Stream task 冻结成模型运行实例，之后一次调用执行整图 | §1.1 官方文档；`vllm_ascend/compilation/acl_graph.py`（`ACLGraphWrapper`） |
| 2. 任务组 | `aclmdlRICaptureTaskGrpBegin/End` ⇄ `torch.npu.graph_task_group_begin/end` | 给"将来要更新的那几个 task"打标记，返回 handle | `vllm_ascend/attention/attention_v1.py:992,1016,1121,1142,1195,1208` |
| 3. 任务更新 | `aclmdlRICaptureTaskUpdateBegin/End` ⇄ `torch.npu.graph_task_update_begin/end` | **在 update stream 上把同一批算子带新实参重新下发一遍** | `vllm_ascend/attention/attention_v1.py:525,538,619,640,809,842`；`attention/utils.py:88,101` |

**vllm-ascend 的实际用法（FULL 图 = 整模前向被捕获，attention 参数每步更新）**：

- 来源：本地 `vllm_ascend/compilation/acl_graph.py`（`update_full_graph_params` → `impl_cls.update_graph_params`）、
  `vllm_ascend/worker/model_runner_v1.py:2669-2725`（`_update_full_graph_params_if_needed` / `_model_forward`）、
  `vllm_ascend/attention/attention_v1.py:476-545`（paged attention 更新路径）、`611-640`（FIA 更新路径）；
  以及 vllm-ascend 设计文档 `ACL_Graph.md` 的「Host-side attention parameter update for full graph replay」一节。
- 机制要点：捕获期把每个 attention task 的 handle / event / workspace / **weak_ref** 存下来；
  每次 replay **之前**，在 **update stream** 上对这些 task 重新下发算子、写入新参数；
  用 `torch.npu.ExternalEvent` 保证 update 与 replay 的先后关系。
- **能力边界（vllm-ascend 设计文档原话）**："The important design point is that Ascend full graph support
  depends on backend-provided `update_graph_params()` hooks. **Without that hook, capture alone is not enough
  to replay the correct attention state.**" —— 即：**整图 replay 是"靠每步 host 侧重下发补丁"才成立的**。
- **证据强度**：上游设计文档 + 本地源码（强）。

### A4. 统一能力边界模型（五轴）

把上面三节的证据归一，可以得到一张"什么能只更新、什么必须重捕获"的判据表。
**这张表是本文最有复用价值的部分**，可直接用来判断任意一个 prepare_input 子步骤的图化可行性。

| 轴 | 能否"只更新/不重捕获" | 依据（简称） | 证据强度 |
|---|---|---|---|
| **拓扑 / 控制流**（op 序列、分支、循环次数） | ❌ 绝对不行 | CUDA：拓扑必须一致；CANN："`TaskGrp` 与 `TaskUpdate` 之间的**任务数量和类型必须相同**" | 官方文档（强） |
| **形状**（grid/block、tensor shape、batch size、memcpy 字节数） | ❌ 基本不行；业界统一用"padding 到离散 capture size"绕开 | CUDA：memcpy 字节数不可变；FlashInfer：`batch_size cannot change during the lifecycle of this wrapper when CUDAGraph is enabled`；vLLM：batch→capture size 分桶 + padding | 官方文档 + 上游代码（强） |
| **参数值**（kernel 标量/指针、block table 地址、seq len 值） | ✅ 可以，但有三种价位↓ | — | — |
| ├ 价位 1：CUDA node-param update | ✅ 直接改 graphExec 节点参数 | `cudaGraphExecKernelNodeSetParams` / `cudaGraphExecUpdate` | 官方文档（强） |
| ├ 价位 2：CANN `graph_task_update` | ✅ 但**代价 > 普通下发** | CANN 原文："**更新任务比单独下发任务更耗时**"；且"task group 类似临界资源，不支持多线程/多流并发更新" | 官方文档（强） |
| └ 价位 3：主流 —— 不 update，直接写固定地址（`copy_` / in-place / 参数 tensor 化） | ✅ 且最省 | torch_npu/vLLM/FlashInfer/TRT-LLM 一致做法 | 官方文档 + 上游代码（强） |
| **host 读 device 值**（`.tolist()` / `.item()` / `.cpu()`） | ❌ 捕获区内**非法** | vLLM #40807 报错原文 | 上游 issue（强） |
| **host 侧 Python 对象**（dict/list/dataclass、`np.repeat`、逐 req 循环） | ❌ 连"图外"都进不了图；只能留 host 或改写 | §1.1 ACL Graph 定义 + §B3 | 官方文档（强） |

### A5. 关键问题：host 侧 Python 构造的 list/tuple 参数，能否被 `graph_task_update` 覆盖？

**结论：能"当作重下发的实参传进去"，但不是"在 device 上就地覆盖"，更不能省掉构造 list 的 host 成本。**

证据链（三条，互相独立）：

1. **CANN 官方语义**：任务更新 = "在 `aclmdlRICaptureTaskUpdateBegin/End` 之间**更新任务的输入信息**"，
   并附三条约束：① 任务数量与类型必须和捕获时一致；② 跨流捕获场景下不能在更新区间向处于捕获态的
   其他流下发任务；③ **task group 不支持多线程/多流并发更新，否则结果不可预期**。
   来源：「Building a Model Running Instance Based on the Capture Mode」
   <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/appdevg/acldevg/aclcppdevg_000519.html>（访问 2026-09-24），
   以及同一接口在 CANN runtime 开源仓库中的说明（`aclmdlRICaptureBegin` / TaskGrp 一节）
   <https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclmdlRICaptureBegin.md>（访问 2026-09-24）。
2. **torch_npu 文档给的更新写法**：更新区间内**把同一个算子再调一遍**，且示例里直接传
   `actual_seq_lengths=length_new`（Python list）。
   来源：`torch_npu.npu.graph_task_update_begin` 文档
   <https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/torch_npu-npu/torch_npu-npu-graph_task_update_begin.md>（访问 2026-09-24）；
   torch_npu 实现 `torch_npu/npu/graphs.py`（`_GraphDispatchMode.update_capture_record` 里
   `graph_dispatch_record.op_cache_entry(*args, **kwargs)`）
   <https://gitcode.com/Ascend/pytorch/blob/df242506eb509201b170357330b48c649f38ef9b/torch_npu/npu/graphs.py>（访问 2026-09-24）。
3. **vllm-ascend 正在这么做**：FIA 的更新路径把
   `seq_lens = attn_metadata[metadata_key].seq_lens_list`（**Python list**）与
   `actual_seq_lengths_q`（**Python list**）作为 `actual_seq_qlen` / `actual_seq_kvlen` 传给
   `torch_npu.npu_fused_infer_attention_score_v2.out(...)`（`attention_v1.py:611-640`）；
   而在捕获路径 `full_graph_fia_v2` 里，kv 长度已经改用 **tensor**（`actual_seq_lengths_kv = attn_metadata.seq_lens`，
   `attention_v1.py:1024-1060`），只有 q 长度仍是 list。
   ⇒ **"FIA 必须吃 list"不是硬约束**（tensor 已被证实可用），但**用 list 也不会让图更新失败**。

**但要小心三个坑（这条是给 GDN builder 判定的直接输入）**：

- **更新 ≠ 免费**：CANN 明说更新比单独下发**更耗时**（§A4 价位 2）。所以"把 list 交给 update 机制"
  在 host 成本上是**负收益**，它的价值是"让整图 replay 正确"，不是"省 host"。
- **更新 ≠ 可变形**：任务数量/类型必须一致 ⇒ **不能靠 update 表达"这一步 3 个 GDN 调用、下一步 2 个"**，
  分支必须自己做 padding/mask 抹平（这一点 vllm-ascend 事实上就是这么做的，
  `_reset_spec_decode_graph_inputs` 把不需要的 spec 分支"清零成 no-op"，见 `gdn_attn_builder.py:371-376`）。
- **update 区间内仍然要构造 list**：如果 list 依赖 device 值（如 `seq_lens` 的精确值），
  就必然引入一次 device→host 读值 ⇒ 与捕获语义冲突（§1.2）。
  vllm-ascend 规避它的办法是**在 host 上维持一份 CPU 镜像**（`seq_lens_cpu` /
  `num_computed_tokens_cpu`），代价是 host 侧要维护这套账本 —— 这正是 `prepare_input` 的一半成本来源。

---

## 3. 谱系 B：把 host 侧循环下沉到 device

### B1. vLLM 里"已经下沉"的清单（可直接引用的先例）

| 已下沉的子步骤 | 位置 | 形态 | 备注 |
|---|---|---|---|
| `slot_mapping` 计算 | `refs/vllm/vllm/v1/worker/block_table.py:166`（`_compute_slot_mapping_kernel`，`@triton.jit(do_not_specialize=["num_tokens","max_num_tokens"])`，末位 program 专门写 padding，"Pad remaining slots for CUDA graph compatibility"）；vllm-ascend 同款 `vllm_ascend/worker/block_table.py` | device kernel，grid=`num_reqs+1` | **本来就是 device 侧**，host 只剩 launch |
| spec-decode：`token_indices_to_sample` / `num_rejected_tokens` | `refs/vllm/vllm/v1/spec_decode/utils.py:136` `eagle_prepare_inputs_padded_kernel` | fused Triton kernel | 注释明说 "Fused kernel…"，即刻意合并 launch |
| spec-decode：`next_token_ids` / `valid_sampled_tokens_count` | 同上 `:178` `eagle_prepare_next_token_padded_kernel` | fused Triton kernel | 枚举-扫描式实现在 device 上完成 |
| spec-decode：draft 输入的 copy+expand | 同上 `:307` / `:457`（Eagle / DFlash） | device kernel | |
| spec-decode：step 级 slot mapping + metadata | 同上 `:28` `eagle_step_slot_mapping_metadata_kernel` | device kernel | |
| **V2 runner 的整套输入准备** | 调用点 `refs/vllm/vllm/v1/worker/gpu/model_runner.py:874`（`prepare_inputs`）；kernel 实现 `refs/vllm/vllm/v1/worker/gpu/input_batch.py:186/222/246/282/304/364`（`_prepare_prefill_inputs_kernel` / `prepare_prefill_inputs` / `_prepare_pos_seq_lens_kernel` / `prepare_pos_seq_lens` / `_combine_sampled_and_draft_tokens_kernel` / `combine_sampled_and_draft_tokens`）与 `:612`（`expand_idx_mapping`） | Triton kernel，写入 `InputBuffers` 持久缓冲 | 见 B1-b（本地 0.26.0 快照已核对） |
| GDN 的 decode 图输入 | `vllm_ascend/ops/gdn_attn_builder.py:332-376`（`_pad_non_spec_decode_graph_inputs` / `_reset_spec_decode_graph_inputs`） | **不是新 kernel，而是"往预分配缓冲里 in-place 写"** | 我们认为这是 GDN 侧最接近"可图化"的既有形态 |

**B1-b. 上游自己怎么描述 V2 的下沉动机**（重要，直接回答了"哪些做不到/为什么"）：

- 来源：vLLM 官方文档「Model Runner V2 Design Document」
  <https://docs.vllm.ai/en/v0.19.1/design/model_runner_v2/>（访问 2026-09-24）
- 原文要点：
  - "V1 introduced **persistent batches** to minimize CPU overhead during input preparation… Building these
    tensors from scratch each step is often **very slow in Python**, especially for large tensors like block tables."
  - "MRV2 **decouples persistent state tensors from per-step input tensors**… Large state tensors are mostly
    stored on **GPU memory, so gather runs in parallel on the GPU with low overhead**."
  - "vLLM now relies heavily on **asynchronous scheduling**. The scheduler and worker prepare inputs for step
    `N+1` while the GPU executes step `N`, overlapping CPU and GPU work."
  - "MRV2 instead assumes the core model execution loop is a **CUDA stream with no CPU synchronization points**.
    CPU entrypoints queue work onto the stream. Both explicit sync… and implicit sync (…) must be avoided."
  - "**MRV2 uses Triton kernels to prepare inputs such as `input_ids`, `positions`, `query_start_loc`, and
    `seq_lens`.** 1. Better async behavior: GPU can derive values (for example with speculative decoding)
    that CPU may not know yet. **2. Lower CPU overhead: input prep is very cheap on GPU and avoids Python bottlenecks.**"
- **能力边界（同文档）**：MRV2 的 CUDA graph 走**独立的 `CUDAGraphManager`**，"capture uses a separate
  dedicated path"；即：**"输入准备"与"图捕获"是两条并行的工程线，输入准备不是被捕获进来的**。
- **证据强度**：官方文档（强）。
- **本地交叉验证**：`refs/vllm/vllm/v1/worker/gpu/README.md` 只有一句 "[Experimental] Model Runner V2…
  Ping Woosuk Kwon"，但目录里 `async_utils.py / attn_utils.py / block_table.py / buffer_utils.py /
  cudagraph_utils.py` 的存在与文档描述一致（`refs/vllm` 为 0.26.0 快照）。

> **对本项目的直接启示**：vLLM 上游对"prepare_input 太重"给出的官方解法是
> **（1）持久状态与每步输入解耦 →（2）device 侧 gather/kernel →（3）async scheduling 重叠 →（4）显式 CUDAGraph 管理**，
> 而**不是**"把 metadata builder 塞进图"。我们的 GDN builder 正处在 V1 架构里，
> 所以它的 3×303 µs 是"V1 账本 + Python 构造"的典型症状。

### B2. FlashInfer 的教训：什么决定了"元数据构造能不能进图"

- **来源**：FlashInfer API 文档 <https://docs.flashinfer.ai/api/attention.html>（访问 2026-09-24）；
  issue #187「Make flashinfer kernels cuda graphs friendly」
  <https://github.com/flashinfer-ai/flashinfer/issues/187>（2024-03-20，访问 2026-09-24）
- **机制 / 边界（原文）**：
  - "The `plan()` method **cannot be used in Cuda Graph or in torch.compile**."（host 侧规划：按输入数据决定
    block size / num_q_tiles / 是否 split-kv）
  - 因此专门提供了 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`（"first proposed in vLLM"）：
    **预分配** `indptr_buffer / indices_buffer / last_page_len_buffer`，并把
    "`batch_size` cannot change during the lifecycle of this wrapper when CUDAGraph is enabled"、
    `q_len_per_req` 也进入"冻结形状"。
  - 为了解决"grid 大小随输入变化"，社区方案是：**grid 固定为 SM 数量的上界 + 用 `block_valid_mask`
    让多余线程块跳过计算**（即"把动态性从 host 侧挪到 device 侧的 mask"）。
  - 另一个更细的技巧：**把捕获期会被冻结的 kernel 实参，改成"指向一个 global memory 地址的指针"**，
    运行期改那块显存的值（"we pass a pointer to a global memory address that stores this value instead"）。
- **能力边界总结**：**凡是"host 根据输入数据做的规划/调度决策"（tile 划分、split 选择、grid 大小），
  都不能进图**；能进图的只有"固定形状 + 固定网格 + device 侧可读参数"的组合。
  这与 §A4 五轴模型完全一致。
- **证据强度**：官方 API 文档（强）+ 上游 issue 讨论（中）。

### B3. 下沉的硬边界（三类"做不到"及原因）

依据 §B1–B2 与本项目源码，`prepare_input` 里能下沉的东西只有一类：**"device 值 → device 值"的纯算术**。
剩下的三类做不到：

| 类别 | 为什么做不到 | 本项目 / 上游例子 |
|---|---|---|
| **需要 host 决策** | 调度决策（谁进 batch、各排多少 token、谁被抢断、块怎么分配）产生于 host scheduler，device 侧没有这个输入 | `_update_states` 的 dict / condense / swap 记账（`docs/01-...md` §8）；vLLM #47924 也说 `_prepare_inputs` 是 eager host 路径 |
| **需要 host 读值**（`.tolist()` / `.item()` / `.cpu()`） | "把 device 值搬回 host"必须有同步；在捕获区间内直接非法 | vLLM #40807（`.tolist()` crash）；`attention_v1.py:332-333`（`query_start_loc_cpu[1:].tolist()` / `seq_lens.tolist()`）；vLLM #32815 反过来把 `.cpu()` 干掉 |
| **需要 Python 对象语义**（dict 查找、dataclass 构造、`list.append`、逐 req for 循环） | 这些不是 Stream task；图里没有对应表示 | `gdn_attn_builder.py:531` 的 `build`；`ops/triton/fla/utils.py:22-37` 的 `prepare_chunk_indices`（**host 上**的 `torch.cat([torch.arange(n) for n in ....tolist()])` —— Python 逐请求循环 + host torch） |

### B4. 反向流动：有些元数据反而应当"留在 host"

不是所有下沉都划算，上游有明确的反向案例：

- **来源**：vLLM PR #32815「[Feature]: Remove DtoH Copy for lfm2_vl On Default Stream」
  <https://github.com/vllm-project/vllm/pull/32815>（访问 2026-09-24）
- **机制**：为了消掉 preprocess 阶段的 D2H，作者**把 ShortConv/Mamba 的 metadata 路径改成使用
  **CPU 侧的 `query_start_loc_cpu`**，"avoiding `.cpu()` conversions in the builder"；
  并把 vision attention 的 `max_seqlen` 保持在 host，避免 `.item()` 触发的隐式同步。
- **能力边界 / 启示**：**"下沉"和"上浮"都要看哪一侧真的需要这个值。**
  如果 host 本来就要用（或正好有一份 CPU 镜像），把它搬到 device 再搬回来是纯亏。
  这条对 GDN builder 的判定很关键：**不要假设"device 化一定更快"**，
  只有当 (a) 该值 host 不再需要 + (b) 该值是 device 原生可得 时，下沉才有正收益。
- **证据强度**：上游已合并 PR（中—强）。

---

## 4. 谱系 C：其他引擎如何把 CPU 移出关键路径

### C1. TensorRT-LLM：把每步账本搬进 C++，把 schedule 与 forward 重叠

- **来源**：
  - 「C++ GPT Runtime」<https://nvidia.github.io/TensorRT-LLM/advanced/gpt-runtime.html>（访问 2026-09-24）
  - 「Architecture Overview」<https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html>（访问 2026-09-24）
  - 「Torch Compile & Piecewise CUDA Graph」
    <https://nvidia.github.io/TensorRT-LLM/1.2.0rc8/features/torch_compile_and_piecewise_cuda_graph.html>（访问 2026-09-24）
- **机制（逐条）**：
  1. **hot loop 在 C++ 里**：`PyExecutor` 只做进程/请求编排，真正的
     `Scheduler` / `KVCacheManager` / `ModelEngine` / `Sampler` / `GptDecoder` 都是 C++ 对象；
     "per-step 元数据准备"因此是**编译后的 C++**，而不是 CPython 解释执行。
     ⇒ 这是"把 `prepare_input` 从解释器里搬走"的**架构级答案**。
  2. **Overlap Scheduler**（官方默认开启）：先为 step `n` 下发 GPU 工作、异步采样，然后再处理 step `n-1` 的结果
     （伪代码：`_schedule()` → `_forward_step()` → `_sample_async()` → `_process_previous_batch()`）。
     ⇒ 这是"**把 host 从关键路径上挪开**"的答案（多一个解码步的流水线深度换吞吐）。
  3. **CUDA Graph padding**：batch size 不匹配已有图时**向上 padding 到最近的已捕获尺寸**；
     官方给的效果是"certain models and hardware 上 e2e 吞吐最高 +22%"。
  4. **Piecewise CUDA Graph 的边界与残留问题（重要负面证据）**：
     - "**Even with Piecewise CUDA Graph enabled, you may still observe bubbles in the context (prefill) phase,
       primarily due to the attention operator's substantial host-side overhead.**"
     - 三条硬约束："Attention MUST NOT have any output. The output tensor should be allocated by CUDA Graph."、
       "Each sub-cudagraph MUST have at least one input tensor that contains the number of tokens in the shape."、
       "Only allow dynamic shape for `num_of_tokens` dim."
     - 结论：**连 NVIDIA 自己的栈也没有"把 attention 的 host 侧开销图化掉"**，只能把它隔离在图外（piecewise）。
- **对本文问题的意义**：TRT-LLM 的答案是"**换语言/换架构 + 重叠**"，不是"图化元数据"。
- **证据强度**：官方文档（强）。

### C2. SGLang：zero-overhead scheduler（重叠），但有明确失败案例

- **来源**：
  - SGLang v0.4 发布博客「Zero-Overhead Batch Scheduler…」
    <https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/>（2024-12-04，访问 2026-09-24）
  - 社区解读（中文，讲透了两线程/两队列结构）
    <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md>（访问 2026-09-24）
  - SGLang issue #19347「Zero-Overhead Batch Scheduler fail to overlap CPU scheduling with GPU computation
    as claimed」<https://github.com/sgl-project/sglang/issues/19347>（2026-02-25，访问 2026-09-24）
  - SGLang issue #27186「[RFE] Re-introduce tunable forward-pipeline depth in overlap scheduler」
    <https://github.com/sgl-project/sglang/issues/27186>（访问 2026-09-24）
  - SGLang 代码 `python/sglang/srt/managers/overlap_utils.py`（`FutureMap`、`needs_cpu_seq_lens`）
    <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/overlap_utils.py>（访问 2026-09-24）
- **机制**：调度器**提前一拍**准备 metadata（CPU-S），把 batch 交给专门做 launch/结果处理的 CPU-L
  （`TpModelWorkerClient` + `input_queue`/`output_queue`），用 CUDA event 串起依赖；
  对 grammar 这类 CPU 密集步骤，直接在上一拍的 `process_batch_result` 里预先把**下一拍**的
  `regex_vocab_mask` 算好（配 dummy batch 触发首拍）。
  `overlap_utils.py` 里还有一个值得借鉴的细节：**每个 backend 用 `needs_cpu_seq_lens` 声明是否需要
  CPU 侧 seq_lens**，只有需要时才做 D2H（用私有 stream + pinned buffer）。
- **能力边界（负面证据）**：
  - issue #19347：实测"**没有观察到 CPU 调度与 GPU 执行的重叠**"（与官方宣称不符；讨论中要求提供更详细的 profile）。
  - issue #27186：移除 overlap 线程后流水线深度被硬编码为 1，
    `pystack` 显示 **48.3% 的时间花在 `torch.cuda.Stream.synchronize`**上；长上下文 VLM 场景每引擎
    生成速率下降 25–30%。⇒ **"重叠"不是免费的，它依赖流水线深度与每步同步点的位置**。
  - SGLang PR #16194（scheduler overlap）自述："does **not** update `seq_lens_cpu` within the Draft model
    because of **host-to-device bound**. This may cause issues if the model relies on this attribute; therefore,
    this optimization is only applicable to models that do not depend on `seq_lens_cpu`."
    —— 与 §B3"需要 host 读值"是同一个边界。
- **证据强度**：官方博客 + 上游代码（强）；issue 中的实测质疑（中，属"社区证据"）。

### C3. NVIDIA Dynamo：它压根不碰 per-step 元数据

- **来源**：「Overall Architecture」<https://docs.nvidia.com/dynamo/dev/design-docs/overall-architecture>（访问 2026-09-24）；
  「Router Design」<https://docs.nvidia.com/dynamo/v1.4.0/knowledge-base/modular-components/router/router-design>（访问 2026-09-24）；
  「Architecture Flow」<https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/architecture-flow>（访问 2026-09-24）
- **机制**：Dynamo 是**分布式编排层**（Frontend / KV-aware Router / Planner / KVBM / NIXL 传输），
  把"请求路由、prefill/decode 分离、KV 生命周期与跨节点搬运"从引擎里拿出来放到独立进程
  （Router 的块索引是 Rust 实现：RadixTree/ConcurrentRadixTree、事件平面 NATS/ZMQ）。
  **per-step 的 input 准备仍然由后端引擎（vLLM / SGLang / TRT-LLM）负责。**
- **能力边界**：对 `prepare_input` 这个具体瓶颈**没有直接帮助**；它降低的是"端到端编排/路由/传输"的 CPU 占用，
  而且引入了一跳。真正相关的只有一条间接经验：**"把与控制流无关的重活挪出关键进程"**。
- **证据强度**：官方文档（强）；"不涉及引擎内 per-step 元数据"是**基于文档内容的推断**，标注为 `[推断]`。

### C4. vLLM 自己：async scheduling / `max_concurrent_batches=2` / DBO / MRV2

- **来源（本地源码，0.26.0）**：
  - `refs/vllm/vllm/config/scheduler.py:158-166`：`async_scheduling: bool | None = None`，
    注释："Async scheduling helps to avoid gaps in GPU utilization, leading to better latency and throughput."
    `get_scheduler_cls()` 在开启时返回 `AsyncScheduler`。
  - `refs/vllm/vllm/config/vllm.py:512-521`：`max_concurrent_batches` —— "**Async scheduling requires
    2 concurrent batches to overlap**"；V1 runner 且 `pp_size<=1` 时返回 **2**；V2 runner 返回 `pp_size+1`。
  - `refs/vllm/vllm/v1/worker/gpu_model_runner.py:541`（`self.use_async_scheduling = ...`）、
    `:1956`（`_prepare_inputs` 里注释 "OPTIMIZATION: Start copying the block table first. This way, we can
    overlap the copy with the following CPU operations."）、`:2073-2095`（async 下的
    `num_accepted_tokens_event.synchronize()`）
  - `docs/01-prepare-input-code-logic.md` §4.3：`synchronize_input_prep()` 只在 `use_async_scheduling` 时生效，
    用 `prepare_inputs_event`（blocking）保护 pinned 缓冲；**ascend 的 record 点在 `_update_states` 之后**。
- **机制**：**把"第 N+1 拍的输入准备"与"第 N 拍的 device 执行"重叠**（双 batch 流水线），
  这是"host 侧耗时 > device 侧耗时"时唯一能提高 GPU 利用率的手段。
- **能力边界 / 负面证据**：
  - vLLM RFC #20727（Async scheduler and Multi-step in v1）
    <https://github.com/vllm-project/vllm/issues/20727>（访问 2026-09-24）里有一条评论级证据：
    "Based on my own experience implementing asynchronous scheduling, this bubble is **almost negligible
    (14ms vs 0.3ms)**, meaning that **asynchronous scheduling is almost useless for decoding models**"。
    —— 这是社区口径，**只适用于"device 远大于 host"的场景**（14 ms/step）；与我们低并发 decode
    （单步 4.9 ms、prepare 2.76 ms）恰好相反，**不能直接引用**，但值得写进"条件依赖"里。
  - vLLM PR #17866「[V1] Fast decode prepare path for `prepare_inputs` logic」被拒，
    WoosukKwon 原话："I really don't want this optimization. We can optimize `prepare_inputs` in different ways"；
    评论区同时给出两条关键观察：① "sglang is doing **async CPU prepares**"；
    ② "it is not easy to optimize the existing prepare_inputs, since it is doing a good use of numpy logic…
    however, it **does not avoid CPU=>GPU transfers**, no matter what you do, unless you avoid them or do it
    async with the GPU somehow"；③ "vllm's cpu overhead becomes more bigger when batch size is more bigger.
    I test **14% cpu overhead when batch size = 128**"。
    <https://github.com/vllm-project/vllm/pull/17866>（访问 2026-09-24）
  - **DBO / ubatching**（Dual Batch Overlap，`--enable-dbo`，仅 DP+EP + DeepEP）：
    把一个 step 的 batch 劈成两个 micro-batch，用两个线程 + 两条流 ping-pong 重叠 **MoE all-to-all** 与计算。
    官方设计文档 <https://docs.vllm.ai/en/latest/design/dbo/>（访问 2026-09-24）；
    引入 PR #20448 / #24845（<https://github.com/vllm-project/vllm/pull/23693>）。
    **注意**：DBO 的目标是**通信重叠**，不是元数据；且上游反馈过"单机小模型上反而变慢"的案例
    （"I observed a negative performance gain, with the per-step latency increasing from 38ms to 49ms"）
    —— 因为它会让权重被读两遍。`[上游 PR 评论，中]`
  - **V2 runner 的 DBO RFC #50738**（<https://github.com/vllm-project/vllm/issues/50738>）
    给出了一个对我们非常有用的**结构性论断**：
    "**Persistent per-microbatch buffers, sliced *before* metadata is built**… Slicing a microbatch's `InputBatch`
    writes into these fixed buffers via `torch.sub(...).clamp_()` in place, then attention metadata is built
    *from* the sliced batch — not sliced *out of* pre-built metadata… buffer addresses are **stable across steps**,
    **which is what CUDA graph replay requires**."
    ⇒ 上游已经把"元数据路径适配图重放"的正确形态写清楚了：**先固定缓冲，再从缓冲建元数据**，
    而不是"先建元数据对象、再想办法塞进图"。

---

## 5. Ascend 专项

### 5.1 机制清单（一句话一个）

| 机制 | 语义 | 与"元数据图化"的关系 |
|---|---|---|
| `aclmdlRICaptureBegin/End` + `aclmdlRIExecuteAsync` | 捕获 Stream task 到模型运行实例，之后异步执行整图 | 只覆盖 device task；**不看 host 代码** |
| `aclmdlRICaptureTaskGrpBegin/End` + `aclmdlRICaptureTaskUpdateBegin/End` | 给少数单算子 task 打组，并在运行期"**更新任务的输入信息**" | 官方定位："**适用于少量单算子调用任务需要更新的场景**"；并明说**更新比单独下发更耗时** |
| `torch.npu.NPUGraph` / `torch.npu.graph` / `make_graphed_callables` | torch_npu 的三档封装；`make_graphed_callables` 自动处理 `copy_` 式输入更新 | "**Only aclnn operators are supported for graph capture**" |
| NPUGraph 的 `auto_dispatch_capture=True` + `_npugraph_handlers` | 对"需要动态参数的算子"（官方举例 sequence length / FlashAttention），由**算子 handler** 在 replay 前刷新参数 | 这是昇腾侧"参数就地更新"的正式入口；**只有注册过 handler 的算子能这么用** |
| `torch.npu.graph_task_*`（vllm-ascend 用的就是这套） | 上表第 2 行的 torch_npu 封装 | 更新 = 在 update stream 上**重下发算子**；约束：**update stream 必须与 capture stream 不同** |
| Npugraph_ex（TorchAir） | **编译期 FX 图优化**（in-place 化、算子融合、可选 static kernel 编译），是 ACLGraph 的加速层 | 它优化的是**模型 FX 图**，不是 host 侧 prepare 代码 ⇒ 对 `_build_attention_metadata` 无直接作用 |
| XliteGraph | 面向特定模型族的另一条图路径（可选安装） | 同上，模型级 |

来源：CANN/昇腾社区 + vllm-ascend 本地文档
（`docs/source/user_guide/feature_guide/graph_mode.md`、`docs/source/developer_guide/Design_Documents/ACL_Graph.md`、
`npugraph_ex.md`），以及 torch_npu 开发指南
<https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/pytorch_npugraph_desc.md>（访问 2026-09-24）。
`[官方文档 + 本地源码]`

### 5.2 官方约束逐条（含两条**很少被引用的硬约束**）

**A. 关于"图更新"的代价与限制**（来自 CANN「Building a Model Running Instance Based on the Capture Mode」）

1. **"updating tasks is more time-consuming than delivering tasks separately"**（更新任务比单独下发更耗时）。
   —— 这一条直接否定了"用图更新来省 host"的设想；它的存在意义是**正确性**（让整图 replay 拿到新参数），
   不是性能。
2. "The **number and types of tasks** between `TaskGrpBegin/End` and `TaskUpdateBegin/End` must be the same."
   —— 不能靠 update 改算子序列/数量（⇒ 分支必须 padding/mask 抹平）。
3. "A task group is similar to a **critical resource** and does **not support concurrent update** of multiple
   threads and streams."
4. 想和模型实例里的其他任务并发，需要**外部事件**（`aclrtCreateEventWithFlag(..., ACL_EVENT_EXTERNAL)`）+
   专门的 `UpdateStream`；且外部事件"规格有限，需要合理复用"。

**B. 关于"捕获本身"的限制**（来自「Single-Stream Capture」CANN 9.0.X / 8.5.0）

5. 捕获期间**流/事件/设备/上下文的查询与同步全部失效**（任何捕获模式）。
6. **默认流上的操作失效**；捕获用的 Begin/End 必须是同一个主流；跨流捕获必须靠 `RecordEvent` + `StreamWaitEvent`
   把其他流"接进/接回"主流，否则捕获结束时报错。
7. 捕获期间**任务用到的 device 内存必须保持不变**（资源只能在模型不再使用后销毁）。
8. `ACL_MODEL_RI_CAPTURE_MODE_GLOBAL` 下 `aclrtMemset/aclrtMemcpy/aclrtMemcpy2d` 这些**非安全函数失效**；
   需要时用 `aclmdlRICaptureThreadExchangeMode` 临时切到 `RELAXED`。
9. 被捕获的异步拷贝若涉及 host 内存，**必须用 acl API 申请的页锁定内存**（否则捕获报错）。
10. **捕获会消耗 Stream 资源**："随着任务数量的增加…更多的 Stream 进入捕获状态，Stream 资源被不断消耗，
    最终可能会导致并发的调度资源不足，因此需提前规划好调度资源的使用。"
    —— vllm-ascend 的 `ACL_Graph.md` 也说 PIECEWISE 捕获最敏感、捕获尺寸覆盖过广会失败。

来源：<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0030.html>、
<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/appdevg/acldevg/aclcppdevg_000519.html>、
<https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclmdlRICaptureBegin.md>（均访问 2026-09-24）。
`[官方文档，强]`

**C. 对本项目的直接结论**：
约束 1 说明"把 `_build_attention_metadata` 藏进 `graph_task_update`"**不可能省钱**；
约束 2/10 说明"每步 op 数量变化的路径"（例如 spec / 非 spec 混合、GDN 的 3 个 KV group 之一为空）
**必须用 padding/mask 抹平成固定 task 序列**；
约束 7 说明"每步 `torch.empty` 新建 metadata tensor"在捕获语义下**根本不可用**。

### 5.3 `ENPU` / `enable_enpu` 到底是什么（本项目代码里能看到，但公开资料很少）

- **它是什么**：`ENPU` = **Enhanced NPU**，即 **vCANN-RT 的 NPU 软切分虚拟化**。
  openEuler `ubs-virt` 的 `vcann-rt` 文档写得很直白：
  "推理任务启动时，会自动拉起 vCANN-RT 服务进行算力控制和显存控制，**软切分服务启动成功之后会设置一个
  进程级环境变量 `ENPU_ENABLE=True`**"，其机制是 `libvruntime.so` 通过 `ld.so.preload` 拦截昇腾 runtime 调用，
  **按时间片（默认 100 ms）轮转算力**。
  来源：<https://gitcode.com/openeuler/ubs-virt/blob/master/ubs-virt-enpu/vcann-rt/README.md>、
  <https://www.hiascend.com/developer/techArticles/20260310-1>（访问 2026-09-24）。
- **vllm-ascend 为什么读它**：PR #8456「[BugFix]: order acl graph updates before model forward for ENPU」
  写得很清楚："For the ENPU scenario, it is required that **device events follow the principle of
  'record first, wait later'**, otherwise the inference process may become stuck. However, in the current
  `model_forward` function, `event.wait` precedes `event.record`. Therefore, for the ENPU scenario,
  **graph parameter updates should be performed before model execution**."
  实现：`vllm_ascend/worker/model_runner_v1.py:469-472`（`get_c_env("ENPU_ENABLE")`）、
  `:2669-2725`（`_update_full_graph_params_if_needed` 里先 `torch.npu.current_stream().synchronize()`，
  并调整 update/replay 顺序）、`vllm_ascend/compilation/acl_graph.py:257-263`
  （`if not self.enable_enpu and need_sync: synchronize()`）。
  来源：<https://github.com/vllm-project/vllm-ascend/pull/8456>（访问 2026-09-24）；本地源码同款。
- **能力边界（重要）**：`enable_enpu` **不是**一种更强的图更新能力，它只做两件事：
  ① 交换"图参数更新"与"模型前向"的**执行顺序**；② 在更新前加一次当前流同步。
  也就是说 —— **它换来的正确性/稳定性，代价是每步多一次 host 侧 sync**（在 host 是瓶颈的场景下这笔钱不小）。
- **证据强度**：上游 PR（含代码）+ 社区文档（中—强）；"ENPU 的图能力"在昇腾官方文档中**没有独立词条**（见 §9）。

### 5.4 GDN builder（`AscendGDNAttentionMetadataBuilder.build`，3×303 µs）能否图化？

**结论：builder 本身不能进图；它的"输出形态"可以改造成可图捕获的，但那不等于收益。**

分步判定（本地源码 `vllm_ascend/ops/gdn_attn_builder.py`）：

| builder 内部 | 现状 | 能否进图 | 说明 |
|---|---|---|---|
| `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`（`:226`） | 上游已声明"GDN 支持 uniform batch 的图" | ✅ 声明层面支持 | 上游 `AttentionCGSupport` 枚举（`refs/vllm/vllm/v1/attention/backend.py:600-614`）里 `UNIFORM_BATCH` 的含义是"每拍 query 长度一致时可图化"；这里说的**"图"指的是 attention 算子，不是 metadata 对象** |
| `_pad_non_spec_decode_graph_inputs`（`:332-369`） | 往 `non_spec_state_indices_tensor` / `non_spec_query_start_loc` **预分配缓冲**里 `fill_` / `copy_` | ✅ **天然可图捕获**（固定地址 + copy_ 语义） | 这是**正确的形态**，也是我们说的"近图化"范例 |
| `_reset_spec_decode_graph_inputs`（`:371-376`） | 把 spec 分支的捕获缓冲清零成 no-op | ✅ 同上 | 这正是对 CANN 约束 2（task 数量/类型必须一致）的**工程解法** |
| `prepare_chunk_indices` 等（`ops/triton/fla/utils.py:22-37`，由 `_build_non_spec_chunked_prefill_metadata` 调用） | **host torch + Python 循环**（`torch.cat([torch.arange(n) for n in ....tolist()])`） | ❌ | 只在 chunked prefill 路径触发；decode 稳态不走这条，但**同文件里 `tuple(...tolist())` 的构造模式**（`:164/200/212`）说明 builder 整体是 host 形态 |
| `GDNChunkedPrefillMetadata.cu_seqlens_host: tuple[int, ...]` 等 | dataclass 里直接存 **Python tuple** | ❌ | Python 对象语义，图里没有对应物 |
| `build()` 整体（`:531`） | 含 `_treat_single_token_prefills_with_state_as_decodes` 的 `.replace()`、`.nonzero()`、`.item()` 判定等多处 host 逻辑 | ❌ | 其中 `.item()`/`bool(...)` 还会引入 host↔device 往返（见 `_compact_empty_segments` 的 `bool(keep.all())`） |

**因此对"GDN builder 能否图化"的正式回答**：

- **不能**把 builder 作为整体图化（它是 host 函数，输出是 Python 对象）。
- **能**把它改造成"**不产生新地址**"的形态：所有需要给图用的量，都写进预分配的持久缓冲（已有范例）；
  所有纯算术（`compute_num_computed_tokens`、`split_decodes` 的掩码、padding 的 `copy_`）要么改成
  device kernel，要么至少改成对固定缓冲的 in-place 操作。
- **必须**先解决它在 host 上重复 3 次这件事（本项目热点：3 × 303 µs = 908 µs，占 `prepare input` 29.3%），
  否则谈图化没有意义 —— 重复的 host 工作**不会被任何图机制消除**。

---

## 6. 反向（负面）证据清单

这一节专门收纳"做了也不赚 / 做不了 / 官方明说更贵"的证据，供决策时对抗乐观预期。

| # | 结论 | 来源 | 类型 |
|---|---|---|---|
| N1 | **"更新任务比单独下发任务更耗时"**（Ascend task update） | CANN 官方文档（§5.2-A1） | 官方定性（强） |
| N2 | **把 slot_mapping kernel 折进 decode cudagraph：字节级一致、无回归，但 e2e 收益 = 0**（省下的 ~130 µs 被 async scheduling 掩盖）；作者结论："the ~3 ms host Python (the prize) is untouched"，win 在 B.2（tensorize `_prepare_inputs` / `_build_attention_metadata`） | vLLM PR #47924（2026-07-07 建、draft、后关闭） | 上游原型（强，且来自"最接近我们问题的那次实验"） |
| N3 | **注意力算子的 host 侧开销即使开了 Piecewise CUDA Graph 依然存在**，只能被隔离在图外 | TensorRT-LLM 官方文档 | 官方文档（强） |
| N4 | **FlashInfer 的 `plan()` 明确"cannot be used in Cuda Graph or in torch.compile"**；只有预分配 + 冻结形状的 wrapper 可图化 | FlashInfer 官方 API 文档 | 官方文档（强） |
| N5 | **在捕获区间内做 `.tolist()` 直接报错**（"unless the CPU tensor is pinned"） | vLLM issue #40807 | 上游 issue（强） |
| N6 | **SGLang 宣称的零开销调度被实测质疑**（"no actual overlap … CPU scheduling phase appears to be strictly sequential with GPU computation"） | SGLang issue #19347 | 社区实测（中） |
| N7 | **SGLang 移除 overlap 线程后流水线深度塌成 1，48.3% 时间花在 `Stream.synchronize`**，长上下文场景生成速率 -25~30% | SGLang issue #27186 | 社区实测（中） |
| N8 | **"部分重叠"的代价是"不能用 seq_lens_cpu"**：SGLang PR #16194 明说 draft 的调度优化不更新 `seq_lens_cpu`，只适用于不依赖它的模型 | SGLang PR #16194 | 上游 PR（中） |
| N9 | **async scheduling 在"device 远大于 host"的场景几乎无用**（"14ms vs 0.3ms… almost useless for decoding models"） | vLLM RFC #20727 评论 | 社区观点（弱，**且与我们的场景相反**） |
| N10 | **DBO 在小 batch/单机场景会变慢**（38 ms → 49 ms/step），因为权重被读两遍 | vLLM PR #23693 评论 | 上游讨论（中） |
| N11 | **快速路径式优化 prepare_inputs 被上游明确拒绝**（"I really don't want this optimization"），且指出根本约束是"避不开 CPU→GPU transfer" | vLLM PR #17866 | 上游 review（中—强） |
| N12 | **保留 CPU 影子变量会炸**：把 `seq_lens_cpu` 影子与 runner 的 pinned 缓冲别名后原地自增，引擎数秒内 `cudaErrorIllegalAddress` | vLLM issue #29134 评论 | 社区实测（中） |
| N13 | **Ascend 上捕获会耗尽 Stream 资源**，捕获尺寸覆盖过广直接失败（PIECEWISE 最敏感） | CANN 官方文档 + `vllm-ascend/ACL_Graph.md` | 官方文档（强） |
| N14 | **有些元数据反向流动更划算**：为消掉 D2H，把 metadata 计算改回用 CPU 侧张量 | vLLM PR #32815 | 上游 PR（中—强） |

---

## 7. `prepare_input` 的可图化分解表

> **口径**：子步骤名与耗时取自本项目已交付文档（`docs/00-INDEX.md` §2.2、`docs/05-hotspots.md`），
> 配置为 **0.8B / TP1 / B=1 / ISL=128 / decode / `FULL_DECODE_ONLY`**，
> 分母是 `prepare input` p50 ≈ 3 294 µs（C2c 臂，含 61 个探针 ≈134 µs 地板）。
> "预估收益"是**量级判断**，不是实测；标注 `[推断]`。
> 三列判定含义：**下沉 device** = 计算本身可以变成 kernel；**纳入图捕获** = 结果写入固定地址后，
> 该 device task 可以在 capture 区间内被录制；**只能留 host** = 目前看不到替代方案的部分。

| # | 子步骤（本项目 scope） | 每步耗时 | 下沉 device | 纳入图捕获 | 只能留 host 的部分 | 预估收益 / 理由 |
|---|---|---|---|---|---|---|
| 1 | `update_states`（含 `_update_states`、req 增删、行搬移） | 100.7 µs | ❌ | ❌ | **全部**：dict/condense/swap 是 host 数据结构账本，输入是 scheduler 的决策 | 不是"图化"的候选；只能靠**增量 diff + 持久状态**（V1 persistent batch / MRV2 `ReqStates`）压缩。`[上游文档]` |
| 2 | `block_table commit`（`block_table.copy_to_gpu`） | 88.8 µs | ❌（block id 由 host 分配） | ⚠️ 部分：若源缓冲固定，可录成 memcpy task；但 CANN 捕获期对 memcpy 有 GLOBAL 模式限制，且需页锁定内存 | host 侧块分配与表维护 | 把"每步 H2D"改成"固定地址 + 增量写"可省大半；真正省掉需要 **device 侧块表**（V2 的 `BlockTables` 方向）。`[上游代码]` |
| 3 | `positions` / `token_indices`（`in.positions_assembly` + S9 前半） | 113.7 µs（positions 部分） | ✅ **最有把握**：`positions = num_computed_tokens + local_pos`，两端在 device 上都有 | ✅（写入持久 `positions` buffer 后，attention 直接读同一地址） | 若 host 仍需 positions 做别的决策则须保留 CPU 镜像 | MRV2 已有 `prepare_pos_seq_lens` kernel；本项目是 numpy 实现。省 ~100 µs 级 + 消除一次 H2D。`[上游文档][推断]` |
| 4 | `input_ids` 的 `torch.index_select`（S9/S14） | 包含在 S9/S14（未单列） | ✅（device gather） | ✅ | `token_ids_cpu_tensor` 这份大矩阵的维护 | 关键前提：token id 不再每步 D2H。async scheduling 已经把 sampled token 留在 device；V2 用 `combine_sampled_and_draft_tokens` 直接在 device 上拼。`[上游文档]` |
| 5 | `_build_attention_metadata` — **GDN**（3× `AscendGDNAttentionMetadataBuilder.build`） | **1091.7 µs（33.1%）**，其中 `gdn.pad_graph_inputs` 294.1 µs、ctor+残差 373.6 µs、`treat_single_token` 167.7 µs、`compute_num_computed_tokens` 129.2 µs、`split_decodes` 117.8 µs | ⚠️ 部分：纯算术可 kernel 化；dataclass 构造不可 | ⚠️ 部分：**输出**中给图用的那几份（state indices / query_start_loc / actual_seq_lengths）已经在写预分配缓冲（`:332-376`）⇒ 那几笔可图化 | **builder 主体**：`.replace()`、`.nonzero()`、`.item()`、tuple 构造、3 个 KV group 各建一份 | **最高优先级不是图化而是去冗余**（3×→1×，预期 -600 µs 级）+ cache 修复（~130 µs，见 `docs/00-INDEX.md` §0 第 3 条）。图化只能吃掉"pad_graph_inputs"那一小块的 launch 开销。`[实测 + 上游代码]` |
| 6 | `_build_attention_metadata` — **full-attn**（`AscendAttentionMetadataBuilder.build` 1×） | 124.0 µs | ⚠️ 部分 | ⚠️ 部分 | `actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()`、`seq_lens_list = seq_lens.tolist()`（`attention_v1.py:332-333`）**必须留 host** | 捕获路径已经能用 **tensor** 传 kv 长度（`full_graph_fia_v2`，`attention_v1.py:1024-1060`）；把 q 长度也 tensor 化是顺理成章的下一步，但**尚未见上游验证**（标注 `[待验证]`）。`[上游代码]` |
| 7 | `slot_mapping`（`in.slot_mapping`） | 138.2 µs | ✅ **已经是 device 侧**（Triton kernel，CUDA/NPU 同款） | ✅ 且**已被上游验证过**（vLLM PR #47924 B.1：byte-identical） | host 只剩 launch + `num_reqs` 等标量 | **反面教材**：B.1 证明它可图化，但 e2e 收益 = 0（被 async scheduling 掩盖）。⇒ 单独图化 slot_mapping 对我们 **没有意义**。`[上游 PR]` |
| 8 | 各 H2D copy（`query_start_loc`、`seq_lens`、`num_computed_tokens`、`req_indices`、`query_pos`、`num_scheduled_tokens`、`discard_request_mask`、spec 的 5 处 `pin_memory().to(device)`） | 分散在 S11/S13/S16/S17/S18/S19/S24 | ✅（多数是"device 可重算的算术"） | ✅（固定缓冲后可作为 memcpy task 或干脆不再需要） | 需要 host 读值的那部分（例如 scheduler 记账用的 `num_computed_tokens_cpu`） | 量大但每笔小。按 920B 单次小操作 3–12 µs 计，抹掉 10+ 次小 H2D 是**百 µs 级**收益。前提是 host 不再需要这些值（async 化）。`[实测 + 推断]` |
| 9 | `sync_input_prep`（`prepare_inputs_event.synchronize()`） | 102.1 µs | ❌ | ❌ | **全部**（同步原语） | 只能靠重叠/流水线减少"暴露时间"；ascend 的 record 点在 `_update_states` 之后（`docs/01` §4.3），本身还有待验证是否该扩大覆盖 |
| 10 | spec-decode metadata（`_calc_spec_decode_metadata`） | 未单独计入（本配置为非 spec） | ✅ 已有先例（spec 的 5 个 Triton kernel） | ✅ | `scheduled_spec_decode_tokens` 字典遍历（host） | 已有上游实现可抄；非 spec 稳态下不适用 |

**读表要点**：

1. **能"纳入图捕获"的格子，几乎全部以"下沉 device / 固定地址"为前提**；
   换句话说，**图捕获是结果，不是手段**。
2. **"只能留 host"的格子集中在两处**：scheduler 决策账本（#1）与 host 读值/list 构造（#5/#6）。
   这两处正是 §8 里 1 ms 预算的**结构性障碍**。
3. 表里唯一"已经图化且已被上游验证"的 #7，恰好是**收益为 0** 的那一条 ——
   这本身说明"图化覆盖率"不是有意义的 KPI，"**host 关键路径上还剩多少工作量**"才是。

---

## 8. TPOT → 1 ms 时的临界判断

### 8.1 预算算术（用本项目实测数字搭骨架）

实测（`docs/00-INDEX.md` §2.1，0.8B / TP1 / decode / graph / pystack off）：

| 阶段 | p50 | 占单步 |
|---|---:|---:|
| `prepare input` | 2 757.6 µs | 55.3% |
| `forward` | 827.6 µs | 16.8% |
| 单步（`Step:Schedule` 口径） | 4 915 µs | 100% |
| 其余（调度 / 采样 / 输出处理 / IPC / 同步） | ≈1 330 µs（推算） | ≈27% |

若单步预算压到 **1 ms**，而 `forward` 与"其余"也各自按比例被压缩，那么留给
`prepare_input` 的**量级上限是几百微秒**（例如 forward 300–500 µs + 其余 200–300 µs ⇒ prepare ≤ 200–500 µs）。
从 **2 757.6 µs** 出发，需要 **5.5×–13×** 的降幅。下面逐条判定。

### 8.2 哪些技术能让它进预算

| 技术 | 能不能进 1 ms 预算 | 依据 |
|---|---|---|
| **去冗余**（3 个 GDN builder → 1 个 + 修 `.replace()` 清缓存） | ✅ **必做**，单独可省 ~600 µs + ~130 µs | 本项目实测（`docs/00-INDEX.md` §0 第 3 条） |
| **静态缓冲 + 去每步分配**（不再 `torch.empty`/新建 dataclass，全部 `copy_`/in-place） | ✅ **必做**（也是任何图化的前置条件） | §A2/A4；上游 V2 DBO RFC #50738 表述 |
| **纯算术下沉 device**（positions / token_indices / seq_lens / logits_indices / slot_mapping / padding mask） | ✅ **必做**，把十几笔 host 小操作换成几次 kernel launch（920B 上单次 3–12 µs） | §B1/B2；MRV2 官方文档 |
| **消掉 host↔device 读值**（`.tolist()` / `.item()`），改用 CPU 镜像或 device 值 | ✅ 必做，但**方向要选**：要么 host 不再需要（async 化），要么保留一份稳定的 CPU 镜像（注意 N12 的别名陷阱） | §B3/B4；vLLM #40807、#32815 |
| **账本 C++/编译化**（`_update_states`、block table 维护、调度决策翻译） | ⚠️ 长期唯一出路：TRT-LLM 用 C++ 做这件事；vLLM 用 V2 runner + async 做这件事 | §C1/C4 |
| **重叠 / 流水线**（async scheduling、`max_concurrent_batches=2`、DBO） | ✅ 能把 host 从"关键路径"变成"与 device 并行"，但**只有当 host 单步 ≤ 2× device 单步**时才有效；且它**不减少 work，只隐藏 work** | §C2/C4 |
| **"把 prepare_input 图化"（图捕获/图更新）** | ❌ **不能单独进预算**：图捕获不覆盖 host（§1），Ascend 的图更新官方明说比下发更贵（§5.2-A1），上游实测收益为 0（§6-N2） | §1/§5.2/§6 |
| **把 list 参数交给 `graph_task_update`** | ❌ 不解决构造成本；仅解决"整图 replay 时参数正确性" | §A5 |

### 8.3 必要条件清单（"想进 1 ms 必须先满足这些"）

按优先级排序；前四条是**必要**，第五条及以后是**充分性**条件（缺一条都到不了 1 ms）：

1. **host 侧不再"需要读 device 值"**：任何 `.tolist()` / `.item()` / `.cpu()` 出现在每步热路径上，
   就等价于给预算加了一次同步（§B3；vLLM #40807）。检验方法：在 `prepare_input` scope 内统计
   跨设备读值次数，目标 **0**。
2. **每步不再分配新的 device 内存**：所有每步变化量写入**地址固定**的持久缓冲
   （`copy_` / `fill_` / in-place kernel）。检验方法：捕获区间内不得出现分配（CANN 约束 7；vLLM V2 DBO RFC）。
3. **attention 算子所需动态参数可以 tensor 化**（至少能通过固定地址表达）。
   GDN/FIA 的 list 形参必须找到 tensor 等价物（**上游已证明 FIA 的 kv 长度可以走 tensor**，
   `attention_v1.py:1024-1060`）。检验方法：`graph_task_update` 里不再出现 Python list。
4. **GDN 的 3× 重复必须消除**：908 µs 本身就是 1 ms 预算的 91%，任何其他优化都救不了它。
5. **batch/形状必须离散化并 padding 到 capture size**（V1 已做；V2 更彻底）。
6. **有真实的重叠结构**：async scheduling / 双 batch / 双线程，把"host 构造"与"device 执行"并行；
   同时必须能容忍"host 读值滞后"（SGLang 的 `needs_cpu_seq_lens`、vLLM 的
   `_seq_lens_cpu=None` 语义就是这个意思，见 §6-N8 与 issue #29134）。
7. **剩余 host 账本（`update_states`、块表、调度翻译）必须离开 CPython**：
   走 C++（TRT-LLM 路线）或高度编译化/持久状态化（V2 路线）。这一条决定 1 ms 的**上限**。

### 8.4 一句话判断

> **`prepare_input` 不会被"图下发"顺带解决。**
> 在 1 ms 预算下，它是**硬约束**：需要在"去冗余 + 静态缓冲 + 下沉 device + 账本编译化"四个方向同时推进，
> 且必须用重叠把剩下的 host 工作藏到 device 后面。
> "图化"在这个组合里只负责**最后 10%**（把已经 tensor 化/固定地址的参数交给 device 的那一步），
> 且它在 Ascend 上的**官方定位**是"正确性机制"，不是"性能机制"。

---

## 9. 明确查不到的（避免后人重复挖）

1. **ENPU 的官方定义/能力说明**：昇腾文档中没有 "ENPU" 的独立词条；
   唯一可靠来源是 openEuler `ubs-virt`/vCANN-RT 的软切分文档（`ENPU_ENABLE` 进程级环境变量）
   与 vllm-ascend PR #8456。**没有任何资料表明 ENPU 与"更强的图更新能力"有关。**
2. **`aclmdlRICaptureTaskUpdate*` 的定量开销**：只有定性表述"更新任务比单独下发任务更耗时"，
   **没有 µs 级数据**，也没有"更新 1 个 task 多少钱"的公式。
3. **CUDA `cudaGraphExecKernelNodeSetParams` vs 普通 kernel launch 的成本对比**：
   NVIDIA 只承诺"比重新 instantiate 轻"，**没有**"比 launch 便宜"的承诺，也没有官方基准。
   ⇒ 想在大 batch/小 kernel 场景下用它省时间，**必须先自测**。
4. **Ascend 上"图更新改变形状"是否可行**：官方只说"任务数量和类型必须一致"，
   **未提形状**；我们的代码是靠 padding 归一形状的，因此**形状变更应当视为不可用**（推断，非引用）。
5. **vLLM 上游是否有"把 attention metadata builder 整体下沉 device"的 PR**：截至 2026-09-24，
   只找到 **B.1（slot_mapping 进图）** 与 **B.2（tensor 化 `_prepare_inputs` / `_build_attention_metadata`）
   的规划表述**（PR #47924 已关闭，未见后续 PR）**，以及 MRV2 的 `prepare_*` Triton kernel。
   **没有**"metadata builder on device"的独立 RFC。
6. **GDN/linear-attention 在昇腾上的 metadata 图化实践**：没有任何公开资料（社区、上游 issue、华为文档）
   讨论过 linear attention/GDN 的 metadata 进图问题；本报告 §5.4 的判断**全部来自本地源码 + 通用约束的外推**。
7. **`graph_task_update` 传 Python list 的底层代价**（是否每步在 C++ 侧把 list 变成 device tensor、
   是否因此产生一次 H2D、是否触发 tiling 重算）：**代码可见 list 被传入，底层实现无公开资料**。
   ⇒ 这是一个**可实测**的空白（与 `docs/10-prepare-input-graphification.md` 的待验证项 V3 同源）：
   在 `update_graph_params` 加探针，测 6 层 FIA 的 update 总时长；若 >100 µs，
   "图化省 host"在本平台**直接证伪**。
8. **SGLang 是否有"attention metadata 下沉 device"的完整方案**：只找到 backend 级的
   `needs_cpu_seq_lens` 开关与 `FutureMap` 的按需 D2H，**没有**"metadata 在 device 上构造"的实现。
9. **Dynamo 是否干预引擎内 per-step 元数据**：其架构文档完全聚焦 Router/Planner/KVBM/NIXL，
   **没有**任何关于引擎内输入准备的描述（本文的"不涉及"是基于文档内容的推断）。
10. **昇腾 Npugraph_ex 静态 kernel 编译对 host 元数据路径的收益**：
    官方只说它"pre-compiles operator binaries with fixed shapes"，**未涉及** host 侧 Python 账本。

---

## 10. 来源索引

### 10.1 官方文档

| 主题 | URL | 访问日期 |
|---|---|---|
| CUDA Graphs 更新机制 | <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html> | 2026-09-24 |
| CUDA Runtime Graph API（`cudaGraphExecKernelNodeSetParams` / `cudaGraphExecUpdate`） | <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html> | 2026-09-24 |
| ACL Graph 简介（CANN 9.0） | <https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/acldevg/runtime_doc_dev_0045.html> | 2026-09-24 |
| 捕获式构建模型运行实例（含 TaskGrp/TaskUpdate 与"更新更耗时"） | <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/appdevg/acldevg/aclcppdevg_000519.html> | 2026-09-24 |
| 单流捕获限制（CANN 9.0） | <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0030.html> | 2026-09-24 |
| `aclmdlRIExecuteAsync` | <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1822.html> | 2026-09-24 |
| `aclmdlRICaptureBegin`（runtime 组件仓库） | <https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclmdlRICaptureBegin.md> | 2026-09-24 |
| torch_npu `NPUGraph` 开发指南（含 update 机制与适用/不适用场景） | <https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/pytorch_npugraph_desc.md> | 2026-09-24 |
| torch_npu `graph_task_update_begin` | <https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/torch_npu-npu/torch_npu-npu-graph_task_update_begin.md> | 2026-09-24 |
| torch_npu `graphs.py` 实现（`_GraphDispatchMode`） | <https://gitcode.com/Ascend/pytorch/blob/df242506eb509201b170357330b48c649f38ef9b/torch_npu/npu/graphs.py> | 2026-09-24 |
| vLLM CUDA Graphs 设计文档 | <https://docs.vllm.ai/en/latest/design/cuda_graphs/> | 2026-09-24 |
| vLLM Model Runner V2 设计文档 | <https://docs.vllm.ai/en/v0.19.1/design/model_runner_v2/> | 2026-09-24 |
| vLLM Dual Batch Overlap | <https://docs.vllm.ai/en/latest/design/dbo/> | 2026-09-24 |
| vllm-ascend ACL Graph 设计文档 | <https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html> | 2026-09-24 |
| FlashInfer attention API（plan vs CUDAGraph） | <https://docs.flashinfer.ai/api/attention.html> | 2026-09-24 |
| TensorRT-LLM C++ GPT Runtime | <https://nvidia.github.io/TensorRT-LLM/advanced/gpt-runtime.html> | 2026-09-24 |
| TensorRT-LLM Architecture Overview（Overlap Scheduler / CUDA Graph padding） | <https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html> | 2026-09-24 |
| TensorRT-LLM Piecewise CUDA Graph（attention host 开销残留 + 三条硬约束） | <https://nvidia.github.io/TensorRT-LLM/1.2.0rc8/features/torch_compile_and_piecewise_cuda_graph.html> | 2026-09-24 |
| SGLang v0.4 零开销调度博客 | <https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/> | 2026-09-24 |
| Dynamo Overall Architecture | <https://docs.nvidia.com/dynamo/dev/design-docs/overall-architecture> | 2026-09-24 |
| Dynamo Router Design | <https://docs.nvidia.com/dynamo/v1.4.0/knowledge-base/modular-components/router/router-design> | 2026-09-24 |
| vCANN-RT / ENPU_ENABLE（软切分） | <https://gitcode.com/openeuler/ubs-virt/blob/master/ubs-virt-enpu/vcann-rt/README.md> | 2026-09-24 |
| NPU 虚拟化软切分参考实践 | <https://www.hiascend.com/developer/techArticles/20260310-1> | 2026-09-24 |

### 10.2 上游 PR / issue

| 编号 | 标题（截断） | 状态 | URL |
|---|---|---|---|
| vLLM #47924 | [RFC+prototype] Compile the decode step e2e incl. KV-cache management（B.1 slot-mapping in cudagraph） | draft，已关闭（2026-07-07 建） | <https://github.com/vllm-project/vllm/pull/47924> |
| vLLM #29134 | [Performance]: Fully Async Spec-Decoding \| Make `seq_lens_cpu` optional | open | <https://github.com/vllm-project/vllm/issues/29134> |
| vLLM #29624 | [Attention] Make `seq_lens_cpu` optional in `CommonAttentionMetadata` | — | 同上 issue 引用 |
| vLLM #40807 | [Bug]: … crashes CUDA graph capture at `query_start_loc.tolist()` | open | <https://github.com/vllm-project/vllm/issues/40807> |
| vLLM #32815 | [Feature]: Remove DtoH Copy for lfm2_vl On Default Stream | — | <https://github.com/vllm-project/vllm/pull/32815> |
| vLLM #17866 | [V1] Fast decode prepare path for `prepare_inputs` logic | 被拒 | <https://github.com/vllm-project/vllm/pull/17866> |
| vLLM #20727 | [RFC]: Async scheduler and Multi-step in v1 | — | <https://github.com/vllm-project/vllm/issues/20727> |
| vLLM #20448 / #23693 / #24845 | Dual-Batch Overlap（microbatching）系列 | 部分合并 | <https://github.com/vllm-project/vllm/pull/23693> |
| vLLM #50738 | [RFC]: Dual Batch Overlap (DBO) for Model Runner V2 | — | <https://github.com/vllm-project/vllm/issues/50738> |
| vLLM #51700 | [2/2][Model Runner V2] FULL CUDA graph capture for microbatched steps | — | <https://github.com/vllm-project/vllm/pull/51700> |
| vLLM #28579 | [Core] Refactor padding logic and pad for CUDA graphs before attention metadata building | 合并 | <https://github.com/vllm-project/vllm/pull/28579> |
| vllm-ascend #8456 | [BugFix]: order acl graph updates before model forward for ENPU | — | <https://github.com/vllm-project/vllm-ascend/pull/8456> |
| SGLang #19347 | Zero-Overhead Batch Scheduler fail to overlap CPU scheduling with GPU computation as claimed | open | <https://github.com/sgl-project/sglang/issues/19347> |
| SGLang #27186 | [RFE] Re-introduce tunable forward-pipeline depth in overlap scheduler | open | <https://github.com/sgl-project/sglang/issues/27186> |
| SGLang #16194 | [scheduler] scheduler overlap | — | <https://github.com/sgl-project/sglang/pull/16194> |
| FlashInfer #187 | Make flashinfer kernels cuda graphs friendly | closed | <https://github.com/flashinfer-ai/flashinfer/issues/187> |

### 10.3 本项目内部证据（本文反复引用）

| 出处 | 用到的内容 |
|---|---|
| `docs/00-INDEX.md` §0/§2 | `prepare_input` 2 757.6 µs / 55.3%；forward 827.6 µs；子 scope 分解；GDN 3×303 µs；frontend_bound 66.01%、IPC 0.771、`aclrtSynchronize*` 0.27% |
| `docs/01-prepare-input-code-logic.md` §2/§4.2/§4.3/§8 | 26 个子步骤、`.tolist()`/list 形参的机制原因、`synchronize_input_prep` 语义、`_update_states` 成本模型 |
| `docs/05-hotspots.md` | GDN builder 的 perf 证据、`_pad_non_spec_decode_graph_inputs` self 5.94% |
| `docs/10-prepare-input-graphification.md` | eager vs graph 的绝对耗时对比（2.843→3.132 ms）、三前提判定、待验证项 V1–V3 |
| `refs/vllm`（0.26.0） | `block_table.py:153/166/184`、`_prepare_inputs`（`gpu_model_runner.py:1937+`）、`_prepare_input_ids`（`:1761+`）、`AttentionCGSupport`（`v1/attention/backend.py:600-627`）、`spec_decode/utils.py`、`config/vllm.py:512`、`config/scheduler.py:158`、`v1/worker/gpu/model_runner.py`（V2） |
| `vllm-ascend`（0.26.0rc1） | `attention/attention_v1.py`（476/525/538/619/640/809/842/992/1016/1024-1060/1121/1142/1195/1208）、`attention/utils.py:88/101`、`ops/gdn_attn_builder.py`（226/332-376/531 与 `ops/triton/fla/utils.py:22-37`）、`worker/model_runner_v1.py`（469-472/1246-1250/2669-2725）、`compilation/acl_graph.py`（93-122/230-290）、`docs/source/.../ACL_Graph.md`、`graph_mode.md`、`npugraph_ex.md` |

---

## 附：给下一个人的最短路径

如果只允许做三件事来推进这个方向，按证据强度排序应当是：

1. **测 `graph_task_update` 的单次 host 成本**（6 层 FIA 的 update 总时长）。
   判据：若 >100 µs，则"图化省 host"在本平台**直接证伪**，可以把这条路从主线里划掉。
   —— 这是 CANN 官方"更新比下发更贵"在我们平台上的定量化，也是唯一能把 §5.2-A1 变成我们自己的数字的实验。
2. **把 GDN builder 的去冗余（3×→1×）+ `.replace()` 缓存修复做掉**（预期 ~730 µs，占 `prepare_input` 26%）。
   —— 与图化无关，但它是 1 ms 预算里最大的单块。
3. **照 V2 的形态改造 GDN/full-attn 的"输出侧"**：所有给图用的量写进持久缓冲，
   消除每步 `torch.empty`/dataclass 分配；并把 `positions`/`token_indices`/`seq_lens` 这类纯算术
   换成 device kernel（可先抄 `prepare_pos_seq_lens` 的语义）。
   —— 这一步才是"接近图下发"的真实形态。
