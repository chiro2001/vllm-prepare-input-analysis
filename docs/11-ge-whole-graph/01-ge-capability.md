# GE / 整图下发能力：机制、边界与可引用证据

> **调研日期**：2026-09-24（本文所有 URL 均于该日访问，并逐个做过 HTTP 200 校验）。
> **调研方式**：纯文献 / 官方文档 / 网络调研。**未**登录任何机器、**未**运行 NPU、**未**起容器、**未**改动环境。
> **事实基线**：同目录 [`00-source-evidence.md`](./00-source-evidence.md)（vllm-ascend 0.26.0rc1 的一手代码/文档证据）与
> [`../10-prepare-input-graphification.md`](../10-prepare-input-graphification.md)（本项目的实测与源码推断）。本文**不重复**挖 vllm-ascend。
>
> **证据强度约定**
>
> | 强度 | 定义 |
> |---|---|
> | **强** | 华为官方用户手册（hiascend.com 文档中心）、官方组件仓文档（`gitcode.com/cann/*`）、官方技术文章、TorchAir 官方仓库文档 |
> | **中** | 官方样例仓文档（cann-recipes-infer）、维护者在官方 issue 下的回复、CANN 开发者社区文章 |
> | **弱** | 第三方博客/论坛，无法与官方文档交叉核实 |
>
> 行文中每条证据用 `E<x.y>` 编号，格式固定为：**来源 / 机制 / 能力边界 / 证据强度**。

---

## 0. 一页结论（可直接进工程决策）

| # | 结论 | 依据 |
|---:|---|---|
| 1 | **`max-autotune` 就是 GE 模式**（官方又称 Ascend IR 模式），而且它是 TorchAir `CompilerConfig.mode` 的**默认值**。直接按官方快速上手写 `torchair.get_npu_backend()`，得到的就是 GE 图模式。 | E1.1 |
| 2 | `mode` 只有三个取值：`max-autotune`（GE/Ascend IR）、`reduce-overhead`（旧 ACLGraph，自 torch_npu 7.3.0 起**不再演进**）、`npugraph_ex`（新 ACLGraph 后端）。vllm-ascend 显式选了后两者之一并 `run_eagerly=True`，**主动避开** IR 变换。 | E1.1 / 00-source-evidence §2 |
| 3 | **"整图下发"省掉的不是"host 开销"，而是"按算子计费的 host 开销"**：GE 把成本从「每算子一次 Host 派发」降到「每步一次模型执行 Task + 每步输入输出元数据转换」。**"完全抹除 host 开销"在官方文档里没有任何依据。** | §2.2 四层表 / E2.1 |
| 4 | **只有静态 shape 图才能真正下沉（sink）**。动态 shape 图在 GE 里是 **Host 调度**：官方原文"需在执行时逐算子进行 InferShape、Tiling 和内存分配，然后将 Kernel 逐一下发至 Device"。 | E2.1 / E2.4 |
| 5 | 下沉后仍有**四项头开销**，官方明确列在第 1 位的就是"模型输入输出 Tensor 到内部 InputData/OutputData **数据结构转换**"。官方给出量级：盘古 71B（约 1600 个输入输出、约 6300 节点、10 个 I/O 地址变更）**头开销约 2 ms**。 | E2.2 |
| 6 | GE 稳态执行**每步仍要付**：Dynamo **Guards 校验** + **Input 转换**（刷新图输入地址）。这是官方 `compile_cache` 文档配图里明确画出来的两个环节。 | E2.3 |
| 7 | **图外的 host Python 代码（对象构造、numpy、dataclass）GE 一个字都省不掉**。官方工程要求是"动态信息**显式传入**，而不是在模型内部临时生成 Python 标量"；图捕获期 `.item()`、基于 Tensor 值的 `if`/`while` 都会断图回 eager。 | E3.2 / E3.4 |
| 8 | **每步变化的值可以进图**，但只有三种形态：① 改成 **Tensor 图输入**（官方为 FIA 专门提供了 `torchair.ops.npu_fused_infer_attention_score`，`actual_seq_lengths*` 收 Tensor 且**仅图模式可用**）；② `dynamic=True` + `mark_static` 让标量变成 `ge.Data`（CPU 占位、shape=[]）从而"值变不重编译"；③ `dynamic=False` 直接编成 `ge.Const`（**值一变就要重编译**）。 | E3.3 / E3.7 / E3.8 |

---

## 1. 分层：GE / TorchAir / ACLGraph 到底谁是谁

### 1.1 三层职责

| 层 | 组件 | 干什么 | 关键 API / 产物 | 证据 |
|---|---|---|---|---|
| **前端捕获层** | PyTorch **Dynamo**（`torch.compile`） | 重写 Python 字节码，把张量算子序列抽成 **FX 图**，生成 **Guards** 并在每次执行前校验 | FX `GraphModule`、Guards | E1.3 |
| **图编译层（两条互斥的路）** | **① TorchAir → Ascend IR → GE**<br>**② TorchAir npugraph_ex / `torch.npu.graph` → ACLGraph** | ① 把 FX 图转成 **Ascend IR**，交给 **GE（Graph Engine）**做图优化、算子编译、内存编排、下沉/调度<br>② **不经过 GE 图编译**，直接对 Stream 做 **Capture & Replay** | ① `ge.Data/ge.Const/ge.MatMulV2…`、`.om`、GE build 图<br>② `aclmdlRI` 句柄 | E1.1 / E1.3 / E1.2 |
| **执行器层** | ① GE Runtime（DavinciModel / TaskSink / ExecuteGraph）<br>② ACL Runtime（model running instance） | ① 静态图下沉后由设备侧自主调度；动态图由 Host 逐算子调度<br>② 回放捕获到的任务序列 | ① `ExecuteGraphWithStreamAsync`（日志可见）、`rtModelExecute`<br>② `aclmdlRIExecuteAsync` | E1.2 / E1.3 / E1.4 |

**一句话关系图**：

```text
                 Python 模型代码
                        │  torch.compile(dynamo)
                        ▼
                     FX 图 (+Guards)
        ┌───────────────┴────────────────┐
        │ mode=max-autotune（默认）        │ backend="npugraph_ex"
        │ torchair.get_npu_backend()      │ 或 mode="reduce-overhead"
        ▼                                ▼
  Ascend IR ──► GE 编译 ──► GE Runtime   torch.npu.graph / aclmdlRICapture*
  （图优化/融合/内存编排/下沉）              （捕获 Stream 任务到 modelRI）
        │                                │
        ▼                                ▼
  ExecuteGraphWithStreamAsync      aclmdlRIExecuteAsync（replay）
```

### 1.2 E1.1 `CompilerConfig.mode` 的取值、默认值与语义（这是"GE 是不是默认"的直接答案）

- **来源**：[CompilerConfig类](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/torchair/compiler_config.md)（API 参考，访问 2026-09-24）；
  [GE图模式快速上手](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/quick_start.md)；
  [TorchAir 简介](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md)；
  [自定义算子入图概述](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/custom_op_graph/overview.md)。
- **机制**：`CompilerConfig` 定义
  `self.mode = OptionValue("max-autotune", ["max-autotune", "reduce-overhead", "npugraph_ex"])`
  —— 第一个参数是**默认值**，后面是合法取值集合。官方快速上手原文：
  "GE图模式一般通过 TorchAir 的 CompilerConfig 属性 **mode="max-autotune"** 开启（**该模式是系统默认模式**），其将 FX 图转换为 Ascend IR 图，并通过 GE（Graph Engine）实现计算图的编译和执行。"
  官方简介进一步说明：`npugraph_ex` 后端采用 Capture&Replay，"一般通过 Runtime 提供的 `aclmdlRICaptureXxx` 系列接口实现"；
  并声明"从 TorchNPU 7.3.0 之后的版本开始，原 **reduce-overhead 模式（aclgraph）**通过 `config.mode` 配置图编译后端的方式**将不再演进**，也不再推荐使用"。
- **能力边界**：
  - `max-autotune` = **GE / Ascend IR 模式**（支持/默认）；
  - `reduce-overhead` = 旧 ACLGraph 模式（**有条件支持，官方已不推荐**）；
  - `npugraph_ex` = ACLGraph 的新入口（**支持**，也更推荐以独立模块/独立后端方式使用）。
- **证据强度**：**强**（官方 API 参考 + 官方快速上手 + 官方简介，三处互相印证）。

### 1.3 E1.2 `aclmdlRIExecuteAsync` / `aclmdlRIDebugJsonPrint` 属于 **ACLGraph 层**，不是 GE

- **来源**：
  - [aclmdlRIExecuteAsync（模型运行实例管理 / Runtime API(C)）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1822.html)；
  - [模型运行实例管理（runtime 仓 API 参考）](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/15_model_running_instance__management.md)；
  - [ACL Graph 任务更新（CANN 9.1.0 应用开发）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0032.html)；
  - TorchNPU 源码（本地只读）：`/home/chiro/projects/vllm/HIST_PROJECT/.research/ascend-pytorch-v2.10.0/torch_npu/csrc/core/npu/NPUGraph.cpp`。
- **机制**：
  - `aclmdlRICaptureBegin/End` 期间"所有在指定 Stream 上下发的任务**不会立即执行**，而是被暂存在系统内部模型运行实例中"，由 `aclmdlRIExecute`/`aclmdlRIExecuteAsync` 触发真正执行，"以此减少 Host 侧的任务下发开销"；
  - `aclmdlRIDebugJsonPrint` 是把模型运行实例信息以 JSON 导出到文件的**维测接口**；
  - torch_npu 侧映射（源码可核对）：`graph_task_group_begin/end` → `AclmdlRICaptureTaskGrpBegin/End`；`graph_task_update_begin/end` → `AclmdlRICaptureTaskUpdateBegin/End`；`NPUGraph::replay()` → `AclmdlRIExecuteAsync`；`NPUGraph::debug_dump()` → `AclmdlRIDebugJsonPrint`。
- **能力边界**：这套 API **只服务于 Capture&Replay（ACLGraph）路径**；GE 路径不使用它们（GE 走 GE Session + `ExecuteGraphWithStreamAsync` 一类的图执行接口）。**不要把它们当成"GE 整图下发"的 API。**
- **证据强度**：**强**（官方 Runtime API + 官方运行时仓文档 + torch_npu 源码三处一致）。

### 1.4 E1.3 TorchAir 自己的分层叙述

- **来源**：[动/静态图-背景介绍](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/dynamic_static_graph/background.md)（访问 2026-09-24）。
- **机制**：官方把 max-autotune 的编译链路写成一个 8 步时序：
  ① Converter 前 FX 优化 → ② Converter（Aten IR→Ascend IR） → ③ Converter 后 FX 优化（含"**符号输入转换为 ge.Data**"）
  → ④ 反序列化加载得到 GE Model → ⑤ 首次执行触发向 **GE Session** 添加 graph → ⑥ 首次执行触发 graph **编译** → ⑦ TorchAir 生成 Executor → ⑧ 调用 **GE 图执行接口（如 `ExecuteGraphWithStreamAsync`）**。
- **能力边界**：图编译与图执行是**两个阶段**，编译可以发生在首次执行时（因此有"首次编译耗时"），也可以被 `cache_compile` 的缓存跳过；执行阶段是 GE Runtime。
- **证据强度**：**强**（官方 TorchAir 文档）。

### 1.5 E1.4 GE Runtime 侧的"下沉"是怎么落地的（TaskSink / `rtModelExecute`）

- **来源**：[GE Runtime 设计（ge 仓设计文档）](https://gitcode.com/cann/ge/blob/master/docs/zh/design/modules/runtime/runtime.md)。
- **机制**：设计文档把 GE 的两代执行器写得很直白：
  - v1（`DavinciModel + TaskDef`）的**下沉（Sink）路径**是"阶段四：DoTaskSink —— 任务下沉"：`BindModelStream` 绑定逻辑流 → `InitTaskInfo + DistributeTask` 把每个 TaskDef（Kernel/Hccl/FftsPlus 等）**预下发到设备** → `aclmdlRIBuildEnd` 通知运行时"模型构建完毕" → 之后 **Host 只需一次 `rtModelExecute` 调用**；文档原文举例"传统 Host 调度：Host 逐算子下发 → Device 执行 → ……（N 次交互）"vs"**Sink 模式：Host 一次 launch → Device 自主执行所有 Task（1 次交互）**"。
  - v2（`ExecuteGraph + Node/Kernel`）走 Host 顺序/拓扑执行，对应动态 shape/单算子场景。
- **能力边界**：**这是"整图下发"在 runtime 层的真实含义** —— 把执行序列在加载期固化到设备，稳态只留一次模型执行调用。注意此处 `aclmdlRIBuildEnd` 是"**构建**模型运行实例"的接口，与 ACLGraph 的"**捕获**（`aclmdlRICapture*`）"是两条不同的路（名字相近，勿混）。
- **证据强度**：**强**（CANN 组件仓官方设计文档；与 §2.1 的官方用户文档口径一致）。

### 1.6 术语澄清表（避免和 vllm-ascend 文档里的叫法混淆）

| 名字 | 真实身份 | 是否经过 GE 图编译 |
|---|---|---|
| **GE / Graph Engine** | Ascend 的图编译器 + 图执行器（图优化、算子编译、内存编排、调度下沉） | — |
| **TorchAir**（`torchair`） | 华为维护的 torch↔Ascend 图编译桥；`npugraph_ex` 是它的一个模块 | 取决于 `mode` |
| **`max-autotune` / Ascend IR 模式** | TorchAir 的 **GE 模式**（默认） | ✅ |
| **ACLGraph** | `torch.npu.graph`（`aclmdlRICapture*`）的 Capture&Replay | ❌ |
| **`npugraph_ex`** | ACLGraph 的新后端封装（含 FX 图优化，但**不做 IR 变换**） | ❌ |
| **`reduce-overhead`** | 旧的 ACLGraph 入口，官方已不推荐 | ❌ |

---

## 2. 【核心交付】"整图下发"到底省掉什么

### 2.1 E2.1 先把官方定义的两种调度模式钉死

- **来源**：
  - [模型下沉调度（CANN 8.5.0 图开发 / 概念和原理介绍）](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/graphdevg/atlasag_25_0087.html)；
  - [深度解读昇腾CANN模型下沉技术，提升模型调度性能](https://www.hiascend.com/zh/developer/techArticles/20240715-1)（2024-07-15）；
  - [深度解读昇腾CANN动态Shape图调度加速技术](https://www.hiascend.com/developer/techArticles/20250911-1)（2025-09-12）。
- **机制（原文级）**：
  - **Host 调度**："Host CPU 将模型中的算子**依次下发**到 Device 执行……每次运行都会触发 Host 把模型上的所有算子遍历下发一遍。"
  - **下沉调度（= 整图下发）**："让模型中的算子**在加载阶段提前以整图的形式下发到 Device 上**，在执行时，只需在 Host 侧下发**一个模型执行的 Task** 即可触发模型在 Device 上调度执行。"
  - 官方对下沉的两个前提说得很直白："对于输入 tensor shape **固定不变**的静态 Shape 模型，在编译时即可确定所有算子的输入输出 shape……可完成**模型级内存编排**；静态 shape 模型在编译时还可**提前完成所有算子的 Tiling 计算**等 Host 侧计算。"
- **能力边界**：
  - **支持**：静态 shape 模型 → 下沉调度（每步 1 个 Task）；
  - **不支持**：动态 shape 模型 → 只能 Host 调度（逐算子下发）；
  - **有条件**：静态 shape 图里若存在"**值依赖算子**"，该算子默认走 Host 调度（见 E3.8）。
- **证据强度**：**强**。

### 2.2 【核心表】四层开销：GE 到底能省哪一层

> 口径说明：本表按 **"稳态 decode 一步"** 计价，对比基准是 **eager（单算子模式）**。
> "有条件"一律指 **静态 shape + 成功下沉** 这个前提；不满足时按"否"处理。

| 层 | 具体是什么 | GE 能省吗 | 为什么（官方依据） |
|---|---|---|---|
| **(a) kernel launch**<br>（把 kernel/Task 发到 device 执行队列） | 每个算子对应的 1..N 个 Task 的 launch；含 device 侧 task 调度间隙 | **有条件能，且是最彻底的一层** | 下沉模式下"模型中的算子在**加载阶段**提前以整图形式下发到 Device"，执行期只有**一个模型执行 Task** ⇒ 稳态每步的 per-op launch 归零（E2.1）。**动态 shape 图不行**：官方明确"逐算子……将 Kernel 逐一下发至 Device"（E2.1/E2.4）。**分档（gear）图可以**：官方称每个档位"转换为静态子图，**能一次全部下发到 Device 侧**"（E3.9）。 |
| **(b) op 派发**<br>（host 侧"每算子一次"的派发动作：Python→C++、ACL 调用、InferShape/Tiling/AllocMem 的 Host Kernel 链） | eager 下一个算子的下发流程含"**Python 处理、Python 到 C++ 数据结构转换、Tiling 计算、申请算子 Workspace 内存和输出内存、Launch**"等 Host 操作（官方原文） | **有条件能，但有两级折扣** | 折扣一：**图模式 Host 调度**（动态 shape）"可以避免总是返回 Python 调用栈，避免冗余流程与数据结构转换，并且可以直接使用图编译阶段完成的 Infer Shape 与 Tiling 计算结果"；官方对其相对 eager 的定位是"**调度性能持平或略优**"——**不是抹除**（E2.1）。折扣二：同一份官方文档说得很明确——动态 shape 下"一个 AI Core 算子的下发会被拆分为 **InferShape、Tiling、AllocMemHbm、Launch 等多个 Host Kernel**"（E2.4）。只有**下沉**才把这一层真正拿掉；且官方给了量化机制："N 个算子聚合成一个静态子图后**只需要一次下发**"（E2.5）。 |
| **(c) 张量元数据构造**<br>（TensorDesc/InputData/OutputData、shape/stride、地址绑定、tiling 参数） | 两类：**图内算子**的元数据、**图输入输出**的元数据 | **图内算子：能省（编译期定型）；图输入输出：不能省，是每步固定成本** | 图内：静态 shape 下"InferShape + Tiling + 内存编排**都在编译期完成**"（E2.1）。图输入输出：官方《下沉头开销》第 1 项就是"**模型输入输出 Tensor 到内部 InputData/OutputData 数据结构转换**"，第 2 项是"刷新相关联算子的相关地址"（E2.2）；官方 `compile_cache` 配图把 **Input 转换**画成**每次执行**（含稳态）都要走的环节（E2.3）；官方 API `frozen_parameter` 的存在也印证这类"地址刷新"是成本，需要专门开关去省（E3.11）。 |
| **(d) host 对象分配**<br>（dataclass / list / numpy / 临时 python 对象 / 新申请的张量对象） | 图**外**的 Python 对象构造，以及图内中间张量的**device 内存分配** | **不能省（图外一律不能省）**；图内 device 内存分配能省 | 图外：Dynamo 只把**张量算子序列**抽成图；官方工程要求明确写着"动态信息**显式传入**……**而不是在模型内部临时生成 Python 标量**"、"KV Cache/常驻 buffer 应**预分配并原地更新**"（E3.4），反过来说明图外对象构造不在图的管辖范围。断图规则也说明：`.item()`、基于 Tensor 值的 `if/while` 会**断图**回 eager（E3.2/E3.4）。图内：静态 shape 的"模型级**内存编排**"在编译期完成（E2.1），因此中间张量不需要每步重新申请。 |

**附加观测项（不属于四层，但每步都要付）**：第 5 项 —— **Dynamo Guards 校验**。
官方 `compile_cache` 配图（E2.3）明确画出：即使**再次执行**（稳态），也要走 `Guards` → `Input 转换` → `图执行`。
GE 把前四层压到"每步一次"，但**压不掉 Guards**；Guards 失效还会触发重编译。

### 2.3 E2.2 下沉不是零成本：官方列出的四项"下沉头开销"

- **来源**：[深度解读昇腾CANN模型下沉技术，提升模型调度性能](https://www.hiascend.com/zh/developer/techArticles/20240715-1) §3.3"下沉头开销"（2024-07-15，访问 2026-09-24）。
- **机制（原文四级）**：
  1. 模型输入输出 Tensor 到内部 **InputData/OutputData 数据结构转换**（stage1_t）；
  2. 若 Feature Map 内存/输入输出内存有变更，**刷新相关联算子的地址**（stage2_t）；
  3. 若输入不支持零拷贝（如输入在 Host 内存），**下发异步拷贝 Task**（stage3_t）；
  4. **下发模型执行 Task**（stage4_t）。
- **官方量化**："以盘古 71B 增量推理模型为例（模型输入输出总个数约 **1600** 个，模型内节点总数约 **6300** 个，模型运行时，Feature Map 内存地址不变更，**10 个输入输出内存地址变更**），**当前模型下沉的头开销约 2 ms**。"
- **能力边界**：这 2 ms 是**每步固定成本**，与图内有多少算子无关；它由"输入输出元数据转换 + 地址刷新 + 模型执行 Task"构成。⇒ 对 1 ms 级 TPOT 目标，**这已经是一个必须先算清楚的预算项**。
- **证据强度**：**强**（官方技术文章，给出模型规模与量级）。

### 2.4 E2.3 稳态每步到底走什么：官方 `compile_cache` 配图

- **来源**：[模型编译缓存功能](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/compile_cache.md)（"图 1 max-autotune模式执行时间分布示意图"，访问 2026-09-24；图源 `raw.gitcode.com/Ascend/torchair/raw/26.0.0/docs/zh/figures/execution_time_1.png`）。
- **机制**：该图把 Ascend IR（GE）模式的时间线画成三段：
  - **首次执行**：`Dynamo编译` → `Guards` → `Ascend IR图编译` → `Input转换` → `图执行`；
  - **再次执行**：`Guards` → `Input转换` → `图执行`（Dynamo 与 IR 编译消失）；
  - **开启模型编译缓存**：`Time save`（覆盖 Dynamo 与 IR 编译）→ `Input转换` → `图执行`。
- **能力边界**：官方**自己承认**稳态 GE 路径每步仍有 `Guards` 与 `Input 转换` 两项 host 工作；"Input 转换"的官方定义（同仓 `appendix/cases/dynamic_static_graph` 与 npugraph_ex 编译缓存文档）是"**更新图内 input 类参数输入地址为图实际运行时的输入地址**"。
- **证据强度**：**强**（官方文档配图，非第三方解读）。

### 2.5 E2.4 动态 shape 下 GE 每算子要付多少层

- **来源**：[深度解读昇腾CANN动态Shape图调度加速技术](https://www.hiascend.com/developer/techArticles/20250911-1) §4（2025-09-12）。
- **机制（原文）**："在动态 Shape 模型的执行图中，**一个算子的下发被拆分为多个 Host 执行单元，每个执行单元称为 Host Kernel**……一个 AI Core 算子的下发会被拆分为 **InferShape、Tiling、AllocMemHbm、Launch** 等多个 Host Kernel。"
- **能力边界**：⇒ 动态 shape 路径下，**(b) op 派发与 (c) 张量元数据构造**并没有省掉，只是实现语言从 Python 变成了 GE 的 C++ Host Kernel；官方给出的优化方向是 Host 缓存（Tiling 缓存）、多核并发（三级流水、`MAX_RUNTIME_CORE_NUMBER=3`）、小 Shape 算子下沉到 Host 执行等——**这些都是在"省一部分"，而不是"抹除"**。
- **证据强度**：**强**。

### 2.6 E2.5 官方对"一次下发省 N 次"的量化表述

- **来源**：[动静子图拆分场景性能优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/static_model_ops_lower_limit.md)；[图内标定SuperKernel范围](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md)。
- **机制**：官方原文"如 **N 个算子聚合成一个静态子图后只需要一次下发**"；同时给出**反作用**："静态子图中算子数量越多，**单次图下发、图执行耗时越长**……静态子图中的算子数量并非越多越好，而是下发与执行的耗时相互抵消"。SuperKernel 同样定位为"与单算子下发相比……可以优化**任务调度的等待时间和调度开销**"。
- **能力边界**：**支持**（GE 静态图 / 静态子图），但这是"用一次更大的下发换 N 次小下发"，收益取决于"省下的派发"是否大于"多出的图下发/执行"。
- **证据强度**：**强**。

### 2.7 结论：把"完全抹除 host 开销"这句话拆开看

```text
eager 每步 host 成本 ≈ N_ops × (Python 处理 + Python→C++ + Tiling + 内存申请 + Launch)
GE 动态图每步成本   ≈ N_ops × (InferShape + Tiling + AllocMem + Launch 的 Host Kernel 链) + Guards + 图输入元数据
GE 静态下沉每步成本 ≈ 1 × 模型执行 Task + 图输入/输出元数据转换 + 地址刷新 + Guards
                      ↑ 官方量级参考：盘古71B ≈ 2 ms（E2.2）
图外 host 代码      ≈ 原样执行（GE 完全不管）                     ← prepare_input 就在这里
```

**能省**：device 侧 per-op launch（下沉后）；host 侧 per-op 的 InferShape/Tiling/内存申请（编译期定型）。
**不能省**：图外 Python 代码；Guards；图输入输出的元数据转换与地址刷新；动态 shape 路径下的逐算子 Host Kernel。
⇒ **"GE 整图下发能完全抹除 host 开销"是夸大。** 官方口径的上限是"大幅降低 Host 侧调度开销"（下沉）+ "调度性能持平或略优"（动态图 Host 调度）。

---

## 3. 能力边界（逐条：来源 / 机制 / 边界 / 强度）

### 3.1 能不能捕获 Python 控制流？

#### E3.1 数据依赖控制流：不支持（会断图）

- **来源**：[Dynamo导图功能](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/dynamo_export.md)（约束："受 Dynamo 功能约束，**不支持动态控制流 if/else**"）；
  [npugraph_ex 快速上手](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/npugraph_ex/quick_start.md)（"与 `torch.cuda.CUDAGraph` 原生接口功能类似，约束与其保持一致（如**不支持 stream sync、动态控制流**等）"）
- **机制**：Dynamo 编译期只能选择一条分支；分支条件依赖 Tensor **值**时无法在编译期确定 ⇒ graph break（断图），该段代码回落到 eager 每步执行。与值无关的控制流（如 `if x.is_prompt` 这类宿主布尔量、常量循环）会被**特化**进图，但特化条件会变成 **Guard**：值一变就重编译。
- **能力边界**：**不支持**（数据依赖 `if/while`）；**有条件支持**（非数据依赖控制流 → 编译期特化 + Guard）。
  官方给出的替代是**把控制流外提**：`cache_compile` 的官方示例正是把 prompt/decode 拆成两个函数，在 `forward` 里用宿主的 `if x.is_prompt` 分派到两张已编译图（见 §4.3）。
- **证据强度**：**强**（官方文档 + 官方示例）。

#### E3.2 图内 print / .item() 之类"看起来无害"的 Python：会断图

- **来源**：[图内Tensor打印功能](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/tensor_print.md)（"在图模式下，由于 Python 原生 `print` 函数会**触发断图（graph break）**，导致图模式下无法使用 print 观察图执行过程中的 tensor 信息"）；[PyTorch: torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)（数据依赖操作 `.item`、`.data_ptr`、数据依赖控制流都会 graph break）。
- **机制**：任何在 trace 期无法用张量算子表达的 Python 副作用，要么被折叠成常量，要么断图。
- **能力边界**：**不支持**（作为图内代码）；TorchAir 提供 `torchair.ops.npu_print` 作为"不断图的 print"替代。
- **证据强度**：**强**（TorchAir 官方 + PyTorch 官方）。

#### E3.3 GE IR 自己是有条件控制算子的（但 TorchAir 能否把 Python/torch.cond 转进去，官方未写）

- **来源**：[图编译多级优化选项](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/oo_level.md)（"死边消除：当图中存在**条件控制算子（如 If、Case 等）**时，会根据输入条件（cond）判断执行哪个分支"）。
- **机制**：Ascend IR / GE **有** If/Case 这类控制算子；死边消除 pass 会在 cond 编译期已知时删掉死分支。
- **能力边界**：**有条件支持**（IR 层面存在控制算子）；但**TorchAir 对 `torch.cond`/`torch.while_loop` 的转换支持，本文没有找到任何官方说明** ⇒ 见 §8"查不到的"。
- **证据强度**：IR 有控制算子 = **强**；TorchAir 侧映射 = **查不到**。

### 3.2 能不能捕获 host 侧 Python 对象的构造（22 字段 dataclass / numpy 算 cu_seqlens）？

#### E3.4 官方给出的判定标准是"动态信息显式传入"，不是"编进图"

- **来源**：[cann-recipes-infer / NPU 图模式优化原理](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/cann/zh/npu_graph_optimization.md) §3.2（官方样例仓，无页面日期，访问 2026-09-24；可在 raw 路径 `https://raw.gitcode.com/cann/cann-recipes-infer/raw/master/docs/cann/zh/npu_graph_optimization.md` 取到全文）。
- **机制（原文）**：
  - "**动态信息显式传入**：典型动态信息包括 `kv_len`、`position_ids`、`actual_seq_lengths_q`、`actual_seq_lengths_kv` 和 `is_prefill`。这些信息应由框架构造后**作为显式输入传给模型**，而**不是**在模型内部临时生成 Python 标量，或依赖隐式全局状态推导。"
  - "**KV Cache 与常驻 buffer 原地更新**：Decode 图复用要求关键输入的 shape 和地址尽量稳定。因此 KV Cache、attention mask、position buffer 等常驻数据应**预分配**，并在运行时**原地更新**。"
  - "**避免 Graph Break 写法**：① `tensor.item()`；② 基于 Tensor 值的 Python `if`/`while`；③ 在 `forward` 内部临时创建影响 shape 的控制分支；④ 根据 Python list/tuple **长度变化**切换图内控制流。"
- **能力边界**：
  - **不支持**：把"每步变化的 host 对象构造"当作可以被图省掉的成本 —— 官方要求**把它挪到图外**，因此它的成本也在图外；
  - **支持**：把它构造的**结果**（如 `actual_seq_lengths`）以 **Tensor / 固定长度结构化输入**喂进图。
- **证据强度**：**中**（官方样例仓工程文档；与 TorchAir 官方 API 文档口径一致）。

#### E3.5 dataclass 作为**输入**是被官方支持的（但构造发生在图外）

- **来源**：[模型编译缓存功能](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/compile_cache.md) / [cache_compile API](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/inference/cache_compile.md)。
- **机制**：官方 cache_compile 示例里直接用了
  ```python
  @dataclasses.dataclass
  class InputMeta:
      data: torch.Tensor
      is_prompt: bool
  ```
  并把 `InputMeta` 实例与 `List[torch.Tensor]` 作为 `torch.compile` 的入参；官方注释写着"InputMeta 为**仿照 vLLM 框架的入参结构**"。也就是说 **dataclass / List 作为"图输入的容器"是官方示例形态**。
- **能力边界**：**有条件支持**（作为输入容器）。官方**没有**任何地方说"图会捕获 dataclass 的构造过程"——示例中构造发生在被编译函数之外；且约束里明确要求 `func 必须能形成整图（必须支持 full graph）`、`缓存要与执行计算图一一对应，若重编译则缓存失效`。
- **证据强度**：**强**（官方示例代码）对"能否作为输入"；"能否捕获构造过程"= **查不到/否定**。

#### E3.6 纯算法（numpy 算 cu_seqlens）能进图吗？

- **结论**：**没有找到任何官方依据支持"host 侧 numpy 算法被 GE 图捕获并消除"**。
  实务上的三条路（每条都有代价）：
  1. 留在图外 → 每步都执行（成本不消失）；
  2. 改写成**张量算子**（cumsum/scatter 等）交给图 → 需要对应 ATen 算子在官方支持清单里（E3.14）；且 Device 侧小算子的派发/同步成本可能比 host 更贵（官方"小 Shape 算子计算优化"正是因为这个：把小 Shape 的 Gather/Concat **保留在 Host 执行**反而更快，见 §5.1 第 4 行）；
  3. 预计算成**常量** → 只有值不变才行，值一变化 Guard 失效、重编译。
- **证据强度**：**强**（对"没有官方依据"这一否定命题，依据是官方支持清单 + 官方断图/守卫规则 + 官方小 Shape 优化说明）。

### 3.3 动态 shape / 动态 batch 的支持

#### E3.7 `dynamic=True/False` 的语义（官方表格）

- **来源**：[动/静态图概念](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/dynamic_static_graph/concepts.md)（"Dynamo 中动/静态图概念"表 1）。
- **机制**：

  | `dynamic` | user_input tensor | `mark_static` 的 tensor | parameter/buffer | scalar 输入 |
  |---|---|---|---|---|
  | `False` | **固定形状** | 固定形状 | 固定形状 | **固定值（常量）** |
  | `True` | **符号形状** | 固定形状 | 固定形状 | **符号** |

- **能力边界**：**支持动态 shape 图**，但"动态"的代价是调度方式变成 Host 调度（见 E2.1/E3.8）。
- **证据强度**：**强**。

#### E3.8 GE 侧的动/静态与调度方式的绑定（"能不能下沉"的判定）

- **来源**：同 E3.7 文档"GE 动/静态图概念"表 2。
- **机制（原文）**：
  - "对于**所有输入 tensor shape 不固定**的图，称为**动态 shape 图**。动态 Shape 图在执行时才能确定 Shape，完成 Tiling 计算，**只能采用 Host 调度**。"
  - "对于**所有输入 tensor shape 固定**的图，称为**静态 shape 图**。静态 shape 图中的算子一般都能采用**下沉调度**。"
  - "特殊情况下，静态 shape 图中存在**值依赖算子**，该算子**默认使用 Host 调度**。"（即：一个值依赖算子就能让"部分算子下沉、部分 Host 调度"，而不是纯下沉）
- **能力边界**：**支持**动态 shape；**不支持**动态 shape 的整图下沉。
- **证据强度**：**强**。

#### E3.9 动态 batch 的官方解法：**分档（gear）**，而不是无限动态

- **来源**：
  - [动态shape图分档执行功能](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/dynamic_gears_merge_policy.md)（TorchAir）；
  - [set_dim_gears](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/inference/set_dim_gears.md)；
  - GE 侧：[动态输入shape范围 / 动态维度（ge 仓文档）](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/graph_dev/more_features/dynamic_shape.md)、[aclgrphBuildModel 支持的配置参数（DYNAMIC_DIMS/INPUT_SHAPE）](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendgraphapi/atlasgeapi_07_0143.html)、[options 参数说明（ge.inputShape / ge.dynamicDims）](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/graphdevg/atlasgeapi_07_0119.html)。
- **机制**：
  - TorchAir 侧：`torchair.inference.set_dim_gears(t, {dim: [档位...]})` + `torch.compile(..., dynamic=True)`；可用 `config.inference_config.dynamic_gears_merge_policy = "zip" | "product"` 控制档位组合；编译日志里能看到 `ge.dynamicDims = 2,2;4,4`、`ge.dynamicNodeType = 1`、`ge.inputShape = arg1_1:-1,2;...`。
  - GE 侧：`ge.inputShape`（用 `-1` 标动态维、用 `8~20` 写**范围**）+ `ge.dynamicDims`（枚举档位）+ `ge.dynamicNodeType`；分档编译成 `Case + N 子图`，运行时按输入 shape 选档。
- **能力边界（硬约束，官方列出）**：
  - 分档**仅适用于 GE 图模式**，且**只适用于整图优化场景**；需与 `dynamic=True` 搭配；参与分档的 tensor 不能是私有格式（FRACTAL_NZ/NC1HWC0 等）；
  - **档位数量约束**：CANN 8.5 文档为 `(1,100]`，**建议 3~4 档**；TorchAir `set_dim_gears` 文档写"生成的总档位数量不超过 100"；**档位值不能包含 0 或 1**（因为动态 FX graph 的符号化范围是 `[2,∞)`，dim 为 0/1 时不会命中动态图、需要重新成图）；
  - 若运行期 shape 不在档位内 → **编译或执行报错**（不是自动回退；GE 内部另有"匹配不到档位则走动态 shape 图"的渐进降级设计，但 TorchAir 文档写的是报错）。
- **证据强度**：**强**（TorchAir + GE 双侧官方文档）。

#### E3.10 关于 `input_shape_ranges` 这个名字

- **核实结果**：官方文档里**没有** `input_shape_ranges` 这个选项名。真实存在的对应机制是：
  - `ge.inputShape` 用**范围写法**（如 `"input_name1:8~20,3,5,-1"`）来表达"shape 范围（动态 shape）"；
  - `ge.dynamicDims`（或 `aclgrphBuildModel` 的 `DYNAMIC_DIMS`）表达"分档（静态 shape 分档）"；
  - 旧选项 `ge.exec.dataInputsShapeRange` 已被官方标注**废弃，请勿使用**。
- **来源**：[options参数说明](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/graphdevg/atlasgeapi_07_0119.html)（含 `ge.exec.dataInputsShapeRange` "该参数已废弃，请勿使用"）；[aclgrphBuildModel 配置参数](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendgraphapi/atlasgeapi_07_0143.html)；[动态输入shape范围（ge 仓）](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/graph_dev/more_features/dynamic_shape.md)。
- **证据强度**：**强**。

### 3.4 静态输入要求：地址必须固定吗？每步变化的标量/长度怎么传？

#### E3.11 输入张量地址：GE **不要求**固定，但"刷新地址"是每步成本；可用开关省掉它

- **来源**：[固定权重类输入地址功能（Ascend IR）](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/frozen_parameter.md)；E2.2（下沉头开销第 2 项）。
- **机制**：
  - `frozen_parameter=False`（默认）：图执行时**不**固定权重类输入地址 ⇒ 每次执行要刷新地址（官方原话："可以开启本功能**缩短图下发时间**，提升下发性能"）；
  - `frozen_parameter=True`：固定 Parameter 类（权重）输入地址；PyTorch ≥ 2.6 还可用 `torch._dynamo.mark_static_address` 把 KV cache 这类地址不变的 tensor 也纳入。
- **能力边界**：**不要求固定**（有条件）；固定能省"地址刷新"这一类开销，但**只对地址本来就不变的输入有效**。
  对比参照（ACLGraph 更严）：npugraph_ex 官方文档写"aclgraph 是**基于固定内存地址执行**"，`mutated_inputs`（如 KV cache）地址变化会**触发 Recapture**（重捕获）（见 §7 差异矩阵）。
- **证据强度**：**强**。

#### E3.12 每步变化的标量/SymInt[] 怎么传：官方给了 4 条编译路径

- **来源**：[典型问题：模型中存在 FA 算子，如何执行整图静态下沉调度](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/dynamic_static_graph/typical_issues.md)（以 `npu_fused_infer_attention_score` 的 `SymInt[] actual_seq_lengths` 为例）。
- **机制（官方原文归纳）**：

  | 路径 | `dynamic` | 输入标记 | Tiling | GE build 图 | SymInt[] 变化时 |
  |---|---|---|---|---|---|
  | 路径1 | `False` | — | 不影响 | **静态图**（SymInt[] 被编成 `ge.Const`） | **触发重新编译** |
  | 路径2 | `True` | 不 mark_static | 不影响 | 动态图 | 不重编译，但**无法下沉**（Host 调度） |
  | 路径3 | `True` | `mark_static` tensor | **不开** | 动态图（FA 走 Host 调度） | 不重编译，FA 仍 Host 调度 |
  | 路径4（**官方指定解**） | `True` | **`mark_static` tensor** | **开 `tiling_schedule_optimize`** | **静态图** | **不重编译，且整图静态下沉** |

  官方原文对路径4的结论："SymInt[] 编译时被泛化为符号，如果模型运行时 SymInt[] 的值发生变更，**不会触发重新编译，仍然使用首次编译的结果**。因此，为了实现包含 FA 算子的**整图静态下沉，只能选择路径4**。"
- **能力边界**：**有条件支持**。三条件缺一不可：`torch.compile(dynamic=True)` + `torch._dynamo.mark_static(inp)`（所有 tensor 输入）+ `config.experimental_config.tiling_schedule_optimize=True`。
- **证据强度**：**强**（官方案例文档，含定位流程）。

#### E3.13 更彻底的做法：把长度直接做成 **Tensor 图输入**（TorchAir 为 FIA 定制）

- **来源**：[torchair.ops.npu_fused_infer_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/ops/npu_fused_infer_attention_score.md) 与 [\..._v2](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/ops/npu_fused_infer_attention_score_v2.md)。
- **机制**：官方定制接口把 `actual_seq_lengths` / `actual_seq_lengths_kv` / `actual_seq_qlen` / `actual_seq_kvlen` 从 **`SymInt[]`（int 数组）改为 `Tensor`（int64）**，并解释动机：
  "该接口在图模式场景下，如果开启 Tiling 调度优化功能，模型中 `actual_seq_length` 类参数会存在**从 Host 到 Device 的拷贝开销**，模型执行性能会下降。为此 TorchAir 提供了相应的定制化接口……Tiling 下沉时 AI CPU 中的 Tiling 分核和 AI Core 中的 Kernel 计算均在 Device 侧，**直接传入 Device 可以降低 Host 到 Device 拷贝**。"
- **能力边界**：**支持，但仅 GE 图模式**——官方约束写明"**本接口只支持图模式，不支持 Eager 模式下调用**"。
- **证据强度**：**强**（官方 API 文档，且动机写得很清楚）。

### 3.5 算子覆盖

#### E3.14 ATen 算子：有官方清单，清单外需要补 Converter

- **来源**：[支持的ATen API清单](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/aten_api.md)（153 行表格）；[算子Converter支持度导出功能](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/converter_export.md)。
- **机制**：官方原文："如果自定义模型用到的 ATen API **不在表1**，说明对应的 API 能力**可能不完备**，用户需根据实际情况进行 Converter 适配实现算子入图"；调试时可用 `config.debug.fx_summary.type = "csv"` 导出"算子名 / Converter 支持度（已支持 / 部分支持 / 未实现）/ 调用次数"。
- **能力边界**：**有条件支持**（清单内支持；清单外需自研 Converter 或改模型）。
- **证据强度**：**强**。

#### E3.15 自定义算子：**必须**是 Ascend C 工程化的 `aclnnXxx`，Kernel 直调不可入图

- **来源**：[自定义算子入图概述](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/custom_op_graph/overview.md)；[In-place算子开发和入图样例](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/custom_op_graph/in_place_op_cases.md)。
- **机制（原文）**："使用 TorchAir **max-autotune** 模式时，需要确保算子 NPU 实现是以 **Ascend C 算子工程化方式**开发的 `aclnnXxx` 接口，**以 Kernel 直调方式开发的 NPU 实现不会生成 Ascend IR 注册逻辑，没有对应的 Ascend IR，无法完成 PyTorch 算子到 Ascend IR 的转换**。"
  另外：非 In-place 算子只需 Meta 推导；In-place 算子需要函数化（官方说 TorchAir 已支持自动化函数化）。
  还有一条容易踩的坑：**以 fallback 形式下发的算子会断图**——官方另一篇文档写"在静态 shape 场景下，`aclnnXX` 接口包含 Host 操作，**无法以整图形式下发，需要断图后以子图形式下发**，这可能会对性能产生较大影响"（[基于fallback形式下发算子](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/graph/graphdevg/atlasag_25_0090.html)）。
- **能力边界**：**有条件支持**（工程化 aclnn 算子）；**不支持** Kernel 直调算子的 IR 入图。
- **证据强度**：**强**。

#### E3.16 Triton 算子：GE 有"Triton 入图"能力，但官方只给 TensorFlow 路径，且实现是 Host 侧 launch

- **来源**：[Triton入图 / 开发流程（CANN 商用版 9.0.0）](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/graphdevg/atlasag_25_0106.html)（同文见 [ge 仓](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/graph_dev/custom_operator_into_graph/triton_into_graph.md)）。
- **机制**：官方描述"将 Triton 算子集成到 GE 图中，可复用 GPU 训练时开发的 Triton 自定义算子"；流程是 ① `triton.compile` 把 kernel 编成 **`*.npubin`**；② "开发**入 TensorFlow 图**的交付件。**当前仅支持 TensorFlow 框架**"；③ 在 GE 图内注册 `EagerExecuteOp`，其 `Execute()` 里 **`aclrtLaunchKernelWithHostArgs(...)`** 启动核函数。
- **能力边界**：
  - **支持（有条件）**：GE 图内可以承载 Triton kernel，但走的是 **EagerExecuteOp + Host 侧 launch**，不是把 Triton 编成 Ascend IR 算子；且官方文档只覆盖 **TensorFlow** 前端。
  - **PyTorch 侧的 Triton-Ascend 是另一条路**：官方 PyTorch 编译模式文档列的是 `torch.compile(backend="inductor")`（默认 Triton 模式，基于 Triton-Ascend 生成融合算子），与 torchair 的 GE 后端是**互斥的后端选择**（[PyTorch编译模式（torch.compile）](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/pytorch_compilation_mode.md)）。
  - **未查到**：TorchAir GE 模式直接消费 Triton kernel 的官方说明 ⇒ 见 §8。
- **证据强度**：**强**（对 TF 路径与 inductor 路径）；**查不到**（对 torchair+PyTorch+Triton）。

### 3.6 图编译时间与稳态负载是否值得

#### E3.17 官方只有定性描述，**没有任何绝对时长数字**

- **来源**：E2.3 的 `compile_cache` 文档（含配图）。
- **机制（原文）**："torch.compile 是一种即时编译器（JIT），**成图首次编译时间通常较长**，而大模型推理场景对时延敏感，因此优化首次编译时长显得尤为重要"；"成图编译涉及**两段耗时**，一段是 **Dynamo 的编译耗时**，另一段是基于 Dynamo 编译出的 FX 图进行再处理的耗时"（GE 模式下即 Ascend IR 图编译）。
  官方给出的应对手段：`torchair.inference.cache_compile(..., ge_cache=True)` 缓存 Ascend IR 编译结果（默认 `ge_cache=False`）。
- **能力边界**：⇒ **decode 稳态负载**理论上完全可以把编译成本放到 warmup 之外（官方推荐做法，见 E3.18），但**"值不值得"取决于图能否被稳定复用**：任何 shape/地址/控制流/代码变化都会让 Guard 失效、缓存失效（官方 cache 失效规则）。
- **证据强度**：**强**（对"耗时长"这一定性结论）；**没有数字**（见 §5.3）。

#### E3.18 官方推荐的落地顺序（把编译放在 warmup）

- **来源**：[cann-recipes-infer / NPU 图模式优化原理](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/cann/zh/npu_graph_optimization.md) §2.1、§4.1、§4.3。
- **机制**：warm-up 阶段先跑一遍 Decode 触发编译并缓存，正式推理只走 `Guards 校验 + Input 处理 + Replay`；验收标准是"正式推理 decode 阶段直接复用 warmup 编译的图、日志中**无 `recompile` 标识**"。
- **能力边界**：**支持**（这是官方样例仓的标准做法）。
- **证据强度**：**中**（官方样例仓）。

### 3.7 怎么验证"图真的下沉了"

#### E3.19 官方判据：dump GE build 图，看 `_graph_unknown_flag`

- **来源**：[动/静态图展示](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/dynamic_static_graph/presentation.md)（"GE 动/静态图展示"节）。
- **机制**：dump GE 图（`DUMP_GE_GRAPH`）后，看 build 图 txt：**"如果存在 `_graph_unknown_flag` 属性值且取值为 true，则为非完全静态下沉调度，否则为完全静态下沉调度"**。
- **能力边界**：**支持**（官方给出的可操作判据）。这条对本项目很有用：**"GE 能不能整图下发"不是一个需要争论的问题，而是一条 dump 出来就能看的属性。**
- **证据强度**：**强**。

#### E3.20 判据二：跑一次 profiling 看是否有 Host 侧下发/空隙

- **来源**：[图模式性能分析](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/performance_cases.md)；E2.1 技术文章对 Host Bound/Device Bound 的 profiling 特征描述。
- **机制**：官方明确"Host Bound 网络，Host 任务下发速度较慢，Device 侧需等待 Host 下发任务"。下沉成功的典型特征是"Device 空隙大幅减少"。
- **证据强度**：**强**。

---

## 4. TorchAir host 侧 API 面（含对几个 API 名的核实）

### 4.1 官方 API 清单（GE 图模式）

- **来源**：[GE图模式API列表](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/api_list.md)、[整体说明（API参考）](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00079.html)、[CompilerConfig类](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/torchair/compiler_config.md)。
- **机制**：官方 API 命名空间只有这几个（**没有 `torchair.npu`**）：

| 命名空间 | 接口 | 作用 |
|---|---|---|
| `torchair` | `CompilerConfig` | 图编译配置（`mode` 在这里） |
| | `get_npu_backend(compiler_config=...)` | 取得 GE 图编译后端，传给 `torch.compile(backend=...)` |
| | `get_compiler` | 取得 NPU 图编译器 |
| | `dynamo_export` | 导出离线 `.air` 图（**不支持动态控制流 if/else**） |
| | `patch_for_hcom` | 集合通信入图补丁 |
| | `register_fx_node_ge_converter` | 注册自定义算子的 GE Converter |
| | `register_replacement` | 注册自定义算子融合规则 |
| | `use_internal_format_weight` | 权重转内部私有格式 |
| `torchair.ge` | `Tensor` / `TensorSpec` / `Const` / `Cast` / `Clone` / `custom_op` / `DataType` / `Format` | 写 Converter 用的构图原语 |
| `torchair.inference` | `cache_compile(func, config=..., dynamic=..., cache_dir=..., ge_cache=...)` | 编译缓存（含 GE 编译结果缓存） |
| | `readable_cache` | 读出缓存内容 |
| | `set_dim_gears` | 设置动态 shape **档位** |
| `torchair.ops` | `npu_print` | 不断图的 print |
| | `npu_fused_infer_attention_score` / `_v2` | FIA 定制接口（`actual_seq_lengths*` 收 **Tensor**，仅图模式） |
| | `record` / `wait` | 图内跨流时序 |
| `torchair.scope` | `npu_stream_switch` / `npu_wait_tensor` / `super_kernel` / `limit_core_num` / `op_never_timeout` / `data_dump` | 图内多流、SuperKernel、限核等 |
| `torchair.llm_datadist` | `create_npu_tensors` | 分布式 KV 传输 |

### 4.2 对用户提到的几个 API 名逐个核实（**重要：请勿沿用错误名字**）

| 用户提到的名字 | 核实结果 | 真正对应什么 |
|---|---|---|
| `torchair.npu` | **不存在**。官方 API 列表里没有这个命名空间 | 想要 ACLGraph 模块：`torch.npu.npugraph_ex`（新）或 `backend="npugraph_ex"`；想要 GE：`torchair.get_npu_backend()` |
| `compile` | 不是 torchair 的接口 | 用 PyTorch 原生 `torch.compile(model, backend=torchair.get_npu_backend(...))` |
| `cache_compile` | ✅ 存在 | `torchair.inference.cache_compile(func, *, config, backend, dynamic, cache_dir, global_rank, tp_rank, pp_rank, ge_cache, **kwargs)`（见 §4.1） |
| `register_frozen_parameter` | **官方 TorchAir 文档里查不到这个名字** | 真实开关是配置项 `config.experimental_config.frozen_parameter = True`（GE）或 `options={"frozen_parameter": True}`（npugraph_ex）；配套接口是 `torch._dynamo.mark_static_address` |
| `inplace` | **没有这个接口名** | 相关的是：npugraph_ex 的 `options={"inplace_pass": ..., "input_inplace_pass": ...}`（把 out-of-place 算子替换为 in-place，**仅 npugraph_ex 后端**）；以及 GE 侧的 In-place **算子函数化**要求 |
| `static_input` | **TorchAir 文档里查不到**；只在 `torch_npu/npu/_graph_tree.py`（ACLGraph 树 / `torch.npu.graphs.make_graphed_callables` 的移植实现）里作为 **`static_input_idxs`** 出现 | GE 侧的对应概念是"图输入地址固定"，用 `frozen_parameter` + `mark_static_address`；张量"形状静态化"用 `torch._dynamo.mark_static` |
| `get_input` / `update` | **TorchAir 文档里查不到** | 真正相关的是：ACLGraph 的 `torch.npu.graph_task_update_begin/end`（任务参数更新，E1.2）；npugraph_ex 的 "Host Tiling 参数刷新"；GE 侧的"Input 转换"（内部步骤，无用户 API） |

> **结论**：`static_input` / `inplace` / `register_frozen_parameter` / `get_input` / `update` 这套名字**不是 TorchAir GE 的官方 API 面**，来源大概率是别的框架/别的层（或 AI 生成内容的臆造）。写方案时请使用上表的真实名字。

### 4.3 官方示例里"如何把每步变化的输入喂进已编译的图"

**形态 A：把变化的值改成 Tensor 图输入（GE 首选，官方推荐）**

```python
# 官方 API 文档示例（E3.13）
actual_seq_lengths = torch.tensor([50]).npu()          # int64 Tensor，不是 list
return tng.ops.npu_fused_infer_attention_score(
    q, k, v, actual_seq_lengths=actual_seq_lengths, ...)   # 仅 GE 图模式可用
```

**形态 B：`dynamic=True` + `mark_static` + `tiling_schedule_optimize`（保留 list/scalar 语义，但仍要下沉）**

```python
# 官方案例文档路径4（E3.12）
inp = torch.randn(100, 128).npu()
torch._dynamo.mark_static(inp)                          # 形状静态化（地址另见 mark_static_address）
config = CompilerConfig()
config.experimental_config.tiling_schedule_optimize = True
model = torch.compile(model, backend=tng.get_npu_backend(config), dynamic=True, fullgraph=True)
```

**形态 C：prompt/decode 拆两张图，用**图外**的宿主分支分派（官方 cache_compile 示例）**

```python
# 官方示例（E3.5）：把每步变化的对象"从图外传入"，而不是在图内构造
def forward(self, x: InputMeta, kv: List[torch.Tensor]):
    if x.is_prompt:                     # ← 宿主布尔量，图外分派
        return self.cached_prompt(x, kv)
    return self.cached_decode(x, kv)    # ← 两个各自独立的编译产物
```

**形态 D：形状按档位收敛（batch 变化时）**

```python
# 官方示例（E3.9）
model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
torchair.inference.set_dim_gears(npu_input0, {0: [2, 4]})   # 档位：batch=2 或 4
```

---

## 5. 公开性能数字（有则列，无则明说）

### 5.1 官方数字（**强**）

| # | 场景 | 数字 | 口径 | 来源 |
|---:|---|---|---|---|
| 1 | LLaMA-7B **Decoding**，Host 调度 vs **模型下沉** | 端到端 **-18 ms**；**吞吐 +37%**；Device "计算时间与空隙接近 1:1" → "空隙大幅减少" | 官方技术文章，未给 batch/序列/卡型 | [20240715-1](https://www.hiascend.com/zh/developer/techArticles/20240715-1) |
| 2 | 盘古 71B 增量推理，下沉**头开销** | ≈ **2 ms/步**；该模型 I/O 约 **1600** 个、图内节点约 **6300** 个、每步仅 **10** 个 I/O 地址变更 | 官方技术文章 | 同上 §3.3 |
| 3 | 动态 shape Host 调度，Tiling 缓存（默认开） | pangu/llama2 上 **Host 侧 Tiling 开销降低 50%+**；参数越大收益越明显；pangu38B 量化网络端到端提升最明显（未给具体数） | 官方技术文章 | [20250911-1](https://www.hiascend.com/developer/techArticles/20250911-1) §4.1 |
| 4 | 小 Shape 算子保留 Host 执行（默认开） | LLaMA2：E2E **1.062 s → 1.009 s**，**吞吐 +5%**；命中算子约 **650+** 个（Pack/Gather/Concat 等） | 官方技术文章 | 同上 §4.3 |
| 5 | GE 模式的阶段构成 | **有图无绝对值**：官方 `compile_cache` 图 1 展示"首次执行 = Dynamo 编译 + Guards + Ascend IR 图编译 + Input 转换 + 图执行"，`Dynamo 编译`与`Ascend IR 图编译`是"耗时占比最大环节" | 官方文档配图，坐标轴无数值 | E2.3 |

### 5.2 第三方 / 社区数字（**中/弱**）

| # | 场景 | 数字 | 可信度评估 |
|---:|---|---|---|
| 6 | vLLM + ACLGraph 的 decode 阶段（**不是 GE**） | Qwen3-30B：eager 150.33 ms → fullgraph 21.9 ms → fullgraph+npugraph_ex 20.2 ms；DeepSeekV3.1：185.6 → 45.0 → 41.2 ms | **中**（CANN 开发者社区文章 [npugraph_ex：CANN aclGraph 的图模式样板间](https://cann.csdn.net/69afcbb00a2f6a37c59646a1.html)，无发布日期，未标卡型/配置） |
| 7 | 大 MoE 模型 GE 图编译耗时 | 自称 Mixtral 8×7B "GE 图编译跑了 15 分钟"、"fast 模式 15 分钟→4 分钟" | **弱**：文章称由 `GE_FUSION_PASS_MODE` / `GE_WARMUP_SHAPES` / `GE_MEM_PROFILING` 等环境变量控制，但这些变量**在官方文档中均未找到**，疑为生成式内容。**不建议引用**（见 §8）。 |
| 8 | vLLM-ascend torchair 集成的冗余开销 | "每次 decode 都存在冗余的 cache_compile lookup 动作"，现场规避后**实测收益 4%** | **中**（TorchAir 仓库 issue [IC3L9G](https://gitee.com/ascend/torchair/issues/IC3L9G)，2025-04，关联 vllm-ascend issue #588） |

### 5.3 明确"没有找到"的数字（宁缺勿编）

1. **GE 图模式 vs eager，同一个 LLM、同一口径的 host 侧开销对比**（官方没有；只有 LLaMA-7B 下沉 vs Host 调度的 E2E/吞吐）；
2. **GE 首次编译的绝对时长**（官方只有"通常较长"；没有模型规模 × 卡型 × 时长的任何数字）；
3. **GE 稳态每步 host 时间的分解**（官方只有"下沉头开销 ≈2 ms"这一个含模型规模的数字）；
4. **在 910 A3（A3）上 GE vs ACLGraph 的头对头数字**（没有找到任何公开数据）；
5. **TorchAir `cache_compile` 命中/未命中的耗时差异百分比**（官方仅说明用途，无数字）；
6. ACLGraph `graph_task_update_*` 每算子每步的开销量级（官方只有"更新任务比单独下发任务更耗时"的定性描述）。

---

## 6. 已知限制与坑

### 6.1 官方明确列出的不支持项

| # | 限制 | 适用范围 | 来源 |
|---:|---|---|---|
| 1 | **动态控制流 if/else** 不支持 | `dynamo_export`（离线导图）；npugraph_ex 同样继承 CUDAGraph 约束（不支持动态控制流、stream sync） | E3.1 |
| 2 | **随机数算子**（randn/bernoulli/dropout 等）不能用于 `cache_compile` 缓存；npugraph_ex 也不支持随机数算子 capture | GE cache / ACLGraph | [compile_cache](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/compile_cache.md)、[overview](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md) |
| 3 | 每个进程**只支持 1 张 NPU 卡**（不支持一进程多卡） | PyTorch 图模式整体约束 | [overview](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md) |
| 4 | 以 **Kernel 直调**方式开发的自定义算子**无法生成 Ascend IR**，不能进 GE 图 | GE max-autotune | E3.15 |
| 5 | fallback 下发（`EnableFallBack`/`aclnn_only`）在静态 shape 下**无法整图下发**，必须断图 | GE | E3.15 |
| 6 | **档位值不能是 0 或 1**、总档位 ≤100、运行期 shape 不在档位内会报错 | GE 分档 | E3.9 |
| 7 | 分档要求 tensor **不能是私有格式**（FRACTAL_NZ/NC1HWC0 等） | GE 分档 | E3.9 |
| 8 | `tiling_schedule_optimize`（Tiling 下沉）**仅支持静态 Shape 模型**，且**当前只支持融合算子**（例如 FIA、IFA） | GE | [Tiling调度优化功能](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/tiling_schedule_optimize.md) |
| 9 | SuperKernel 仅 GE + **静态图**；遇到不可融合算子会截断成多段 | GE | [super_kernel](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md) |
| 10 | `set_dim_gears` 不支持对同一 tensor 设置两次不同档位；首次执行必须设置档位，否则后续不能补 | GE 分档 | E3.9 |
| 11 | npugraph_ex/aclgraph：不支持反向 capture、随机数 capture、动态控制流、stream sync；**aclgraph 本身不支持动态 shape**，shape 变化会**重新捕获**（有 `capture_limit`，默认 64 次后回退 eager） | ACLGraph | [capture_limit](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/npugraph_ex/basic/capture_limit.md) |
| 12 | ACLGraph 任务是**基于固定地址**的；输入地址变化会对 `mutated_inputs` 触发 Recapture，对 `user_inputs` 靠 clone 兜底（有内存代价） | ACLGraph | [aclgraph间内存复用](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/npugraph_ex/basic/memory_reuse.md) |
| 13 | ACLGraph 任务更新：**更新任务比单独下发任务更耗时**；任务组内**任务数量与类型必须严格一致**；**不支持多线程/多 Stream 并发更新**；单 Device 同时更新任务上限 1024×1024 | ACLGraph | [任务更新](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0032.html) |

### 6.2 "不是错误，但不满足就没收益"的工程约束

| # | 约束 | 后果 |
|---:|---|---|
| 1 | 稳态必须**不重编译**（Guards 全命中） | 一旦重编译，收益被编译时间吃掉（官方样例仓验收标准：日志无 `recompile`） |
| 2 | KV cache / mask / position buffer 要**预分配 + 原地更新** | 否则 shape/地址变化 → Guard 失效 / Recapture |
| 3 | 动态信息要**显式传入**，不要在 forward 内临时造 Python 标量 | 否则断图或每组输入重编译 |
| 4 | Prefill/Decode 要拆开（官方样例仓：Prefill 保持 eager，Decode 走图；官方 TorchAir 也建议拆 func） | 否则 prefill 的动态 shape/控制流污染 decode 图 |
| 5 | GE 的 Tiling 下沉需要 `tiling_schedule_optimize` + `dynamic=True` + `mark_static` 三件套 | 缺一个就退回 Host 调度（E3.12） |
| 6 | 编译缓存对**模型代码/输入规格/编译配置/cache_dir** 任一变化都失效；CANN 跨版本不保证兼容 | 需要清理重建 |

### 6.3 社区反馈（中）

| # | 反馈 | 出处 | 说明 |
|---:|---|---|---|
| 1 | vLLM-ascend 维护者（2026-05-27）："**The torchair graph mode has been sunset. In the new version, the aclgraph graph mode is used.**" | [vllm-ascend issue #588](https://github.com/vllm-project/vllm-ascend/issues/588)（该 issue 被标 `wontfix` 关闭） | 这是 vllm-ascend **集成方向**的表态，不等于 GE 能力消失；但说明"指望 vllm-ascend 短期内置 GE"是不现实的 |
| 2 | 用户实测"0.10.0rc1 torchair 还不支持 Qwen 系列；目前只支持盘古、DeepSeek、Kimi"；另有维护者回复"现在默认是 acl graph，**使用 ge 图模式可能有提升**" | [vllm-ascend issue #2310](https://github.com/vllm-project/vllm-ascend/issues/2310)（2025-08） | **弱-中**：社区问答，非官方能力矩阵；但反映了模型覆盖的现实约束 |
| 3 | TorchAir cache_compile 每次 decode 有冗余 `lookup`；规避后 **+4%** | [TorchAir issue IC3L9G](https://gitee.com/ascend/torchair/issues/IC3L9G)（2025-04，状态 DONE） | 说明 GE 路线的 host 侧"残余成本"是可观测、可优化的 |
| 4 | 第三代 MindIE-SD 仓库的编译后端对比（官方）指出：`torchair_ge` 会"完全绕过 aot_autograd → 无 functionalization → 无 `_to_copy` → 无 Copy 膨胀"，在 Wan2.2（910B）上 timed 推理 7007 ms(no-compile) / 7023 ms(torchair_ge) / 7632 ms(default) | [MindIE-SD backend-comparison.md](https://gitcode.com/Ascend/MindIE-SD/blob/master/.agents/skills/compilation-dev/references/backend-comparison.md) | **中**：官方仓库文档，但场景是扩散模型（非 LLM 稳态 decode），不能外推到 TPOT |

### 6.4 工程风险清单（本文的归纳，非官方原文）

1. **收益与"能不能稳定复用一张图"强绑定**：只要 prepare_input 产出的 shape/长度/地址每步变化，GE 就会在"重编译 / Host 调度 / 反复 Recapture"里选一个。
2. **下沉不等于零成本**：每步固定 2 ms 量级（官方 71B 数据）在 1 ms TPOT 目标下是硬预算。
3. **把 host 计算改成 device 张量未必更快**：官方"小 Shape 算子优化"恰恰说明小 Shape 的 device 派发可能比 host 更贵（§5.1 第 4 行）。
4. **自定义/非标准算子有硬门槛**：Kernel 直调不行、Triton 只有 TF 路径、ATen 清单外要自研 Converter。
5. **算子覆盖与模型覆盖是两件事**：即使算子都能入图，也要逐模型验证（社区反馈 Qwen/Kimi 支持滞后）。

---

## 7. 与 ACLGraph（当前 vllm-ascend 用的）的差异矩阵

> 左列 GE = `mode="max-autotune"`（Ascend IR）；右列 ACLGraph = `torch.npu.graph` / `aclmdlRICapture*`，工程入口是 `backend="npugraph_ex"` 或 `mode="reduce-overhead"`。

| 维度 | **GE / Ascend IR（max-autotune）** | **ACLGraph（npugraph_ex / reduce-overhead）** |
|---|---|---|
| 核心机制 | FX → **Ascend IR** → GE 编译（图优化/融合/内存编排/下沉） | **Capture & Replay**：把 Stream 上的任务抓进 `aclmdlRI` |
| 是否需要"算子能转成 IR" | **需要**：ATen 必须在清单内或有 Converter；自定义算子必须 Ascend C 工程化 | **不需要**：只要能 eager 跑通（官方：npugraph_ex 模式下"只需要保证 Torch 算子能在 Eager 模式下正常工作"） |
| 每步 host 入口 | GE 图执行接口（`ExecuteGraphWithStreamAsync` 一类）+ Guards + Input 转换 +（下沉后）1 个模型执行 Task | `aclmdlRIExecuteAsync`（`NPUGraph::replay`）+ 每步 `graph_task_update_begin/end` 刷新参数 + ExternalEvent 排序 |
| 动态 shape | **支持**（`dynamic=True` 符号化；GE 侧还有 `ge.inputShape` 范围 + `dynamicDims` 分档）；但动态 shape 图**只能 Host 调度** | **aclgraph 本身不支持动态 shape**：shape 变化会**重新捕获**（`capture_limit` 默认 64 次后整体回退 eager） |
| 输入地址 | **不要求固定**；地址变化在"下沉头开销"里刷新（可被 `frozen_parameter` 削弱） | **基于固定地址执行**；`mutated_inputs` 地址变化触发 Recapture，`user_inputs` 靠 clone 复用内存池 |
| 每步变化的算子参数 | 通过 **Tensor 图输入**（官方 FIA 定制接口）或 `dynamic=True` 符号输入；**无 per-op update API** | 通过 **`graph_task_update_begin/end`** 显式重发算子；官方警告"更新任务比单独下发任务更耗时"，且任务组**数量/类型必须一致**、不支持多线程多 Stream 并发更新 |
| 每步固定成本 | 下沉头开销（71B ≈ **2 ms**，与 I/O 个数强相关）+ Guards | Replay + 所有需更新算子的 update + Guards；官方无公开量级 |
| 图内优化空间 | **大**：算子融合、常量折叠、死边消除、Tiling 下沉、图内多流（`npu_stream_switch`/`npu_wait_tensor`）、限核（算子级/全局）、SuperKernel、通信入图 | **小**：主要是 FX 层 pass（inplace_pass、pattern fusion、cat/noop 消除、内存复用、静态 Kernel 编译）+ 流级限核 + SuperKernel（npugraph_ex 也有） |
| 集合通信 | 官方提供 `patch_for_hcom`（GE 路线） | npugraph_ex 默认支持通信算子入图 |
| 编译/捕获时间 | **两段编译**（Dynamo + Ascend IR 编译），官方只给"通常较长"；可用 `cache_compile(ge_cache=True)` 跨进程缓存 | **捕获**（Dynamo + capture）；官方 vllm-ascend 文档提到开启 static kernel 会额外增加"**数分钟到数十分钟**"启动时间（该数字属于静态 Kernel 编译，不是 GE 图编译） |
| 资源约束 | 无"每图一条 Stream"这类约束（下沉是设备侧 Task 序列） | 有：官方 vllm-ascend 文档写"一个 graph 至少需要一条独立 stream"，上限受 stream 数（2048）限制（ACLGraph 捕获数上限） |
| 失败模式 | 断图（回 eager）、Guard 失效重编译、档位不命中报错、值依赖算子退 Host 调度 | 断图、地址/shape 变化触发 Recapture、超过 `capture_limit` 整体回退 eager、stream 资源不足（错误码 207008） |
| 可观测性 | GE dump 图（`_graph_unknown_flag` 判是否完全下沉）、graph_dump pbtxt、GE Session 日志 | `aclmdlRIDebugJsonPrint`（维测）、`torch_compile_debug`、capture_limit 告警 |
| 官方成熟度表述 | TorchAir 简介中 GE 是主推图模式（max-autotune 是默认模式） | 官方明确"**原 reduce-overhead 模式（aclgraph）……将不再演进**"，推荐改用 **npugraph_ex** 后端；样例仓建议"优先选择 npugraph_ex" |
| 当前 vllm-ascend | **未使用**（显式 `mode="reduce-overhead"` + `run_eagerly=True`，注释写明 avoid IR transformation；见 00-source-evidence §2） | **默认使用** |

---

## 8. 明确查不到的（宁缺勿编）

1. **GE 首次图编译的绝对时长**（任何模型规模 × 卡型的公开数字）。官方只有"通常较长"与"两段耗时"的定性描述 + 缓存机制。第三方那篇自称 Mixtral 8×7B 编译 15 分钟的文章所引用的环境变量（`GE_FUSION_PASS_MODE`、`GE_WARMUP_SHAPES`、`GE_MEM_PROFILING`、`GE_TILING_SEARCH_MODE`）**在官方文档中均查不到**，故不采信（§5.2 第 7 条）。
2. **官方给出的 "GE 稳态 host 时间 vs eager host 时间" 的百分比/绝对值对比**。官方只给了"下沉 vs Host 调度"的 E2E 收益（LLaMA-7B：-18 ms / +37%）和下沉头开销（71B ≈2 ms）。
3. **TorchAir 对 `torch.cond` / `torch.while_loop`（结构化数据依赖控制流）的转换支持**。官方文档只在 GE pass 说明里提到 IR 层存在 If/Case 控制算子，**没有**任何"PyTorch 侧怎么写、TorchAir 能不能转"的说明。
4. **TorchAir（PyTorch 前端）对 Triton kernel 的直接入图支持**。官方只有 TensorFlow 前端的 Triton 入图文档；PyTorch 侧的 Triton 走的是 `backend="inductor"`（Triton-Ascend），与 torchair GE 是两条后端路线。
5. **`torchair.npu` 命名空间、`register_frozen_parameter` / `inplace` / `static_input` / `get_input` / `update` 这套 API 名**：在官方 TorchAir 文档与本地 torchair 仓库文档中均查不到（详见 §4.2）。
6. **`input_shape_ranges` 这个选项名**：查不到；真实名字是 `ge.inputShape`（范围写法）+ `ge.dynamicDims` / `DYNAMIC_DIMS`（分档），旧的 `ge.exec.dataInputsShapeRange` 已废弃。
7. **GE 模式下 Python dataclass 作为图输入时，字段值变化是否触发 guard/重编译的细节**。官方只有 cache_compile 示例用了 dataclass 入参，没有约束说明。
8. **A3（910 A3）上 GE 与 ACLGraph 的公开头对头性能数字**。
9. **GE "下沉头开销" 的分解比例**（1600 个 I/O 里，元数据转换占多少、地址刷新占多少）——官方只给了总量 ≈2 ms。
10. **GE 图模式下 `.item()` 的官方替代要求**（官方只给了断图事实与 `npu_print`；样例仓给了"显式传入"工程要求，但没有 API 级替代品）。

---

## 9. 对本项目问题的边界含义（**推断，非官方结论**）

> 本节是本文依据上述证据做出的推断，用于衔接 `02`/`03` 两篇，不宣称是官方口径。

1. **`prepare_input` 全部在图外。** 它产出的是 forward 的输入（`位置/长度/block table/slot mapping` 等），官方把这类东西定义为"应由框架构造后**作为显式输入传给模型**"（E3.4）。⇒ GE 能把它的**产物**喂进图（形态 A/B），但**不能省掉构造过程本身**（§2.2 第 (d) 层）。
2. **"GE 整图下发 = 抹除 host 开销"不成立**：即使全部下沉，每步仍要付 `Guards + Input 转换 + 沉头开销`（E2.2/E2.3）。以官方 71B 的 ≈2 ms 为锚点，**"零 host 开销"不是可达目标，而是不可达下界**。
3. **本项目 920B 上 2.76 ms 的 `prepare_input` 若原样保留，GE 也救不了它**；能救它的只有"减少对象数量 + 预分配缓冲 + 把结果变成固定 shape 的图输入"，这与 `docs/10` 的结论一致（**图化给的是"静态缓冲"这条路径，而不是"图"本身**）。
4. **本项目的 FIA 元数据（`actual_seq_lengths*`）在 GE 路线里有官方正解**：改 Tensor 输入 + `tiling_schedule_optimize`（E3.12/E3.13）。但要注意这两条都**要求改模型侧的算子调用方式**（用 `torchair.ops` 的 FIA 而非 `torch_npu` 的），且仅 GE 模式可用——这与 vllm-ascend 当前"ACLGraph + 每步 update"的形态是**互斥的实现路线**。
5. **可验证性**：是否真的下沉，有官方判据（dump GE build 图看 `_graph_unknown_flag`，E3.19），不需要靠猜。这是后续任何方案评估必须带上的验收项。

---

## 附录 A：关键文档 URL 清单（全部于 2026-09-24 访问，HTTP 200）

**TorchAir 官方文档（26.0 分支，hiascend.com）**

| 主题 | URL |
|---|---|
| TorchAir 简介（含两种图模式、aclgraph 不再演进） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md |
| GE 图模式快速上手（max-autotune 是默认） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/quick_start.md |
| CompilerConfig 类（mode 取值与默认） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/torchair/compiler_config.md |
| GE 图模式 API 列表 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/api_list.md |
| 模型编译缓存（含"首次/再次执行"配图） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/compile_cache.md |
| cache_compile API | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/inference/cache_compile.md |
| 固定权重类输入地址（frozen_parameter） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/frozen_parameter.md |
| Tiling 调度优化（Tiling 下沉） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/tiling_schedule_optimize.md |
| 动态 shape 图分档执行 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/dynamic_gears_merge_policy.md |
| set_dim_gears | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/inference/set_dim_gears.md |
| 动静子图拆分场景性能优化（N 算子一次下发） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/static_model_ops_lower_limit.md |
| SuperKernel 范围标定 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md |
| torchair FIA 定制接口（Tensor actual_seq_lengths） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/api/ops/npu_fused_infer_attention_score.md |
| 支持 ATen API 清单 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/aten_api.md |
| 自定义算子入图概述 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/custom_op_graph/overview.md |
| In-place 自定义算子入图样例（Kernel 直调不可入图） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/custom_op_graph/in_place_op_cases.md |
| Dynamo 导图（不支持动态控制流） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/dynamo_export.md |
| 图内 Tensor 打印（print 会断图） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/advanced/tensor_print.md |
| 动/静态图概念（Dynamo/GE 两张表） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/dynamic_static_graph/concepts.md |
| 动/静态图展示（含 `_graph_unknown_flag` 判据） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/dynamic_static_graph/presentation.md |
| 典型问题：FA 的整图静态下沉四路径 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/dynamic_static_graph/typical_issues.md |
| 入图失败定界与定位 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/cases/graph_failed_cases.md |
| npugraph_ex 快速上手（约束、与 CUDAGraph 对齐） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/npugraph_ex/quick_start.md |
| 重捕获次数限制（aclgraph 不支持动态 shape） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/npugraph_ex/basic/capture_limit.md |
| aclgraph 间内存复用（固定地址、Recapture、clone） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/npugraph_ex/basic/memory_reuse.md |
| npugraph_ex 场景 Host Tiling 参数刷新 | https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/appendix/host_tiling_update.md |

**CANN / GE / Runtime 官方文档**

| 主题 | URL |
|---|---|
| 模型下沉调度（Host 调度 vs 下沉调度） | https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/graphdevg/atlasag_25_0087.html |
| 深度解读 CANN 模型下沉技术（LLaMA-7B +18ms/+37%；盘古71B 头开销 2ms） | https://www.hiascend.com/zh/developer/techArticles/20240715-1 |
| 深度解读 CANN 动态 Shape 图调度加速（Host Kernel 拆分、Tiling 缓存、小 Shape 优化） | https://www.hiascend.com/developer/techArticles/20250911-1 |
| aclmdlRIExecuteAsync（模型运行实例管理） | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1822.html |
| 模型运行实例管理（runtime 仓 API 参考） | https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/15_model_running_instance__management.md |
| ACL Graph 任务更新（更新任务更耗时、任务组一致性限制） | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0032.html |
| ACL Graph 任务更新（runtime 仓 dev_guide） | https://gitcode.com/cann/runtime/blob/master/docs/zh/dev_guide/04-03_task_update.md |
| 动态输入 shape 范围 / 动态维度（ge 仓） | https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/graph_dev/more_features/dynamic_shape.md |
| 动态分档设计（Case + N 子图、aclmdlSetInputDynamicDims） | https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/dynamic_gear.md |
| aclgrphBuildModel 配置参数（INPUT_SHAPE / DYNAMIC_DIMS） | https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendgraphapi/atlasgeapi_07_0143.html |
| options 参数说明（ge.inputShape / ge.dynamicDims / 已废弃的 dataInputsShapeRange） | https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/graphdevg/atlasgeapi_07_0119.html |
| 基于 fallback 形式下发算子（静态 shape 下会断图） | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/graph/graphdevg/atlasag_25_0090.html |
| Triton 入图（仅 TensorFlow 前端 + Host launch） | https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/graphdevg/atlasag_25_0106.html |
| GE Runtime 设计（TaskSink / rtModelExecute / ExecuteGraph） | https://gitcode.com/cann/ge/blob/master/docs/zh/design/modules/runtime/runtime.md |
| aclGraph 设计（TaskGroup、捕获、条件控制流） | https://gitcode.com/cann/runtime/blob/master/docs/zh/design/features/aclgraph.md |

**官方样例仓 / 社区**

| 主题 | URL |
|---|---|
| cann-recipes-infer：NPU 图模式优化原理（GE vs npugraph_ex、动态信息显式传入、禁 graph break 清单） | https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/cann/zh/npu_graph_optimization.md |
| cann-recipes-infer：raw 全文 | https://raw.gitcode.com/cann/cann-recipes-infer/raw/master/docs/cann/zh/npu_graph_optimization.md |
| PyTorch 编译模式（inductor / npugraphs / npugraph_ex） | https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/pytorch_compilation_mode.md |
| MindIE-SD 编译后端对比（torchair_ge vs npugraph_ex vs aclgraph） | https://gitcode.com/Ascend/MindIE-SD/blob/master/.agents/skills/compilation-dev/references/backend-comparison.md |
| vLLM-ascend 图模式指南（ACLGraph 默认、static kernel 启动代价） | https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/graph_mode.html |
| torch.compile troubleshooting（数据依赖操作 graph break） | https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html |
| TorchAir issue：cache_compile lookup 冗余（+4%） | https://gitee.com/ascend/torchair/issues/IC3L9G |
| vllm-ascend issue：torchair graph mode sunset（维护者回复） | https://github.com/vllm-project/vllm-ascend/issues/588 |
| vllm-ascend issue：模型覆盖与 ge 图模式的社区问答 | https://github.com/vllm-project/vllm-ascend/issues/2310 |

## 附录 B：本地只读证据文件（本次调研使用，未做任何修改）

| 文件 | 用途 |
|---|---|
| `/home/chiro/projects/vllm/HIST_PROJECT/.research/torchair-docs/docs/zh/**`（仓库 `Ascend/torchair`，commit `d29ef3c`，2026-09-14） | TorchAir 26.1 官方文档全文（含 `ascend_ir/`、`npugraph_ex/`、`appendix/`、`custom_op_graph/`） |
| `/home/chiro/projects/vllm/HIST_PROJECT/.research/ascend-pytorch-v2.10.0/torch_npu/csrc/core/npu/NPUGraph.cpp` | `graph_task_group_*` / `graph_task_update_*` / `replay` / `debug_dump` → aclmdlRI* 的源码级映射（E1.2） |
| `/home/chiro/projects/vllm/HIST_PROJECT/.research/ascend-pytorch-v2.10.0/torch_npu/npu/graphs.py`、`_graph_tree.py` | `make_graphed_callables` / `static_input_idxs` 的真实归属（ACLGraph 树，不是 GE），用于 §4.2 的 API 名核实 |
| `/home/chiro/projects/vllm/preparing-input-phase/docs/11-ge-whole-graph/00-source-evidence.md` | vllm-ascend 侧一手证据（本文不重复挖） |
| `/tmp/cann_graph.md`、`/tmp/exec1.png` | 本次网络调研的临时抓取（官方样例仓文档全文、官方 compile_cache 配图） |
