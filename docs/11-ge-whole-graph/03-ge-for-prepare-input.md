# GE 整图下发能用于 `prepare_input` 吗？——分层可行性判定

> 对象：vLLM 0.26.0（`568afb3a1`）+ vllm-ascend 0.26.0rc1（`f2f74a16c`），Ascend 910 A3 / Kunpeng 920B
> 实测口径：0.8B / TP1 / B=1 / ISL=128 / decode / `FULL_DECODE_ONLY` / pystack off
> 证据分级：`[实测]` 本项目真机数据 ｜ `[源码]` 可核对的仓库行号 ｜ `[官方]` 华为/昇腾/上游官方文档 ｜ `[推断]` 写清验证方法
> 访问日期：所有 URL 均为 2026-09-24 访问
>
> **口径校正（先记一笔）**：本目录根任务书把 `prepare_input` scope 写成
> `model_runner_v1.py:1859–2098`。逐行核对缩进后，scope 实际结束于 **2082**
> （`_sanitize_placeholder_input_ids_for_forward` 的最后一行）；
> `2084–2095` 的 `_preprocess()` 与 `2098` 的 `update_cos_sin(positions)` 缩进已收回
> 12 空格，**在 scope 之外**（`[源码]` `vllm_ascend/worker/model_runner_v1.py:2079–2100`）。
> 若按 2098 记账，会把 MM/embedding 查表与 cos/sin 更新算进本阶段，**高估**它的占比。
> 本文件全部按 **1859–2082** 口径。

---

## 0. 直接回答

### 0.1 分层结论（L1–L4）

| 层 | 内容 | GE 整图能否省 | 一句话理由 | `prepare_input` 内的可省量级 |
|---|---|---|---|---:|
| **L1** | kernel launch（device 侧提交） | **能，但这里几乎没有可省的** | GE/ACLGraph 省的就是"逐 task 下发"；可 `prepare_input` 里真正走 kernel launch 的只有 `compute_slot_mapping` 等极少数 | **≈0–140 µs** |
| **L2** | 算子派发（Python→C++→driver 的每次调用） | **能省一部分，仅限"可表达为张量运算"的调用** | GE 图内节点不再需要每次 Python 派发；**但图的输入必须由 host 每步准备** | **≈400–500 µs（上限）** |
| **L3** | 张量元数据构造（`cu_seqlens`/`slot_mapping` 等小张量算术） | **算术可下沉，值不可下沉** | 前提是固定 shape + 固定地址 + 输入已是设备张量；**值的来源是 host 账本** | **≈150–300 µs（与 L2 重叠）** |
| **L4** | host 对象与决策（dataclass、dict 记账、请求调度、控制流分支） | **完全不能省** | GE 图里没有 dict/deque/dataclass/请求对象；capture 模式还明令禁止同步与查询 | **0** |

### 0.2 一句话总结

> **`prepare_input` 的输入是 host 侧的调度决策（Python 对象），输出是 forward 的设备输入张量。
> GE 只能接管"从张量到张量"的那一段，接管不了"从 Python 对象到张量"的那一段。
> 因此 GE 在这条路径上能抹掉的只是「派发税」（≈0.4–0.6 ms / 步），
> 抹不掉「记账与决策税」（剩余 ≈2.2 ms 里，绝大部分不因 GE 而改变）。**

### 0.3 三个决定性的边界事实（后文逐一展开）

1. **GE 图里的"输入"是 `ge.Data(dtype, shape, placement)`，不是 Python 的 list/dict**
   （`[官方]` TorchAir 图 dump 原文，§1.2）。
   "这段 Python 每步还要跑一遍"不是修辞，而是官方机制的直接推论。
2. **官方文档自己承认了 GE 化的障碍**（`[源码]` `vllm-ascend/docs/.../ACL_Graph.md`，§4.1）：
   > "some attention operators need runtime metadata updates **even when the overall graph is static**"
   > "**Without that hook, capture alone is not enough to replay the correct attention state.**"
3. **`prepare_input` 在图之外**——官方把它定义为"**how we obtain the inputs and their
   corresponding attention metadata**"（`[源码]` `docs/.../ModelRunner_prepare_inputs.md:1–20`，§4.1）。
   它产出的是图的输入，不是图内节点。

---

## 1. 判定所依据的第一性原理：GE 的图里"输入"必须是什么形态

这一节回答关键判断点 1。**结论先行：GE 捕获/编译的是计算图（device 任务序列 + 可选 host
算子），不是 Python 执行流；Python 里算出来的 list/dict 只要没变成图输入张量，就在图外，
每步都要重跑一遍。**

### 1.1 GE 与 ACLGraph 是两条不同的技术，但"捕获对象"是同一样东西

先把三个容易混的名字钉死（`[官方]` TorchAir 概览
<https://gitcode.com/Ascend/torchair/blob/master/docs/zh/overview.md>）：

| 名字 | 机制 | 捕获/编译什么 |
|---|---|---|
| **ACLGraph**（`aclmdlRICaptureBegin/End` + `aclmdlRIExecuteAsync`） | **运行期 Stream 捕获**，不经过 GE 图编译 | **指定 Stream 上下发的 device task 序列** |
| **GE 图模式（Ascend IR / `mode="max-autotune"`）** | 编译期把 FX 图转成 Ascend IR，交给 GE Compiler 编译 | **计算图（算子 + 张量依赖）**，编译产物可含 host kernel |
| **GE 模型下沉（sink）** | 静态 shape 图的加载期整图下发 | **图上全部算子的 device task 序列**，执行期 1 次 launch |

ACLGraph 的捕获范围有官方明文（`[官方]` CANN 9.0.0《ACL Graph 简介》
<https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/acldevg/runtime_doc_dev_0045.html>）：

> "在 `aclmdlRICaptureBegin` 和 `aclmdlRICaptureEnd` 接口之间，**所有在指定 Stream 上下发的任务**
> 不会立即执行，而是被暂存在模型的运行实例中，只有在调用 `aclmdlRIExecuteAsync` 接口执行模型时
> 这些任务才会被真正执行。"

⇒ **捕获的宾语是"Stream 上的任务"**。host 侧的 Python 逻辑（记账、分支、`.item()`）不是
Stream 任务，**不在捕获范围内**。

GE 侧同理：Ascend IR 的构造单元是**算子与张量**——`[官方]`《什么是 GE 图引擎》/ GE 架构文档：

> "AscendIR 是 GE 编译流程使用的核心 IR……采用静态计算图的方式表达模型的**计算逻辑与数据依赖结构**"
> "**AscendIR 表示的是静态图，其图结构在编译期固定，不会在执行过程中动态改变。**"
> —— <https://gitcode.com/cann/ge/blob/a74342b574f03ffe45e059c1a0fa74e50ccf5733/docs/architecture.md>

图里没有"Python 对象"这种类型。`CachedRequestState`、`requests: dict[str, ...]`、
`input_batch` 的 deque/数组、`spec_decode_metadata` 的闭包——**没有任何一张 GE 图能表达它们**。

### 1.2 图输入的形态：`ge.Data(dtype, shape, placement)` —— 这是本问题的"铁证"

TorchAir 官方文档给出了 FX 图转成 GE 图后的**原始 dump**（`[官方]`《动/静态图展示》
<https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/cases/dynamic_static_graph/presentation.md>）。
三个场景放在一起看，结论非常干净：

```python
# 场景 A：dynamic=False（静态 GE 图）—— python 标量被折成常量节点
primals_1_0 = ge.Data(index=0, dtype=0, shape=[64, 128], placement="NPU", node_name="primals_1")
primals_3_0 = ge.Data(index=2, dtype=0, shape=[100, 128], placement="NPU", node_name="primals_3")
Mul_1_0     = ge.Mul(Sub_0, ge.Const(100, dtype=0), node_name="Mul_1")   # <- python 标量 100 -> Const

# 场景 B：dynamic=True（动态 GE 图）—— python 标量变成"host 侧输入节点"
primals_3_0 = ge.Data(index=2, dtype=9, shape=[], placement="CPU", node_name="primals_3")
primals_6_0 = ge.Data(index=5, dtype=9, shape=[], placement="CPU", node_name="primals_6")
```

逐条读这张 dump，能直接回答关键判断点 1 和 3：

1. **图输入 = 有 `dtype` + `shape` + `placement` 的 Data 节点**。不是 list，不是 dict，
   不是对象。**"把 `actual_seq_lengths_kv` 从 Python list 换成 tensor"之所以是 GE 化的前提，
   不是风格问题，而是类型问题**——list 根本无法成为 `ge.Data`。
2. **Python 标量在图里要么被折成常量（`ge.Const`），要么被升格成一个 host 侧输入节点
   （`ge.Data(shape=[], placement="CPU")`）**。后者意味着：**每次执行都必须由 host
   重新把这个值喂进去**——这正是"图外的 Python 每步还要跑一遍"的机器可读证据。
   `ge.Const` 的语义更严厉：**值一旦在编译期固定，运行期就不能变**。
   ⇒ 任何"每步变化的 host 标量"（本步 `num_reqs`、`num_tokens`、`num_prefills`、
   `split_decodes` 的分类结果）在静态 GE 图里都无处安放，只能变成 host 输入或触发重编译。
3. **`placement="CPU"` 的存在说明 GE 图内确实可以放 host 侧的东西**（见 §1.3），
   但它同时说明：**这些 host 侧的工作仍然要在 host 上执行**，只是执行体从 Python/ATen
   变成了 GE 的 Host Kernel。

> `[官方]` 同页还给了判定指令：dump 出的 build 图里若有 `_graph_unknown_flag = true`，
> 就是"非完全静态下沉调度"；只有它为 false 才是完全下沉。
> ⇒ **"GE 整图"不是一个开关，而是一个谱系**；带 `placement="CPU"` 的 Data 节点、
> 带动态 shape 的图，都不在"一次下发整图"的那一端。

### 1.3 图内可以包含 host 计算——但那是"C++ 化的 host 工作"，不是"消失的 host 工作"

这一条是本文与 `docs/10` 的一个重要补丁，也是关键判断点 3 的答案。

GE 的**动态 shape Host 调度**模式，官方描述得比多数人想象的更"host 友好"
（`[官方]`《深度解读昇腾 CANN 动态 Shape 图调度加速技术》2025-09-12
<https://www.hiascend.com/developer/techArticles/20250911-1>）：

> "一个 AI Core 算子的下发会被拆分为 InferShape、Tiling、AllocMemHbm、Launch 等多个
> **Host Kernel**……执行图上的节点与 Host Kernel 一一对应。"
> "对于一些小 Shape 算子，**输入 Tensor 在 Host 侧**，本身计算量很小，下发到 Device 执行的
> 调度开销往往大于算子的实际计算开销。**将这类算子保留在 Host 侧执行**，可以有效减少
> 调度开销带来的性能影响。……GE 识别将这部分算子保留在 Host 侧执行"
> （默认启用；LLaMA2 上刷新 650+ 个算子，E2E 1.062s → 1.009s）

⇒ 正确的说法是：

- **GE 图可以包含 host 侧计算**（甚至有"小 shape 算子刻意留 host"的官方优化 pass）；
- **但这些 host 计算每次执行都要重新跑一遍**，它们省掉的是**每算子一次的
  InferShape/Tiling/内存分配/Python↔C++ 转换**，不是计算工作本身；
- 因此 **L2（派发税）对这部分是真实可省的，L3 的"值"和 L4 的"决策"依然不可省**。

官方对"Host 调度 vs 下沉"的定性同样直接（`[官方]`《深度解读昇腾 CANN 模型下沉技术》2024-07-15
<https://www.hiascend.com/developer/techArticles/20240715-1>）：

> "静态 shape 模型在编译时即可确定所有算子的输入输出 shape……因此，GE 提供了静态图下沉
> 调度模式，让模型中的算子在加载阶段提前以整图的形式下发到 Device 上，在执行时，
> **只需在 Host 侧下发一个模型执行的 Task** 即可触发模型在 Device 上调度执行。"
> "**每次模型下发时，支持更新模型的 Feature Map 内存地址和输入输出内存地址**……
> 以盘古 71B 增量推理模型为例（模型输入输出总个数约 1600 个，模型内节点总数约 6300 个，
> 模型运行时 Feature Map 内存地址不变更，**10 个输入输出内存地址变更**），
> 当前模型下沉的**头开销约 2 ms**。"

请把这句"头开销约 2 ms"和我们的实测放在一起：**B=1 单步才 4.9 ms，
`prepare_input` 2.76 ms**（`[实测]` `docs/05` §5bis.8）。在一个**每步都要刷新输入**的
自回归 decode 里，"下沉"并不是免费的；它的头部开销与**每次执行需要刷新的输入/输出个数**同阶。
这构成 §4 对比的最硬素材。

### 1.4 capture 模式的硬约束：`prepare_input` 直接违规

即便走 ACLGraph 这条路（把 `prepare_input` 的产物"捕获"进图也不行），
运行时文档列出的捕获期禁令会直接命中我们的代码（`[官方]` CANN runtime《单流捕获》
<https://gitcode.com/cann/runtime/blob/54cd3f51d17924a9a1f130a653e8939d2705c142/docs/02_dev_guide/04-01_%E5%8D%95%E6%B5%81%E6%8D%95%E8%8E%B7.md>）：

> 2. "……**对 Stream 或 Event 的查询或同步均为非法操作**。同样，对 Device 或 Context 的
>    查询或同步也是非法的……在任何捕获模式下都是非法的。"
> 5. "若捕获的异步内存复制任务涉及 Host 内存，则**只支持使用 `aclrtMallocHost`
>    申请的 Host 锁页内存**，否则在捕获过程中将返回报错。"

对照我们的 `prepare_input`：

| capture 禁令 | `prepare_input` 里的对应代码 | 后果 |
|---|---|---|
| 禁止 Event 同步 | `num_accepted_tokens_event.synchronize()`（`[源码]` `model_runner_v1.py:1103–1104`） | **捕获期直接非法** |
| 禁止查询/同步 | `_treat_single_token_prefills_with_state_as_decodes` 的 `torch.any(...).item()`（`[源码]` `gdn_attn_builder.py:73`）、`split_decodes_and_prefills` 返回 Python int、异步 `_prepare_input_ids` 的逐请求 `.item()`（`[源码]` core `gpu_model_runner.py:1809`） | 数据依赖控制流，**捕获期非法 / 无法捕获** |
| H2D 必须来自锁页内存 | `query_start_loc.copy_to_gpu()`、`req_indices.copy_to_gpu()`、`commit_block_table` 等（`[源码]` `model_runner_v1.py:1039/1160–1167`、`vllm_ascend/worker/block_table.py:288–289`） | 需全部改成 pinned（部分已是） |

上游 PyTorch 对"为什么数据依赖控制流不能被捕获"有标准解释（`[官方]`
<https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html>）：
**数据依赖控制流（`if` 依赖张量值）与直接张量数据访问（`.item()`、`.data_ptr()`）会触发
graph break**；`fullgraph=True` 时直接报错。绕过办法只有 `torch.cond`（把两个分支都编译进图）
或 `capture_scalar_outputs`（把标量搬进图，但**"host 需要知道这个值才能分支"这一事实不变**）。

---

## 2. 四层开销逐层判定

### 2.1 L1 kernel launch（device 侧提交）—— **能省，但这里几乎没得省**

**判定：GE 的收益点，但在 `prepare_input` 里不是主要矛盾。**

`prepare_input` 里真正产生 device op 的地方，逐个数：

| 位置 | 形态 | 实测/估计成本 |
|---|---|---:|
| `compute_slot_mapping`（Triton kernel，`[源码]` `block_table.py:179`；kernel 体 `vllm_ascend/ops/triton/compute_slot_mapping.py:12`） | 1 次 launch / 步，计算已在 device | `in.slot_mapping` **136.3 µs**（`[实测]` `docs/05` §5bis.1）——但**其中绝大部分是 Python 侧包装与参数准备，不是 launch 本体** |
| `_pad_non_spec_decode_graph_inputs` 的 `fill_`/`copy_`/`expand_as` | 3 组 × ~4 op ≈ 12 次 device op | `pad_graph_inputs` **285.9 µs**（同上），均值 ≈24 µs/op（**高于**基准 3–12 µs） |
| `_build_actual_seq_lengths`（`empty_like`+`copy_`+`sub(out=)`） | 3 组 × 3 op = 9 次 | **163.0 µs**，均值 ≈18 µs/op（`[实测]` 同上） |
| 各种 `copy_to_gpu()` / `fill_(-1)` / `seq_lens` 赋值 | 十余次小 H2D + device op | 分散在 S11/S12/S16/S18/S19/S21 |

`[实测]` 920B 一次小 NPU 算子派发 **3–12 µs**（`docs/05` §5bis.5，n=3000）。
把上表按 3–12 µs 折算，`prepare_input` 里"纯 launch 本体"的物理下限是
**约 20–30 次小 op × 3–12 µs ≈ 100–300 µs**——但请注意：这个区间里**只有很小一部分是
"launch"本身**，其余是 Python→ATen→driver 的**派发路径**，属于 L2。

**为什么 `pad_graph_inputs` 的单位成本（≈24 µs/op）明显高于基准（3–12 µs）？**
`[推断]`：`expand_as` + `copy_` 在 NPU 上要经 ATen dispatcher、TensorIterator 构造、
以及非阻塞拷贝的 stream 记账；小 tensor 下这些"固定税"占满。验证方法见 §7 V4。

**L1 小结**：能省，但可省量级仅 **≈0–140 µs**（把 `slot_mapping` 的 launch 部分算作上限）。
**"GE 能省 launch"这句话是对的，但它解释不了 `prepare_input` 的 2.76 ms。**

### 2.2 L2 算子派发（Python→C++→driver 的每次调用）—— **能省，但有严格前提**

**判定：这是 GE 在 `prepare_input` 里唯一有实质收益的层；前提是"可表达为张量运算"+
"固定 shape"+"参数可由 device/固定缓冲提供"。**

为什么这是真收益，官方话讲得很直白（`[官方]` 2024-07-15 下沉文章）：

> "Eager 模式也叫单算子模式……**一个算子的下发流程包含 Python 处理、Python 到 C++ 数据结构转换、
> Tiling 计算、申请算子的 Workspace 内存和输出内存、Launch 等 Host 操作。**"
> "相比于单算子模式，**图模式的 Host 调度可以避免总是返回 Python 调用栈，避免冗余流程与
> 数据结构转换，并且可以直接使用图编译阶段完成的 Infer Shape 与 Tiling 计算结果。**"

这正好对着我们的实测指纹：

| 观测（`[实测]` `docs/05` §2.2 / §4.2） | 对应的开销层 |
|---|---|
| `frontend_bound 66.01%`（`frontend_latency 59.89%`）、`retiring 12.85%`、**IPC 0.771** | CPython 逐条派发的特征（L2/L4） |
| `aclrtSynchronize*` 全族 **0.27%** | **不是等卡**（排除"GE 省等待"的假设） |
| top-1 `_PyEval_EvalFrameDefault` 仅 10.35%、top-10 合 22.83% | 极平——"调用条数多、每次都不重" |
| `_PyType_Lookup` 1.67% + `_PyObject_GenericGetAttrWithDict` 1.36% + `unicodekeys_lookup_unicode` 1.68% | 属性/关键字查找（22 字段 dataclass 构造） |
| `_PyObject_Malloc` 1.16% + `malloc` 1.06% + `gc_collect_main`（self 占 gc 子树 54.15%） | 每步新建对象 |

**GE 能吃到的那一块**，是 §2.1 表里那些"device op"：
`pad_graph_inputs`(285.9) + `build_actual_seq_lengths`(163.0) + `attach_decode`(40.2) ≈
**489 µs/步的 device 交互段**（`[实测]` `docs/05` §5bis.4 的分类小结：device ≈494 µs = 44%）。
若这些 op 成为图内节点，其 host 侧派发开销**可以降到接近 0**（只剩一次图执行）。

**但有三个"但是"**：

1. **图的输入仍要 host 准备**：上述 device op 的**源数据**（`block_table_tensor[:,0]`、
   `gdn_query_start_loc`）都是 host 每步写出来、再 H2D 上去的。图能省"加工"，省不了"喂料"。
2. **需要固定 shape**：`_pad_non_spec_decode_graph_inputs` 的 `graph_batch_size` 之所以存在，
   就是为了让每步 batch 长得像捕获时的 batch。`FULL_DECODE_ONLY` 已经把这件事做了
   （`[源码]` `gdn_attn_builder.py:238` 的 `decode_cudagraph_max_bs`）。
3. **收益与"是否真的进了图"强绑定**：现在的 `pad_graph_inputs` **已经在写预分配 buffer**，
   但它**每步仍有约 12 次 host→driver 调用**（`[源码]` `gdn_attn_builder.py:332–369`）。
   ⇒ **"预分配 buffer"不等于"入图"**，这正是 `docs/10` §4 标题"已有雏形"要补的一刀：
   **形态对了，但仍在图外，所以每步照收派发税。**

**L2 小结**：可省 **≈400–500 µs/步（乐观上限）**，且**必须**同时满足固定 shape + 固定地址
+ 参数 tensor 化。这是 GE 路线唯一的实质抓手，占 `prepare_input` 的 **~18%**。

### 2.3 L3 张量元数据构造—— **算术可下沉，值不可下沉**

**判定：`cu_seqlens` 的 cumsum、`slot_mapping` 的 gather 这类"纯算术"可以下沉；
但它们要用的"值"来自 host 账本，因此下沉只改变"算在哪"，不改变"谁提供输入"。**

先按"纯 host 记账"与"可表达为张量运算"把 `prepare_input` 劈开（回答关键判断点 2）：

| 类别 | 具体内容 | 能否表达为张量运算 |
|---|---|---|
| **纯 host 记账**（不可表达） | `_update_states`：`requests.pop` / `remove_request`、`CachedRequestState` 构造、`input_batch.add_request` 写 `token_ids_cpu` 行与采样标量、`condense()` 的空位下沉与行拷贝、`swap_states`、`refresh_metadata` 重建 `SamplingMetadata`（`[源码]` core `gpu_model_runner.py:1169–1543`；`gpu_input_batch.py:338/569/686/814`） | 不能。这是 Python 对象图的操作 |
| **可表达为张量运算** | `np.repeat`/`np.cumsum`/`np.subtract` 生成 `req_indices`、`cu_num_tokens`、`query_pos`、`positions_np`（`[源码]` `model_runner_v1.py:908–937`）；`index_select` 取 `input_ids`（`977–997`）；`seq_lens = num_computed + num_scheduled`（`1227–1230`）；`optimistic_seq_lens_cpu`（`1052–1061`）；`compute_num_computed_tokens`（core `backend.py:530`） | 能（但需固定 shape + 设备张量输入） |
| **已经是 device kernel，host 只剩 launch** | `compute_slot_mapping`（Triton，`[源码]` `vllm_ascend/ops/triton/compute_slot_mapping.py:12`，launch 在 `block_table.py:179`）；`update_num_computed_tokens_for_batch_change` kernel（`model_runner_v1.py:1146–1153`） | 已经是 |

**关键判断点 4：`_build_attention_metadata` 的部分下沉是否已经发生？——已经发生，且它说明了三件事。**

证据 1：**`slot_mapping` 已经是 device kernel**。core 与 ascend 都是 Triton/NPU kernel
（`[源码]` core `vllm/v1/worker/block_table.py:166`；ascend `vllm_ascend/ops/triton/compute_slot_mapping.py:12`），
CPU 只剩 launch，实测 `in.slot_mapping` **136.3 µs** 基本是 Python 侧包装
（`[实测]` `docs/05` §5bis.1）。

证据 2：**`full_graph_fia_v2` 已经用 tensor 传 `actual_seq_lengths_kv`，而 eager 路径仍用 Python list**：

```python
# 图路径（[源码] vllm_ascend/attention/attention_v1.py:1028–1029）
key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(...)
actual_seq_lengths_kv = attn_metadata.seq_lens          # <- tensor

# eager 路径（[源码] 同文件 _get_fia_params：1248 / 1258 / 1269）
actual_seq_lengths_kv = attn_metadata.seq_lens_list     # <- Python list
```

文件里的注释把原因写明了（`[源码]` `attention_v1.py:348–354`）：

> "`full_graph_fia_v2` passes the **seq_lens tensor (not `seq_lens_list`)** as
> `actual_seq_kvlen` during graph capture, and `_get_fia_params` derives the PrefillCacheHit
> batch size from `seq_lens.shape[0]`, so the tensor has to carry the dummy request too."

> **交叉印证**：同目录 `01-ge-capability.md` §E3.3 记录，TorchAir 为 FIA 专门提供了
> 收 **Tensor** 形参的 `torchair.ops.npu_fused_infer_attention_score`
> （`actual_seq_lengths*` **仅图模式可用**）。
> ⇒ **"改算子接口让参数以 tensor 进来"是 GE 化的硬前提，这一点在两条独立调研路径上一致。**

证据 3：`AscendAttentionMetadataBuilder.build` 里 `seq_lens_list = seq_lens.tolist()`
与 `actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()`
（`[源码]` `attention_v1.py:332–333`）**在非图路径仍在每步执行**。

**这三条合起来说明什么？**

1. **"元数据下沉"不是设想，而是已经在跑的工程实践**——`slot_mapping` 走 kernel，
   FIA 的图路径走 tensor。**技术可行性已被本公司代码证明。**
2. **下沉是"按路径"发生的，不是"按模块"发生的**。`slot_mapping` 全面下沉了；
   `actual_seq_lengths_kv` 只在图路径下沉；`seq_lens_list` / `actual_seq_lengths_q` 的
   list 形态在 eager 与 builder 里仍存活。⇒ **`prepare_input` 的下沉是"块状补丁"式的，
   不是整体重写。**
3. **`.tolist()` / list 形态的存在，本身就是"这个值要回 host"的信号**：
   FIA 算子的 eager 路径要吃 list 形参——**只要算子签名要 list，host 就一定得算出来并传进去**，
   这段 Python 不可省。反过来，**只要算子签名接受 tensor，host 侧就只剩"把值写进张量"**。
   ⇒ **"下沉"的真正杠杆是「改算子接口 + 把值留在 device」，不是"把 Python 换成 GE"。**

**关键判断点 3：要把 L3 搬进 GE 图，需要满足什么条件？**

对照三个前提，逐条给现状与证据：

| # | 前提 | `prepare_input` 现状 | 证据 |
|---:|---|---|---|
| 1 | **控制流固定**（分支形状不随 step 变） | **仅 `FULL_DECODE_ONLY` + 纯 decode 下成立**；一旦进入 prefill/chunked/投机，`_build_attn_state` 会在 5 个 `AscendAttentionState` 里选一个（`[源码]` `model_runner_v1.py:1320–1347`），`_treat_single_token_prefills_with_state_as_decodes` 还会按数据改 `is_prefilling`（`gdn_attn_builder.py:66–78`） | `[源码]` |
| 2 | **内存地址固定** | **最大缺口**。`_build_actual_seq_lengths` 仍 `torch.empty_like`（`gdn_attn_builder.py:133–135`，虽然 `out=` 参数已被用于图分支：第 134 行的三元表达式）；`GDNAttentionMetadata(...)` 这个 22 字段 dataclass 每步新建（`gdn_attn_builder.py:860–885`） | `[源码]` |
| 3 | **参数可由 device 提供** | **部分满足**：`seq_lens` 是 device tensor（`model_runner_v1.py:1227–1230`），`query_start_loc` 有 device 副本；但 `block_table`/`input_ids`/`positions` 每步由 host 写 pinned → H2D | `[源码]` |
| 4 | **补充：形状必须静态**（官方下沉前提） | **在 FULL_DECODE_ONLY 下被 padding 强行满足**，代价见 §4.3 | `[官方]` 2024-07-15；`[源码]` `_pad_query_start_loc_for_fia`（`model_runner_v1.py:834–881`） |

⇒ **L3 的结论**：**算术可下沉，且不需要 GE 也能下沉（kernel 化即可）；
要沉到 GE 图里则必须先补上"固定地址"这一条，而这一条与"每步值都变"的 L4 天然冲突。**
把 L3 搬进图的**边际收益**（相对"kernel 化 + 固定缓冲"）主要是**省掉 launch 与派发**，
即已被 L2 计过的那部分——**不要重复计收益**。

### 2.4 L4 host 对象与决策—— **完全不能省**

**判定：不可省，且不可"图化"。这不是工程量问题，是类型系统问题。**

三条独立的理由：

**(a) GE 图里没有这些类型。** `_update_states` 的核心是
`requests: dict[str, CachedRequestState]` 的增删、`input_batch.condense()` 的空位下沉、
`swap_states(i1, i2)` 的行交换、`refresh_metadata()` 的 `SamplingMetadata` 重建
（`[源码]` core `gpu_model_runner.py:1169–1543`；`gpu_input_batch.py:338/569/686/814`）。
这些操作的**数据结构**是 Python dict/deque/`CachedRequestState`/闭包
（如 `deferred_spec_decode_corrections`，`gpu_model_runner.py:1363–1365/1511–1541`）。
`ge.Data` 的 `dtype` 枚举里没有"dict"，AscendIR 的算子原型里没有"pop 一个请求"。

**(b) 决策必须回 host 才能选图。** 图执行是 host 发起的
（`aclmdlRIExecuteAsync(modelRI, stream)`，`[官方]` CANN ACL Graph 文档），
所以**"本步用哪张图、按哪个 batch descriptor"必须在 host 决定**。
`_determine_batch_execution_and_padding` / `CudagraphDispatcher.dispatch`
（`[源码]` core `gpu_model_runner.py:3877–3922`、`v1/worker/gpu/cudagraph_utils.py:360`）
吃的输入是 `num_tokens`（Python int）——**这是 host 值，不是张量**。

**(c) capture 模式禁止它需要的操作。** `num_accepted_tokens_event.synchronize()`
（`[源码]` `model_runner_v1.py:1103–1104`）与 `.item()` 系列在捕获期非法（§1.4）。
即便改用 `torch.cond` 把分支都编进图（`[官方]` PyTorch graph break 文档），
**host 仍然需要知道布尔结果才能决定"重放哪张图 / 走哪条 Python 路径"**——
这就是 `treat_single_token` 那 168 µs 的本质：
**它的成本不是"算了很久"，而是"必须让 host 知道一个布尔值"。**

**具体到关键判断点 5：GDN builder 那 908 µs 里，可下沉的比例有多大？**

先把 `docs/05` §5bis.4 的逐段分解（3 次调用合计，p50，**含探针地板 ≈45 µs/步**）摊开：

| 段 | µs/步 | µs/次 | 性质 | 可下沉？ |
|---|---:|---:|---|---|
| `gdnb.pad_graph_inputs` | 285.9 | 95.3 | device（已在写预分配 buffer） | **形状对了、地址对了 → 可进图**；但**组间不可去重**（每组 state indices 不同） |
| `gdnb.treat_single_token` | 169.3 | 56.4 | 纯 CPU/Python + **`.item()` 决策** | **不可下沉**（host 需知布尔值）；但**decode-only 可短路跳过**（见 §6.4） |
| `gdnb.build_actual_seq_lengths` | 163.0 | 54.3 | device（3 op/call） | **可进图**，且 **3 组结果完全相同 → 可去重 2/3** |
| `gdnb.compute_num_computed_tokens` | 129.5 | 43.2 | **缓存被 `.replace()` 击穿**，走 4+ 次 torch 派发 | 不该重算；修缓存即可省 ~130，**不需要 GE**（`[实测]` `docs/05` §5bis.5.1：子树 67.5% 在 `PyNumber_Subtract`） |
| `gdnb.split_decodes` | 121.1 | 40.4 | 纯 CPU 分类（返回 Python int） | 决策不可下沉，但 decode-only 可短路 |
| `gdnb.attach_decode` | 40.2 | 13.4 | 混合（含 1 次 `_build_actual_seq_lengths` + ctor） | 部分可进图 |
| `gdnb.ctor`（22 个 kwargs 的 dataclass） | 21.6 | 7.2 | 纯 CPU 对象构造（`gdn_attn_builder.py:860–885`） | 不可下沉；**但可改为 `__slots__`/直接赋值**（perf 里 `unicodekeys_lookup_unicode`=1.68% 是它的指纹） |
| `gdnb.mamba_block_table` / `attach_prefill` / `attach_spec` | 10.4 | 3.5 | 混合 | 视路径 |
| **残差**（含 ~45 µs/步探针记账） | 172.9 | 57.6 | 其中 **真正未解释 ≈123 µs/步** | 未知，需 perf 细分（`docs/05` §5bis.4 已诚实标注） |

**分类小结（`[实测]` `docs/05` §5bis.4）**：device ≈494 µs（44%）；纯 CPU/Python ≈317 µs（28%）；
缓存+残差 ≈302 µs（27%）。

**逐条回答关键判断点 5：**

| 段 | 给定描述 | 判定 |
|---|---|---|
| `pad_graph_inputs`（294 µs，已在写预分配 buffer） | 可下沉？ | **可进图**（形态已对），但**收益不是"省下全部 294"**：其中一部分是 12 次 device op 的派发税（≈100–200 µs，可省），另一部分是 `fill_`/`copy_` 的真实搬运（不可省，只是不再占 host）。且**3 组不能去重**（每组 state indices 不同） |
| `treat_single_token`（168 µs，纯 host 布尔索引 + `.item()`） | 可下沉？ | **原理上不可**：`torch.any(...).item()` 的语义就是"把张量的值拿到 host 做分支"。**但**在 `FULL_DECODE_ONLY` 的纯 decode 下这个分支恒为 False，**整个函数可以直接跳过**（见 §6）——**这是不需要 GE 的 168 µs** |
| `compute_num_computed_tokens`（129 µs，4+ 次 torch 派发） | 可下沉？ | **不该下沉，该修 bug**：`.replace()` 清空 `_num_computed_tokens_cache`（`[源码]` `gdn_attn_builder.py:78` + core `backend.py:528–534`），或直接用 host numpy 算（数据本来就在 CPU）。**≈130 µs，与 GE 无关** |
| `split_decodes`（118 µs） | 可下沉？ | 返回 Python int（`num_decodes/num_prefills/num_decode_tokens`），**必须回 host**；纯 decode 下可短路 |
| dataclass 构造（374 µs 残差中的大头） | 可下沉？ | 不可下沉；**可省一部分**（减少 kwargs、`__slots__`、复用对象）。注意 22 个字段里约 10 个是 device tensor 引用，`GDNAttentionMetadata` 本身只是 host 侧"指针包"，**它天然是 host 对象** |

**可下沉比例（我的估算，`[推断]`，方法见 §7）**：

```text
908 µs / 步（3 次 build）
├─ 可进 GE 图（device 段）              ≈ 400–500 µs   <- 上限，仍需固定地址改造
├─ 不可下沉但可短路/修复（非 GE 手段）  ≈ 300–430 µs   <- treat_single_token 168 + 缓存修复 130 + ctor 部分
└─ 结构性不可省（决策 + 残差）         ≈ 100–200 µs
```

⇒ **"908 µs 全部能靠 GE 抹掉"是错的；"GE 能在其中拿到 ~400–500 µs"是站得住的。**

### 2.5 四层小结（把上面的判定收敛成一张表）

| 层 | GE 能否省 | 省多少（`prepare_input`，µs/步） | 省的方式 | 前提 |
|---|---|---:|---|---|
| L1 kernel launch | 能 | ≈0–140 | 图内任务序列代替逐次下发 | 固定 shape |
| L2 算子派发 | **部分** | **≈400–500** | 图内 Host Kernel/device kernel 取代逐次 Python 派发 | 固定 shape + 固定地址 + tensor 输入 |
| L3 元数据构造 | 部分（只省算术，不省值） | ≈150–300（**与 L2 重叠，勿重复计**） | cumsum/gather 进图或 kernel 化 | 设备侧输入 + 固定缓冲 |
| L4 host 对象与决策 | **不能** | **0** | — | — |
| **合计（去重后）** | | **≈400–600** | | |

---

## 3. `prepare_input` 组件的 GE 适用性分解表

标注口径：**可进 GE 图** = 满足固定 shape/地址后可作为图内节点或图输入；
**可下沉 device kernel** = 不做 GE 也能/已能变成 device kernel；
**只能留 host** = 类型或语义上必须留在 host。
收益 = "在本阶段实现该改造可省下的 host 时间（µs/步）"，`[实测]` 有真机 p50 支撑，
`[推断]` 为估算（验证方法见 §7）。本表 15 行，覆盖 `data/static/prepare-input-steps.json`
的全部 A/S/U/M 步骤。

| # | 子步骤（对应 step id） | 源码 | 判定 | 理由 | 预估收益（µs/步） |
|---:|---|---|---|---|---:|
| 1 | `synchronize_input_prep` 事件协议（A02） | `model_runner_v1.py:1860–1883` | **只能留 host** | 本质是"等上拍 H2D 读完 pinned buffer"的 host 事件等待；捕获期 Event 同步**非法**（`[官方]` runtime 单流捕获 §2） | 0；`[实测]` 102.4（可优化，但非 GE 能给的） |
| 2 | 请求生命周期记账 + sampling metadata（U01–U08, U11） | core `gpu_model_runner.py:1179–1543`；`gpu_input_batch.py:338–484/814–938` | **只能留 host** | `requests` dict 增删、`CachedRequestState` 构造、`output_token_ids.extend`、`deferred_spec_decode_corrections` 闭包、写 `token_ids_cpu` 行、惰性分配 `allowed_token_ids_mask_cpu_tensor` | 0 |
| 3 | `input_batch.condense()`（U09） | core `gpu_input_batch.py:686–812` | **只能留 host** | Python while 循环 + 空位下沉 + 行拷贝，依赖"哪些槽位被释放"这一 host 知识 | 0（稳态 decode 早退 O(1)） |
| 4 | `_may_reorder_batch` / `swap_states`（U10） | core `gpu_model_runner.py:1494–1495`；`gpu_input_batch.py:569–679` | **只能留 host** | `needs_swap.any()` 判定后做 host 数组交换 | 0（decode-only 恒不重排） |
| 5 | `commit_block_table` H2D（S01） | `model_runner_v1.py:906`；`vllm_ascend/worker/block_table.py:288–289` | **可进 GE 图**（作为图内 H2D 节点） | 是"host pinned → device"的异步 memcpy；捕获期允许**若源是 `aclrtMallocHost` 锁页内存**（`[官方]` 单流捕获 §5） | ≈10–30（省一次 Python→ATen 调用；搬运本身不可省）`[推断]` |
| 6 | token/request 级 numpy 算术（S02–S05, S13, S16） | `model_runner_v1.py:908–937, 1052–1061, 1085–1099` | **可进 GE 图 / 可下沉 kernel** | `repeat`/`cumsum`/`subtract`/比较/`nonzero` 全是张量可表达的算术；但**输入是 host 账本数组** | ≈50–120 `[推断]` |
| 7 | `index_select` 取 `input_ids`（S09） | `model_runner_v1.py:977–997` | **可下沉 device kernel** | 已是 ATen CPU 内核；若 token 池放 device 则变成 device gather（但池子随请求生命周期变，需固定化） | ≈20–50 `[推断]` |
| 8 | `query_start_loc` 家族（S11, S12, A12） | `model_runner_v1.py:1037–1049, 834–881` | **混合：可进图（数值）/ 只能留 host（填充值）** | 数值可张量化（cumsum/填充）；但 filler 值 = `num_reqs_padded`、`uniform_decode_query_len` 等 **host 标量**；且现在是**全量 copy_to_gpu**（`1039`） | ≈30–80 `[推断]` |
| 9 | `_prepare_input_ids`（S14） | `model_runner_v1.py:1067`；core `1761–1890` | **混合：同步路径可进图；异步路径只能留 host** | 同步路径 = O(T) memcpy（可入图）；异步路径 = 逐请求 Python 循环 + `.item()` + 构造 4 个 list + `torch.tensor(list, pin_memory=True).to()` | 同步 ≈10–30；异步 0（需架构改造） |
| 10 | `num_accepted_tokens` 同步（S17） | `model_runner_v1.py:1101–1127` | **只能留 host** | 唯一的强同步点；捕获期 Event 同步非法。仅在 spec+hybrid 触发（`docs/01` §6 三档） | 0 |
| 11 | 上屏家族（S18–S21） | `model_runner_v1.py:1129–1230` | **可进 GE 图** | `copy_`/`fill_`/gather+add 全是 device op，已是固定 buffer（`seq_lens`/`num_computed_tokens`/`positions`） | ≈60–120 `[推断]` |
| 12 | `compute_slot_mapping`（S23） | `model_runner_v1.py:1246–1250`；`block_table.py:150–190`；`ops/triton/compute_slot_mapping.py:12` | **已经是 device kernel** | CPU 只剩 Triton launch 包装；`[实测]` 136.3 µs/步 | 入图后 ≈80–120（省 launch 包装）`[推断]` |
| 13 | 图分派 + 占位符清理（A09, A14） | core `gpu_model_runner.py:3877–3922`；`cudagraph_utils.py:360`；`model_runner_v1.py:2079–2082` | **只能留 host** | 图重放由 host 发起（`aclmdlRIExecuteAsync`），**"选哪张图"必然是 host 决策**；输入是 Python int | 0（但可省其内部冗余计算） |
| 14 | `_build_attention_metadata`：GDN builder ×3（A13/M03–M06） | `model_runner_v1.py:3141–3166`；`gdn_attn_builder.py:531–897` | **混合（见 §2.4 分解）** | device 段可进图；`.item()`/Python int 分类不可；缓存击穿可修 | **≈400–500 可进图** + **≈300–430 靠非 GE 手段**（`[实测]` 总量 908） |
| 15 | `_build_attention_metadata`：full-attention builder ×1（M04） | `attention_v1.py:291–395`（`tolist` 在 `332–333`，`pin_memory().to()` 在 `330`） | **混合：list 部分只能留 host；device op 可进图** | `seq_lens_list` / `actual_seq_lengths_q` 的 list 形态**只在 eager 路径必需**（图路径已改用 tensor，`1029`）；`pin_memory()` 每拍新建是纯 host 浪费 | ≈40–100（`[实测]` 该类 self 35.9 + 子探针 88.3） |

> **表的用法**：第 1–4、10、13 行是**结构性不可动**的部分（与 GE 无关）；
> 第 5–12、15 行是**GE 或 kernel 化能吃到**的部分；第 14 行是最大的一块，
> 也是**唯一必须靠 GE 才能榨干**的一块。

---

## 4. 与 ACLGraph 方案的对比

### 4.1 官方文档已经把两者的边界说清楚了

`vllm-ascend` 的官方 ACLGraph 设计文档（`[源码]`
`docs/source/developer_guide/Design_Documents/ACL_Graph.md`，
中文镜像 <https://docs.vllm.ai/projects/ascend/zh-cn/main/developer_guide/Design_Documents/ACL_Graph.html>）
有一节 **"Host-side attention parameter update for full graph replay"**，原文：

> "Full graph replay on Ascend has an extra problem that upstream generic documentation does not
> cover in detail: **some attention operators need runtime metadata updates even when the overall
> graph is static.** The Ascend implementation handles this by **separating graph capture from
> host-side task parameter updates.**"
>
> 流程：捕获期记录 per-graph task handles / events / workspaces / **weak references to the
> tensors or metadata that must be refreshed**；重放前 `update_full_graph_params()` →
> `update_graph_params()`，在 update stream 上用 `graph_task_update_begin/end` 包住算子调用，
> 并用 `torch.npu.ExternalEvent` 保证顺序。
>
> "**The important design point is that Ascend full graph support depends on backend-provided
> `update_graph_params()` hooks. Without that hook, capture alone is not enough to replay the
> correct attention state.**"

对应源码里的形态（`[源码]` `vllm_ascend/attention/attention_v1.py:476–540`）：

```python
for key, param, handle, event in zip(forward_context.attn_metadata,
                                     graph_params.attn_params[num_tokens],
                                     graph_params.handles[num_tokens],
                                     graph_params.events[num_tokens]):
    seq_lens = forward_context.attn_metadata[key].seq_lens      # <- 每步从 host 侧容器取
    torch.npu.graph_task_update_begin(update_stream, handle)
    torch_npu._npu_paged_attention(..., context_lens=seq_lens, out=output, workspace=workspace)
    torch.npu.graph_task_update_end(update_stream)
    event.record(update_stream)
```

⇒ **当前的"图"是两段式：图（device 任务序列）+ 每步 host 参数更新。**
每层每次 replay 都要在 host 上重新发起一次算子调用——**派发税没有消失，只是搬了家**
（从 launch 变成 task update）。

### 4.2 GE 整图 vs ACLGraph 分段图：对 `prepare_input` 各自意味着什么

| 维度 | 现状：ACLGraph（分段图 + 每步 update） | 假设：GE 整图下沉（sink） | 对 `prepare_input` 的影响 |
|---|---|---|---|
| 图的粒度 | 分段捕获；attention 等作为 break point 或走 `update_graph_params` | **整图一条执行序列**，执行期 1 次 launch | **都不覆盖 `prepare_input`**：它在 capture/replay 之外、replay 之前 |
| `prepare_input` 的产物 | forward 的输入张量 + 每步 update 用的 metadata | forward 的输入张量 + 输入地址刷新 | **同样都只是"喂料"**，不因图变整而消失 |
| 变化参数怎么办 | `graph_task_update_begin/end` 每层刷新（官方承认 "capture alone is not enough"） | 官方下沉机制允许**"每次模型下发时更新 Feature Map 内存地址和输入输出内存地址"**，代价计入"下沉头开销" | **GE 不会消灭"每步刷新"，只会改变它的形态** |
| 头开销量级 | 每次 replay 1 次 `aclmdlRIExecuteAsync` + N 层 task update | 官方数据点：盘古 71B（6300 节点 / **1600 个输入输出**、10 个地址变更）**≈2 ms**（`[官方]` 2024-07-15） | **对 B=1 小 batch decode（单步 4.9 ms）是数量级相同的开销**，不可忽略 |
| 前提 | 固定 batch descriptor + 固定地址（debug 模式下 `ACLGraphWrapper` 会 assert 重放地址与捕获一致） | 固定 shape + 固定地址 + 参数可 device 提供 | 现状**部分满足**；`FULL_DECODE_ONLY` 的 padding 就是在买"固定 shape" |
| 对 `prepare_input` 的直接收益 | 无（它不覆盖这一段） | 无（同上） | **两条路线都不直接优化 `prepare_input`** |
| 间接收益 | 省 forward 的逐算子下发 | 省 forward 的逐算子下发（更彻底） | 见 §6.5 |

### 4.3 一个必须点破的悖论：**图化的前提，正在由 `prepare_input` 买单**

`FULL_DECODE_ONLY` 要能捕获，前提是"每步 batch 长得一样"。而让 batch 长得一样的那部分工作，
落到了 `prepare_input` 头上：

- `_pad_query_start_loc_for_fia`（`[源码]` `model_runner_v1.py:834–881`）每步补 dummy request；
- `_pad_non_spec_decode_graph_inputs`（`[源码]` `gdn_attn_builder.py:332–369`）每步
  `fill_` 空位 + `copy_` 有效行 + `expand_as` 补齐 —— **285.9 µs/步，是 `prepare_input` 内部
  第 3 大子步骤**（`[实测]` `docs/05` §5bis.1）；
- builder 里那些 `if self.use_full_cuda_graph and num_prefills == 0 and ...` 的图专用分支
  （`gdn_attn_builder.py:783/843`）本身也是每步要走的 host 判定。

⇒ **"图化让 forward 变快"是真的（0.8B 上 36.26 ms → 0.95 ms，38×）；
但"图化让 batch 固定"的成本，被转移到了 `prepare_input` 里（55.3% 的占比就是这么来的，
`[实测]` `docs/10` §1 的表）。** 所以"用 GE 整图去优化 `prepare_input`"这个提法，
在逻辑上要格外小心：**GE 图要成立，恰恰需要 `prepare_input` 先把 batch 摆成图的形状。**

### 4.4 ACLGraph 与 GE 在"能不能覆盖 `prepare_input`"上的共同答案

| 问题 | ACLGraph | GE 整图 |
|---|---|---|
| 能捕获/编译 host 侧 Python 记账吗？ | 否，只捕获 Stream 任务 | 否，只表达算子与张量 |
| 能捕获 `.item()` 分支吗？ | 否，捕获期非法 | 否，graph break / 需 `torch.cond` |
| 能捕获 `event.synchronize()` 吗？ | 否，明令非法 | 否，同上 |
| 能省 `prepare_input` 里的 device op 派发吗？ | 只能省"进了捕获段"的那些（attention 反而被排除） | **能省更多**（可把 metadata 加工一起入图） |
| 能省 `prepare_input` 的 host 记账吗？ | 否 | 否 |

### 4.5 一条工程史证据：这套栈**试过 GE 路径，并且主动放弃了它**

同目录 `02-why-not-enabled.md` §0.1 从 vllm-ascend 的文档 diff 挖出的事实：

- **v0.9.x ~ v0.11.0 期间，vllm-ascend 正式提供过 GE 图模式路径，名为 `TorchAirGraph`**
  （旧文档原话："**TorchAirGraph: This is the GE graph mode**"，
  见 <https://github.com/vllm-project/vllm-ascend/pull/4814/files> 中被删除的
  `docs/source/user_guide/feature_guide/graph_mode.md` 段落）；
- **v0.12.0rc1 起被移除**，理由是 "aclgraph is stable and fast now"。

⇒ 对本文的判定，这条史实有两层含义：

1. **"GE 能不能接上 vLLM 的推理路径"早就被回答过了：能。**
   所以本文的结论不是"GE 技术上不可行"，而是"**GE 接上之后也覆盖不到 `prepare_input`
   的 L4，且覆盖 L1/L2 的收益要用下沉头开销与启动编译成本换**"。
2. **它是"整图 vs 分段图"在真实工程里的一次对照实验，结论倾向 ACLGraph。**
   本文 §4.2 的对比与这条史实方向一致：**在每步都要刷新输入的自回归 decode 上，
   "整图下沉"的头开销与本场景的收益不成正比。**
   （注意：这条史实**不能**单独证明"GE 对 `prepare_input` 无用"，只能说明该路线在本栈
   被判定为性价比不足；`01-ge-capability.md` 提醒 `mode="max-autotune"` 仍是 TorchAir
   的默认值，两条路线是并存而非一者被淘汰。）

---

## 5. 理论下限：即使 GE + 全下沉都做到，`prepare_input` 最少还剩多少？

### 5.1 先定义"下限"的口径

`prepare_input` 的**不可消除的工作**由三件事决定，与 GE 无关：

1. **必须消费调度决策**：scheduler 给的 `SchedulerOutput` 是 Python 对象
   （`num_scheduled_tokens: dict[str,int]`、请求增删列表、preempt 列表），
   worker 必须读它、更新自己的账本；
2. **必须把决策翻译成设备格式**：写块表、写 token、写位置、算 cu_seqlens/slot_mapping；
3. **必须发起图重放**：选 batch descriptor、下发 `aclmdlRIExecuteAsync`。

### 5.2 四层下限（逐层收窄）

| 层级 | 假设 | `prepare_input` 下限 | 由什么决定 | 支撑 |
|---|---|---:|---|---|
| **T0 现状** | 不改造 | **2757.6 µs** | 见 `docs/05` §5bis.8 | `[实测]` |
| **T1 只做"非 GE"优化**：修缓存击穿、decode-only 快路径、GDN 去重、减少对象分配 | 不动架构 | **≈1.9–2.2 ms** | 记账与翻译的 Python 条数 | `[推断]`，见 §6.4 |
| **T2 再做"GE / kernel 化能吃到"的部分**：device 段入图、元数据 kernel 化、固定缓冲 | 固定 shape + 固定地址 | **≈1.6–1.9 ms** | 剩 host 侧的逐请求循环 + 图输入准备 | `[推断]`，见 §6.4 |
| **T3 架构级**：账本迁 device、scheduler 直接产出定长张量、metadata 路径 C++/编译化 | 超出"GE"范畴 | **≈150–400 µs** | **"决策翻译成本"**（见 §5.3） | `[推断]` |

### 5.3 为什么 T3 也不会到 0

**下限的物理含义 = "把 host 侧的调度决策翻译成设备可消费的定长张量"的成本。**
即使 `_update_states` 的账本搬到 device、`condense` 变成 device compaction kernel，
以下三件事依然存在：

1. **scheduler 与 worker 之间的决策传输**：跨进程 RPC（engine core → worker）本身是
   Python/事件通知路径。`[实测]` 真机 perf 里 `eventfd_write` = **1.93%**、
   `pthread_mutex_lock` = **1.78%**（`docs/05` §2.2）——**这些不在 `prepare_input` scope 内，
   但它们构成"决策到达"的绝对下限**（`docs/10` §5.1 估 ~50–150 µs）。
2. **写 pinned staging buffer + H2D 启动**：几个 coalesced memcpy，
   ≈10–40 µs（`[推断]`）。
3. **图/执行分发**：1 次 `aclmdlRIExecuteAsync` + 若干 `ExternalEvent` 的 record/wait，
   ≈5–20 µs（`[推断]`，取决于 §7 V7）。

⇒ **`prepare_input` 的理论下限 ≈ 150–400 µs/步**，
**不是 0，也不由 GE 决定，而由"调度决策的形态"决定**
（是 Python 对象，还是定长张量）。

### 5.4 哪些机制"原理上"不可下沉（关键判断点 5 的收口）

| 机制 | 为什么原理上不可下沉 |
|---|---|
| `_update_states` 的 dict/condense/swap | 操作对象是 Python 容器与请求对象；AscendIR 无此类型 |
| `treat_single_token_prefills_with_state_as_decodes` 的 `.item()` | **host 需要知道布尔结果才能选路径**；把两个分支都编进图（`torch.cond`）也不消除"host 需要知道结果"这一语义前提 |
| `_build_attn_state` 的 5 路选择 | 同上：选中的 `AscendAttentionState` 决定后续 kernel/mask 形态，而重放由 host 发起 |
| `_determine_batch_execution_and_padding` | 输入是 Python int（`num_tokens`），输出是"用哪张图"——**图选择必然在 host** |
| `num_accepted_tokens_event.synchronize()` | 语义就是等 device 结果回 host；捕获期非法 |
| `refresh_metadata` / `SamplingMetadata` | 采样参数、惩罚项、allowed tokens 都是 host 标量与掩码对象 |

> 补充一句"看起来可以但实际不行"的边界：GE **确实**支持图内控制流算子
> （If/While/Case）与 CPU placement 的 Data 节点，所以"把决策放进图"在理论上不是绝对不可能。
> 但那条路要求：①决策只依赖**图内张量**（而我们的决策依赖 `SchedulerOutput` 这个 Python 对象）；
> ②两个分支都能编译成固定结构的子图（我们这里分支会导致 batch 形状变化，
> 与"固定 shape"直接冲突）；③host 仍要发起重放，仍需知道"用哪张图"。
> ⇒ 对本场景，**这三条同时不成立**，所以结论仍是"决策留 host"。

---

## 6. 对 `docs/10-prepare-input-graphification.md` 的审查

`docs/10` 的方向判断我基本同意（**"决定不可图化、构造可图化"**），但**两处论证偏松、
一处估计偏高**。逐条给结论。

### 6.1 支持：判据 1（成本与 device 图化无关）

**成立，且证据比 `docs/10` 写的更硬**：`aclrtSynchronize*` 全族只占 **0.27%**
（`[实测]` `docs/05` §2.4），说明 `prepare_input` 的 2.76 ms 里几乎没有等 device 的成分；
topdown（`frontend_bound 66.01%` / `retiring 12.85%` / IPC 0.771）指向解释器派发。
**"图化不会顺带解决 `prepare_input`"这个结论我完全支持。**

小修正：`docs/10` §1 表中 2.843 → 3.132 ms 的 "+10%" 是**跨模式对比**（eager 臂 vs graph 臂），
不是同一次测量的同配置重复；引用时应标注为"不同臂"，避免被读成"图化让 `prepare_input` 变慢 10%"。

### 6.2 部分挑战：判据 2（三个前提）

`docs/10` 说"#3 已满足、#1 接近满足、唯一缺口是 #2 固定地址"。
**我同意 #2 是最大缺口，但要补三点：**

1. **#2 是必要不充分。** 官方 ACLGraph 文档的原文说明：**"even when the overall graph is
   static"**，attention 仍需 host 侧每步更新参数（§4.1）。
   ⇒ 固定地址能让**图**成立，但**不能消除每步的 host 参数更新**。
   `docs/10` §3 表格里"#3 已满足"的写法容易被误读成"参数已经能在 device 上自洽"；
   实际是"**图的输入可以接受 tensor**"，而不是"**参数不再需要 host 参与**"。
2. **#1"控制流固定"只在 `FULL_DECODE_ONLY` + 纯 decode 下成立**，且是被 padding
   强行钉住的（§4.3）。它不是能力，而是**用 `prepare_input` 的时间买来的状态**。
3. **`docs/10` §3.1 把 `full_graph_fia_v2` 的 tensor 传参当作"前提 3 已满足"的证据**——
   方向对，但要补一句限定：这只覆盖**图捕获那一小段**；eager 路径的
   `seq_lens_list`（`attention_v1.py:1248/1258/1269`）**仍然是 list**，
   而**只要算子签名要 list，host 就必须每步算出来**（§2.3）。
   ⇒ 正确表述是"**元数据 tensor 化已有先例，但覆盖面很小**"。

### 6.3 支持并强化：判据 3（"决定"不可图化）

成立。`docs/10` §5.3 说"下限不是 0，而是把调度决策翻译成设备格式的成本"——
本文 §5 把它量化成 **150–400 µs**，并给出了三条不因 GE 消失的成分。

### 6.4 推翻：§5.2 的收益估计 "2.76 ms → 1.2–1.5 ms" **站不住**

`docs/10` §5.2 清单里，第 1 项给的是
**"3 个 GDN KV group 共享 metadata，若 2/3 是重复，可省 ~600 µs"**。这个数偏乐观，三个原因：

1. **"2/3 是重复"只对一部分子步骤成立**。三组的**输入共享**（`gdn_query_start_loc` 相同，
   `cm` 是 `copy(cm_base)` 后只改 `block_table_tensor`/`slot_mapping`，
   `[源码]` `model_runner_v1.py:3150–3156`），但**产出不共享**：
   `non_spec_state_indices_tensor`（每组不同的 cache group）、`_pad_non_spec_decode_graph_inputs`
   的约 12 次 device op **必须每组各做一遍**。按 §2.4 的分解，
   **跨组真正相同的部分大约是 `treat_single_token`(56.4) + `split_decodes`(40.4) +
   `compute_num_computed_tokens`(43.2) + `build_actual_seq_lengths`(54.3) + `ctor`(7.2)
   ≈ 200 µs/次**，3 组去重后省 **≈400 µs（不是 600）**，且其中
   `compute_num_computed_tokens` 的 ≈130 µs 应该通过**修 `.replace()` 缓存**拿到，
   **不能与去重重复计**。
2. **第 3 项"静态缓冲省 200–400 µs"与第 1 项高度重叠**：`build_actual_seq_lengths` 的
   `empty_like`、`pad_graph_inputs` 的 `copy_` 都已经被第 1 项算过一次。
3. **`_pad_non_spec_decode_graph_inputs`（285.9 µs）不可去重**，且它的单位成本
   （≈24 µs/op）说明其中很大一块是"小 op 的固定税"——**只有入图才能省**，
   而"入图"还需要先补前提 #2（固定地址）。

**我的修正估计**（方法：从 `docs/05` §5bis 的 p50 逐段加减，重叠项去重）：

| 阶段 | 措施 | 收益 | 残余 |
|---|---|---:|---:|
| 起点 | — | — | **2757.6 µs** |
| S1 | 修 `.replace()` 缓存击穿（**不依赖 GE**） | −130 | 2628 |
| S2 | decode-only 快路径：短路 `treat_single_token` + `split_decodes` + GDN 三组去重（**不依赖 GE**） | −430 ~ −500 | ≈2100–2200 |
| S3 | full-attn builder 去 `tolist()` / `pin_memory()` 重建 | −40 ~ −100 | ≈2000–2160 |
| S4 | **GE / kernel 化**：device 段入图（`pad_graph_inputs` + `build_actual_seq_lengths` + 上屏家族 + slot_mapping launch） | −300 ~ −450 | **≈1600–1850** |
| S5 | 要到 1.2–1.5 ms | 还需 **C++ 化 metadata 路径 / 账本迁 device**，即架构改造 | — |

⇒ **结论：`docs/10` 的方向对，"图化 + 静态缓冲 + 去冗余"确实能把 `prepare_input` 砍下去，
但「只靠 GE / 图化」的落点大约是 ≈1.6–1.9 ms，不是 1.2–1.5 ms；
1.2–1.5 ms 是一个"图化 + C++/账本级改造"的联合估计，不应记在图化账上。**
（这个估计的主要误差来自两块未知量：GDN builder 残差 ≈123 µs/步的真实构成，
以及 ≈207 µs 的未归因时间——`docs/05` 已如实标注。）

> 顺带肯定 `docs/10` 的一点：它列的第 2 项（`.replace()` 缓存击穿，~130 µs）与
> 第 5 项（metadata 部分下沉）都是对的，只是**第 2 项根本不需要 GE**，
> 属于"低垂果实"，应优先于任何 GE 化工程。

### 6.5 补一条 `docs/10` 没提的间接收益（关键判断点 6）

问题是：**GE 让整个 forward 变成 1 次 `aclmdlRIExecuteAsync`，host 从 forward 里省出时间——
这对 `prepare_input` 算不算收益？**

**答案：算 E2E 收益，不算 `prepare_input` 收益；而且必须分开记账。**

- **不算**：`prepare_input` 是独立 scope（`[源码]` `model_runner_v1.py:1859–2082`），
  forward 的 host 时间**不在这段里**。任何"`prepare_input` 变快了"的说法都不能引用它。
- **算 E2E**：异步调度下 `T_step ≥ max(T_cpu_engine_step, T_dev_step)`。
  当 **CPU 是瓶颈时**（本例 `T_cpu` 侧有 2.76 ms 的 `prepare_input`），
  **释放 forward 的 host 时间不会提高吞吐**——瓶颈在别处。
  只有当 `T_cpu` 降到与 `T_dev` 同阶、且 forward 的 host 段仍在 CPU 预算内时，
  这笔收益才会兑现。
- **可量化**：`[实测]` forward p50 = **830.6 µs**（B=1，`docs/05` §5bis.8），
  其中相当部分是 host 侧逐算子派发（`docs/10` §2.2 的估算 + §7 V5 待测）。
  ⇒ **上限 ≈800 µs 的 CPU 预算**，但**它只在 `T_cpu` 已不是瓶颈之后才有意义**。

> 记账纪律：**GE 对 forward 是"降 CPU"，对 `prepare_input` 是"降派发"，
> 两者不可混入同一个百分比。** 若 `docs/10` §5.1 的 1 ms 预算表要修订，建议加一行
> "forward 的 host 段（与 `prepare_input` 顺序串行、不占同一预算）"，
> 并明确它不属于 `prepare_input` 的优化项。

---

## 7. 不确定的、需要实测验证的

每条给出可执行方法。V1–V4、V8、V9 可在无卡 harness 上先做；V5–V7 需真机
（**本文未执行任何验证，全部标注为待办**）。

| # | 不确定项 | 为什么重要 | 验证方法 | 判据 |
|---:|---|---|---|---|
| V1 | GDN 三组**真正可去重**的比例 | 决定 §6.4 S2 的 430–500 µs 是否成立 | 让 3 个 GDN builder 共享"分类结果 + `actual_seq_lengths`"（各组 buffer 仍独立），A/B 测 `pi: am.builder_build@AscendGDNAttentionMetadataBuilder` | 若省 ≥350 µs，"去重"优先级最高 |
| V2 | `.replace()` 缓存修复的真实收益 | 机理已验证（perf 子树 67.5% 在 `PyNumber_Subtract`），数值未验证 | 改 `_treat_single_token_prefills_with_state_as_decodes` 保留 `_num_computed_tokens_cache`，A/B | 预期 ~130 µs |
| V3 | decode-only 快路径（短路 `treat_single_token` + `split_decodes`） | ≈290 µs，且**不需要 GE** | 在 `build()` 入口加 `num_prefills == 0 and spec 关闭` 短路，A/B | 若 ≥250 µs，与 V2 一起构成"低垂果实" |
| V4 | `pad_graph_inputs` 285.9 µs 里 launch 与真实拷贝各占多少 | 决定"入图"能拿多少 | 逐 op 插桩（`fill_`/`copy_`/`expand_as` 各一次计时）+ 用 `torch.npu.graph_task_group_begin/end` 把整个函数包成 1 次调用做对照 | 若 host 侧占 ≥60%，入图收益 ≈170 µs+ |
| V5 | 每层 `graph_task_update_begin/end` 的 host 成本 | 决定 ACLGraph 路线的收益上限（`docs/10` §8 V3） | 在 `update_graph_params` 加探针，测 6 层 FIA + 3 组 GDN 的 update 总时长 | >100 µs 则"图化省 host"在 920B 不成立 |
| V6 | GDN builder 残差 **≈123 µs/步**的构成 | 最大的未解释块 | `perf report` 的 children 分解 + 对照 `_copy_sequence_indices_to_device` / `.replace()` / 属性赋值 | 定位到具体符号才能计入改造清单 |
| V7 | 920B 上一次**图执行头开销**（`aclmdlRIExecuteAsync` + 输入/地址刷新） | 决定 GE 整图 vs 分段图的性价比 | 微基准：捕获一个 10 节点小图，测纯 replay 的 host 时间；再按"刷新 N 个输入"扫描 | 与官方盘古 71B 的 ≈2 ms/1600 输入做标定；若头开销 >200 µs，"整图"对 B=1 可能是负收益 |
| V8 | `unicodekeys_lookup_unicode`(1.68%) 中有多少来自 `GDNAttentionMetadata(...)` 的 22 kwargs | 决定 ctor 优化收益 | 改成位置参数/`__slots__`/直接赋值，A/B + perf 对照该符号 | 若该符号降 ≥1 pp，≈省 30–50 µs |
| V9 | 空 batch / DP `_dummy_run` 路径对均值的影响 | 防止把空转算成收益 | 按 `total_num_scheduled_tokens == 0 / > 0` 分层重算（`docs/01` §1.3 已提示） | 分层前后占比差 >3 pp，则所有结论需分层重述 |

补充两条**结论性待验证**（不属于性能而是可行性）：

| # | 待验证 | 方法 |
|---:|---|---|
| V10 | 把 `prepare_input` 的产物"捕获"进图在技术上是否可能 | 只做**只读实验**：写一个最小脚本，在 `aclmdlRICaptureBegin/End` 之间调用 `AscendGDNAttentionMetadataBuilder.build`，观察是否在 `event.synchronize()` / `.item()` 处报错（预期会失败，用于把 §1.4 的静态判断变成实测证据） |
| V11 | 若把 `_pad_non_spec_decode_graph_inputs` 改成"提前切片、不做 padding"，是否仍满足 FIA 的 TND 约束 | 在 `FULL_DECODE_ONLY` 下关闭 padding 做 A/B，观察是否报 561002（`[源码]` `attention_v1.py:336–354` 注释里的批处理长度校验错误码） |

---

## 8. 附录：证据索引

### 8.1 官方文档（均 2026-09-24 访问）

| 用途 | 出处 |
|---|---|
| ACLGraph 捕获边界（捕获的是 Stream 任务） | <https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/acldevg/runtime_doc_dev_0045.html> |
| 捕获期禁令（Event/Stream 同步非法、H2D 必须锁页） | <https://gitcode.com/cann/runtime/blob/54cd3f51d17924a9a1f130a653e8939d2705c142/docs/02_dev_guide/04-01_%E5%8D%95%E6%B5%81%E6%8D%95%E8%8E%B7.md> |
| GE 模型下沉原理（整图下发、头开销 ≈2 ms 数据点） | <https://www.hiascend.com/developer/techArticles/20240715-1> |
| GE 动态 shape Host 调度（Host Kernel、小 shape 算子留 host、Host 缓存） | <https://www.hiascend.com/developer/techArticles/20250911-1> |
| GE 小 shape 算子计算优化 | <https://www.hiascend.com/developer/techArticles/20240726-1> |
| AscendIR 是静态计算图，算子+张量为基本单元 | <https://gitcode.com/cann/ge/blob/a74342b574f03ffe45e059c1a0fa74e50ccf5733/docs/architecture.md> |
| 图输入形态 `ge.Data(dtype, shape, placement)` / `ge.Const` / `_graph_unknown_flag` | <https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/cases/dynamic_static_graph/presentation.md> |
| aclgraph 与 GE 两条图模式的定性区别 | <https://gitcode.com/Ascend/torchair/blob/master/docs/zh/overview.md> |
| npugraph_ex 基础功能与 `fullgraph`/图中断语义 | <https://github.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/quick_start.md> |
| 静态 kernel 编译的启动代价（分钟~数十分钟） | torchair `docs/zh/npugraph_ex/basic/static_kernel_compile.md`；vllm-ascend `docs/source/user_guide/feature_guide/graph_mode.md` |
| graph break（数据依赖控制流 / `.item()`） | <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html> |
| vLLM 侧 CUDA Graph 模式语义（PIECEWISE / FULL / FULL_DECODE_ONLY） | <https://docs.vllm.ai/en/stable/design/cuda_graphs/> |
| vllm-ascend ACLGraph 设计文档（含 host 侧参数更新原文） | <https://docs.vllm.ai/projects/ascend/zh-cn/main/developer_guide/Design_Documents/ACL_Graph.html> |

### 8.2 本项目内部证据

| 用途 | 文件 |
|---|---|
| scope 边界、四层口径、S1–S26 与 `_update_states` 分解 | `docs/01-prepare-input-code-logic.md` |
| 真机 p50 / topdown / 火焰图 / GDN 六段分解 / 微基准 | `docs/05-hotspots.md` §2.3、§2.4、§4、§5bis |
| 本文审查对象 | `docs/10-prepare-input-graphification.md` |
| 静态步骤表（§3 的 id 对应关系） | `data/static/prepare-input-steps.json` |
| GE / ACLGraph 的一手源码与官方文档证据 | `docs/11-ge-whole-graph/00-source-evidence.md` |
| GE / TorchAir 的能力清单与官方依据（本文 §2.3、§4.5 交叉引用） | `docs/11-ge-whole-graph/01-ge-capability.md` |
| vllm-ascend 历史上使能/移除 GE 路径的脉络（本文 §4.5 引用） | `docs/11-ge-whole-graph/02-why-not-enabled.md` |

### 8.3 关键源码位置

**vllm-ascend 0.26.0rc1（`f2f74a16c`）**

```text
vllm_ascend/worker/model_runner_v1.py:1859–2082   prepare_input scope（2065 调 _build_attention_metadata）
vllm_ascend/worker/model_runner_v1.py:1320–1347   _build_attn_state（5 路 host 分类）
vllm_ascend/worker/model_runner_v1.py:834–881     _pad_query_start_loc_for_fia
vllm_ascend/worker/model_runner_v1.py:1101–1127   num_accepted_tokens 同步（唯一强同步点）
vllm_ascend/worker/model_runner_v1.py:1246–1250   compute_slot_mapping 调用点
vllm_ascend/worker/model_runner_v1.py:3150–3166   per-KV-group cm 复制与 builder 分发
vllm_ascend/ops/gdn_attn_builder.py:66–78         _treat_single_token_prefills_with_state_as_decodes（.item()）
vllm_ascend/ops/gdn_attn_builder.py:128–142       _build_actual_seq_lengths（empty_like / out=）
vllm_ascend/ops/gdn_attn_builder.py:332–369       _pad_non_spec_decode_graph_inputs（预分配 buffer + ~12 次 device op）
vllm_ascend/ops/gdn_attn_builder.py:783/843–860   full-graph 专用分支
vllm_ascend/ops/gdn_attn_builder.py:860–885       GDNAttentionMetadata(...) 22 个 kwargs 的 ctor
vllm_ascend/attention/attention_v1.py:291–392     AscendAttentionMetadataBuilder.build（tolist / pin_memory）
vllm_ascend/attention/attention_v1.py:476–540     每步 graph_task_update（paged attention）
vllm_ascend/attention/attention_v1.py:1028–1029   full_graph_fia_v2 用 tensor
vllm_ascend/attention/attention_v1.py:1248/1258/1269  eager 路径用 seq_lens_list
vllm_ascend/ops/triton/compute_slot_mapping.py:12  slot_mapping 已是 Triton kernel
vllm_ascend/compilation/compiler_interface.py:132–135  mode="reduce-overhead"，不做 FX→Ascend IR 变换
```

**vLLM 0.26.0（`refs/vllm`）**

```text
vllm/v1/worker/gpu_model_runner.py:1169–1543      _update_states
vllm/v1/worker/gpu_model_runner.py:3877–3922      _determine_batch_execution_and_padding
vllm/v1/worker/gpu_input_batch.py:338/569/686/814 add_request / swap_states / condense / refresh_metadata
vllm/v1/attention/backend.py:528–534              compute_num_computed_tokens（缓存被击穿点）
vllm/v1/worker/block_table.py:166                 slot_mapping 也已是 kernel
vllm/v1/worker/gpu/cudagraph_utils.py:360         CudagraphDispatcher.dispatch
```

---

## 9. 一页结论（给不想读全文的人）

1. **`prepare_input` 不能"使能 GE"**——它不是一段可被编译的计算，而是**图的输入准备**。
   官方文档把它定义为"obtain the inputs and their corresponding attention metadata"。
2. **GE 能省 L2 派发（≈400–500 µs）与部分 L1（≈0–140 µs），不能省 L4（0）**；
   L3 只省算术不省值，且与 L2 重叠。
3. **前提要付代价**：静态 shape 靠 `prepare_input` 每步 padding 买单
   （`pad_graph_inputs` 285.9 µs 就是账单的一部分）；静态地址仍是缺口。
4. **理论下限 ≈150–400 µs/步**，由"调度决策的形态"决定，不由 GE 决定。
5. **最优路径不是上 GE，而是**：①修 `.replace()` 缓存（−130 µs）；
   ②decode-only 快路径 + GDN 三组去重（−430 ~ −500 µs）；
   ③再做 device 段入图/kernel 化（−300 ~ −450 µs）⇒ **≈1.6–1.9 ms**；
   要进 1 ms 必须做 **C++ 化 / 账本迁 device** 的架构改造。
