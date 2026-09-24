# 03 · 降低 torch 层与 LAL 层 host 开销：公开做法调研

> 调研对象：vLLM 0.26.0（`568afb3a1`）+ vllm-ascend 0.26.0rc1（`f2f74a16c`），
> Kunpeng 920B（aarch64）+ Ascend A3。
> 调研方式：**纯文献 / 网络 + 源码**（未上机、未跑 NPU）。所有引用给 URL + 访问日期
> **2026-09-24**（Asia/Shanghai）。本地源码引用一律给 `文件:行号`（`refs/vllm` 只读快照 +
> `/home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend`）。
>
> 本文与 [`../10-prepare-input-graphification.md`](../10-prepare-input-graphification.md) 互补：
> 那篇回答"prepare_input 能不能图化"，本篇回答"**在 torch 层与 LAL 层还能省什么**"，
> 并把每条结论挂到公开证据上。

---

## 0. 结论速览（先看这 8 条）

| # | 结论 | 依据 |
|---|---|---|
| 1 | **我们的 3–12 µs/op 是"整链"数字，不是 dispatcher 数字。** 920B 上纯 ATen 派发实测 **1.0–2.3 µs/次**（本项目 `data/model/microbench-torch.csv`），与 PyTorch 官方文档的"x86 上约 2 µs/次"同量级 ⇒ **ARM 的 dispatcher 不慢**，多出来的部分是 NPU 下发层（aclnn/aclrtLaunchKernel）+ 框架层。 | A |
| 2 | `at::_ops::*::call` 的固定成本是**几千条指令**量级（`at::empty` 微基准里 `wrap_kernel_functor_unboxed::call` ≈ 2 000 条指令 / 1.24%），不是"几十 µs"。**想靠 dispatcher 层的微优化拿回 900 µs 是拿不回来的。** | A |
| 3 | 官方确有"快路径"API：`torch._C._DisableTorchDispatch`（别名 `torch.utils._mode_utils.no_dispatch`）、`torch._C.DisableTorchFunctionSubclass` / `DisableTorchFunction`、`torch._C._get_tracing_state()` 短路。但它们**只在有活跃 dispatch mode / Tensor 子类 / JIT tracing 时才有收益**；我们基线三者都没有 ⇒ 收益 ≈ 0。 | A |
| 4 | 反例（重要）：**用 `TorchDispatchMode` 做轻量拦截一点都不轻**——PyTorch 自己的 FLOP counter 实测 **+30–40 µs/op**。任何"给每个算子挂一层 Python"的方案在我们这里都会把 908 µs 变成几 ms。 | A |
| 5 | **`torch.compile` 对"元数据构造"是净负收益。** 官方文档承认 guard 求值/前图字节码**每次调用都要跑**；小图上编译后的运行时开销大到 PyTorch 开了专门 issue；SGLang 更直接：prefill 图切分从 `torch.compile` 换成自研 BCG 后，**每次 replay 少付 Dynamo guard + dispatch，快 17%**，且图构建快 3.8–5.2×。Ascend 侧更硬：vllm-ascend 明确 **ACL graph 路径下禁用 Inductor**。 | A |
| 6 | 图下发**不会减少 host 计算**，它减少的是 host 与 device 之间的"每次下发"。NVIDIA 官方数字：同一 kernel，无图 9.6 µs/次（含开销，kernel 本体 2.9 µs）→ 图 replay **3.4 µs/次**（V100）；CUDA 12.6 上直链图 repeat launch ≈ **2.5 µs + ~1 ns/node**。我们自己的数据也印证：图化后 `prepare input` **2.843 ms → 3.132 ms（+10%）**，变的是分母（单步 −87%）。 | A |
| 7 | **我们不是第一个撞上这条墙的。** 上游 vLLM 有一个 PR 的标题就是"[Attention] Move GDN common metadata compute out of per-group build"（#52297），自述 `GDNAttentionMetadataBuilder.build()` **900 µs → 300 µs**、BS=1 端到端 **+61%**（H200，DFlash）；vllm-ascend 有一个 PR 的标题是"[Performance][GDN] Share group-invariant metadata across kv-cache groups"（#16246），自述 3 组 **6.3 ms → 4.5 ms/步（−28.6%）**。两个都还**未合并**。 | B |
| 8 | 结论：**我们的两份钱要分开花** —— "把 3 组 GDN build 压成 1 组"（对应上游两个 PR，收益 260–600 µs/步）是**重复劳动但可以直接抄**；"`.replace()` 造的临时对象让 `num_computed_tokens` 缓存每次落空"（vLLM main 至今未修，已逐行核对源码）是**我们的独有发现**，收益 80–200 µs/步。 | A（机制）／B（收益折算） |

---

## 1. 口径与证据强度分级

本文所有数字按三级标注：

| 级别 | 含义 | 例子 |
|---|---|---|
| **A** | 一手来源（官方文档 / merged PR / 项目自测 CSV），有可核对数字 | NVIDIA 官方博客、PyTorch 官方 benchmark README、本项目 `data/model/*.csv` |
| **B** | 一手来源但只有自述数字，无第三方复核（未合并 PR、issue 里的个人测量） | vLLM #52297 的 "900 µs → 300 µs" |
| **C** | 二手转述 / 博客推断 / 需要我们自己换算 | 行业博客里的"launch overhead 20–200 µs" |

**一条纪律**：本文出现的"µs/步"预测，凡不是引用他人数字的，都写清换算过程（派发次数 × 单次成本），
不接受"看起来能省一半"这种说法。我们的单次成本基准取本项目实测：

| 基准 | 值 | 来源 |
|---|---|---|
| 纯 ATen CPU 派发（`add(out=)` / `empty`） | **1.61 µs/次**（p10/p90 = 1.58/1.63） | `data/model/microbench-torch.csv`，920B aarch64，torch 2.9.0+cpu，单线程 |
| 纯 ATen 分配（`zeros(3,8)`） | 2.30 µs/次 | 同上 |
| `torch.index_select` / numpy→tensor | 1.00–1.03 µs/次 | 同上 |
| numpy 小算子（`np.array(list)` / `np.nonzero`） | 0.40 / 0.53 µs/次 | `data/model/microbench-numpy.csv` |
| numpy 复合（`_get_cumsum_and_arange`） | 8.08 µs/次 | 同上 |
| **一次小 NPU 算子派发（整链：Python→torch→aclnn→device 队列）** | **3–12 µs/次** | `docs/03-share-in-inference.md:127`（`k_h2d_small`，真机微基准 `scripts/npu_dispatch_bench.py`） |
| 单步 `prepare_input`（B=1 decode，`FULL_DECODE_ONLY`） | **2.76 ms**，占单步 55.3% | `docs/00-INDEX.md` §0 |
| 头号热点 `AscendGDNAttentionMetadataBuilder.build` | **3 次 × 303 µs = 908 µs/步（29.3%）** | `docs/00-INDEX.md` §3 |

> ⚠️ 注意 1.61 µs 与 3–12 µs 的差别：前者是**纯 host 计算**（不碰 device），
> 后者含**下发到 NPU**（驱动 + 队列）。二者不可混用。说"省一次 torch 算子"
> 到底值 1.6 µs 还是 12 µs，取决于该算子是不是 device 算子。

---

## 2. PyTorch dispatcher 的 host 成本结构

### 2.1 一次 `at::_ops::*::call` 到底经过什么

PyTorch 官方 dev-discuss《Dispatcher Performance and Inlining》（2021-01-28，
<https://dev-discuss.pytorch.org/t/dispatcher-performance-and-inlining-a-report-on-two-days-spent-on-dispatcher-performance/109>）
给出的调用链（以 `at::empty` 为例）：

```
at::empty
 → Dispatcher::call
   → Dispatcher::callWithDispatchKey
     → OperatorEntry::lookup → KernelFunction::isValid
     → KernelFunction::call
       → KernelFunction::callUnboxedKernelFunction
         → 注册的 unboxed kernel 函数指针（此处不可内联）
   → Dispatcher::singleton …
```

- **关键点 1**：`callWithDispatchKey` 会被内联进每个算子实现里，所以"参数处理"与派发逻辑交错，
  优化它需要逐个 slow path 外提（该文原话）。
- **关键点 2**：作者用 `perf stat --repeat 5` + 关 turbo 才能测出改进，且 "changes on the order of
  **100 ns**" —— **这是 dispatcher 层微优化的量级**。

更硬的指令数证据来自 PyTorch PR #50848（<https://github.com/pytorch/pytorch/pull/50848>，2020）：

| 微基准 | 热点函数 | 指令数 | 占比 |
|---|---|---|---|
| `at::add`（scalar 重载） | `function_ref::callback_fn` | 6 700 | 0.36% |
| `at::empty` | `wrap_kernel_functor_unboxed::call` | 2 000 | 1.24% |

即：**一次算子调用的 dispatcher 本体是"几千条指令"，在 3 GHz x86 上约 0.5–1 µs**，
而**不是** 3–12 µs。这个差距必须由别的东西解释（见 §2.4）。

### 2.2 公开的"µs/op"数字（汇总表）

| 来源（时间） | 场景 | 公开数字 | 证据 |
|---|---|---|---|
| PyTorch `benchmarks/overrides_benchmark/README.md`（现行 main）<br><https://github.com/pytorch/pytorch/blob/main/benchmarks/overrides_benchmark/README.md> | `torch` 函数作用在 `torch.Tensor` 上 | **~2 µs/次**；`__torch_function__` 对普通 `Tensor` **零额外开销**；对 Tensor 子类"很小"；对 Tensor-like "几 µs" | A |
| pytorch#41383（2020-07-14）<br><https://github.com/pytorch/pytorch/issues/41383> | 小张量最简单的逐元素算子（torch 1.5） | 报告者实测 **~7 µs/次**（含 autograd）；`requires_grad=False` 时约 5 µs；maintainer：`x*x` **≈2 µs**，其中"**约一半是 Python 解析开销，C++ API 约 1 µs**" | A |
| dev-discuss《Skipping Dispatcher with LazyTensor》（2022-06-12）<br><https://dev-discuss.pytorch.org/t/skipping-dispatcher-with-lazytensor/634> | 绕过整个 dispatcher 直调 LazyNativeFunctions | 微基准（8–32 个可融合算子）**整体快 10–12%**；张量越大收益越低（算子本身耗时占比上升） | A |
| pytorch#187949 "PyObject Dispatch"（2026，commit `f7155a0`）<br><https://github.com/pytorch/pytorch/commit/f7155a0e49678bac796f36ed11ac741e1c3ed254> | 自定义算子（Python kernel）**一次 Python↔C++ 往返** | 10 输入 / 3 输出 no-op 自定义算子：**2.6 µs → 300 ns** | A |
| dev-discuss《The "Ideal" PyTorch FLOP Counter (with `__torch_dispatch__`)》（2022-02-18）<br><https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505> | `TorchDispatchMode` / `__torch_dispatch__` 拦截**每个**算子 | **+30–40 µs/op** | A |
| pytorch#72746（TorchScript 小模型 CPU-bound） | 单次 `aten::addmm` 下发到 GPU | **~70 µs**（含 launch，小批量载体） | A |

### 2.3 官方的"快路径"API 与其适用条件

（源码核对日期 2026-09-24）

| API | 位置 | 作用 | 什么时候才有收益 |
|---|---|---|---|
| `torch._C._DisableTorchDispatch` / `torch.utils._mode_utils.no_dispatch` | `torch/utils/_mode_utils.py`（`no_dispatch = torch._C._DisableTorchDispatch`）；stub 见 `torch/_C/__init__.pyi.in:1582` | 临时**清空 Python dispatch key**，使算子不再回到 `__torch_dispatch__` | **仅当有活跃的 dispatch mode / Tensor 子类时**。典型用法是在 mode 的 `__torch_dispatch__` 内部调用以避免无限递归。我们基线没有 mode ⇒ 无效 |
| `torch._C.DisableTorchFunctionSubclass` / `DisableTorchFunction` | `torch/_C/__init__.pyi.in:238-240` | 跳过 `__torch_function__` 协议检查 | 仅当输入可能是 Tensor 子类/Tensor-like 时。我们全用原生 `torch.Tensor` ⇒ 无效 |
| `torch._C._get_tracing_state()` 短路 | 用于 `torch/_tensor.py`、`torch/nn/functional.py` 等（GitHub code search：repo 内 17 处） | 在 JIT tracing 下跳过 `__torch_function__` 处理 | 非 tracing 时该函数只是一次 C 调用（几十 ns）⇒ **不是瓶颈** |
| `torch.utils._python_dispatch._disable_current_modes()` | `torch/utils/_python_dispatch.py` | 纯 Python 地逐个弹出/恢复 mode 栈 | 调试工具用；成本比 `_DisableTorchDispatch` 高 |

**我们的基线（eager + 无 mode + 无子类 + 非 tracing）这四条路全部收益 ≈ 0。**
这一点值得写进结论，因为它能挡掉一类"看起来很美"的优化提案。

### 2.4 那 3–12 µs/op 从哪来？（以及 ARM 是否异常）

把数字摆在一起：

| 环节 | x86 公开值 | 920B 实测 | 结论 |
|---|---|---|---|
| 纯 ATen 派发（host-only，无 device） | **~2 µs**（官方 README，2020–2021） | **1.61 µs**（`add(out=)`，2026 实测） | **同量级，ARM 不慢** |
| 含 device 下发的整链（Python + torch + 驱动 + 队列） | CUDA 生态公开值 **5–10 µs/kernel**；NVIDIA 官方口径"DL 应用里每操作 20–200 µs（含框架层）"<br><https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html> | **3–12 µs/op** | 落在同一区间内，**合理** |

拆解我们的 3–12 µs：`ATen 派发 1.6 µs` + `Python/C 胶水（aclnn 包装、参数转换、tensor 构造）` +
`aclrtLaunchKernel / aclnn 下发` + 可能的 queue 背压。**其中只有第一项是"torch 的锅"**，
其余是 LAL（Ascend 下为 aclnn/CANN）与框架层的锅。这条结论决定了优化方向：

> **不是"把 torch 换掉"，而是"减少调用条数 + 减少重复调用 + 让每次调用更便宜"。**

这恰好与我们火焰图的指纹一致（`docs/00-INDEX.md` §2）：`frontend_bound 66.01%`、
`_PyType_Lookup 1.67%`、`_PyObject_GenericGetAttrWithDict 1.36%` —— 典型"小对象多、
属性查找多、逐条派发"的形态。

### 2.5 本节对我们负载的适用性

| 做法 | 适用性 | 预期收益 | 风险 |
|---|---|---|---|
| `no_dispatch` 包住元数据构造 | ❌ 不适用（无活跃 mode） | ~0 | 无 |
| `DisableTorchFunctionSubclass` 包住元数据构造 | ❌ 不适用（无子类） | ~0 | 无 |
| `_get_tracing_state` 短路 | ❌ 非 tracing | ~0 | 无 |
| **减少算子条数（合并/缓存/去重）** | ✅ 直接适用 | 每条 device 算子 3–12 µs | 需逐条做语义等价性证明 |
| **把 CPU 侧簿记从 torch 张量换成 numpy/原生 list** | ✅ 适用（仅 CPU 张量场景） | 1.6 µs → 0.4–0.9 µs/次 | 与 device 张量混用会引入 D2H |

---

## 3. `torch.compile` / TorchInductor 对"元数据构造"是净负收益

### 3.1 机制：编译后的函数**每次调用**都要付固定成本

PyTorch 官方文档《Reducing Guard Overhead》
（<https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.reducing_guard_overhead.html>，访问 2026-09-24）
把成本构成写得很直白：

1. **guards**："This check happens on **every call**, so for functions that are **cheap relative to their
   guard set**, guard evaluation can become a **measurable fraction of runtime**."
2. **前图字节码（pre-graph bytecode）**：每次调用都要为形参/缓冲做 marshal，同样每次都要跑。
3. 官方给的四个"减 overhead"旋钮（`install_free_tensors`、`guard_filter_fn=skip_guard_on_all_nn_modules_unsafe`、
   `use_recursive_dict_tags_for_guards`、`set_stance(skip_guard_eval_unsafe=True)`）**都要求放弃部分 guard 正确性**。

### 3.2 公开量化数字

| 来源（时间） | 场景 | 数字 | 证据 |
|---|---|---|---|
| pytorch#185886（2026，inference 大模型）<br><https://github.com/pytorch/pytorch/issues/185886> | "TorchDynamo Cache Lookup"（纯 guard 求值，**不含**模型计算） | 2.7 → 2.12：**~1.36 ms → ~5.65 ms**；根因是 1 755 个 `LAMBDA_GUARD`（float 属性 + 对象 alias），`inline_inbuilt_nn_modules` 强制开启后放大；组合 `specialize_float=True` + 递归 dict tag 只回到 ~3.57 ms | A |
| pytorch#161783（2025-08-29）"torch.compile has runtime overhead on small graphs"<br><https://github.com/pytorch/pytorch/issues/161783> | 用小图（rope）当"kernel 生成器" | "AOTAutograd is known to be **a lot, if not most, of the overhead**"；后续定位：`record_function` 低效占"最多一半"，`eval_frame.compile_wrapper` 是 cProfile 第一大项，AOT 的 prologue/epilogue 去掉可省 **~30%**；随后 PR #163747 落地，本地测试"**运行时开销降低近 50%**"（即：**砍半之后还是很贵**） | A |
| pytorch#184133（2026，AOT 运行时 wrapper 微优化） | 继续削 wrapper | 4 项微优化（跳过 profiler 闭包、单例 `nullcontext`、去掉空 dict、退化推理直接返回） | B |
| **SGLang《Breakable CUDA Graph》**（2026，实战对照）<br><https://www.sglang.io/blog/breakable-cuda-graph> | prefill 图切分：`torch.compile` 版（tc_piecewise）vs 自研无编译器版（BCG），同 forward | ① `torch.compile` 占"准备 prefill 图"时间的 **78–86%**；② 图构建 BCG 比 tc_piecewise **快 3.8–5.2×**（GLM-5.2：35.2 s vs 183.1 s）；③ **replay 阶段** BCG 1.70× vs tc_piecewise 1.45×（相对 eager）⇒ 仅因为"每次 replay 要回 Dynamo 的 guard + dispatch"，就慢 ~17% | A |
| Fireworks《Speed, Python: Pick Two》(2023-08-29)<br><https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning> | LLaMA-2-7B, bs=1, A100 | 手写 CUDA Graph 与 `torch.compile` **都到 69 tok/s**（vs 30 tok/s 无图）；但 `torch.compile` **warm-up 约 3 分钟**，手写图 <1 s | A |

### 3.3 graph break 的代价

官方 Dynamo 文档（`programming_model.dynamo_core_concepts`）：一次 graph break =
**编译已捕获的子图 → 回到普通 Python 跑不支持的部分 → 重新开始 trace**。对"元数据构造"这类代码，
graph break 几乎是必然的，因为里面有：

- `dataclasses.replace` / dict / list 操作（`CommonAttentionMetadata.replace`，`refs/vllm/vllm/v1/attention/backend.py:497`）
- 数据相关的控制流（`if not torch.any(...).item(): return ...`，见 vllm-ascend `vllm_ascend/ops/gdn_attn_builder.py:78`）
- `.tolist()` / `.item()`（把 device 值搬到 host，见 `vllm_ascend/attention/attention_v1.py:337-339`）

⇒ 每个 break 都要重付 guard，且 Dynamo 会把 `__torch_dispatch__`、Tensor 属性访问都拉进 Python，
正好击中我们火焰图里最热的 `_PyType_Lookup` / `_PyObject_GenericGetAttrWithDict`。

### 3.4 结论（可直接引用）

> 对 ~10 个张量算子 + 控制流 + dataclass 构造的"元数据构造"代码，
> `torch.compile` 的**每次调用固定成本（guard 求值 + 前图字节码 + AOT wrapper）在数十 µs 量级**，
> 与被编译代码本身的 **20–60 µs** 同量级甚至更大；再叠加 graph break 后的
> "Python 段"成本，**净收益为负**。
>
> 硬证据：① PyTorch 自己为"小图上的运行时开销"开了 issue；
> ② SGLang 用自研 BCG 取代 `torch.compile` 做图切分，明确指认 Dynamo guard+dispatch 是 replay 慢 17% 的原因；
> ③ **vllm-ascend 官方文档明确写 `use_inductor` 在 ACL graph 路径下被禁用**
> （<https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html>）——
> Ascend 上连 Inductor codegen 都没有，指望 torch.compile 加速元数据是方向性错误。

---

## 4. 图下发对 host 侧的解放：公开数字（支撑"prepare_input 能否图化"）

> 📎 本节只负责"**公开数字**"。图谱系、ACLGraph 的三层机制、
> `graph_task_update` 的能力边界、`prepare_input` 逐项可图化分解，见同目录
> [`05-graph-dispatch-host-side.md`](./05-graph-dispatch-host-side.md)（另一路调研）；
> 两篇的结论一致：**图化省下发次数，不省 host 计算**。

### 4.1 NVIDIA 官方基础数据（这是所有"图能省多少"的天花板参照）

| 来源（时间） | 场景 | 数字 | 证据 |
|---|---|---|---|
| NVIDIA《Getting Started with CUDA Graphs》(2019-09-05)<br><https://developer.nvidia.com/blog/cuda-graphs/> | V100，1 000 次迭代 × 20 个小 kernel，kernel 本体 2.9 µs | 无图 **9.6 µs/kernel**（含开销）；多流预热 3.8 µs；**CUDA Graph 3.4 µs**。图创建/实例化 ~400 µs（一次性） | A |
| NVIDIA《Constant Time Launch…》(2024-09-11)<br><https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/> | Ampere，直链图 repeat launch CPU 开销 | 从 CUDA 11.8 的 `2 µs + 200 ns/node` → 12.6 的 **`~2.5 µs + ~1 ns/node`**；首次 launch CPU 开销：10 节点 4 µs、100 节点 15–25 µs、1 025 节点 175–278 µs；空流 repeat launch 端到端：10 节点 **9 µs**、100 节点 55 µs | A |
| NVIDIA CUDA Graph Best Practices<br><https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html> | 框架 + launch 的综合开销 | "**20–200 µs per operation** in Deep Learning applications"；整图 launch **~10 µs** | A |
| NVIDIA《Employing CUDA Graphs in a Dynamic Environment》(2021-11-03) | A100，504 / 2 520 个 2–8 µs 的 kernel | 图化片段 **+16~26%** | A |
| NVIDIA llama.cpp 图化 (2024-08-07)<br><https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/> | H100，Llama-7B… | 最高 **1.2×**；作者同时指出**剩下的是 CPU 侧"GGML 图准备 + 采样"，预计再 ~10%** | A |

**读法**：图化省的是"每次下发的固定开销"（每 kernel 从 ~9.6 µs 降到图内 ~1 µs 级），
**它一点也没有减少 host 侧 Python/框架的计算**。我们自己的数据（`docs/10-prepare-input-graphification.md` §1）
完全吻合：eager → graph 后 `prepare input` 2.843 → 3.132 ms（**+10%**），单步 42.46 → 5.66 ms（−87%）。

### 4.2 LLM 推理引擎的公开战果

| 来源（时间） | 引擎/模型 | 数字 | 证据 |
|---|---|---|---|
| Fireworks (2023-08-29) | LLaMA-2-7B, bs=1, A100 | 30 → **69 tok/s（2.3×）** | A |
| vLLM PR #16072 "Support full cuda graph in v1"（merged 2025-05-08）<br><https://github.com/vllm-project/vllm/pull/16072> | 单图 vs piecewise；profiling 模式下"决定一个 token" | **13 ms → 6 ms**（profiling 口径）；低并发实测 p50 53.0 → 48.0 ms | B |
| vLLM #27222（2025-10-20，open）<br><https://github.com/vllm-project/vllm/issues/27222> | **Qwen3-Next-80B-A3B（混合 GDN！）**，bs=1024，B200 | 只把 cudagraph capture size 从 ≤512 提到 1 024：**13 900 → 21 000 gen tok/s（+51%）**；作者结论"**主要原因是 GDN，我们在 CPU 上花的时间比 GPU 多**"；后续有人认领"GDN attn CPU-overhead"调查 | B |
| vLLM PR #23569 "[Perf][V1] Fully overlap model execution"（merged 2025-09-06）<br><https://github.com/vllm-project/vllm/pull/23569> | Llama-3.2-1B，H100，concurrency=1，async sched + full cudagraph | 让 model runner 跑在 GPU 前面、把 input prep 与 forward **完全重叠**：TPS 549.85 → 685.11（**+24.6%**）；Qwen3-235B TP16：TPOT 99 → 92 ms，但 TTFT 116 → 200 ms | A |
| SGLang《Zero-Overhead Batch Scheduler》官方博客（2024-12-04）<br><https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/> | Llama-3.1-8B | 调度与 GPU 计算重叠：**+10% 吞吐**（关掉 radix 也成立）；nsys 显示连续 5 个 batch GPU 无 idle | A |
| SGLang 官方文档《Piecewise CUDA Graph》<br><https://docs.sglang.io/docs/advanced_features/piecewise_cuda_graph> | 通用经验值 | decode 图：bs 1–32 **+10~30%**、32–128 **+5~15%**、>256 递减；prefill PCG：短序列 **+15~40%** | A |
| SGLang-omni PR #503（2026-05-21）<br><https://github.com/sgl-project/sglang-omni/pull/503> | AR decode 图化，c=32 | 图化消掉 "Python + launch dispatch **~3–5 ms/step，占单步 30–40%**"：吞吐 **+69%**，延迟 −39% | B |
| **SGLang《Breakable CUDA Graph》(2026)** | gpt-oss-120b，TP4×GB300，prefill-only | 相对 eager：full capture **1.93×**、BCG 1.70×、`torch.compile` 切分 1.45× | A |
| TensorRT-LLM 架构文档<br><https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html> | CUDA Graph padding | "up to **22% end-to-end throughput increase** on certain models/hardware" | A |
| TensorRT-LLM issue #12551（2026-03-25）<br><https://github.com/NVIDIA/TensorRT-LLM/issues/12551> | Qwen2-0.5B，30 in / 30 out，bs=1 | 纯 C++ TRT runtime P50 **36.67 ms** vs PyTorch runtime **58.24 ms**（都开了 CUDA Graph）；NVIDIA 回复："both graph execution time and **bubble** between executions are slower… **the latter one should be due to python runtime overhead (compared to C++)**" | A |
| TensorRT-LLM PR #12148（2026-03-12） | 新增 host 回归测试 | 明确以"host (CPU) overhead 回归"为测试对象（scheduler/sampler/KV manager，BS 1–256，nsys 确认 GPU util <1%） | B |

### 4.3 Ascend 侧（我们真正能用的）

| 来源 | 内容 | 数字 | 证据 |
|---|---|---|---|
| vllm-ascend《ACL Graph》设计文档<br><https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html> | ACLGraph = Ascend 的静态图执行：上游 vLLM 负责 runtime mode/batch descriptor 决策，`ACLGraphWrapper` 负责 capture/replay 缓存；**full graph 需要 attention backend 提供 `update_graph_params()` 钩子**，用 `torch.npu.graph_task_update_begin/end` + `ExternalEvent` 保证 replay 不越过参数更新 | 文档未给百分比，但明确"设计目标就是 reduce host launch overhead for small/medium shapes" | A（机制） |
| vllm-ascend graph mode 用户指南<br><https://docs.vllm.ai/projects/ascend/en/v0.22.1rc/user_guide/feature_guide/graph_mode.html> | `FULL_AND_PIECEWISE`（默认）/`FULL`/`FULL_DECODE_ONLY`/`PIECEWISE`/`NONE` 与 Npugraph_ex 的配对关系；`use_inductor` 在 ACL graph 下禁用；static kernel 编译会让启动多花"几分钟到几十分钟" | — | A |
| vllm-ascend RFC #4715 + PR #4700（npugraph_ex 后端）<br><https://github.com/vllm-project/vllm-ascend/issues/4715> | torchair/npugraph_ex 在 FX 图上做 NPU 亲和融合（如 add+rms_norm → npu_add_rms_norm）+ static kernel | 离线 **TPOT +5~8%**（Qwen/DeepSeek）；在线 DeepSeek-V3.1-w4a8 **TPOT 47.36 → 46.27 ms（+2.5%）** | B |

### 4.4 对"prepare_input 能否图化"的三句话回答

1. **图化能省的是"下发次数"，不是"host 计算量"**——NVIDIA 官方 9.6 → 3.4 µs/kernel 是天花板，
   我们自己的 2.843 → 3.132 ms 是下界（host 计算一点没少）。
2. **decode 路径的算子已图化，而 `prepare_input` 的 host 计算永远在图外**：它决定"下一步跑什么"，
   输入是 scheduler 的 host 决策。
3. **可图化的只是"构造与搬运"**：把每步新建的张量换成预分配缓冲的原地写、把参数张量化
   （vllm-ascend 的 full graph 已经用 tensor 传 `actual_seq_lengths_kv`，见 `attention_v1.py:1029` 附近），
   这与 `docs/10-prepare-input-graphification.md` 的结论一致。

---

## 5. "host-side metadata / scheduler 用 C++ 或编译化"的工程案例

### 5.1 vLLM V1：把 EngineCore 从 API 进程里切出去

| 来源 | 机制 | 公开数字 |
|---|---|---|
| vLLM 官方博客《v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction》(2024-09-05)<br><https://vllm.ai/blog/2024-09-05-perf-update> | ① API server 与 engine 分进程（ZMQ）；② 多步调度；③ 异步输出处理；④ **对象缓存**；⑤ 能异步的拷贝一律异步 | 改造前的 breakdown：**HTTP API 33% / 调度+输入准备 29% / GPU 38%**；多步调度使 Llama-70B on 4×H100 **吞吐 +28%**；异步输出处理使 **TPOT +8.7%**；**"对象缓存"（#7162）单独带来端到端吞吐 +24%** |
| vLLM 官方博客《V1: A Major Upgrade to vLLM's Core Architecture》(2025-01-27)<br><https://vllm.ai/blog/2025-01-27-v1-alpha-release> | 隔离的 `EngineCore` 执行循环（只管 scheduler + model executor）；worker 侧缓存请求状态、每步只传 diff；大量用 numpy 替代 native torch 做张量更新 | **吞吐最高 1.7× vs V0**（无多步调度时）；作者明确归因"**comprehensive CPU overhead reductions**" |
| vLLM PR #19970（async scheduling，2025）<br><https://github.com/vllm-project/vllm/pull/19970> | scheduler 提前一步（NanoFlow 思路），用 output placeholder 解依赖 | **吞吐 +3~15%**，小模型/大批量更明显；BS=1 几乎无提升（"调度开销本来就小"）。**代价**：scheduler 与 worker 必须分进程 ⇒ 输入/输出序列化开销；spec decode / 结构化输出 / PP 当时不支持；TTFT 略升 |
| vLLM V1 worker 侧持久批（persistent batch） | 状态张量常驻 + 每步只写 diff，而不是重建 | 见 MRV2 文档对 V1 的评价："building these tensors from scratch each step is often very slow in Python, **especially for large tensors like block tables**" |

**代价（要写进我们的方案评估）**：跨进程 = 序列化 + 调试复杂度上升；vLLM #19970 自己点名
"multimodal models with large inputs" 会受额外序列化影响。

### 5.2 vLLM MRV2：把 input prep 搬到 GPU（最激进也最有参考价值）

| 来源 | 机制 | 公开数字 |
|---|---|---|
| MRV2 设计文档<br><https://docs.vllm.ai/en/latest/design/model_runner_v2/> | ① 持久状态与每步输入解耦（gather 代替 reorder）；② **Triton kernel 在 device 上构造 `input_ids`/`positions`/`query_start_loc`/`seq_lens`**；③ `StagedWriteTensor`：block table 常驻 GPU，每步把 diff 打包成连续缓冲再一个 kernel 落地；④ UVA 让 GPU 直接读 CPU 常驻大张量；⑤ 每个 **CUDA graph 显式由 `CUDAGraphManager` 管** | 文档原话列出收益："**Lower CPU overhead** by avoiding a large amount of Python and CPU tensor manipulation"、更好的 async/spec-decode 兼容性 |
| MRV2 发布博客 (2026-03-24)<br><https://vllm.ai/blog/2026-03-24-mrv2> | 同上 | 用**故意挑的小模型**放大 host 占比：Qwen3-0.6B on 1×GB200 **吞吐 16K → 25K output tok/s（+56%）**；4×GB200 + GLM-4.7-FP8 + MTP=1：**TPOT −6.3%** |
| PR #25266 | MRV2 实现 | 仍未 feature-complete；移植过程中打破过 PP / LoRA（review 明确指出） |

**对我们的意义**：MRV2 是最接近"把元数据构造从 Python 挪走"的公开工程答案，
但它是 **GPU-first** 的（Triton kernel、UVA、CUDA graph），Ascend 侧只能借鉴**设计原则**：
"能在 device 上算的就别在 host 上算"、"每步只搬 diff"、"持久张量 + 原地写"。
注意 vllm-ascend 已经在做 MRV2 适配（如 PR #16432 "[Performance][MRV2] Vectorize seq_lens_cpu host update"，open）。

### 5.3 SGLang：调度与计算重叠 + 放弃 `torch.compile` 做图切分

| 来源 | 机制 | 数字/代价 |
|---|---|---|
| 《SGLang v0.4: Zero-Overhead Batch Scheduler》(2024-12-04)<br><https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/> | scheduler 提前一个 batch，与 GPU 计算重叠（CPU-S 调度 / CPU-L 下发两段）；用 CUDA event + future token 环状缓冲解依赖 | 吞吐 **1.1×**（对比自身前版）、**1.3×**（对比当时其他 SOTA）；"小模型 + 大 TP 收益最明显" |
| SGLang PR #1738 / PR #4790 | 调整 launch 与取结果顺序；删掉冗余 event | +10% / +1.04% 吞吐 |
| 《Breakable CUDA Graph》(2026)<br><https://www.sglang.io/blog/breakable-cuda-graph> | **用 capture 期插入 eager break 取代 `torch.compile` 的图切分**（`@eager_on_graph`） | 图构建 **3.8–5.2× 快**、代码量 521 vs 1 771 行、replay 比 `torch.compile` 版**快 17%**；**代价**：capture 期稍慢、跨边界的张量要拷进固定地址的 boundary buffer |
| 学术量化《Can Scheduling Overhead Dominate LLM Inference Performance?》(2024-09-10)<br><https://mlsys.wuklab.io/posts/scheduling_overhead/> | 对比 vLLM 0.5.4 与 SGLang | **vLLM 调度开销可占单次迭代总时间 >50%**；SGLang 最多 18%（小模型）。作者结论：主要省在"张量前后处理"与"不必要的 detokenize" |

### 5.4 TensorRT-LLM：纯 C++ runtime + host 性能回归门禁

| 来源 | 机制 | 数字 |
|---|---|---|
| TRT-LLM 架构文档 | CUDA Graph + **overlap scheduler（默认开）**：让一次 decode 的 GPU 计算与下一次的 CPU 调度重叠（代价：多引入一步 decode） | CUDA Graph padding "up to 22% e2e throughput" |
| TRT-LLM issue #12551（2026-03-25） | 同一模型、同一 CUDA Graph 配置，只换 runtime | C++ runtime P50 36.67 ms vs **Python runtime P50 58.24 ms**；NVIDIA 归因"python runtime overhead"；建议"用 CUDA Graph + overlap scheduler 缓解，但负载太小时仍然 host-bound" |
| TRT-LLM PR #12148（2026-03-12） | 新增 **host 性能回归测试套件**（scheduler 21 场景 / sampler 6 / KV manager 6 + LLM API + serving 两层），nsys 确认 GPU util <1% | 说明"host 开销回归"已被当成一类必须持续防的缺陷 |

### 5.5 NVIDIA Dynamo：Rust 承担性能敏感路径

- 官方架构文档明确："**Rust for performance-sensitive runtime components. Python for backend integration
  and extensibility**"（<https://docs.nvidia.com/dynamo/v1.4.0/knowledge-base/overview>）；
  `DistributedRuntime` 完全用 Rust 实现（etcd/NATS/NIXL 客户端、路由、KV 事件），只通过 Python binding 暴露。
- **但它解决的是"分布式/路由/传输"的 host 开销，不是"单步模型输入构造"的 host 开销**。
  公开材料里没有"Dynamo 让单步 host 时间降低 X%"的数字 ⇒ 对我们的直接借鉴价值有限，**证据强度 C**。

### 5.6 横向对比：省了什么 / 代价是什么

| 方案 | 省掉的东西 | 代价 | 对 vllm-ascend 0.26.0 的可用性 |
|---|---|---|---|
| vLLM V1 EngineCore 进程化 | GIL 争用（HTTP/分词与调度互相排队） | 跨进程 IPC 序列化；调试复杂 | ✅ 已在用（0.26.0 默认 V1） |
| vLLM 多步调度 / 异步输出 | 每步调度固定成本摊销到 n 步 | TTFT 变差、ITL 抖动 | ⚠️ V1 已内化（无 `--num-scheduler-steps`） |
| vLLM async scheduling | 调度与 forward 重叠 | 分进程 + 部分特性不兼容 | ⚠️ Ascend 可用性与特性兼容需实测 |
| MRV2 | 大量 Python/CPU 张量操作、CPU↔GPU 同步 | 特性不全、实验性、GPU-centric | ❌ 短期不可直接用；**设计原则可借** |
| SGLang zero-overhead scheduler | 调度泡 | GIL；部分特性（惩罚项/约束解码）最初不支持 | ⚠️ 概念可借，代码不可直接搬 |
| SGLang BCG（无编译器图切分） | `torch.compile` 的编译期与每步 guard | 跨边界拷贝；capture 期约束 | ⚠️ 概念可借；Ascend 已有 ACLGraph，不需要再造 |
| TRT-LLM 纯 C++ runtime | Python 解释器与框架层 | 可扩展性/迭代速度 | ❌ 对 0.26.0 不现实 |
| Dynamo（Rust 运行时） | 分布式路径的 host 成本 | 与单步输入构造无关 | ❌ 无关 |

---

## 6. vLLM 上游动态：我们是不是在重复别人的工作？

### 6.1 直接命中（同一根因、同一个热点），全部核对到 2026-09-24

| 编号 | 类型 | 状态（核对日 2026-09-24） | 标题 | 关键数字/结论 |
|---|---|---|---|---|
| **vLLM #52297** | PR | **open**（created 2026-08-14，updated 2026-09-24；非 draft） | `[Attention] Move GDN common metadata compute out of per-group build` | 作者原话："`GDNAttentionMetadataBuilder.build()` takes a long time (**~900µs**)，because the batch level GDN spec decode metadata … is computed **repeatedly in each group build()**"。改法：把 batch-level 计算提到 `compute_spec_metadata`（每步一次），只在每组保留 block-table 相关的 state index gather。**build 900 µs → 300 µs**；e2e 输出吞吐 **BS=1 +61%、BS=16 +12%**（H200，DFlash） |
| **vllm-ascend #16246** | PR | **open**（created 2026-09-10） | `[Performance][GDN] Share group-invariant metadata across kv-cache groups` | 与我们完全同构：混合模型 3:1 导致 **GDN 被拆到 3 个 kv-cache group**，`build()` 每步被调 3 次；新增 `GDNGroupInvariantCache`，第一组全量算、后两组复用，**只重算与自身 block table 相关的 state index**。自述：**6.3 ms → 4.5 ms/步（−28.6%）**；端到端 **吞吐 +2.29%、TPOT −2.03%**（Qwen3.6-35B-A3B，TP2）；GPQA 精度不变。**当前 0.26.0rc1 源码里没有这个类**（本地 `grep -rn "GroupInvariantCache"` 无命中） |
| **vLLM #57962** | PR | **closed，未合并**（closed 2026-09-21） | `[Perf][GDN] Reuse speculative metadata and fuse block-table gathering` | 同一目标：把元数据缓存进 `MambaHybridAttnMetadata` 供其他 GDN 层复用 + 用一个 Triton kernel 融合 block-table gather。PR 自己注明与 **#52297 目标相同、应协调** ⇒ 上游最终大概率走 #52297 这条更彻底的路线 |
| **vLLM #49730** | issue | **open**（created 2026-07-24，updated 2026-09-21） | `[Bug][Qwen 3.5 4B][H100]: Performance of DFlash is lower than expected` | #57962 的动机来源；同一"重复的 batch-level GDN 计算"现象 |
| **vllm-ascend #16887** | PR | **open**（2026-09-18） | `[Performance][GDN]Skip unused device metadata copies for fused prefill` | 通用 GDN builder 每步为 prefill 生成 **6 个 device chunk 元数据张量**，但在"融合 chunk 算子可用 + 非 PCP"时一个都不用；新增 **host-only builder**（只留 `cu_seqlens_host` / `chunk_indices_chunk64_host` / `keep_meta`）。**只对 prefill 有效** |
| **vllm-ascend #17219 / #16865** | PR / issue | **open**（2026-09-22 / 2026-09-18） | `[Performance][GDN] Avoid AiCPU fallback for boolean token sorting` / `Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu` | GDN 用 `argsort` 稳定划分 spec / 非 spec token；bool mask 先被 cast 成 `int32` 会导致 **AiCore 排序回退到 AiCPU**。改法：**只把 bool mask cast 成 float32**（0/1 可精确表示），保留 `stable=True`，索引仍是 int64。**本地 `gdn_attn_builder.py:49-52` 就是 `tensor.to(torch.int32)` 这个写法**，但只在 spec-decode 路径（`:651`）触发 ⇒ 我们当前基线（不开投机）用不到 |
| **vllm-ascend #16629** | PR | **open**（2026-09-15） | `[Performance][DSA] Eliminate per-step re-JIT of the local metadata build with a fixed-capacity Triton kernel` | 元数据构造"每步重新 JIT"的另一种病；与我们 DSA 无关，但证明**按 batch size 变化的元数据 kernel 会反复重编译** |
| **vllm-ascend #13121** | PR | **open**（2026-07-29） | `[Performance][Ops] Remove obsolete GDN prefill metadata` | 与 #16887 同族，继续删无用字段 |
| **vllm-ascend #14639** | PR | **merged**（2026-09-21） | `[Refactor][GDN] Unify request-level metadata building` | 已经把 request 级元数据构造统一，说明 Ascend 侧**正在**持续削这块 |
| **vllm-ascend #11161** | PR | **merged**（2026-07-10） | `[BugFix][Ops] Move GDN conv1d metadata to device tensors` | conv1d 元数据搬到 device 张量（对应 RFC #11723，仍 open） |
| vllm-ascend #8985 | PR | **closed，未合并**（2026-05-08） | `[Performance]Refactor Ascend GDN attention metadata builder and erase the synchronization operations caused by some tensor copies` | 把 monkey patch 改成 `AscendGDNAttentionMetadataBuilder` 后端、**用 CPU/pinned 缓冲 + device `index_select` 替代"bool mask 直接切 device 张量"以避免同步**。⚠️ 该 PR 本身未合并，但**同样形态的代码已经出现在我们手上的 0.26.0rc1 里**（`vllm_ascend/ops/gdn_attn_builder.py` 的 `_copy_sequence_indices_to_device`）⇒ 说明它经别的 PR 落地，**别用 #8985 的 diff 去 cherry-pick** |

### 6.2 相邻但重要（同步点 / 分配 / 图 / 异步）

| 编号 | 状态 | 内容 | 我们能用的部分 |
|---|---|---|---|
| vLLM **#29134** | issue **open**（2025-11-21） | `seq_lens_cpu` 阻碍 spec-decode 全异步；建议改成可选 + 提供带 warning 的 property；issue 里有一条 2026 年的实测评论：v0.26.0 + MTP=3 + 混合 GDN，**draft 3 趟占 52.5 ms CPU 只为发起 2.3 ms GPU 工作**，阻塞点定位到 `build_for_drafting → FlashInferMetadataBuilder.build → seq_lens_cpu → Tensor.to("cpu")` | 与我们的 `.replace()` 缓存问题同类："元数据 property 的隐式物化 + 每步重算"。**注意它指出 `num_computed_tokens_cpu` 同样是 `seq_lens_cpu - query_lens_cpu` 的派生物**，与我们 §7-A1 同源 |
| vLLM **#29624** / #29449 | 均 **merged**（2025-11-27 / 2025-11-25） | 让 `seq_lens_cpu` 在 `CommonAttentionMetadata` 里可选；去掉 FlashAttention 在 DCP>1 时对它的依赖 | 上游的"少一次 D2H"路线；与我们的 device 张量 `compute_num_computed_tokens` 是同一思路的两个方向 |
| vLLM **#41434** | **merged**（2026-05-08） | `[Perf][3/n] Eliminate GPU<->CPU syncs in attention impls`：`tensor[0] = x` → `tensor[:1] = x` / `.fill_()`；把 `max()/max_seq_len` 预计算到 CPU；用 pinned `async_tensor_h2d` 代替 `torch.tensor(..., device=cuda)`；把 `torch.nonzero`/`torch.bincount` 换成无同步等价物 | **一份可以直接照抄的"同步点黑名单"**，逐条对照我们的 GDN builder 自查 |
| vLLM **#45074** | **merged**（2026-06-10） | MLA chunked-context 元数据里 `chunk_starts` / `token_to_seq_tensor_cpu` 是 pageable 内存 + `non_blocking=True` ⇒ **假异步、实际同步**；`cudaMemcpyAsync` 会等整条流排空。修法：`.pin_memory()` | 元数据 build 一度 **67–101 ms/4096-token step**；修后端到端输入吞吐 **+3.3%**、P99 TTFT −5.4%。**教训**：`non_blocking=True` 必须配 pin memory，否则元数据构造会被"隐藏的流排空"拖死 |
| vLLM **#39936** | **open**（2026-04-15） | `Reduce cpu launch overheads for MultiGroupBlockTable and attn_meta_data handling on Blackwell`：用 Triton kernel **批量处理**同一 `MultiGroupBlockTable` 下的所有 block table，替代逐个 `compute_slot_mapping`/`commit_block_table`/`split_decodes_and_prefills` | **"元数据批量化 + kernel 化"的正面案例**：与 #52297 的思路一致（把 per-group 变成 per-step 一次）。review 里也有人建议先试 `VLLM_USE_V2_MODEL_RUNNER=1` |
| vLLM **#46112** | **closed，未合并**（2026-06-18） | `Reduce update_from_output CPU overhead for decode batches`：微基准 BS=128 **0.174 → 0.135 ms（−22%）**；decode-bound 端到端 **吞吐 +5.1%、TPOT −5.0%** | 说明**"只削 Python 循环"也能拿到 5%**；但该 PR 未合并（方案可能被视为过度特化） |
| vLLM **#36868** | **closed，未合并**（2026-03-12） | `Skip np.repeat in _prepare_inputs when all requests are decodes`：`np.repeat` 实测 **1.44 µs/次**，纯 decode 步可直接返回切片 | 单点收益极小（µs 级），价值在示范"纯 decode 快路径"这个手法 |
| vLLM **#44288** | **open**（2026-06-02） | 复用 `input_batch.num_tokens_no_spec`，避免每步重建 `list` + `np.array` | 与我们"减少每步新对象"完全同向；作者自述主要收益是**减少 GC 压力与 P99 抖动**而非吞吐 |
| vLLM **#58114** | **open**（2026-09-22） | `[Perf][Qwen3.8] Reduce PLE metadata construction overhead`：混合投机 batch 里的同步、无用临时张量、`repeat_interleave` 输出尺寸、`index_fill_` 代替标量赋值 | **最新（本周）的同类工作**；其手法清单可直接对照我们的 `_pad_non_spec_decode_graph_inputs` |
| vLLM **#27222** | issue **open**（2025-10-20） | Qwen3-Next（混合 GDN）CPU 开销：cudagraph capture 只放到 512 时 **13 900 gen tok/s**，提到 1 024 后 **21 000（+51%）** | **"混合 GDN + host-bound"在上游的经典案例**；我们的问题在 GPU 上已被同一现象反复报告 |
| vLLM **#17866** | PR **closed，未合并**（2025-05-08） | `[V1] Fast decode prepare path for prepare_inputs logic`：为重复 decode 做快路径，自述 Llama-3.2-1B TPOT **~10%** | 被 WoosukKwon 明确拒绝："I really don't want this optimization. We can optimize `prepare_inputs` in different ways." ⇒ **上游态度：不接受"绕开通用路径的快路径"，只接受通用路径本身的优化**。我们提方案时要按这个口径写 |
| vLLM **#5561** | issue **closed**（2024-06-14） | 投机解码里 sampler 的 CPU 拷贝+序列化 **~441 µs**（draft 前向+采样才 859 µs）；预期 `prepare_inputs` 还能从 ~300 µs → ~150 µs | 早期量化"host 侧串行化"的经典数字 |
| vLLM **#914** | issue **closed**（2023-08-31） | CUDA Graph 支持的最初讨论（引 Fireworks 2.3×） | 历史脉络 |

### 6.3 判定：我们重复了哪部分，增量在哪

| 我们的工作 | 是否重复 | 结论 |
|---|---|---|
| "GDN builder 每步被调 3 次、占 29.3%" | ⚠️ **部分重复**：上游 #52297（GPU，open）与 vllm-ascend #16246（Ascend，open）已经独立发现并给出方案 | 价值在于**我们给出了 Ascend 0.26.0rc1 + 0.8B/TP1/B=1 的实测口径**（他们给的是 H200 BS=1/16 与 Ascend 35B-A3B TP2），以及"3 组 → 1 组"在**我们这套小模型上**的实际收益 |
| "`.replace()` 造的临时对象让缓存落空" | ✅ **未重复**：vLLM main（今日读取）的 `CommonAttentionMetadata.replace()` 仍是裸 `dataclasses.replace`，缓存字段仍是普通字段；没有任何 PR 处理它 | **这是我们可以直接提 upstream 的点**，且改动极小（见 §7-A2） |
| "想用 torch.compile 加速元数据构造" | ✅ 已排除 | 有 4 条独立公开证据说明是负收益（§3），不需要再做实验 |
| "想用 graph 包住 prepare_input" | ⚠️ 已被 `docs/10` 论证 | 本篇补充公开天花板数字（§4.1） |

### 6.4 可直接 cherry-pick 的判定

| 候选 | 能否直接 cherry-pick | 说明 |
|---|---|---|
| vllm-ascend #16246 | **可以，但需要移植** | 它基于更新的 vLLM main（`84030bbe`）与更新的 vllm-ascend 代码；核心是"新增 cache 类 + build 增加可选参数 + model runner 在 group 循环外建 cache"。我们树里 `AscendGDNAttentionMetadataBuilder.build()`（`gdn_attn_builder.py:531`）签名与它一致，**移植面可控**；必须补"field-by-field 等价"单测 |
| vLLM #52297 | **思路可抄，代码不可直接合** | 它改的是上游 `gpu_model_runner.py` 的 `compute_spec_metadata` + `GDNAttentionMetadataBuilder` 的分层；我们 Ascend 的 builder 是**另一份**实现（`AscendGDNAttentionMetadataBuilder`），要按同样的分层改 Ascend 版本 |
| vLLM #57962 | ❌ | 已 close 未合并；上游方向是 #52297 |
| vllm-ascend #16887 | 部分 | 只对 prefill 有效，decode 基线不用 |
| vllm-ascend #17219 | 条件可用 | 只在我们开启投机解码（走 `:651` 的 bool argsort）时才需要 |
| vLLM #41434 的同步点黑名单 | ✅ 当 checklist 用 | 逐条自查我们 builder 里的标量赋值 / `.max().item()` / `torch.tensor(..., device=...)` |

---

## 7. 可借鉴清单（按"预期收益 × 落地难度"排序）

排序原则：**先抄上游已经验证过的、收益最大的**；再动只有我们发现的点；最后才是工程改造。
每条都给出"改哪段代码 / 预期省多少 / 风险 / 证据强度"。µs 数一律给换算过程。

---

### A1 · 把 3 组 GDN build 的 batch-level 计算压成 1 组 ⭐ 最高优先级

| 项 | 内容 |
|---|---|
| **改哪里** | ① 首选照 **vllm-ascend #16246**：在 `vllm_ascend/ops/gdn_attn_builder.py`（本地 940 行版）的 `AscendGDNAttentionMetadataBuilder.build()`（`:531`）增加可选 `group_invariant_cache` 参数，新增 `GDNGroupInvariantCache` 类；**只在 model runner 的 kv-cache-group 循环外建一次 cache**（上游对应位置：`refs/vllm/vllm/v1/worker/gpu_model_runner.py:2446` 的 `_build_attn_group_metadata` 闭包 + `:2573` 的调用点）。② 若要更彻底，照 **vLLM #52297** 的分层：把 batch-level（spec mask / decode-prefill 划分 / token index / `query_start_loc` cumsum）提到 `compute_spec_metadata` 每步一次，每组只保留 block-table 相关的 state index gather。 |
| **预期收益** | 按 #16246 的实测比例（6.3 → 4.5 ms，**−28.6%**）线性折算到我们 **908 µs/步**：**≈ 260 µs/步**。按 #52297 的"900 → 300 µs"（**−67%**）折算：**上限 ≈ 600 µs/步**（= 我们理论上能省掉的 2/3）。区间 **260–600 µs/步**，占 `prepare_input` 的 9–22%。 |
| **风险** | ① **语义等价必须逐字段验证**（#16246 用 field-by-field UT + GPQA 精度回归）；② 缓存张量必须**只读共享**，每组的 `spec_state_indices_tensor` / `non_spec_state_indices_tensor` / `prefill_state_indices` 仍要按各自 block table 重算；③ **图模式下不能把每组的持久缓冲换成共享缓冲**（会改变 capture 时的地址）；④ PR 里明确"ubatch 场景退化为全量计算"（shape 级 fingerprint 无法区分同形 ubatch）。 |
| **证据强度** | **A**（两个独立 PR + 我们本地源码确认同一调用结构 `.replace()` / 3 组 / `build()` 签名一致） |
| **前置** | 无。建议第一步就做，并且**先做等价性 A/B（关图 / 开图两个口径）**。 |

---

### A2 · 修 `.replace()` 造成的"缓存永远落空"（我们的独有发现）⭐

| 项 | 内容 |
|---|---|
| **精确机制（已核对源码，2026-09-24）** | `CommonAttentionMetadata.replace()` 是裸的 `dataclasses.replace`（`refs/vllm/vllm/v1/attention/backend.py:497`）。缓存字段 `_num_computed_tokens_cache`（`:487`）是**普通 dataclass 字段**，所以 `replace()` 复制的只是**当前值**——而调用者是 vllm-ascend 的 `_treat_single_token_prefills_with_state_as_decodes()`（`gdn_attn_builder.py:59-83`），它**在任何人计算之前**就 `replace()` 出一个新对象 `m`，随后 `m.compute_num_computed_tokens()`（`:543`）把结果**写进这个临时对象**。父对象 `cm` 的缓存**永远是 None** ⇒ 下一个 group 重新 `replace()` 又拿到 None ⇒ **3 组各算一次**。 |
| **改哪里** | 三种改法（建议 (a)+(b) 组合）：<br>(a) **让 `replace()` 只在"派生输入未变时"保留缓存**：在 `CommonAttentionMetadata.replace()` 里判断 `kwargs` 是否触及 `query_start_loc`/`seq_lens`，未触及则透传 `_num_computed_tokens_cache` / `_token_to_req_indices_cache`（并给这两个字段标 `field(compare=False)`）——**不能无条件透传**，否则 `replace(seq_lens=...)` 这类调用会拿到过期缓存；<br>(b) **把 `_treat_…` 的结果缓存到父对象**（例如内部把 `is_prefilling` 覆盖写回 `cm`，或加一个 `_treat_cache` 字段），或干脆把这段判断从"每组一次"提到"每步一次"（那就是 A1 的一部分）；<br>(c) 用 **numpy** 完成同一判断（`query_start_loc_cpu` / `seq_lens_cpu_upper_bound` 都是 CPU host 张量，`.numpy()` 是零拷贝视图），把 10 次 torch 派发换成 numpy。 |
| **预期收益（含换算）** | ① CPU 侧：`_treat_…` 每组的 `torch.diff`(1) + `==`(1) + `>`(1) + `&`(2) + `torch.any`(1) + `.item()`(1) + `clone()`(1) + mask 赋值(1) ≈ **9 次 torch 派发/组**，按纯 ATen 1.6–2.3 µs/次 ⇒ 15–21 µs/组 ⇒ **45–63 µs/步**（3 组）。<br>② device 侧：`compute_num_computed_tokens()` 里 `query_start_loc[1:] - query_start_loc[:-1]` + `seq_lens - query_lens` ≈ **4 次 device 算子/组**，按 3–12 µs/次 ⇒ 12–48 µs/组 ⇒ **36–144 µs/步**。<br>③ 合计 **80–200 µs/步**，与项目既有估计 **~130 µs/步**（`docs/00-INDEX.md` §3）自洽。**保守取 130 µs/步**。 |
| **风险** | ① 改 `replace()` 属**上游核心数据结构**，要保证"传出去的缓存"在后续被 mutate 的语义下仍成立（本例中 `query_start_loc` / `seq_lens` 在 group 循环内**不会变**，只有 `block_table_tensor` / `slot_mapping` 变 ⇒ 缓存安全）；② 不要引入新的 `.item()`/`.tolist()` 同步；③ 该缓存字段带 `WARNING: Deprecated` 注释，上游可能直接删字段 —— 提 PR 时要按"保留派生缓存或改成 device 端一次算好"的口径写。 |
| **证据强度** | **A**（机制在源码里逐行核对）；收益换算为 **B**（基于我们的单次成本基准） |
| **前置** | 无（可与 A1 合并成一个 PR） |

---

### A3 · 削 build 内的一次性对象与重复 tensor 构造

| 项 | 内容 |
|---|---|
| **改哪里** | `gdn_attn_builder.py`：<br>① `_copy_sequence_indices_to_device`（`:305-329`）每组 2× `torch.nonzero` + 2× CPU `copy_` + 2× H2D `copy_` —— 可改为"预分配 + 单次写入"或合并成一个 kernel/一次 H2D（上游 #41434 的 `scatter_add_` 替代 `bincount`、#52297 的"融合 block-table gather 成一个 Triton kernel"都是同款）；<br>② `_collapse`/`_compact_empty_segments`（`:137-165`）里的 `torch.tensor(cu_seqlens_host)` + 比较 + `keep.all()` + `cat` + `tolist()` 链路，decode 路径可短路；<br>③ `_pad_non_spec_decode_graph_inputs`（我们文档里 self 最高的子项 5.94%）已经是 in-place 写，继续把同层其它"每步新建"的字段改造成 in-place 写。 |
| **预期收益** | 每步 3 组 × (6 次 index/copy + 4~6 次 CPU 判断) ≈ 30–45 次算子 ⇒ 视其中 device 算子占比，**20–40 µs/步**。 |
| **风险** | 低（都是等价替换），但 **in-place 改造会改变张量地址** ⇒ 与 graph capture 的持久缓冲约定强耦合，必须对照 `docs/10-prepare-input-graphification.md` §3 的"固定地址"约束一起做。 |
| **证据强度** | **B**（收益为自估；机制有上游同类 PR 支撑） |

---

### B1 · 用 numpy/原生结构替换"纯 CPU 张量上的 torch 运算"

| 项 | 内容 |
|---|---|
| **改哪里** | 所有"输入是 CPU 张量、输出也是 host 值"的地方：`_treat_single_token_prefills_with_state_as_decodes`、`seq_lens_cpu_upper_bound` 相关判断、`num_tokens`/`discard_request_mask` 之类的簿记（上游 #44288、#36868 就在做这件事）。 |
| **预期收益** | 单次成本从 1.6–2.3 µs（torch CPU）降到 0.4–0.9 µs（numpy，实测 `data/model/microbench-numpy.csv`）；若每步有 30–50 次这类调用，**20–50 µs/步**。 |
| **风险** | ① 必须确认输入是 **CPU 张量**（device 张量 `.numpy()` 会触发 D2H 同步，**严禁**）；② pinned 张量的 `.numpy()` 合法但要小心生命周期；③ numpy 默认 dtype 与 torch 的隐式提升规则不同（int64/int32 溢出、bool 语义），要做 dtype 显式化。 |
| **证据强度** | **B**（成本有实测；总收益需 A/B） |

---

### B2 · CPython 层的"少一次属性查找、少一个对象"

| 项 | 内容 |
|---|---|
| **改哪里** | 热点循环里的 `obj.attr` / `self.x` / dataclass 构造：把 `common_attn_metadata.seq_lens`、`m.query_start_loc` 这类反复访问绑到局部变量；把每步新建的中间 dataclass 改成 tuple/预分配；避免在循环里 `getattr`/`hasattr`。 |
| **公开经验** | vLLM v0.6.0 里**单独一个"对象缓存"（#7162）就带来端到端吞吐 +24%**（<https://vllm.ai/blog/2024-09-05-perf-update>）；我们的火焰图 `_PyType_Lookup 1.67%` + `_PyObject_GenericGetAttrWithDict 1.36%` 正是这一类的指纹（`docs/00-INDEX.md` §2）。 |
| **预期收益** | **10–40 µs/步**（自估，必须 A/B）；好处是**零语义风险**。 |
| **风险** | 低。唯一风险是"为了省属性查找把代码改得不可读"，要配合 `docs/04-profiling-methodology.md` 的噪声门验证。 |
| **证据强度** | **C**（上游那 24% 是另一套代码路径，不能直接套用） |

---

### C1 · 打开 async scheduling / 评估 MRV2（架构级，中长期）

| 项 | 内容 |
|---|---|
| **做法** | 让"下一步输入准备"与"当前步 forward"重叠，从"减少 host 成本"变成"**隐藏** host 成本"。vLLM 侧：`--async-scheduling`（#19970，吞吐 +3~15%）、完全重叠（#23569，+24.6% TPS @c=1）；Ascend 侧要自己确认兼容矩阵（spec decode / 结构化输出 / PP 的历史限制）。 |
| **预期收益** | 若 host 开销能被完全藏住：**理论上界 = 我们 2.76 ms 的 `prepare_input` 全部不落在关键路径上**；现实中受同步点、feature 兼容性、前后处理限制。上游 GPU 实测 +3~25%。 |
| **风险** | 高：① 需要额外进程/线程与同步设计；② 与投机解码、结构化输出、PP 的兼容性历史上反复出问题（#19970、#23569 的长评论）；③ Ascend 侧能否用**必须实测**（我们无法从公开材料判断）；④ 一旦有隐藏 D2H，收益归零。 |
| **证据强度** | **A**（上游数字）／**C**（Ascend 适用性） |

---

### C2 · 只在需要时启用 prefill 侧专用优化（#16887 / #17219）

| 项 | 内容 |
|---|---|
| **做法** | ① 当 workload 变成"长 ISL / chunked prefill"时，接 **#16887** 的 host-only GDN builder（跳过 6 个 device chunk 元数据张量）；② 当开启投机解码时，接 **#17219** 的 bool→float32（避免 AiCore→AiCPU argsort 回退）+ 修 `_stable_argsort_for_npu`（本地 `gdn_attn_builder.py:49-52`）。 |
| **预期收益** | 我们**当前基线（B=1 decode、不开投机、非长 prefill）≈ 0**；在对应场景下按 PR 自述是"每次调用省数百 µs 级"。 |
| **风险** | 低（都是"少做无用功"），但需要对应 workload 才能验证。 |
| **证据强度** | **B** |

---

### D · 不建议做（但列出来省得后人再试）

见 §8。

---

## 8. 反面清单（不要再试的路）

1. **不要对元数据构造用 `torch.compile`**：4 条独立证据（§3.4）+ Ascend 禁用 Inductor。若真想试，唯一合理形态是
   **只编译"纯张量、无控制流"的那一小段**，且必须把 `TorchDynamo Cache Lookup` 时间打出来与收益对比（官方给出的诊断方法）。
2. **不要用 `TorchDispatchMode` / Tensor 子类做"轻量"拦截**：官方 FLOP counter 实测 **+30–40 µs/op**；
   我们 908 µs 的 build 里若每个算子都过一层 Python，成本会翻几倍。
3. **不要指望 `no_dispatch` / `DisableTorchFunctionSubclass` / `_get_tracing_state` 能省时间**：
   它们的前提条件（活跃 mode / 子类 / tracing）在我们的 eager 基线下**都不成立**（§2.3）。
4. **不要把 FIA 的 `actual_seq_lengths` 从 Python list 改成 `torch.Tensor` 来"省一次 `.tolist()`"**：
   `vllm_ascend/attention/attention_v1.py:337-339` 附近的注释明确写了这是**为了规避 D2H 同步的刻意设计**；
   真要改，必须走 full graph 的 tensor 参数路径（且那条路已经存在）。
5. **不要在"每步新建张量"的同时谈图化**：图捕获要固定地址（`docs/10-prepare-input-graphification.md` §3），
   两件事必须一起做，否则白忙。
6. **不要照抄未合并 PR 的 diff 而不看它的 review**：例如 vLLM #39936 的 review 就发现了
   `utils.py` 里引用未定义变量、张量未先转 list 就做逻辑运算的 bug；vllm-ascend #8985 的形态虽已进主线，
   但该 PR 本身是 closed 未合并。
7. **不要只优化 dispatcher 层**：那一层的量级是 **100 ns**（PyTorch 官方原话），
   我们的目标是 **µs × 调用条数**。

---

## 9. 引用清单（全部访问于 2026-09-24）

**PyTorch dispatcher / torch.compile**

1. 《Dispatcher Performance and Inlining》，dev-discuss，2021-01-28 — <https://dev-discuss.pytorch.org/t/dispatcher-performance-and-inlining-a-report-on-two-days-spent-on-dispatcher-performance/109>
2. PyTorch `benchmarks/overrides_benchmark/README.md`（"on the order of 2 μs"）— <https://github.com/pytorch/pytorch/blob/main/benchmarks/overrides_benchmark/README.md>
3. pytorch#41383《Large overhead (7 microseconds) for PyTorch operation》，2020-07-14 — <https://github.com/pytorch/pytorch/issues/41383>
4. pytorch#50848《inlining tensor.device()》（指令数证据）— <https://github.com/pytorch/pytorch/pull/50848>
5. 《Skipping Dispatcher with LazyTensor》，dev-discuss，2022-06-12 — <https://dev-discuss.pytorch.org/t/skipping-dispatcher-with-lazytensor/634>
6. 《The "Ideal" PyTorch FLOP Counter (with `__torch_dispatch__`)》，dev-discuss，2022-02-18（+30–40 µs/op）— <https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505>
7. pytorch#187949 / commit `f7155a0`《PyObject Dispatch》（2.6 µs → 300 ns）— <https://github.com/pytorch/pytorch/commit/f7155a0e49678bac796f36ed11ac741e1c3ed254>
8. `torch/utils/_mode_utils.py`（`no_dispatch = torch._C._DisableTorchDispatch`）— <https://github.com/pytorch/pytorch/blob/main/torch/utils/_mode_utils.py>
9. `torch/_C/__init__.pyi.in`（`_DisableTorchDispatch` / `DisableTorchFunctionSubclass` / `DisableTorchFunction`）
10. `torch/utils/_python_dispatch.py`（mode 栈内部实现）
11. PyTorch 官方文档《Reducing Guard Overhead》— <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.reducing_guard_overhead.html>
12. pytorch#185886《Dynamo cache lookup guard eval regression 2.7 → 2.12》（1.36 ms → 5.65 ms）— <https://github.com/pytorch/pytorch/issues/185886>
13. pytorch#161783《torch.compile has runtime overhead on small graphs》，2025-08-29（+ PR #163747 / PR #184133）— <https://github.com/pytorch/pytorch/issues/161783>
14. pytorch#72746《CPU execution/dispatch time dominates and slows down small TorchScript GPU models》

**CUDA Graph / NPU Graph**

15. NVIDIA《Getting Started with CUDA Graphs》，2019-09-05 — <https://developer.nvidia.com/blog/cuda-graphs/>
16. NVIDIA《Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements》，2024-09-11 — <https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/>
17. NVIDIA《CUDA Graph Best Practice for PyTorch》（"20–200 µs per operation"，图 launch ~10 µs）— <https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html>
18. NVIDIA《Employing CUDA Graphs in a Dynamic Environment》，2021-11-03 — <https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/>
19. NVIDIA《Optimizing llama.cpp AI Inference with CUDA Graphs》，2024-08-07 — <https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/>
20. Fireworks《Speed, Python: Pick Two》，2023-08-29 — <https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning>
21. SGLang 官方文档《Piecewise CUDA Graph》— <https://docs.sglang.io/docs/advanced_features/piecewise_cuda_graph>
22. SGLang 博客《Breakable CUDA Graph in SGLang: 5× faster graph builds, 1.93× faster prefill》(2026) — <https://www.sglang.io/blog/breakable-cuda-graph>
23. SGLang 博客《SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer…》，2024-12-04 — <https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/>
24. SGLang PR #1738 / #4790（overlap scheduler 的量化收益）
25. vLLM 文档《CUDA Graphs》（`FULL_AND_PIECEWISE` 语义）— <https://vllm.website.cncfstack.com/design/cuda_graphs.html>
26. vllm-ascend《ACL Graph》设计文档 — <https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html>
27. vllm-ascend《Graph Mode Guide》— <https://docs.vllm.ai/projects/ascend/en/v0.22.1rc/user_guide/feature_guide/graph_mode.html>
28. vllm-ascend RFC #4715《npugraph_ex backend》+ PR #4700 — <https://github.com/vllm-project/vllm-ascend/issues/4715>
29. TensorRT-LLM《Architecture Overview》— <https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html>
30. TensorRT-LLM issue #12551《Performance degradation using Pytorch backend compared to TensorRT backend on Qwen2 0.5B model》，2026-03-25 — <https://github.com/NVIDIA/TensorRT-LLM/issues/12551>
31. TensorRT-LLM PR #12148《Add host performance regression test suite for PyExecutor》，2026-03-12 — <https://github.com/NVIDIA/TensorRT-LLM/pull/12148>
32. NVIDIA Dynamo 文档《Overall Architecture》— <https://docs.nvidia.com/dynamo/v1.4.0/knowledge-base/overview>

**vLLM / vllm-ascend 架构与上游动态**

33. vLLM 博客《v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction》，2024-09-05 — <https://vllm.ai/blog/2024-09-05-perf-update>
34. vLLM 博客《V1: A Major Upgrade to vLLM's Core Architecture》，2025-01-27 — <https://vllm.ai/blog/2025-01-27-v1-alpha-release>
35. vLLM 博客《Model Runner V2: A Modular and Faster Core for vLLM》，2026-03-24 — <https://vllm.ai/blog/2026-03-24-mrv2>
36. vLLM 设计文档《Model Runner V2》— <https://docs.vllm.ai/en/latest/design/model_runner_v2/>
37. vLLM PR #19970《Implement Async Scheduling》— <https://github.com/vllm-project/vllm/pull/19970>
38. vLLM PR #23569《[Perf][V1] Fully overlap model execution》— <https://github.com/vllm-project/vllm/pull/23569>
39. vLLM PR #25266《GPU Model Runner V2》— <https://github.com/vllm-project/vllm/pull/25266>
40. vLLM PR #16072《[Core] Support full cuda graph in v1》— <https://github.com/vllm-project/vllm/pull/16072>
41. vLLM PR #17866《[V1] Fast decode prepare path for prepare_inputs logic》（被拒）— <https://github.com/vllm-project/vllm/pull/17866>
42. vLLM PR #52297《[Attention] Move GDN common metadata compute out of per-group build》— <https://github.com/vllm-project/vllm/pull/52297>
43. vLLM PR #57962《[Perf][GDN] Reuse speculative metadata and fuse block-table gathering》— <https://github.com/vllm-project/vllm/pull/57962>
44. vLLM issue #49730《[Bug][Qwen 3.5 4B][H100]: Performance of DFlash is lower than expected》— <https://github.com/vllm-project/vllm/issues/49730>
45. vLLM issue #27222《[Performance][Qwen3-next] Decrease huge CPU overhead》— <https://github.com/vllm-project/vllm/issues/27222>
46. vLLM issue #29134《[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional》— <https://github.com/vllm-project/vllm/issues/29134>
47. vLLM PR #29624 / #29449（`seq_lens_cpu` 可选 / 去掉 DCP 依赖，均已合并）
48. vLLM PR #41434《[Perf][3/n] Eliminate GPU<->CPU syncs in attention impls》— <https://github.com/vllm-project/vllm/pull/41434>
49. vLLM PR #45074《[Perf][Attention] Pin MLA chunked-context metadata tensors…》— <https://github.com/vllm-project/vllm/pull/45074>
50. vLLM PR #39936《[Performance] Reduce cpu launch overheads for MultiGroupBlockTable and attn_meta_data handling on Blackwell》— <https://github.com/vllm-project/vllm/pull/39936>
51. vLLM PR #46112《[Performance] Reduce update_from_output CPU overhead for decode batches》— <https://github.com/vllm-project/vllm/pull/46112>
52. vLLM PR #36868《[Core] Skip np.repeat in _prepare_inputs when all requests are decodes》— <https://github.com/vllm-project/vllm/pull/36868>
53. vLLM PR #44288《perf: Avoid per-step list+np.array allocation in _prepare_inputs…》— <https://github.com/vllm-project/vllm/pull/44288>
54. vLLM PR #58114《[Perf][Qwen3.8] Reduce PLE metadata construction overhead》— <https://github.com/vllm-project/vllm/pull/58114>
55. vLLM issue #5561《[Performance][Speculative decoding] Speed up autoregressive proposal methods…》— <https://github.com/vllm-project/vllm/issues/5561>
56. vLLM issue #914《CUDA Graph support》— <https://github.com/vllm-project/vllm/issues/914>
57. vllm-ascend PR #16246《[Performance][GDN] Share group-invariant metadata across kv-cache groups》— <https://github.com/vllm-project/vllm-ascend/pull/16246>
58. vllm-ascend PR #16887《[Performance][GDN] Skip unused device metadata copies for fused prefill》— <https://github.com/vllm-project/vllm-ascend/pull/16887>
59. vllm-ascend PR #17219 / issue #16865（bool argsort → AiCPU 回退）— <https://github.com/vllm-project/vllm-ascend/pull/17219>
60. vllm-ascend PR #16629《[Performance][DSA] Eliminate per-step re-JIT of the local metadata build…》— <https://github.com/vllm-project/vllm-ascend/pull/16629>
61. vllm-ascend PR #13121《[Performance][Ops] Remove obsolete GDN prefill metadata》— <https://github.com/vllm-project/vllm-ascend/pull/13121>
62. vllm-ascend PR #14639《[Refactor][GDN] Unify request-level metadata building》（已合并）— <https://github.com/vllm-project/vllm-ascend/pull/14639>
63. vllm-ascend PR #11161《[BugFix][Ops] Move GDN conv1d metadata to device tensors》（已合并）+ RFC #11723
64. vllm-ascend PR #8985（同形态代码已进主线，PR 本身未合并）— <https://github.com/vllm-project/vllm-ascend/pull/8985>
65. vllm-ascend PR #16432《[Performance][MRV2] Vectorize seq_lens_cpu host update》— <https://github.com/vllm-project/vllm-ascend/pull/16432>
66. 《Can Scheduling Overhead Dominate LLM Inference Performance?》，WukLab，2024-09-10 — <https://mlsys.wuklab.io/posts/scheduling_overhead/>

**本项目内部引用**

67. `docs/00-INDEX.md` §0/§2/§3（55.3% 占比、火焰图指纹、908 µs 热点、~130 µs 优化估计）
68. `docs/03-share-in-inference.md:127`（`k_h2d_small` = 3–12 µs）
69. `docs/05-hotspots.md` §5bis.5（单次小 NPU 算子 ≈ 3–12 µs 的推导）
70. `docs/10-prepare-input-graphification.md` §1/§3（图化前后 2.843 → 3.132 ms；固定地址约束）
71. `data/model/microbench-torch.csv` / `microbench-numpy.csv` / `microbench-meta.json`（920B aarch64 单次成本基准）
72. 源码：`refs/vllm/vllm/v1/attention/backend.py:486-499,530-535`；`refs/vllm/vllm/v1/worker/gpu_model_runner.py:2446-2576`；
    `vllm_ascend/ops/gdn_attn_builder.py:49-83,137-165,305-329,531-570`；`vllm_ascend/attention/attention_v1.py:320-360,1029`
