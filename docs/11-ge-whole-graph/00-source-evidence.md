# GE / 整图下发：vllm-ascend 官方源码与文档的直接证据

> 本文**只记录可直接核对的一手证据**（官方仓库内的文档 + 源码行号），不含推测。
> 由 root 直接查阅 `vllm-ascend` 0.26.0rc1 仓库（commit `f2f74a16c`）写成，
> 是本目录另外三篇（`01`/`02`/`03`）的事实基础。
> 所有路径相对 `/home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend/`。

---

## 1. 结论速览（三条，均有行号）

| # | 事实 | 证据 |
|---:|---|---|
| 1 | **vllm-ascend 确实集成了 torchair 项目，但刻意配置为"不做 fx graph → Ascend IR 变换"** | `vllm_ascend/compilation/compiler_interface.py:132–135` |
| 2 | **官方图模式文档只列两条路径：ACLGraph(+Npugraph_ex) 与 XliteGraph；没有 GE 整图路径** | `docs/source/user_guide/feature_guide/graph_mode.md:22–47` |
| 3 | **官方 ACLGraph 设计文档明确写出"为什么 capture 单独不够"：attention 需要每步的 host 侧参数更新** | `docs/source/developer_guide/Design_Documents/ACL_Graph.md`，见 §4 |

---

## 2. 证据一：vllm-ascend 用了 torchair，但刻意不做 IR 变换

`vllm_ascend/compilation/compiler_interface.py` 的 `npugraph_ex_compile()`：

```python
# 先用 npugraph_ex，失败才回退到 torchair（向后兼容）
try:
    import npugraph_ex as nge
    config = nge.CompilerConfig()
    _configure_backend(config, ..., process_kwargs_options=_process_kwargs_options)
    backend = nge.get_npu_backend(compiler_config=config)
    ...
except ImportError:
    import torchair
    config = torchair.CompilerConfig()
    _configure_backend(config, ascend_compilation_config, vllm_config)
    backend = torchair.get_npu_backend(compiler_config=config)
```

`_configure_backend()` 的两条分支（`compiler_interface.py:105–150`）都指向 **eager 执行 FX 图**：

```python
if process_kwargs_options is not None:
    # npugraph_ex 路径
    options = {
        "force_eager": True,        # ← FX 图在 capture 前按 eager 执行
        "inplace_pass": False,      # 避免 gelu 回退 CPU
        "clone_input": False,
        "clone_output": False,
    }
    ...
else:
    # torchair 路径
    # mode="reduce-overhead": use aclgraph mode, avoid fx graph to Ascend IR transformation.
    config.mode = "reduce-overhead"      # ← 显式选 ACLGraph 模式
    config.debug.run_eagerly = True      # ← 显式要求 eager 执行
```

**这条注释就是用户问题的直接答案**：`avoid fx graph to Ascend IR transformation` —— 
vllm-ascend 主动避开 GE 的 IR 变换，选择 ACLGraph 模式。

> 术语澄清（避免混淆）：
> - **torchair** = 华为维护的 torch↔Ascend 图编译桥（仓库 `gitcode.com/Ascend/torchair`），
>   `npugraph_ex` 就是它的一个模块。
> - **GE（Graph Engine）** = Ascend 的图编译器/执行器，torchair 在**非** `reduce-overhead`
>   模式下会把 FX 图变成 Ascend IR 再交给 GE 编译。
> - **ACLGraph** = `torch.npu.graph` 的捕获/回放（`aclmdlRI` + `aclmdlRIExecuteAsync`），
>   **不经过 GE 的图编译**。
> ⇒ 三者关系：vllm-ascend **用了 torchair 这个项目**，但**只用它的 ACLGraph 模式**，
> **没有用 GE 编译**。

## 3. 证据二：官方图模式文档只列两条路径，无 GE

`docs/source/user_guide/feature_guide/graph_mode.md:22–47`：

| Graph Path | Default | Description | Since |
|---|---|---|---|
| **ACLGraph (+ Npugraph_ex)** | **Yes** | Compile-time FX optimization (Npugraph_ex) + runtime capture/replay (ACLGraph) | v0.9.0rc1（Npugraph_ex 自 v0.15.0rc1） |
| XliteGraph | No | Preconfigured graph path for selected model families. Requires separate installation | v0.11.0 |

同一文档给出的编译期/运行期分工：

| `cudagraph_mode` | Compile-time | Runtime | Npugraph_ex |
|---|---|---|---|
| FULL_AND_PIECEWISE | Piecewise 编译路径 | 混合：混合 batch 用 PIECEWISE，均匀 decode 用 FULL | Disabled |
| **FULL / FULL_DECODE_ONLY** | **Npugraph_ex FX 优化** | **ACLGraph capture/replay** | **Enabled** |
| PIECEWISE | 仅基础 FX 融合 pass | ACLGraph capture/replay | Disabled |
| NONE | 无 | Eager | Disabled |

**全表没有 GE / Ascend IR / graph engine 这一行。**

## 4. 证据三：官方文档解释了"为什么 capture 单独不够"——即 GE 整图的真正障碍

`docs/source/developer_guide/Design_Documents/ACL_Graph.md`，"Host-side attention parameter
update for full graph replay"一节（**官方原文**）：

> Full graph replay on Ascend has an extra problem that upstream generic documentation does
> not cover in detail: **some attention operators need runtime metadata updates even when the
> overall graph is static.** The Ascend implementation handles this by **separating graph
> capture from host-side task parameter updates.**
>
> The flow is:
> 1. During capture, attention backends record per-graph task handles, events, workspaces,
>    and weak references to the tensors or metadata that must be refreshed.
> 2. Before replay, `update_full_graph_params()` calls the backend specific
>    `update_graph_params()` implementation.
> 3. That backend runs parameter refresh on an update stream with
>    `torch.npu.graph_task_update_begin(...)` and `torch.npu.graph_task_update_end(...)`
>    around the underlying attention operator launch.
> 4. `torch.npu.ExternalEvent` objects are used to enforce ordering between the host-side
>    update stream and the replay stream.
>
> **The important design point is that Ascend full graph support depends on
> backend-provided `update_graph_params()` hooks. Without that hook, capture alone is not
> enough to replay the correct attention state.**

**这段话是本项目最关键的官方依据**，它承认了：

1. 即使整体图是静态的，**attention 算子仍需要每步的运行时元数据更新**；
2. 所以现在的方案是"**图捕获 + host 侧参数更新**"两段式，
   而**不是**"把一切都编进一张静态图"；
3. 若没有 `update_graph_params()` 这个 hook，**光有捕获不足以正确回放**。

⇒ **GE 整图编译要求把参数变成静态的/图内的；而 attention 元数据每步都变，
这就是 GE 无法直接套用的根本原因。** 这与我们在 `docs/10` §3 推出的"前提 2：固定内存地址
是唯一缺口"完全一致。

## 5. 证据四：capture 规模与资源约束（官方列出的硬限制）

同文档"Platform mode normalization is stricter than generic upstream behavior"：

- Encoder-decoder 模型被强制为 `PIECEWISE`；
- **`use_inductor` 在 ACL graph 路径上被禁用**；
- `ASCEND_LAUNCH_BLOCKING=1` 与 ACL graph 不兼容；
- Xlite graph 可关闭 ACL graph full mode 或回退到 `FULL_DECODE_ONLY`。

以及 "Capture breadth is still constrained by runtime resources"：

> Unlike CUDA Graph on CUDA devices, ACL graph capture on Ascend **can still fail when the
> selected graph sizes consume more runtime resources than the current backend can supply.
> Piecewise mode is the most sensitive case because it captures many subgraphs and the total
> capture cost scales with model depth and configured size coverage.**

⇒ capture 本身**有资源上限**，PIECEWISE 尤其敏感（捕获数随模型深度增长）。
这对"把 `prepare_input` 也纳入捕获"是直接的负面信号：会进一步增加捕获数量。

## 6. 证据五：vllm-ascend 已经有"预编译算子"的选项，但代价明确

`vllm_ascend/ascend_config.py:599–647` 的 `AscendCompilationConfig`：

```python
enable_npugraph_ex: bool = True,      # 默认开
enable_static_kernel: bool = False,   # 默认关
```

文档对 `enable_static_kernel` 的说明（`ascend_config.py:615–621` 与官方 user guide）：

> Static kernel is suitable for scenarios with **purely static shapes or minimal shape
> changes**... when during graph capture, it will compile operator binary files with the
> corresponding shapes based on the current batch_size, **which usually takes some time.**

官方 user guide 补充了代价的**量级**（`graph_mode.md`，"Static kernel compilation"节）：

> Enabling static kernel triggers a compilation pass during the graph capture phase at
> service startup. This may add **several minutes to tens of minutes** to the startup time
> depending on the number of operators to compile and model complexity.

⇒ **"提前把算子编译成固定 shape"这条路 vllm-ascend 已经提供了**，
但它默认关闭，且**启动代价是分钟到数十分钟级**。这是理解"为什么没默认开 GE 类能力"的
关键成本锚点。

## 7. 证据六：`prepare_input` 在官方文档里的定位 = 纯 host 侧数据准备

vllm-ascend 仓库里有一篇**专门讲 prepare_inputs 的设计文档**：
`docs/source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md`（286 行）。

它的开篇（官方原文）：

> Information required to perform model forward pass: **the inputs** and
> **the corresponding attention metadata**...
> Therefore, as long as we have these two pieces of information mentioned above,
> we can perform the model's forward propagation.
> This document will explain **how we obtain the inputs and their corresponding attention
> metadata**.

它列出的"要准备的东西"**全部是 host 侧算出来的标量/数组**：

- `query start location`（prefix sum）、`sequence length`、`number of computed tokens`、
  `number of requests`、`number of tokens`、`block table`、`max query len`、
  `slot mapping`、`attention mask`
- 并明确说 **"Both of these two tables come from the `_update_states` method before
  preparing inputs"**

文档里给出的算式（`block table indices = request indices × K + positions / block size`、
`slot mapping = device block number × block size + block offsets`）**都是纯算术**。

⇒ **官方自己把 prepare_input 定义为"host 侧的数据准备"**，它的产物是
forward 图的**输入**，而不是图内的算子。这决定了它**天然在图之外**。

## 8. 由以上证据可直接导出的推论（不再需要外部调研）

| # | 推论 | 依据 |
|---:|---|---|
| 1 | **GE 整图不能"顺带"解决 `prepare_input`** | §7：它产出的是图的输入，不是图内算子 |
| 2 | **GE 的障碍与 `prepare_input` 的障碍是同一个**：元数据每步变化 | §4 官方承认"attention 需要每步运行时元数据更新" |
| 3 | **当前架构是"图捕获 + host 侧参数更新"两段式**，host 开销只是被**搬到了 update 阶段**，没有消失 | §4 的 `update_graph_params` 流程 |
| 4 | **"把元数据下沉到 device"是 GE 化的前置条件**，不是可选项 | §4：参数必须在图内/静态，否则 `update_graph_params` 不可省 |
| 5 | **成本上有先例可参照**：预编译静态 shape 算子的启动代价是分钟~数十分钟 | §6 |
| 6 | **capture 有资源上限，PIECEWISE 随模型深度增长**，再塞更多捕获有风险 | §5 |

## 9. 本文未覆盖、留给子代理的（见同目录 `01`/`02`/`03`）

- GE / TorchAir 的**完整能力清单与官方文档引用**（`01-ge-capability.md`）
- **为什么历史上没使能、有没有公开计划、MindIE 等对照组怎么做**（`02-why-not-enabled.md`）
- **`prepare_input` 各子步骤的 GE 适用性逐条分解**（`03-ge-for-prepare-input.md`）

## 10. 本地复现本文证据的命令

```bash
cd /home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend   # 0.26.0rc1, f2f74a16c

# 证据一：torchair 的 reduce-overhead + run_eagerly
sed -n '105,150p' vllm_ascend/compilation/compiler_interface.py
sed -n '149,220p' vllm_ascend/compilation/compiler_interface.py

# 证据二：官方图模式文档（只有两条路径）
sed -n '18,47p' docs/source/user_guide/feature_guide/graph_mode.md

# 证据三：官方 ACLGraph 设计文档的 host 侧参数更新一节
grep -n "Host-side attention parameter update" -A 30 \
  docs/source/developer_guide/Design_Documents/ACL_Graph.md

# 证据五：static kernel 的默认值与代价
sed -n '599,648p' vllm_ascend/ascend_config.py
grep -n "Static kernel compilation" -A 12 docs/source/user_guide/feature_guide/graph_mode.md

# 证据六：官方 prepare_inputs 设计文档
sed -n '1,60p' docs/source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md
```
