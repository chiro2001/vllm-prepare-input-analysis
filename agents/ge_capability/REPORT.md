# ge_capability —— GE / 整图下发能力调研交接报告

**Agent**: `/root/ge_capability`
**状态**: 完成（纯文献/官方文档/网络调研；未登录任何机器、未跑 NPU、未起容器、未改任何环境）
**时间**: 2026-09-24
**写入范围**: 仅 `/home/chiro/projects/vllm/preparing-input-phase/`（下两处）；网络抓取临时物在 `/tmp`

---

## 1. 交付物

| 路径 | 内容 |
|---|---|
| `docs/11-ge-whole-graph/01-ge-capability.md` | **主交付**：704 行，9 节 + 2 附录；含四层开销表、与 ACLGraph 差异矩阵、官方 URL 清单（全部 HTTP 200 校验） |
| `agents/ge_capability/REPORT.md` | 本文件 |
| `/tmp/cann_graph.md`、`/tmp/exec1.png` | 抓取物：官方样例仓文档全文、官方 compile_cache 配图（临时，可删） |

---

## 2. 对 root 六个问题的直接答复（A–F）

### A. `CompilerConfig.mode` 有哪些取值？`max-autotune` 是不是 GE 模式？默认是什么？

- 取值**只有三个**：`max-autotune`（默认）、`reduce-overhead`、`npugraph_ex`。
  依据：官方 `CompilerConfig` 定义 `OptionValue("max-autotune", ["max-autotune", "reduce-overhead", "npugraph_ex"])`（第一个参数=默认值）。
- **`max-autotune` 就是 GE 模式**（官方别名"GE 图模式"/"Ascend IR 模式"）。
- **默认值就是 `max-autotune`** ⇒ 直接按官方快速上手写 `torchair.get_npu_backend()` 得到的就是 GE 模式；
  vllm-ascend 是**显式**改成 `reduce-overhead` + `run_eagerly=True` 才躲开 GE 的 IR 变换。
- 补充：官方已声明 `reduce-overhead`（aclgraph 的旧入口）"**将不再演进**"，推荐改用独立的 `npugraph_ex` 后端。
- **证据强度：强**（官方 API 参考 + 快速上手 + 简介，三处互证）。

### B. 图内输入的约束？`static_input` / `inplace` / `register_frozen_parameter` / `get_input` / `update` 分别解决什么？host 侧 Python list 能当图输入吗？

**先纠名**：`register_frozen_parameter`、`inplace`、`static_input`、`get_input`、`update` **这套名字在官方 TorchAir 文档和 torchair 仓库文档里都查不到**，`torchair.npu` 这个命名空间也不存在。真实存在的是：

| 想干的事 | 真实 API / 配置 |
|---|---|
| 固定权重/kv_cache 的**内存地址**（省"地址刷新"） | `config.experimental_config.frozen_parameter = True` + `torch._dynamo.mark_static_address`（GE）；npugraph_ex 是 `options={"frozen_parameter": True}` |
| 固定 tensor 的**形状**（不因 shape 变而重编译） | `torch._dynamo.mark_static(inp)` |
| 让动态维度可选档位 | `torchair.inference.set_dim_gears(t, {dim: [档位...]})` + `dynamic=True` |
| 跨进程复用编译产物 | `torchair.inference.cache_compile(..., ge_cache=True)` |
| 每步刷新算子参数（**ACLGraph 专用**） | `torch.npu.graph_task_update_begin/end`（底层 `aclmdlRICaptureTaskUpdateBegin/End`） |
| `static_input_idxs` | 只存在于 `torch_npu/npu/_graph_tree.py`（CUDA 式 `torch.npu.graphs.make_graphed_callables` 的移植实现），**属于 ACLGraph 树，不是 GE** |

**host 侧 Python list 能不能当图输入？能，但只有三种合法形态**（官方有原文依据）：
1. **改成 Tensor 图输入**（GE 首选）：官方为 FIA 定制了 `torchair.ops.npu_fused_infer_attention_score`，把 `actual_seq_lengths*` 从 `SymInt[]` 改成 **int64 Tensor**，动机原文就是"减少 Host→Device 拷贝、保障 Tiling 下沉"；约束是"**只支持图模式，不支持 Eager**"。
2. **`dynamic=True` + `mark_static` + `tiling_schedule_optimize`**：官方"路径4"，此时 `SymInt[]` 被泛化为符号 → 值变更**不触发重编译**，且 GE build 图仍是静态图（可整图下沉）。三条件缺一不可。
3. **`dynamic=False`**：scalar/SymInt[] 被编成 **`ge.Const`**（常量），**值一变化就重新编译**。

另外官方明确要求：动态信息"应由框架构造后**作为显式输入传给模型**，而**不是**在模型内部临时生成 Python 标量"。
**证据强度：强**（TorchAir API 文档 + 官方案例文档；API 名的否定结论基于官方 API 列表 + 全仓文档检索）。

### C. 动态 shape 的支持：`dynamic_dims` / `input_shape_ranges` / 分档；decode 变 batch 怎么办？

- **官方的两条路**：① **shape 范围**（动态 shape，`ge.inputShape` 用 `8~20` 写法）；② **分档 gear**（`ge.dynamicDims` / `DYNAMIC_DIMS`，TorchAir 侧是 `set_dim_gears`）。**没有 `input_shape_ranges` 这个选项名**；`ge.exec.dataInputsShapeRange` 官方已标"**已废弃，请勿使用**"。
- **最关键的一条绑定**：官方原文"动态 shape 图……**只能采用 Host 调度**"；静态 shape 图"一般都能采用**下沉调度**"。⇒ **动态 batch 想拿到"整图下发"，必须先分档把每个档位变成静态子图**（官方："每个档位的图转换为静态子图，能一次全部下发到 Device 侧"）。
- **分档的硬约束**：仅 GE 图模式 + 仅整图优化场景；需 `dynamic=True`；档位总数 ≤100（CANN 8.5 文档写 `(1,100]`，建议 3~4 档）；**档位值不能含 0 或 1**；形状不在档位内**会报错**；参与分档的 tensor 不能是私有格式（FRACTAL_NZ/NC1HWC0）。
- 另一个必须知道的坑：静态 shape 图里若存在**值依赖算子**，"该算子默认使用 **Host 调度**"⇒ 一个值依赖算子就能破坏"纯下沉"。
- **证据强度：强**。

### D. GE 到底省哪一层？官方有没有 host 侧开销对比数字？

- **省哪一层**：见主文档 §2.2 的四层表。一句话：**GE 把 host 成本从"按算子计费"降到"按步计费"，同时把低于"按步"的部分留给了你**。
  - (a) kernel launch：**有条件下沉后能省**（执行期只有 1 个模型执行 Task）；动态 shape 不行。
  - (b) op 派发：**有条件下沉后能省**；动态 shape 下"图模式 Host 调度"只做到官方口径的"**调度性能持平或略优**"（相对 eager 双线程），并省掉 Python 栈与冗余数据结构转换；此时每个算子的下发被拆成 **InferShape + Tiling + AllocMemHbm + Launch** 多个 Host Kernel。
  - (c) 张量元数据构造：**图内算子的能省**（InferShape/Tiling/内存编排都在编译期）；**图输入输出的不能省**——官方"下沉头开销"第 1 项就是"输入输出 Tensor 到内部 InputData/OutputData 数据结构转换"，第 2 项是地址刷新。
  - (d) host 对象分配：**图外一律不能省**；图内 device 内存可在编译期编排。
  - 附加第 5 项：**Guards 每步都在**（官方配图明确画出稳态执行 = Guards + Input 转换 + 图执行）。
- **数字**：
  - 有：LLaMA-7B Decoding 下沉 vs Host 调度 **E2E −18 ms、吞吐 +37%**；盘古 71B（~1600 I/O、~6300 节点、10 个地址变更）**下沉头开销 ≈2 ms/步**；Tiling 缓存使 Host 侧 Tiling 开销 **−50%+**（pangu/llama2）；小 Shape 算子优化 LLaMA2 **1.062 s → 1.009 s（+5%）**。
  - **没有**：GE vs eager 同口径的 host 侧开销对比（任何模型）；GE 首次编译的**任何绝对时长**；A3 上 GE vs ACLGraph 的头对头数字。
- **结论**：**官方文档里没有任何"GE 能完全抹除 host 开销"的依据**，相反官方自己画出了稳态仍要走的 `Guards + Input 转换` 和 ≈2 ms 的下沉头开销。**证据强度：强（对上述全部数字与定性结论）**。

### E. 算子覆盖：自定义算子 / Triton？

- **ATen**：官方有 153 行支持清单，**清单外需自研 Converter**；用 `config.debug.fx_summary.type="csv"` 可导出"已支持/部分支持/未实现"。
- **自定义算子**：GE（max-autotune）**只接受 Ascend C 工程化的 `aclnnXxx`** 实现；官方原文"以 **Kernel 直调**方式开发的 NPU 实现不会生成 Ascend IR 注册逻辑……无法完成 PyTorch 算子到 Ascend IR 的转换"。另有一条坑：**fallback 形式下发的算子在静态 shape 场景无法整图下发，必须断图**。
- **Triton**：GE 有"Triton 入图"官方文档，但 ① **当前只支持 TensorFlow 前端**；② 实现方式是 `EagerExecuteOp` + `aclrtLaunchKernelWithHostArgs`（**Host 侧 launch**），不是编进 Ascend IR。PyTorch 侧的 Triton-Ascend 走的是 `torch.compile(backend="inductor")`，与 torchair GE 是**两条互斥后端路线**。
- **FIA（与本项目最相关）**：GE 有专用 `torchair.ops.npu_fused_infer_attention_score`，`actual_seq_lengths*` 支持 Tensor + 配合 Tiling 下沉；**仅 GE 图模式可用**。
- **证据强度：强**（除"torchair+PyTorch+Triton 直接入图"= 查不到）。

### F. 官方文档链接

主文档**附录 A** 给出 46 条 URL，分三组（TorchAir 官方文档 / CANN·GE·Runtime 官方文档 / 官方样例仓与社区），全部于 2026-09-24 做过 HTTP 200 校验。

---

## 3. 本次调研中"可能改变决策"的几条

1. **`max-autotune` 是默认值**：也就是说 vllm-ascend 现在的形态不是"默认没开 GE"，而是"**显式关掉了默认的 GE 模式**"。这对 `02-why-not-enabled.md` 的论证是正面支撑。
2. **GE 官方配图显示稳态仍付 `Guards + Input 转换`**：任何"GE 能不能省掉 X"的讨论，必须先扣掉这两项和下沉头开销（71B 量级 2 ms）。
3. **官方为 FIA 元数据给了正解**（Tensor 输入 + Tiling 下沉 + `mark_static` 三件套，或 `torchair.ops` FIA 接口）：这条与本项目 `docs/10` 的推论**独立地吻合**——"参数 Tensor 化 + 固定缓冲"是可行方向，但**要求换 FIA 调用接口且只在 GE 模式下成立**。
4. **下沉与否有官方判据**：dump GE build 图看 `_graph_unknown_flag` 是否为 true。后续任何"GE 是否真的整图下发"的争论都可以用这条一次性收敛。
5. **vllm-ascend 维护者 2026-05 明确说 torchair graph mode 已 sunset**（issue #588 被 `wontfix` 关闭）。⇒ 不要把"短期内 vllm-ascend 会切 GE"写进方案假设。

---

## 4. 证据质量与自查

- 主文档每条边界都挂了 `E<x.y>`，格式为 **来源 / 机制 / 能力边界 / 证据强度**；表格与正文引用一一对应。
- **只用了两类来源**：华为官方文档/官方仓库（强）、官方样例仓/官方 issue 维护者回复/社区文章（中/弱）。**没有任何一条结论来自模型记忆或推测**；第 9 节明确标注为"推断"。
- 所有 URL 逐个 `curl -o /dev/null -w '%{http_code}'` 校验为 200（含 hiascend.com、gitcode.com、docs.vllm.ai、docs.pytorch.org、github、gitee）。
- 唯二**不采信**的材料：① CSDN 那篇"Mixtral 编译 15 分钟"（其引用的 `GE_FUSION_PASS_MODE`/`GE_WARMUP_SHAPES` 等环境变量在官方文档中查不到，已标注 **弱/不建议引用**）；② 任何把 `aclmdlRI*` 当作 GE API 的说法（已用官方 Runtime 文档 + torch_npu 源码澄清）。

## 5. 遗留 / 建议的下一步（未做，仅建议）

1. 若要坐实"GE 到底能压到多少 ms/步"，只能在有卡环境上按官方判据做一次实验：
   `dynamic=True + mark_static + tiling_schedule_optimize` 编译，dump GE build 图确认 `_graph_unknown_flag`，再量稳态每步 host 时间。本任务范围（纯文献）不含此步。
2. 若需要 GE 首次编译时长的真实数字，建议在实验环境里量 `cache_compile` 冷/热两次启动的差值，而不是继续找公开数字（公开数字不存在）。
3. `torchair` / `npugraph_ex` 的上游文档在 26.0 与 26.1 之间有重组（`config.mode` vs 独立 `npugraph_ex` 模块）。写方案引用时**优先引 26.0 分支的 `docs/zh/**` 路径**（本文已验证这些 URL 稳定 200）。
