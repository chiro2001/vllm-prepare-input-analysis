# GE 整图下发调研：导航与结论汇总

> 回答两个问题：
> **Q1（本文档集的核心）**：昇腾有 GE 整图下发特性，据称能完全抹除 host 开销，
> 为什么目前没在 vllm-ascend 使能？后续会不会？**`prepare_input` 能否使能 GE？**

---

## 0. 三句话结论

1. **"完全抹除 host 开销"这个前提在官方文档里没有依据。**
   GE 下沉后仍有四项头开销，官方明确列在第 1 位的就是"模型输入输出 Tensor 到内部
   InputData/OutputData 的**数据结构转换**"；官方给出的量级是
   **盘古 71B（约 1600 个 I/O、约 6300 节点）下沉头开销 ≈ 2 ms/步**。
   且官方 `compile_cache` 配图明确画出**稳态每步仍要付 `Guards + Input 转换`**。

2. **vllm-ascend 曾经使能过 GE，在 v0.12 主动移除了。**
   旧的 `TorchAirGraph` 在官方文档里写着 *"This is the GE graph mode"*；
   移除 PR [#4814](https://github.com/vllm-project/vllm-ascend/pull/4814)（2025-12-10 合并）
   的理由只有一句：**`aclgraph is stable and fast now.`**
   官方 release notes（v0.12.0rc1）确认：
   *"Torchair graph mode is removed... Please use aclgraph instead."*

3. **`prepare_input` 不能靠 GE 解决。**
   它的**输入是 host 侧调度决策**（Python 对象 / list / dict），
   而 GE 图能接受的输入是张量（`ge.Data`）；两者之间那段 Python **每步必须再跑一遍**。
   GE 能抹掉的是**派发税**（≈400–600 µs），抹不掉**记账与决策税**。

---

## 1. 文档地图

| 文档 | 内容 | 作者 |
|---|---|---|
| **`00-source-evidence.md`** | **vllm-ascend 现状的一手源码/文档证据**（torchair 配置、官方图模式文档、ACLGraph 设计文档、`prepare_inputs` 设计文档） | root |
| `01-ge-capability.md` | GE/TorchAir 的能力、四层开销判定、能力边界、host 侧 API、46 条 URL | 子代理 |
| `02-why-not-enabled.md` | **为什么移除**：历史脉络、阻塞点清单、公开计划、MindIE 对照组、上游视角 | 子代理 |
| `03-ge-for-prepare-input.md` | **`prepare_input` 的 GE 适用性**：逐子步骤分解表、理论下限、对 `docs/10` 的审查 | 子代理 |

---

## 2. 关键事实速查

### 2.1 GE / TorchAir / ACLGraph 谁是谁

| 名字 | 是什么 | vllm-ascend 用了吗 |
|---|---|---|
| **GE（Graph Engine）** | Ascend 的图编译器 + 执行器；接收 Ascend IR，做整图编译与下沉 | **❌ 没走** |
| **torchair** | 华为维护的 torch↔Ascend 图编译桥（`gitcode.com/Ascend/torchair`） | **✅ 用了这个项目** |
| **`npugraph_ex`** | torchair 的一个模块（新版入口） | ✅ 用其 ACLGraph 模式 |
| **ACLGraph** | `torch.npu.graph` 的捕获/回放（`aclmdlRI` + `aclmdlRIExecuteAsync`），**不经 GE 编译** | ✅ **当前唯一路径** |

**关键配置证据**（`vllm_ascend/compilation/compiler_interface.py:132–135`）：

```python
else:  # torchair 路径
    # mode="reduce-overhead": use aclgraph mode, avoid fx graph to Ascend IR transformation.
    config.mode = "reduce-overhead"
    config.debug.run_eagerly = True
```

`max-autotune` 才是 GE 模式，而且**是 TorchAir 的系统默认**：
官方 `OptionValue("max-autotune", ["max-autotune", "reduce-overhead", "npugraph_ex"])`。
⇒ **vllm-ascend 是显式关掉默认，才躲开了 IR 变换。**

### 2.2 四层开销：GE 到底能省哪一层

| 层 | 内容 | GE 能否省 | 量级 |
|---|---|---|---:|
| **L1** | kernel launch（device 侧提交） | 能，但这里几乎没得省 | 0–140 µs |
| **L2** | 算子派发（Python→C++→driver 每次调用） | **部分能**（只有静态 shape 成功下沉才省） | **400–500 µs（上限）** |
| **L3** | 张量元数据构造 | **图内省、图输入输出不省** | 150–300 µs（与 L2 重叠） |
| **L4** | host 对象与决策（dataclass / dict / 调度） | **完全不能** | **0** |

**为什么 L4 是 0**：Ascend IR 里没有 dict/deque/dataclass；GE 图输入是
`ge.Data(dtype, shape, placement)`，Python 标量要么编成 `ge.Const`（编译期固定），
要么升格为 `ge.Data(shape=[], placement="CPU")`（**每步由 host 重喂**）。
⇒ **list/dict 根本无法成为图输入** —— 这就是 `seq_lens_list` 意味着
"这段 Python 每步必须跑"的根因。

> 一条反直觉的补充：**GE 图内可以有 host 计算**（官方对小 shape 算子刻意留 host）。
> 所以正确结论不是"host 工作不可入图"，而是
> **"入图的 host 工作仍每步在 CPU 跑，只省掉 InferShape/Tiling/分配/派发转换"**。

### 2.3 为什么移除：三类原因

| 类别 | 内容 | 证据 |
|---|---|---|
| **成本阻塞（主因）** | 编译"数分钟到数十分钟"；**图缓存不稳定、默认每次启动删除重编**；GE 路径需要一整套平行模型实现（`vllm_ascend/torchair/{models,ops,quantization,...}`），而 ACLGraph 复用上游 vLLM 抽象 | v0.11 `platform.py` 注释；`graph_mode.md` 的 static kernel 章节 |
| **组织原因（扳机）** | 上游 TorchAir 自己弃用 `reduce-overhead` 转推 `npugraph_ex`；vllm-ascend 的硬约束是跟住 vLLM 主线 | PR #4814 正文 |
| **硬阻塞（很少）** | GE 动态分档不接受 `FRACTAL_NZ`、档位值不能含 0/1（batch=1 无法符号化）、未命中档位直接失败；自定义算子须补 Ascend Converter | `01` §3.3 |

**用户猜测的 6 个候选阻塞点里只有 2 个是真的**：
动态 shape 对 GE 分档路径是真阻塞；**MTP/投机解码不是阻塞**（GE 时代已跑通，PR #2145）。

### 2.4 MindIE 对照组（出人意料）

MindIE 的整图是 **ATB 图**，且**它自己也同时支持 ACLGraph**。
更关键的是：MindIE 处理每步变化元数据的方式——把
`input_ids / position_ids / seq_len / block_tables` 声明为**图的输入张量**、
每步 host 刷新，另有 host 侧 `runtime_param`——
**与 vllm-ascend 现有的 `update_full_graph_params()` 是同一类解法**。

⇒ **"每步元数据"从来不是障碍**；不可照搬的是**算子与模型生态**
（ATB 是自有 C++ 算子库 + 手写 `build_graph` + 每模型单独实现）。

### 2.5 公开计划：**未找到**

2025-12-10 之后仓库里所有图相关动作都指向 ACLGraph / npugraph_ex；
`max-autotune` 检索命中 **0** 条。下一步是 Q3 2026 roadmap 的
`npugraph_ex support superkernel to reduce TPOT`（#15067，未完成）。

另有一条组合信号：给 GE 补 e2e 测试的 #3780/#3783/#3788 与降低 GE 成本的
#4756/#4757 **5 个 PR 全部 closed 未合并**，而同期"拆"的 #4814/#4856/#4927 全部合并
——移除前社区的推进方向已经是"拆"而不是"补"。

---

## 3. 直接回答："`prepare_input` 能否使能 GE？"

**不能靠 GE 解决，但 GE 能吃掉其中一部分。**

| 问题 | 答案 |
|---|---|
| 能否"把 `prepare_input` 整体编进 GE 图"？ | **不能**。它的输入是 host 决策（Python 对象），GE 图输入是张量 |
| 那 host 侧那段 Python 会怎样？ | **每步仍要跑**。GE 只让它的产物"作为图输入"被消费 |
| GE 能吃掉多少？ | **≈400–600 µs/步**（L2 派发税 + 部分 L3），占 2.76 ms 的 15–22% |
| 抹不掉的是什么？ | **L4 记账与决策税**（`_update_states` 的 dict/condense/swap、需 host 判定的数据依赖控制流） |
| 理论下限？ | **≈150–400 µs/步**，由"**调度决策的形态**"决定，不由 GE 决定 |
| 值得为它单独引入 GE 吗？ | **不值得**——移除它的全部成本理由仍然成立，而收益只有 15–22% |

### 3.1 `prepare_input` 的优化阶梯（修正后，建议以此为准）

| 阶段 | 措施 | 落点 | 是否需要 GE |
|---|---|---:|---|
| **T1** | 修 `.replace()` 缓存击穿 + decode-only 快路径 + GDN 去重 | **≈2.2–2.4 ms** | **不需要** |
| **T2** | device 段入图 / 元数据 kernel 化 / 固定缓冲 | **≈1.6–1.9 ms** | 部分需要 |
| **T3** | 架构级：账本迁 device、scheduler 直接产出定长张量、metadata 路径 C++/编译化 | **≈150–400 µs** | 不需要（超出 GE 范畴） |

⇒ **要到 TPOT=1 ms 的预算（`prepare_input` ≤550 µs），必须做 T3**，
而不是把 T1+T2 做到极限。

### 3.2 一个必须点破的悖论

**图化的前提正在由 `prepare_input` 买单**：
`FULL_DECODE_ONLY` 的 batch 固定，是靠
`_pad_query_start_loc_for_fia` + `_pad_non_spec_decode_graph_inputs`
**每步 padding** 换来的（实测 `pad_graph_inputs` = 285.9 µs/步、约 12 次 device op）。
⇒ 用"图化"论证"`prepare_input` 能变快"时要小心：
**它的一部分成本正是图化的代价。**

> 另补一刀：**"预分配 buffer ≠ 入图"**。`_pad_non_spec_decode_graph_inputs`
> 已经在写预分配 buffer，却仍要 ~12 次 device op / 步 ——
> 预分配消除的是**分配**，不是**每步的 host 调用与派发**。
> 静态缓冲是**必要**条件，不是**充分**条件。

---

## 4. 可直接落地的行动项（都不需要 GE）

| 优先级 | 措施 | 预期 | 依据 |
|---|---|---:|---|
| **P0** | 修 `.replace()` 导致的 `_num_computed_tokens_cache` 击穿 | **80–200 µs/步** | `docs/05-hotspots.md` §5bis.5.1 + `docs/10-cpython-directions/03` §3 |
| **P0** | GDN metadata 3 组重复构造 → 共享（**上游已有 PR 可抄**） | **260–600 µs/步** | vllm-ascend #16246 / vLLM #52297 |
| **P1** | 加 decode-only 快路径（跳过 spec/prefill 分支的构造） | 百 µs 级 | `03-ge-for-prepare-input.md` §7 |
| **P1** | 减少 torch 算子条数（同时打掉 `eventfd_write`/`pthread_mutex_lock`） | 1–3% | `docs/10-cpython-directions/02` §3 |
| **P2** | 属性查找提前 / cache 友好 | 1–2% | 同上 |

**明确不要做的**（有硬证据的负收益或零收益）：GC 调优（真机权重 = 0）、
immortal / deferred RC（3.12.13 无公开 API）、free-threading（本栈无 `t` 轮子，
且 `eventfd`/`mutex` 不是 GIL）、`torch.compile` 化元数据构造（ACL graph 路径已禁 Inductor，
小图 guard 开销净负）、`TorchDispatchMode` 拦截（实测 +30–40 µs/op）。

---

## 5. 验证"图真的下沉了吗"的官方判据

如果将来有人声称"整图下发生效"，用官方判据一次性终结争论：

> dump GE 图（`DUMP_GE_GRAPH`），看 build 图 txt：
> **"如果存在 `_graph_unknown_flag` 属性值且取值为 true，则为非完全静态下沉调度，
> 否则为完全静态下沉调度。"**

在当前 vllm-ascend 上跑这个 dump，预期会看到 `_graph_unknown_flag` 存在
—— 这本身就是"**没有走 GE 静态下沉**"的直接证据。
