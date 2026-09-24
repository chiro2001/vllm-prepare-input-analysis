# `prepare_input` 能否也走向"图下发"？—— 基于源码与实测的向前判断

> 问题：未来 TPOT 降到 1 ms 时，decode 阶段的算子下发已经从 eager 走向图下发
> （cudagraph / ACLGraph 整图 replay）。那 `prepare_input` 阶段是否也会出现类似的
> "图下发"技术？
>
> 本文每条判断都挂在我们自己的实测或可核对源码上。`[实测]` = 本项目在 chip3 上量到的
> 数字；`[源码]` = 给出文件:行号；`[推断]` = 写清验证方法。

---

## 0. 结论速览

1. **这件事已经发生了——而且正是它让 `prepare_input` 变成瓶颈。**
   同一模型同一负载，eager 下 `prepare input` 占单步 **6.7%**，图模式下占 **55.3%**；
   但它的**绝对耗时几乎没变**（2.843 ms → 3.132 ms，+10%），变的是分母
   （单步 42.46 ms → 5.66 ms，**7.5×**）。`[实测]`
   ⇒ **图化没有减少 host 开销，只是把 device 时间挤掉了。**

2. **"算子图化"不等于"元数据图化"。** Ascend 现在的 decode 已经是"图捕获 +
   每步参数更新"（`graph_task_group_begin/end` + `graph_task_update_begin/end`），
   但**每个 attention 算子每步仍要从 Python 重新发起一次调用**。
   host 侧派发成本**没有消失，只是换了位置**（从 launch 挪到 task update）。`[源码]`

3. **好消息：图化的三个前提，两个已满足。**
   - 固定控制流：`FULL_DECODE_ONLY` 已把 decode 的 batch 钉到 capture size ✅
   - 参数可由 device 提供：`full_graph_fia_v2` **已经用 tensor** 传
     `actual_seq_lengths_kv`（`attention_v1.py:1029`），只有 eager 路径还在用
     Python list（`1248/1258/1269`）✅
     ⇒ **"FIA 必须吃 list"不是硬约束，图路径已经绕开了它**
   - **固定内存地址：❌ 这是唯一真正的缺口** —— 现在每步 `torch.empty/zeros`
     与 dataclass 构造都在造新对象，地址漂移，无法被图捕获引用。

4. **所以可行的技术形态不是"把 `prepare_input` 塞进图"，而是"把 `prepare_input`
   改造成固定缓冲的 in-place 写入 + 参数 tensor 化"。** 这个形态在代码里**已有雏形**：
   `_pad_non_spec_decode_graph_inputs`（`gdn_attn_builder.py:332`）就在往预分配的
   `non_spec_state_indices_tensor` / `non_spec_query_start_loc` /
   `non_spec_actual_seq_lengths` 里写。`[源码]` **方向已经对了，只是覆盖不全。**

   > ⚠️ **但"预分配 buffer ≠ 入图"**（子代理补的刀）：
   > `_pad_non_spec_decode_graph_inputs` 实测仍是 **285.9 µs/步、约 12 次 device op**。
   > 预分配只消除了**分配**，没有消除**每步的 host 调用与 device 派发**。
   > ⇒ 静态缓冲是**必要**条件，但远不是**充分**条件。

5. **但有一个不可图化的内核**：`prepare_input` 的输入是 **host 侧的调度决策**
   （哪些请求、各调度多少 token、谁被 preempt），由 scheduler 在 host 决定；
   `_update_states` 的 dict/condense/swap 记账本质是 host 数据结构操作。
   ⇒ **"决定"不可图化，"构造 + 搬运"可图化。**

6. **收益上限的诚实估计**：图化 + 静态缓冲 + 去冗余最多把 `prepare_input`
   从 **2.76 ms 压到 ~1.2–1.5 ms**（工程量大）。**这仍然 > 1 ms。**
   ⇒ 若真上 TPOT=1 ms，`prepare_input` 会成为**硬约束**，除非做架构级改造
   （C++/编译化元数据路径，或模型结构变化）。

---

## 1. 先看一个已经发生的事实：图化 decode，正是 `prepare_input` 变瓶颈的原因

我们自己的历史 phase 数据（0.8B / TP1 / decode，`[实测]`，
`data/historical/phase_per_step_configs.csv`）：

| 模式 | `prepare input` | 单步 | **占比** | `forward` |
|---|---:|---:|---:|---:|
| eager | 2.843 ms | 42.459 ms | **6.7%** | 36.261 ms |
| graph（`FULL_DECODE_ONLY`） | 3.132 ms | 5.661 ms | **55.3%** | 0.953 ms |
| **变化** | **+10%** | **−87%** | **8.3×** | −97% |

**这张表是回答整个问题的关键。** 逐条读：

1. `prepare input` 的绝对成本**几乎不动**（2.843 → 3.132 ms）。这符合它**纯 host 侧**的
   性质：真机 topdown 显示同步调用只占 **0.27%**，它的成本**与 device 是否图化无关**。`[实测]`
2. 图化把 `forward` 从 36.26 ms 砍到 0.95 ms（**38×**），单步砍了 7.5×。
3. 于是 `prepare input` 的**占比**从 6.7% 跳到 55.3% —— **不是它变慢了，是分母塌了**。

⇒ **外推**：TPOT 继续下降（device 继续变快），`prepare_input` 的占比只会继续上升，
直到它从"占比问题"变成"**绝对时间问题**"—— `2.76 ms > 1 ms`，物理上无法满足。

> ### ⚠️ 一个必须同时讲的限定条件（来自上游的实测反例）
>
> 占比上升 ≠ 优化它一定有端到端收益。vLLM 上游有一个**直接对口的实验**
> （PR **#47924**，draft 后关闭）：把 slot-mapping 折进 FULL decode cudagraph，
> 结果**字节级一致、无回归，但 e2e 收益 = 0** ——
> 省下的约 130 µs **被 async scheduling 掩盖掉了**（CPU 有"一整拍 device 时间"的预算，
> 只要 `T_cpu < T_dev` 就不会体现在吞吐上）。
>
> ⇒ **判据要精确**：只有当 `T_prepare` **真的挤到 `T_dev` 之上**（或 T_cpu 合计 > T_dev），
> 优化才转化为端到端收益。`docs/07` §4 给出的临界条件 `T_prepare > B_cpu` 就是这个意思。
> 反过来说：**在 `T_prepare < T_dev` 的区间里，`prepare_input` 的高占比只是"CPU 占用率高"，
> 不是吞吐瓶颈** —— 这与 `docs/07` 的"被掩盖"形态是同一件事。

## 2. "算子图化"到底改变了什么：device 侧省了，host 侧没省

### 2.1 捕获侧：把算子放进 task group

```python
# vllm_ascend/attention/attention_v1.py:992
torch.npu.graph_task_group_begin(stream)
torch_npu.npu_fused_infer_attention_score.out(
    query=query, key=key, value=value,
    actual_seq_lengths=actual_seq_lengths_q,        # <- 捕获时的参数
    actual_seq_lengths_kv=actual_seq_lengths_kv, ...)
handle = torch.npu.graph_task_group_end(stream)
graph_params.handles[num_tokens].append(handle)
```

### 2.2 重放侧：每步**重新发起**每个算子的调用

```python
# vllm_ascend/attention/attention_v1.py:476  update_graph_params(...)
for key, param, handle, event in zip(...):
    (query, key_cache, ..., output) = param
    seq_lens = forward_context.attn_metadata[key].seq_lens
    torch.npu.graph_task_update_begin(update_stream, handle)
    torch_npu.npu_fused_infer_attention_score_v2.out(   # <- 每步每层一次 Python->C++ 调用
        ..., actual_seq_qlen=actual_seq_lengths_q,
             actual_seq_kvlen=seq_lens, ...)
    torch.npu.graph_task_update_end(update_stream)
    event.record(update_stream)
```

**这就是要点**：图化省掉的是 **device 侧的 kernel launch 与调度**，
而 **host 侧"每算子每步一次"的 Python→C++→driver 调用仍然存在**，
只是从 "launch" 变成了 "update"。

> ### ⚠️ 重要修正（子代理查证 CANN 官方文档后得出，比原文更悲观）
>
> **CANN 官方文档明说：`graph_task_update` 比单独下发更贵。**
>
> > *"updating tasks is more time-consuming than delivering tasks separately"*
> > —— CANN 文档，转引自 `docs/10-cpython-directions/05-graph-dispatch-host-side.md` §2
>
> 也就是说：**"图化能省 host 开销"这个直觉在昇腾上不成立**。
> `aclmdlRICaptureTaskUpdate*` 的语义是"在 update 区间内**把同一批算子带新实参再下发一遍**"，
> 且要求**任务数量与类型与捕获时一致**、不支持并发更新。
> 它存在是为了**保证整图 replay 的参数正确性**，不是为省 host 时间。
>
> 这条直接影响到下面 §5.2 的排序与 §7 的判断：**不要把"图化"当成 host 开销的解药**，
> 真正的解药是**减少算子/对象条数**与**把构造下沉到 device**。

> 我们实测 920B 上一次小 NPU 算子派发 **3–12 µs**（`[实测]`，
> `scripts/npu_dispatch_bench.py`）。6 层 full-attention × 每步一次 update ≈
> 数十 µs 量级 —— 这解释了为什么 `forward` 在图模式下仍有 830 µs 而不是接近 0。

### 2.3 推论

**"把 `prepare_input` 也图化"不会自动让它变快**，因为 `prepare_input` 里
**没有"可以被图捕获省掉的 kernel launch"**（它的 device 交互本来就少而小）。
它的成本是 **host 侧的 Python 对象构造 + 属性查找 + 小算子派发**。

图化能给它的帮助是**另一条路**：把"每步新建对象"换成"每步写固定缓冲"，
从而**减少对象数量与分配**（见 §4）。这与"图"的关系是间接的 ——
真正起作用的是**静态缓冲**，不是图本身。

## 3. 图化元数据的三个前提：两个已满足，一个是大缺口

要让一段 host 逻辑的产物能被图捕获引用，需要同时满足三个条件。逐个对照：

| # | 前提 | 现状 | 证据 |
|---:|---|---|---|
| 1 | **控制流固定**（分支形状不随 step 变） | **接近满足**：`FULL_DECODE_ONLY` 把 batch 钉到 capture size；GDN builder 已按 `decode_cudagraph_max_bs` 预分配 | `[源码]` `gdn_attn_builder.py:238`、`build_for_cudagraph_capture` 分支（`attention_v1.py:3197` 附近） |
| 2 | **内存地址固定**（图引用固定 buffer） | **❌ 最大缺口**：每步 `torch.empty_like`/`torch.zeros`/22 字段 dataclass 全新建 | `[源码]` `gdn_attn_builder.py:128`（`_build_actual_seq_lengths` 的 `torch.empty_like`）、`build()` 里的 `GDNAttentionMetadata(...)` 构造 |
| 3 | **参数能由 device 侧提供**（避免 host list） | **✅ 已经做到** | `[源码]` 见 §3.1 |

### 3.1 前提 3 已有决定性先例：图路径已经不用 Python list 了

```python
# 图路径：attention_v1.py:1020  full_graph_fia_v2()
key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(...)
actual_seq_lengths_kv = attn_metadata.seq_lens        # <- tensor

# eager 路径：_get_fia_params() 的 DecodeOnly / PrefillCacheHit / chunked 分支
actual_seq_lengths_kv = attn_metadata.seq_lens_list   # <- Python list（1248/1258/1269）
```

同文件注释把这件事说得很直白（`attention_v1.py:350–354`）：

> `full_graph_fia_v2` passes the **seq_lens tensor (not `seq_lens_list`)** as
> `actual_seq_kvlen` during graph capture, and `_get_fia_params` derives the
> PrefillCacheHit batch size from `seq_lens.shape[0]`, so the tensor has to
> carry the dummy request too.

⇒ **"FIA 算子必须吃 Python list"这个看似硬约束，在图路径上已经被绕开了。**
这说明"元数据 tensor 化"**不是理论设想，而是已经在跑的工程实践**，
只是目前只覆盖了图捕获那一小段，没有覆盖 `prepare_input` 的其余部分。

eager 路径继续用 list 的真实原因也值得说清：`seq_lens_list = seq_lens.tolist()`
是**为了规避 GPU→CPU 的 D2H 同步**（源码原本 `seq_lens.cpu()` 要等设备），
而 `seq_lens_list` 来自 host 侧账本（`optimistic_seq_lens_cpu`）。
⇒ **list 是"host 账本"的自然形态，tensor 是"图上路径"的自然形态**，
两者的分歧根源是**账本存在 host 还是 device**。

## 4. 已有雏形：静态缓冲模式就写在代码里

`prepare_input` 的图化改造不是从零开始。GDN builder 里已有一个**教科书式范例**：

```python
# vllm_ascend/ops/gdn_attn_builder.py:332
def _pad_non_spec_decode_graph_inputs(self, state_indices, query_start_loc, ...):
    """Refresh fixed buffers consumed by a non-spec decode graph."""
    padded_state_indices = self.non_spec_state_indices_tensor[:graph_batch_size]   # 预分配
    padded_state_indices[num_decode_tokens:].fill_(NULL_BLOCK_ID)                  # in-place
    padded_state_indices[:num_decode_tokens].copy_(state_indices[...], non_blocking=True)

    padded_query_start_loc = self.non_spec_query_start_loc[:graph_batch_size + 1]  # 预分配
    padded_query_start_loc[:num_decode_tokens + 1].copy_(query_start_loc[...], ...)
    query_padding.copy_(padded_query_start_loc[num_decode_tokens].expand_as(query_padding), ...)
```

`__init__` 里对应的预分配（`gdn_attn_builder.py:238–265`）：

| 缓冲区 | 被谁消费 |
|---|---|
| `non_spec_state_indices_tensor` / `spec_state_indices_tensor` | 图读的 state 索引 |
| `non_spec_query_start_loc` / `spec_query_start_loc` | 图读的 cu_seqlens |
| `non_spec_actual_seq_lengths` / `spec_actual_seq_lengths` | 图读的 actual seq len |
| `spec_sequence_masks` / `num_accepted_tokens` / `spec_token_indx` | 投机解码输入 |

**这套模式正是"元数据图化"的标准形态**：预分配 + in-place 写 + 图引用固定地址。
它已经覆盖了 GDN 的 decode 分支，但**没覆盖**：

- `_build_actual_seq_lengths` 仍然 `torch.empty_like`（`gdn_attn_builder.py:128`）
- `GDNAttentionMetadata(...)` 这 22 字段 dataclass 仍然每步新建
- 3 个 KV group 各建一套，**互不共享**
- `_treat_single_token_prefills_with_state_as_decodes` 的 `.replace()` 造新对象
  ⇒ 击穿 `_num_computed_tokens_cache`（见 §5.2 第 2 条）

## 5. TPOT→1 ms 的预算推演：图化是必要的，但远远不够

### 5.1 预算缺口

异步调度下 `T_step ≥ max(T_cpu_engine_step, T_dev_step)`，所以 **CPU 与 device 都必须 < 1 ms**。

| 预算项 | 现状（B=1 实测） | 1 ms 目标 | 缺口 |
|---|---:|---:|---:|
| schedule + RPC | ~50–150 µs `[推断]` | ≤150 µs | OK |
| **`prepare_input`** | **2757 µs** `[实测]` | **≤550 µs** | **5.0×** |
| forward（含图参数 update） | 831 µs `[实测]` | ≤250 µs | 3.3× |
| post process + sample | ~300 µs `[实测]` | ≤150 µs | 2.0× |

### 5.2 可做的清单（按预期收益排序，全部是本项目已定位的具体项）

| # | 措施 | 预期收益 | 依据 | 难度 |
|---:|---|---:|---|---|
| 1 | **3 个 GDN KV group 共享 metadata**（去重复构造） | **最大**：908 µs 里若 2/3 是重复，可省 ~600 µs | `[实测]` 3 × 303 µs；三个 group 的 `cm` 只在 `block_table_tensor`/`slot_mapping` 上不同（`model_runner_v1.py:3155` 的 `copy(cm_base)`），**公共部分已共享但 builder 内部仍在重建** | 中 |
| 2 | **修 `.replace()` 缓存击穿** | **~130 µs**（机理已验证） | `[实测]` `compute_num_computed_tokens` 子树 67.5% 在 `PyNumber_Subtract`；3 次重算 | **低** |
| 3 | **静态缓冲**：消掉每步 `torch.empty_like`/dataclass 构造 | ~200–400 µs `[推断]` | `[源码]` `gdn_attn_builder.py:128`；实测 `_PyObject_Malloc` 1.16% + `malloc` 1.06% | 中 |
| 4 | **FIA 参数 tensor 化**：eager 路径的 `seq_lens_list` 也换 tensor | 数十 µs `[推断]` | `[源码]` 图路径已验证可行（§3.1）；收益主要在去掉 `tolist()` 与后续 `len()`/索引 | 中 |
| 5 | **metadata 部分下沉 device**（把"构造"变成 kernel） | 未知，**需实测** | `[源码]` 先例：`slot_mapping` 已是 Triton/NPU kernel（`in.slot_mapping` 仅 138 µs，CPU 只剩 launch） | 高 |

> ### ⚠️ 本表已被子代理审查并**修正**（`docs/11-ge-whole-graph/03-ge-for-prepare-input.md` §5–§6）
>
> 我原来的 "2.76 ms → **1.2–1.5 ms**" 估计**站不住**，两处错误：
> 1. **与"静态缓冲"重复计入收益**（GDN 去重省下的部分本就来自静态缓冲）；
> 2. **GDN 去重按 600 µs 高估**：按 PR #16246 的实际比例应约 **400 µs**。
>
> **修正后的阶梯**（建议以这个为准）：
>
> | 阶段 | 措施 | 落点 |
> |---|---|---:|
> | T1 | 修 `.replace()` 缓存击穿 + decode-only 快路径 + GDN 去重 | **≈2.2–2.4 ms** |
> | T2 | 再做 device 段入图 / 元数据 kernel 化 / 固定缓冲 | **≈1.6–1.9 ms** |
> | T3 | 架构级（账本迁 device、scheduler 直接产出定长张量、metadata 路径 C++ 化） | **≈150–400 µs** |
>
> **理论下限 ≈150–400 µs/步**，由"**调度决策的形态**"决定，不由 GE 决定。
> 要到 ≤550 µs，**必须做 T3**，而不是把 T1+T2 做到极限。

### 5.3 一个必须承认的下限

即使全部做到，**"决定"这部分工作不可能归零**：
哪些请求被调度、各分配多少 token、谁被 preempt/condense —— 由 scheduler 在 host 决定，
`_update_states` 的 `add_request`/`condense`/`swap_states` 是 host 数据结构操作
（`[源码]` `gpu_input_batch.py:338/686/569`）。
所以 `prepare_input` 的**理论下限不是 0，而是"把调度决策翻译成设备格式"的成本**。
这部分若也要压到 µs 级，只能靠**改数据结构 + 编译化**，而不是图化。

## 6. 反向风险：什么都不做的话，情况只会更糟

1. **占比会持续上升**：eager 6.7% → graph 55.3% 已经证明了机制。TPOT 继续降 ⇒ 占比继续升。
2. **TP 下会被 straggler 放大**：每个 rank 各跑一份 `prepare_input`，步时间由最慢 rank
   决定，CPU 段的抖动以 straggler 形式直接吃吞吐（`[推断]`，`docs/07` §4.4 给了公式）。
3. **async scheduling 只能掩盖一拍**：`max_concurrent_batches=2` + `prepare_inputs_event`
   把 CPU 领先量限制在约一拍 device 时间。当 `T_cpu > T_dev` 时，
   **CPU 就是真瓶颈，掩盖机制失效**（`[源码]` `gpu_model_runner.py:3809–3822`）。

## 7. 直接回答："会不会出现类似 `prepare_input` 的图下发技术？"

**会，但形态与 decode 算子图化不同。** 具体判断：

| 阶段 | 技术形态 | 我们的证据 / 现状 |
|---|---|---|
| **短期（已在发生）** | **静态缓冲 + in-place 写 + 参数 tensor 化**；把"算子图化"的同款思路用到元数据上 | `_pad_non_spec_decode_graph_inputs` 已是范例；`full_graph_fia_v2` 已用 tensor |
| **中期（可预见）** | **metadata 构造下沉 device**（把"构造"变成 kernel）+ **异步重叠**；host 只写少量标量 | 上游 MRV2 明写用 Triton kernel 准备 `input_ids`/`positions`/`query_start_loc`/`seq_lens`，理由是 *"avoids Python bottlenecks"*；先例：`slot_mapping` 已是 Triton/NPU kernel |
| **长期（不确定）** | **元数据路径 C++/编译化**，或**账本迁到 device** | 业界先例：TensorRT-LLM 纯 C++ runtime；SGLang 的 overlap scheduler |

> **注意"中期"这一行与"图化"的区别**：上游选的路线是 **"下沉 device + 异步重叠"**，
> **不是"把元数据编进图"**。这两条路的差别是本质性的：前者减少 host 工作量，
> 后者只是把 host 工作换个地方做（且 CANN 明说 update 比 dispatch 更贵）。
> `docs/10-cpython-directions/05-graph-dispatch-host-side.md` §7 给出了完整的
> "可图化分解表"，结论是：**`prepare_input` 里真正该做的是下沉，不是图化。**

**三条判据**：

1. `prepare_input` 的成本与 device 图化**无关**（§1 实测证明）⇒ 它**不会**"被图化顺带解决"。
2. 图化三前提里 **#3 已满足、#1 接近满足**，唯一缺口是 **#2 静态缓冲**，
   而这在工程上**可做**（§4 已有范例）⇒ 技术上会走这条路。
3. 但"决定"部分不可图化（§5.3）⇒ **图化只能拿到一部分收益**，
   真正的天花板由"调度决策翻译成本"决定。

**最关键的量化结论**：

> **1.** TPOT 降到 1 ms 时，`prepare_input` 目前是 2.76 ms —— **不降就是硬失败**。
> **2.** 但它**不会**被"图化"顺带解决（§2 的 CANN 证据：update 比 dispatch 更贵）。
> **3.** 真正的解药是**减少工作量**，且**上游已经在做**（可直接抄）：
>    - vLLM **#52297**（open）：*Move GDN common metadata compute out of per-group build*，
>      自述 `build()` **900 µs → 300 µs**、BS=1 e2e **+61%**（H200/DFlash）
>    - vllm-ascend **#16246**（open）：*Share group-invariant metadata across kv-cache groups*，
>      就是我们识别的"3 组 GDN 重复构造"，自述 **6.3 → 4.5 ms/步（−28.6%）**，
>      做法是新增 `GDNGroupInvariantCache`（**我们手上的 0.26.0rc1 里没有这个类**）
>    - 折算到我们的 **908 µs/步**：**260–600 µs/步**（按两家自述比例线性折算）
> **4.** 再叠加 `.replace()` 缓存修复（**80–200 µs/步**，见 `docs/05-hotspots.md` §5bis.5.1）
>    与静态缓冲，才可能接近 1 ms 预算。
> **5.** 当 `T_prepare < T_dev` 时，以上优化**不一定体现为端到端收益**（可能被
>    async scheduling 掩盖，见 §1 的 PR #47924 反例）——**收益兑现的前提是判据成立**。

## 8. 需要实验验证的三件事

| # | 待验证 | 方法 | 判据 |
|---:|---|---|---|
| V1 | 3 个 GDN builder 的 908 µs 里**有多少是真重复** | 让 3 个 KV group 共享一份 `GDNAttentionMetadata`（只改 `block_table`/`slot_mapping`），测 `pi: am.builder_build@AscendGDNAttentionMetadataBuilder` 的下降 | 若省 ≥400 µs，#1 就是最高优先级 |
| V2 | `.replace()` 缓存修复的真实收益 | 改 `_treat_single_token_prefills_with_state_as_decodes` 保留 `_num_computed_tokens_cache`，A/B 对比该函数的 wall | 预期 ~130 µs |
| V3 | `graph_task_update` 的单次 host 成本 | 在 `update_graph_params` 加探针，测 6 层 FIA 的 update 总时长 | 若 >100 µs，"图化省 host"在本平台不成立 |

前两项**低风险、可快速验证**；V3 决定"图化"这条路在本平台的实际收益上限。

## 附：相关文档

- `docs/05-hotspots.md` §2.3/§2.4 —— GDN builder 的 perf 证据、"不是等卡"的判定
- `docs/07-bottleneck-analysis.md` §4 —— 瓶颈判据、临界条件、TP straggler 放大
- `docs/10-cpython-directions/` —— CPython 优化方向的文献调研（5 个子代理产出）
