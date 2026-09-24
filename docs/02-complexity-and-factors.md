# prepare_input 复杂度模型与影响因素（vLLM 0.26.0 + vllm-ascend 0.26.0rc1）

> 配套文档：`01-prepare-input-code-logic.md`（行号与调用链）。
> 本文的记号：**B** = `input_batch.num_reqs`；**T** = `total_num_scheduled_tokens`；
> **S** = 单请求已算 token 数；**P** = prompt 长度；**K** = `cdiv(max_model_len, 128)`（每请求最大块数）；
> **G** = `len(kv_cache_config.kv_cache_groups)`（KV 组数，非 hybrid 模型为 1）；
> **L** = 模型层数；**n** = 本拍 draft token 数（MTP）。

---

## 1. 一阶复杂度模型

把 `prepare input` 的 CPU 时间拆成 6 个可独立测量的项：

```
T_prepare(B, T, S, G, spec, flags)
  ≈ T_pystatic(B, G)            # ① Python 固定循环
  + T_numpy(T, B)               # ② numpy 向量计算
  + T_aten(B, T, S)             # ③ ATen/CPU 张量算子
  + T_h2d(B·K·G + c·B + T)      # ④ 小包 H2D 搬运 + launch
  + T_build(B, G, L)            # ⑤ attention metadata 构造（含 .tolist()）
  + T_upd(B, S, churn)          # ⑥ _update_states（含 condense）
  + T_sync(device)              # ⑦ 同步等待（非 CPU 计算，但计入 wall）
```

| 项 | 主导因子 | 阶 | 主要来源 |
|---|---|---|---|
| ① `T_pystatic` | B | **O(B)**，常数因子大（≈0.1–0.3 µs/req/循环） | `_compute_prev_positions`（core `1758–1759`）、`_prepare_input_ids` 异步循环（`1799–1826`）、`_build_attn_state` list comp（ascend `914–921`）、`num_tokens = [...for r in req_ids]`（ascend `1087`）、`discard_request_indices` |
| ② `T_numpy` | T | **O(T)**（含 1 次分配） | `np.repeat`（两步）、`np.cumsum`、`np.subtract`、`np.add(positions)`、`np.array(tokens)` |
| ③ `T_aten` | T | **O(T)** | `torch.index_select`（input_ids）、`torch.add`（optimistic seq_lens）、`copy_` 切片 |
| ④ `T_h2d` | B·K·G, T | **O(B·K·G + T)** 字节 + **O(#调用) launch** | `commit_block_table`、`query_start_loc.copy_to_gpu()`（**全量** `max_num_reqs+2`）、`req_indices`/`query_pos`/`num_scheduled_tokens`/`discard_*`/`prev_positions`/`num_computed_tokens`/`positions`/`seq_lens`/M-RoPE 等 **≥10 次** |
| ⑤ `T_build` | B, G, L | **O(B·G)**（**与 T 无关**）+ 每 KV group 一次 metadata 对象构造 | `seq_lens.tolist()`、`query_start_loc_cpu[1:].tolist()`、`query_start_loc_cpu.pin_memory().to()`、`_get_block_table_and_slot_mapping`、`attn_group.layer_names` 逐层赋值（ascend `3138–3139`，**O(L)**） |
| ⑥ `T_upd` | B, S, churn | **O(B)** 常态；**O(B·S)** 最坏（condense/swap） | `InputBatch.condense()`（core `686–812`）、`swap_states()`（`569–679`）、`add_request()`（`338–484`） |
| ⑦ `T_sync` | device | 非 CPU | `num_accepted_tokens_event.synchronize()`（ascend `1104`）、`prepare_inputs_event.synchronize()`（core `3818`） |

### 1.1 "小 T 固定开销"的量化直觉

④ 的**调用次数**在小 batch 时比字节数更重要：稳态 1-token decode（T=B=1）下，
仍有 **≥10 次** Python→ATen→runtime 的 H2D 调用链 + ≥3 次 kernel launch（slot_mapping、
`query_start_loc.gpu[..].fill_(-1)`、`seq_lens[..].fill_(0)`）。若单次 launch 在 Kylin/920B 上
约 3–10 µs，则 **固定开销约 50–150 µs/step**，与 batch 无关 —— 这决定了
"极低负载下 prepare_input 的 wall 下限"。

> **[待验证]** 用 `perf stat`/`perf record` 在 T=B=1 的最简 decode 上测：
> 若 `prepare input` wall 稳定落在 50–150 µs 且与 B 无关，则该模型成立。

### 1.2 `T_build` 的"与 T 无关"特性

`AscendAttentionMetadataBuilder.build()`（`vllm_ascend/attention/attention_v1.py:291–395`）内部
**没有对 token 维度的循环**：它只做

| 行 | 操作 | 阶 |
|---|---|---|
| `299` | `query_start_loc_cpu[:num_reqs+1]` | O(1) slice |
| `301–303` | `split_decodes_and_prefills` | O(B) + 若混合批次含 `argmax().item()`（`utils.py:443`）→ **一次 `.item()`**（CPU 张量，不阻塞） |
| `327` | `attn_mask_builder.get_attention_mask` | 首次 O(seq²) 生成后**缓存复用**（`attention_mask.py:42–49`，`@singleton`），稳态 O(1)（一次 `.to(device)`） |
| `330` | `query_start_loc_cpu.pin_memory().to(device, non_blocking=True)` | **每拍新分配 pinned 内存 + H2D**，O(B) |
| `332` | `actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()` | **O(B) Python list 构造** |
| `333` | `seq_lens_list = seq_lens.tolist()` | **O(B) Python list 构造** |
| `354–366` | padding 补齐（仅 FIA 需要时） | O(B)，**`torch.cat` 分配** |
| `371` | `query_lens=query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]` | O(B) ATen |
| `376–394` | 构造 `AscendMetadata` dataclass | O(1) |

**结论**：`T_build` ∝ **B·G**，与 T 完全解耦。长 prefill（T 大、B 小）时 CPU 占比下降；
大 B decode 时 CPU 成本随 B 线性上涨，同时 device 侧 `num_tokens_padded` 也随 B 上涨，
**两者赛跑** —— 这是"什么时候 prepare_input 成为瓶颈"的核心。

### 1.3 `T_upd` 的 churn 依赖

- **稳态**（无请求进出、无抢占）：`remove_request` 不触发、`condense()` 在
  `if not (empty_req_indices := self.batch_update_builder.removed): return`（core `698–701`）
  直接返回，`refresh_metadata()` 在 `batch_update` 为空时不重建 `SamplingMetadata`（`831–832`）。
  → `T_upd ≈ O(B)` 的轻量标量更新 + `append_row`（无新块时连 `new_block_ids is None`，core `1450`）。
- **高 churn**（请求快速完成/新请求进来）：每次 `condense()` 搬运 `O(active_tokens)` 的
  `token_ids_cpu` 行（core `743–748`），最坏 O(B·S)；`swap_states` 同理（`609–617`）。
- **抢占（preemption）**：`resumed_req_ids` 走 `req_state.block_ids = new_block_ids` 重建路径
  （core `1424–1429`），外加 `add_request` 的整行 `token_ids_cpu` 写入（O(P)）。

---

## 2. 影响因素总表

影响方向标记：↑ = 成本上升，↓ = 下降，— = 无影响，∝ = 线性比例。

| # | 因素 | 影响步骤 | 方向/阶 | 数据内容是否影响控制流？ |
|---|---|---|---|---|
| 1 | **T（本拍调度 token 总数）** | S2/S4/S5/S9/S13/S19/S20/S23 | ↑ **O(T)**（numpy + ATen + H2D 字节） | 否（只影响规模） |
| 2 | **B（并发请求数）** | ①③④⑤⑥ 全线 | ↑ **O(B)**；⑤ 含 `.tolist()` 的常数因子最高 | 否（规模） |
| 3 | **ISL/P（prompt 长度）** | ⑥ `add_request` 的 `token_ids_cpu[req,:P]=prompt`；`swap_states`/`condense` 的 `max_active_token_count` | 首拍 ↑ **O(P)**；稳态每拍 ↑ 通过 `S` 影响 `condense`/`swap` 的拷贝量 | 否 |
| 4 | **OSL/S（已生成长度）** | ⑥ `condense`/`swap_states` 行拷贝；S16 `num_tokens` 取属性 | ↑ **O(B·S)**（仅 churn 时） | 否 |
| 5 | **chunked prefill** | S3 `_build_attn_state`；S16 `discard_request_mask` 必非空；S23 slot_mapping 覆盖 T | ↑ 使 T 与 B 同时高 | **是**：`enable_chunked_prefill` 直接改 `attn_state` 分支（ascend `1337–1338`） |
| 6 | **prefix caching 命中** | S1 `commit_block_table` 的 K 有效长度；`_update_states` 的 `new_block_ids` | ↓ 命中越多，新块越少、`append_row` 越轻 | **是**：命中数改变 `num_computed_tokens` 初值 → `_build_attn_state` 分支与 `is_prefilling` |
| 7 | **block_size / kernel_block_size** | S1（K 的宽度）、S23（Triton grid/constexpr） | K ↓ ⇒ H2D 字节 ↓ **O(B·K)**；`blocks_per_phys_block` 影响 logical table 宽度（ascend `83–90`） | **是**：`use_hybrid_blocks` 切换逻辑块展开路径（`118–119`, `295–310`） |
| 8 | **MTP / spec decode（n）** | S24 `_calc_spec_decode_metadata`（5×`pin_memory().to()`）、S14 异步 `_prepare_input_ids` 的 spec scatter、S17 同步、`_sanitize_placeholder...` | ↑ **O(B)**（每请求 n 个 draft）+ **固定 5 次 pin+H2D**；n≤16 受 FIA TND 限制（`attention_v1.py:239–243`） | **是**：`num_spec_tokens>0` 改变 `num_accepted_tokens_event` 存在性；`scheduled_spec_decode_tokens` 非空改变 `use_spec_decode` 分支 |
| 9 | **draft 长度（动态 spec）** | `num_spec_tokens_to_schedule`（`scheduler/output.py:260`） | ↑ 每拍 pinned 缓冲大小 O(B·n) | 是 |
| 10 | **TP / DP** | `_determine_batch_execution_and_padding` → `_sync_metadata_across_dp`（ascend `702–740`） | DP>1 时 **每拍一次 `dist.all_reduce` on CPU group**（`725`）+ `packed_tensor` 分配（O(dp_size)） | **是**：`dp_size==1` 直接短路返回（`715–716`），完全改变路径 |
| 11 | **async scheduling** | S7（`prev_positions.copy_to_gpu`）、S14 异步分支、`synchronize_input_prep`、S17 的 `prev_positions` 修正、S22 | ↑ 显著：多一次 H2D + **O(B) Python 循环** + 更强的依赖链 | **是**：`prev_sampled_token_ids is None` 切换整条 `_prepare_input_ids` 实现 |
| 12 | **NPU graph 模式** | `_pad_query_start_loc_for_fia`（ascend `834–881`）、`_build_attention_metadata` 的 padding 补齐（`2951–2968`）、`slot_mapping[num_tokens:padded].fill_(-1)` | ↑ 引入 padding：多 `torch.cat`、多 `.tolist()` 元素、多 device fill | **是**：`FULL` vs `FULL_DECODE_ONLY` vs `NONE` 改变 `num_reqs_padded` 与 `pad_attn` |
| 13 | **LoRA** | S25 `set_active_loras` → `make_lora_inputs`（core `979–1002`：`tuple(np.repeat(...))`，**O(T) tuple 构造**） + `_set_active_loras`（adapter 切换） | ↑ **O(T)**（`tuple()` 物化是 Python 层 O(T)） | **是**：`lora_config` 非空才走；`lora_id_to_request_ids` 内容影响 `should_switch` |
| 14 | **多模态（MM）** | S10 prompt embeds 逐请求 scatter；S14 的 `is_token_ids`；`_preprocess` 的 `_execute_mm_encoder`/`_gather_mm_embeddings`（**在 scope 外**） | ↑ **O(B)** Python + O(命中 embeds 的 token) 拷贝 | **是**：`req_prompt_embeds` 是否为空决定整段是否执行 |
| 15 | **prompt logprobs** | `k_sharing_fast_prefill` 断言（ascend `1911–1916`）、`num_prompt_logprobs` 字典 | — / ↓（禁用 fast prefill） | **是** |
| 16 | **logprobs / 采样参数个数** | `_make_sampling_metadata`（core `834–938`），**仅 batch_update 非空时**、`allowed_token_ids` 触发 **惰性分配 `(max_num_reqs, vocab_size)` bool**（`431–445`） | ↑ 首次分配 **O(B·V)** 字节（V=151k 时 B=256 → 38 MB）；之后 O(1) | **是**：`all_greedy`/`no_top_p`/... 等谓词（`1093–1128`）改写元数据构造分支 |
| 17 | **结构化输出（grammar）** | 不在 prepare_input（在 `sample_tokens` 的 `apply_grammar_bitmask`，core `4528`） | — | — |
| 18 | **hybrid 模型（Mamba/GDN）** | S17 同步、`mamba_utils.preprocess_mamba`（ascend `1994–2031`）、`gdn_query_start_loc`（S12） | ↑ 额外一次全量 H2D + 可能的同步 | **是**：`_has_gdn`、`mamba_cache_mode ∈ {align, all, ...}` 切换分支 |
| 19 | **KV connector（P/D 分离）** | `handle_preemptions`（ascend `1851–1856`）、`maybe_get_kv_connector_output`（**scope 外**） | ↑ O(#preemptions) | **是** |
| 20 | **KV cache block zeroing / CoW** | `_zero_block_ids(new_block_ids_to_zero)`（core `1197–1198`）、`copy_kv_cache_blocks_inplace`（`1199–1204`） | ↑ 与**新分配块数**成正比（DMA，非 CPU 计算，但占用 CPU 侧的 launch 时间） | **是**：列表为空则整段跳过 |
| 21 | **EPLB（dynamic expert 路由）** | `update_eplb_heat_collection_status`（ascend `1986`）、`eplb_updator.forward_before`/`forward_end`（`2100–2101`, `2159–2160`） | ↑ O(1)+可能的统计归约 | **是** |
| 22 | **batch reorder（混合 prefill/decode）** | `_may_reorder_batch`（core `1106–1129` → `reorder_batch_to_split_decodes_and_prefills`, `attention/backends/utils.py:665–742`） | ↑ 需重排时调用 `swap_states` × 环数；**稳态返回 False（O(B) 判定）** | **是**：`needs_swap.any()` 决定是否进入 O(B·S) 的 swap 环 |
| 23 | **DP padding** | `num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())`（ascend `2849`，**`.item()` on CPU tensor**） | ↑ 重跑 dispatch + padding | **是** |
| 24 | **`profiling_chunk_config.need_timing`** | ascend `1801–1810`：`self._sync_device()` + `time.perf_counter()` | ↑↑ 每拍一次**全局 device 同步**（仅校准窗口） | **是**：标志位来自 `scheduler_output.disable_profiling_timing` |

### 2.1 "数据内容影响控制流"清单（决定是否需要 record & replay）

以下 9 处**由数据内容（而非仅规模）决定走哪条路径**，因此无卡复刻不能只按参数矩阵合成随机数据：

| 位置 | 判定式 | 分支后果 |
|---|---|---|
| ascend `911` | `not scheduler_output.scheduled_spec_decode_tokens` | `num_valid_tokens` 是零拷贝视图 vs O(B) list comp |
| ascend `925` / `1320–1347` | `_build_attn_state` 三个 `np.all(...)` | 5 种 `AscendAttentionState` → 不同 attention kernel / mask |
| ascend `951–957` | `use_async_scheduling and prev_sampled_token_ids is not None and prev_req_id_to_index` | 是否额外 H2D |
| ascend `1103` | `num_accepted_tokens_event is not None` | 是否**阻塞同步** |
| ascend `1138–1158` | `use_async_spec_decode and valid_sampled_token_count_gpu is not None and prev_req_id_to_index` | kernel 修正 vs 直接 H2D |
| ascend `1243` | `_needs_seq_lens_cpu_sync and async_spec_decode_active` | 是否跑 CPU 端乐观值修正 |
| ascend `1259` | `len(scheduled_spec_decode_tokens) > 0` | spec 元数据（5×pin+H2D）vs 单个 device op |
| ascend `1286–1291` | `is_kv_consumer and req_id in new_schedule_reqs` **或** `num_computed_tokens_cpu >= num_prompt_tokens` | `num_decode_draft_tokens` 取值 |
| core `698` / `831` | `batch_update_builder.removed` / `batch_update` 是否非空 | `condense()` / `_make_sampling_metadata()` 是否执行 |

> **复刻策略建议**：以 **record & replay** 为主（捕获 `SchedulerOutput` 序列 + `InputBatch`
> 关键 CPU 数组快照），因为上表 9 处中至少 6 处依赖请求级 token/length 语义。
> 纯参数化合成只适合做**单因素对照实验**，不适合作为"与真机一致"的主证据。

---

## 3. prefill / decode / 混合 的复杂度对照

设稳态 decode：T = B（每请求 1 token），chunked prefill：T = B·m。

| 项 | decode (T=B) | chunked prefill (T=B·m) | 变化 |
|---|---|---|---|
| ② numpy | O(B) | O(B·m) | ↑ m 倍 |
| ③ ATen index_select | O(B) | O(B·m) | ↑ m 倍 |
| ④ H2D 字节 | O(B·K·G) + O(B) | O(B·K·G) + O(B·m) | 块表项不变，token 项 ↑ |
| ④ H2D **调用次数** | ≈10–14 次（**不变**） | ≈10–14 次 | — |
| ⑤ metadata build | O(B·G) | O(B·G) | **—（与 T 解耦）** |
| ⑥ `_update_states` | O(B) 常态 | O(B)，churn 时 O(B·S) | 取决于 churn |
| device 计算时间 | ∝ B·(KV 长度) | ∝ T·(P+T) 主导 | ↑↑ |
| **CPU/device 比值** | **危险（同阶）** | 安全（device 涨得快） | ↓ |

**推论**：

1. `prepare_input` 成为瓶颈的**首选工况**：**大 B、短序列、纯 decode、小模型、无 MTP**。
   此时 device 时间 ∝ B（每 token 的权重计算 + KV 长度很短的 attention），
   而 CPU 成本也 ∝ B，但 CPU 侧有**固定开销底噪**（④ 的 launch 次数），所以只要 B 够大、
   模型够小、on-chip 计算够快，CPU 就会露头。
2. 长 prefill 让 device 时间 ∝ T 而 CPU 侧 O(T) 项系数很小 →
   **prefill 阶段几乎不可能让 prepare_input 成为瓶颈**（除 O(B) 的 metadata build 在 B 大时）。
3. MTP / spec decode 是**双向放大**：device 侧一次 forward 产 n+1 token（device 时间摊薄），
   但 CPU 侧多 5 次 pin+H2D 与 O(B) 的 spec 循环 → **净效应对 CPU 不利**。

---

## 4. 与 device 侧的耦合：什么时候"露头"

把单步 engine-core 时间写成

```
T_step ≈ max( T_device_forward , T_cpu_prepare )   # 异步调度 + overlap 理想情况
      或 T_device_forward + T_cpu_prepare            # 串行（无 overlap）
```

（实际 vLLM 是单线程 worker：`execute_model` 发完 kernel 就返回，下一拍的 `prepare_input`
与上一拍的 device kernel **可天然重叠**；因此接近 `max()` 语义。）

`prepare_input` 成为瓶颈的判据：

```
T_prepare(B, T, flags) ≥ T_forward_device(B, T, model)
```

对**纯 decode**：

- `T_forward_device ≈ B·(2·L·H·d_model² / FLOPS) + KV-attention 项`，
  KV 项 ∝ `B·S`（短 S 时可忽略）；
- `T_prepare ≈ a·B + c`，`c` 为固定底噪，`a ≈ 1–3 µs/req`（**待实测**）。

⇒ 临界条件大致是 **`a ≥ 单请求 forward 时间`**。单请求 decode forward 时间越短
（小模型 / 小 TP 分片 / 短上下文），临界 B 越小。

### 4.1 影响 `T_forward_device` 的因素（用于找临界点）

| 因素 | 对 `T_forward_device` | 对 `T_prepare` | 净效应 |
|---|---|---|---|
| ISL/OSL ↑ | ↑（attention ∝ S） | — | 更安全 |
| B ↑ | ↑（∝ B，但 kernel 效率也 ↑） | ↑ ∝ B | **赛跑**，看谁的斜率小 |
| TP ↑ | ↓（单卡计算量 ↓，但通信开销 ↑ 且小 batch 下 kernel 变小） | **—**（TP 不改变 CPU 逻辑） | **更危险**：CPU 不变、device 变快 |
| NPU graph（FULL/FULL_DECODE_ONLY） | ↓↓（发射开销消除） | ↑（padding 引入额外 `.tolist()`/`cat`） | 双向，但 device 降得更多 → **更危险** |
| MTP（n ↑） | ↓/token（摊销） | ↑（O(B) + 5×pin） | 双向 |
| 模型变小（0.8B vs 27B） | ↓↓ | — | **更危险** |
| DP ↑ | 每卡 B 不变 | ↑（all_reduce） | 双向 |

> **给 profiling agent 的实验设计建议**：扫 (B, ISL, OSL, 模型大小, MTP, graph 模式)，
> 每格同时记录 `prepare input` phase wall（LiteProfiler patch）与 `forward` phase wall，
> 画出 **CPU/device 比值热力图**；瓶颈区即比值 → 1 的格子。

---

## 5. 同步与"掩盖"关系（profiling 必须区分）

| 现象 | 表现 | 正确的测量方式 |
|---|---|---|
| CPU 真忙（执行指令） | `perf` 采样有栈、IPC 正常、cycles 高 | `perf record` + `perf stat` 的 cycles/instructions |
| CPU 被同步阻塞 | wall 长但 `perf` 采样少、`cycles` 低（或落在 `sched_yield`/`futex`） | 对比 wall 与 on-CPU cycles；`perf sched` 看 sched_stat_blocked |
| device 慢导致 CPU 段变长 | `prepare input` wall 随 device 变慢而变长 | A/B：仅改 device 侧负载（如把 batch 减半）看 CPU 段是否跟着变 |

**已识别的会"吸收" device 时间的点**（仅 3 处）：
`num_accepted_tokens_event.synchronize()`（ascend `1104`）、
`prepare_inputs_event.synchronize()`（core `3818`）、
`self._sync_device()`（ascend `1809`，仅 timing 校准窗口）。
除此之外 `prepare_input` 内**没有** `.cpu()`/`.item()`/`synchronize()` 阻塞 device。

---

## 6. 无卡复刻的"数据依赖"判定

**结论：需要 record & replay。** 理由：

1. §2.1 列出的 9 处控制流依赖数据内容，其中 6 处依赖**请求级 token/length 语义**
   （而非仅规模参数），合成数据难以覆盖（如 `is_kv_consumer and req_id in new_schedule_reqs`）。
2. `attn_state` 的 5 分类（ascend `1320–1347`）直接决定 `_build_attention_metadata` 的
   builder 分支与 mask 选择，是 CPU 时间二阶导数的来源。
3. `_update_states` 的 `condense()`/`refresh_metadata()` 是否执行取决于
   `batch_update_builder` 的内部状态——这是**跨 step 的粘性状态**，纯静态参数无法表达。

**建议的 replay 载荷**（每 step 一条记录）：

```jsonc
{
  "step": 1234,
  "scheduler_output": { /* pickle 或结构化编码：num_scheduled_tokens、scheduled_new_reqs、
                           scheduled_cached_reqs、scheduled_spec_decode_tokens、
                           finished_req_ids、num_common_prefix_blocks、... */ },
  "input_batch_state": {  // 只需"进入 _prepare_inputs 之前"的 CPU 侧关键数组
    "req_ids": [...],
    "num_computed_tokens_cpu": [...],   // int32 list
    "num_prompt_tokens_cpu": [...],
    "num_accepted_tokens_cpu": [...],
    "token_ids_cpu_rows": [[...], ...],  // 只存 [:num_tokens_no_spec+spec] 的活跃前缀
    "block_table_rows": [[...], ...],
    "prev_req_id_to_index": {...},
    "prev_sampled_token_ids_rows": [[...], ...] | null,
    "spec_token_ids": [[...], ...]
  },
  "flags": { "use_async_scheduling": true, "num_spec_tokens": 2,
             "cudagraph_mode": "FULL_DECODE_ONLY", "has_gdn": false,
             "use_dcp": false, "lora": false, "enable_prompt_embeds": false }
}
```

**一致性判据**（建议，供一致性校验 agent 使用）：

| 指标 | 判据 |
|---|---|
| topdown 各分量（frontend/backend/retiring/bad spec） | ±2–3 pp |
| IPC | ±10% |
| 热点函数 top-20 重合度 | ≥80% |
| per-step `prepare input` wall 分布 | 同形（KS 检验 p>0.05 或分位数包络） |

---

## 7. 优化机会清单（供后续文档引用）

| # | 位置 | 问题 | 可能收益 |
|---|---|---|---|
| 1 | ascend `1039` / `1049` / `1073` | `copy_to_gpu()` **无参全量拷贝**（`max_num_reqs+2` / 全 `(2,T)`），而实际只需 `B+1` / `[:T]` | 小 B 时 DMA 字节数降 1–2 个数量级 |
| 2 | `attention_v1.py:330` | `query_start_loc_cpu.pin_memory()` **每拍新建 pinned 内存** | 改为持久 pinned 缓冲，省分配 + 页锁定 |
| 3 | `_calc_spec_decode_metadata:1412–1416` | 5 次 `torch.from_numpy(...).pin_memory().to()` **每拍新建 pinned 缓冲** | 复用预分配 pinned 缓冲 |
| 4 | `attention_v1.py:332–333` | `actual_seq_lengths_q` / `seq_lens_list` 每拍 `.tolist()` | 若 FIA 支持 tensor 形参可省；否则用预分配 list 索引赋值 |
| 5 | core `1758–1759` | `_compute_prev_positions` O(B) Python 循环 | numpy 向量化（dict→数组映射） |
| 6 | `_prepare_input_ids:1799–1826` | 异步路径 O(B) Python 循环 + 4 个 list 构造 | 向量化 / 分块 |
| 7 | core `686–812` | `condense()` Python while 循环 + O(B·S) 行拷贝 | 仅在确有 removed 时执行（已有早退）；进一步可用整批 `index_copy_` |
| 8 | `attention_v1.py:3138–3139` | 逐 layer 赋值 metadata（O(L)） | 用同层共享 dict 引用（已是引用，但循环本身 O(L)） |

