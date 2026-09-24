# prepare_input 阶段代码逻辑分析（vLLM 0.26.0 + vllm-ascend 0.26.0rc1）

> 版本口径（冻结，不可混池）：
> - vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`（0.26.0）
>   本地权威副本：`refs/vllm`（亦可用 `/home/chiro/projects/vllm/HIST_PROJECT/vllm`，`git status` 干净）
> - vllm-ascend `f2f74a16c3c50a76f4349d807918e83edec1e35c`（0.26.0rc1）
>   本地权威副本：`/home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend`
>
> 下文所有行号均指上述两个仓库的工作区文件。**core** 指 vLLM 上游，
> **ascend** 指 vllm-ascend。

---

## 0. 结论先行（TL;DR）

1. **存在两个"prepare input"边界，且互不相等**：
   - core 侧没有叫 `prepare input` 的 scope，等价物是 `"gpu_model_runner: preprocess"`
     （`refs/vllm/vllm/v1/worker/gpu_model_runner.py:4148`），覆盖
     `_update_states → _prepare_inputs → _determine_batch_execution_and_padding
     → mamba preprocess → _get_slot_mappings → _build_attention_metadata → _preprocess`（`4148–4347`）。
   - ascend 侧显式命名为 `"prepare input"`（`vllm_ascend/worker/model_runner_v1.py:1859`），
     覆盖 `_update_states → _prepare_inputs → cascade lens → batch padding 决策
     → mamba/DSA 前处理 → _pad_query_start_loc_for_fia → _build_attention_metadata
     → _sanitize_placeholder_input_ids`（`1859–2082`）。
   - **差别**：ascend 的 `prepare input` **不含** `_preprocess()`（`2084–2095`）与
     `update_cos_sin(positions)`（`2098`）；core 的 `preprocess` 二者都含。
     `synchronize_input_prep()` 在两边都**只包住** `_update_states`（ascend `1860–1883`，
     core `4150–4153` 之后的 `with` 块）。
2. **CPU 侧主成本不在 numpy 数学，而在四类操作**：
   (a) Python 层逐请求循环（`_compute_prev_positions`、`_prepare_input_ids` 异步分支、
   `discard_request_indices`、`_build_attn_state` 的 list comprehension）；
   (b) 每步新建/固定的 tensor 与 `.tolist()`（ascend `attention_v1.py:330/332/333/371`，
   `_calc_spec_decode_metadata` 的 `pin_memory().to()` `model_runner_v1.py:1412–1416`）；
   (c) 大量 H2D `copy_to_gpu()` 调用（每步 >10 次小拷贝）；
   (d) `_update_states` 的请求增删/condense/采样元数据重建。
3. **`slot_mapping` 不是 CPU 热点**：core 与 ascend 都改成 Triton kernel 下发
   （`refs/vllm/vllm/v1/worker/block_table.py:166`、
   `vllm_ascend/ops/triton/compute_slot_mapping.py:12`）。CPU 成本 ≈ kernel launch。
4. **唯一的强阻塞点是 `num_accepted_tokens_event.synchronize()`**
   （ascend `1103–1104`，core `2088–2093`），只在 hybrid/Mamba 模型（`num_accepted_tokens_event` 非 None）时生效。

---

## 1. 调用链与边界

### 1.1 引擎主循环位置

```
EngineCore.step()
  └─ Scheduler.schedule()            → SchedulerOutput
      └─ Executor.execute_model(scheduler_output)   # 跨进程 RPC（engine core → worker）
          └─ NPUModelRunner.execute_model()          # ascend:1792
              ├─[A] "prepare input" scope            # ascend:1859–2082
              │    ├─ synchronize_input_prep()       # 1860–1883
              │    ├─ _update_states()               # 1881
              │    ├─ _prepare_inputs()              # 1933
              │    ├─ _determine_batch_execution_and_padding()  # 1955
              │    ├─ _pad_query_start_loc_for_fia() # 2056
              │    └─ _build_attention_metadata()    # 2065
              ├─[B] _preprocess()                    # 2091   ← 不在 prepare input 内
              ├─[C] "forward" scope                  # 2119–2146
              ├─[D] "post process"                   # 2147
              └─ return None                         # detach 到 sample_tokens()
```

`execute_model()` 在本版本 **不再返回 `ModelRunnerOutput`**，而是把状态挂到
`self.execute_model_state = ExecuteModelState(...)`（ascend `2195–2208`）后在下一拍
`sample_tokens()`（ascend `2218`）里产出输出。因此 `prepare input` 的耗时**不在**同一拍里
被 `sample` 串行拖累，但它与上一拍的 device 执行是重叠（overlap）关系 —— 这正是
"CPU 侧负载能否被 device 掩盖"的物理基础（见 §7）。

### 1.2 精确边界（ascend，逐行核对缩进）

`with record_function_or_nullcontext("prepare input"):` 位于 `1859`，其 `with` 体结束于 `2082`
（`_sanitize_placeholder_input_ids_for_forward` 的最后一行），下一行 `2084` 缩进从 16 降回 12。

**scope 内（计入 prepare_input 计时）**

| 行号 | 内容 | 说明 |
|---|---|---|
| `1859` | `record_function_or_nullcontext("prepare input")` | 打点入口（供 torch_npu profiler / msprof 打 phase 标签） |
| `1860–1883` | `with self.synchronize_input_prep():` 包住 async-scheduling 修补 + `_update_states()` | event wait/record 只在 `use_async_scheduling` 时非空 |
| `1885–1893` | EC（encoder cache）传输分支的提前 return | 多模态 EC 场景 |
| `1895–1910` | `num_scheduled_tokens == 0` 提前 return；DP+external_launcher 时掉 `_dummy_run(1, skip_gdn_state_update=True)` | **空 batch 路径也在 scope 内** |
| `1911–1916` | `kv_sharing_fast_prefill` 断言 | O(1) |
| `1918–1927` | 取 `num_reqs`、`req_ids`、`tokens` list、`np.array` | O(B) |
| `1929–1936` | `self._prepare_inputs(...)` | **主负载**（§2） |
| `1939–1947` | `_compute_cascade_attn_prefix_lens` | 仅在 `cascade_attn_enabled and not enable_dbo` |
| `1949–1963` | `_determine_batch_execution_and_padding` | cudagraph dispatcher + DP all_reduce |
| `1975–1983` | `maybe_create_ubatch_slices` | ascend 恒返回 `(None, None)` |
| `1985–1986` | `dynamic_eplb` 热统计状态 | 仅 EPLB |
| `1988` | `pad_attn = cudagraph_mode == FULL` | O(1) |
| `1994–2031` | `mamba_utils.preprocess_mamba` + `stage_postprocess_inputs_to_gpu` | 仅 `mamba_cache_mode == "align"` |
| `2032–2043` | DSA/compress 的 `dsa_positions` 计算 | 仅 `self.use_compress` |
| `2045–2046` | `use_spec_decode`、`ubatch_slices_attn` | O(1) |
| `2048–2063` | `_pad_query_start_loc_for_fia` | 仅 FULL graph 或 SP |
| `2065–2077` | `_build_attention_metadata` | **第二主负载**（§4） |
| `2079–2082` | `_sanitize_placeholder_input_ids_for_forward` | O(T) device op |

**scope 外（不计入）**：`_preprocess()`（`2084–2095`，含 MM encoder + embedding 查表）、
`update_cos_sin(positions)`（`2098`）、`forward`（`2119–2146`）、`post process`（`2147+`）。

### 1.3 提前返回（early-return）路径归属

| 路径 | 条件 | 是否在 scope 内 | 行为 |
|---|---|---|---|
| 编码器专用 fast path | `has_ec_transfer() and not is_consumer` | ✅ 是（`1885–1893`） | 跑 MM encoder 后 `return make_empty_encoder_model_runner_output()` |
| 空 batch | `not num_scheduled_tokens` | ✅ 是（`1895–1910`） | 返回 `EMPTY_MODEL_RUNNER_OUTPUT`；DP 下先 `_dummy_run(1, skip_gdn_state_update=True)`（`1906`） |
| 空 batch 二次兜底 | `total_num_scheduled_tokens<=0 or not tokens or sum(tokens)==0` | ✅ 是（`1921–1925`） | 直接返回空输出 |
| `_update_states` 抛异常 | — | — | 不在 scope 内被吞，直接冒泡 |

> **对 profiling 的含义**：`prepare input` 的 **wall 平均值会被空 batch step 稀释**。
> 统计时必须按 `total_num_scheduled_tokens == 0 / > 0` 分层，否则 decode 空转轮会拉低占比。

### 1.4 core 侧对照（用于差异分析）

core 的 `"gpu_model_runner: preprocess"`（`4148–4347`）比 ascend 的 `prepare input` **更宽**：
额外包住了 `_get_slot_mappings`（`4311`）与 `_preprocess`（`4345`）。
若直接拿 core 的 `preprocess` 打点数据去解释 ascend 行为，会高估 ascend 的 `prepare input` 占比。

---

## 2. `_prepare_inputs` 逐子步骤（ascend 覆盖版为主）

`NPUModelRunner._prepare_inputs` = `vllm_ascend/worker/model_runner_v1.py:883–1318`
（返回三元组，比 core 多返回 `total_num_scheduled_tokens`）。
入参：`scheduler_output: SchedulerOutput`、`num_scheduled_tokens: np.ndarray[int32] (B,)`。
出参：`(logits_indices: torch.Tensor[np.int64, GPU], spec_decode_metadata: SpecDecodeMetadata|None, total_num_scheduled_tokens: int)`。

记号：**B** = `num_reqs`（本拍 batch 内请求数）；**T** = `total_num_scheduled_tokens`（本拍 token 数）；
**S** = 单请求已算 token 数（seq len）；**K** = `max_num_blocks_per_req`（= `cdiv(max_model_len, 128)`）。

### S1 `commit_block_table` — 块表 H2D（`906`）

- 输入：`input_batch.block_table`（每一 KV cache group 一张 `CpuGpuBuffer[int32]`，形状 `(max_num_reqs, K)`）。
- 输出：device 侧 `block_table.gpu[:, :B]`。
- 语义：把 CPU 权威块表整行同步到 NPU，供 Triton `compute_slot_mapping` 与 attention kernel 使用。
- 复杂度：**O(B·K)** 字节拷贝（纯 memcpy，非 CPU 计算）；ascend 实现在
  `vllm_ascend/worker/block_table.py:288–289`，`MultiGroupBlockTable` 逐 group 循环（`458–460`）。
- **设计意图**：注释 `904–905` 明写"先发起拷贝，让 H2D 与后续 CPU 计算重叠"。
- 设备交互：**H2D**（`non_blocking`）。无同步。

### S2 `req_indices` — 展开请求下标（`908`）

```python
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
```
- 语义：`[2,5,3] → [0,0,1,1,1,1,1,2,2,2]`，长度 T。
- 复杂度：**O(T)**（numpy C 层，分配新数组）。
- 无设备交互。

### S3 `num_valid_tokens` + `_build_attn_state`（`910–926`，`1320–1347`）

- 无 spec token 时 `num_valid_tokens = num_scheduled_tokens`（`911–912`，零拷贝）；
  有 spec token 时走 **Python list comprehension**（`914–921`）：`O(B)` 次 dict `.get()` + `len()`。
- `_build_attn_state`（`1320–1347`）用 `np.all(...)`（O(B) 或 O(B·S) 若走逐元素比较）判决
  `AscendAttentionState ∈ {PrefillNoCache, DecodeOnly, SpecDecoding, ChunkedPrefill, PrefillCacheHit}`。
- 这是**数据内容影响控制流**的典型点：`num_computed_tokens_cpu == 0`、`num_scheduled_tokens == 1`、
  `num_valid_tokens == 1` 三个分支决定后续 attention kernel 与 mask 形态。

### S4 `_get_cumsum_and_arange`（`929–931`，实现在 core `1720–1744`）

```python
cu_num_tokens = np.cumsum(num_tokens)                       # [2,5,3] → [2,7,10]
cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, n)  # → [0,0,2,2,2,2,2,7,7,7]
np.subtract(arange_np[:T], cumsums_offsets, out=query_pos.np[:T])
```
- 输出：`cu_num_tokens (B,)` + `query_pos.np[:T]`（每请求内 0..n-1）。
- 复杂度：**O(B) + O(T)**，3 次 numpy 调用，`out=` 复用，仅 `repeat` 分配。
- 无设备交互。

### S5 `positions_np`（`932–937`）

```python
positions_np = self._positions_np_buf[:T]
np.add(self.input_batch.num_computed_tokens_cpu[req_indices],   # O(T) gather
       self.query_pos.np[:T], out=positions_np)
```
- 语义：绝对位置 = 已算 token 数（按请求展开）+ 请求内偏移。
- 复杂度：**O(T)**，`out=` 复用 `_positions_np_buf`。

### S6 DCP 分支（`939–945` / `959–973`）

- `use_dcp` 时 `dcp_manager.init_batch_info(...)`，并可能 `generate_dcp_mtp_input(...)`。
- 仅 `decode_context_parallel_size > 1` 生效（含 spec 时更重）。

### S7 `_compute_prev_positions` — Python 逐请求循环（`947–957`，实现 core `1746–1759`）

```python
for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
    prev_positions[i] = prev_req_id_to_index.get(req_id, -1)
```
- 复杂度：**O(B) 次 Python 循环 + dict 查找**（B=256 时约 256 次解释器迭代）。
- 输出：`prev_positions.np[:B]`；`use_async_scheduling and prev_sampled_token_ids is not None` 时
  再 `copy_to_gpu(B)`（**H2D**，`956`）。
- **[待验证]**：B 很大时该循环进入 perf top 的概率 —— 用 `perf record -g` 看
  `_compute_prev_positions` 的帧占比即可证伪/证实。

### S8 `query_lens`（`975`）

`self.query_lens = torch.from_numpy(num_scheduled_tokens)` —— 零拷贝视图（共享 `num_scheduled_tokens` 内存）。
**注意**：`num_scheduled_tokens_np` 由 `execute_model:1927` 每拍新建，所以此处不产生额外拷贝。

### S9 `token_indices` + `torch.index_select`（`981–997`）

```python
token_indices = positions_np + req_indices * token_ids_cpu.shape[1]   # O(T) numpy
torch.index_select(input_batch.token_ids_cpu_tensor.flatten(), 0, token_indices_tensor,
                   out=self.input_ids.cpu[:T])
```
- 语义：从 `(max_num_reqs, max_model_len)` 的 CPU 扁平常量池里 gather 出本拍 T 个 input id。
- 复杂度：**O(T)**（`index_select` 走 ATen CPU 内核，注释明说不改用 `np.take`）。
- 该 ATen 调用是 **per-step 固定开销**（含 dispatcher / 张量包装），小 T 时相对占比高。
- 无设备交互（目标 `self.input_ids.cpu`）。

### S10 prompt embeds 逐请求 scatter（`999–1035`）

- 条件：`input_batch.req_prompt_embeds and (is_multimodal_model or enable_prompt_embeds)`。
- 复杂度：**O(B) Python 循环 + 每请求一次 `copy_`**（真实拷贝量 O(命中 embeds 的 token 数)）。
- 无设备交互（CPU 侧拼装）。

### S11 `query_start_loc` 填充 + H2D（`1037–1039`、`1064`）

```python
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1:num_reqs+1] = cu_num_tokens
self.query_start_loc.copy_to_gpu()                     # 注意：全量拷贝，未切片！
...
self.query_start_loc.gpu[num_reqs+1:].fill_(-1)        # device op，为 attention_cp 的 reshape_and_cache
```
- **注意 `1039` 是 `copy_to_gpu()` 无参调用**（core 同为无参，`2055`），即每次搬运整个
  `(max_num_reqs + 2,)` int32 缓冲，而非只拷 `num_reqs+1`。B 小时这是**固定开销**。
- 设备交互：**H2D** + 一次 device `fill_`。

### S12 GDN 专用 `query_start_loc`（`1045–1049`）

- 仅 `self._has_gdn`（`check_gdn_layer(vllm_config)`，见 `328`）：额外一组
  `gdn_query_start_loc` 的填充 + `copy_to_gpu()`（**第二次全量 H2D**）。
- 复杂度 O(B) + H2D。

### S13 `optimistic_seq_lens_cpu`（`1052–1061`）

```python
torch.add(num_computed_tokens_cpu_tensor[:B], torch.from_numpy(num_scheduled_tokens),
          out=self.optimistic_seq_lens_cpu[:B])
self.optimistic_seq_lens_cpu[B:].fill_(0)
```
- 语义：**乐观** seq len（假定上拍 draft 全接受）。它同时供给 `max_seq_len`（`2920`）、
  `discard_request_mask`（`1091`）与 `AscendCommonAttentionMetadata._seq_lens_cpu`（`3004`）。
- 复杂度 O(B) + O(max_num_reqs)（尾部清零）。

### S14 `_prepare_input_ids`（`1067`，实现在 core `1761–1890`）

两条路径：

**同步调度路径（`prev_sampled_token_ids is None`，`1778–1784`）**：只做
`input_ids.copy_to_gpu(T)` 等 H2D，**O(T) memcpy**。

**异步调度路径（`1789–1890`）**：**Python `for cur_index in range(num_reqs)` 循环**（`1799–1826`），
内部做
- `scheduled_spec_tokens.get(req_id, ())`、`cu_num_tokens[cur_index].item()`（**`.item()` 逐请求标量提取**）
- 构造 4 个 Python list（`sample_flattened_indices` / `spec_flattened_indices` /
  `prev_draft_token_indices` / `prev_indices`）
- 之后 2–3 次 `torch.tensor(list, pin_memory=True).to(device)`（**新建 pinned 缓冲 + H2D**，`1856–1880`）
- 走 `scatter_` 把上拍采样 token / draft token 写进 `input_ids.gpu`（device op）

> 复杂度：同步路径 O(T)；异步路径 **O(B) Python + O(T) list 构造 + O(#spec) H2D**。
> 这是 `prepare_input` 里最典型的 "Python 开销随 batch 线性放大" 点。
> **[待验证]** 用 `perf record -g` 对比 `scheduled_spec_decode_tokens` 空/非空两组，
> 看 `_prepare_input_ids` 帧占比差值。

### S15 M-RoPE / XD-RoPE（`1068–1083`）

- 仅 `uses_mrope`（Qwen2-VL 系）或 `uses_xdrope_dim > 0`（HunYuan-VL）时，
  额外 `_calc_mrope_positions()`（core `2725`）+ H2D 全量 `copy_`（`1073–1076`；注意是 `.gpu.copy_(.cpu)` 全量，
  不是 `[:T]` 切片）。
- 复杂度：O(B) + O(T) + H2D。

### S16 discard 统计（`1085–1099`）

```python
num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]   # O(B) Python
num_tokens_np = np.array(num_tokens, dtype=np.int32)                            # O(B) 分配
discard_requests_mask = optimistic_seq_lens_cpu[:B].numpy() < num_tokens_np      # O(B)
discard_request_indices = np.nonzero(discard_requests_mask)[0]                   # O(B)
self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)
self.discard_request_mask.copy_to_gpu(num_reqs)
```
- 两次 **H2D**。注意 `1096` 用 `self.num_discarded_requests` 作为拷贝长度，
  **长度为 0 时等价于空拷贝**（但仍有一次 Python→ATen 调用）。

### S17 `num_accepted_tokens` — **唯一强同步点**（`1101–1127`）

```python
if self.num_accepted_tokens_event is not None:
    self.num_accepted_tokens_event.synchronize()      # ← 阻塞等 device
```
- 触发条件（**三档，必须区分**）：
  1. `num_spec_tokens == 0`（未开投机解码）→ `num_accepted_tokens_event is None`
     （core `914–917`：只有 `if self.num_spec_tokens:` 才创建）→ 走 `else`（`1125–1127`，纯 CPU fill + device fill），**无同步**。
  2. 开了 spec decode 但模型**非 hybrid** → event 存在，但 `_update_states_after_model_execute`
     在 `if not self.speculative_config or not self.model_config.is_hybrid: return`（core `1556–1557`）处提前返回，
     **event 从未被 record**；`Event.synchronize()` 对未 record 的 event 是空操作，**实测不阻塞**。
  3. spec decode **且 hybrid**（MTP/EAGLE + 线性注意力/Mamba）→ `sample_tokens` 里
     core `4535` → `_update_states_after_model_execute` → `record()`（`1584` / `1590`），
     下一拍 `_prepare_inputs` 在这里**真正阻塞**。
- core 版本更保守：多一个 `not (use_async_scheduling and mamba_cache_mode != "align")` 条件（core `2088–2090`）。
- **这是 prepare_input 内唯一会被 device 时间"反向掩盖"的点**：device 越慢，这里等得越久，
  会把 device 的延迟错误地算进 CPU 阶段。

### S18 `num_computed_tokens` 上屏（`1129–1158`）

- 非异步：`self.num_computed_tokens[:B].copy_(num_computed_tokens_cpu_tensor[:B], non_blocking=True)`（**H2D**）。
- 异步 spec decode：改调 `update_num_computed_tokens_for_batch_change(...)` kernel（`1146–1153`），
  额外一次 `computed_token_tensor_cpu = ...to(device, non_blocking=True)`（`1135–1137`，
  **新建 H2D 张量**，非复用缓冲）。

### S19 `req_indices` / `query_pos` / `num_scheduled_tokens` 上屏（`1160–1167`）

连续 4 次 `copy_to_gpu`（**H2D ×4**）：
`req_indices[:T]`、`query_pos[:T]`、`num_scheduled_tokens[:B]`。
每次都是一次独立的 Python→ATen→DMA 链，**小 batch 时 launch 开销占主导**。

### S20 DCP async rebuild / `positions`（`1169–1225`）

- 无 DCP：`dcp_manager is None` → `cp_async_rebuild = DCPAsyncSpecDecodeRebuildResult(rebuilt=False, ...)`（`1203–1207`，纯构造）。
- 随后 `self.positions[:T] = num_computed_tokens[req_indices_gpu] + query_pos.gpu[:T]`（`1222–1225`）：
  3 个 device op（gather + add + slice-assign）。

### S21 `seq_lens`（`1227–1230`）

`self.seq_lens[:B] = num_computed_tokens[:B] + num_scheduled_tokens_gpu`；`seq_lens[B:].fill_(0)`。
2 个 device op，**无 D2H**（这是 ascend 的关键优化：CPU 副本用 `optimistic_seq_lens_cpu` 顶掉）。

### S22 `_correct_optimistic_seq_lens_cpu`（`1243–1244`，实现 `1432–1459`）

- 仅 `use_async_spec_decode and _needs_seq_lens_cpu_sync`。用 CPU 侧 valid-count 修正乐观值，
  **刻意避免一次 D2H + synchronize**（注释 `1232–1241`）。

### S23 `compute_slot_mapping` — Triton kernel（`1246–1250`）

- 入参：`num_reqs`、`query_start_loc.gpu[:B+1]`、`positions[:T]`。
- 实现：`vllm_ascend/worker/block_table.py:150–189`，
  grid = `(num_reqs + 1,)`，最后一 program 专门写 padding（`TILE_BLOCK_SIZE=1024`）。
- 复杂度：**CPU 侧 O(1) + kernel launch**（Triton JIT 缓存命中后为纯 launch）；
  真正的 O(T) 计算在 NPU 上。
- **[待验证]** 首次运行有 Triton JIT 编译（秒级），属 warmup 阶段，稳态应忽略；
  用 `perf` 里是否出现 `triton` 编译相关帧可证伪。

### S24 spec decode 元数据（`1259–1303`）

- `use_spec_decode = len(scheduled_spec_decode_tokens) > 0`。
- 非 spec：`logits_indices = query_start_loc.gpu[1:B+1] - 1`（1 个 device op），**非常廉价**。
- spec：`num_draft_tokens`/`num_decode_draft_tokens` numpy 填充 + 逐 req dict 循环（`1279–1291`）
  + `_calc_spec_decode_metadata`（`1293`，实现 `1366–1430`）。
  `_calc_spec_decode_metadata` 内有 **5 次 `torch.from_numpy(...).pin_memory().to(device)`**
  （`1412–1416`）→ 每步新建 pinned 缓冲 + 独立 H2D，**是 spec decode 下最贵的 CPU 段之一**；
  另有 `self.input_ids.gpu[logits_indices]` 的 device gather（`1420–1421`）。

### S25 LoRA / lmhead-TP 尾巴（`1306–1312`）

- `lora_config` 非空：`set_active_loras(...)`（含 `np.sum` + 可能的 adapter 切换，见 §6）。
- `lmhead_tp_enable()`：对 `logits_indices` 做 `F.pad`（`1312`）—— 每次分配新张量。

### S26 返回（`1314–1318`）

---

## 3. 数据结构与 IO 语义

### 3.1 上游契约：`SchedulerOutput`

定义于 `refs/vllm/vllm/v1/core/sched/output.py:190–274`。**生产方**：`Scheduler`（engine core 进程）；
**消费方**：`NPUModelRunner.execute_model`（worker 进程）。因此它是**跨进程零拷贝不了的 pickle 消息**，
字段大小直接影响 RPC 与反序列化成本。

| 字段 | 行号 | 类型/形状 | 单元语义 | 主要消费者 |
|---|---|---|---|---|
| `scheduled_new_reqs` | `195` | `list[NewRequestData]` | 首拍调度的请求全集（含 prompt_token_ids、block_ids、sampling_params） | `_update_states:1243` |
| `scheduled_cached_reqs` | `199` | `CachedRequestData` | 已在 worker 缓存的请求的**增量 diff** | `_update_states:1332` |
| `num_scheduled_tokens` | `203` | `dict[str,int]` | req→本拍调度 token 数 | `_prepare_inputs`、cascade、`_get_encoder_seq_lens` |
| `total_num_scheduled_tokens` | `206` | `int` | `sum(...)`，即 T | 决定 T 维度缓冲切片 |
| `scheduled_spec_decode_tokens` | `210` | `dict[str,list[int]]` | req→本拍投喂的 draft token ids | `_prepare_inputs:1259+`、`_prepare_input_ids` |
| `scheduled_encoder_inputs` | `214` | `dict[str,list[int]]` | MM 编码器待处理的输入下标 | `_preprocess`→`_execute_mm_encoder` |
| `num_common_prefix_blocks` | `217` | `list[int]`（每 KV group） | 公共前缀块数，用于 cascade attention | `_compute_cascade_attn_prefix_lens` |
| `finished_req_ids` | `222` | `set[str]` | 上拍以来结束的请求 | `_update_states:1180/1192` |
| `free_encoder_mm_hashes` | `225` | `list[str]` | 释放 encoder 缓存 | `_update_states:1207` |
| `new_block_ids_to_zero` | `253` | `list[int]` | 新分配块需清零 | `_update_states:1197`→`_zero_block_ids` |
| `kv_cache_block_copies` | `256` | `list[KVCacheBlockCopy]` | CoW 块拷贝 | `_update_states:1199`→device copy |
| `preempted_req_ids` | `231` | `set[str]` | 抢占（仅 v2 runner） | — |
| `num_spec_tokens_to_schedule` | `260` | `int` | 动态 spec decode 的 K | scheduler 侧 |

`CachedRequestData`（`113–179`）关键字段：`req_ids`、`resumed_req_ids`（抢占恢复集合，**按集合成员决定追加还是替换块表**，`_update_states:1419–1429`）、`new_token_ids`（**仅 PP 使用**，`120–122` 注释）、`all_token_ids`（async scheduling 恢复用）、`new_block_ids`、`num_computed_tokens`、`num_output_tokens`。

### 3.2 粘性状态：`InputBatch` / `NPUInputBatch`

`InputBatch`（core `vllm/v1/worker/gpu_input_batch.py:92`）是 worker 侧的**持久批状态**；
`NPUInputBatch`（`vllm_ascend/worker/npu_input_batch.py:33`）继承并替换块表为 ascend 版。

关键缓冲（都在 CPU，多数为 pinned 或 `with_numpy` 直通 numpy）：

| 属性 | 形状 / dtype | 语义 | 消费点 |
|---|---|---|---|
| `token_ids_cpu_tensor` / `token_ids_cpu` | `(max_num_reqs, max_model_len)` int32，**非 pinned** | 每请求完整 token 序列（含输出） | `_prepare_inputs` 的 `index_select` |
| `is_token_ids_tensor` / `is_token_ids` | 同上 bool | 该位置是 token id 还是 prompt embeds | `enable_prompt_embeds` 分支 |
| `num_computed_tokens_cpu` | `(max_num_reqs,)` int32 | 已算 token 数 | positions/seq_lens/`_build_attn_state` |
| `num_prompt_tokens_cpu` | `(max_num_reqs,)` int32 | prompt 长度 | `is_prefilling`、chunked 判定 |
| `num_tokens_no_spec` | `(max_num_reqs,)` int32 | 不含 spec 的 token 数 | `_update_states`、condense |
| `num_accepted_tokens_cpu` | `(max_num_reqs,)` int32 | 上拍接受数 | `num_accepted_tokens` 同步链 |
| `temperature_cpu` / `top_p_cpu` / `top_k_cpu` / penalties | `(max_num_reqs,)` | 采样参数 | `_make_sampling_metadata` |
| `request_lora_mapping` | `(max_num_reqs,)` int32 | req→lora_id | `set_active_loras` |
| `prev_req_id_to_index` / `prev_sampled_token_ids` | dict / tensor | 异步调度用上一拍索引与采样结果 | `_compute_prev_positions`、`_prepare_input_ids` |
| `spec_token_ids` | `list[list[int]]` | 每请求 spec 占位 | `_get_active_token_count`、condense |

`CachedRequestState`（core `35–89`）是**每请求**状态：`prompt_token_ids`、`output_token_ids`（Python list）、
`block_ids: tuple[list[int],...]`（每 KV group 一个 list）、`num_computed_tokens`、
`mm_features`、`lora_request`、`prev_num_draft_len`（异步 spec）等。
`num_tokens` 是 property（`75–77`）：`num_prompt_tokens + len(output_token_ids)`。

> **重要**：`_update_states` 里 `req_state.output_token_ids.extend(...)`（`1361/1401–1405`）
> 与 `num_tokens` 的重复计算（`_prepare_inputs:1087` 每拍遍历所有 req 取其 `num_tokens`）
> 都是 O(B) 的 Python 属性访问 —— 属于"看起来无害、实测可能进 top"的候选热点。

### 3.3 `BlockTable` / `MultiGroupBlockTable`

| 项 | core | ascend |
|---|---|---|
| 文件 | `refs/vllm/vllm/v1/worker/block_table.py` | `vllm_ascend/worker/block_table.py` |
| 存储 | `CpuGpuBuffer[int32] (max_num_reqs, max_num_blocks_per_req)`（core `81–83`），`slot_mapping` 缓冲见 core `86–90` | 同（ascend `97`），但 DCP 时可 `duplicate_size = 1 + num_speculative_tokens`（ascend `94–97`）；`slot_mapping` 缓冲预留 MTP 额外槽位（`99–105`） |
| `append_row` | numpy 切片写（core `114–130`） | 同（ascend `110–125`），但支持 hybrid 逻辑块展开（ascend `118–119`, `295–310`） |
| `compute_slot_mapping` | **Triton kernel**（`166–182`） | **Triton kernel**（`179–189`），另有 DCP/纯 numpy 分支（`241–286`） |
| `kernel_block_size` | 支持 `kernel_block_size != block_size`（block splitting） | 用 `kernel_sizes` 列表推导 `blocks_per_phys_block`（`61–88`） |

**`slot_mapping` 语义**：`token_idx → KV cache 物理槽位`。Triton kernel 里
`block_indices = pos // block_size; slot_offsets = pos - block_indices*block_size;
slot = block_table[req, block_indices]*block_size + slot_offsets`（`compute_slot_mapping.py:52–84`），
padding 槽写 `PAD_SLOT_ID = -1`。

### 3.4 `query_start_loc` / `cu_seqlens` / `seq_lens` / `positions`

| 张量 | 形状 | 语义 | 生产 | 消费 |
|---|---|---|---|---|
| `query_start_loc` | CPU/GPU `(max_num_reqs+2,)` int32 | 前 B+1 项为 `[0, c1, ..., cB]`（累积查询长度），尾部填 `cu_num_tokens[-1]` 或 `-1` | `_prepare_inputs:1037–1039`、`_pad_query_start_loc_for_fia:834–881` | **从不出现在 FIA**、但等价于 FIA 的 `actual_seq_lengths_q` 前缀和；GDN 用单列版 |
| `gdn_query_start_loc` | 同上 | GDN 的**未 padding** 版本（注释 `1041–1044`：FIA 的 check 需要） | `1045–1049` | GDN metadata builder（`3164–3166`） |
| `seq_lens` | GPU `(max_num_reqs,)` int32 | 每请求总长度（已算+本拍） | `_prepare_inputs:1227–1230` | attention `actual_seq_lengths_kv` |
| `optimistic_seq_lens_cpu` | CPU `(max_num_reqs,)` int32 | `seq_lens` 的 CPU 镜像（乐观） | `_prepare_inputs:1056–1061` | `max_seq_len`（`2920`）、`discard mask`（`1091`）、`_seq_lens_cpu`（`3004–3005`） |
| `positions` | GPU `(max_num_tokens,)` int64 | 每 token 绝对位置 | `_prepare_inputs:1222–1225` | attention / RoPE / `compute_slot_mapping` |
| `num_scheduled_tokens` | CPU/GPU `(max_num_reqs,)` int32 | 本拍每请求调度数 | `_prepare_inputs:1165–1167` | GDN builder、`num_accepted_tokens` 修正 |
| `query_pos` | CPU/GPU `(max_num_tokens,)` int32 | 请求内偏移 0..n-1 | `_get_cumsum_and_arange` | positions、`_prepare_input_ids` |
| `logits_indices` | GPU `(≤ num_sampled,)` int64 | 需要出 logits 的 token 在 hidden_states 里的下标 | `_prepare_inputs:1269` 或 `spec_decode_metadata.logits_indices` | `forward` 后 `hidden_states[logits_indices]`（`2171`） |

### 3.5 `slot_mapping` 的两种消费方

1. attention kernel 的 `reshape_and_cache`（写 KV）；
2. `_build_attention_metadata:2963` 取 `blk_table.slot_mapping.gpu[:num_tokens_padded]`，
   并在 `2967` 对 padding 区 `fill_(-1)`。**注意这是 device 侧写 —— 每拍一次**。

### 3.6 attention metadata

- 中间态：`AscendCommonAttentionMetadata`（`vllm_ascend/attention/utils.py:208`），继承 core 的
  `CommonAttentionMetadata`（`refs/vllm/vllm/v1/attention/backend.py:412`）。
- 终态：`AscendMetadata`（`vllm_ascend/attention/attention_v1.py:150–201`），关键字段：
  `attn_mask`、`attn_state`、`num_actual_tokens`、`num_decode_tokens`、`num_prefills`、`num_decodes`、
  `seq_lens`、`seq_lens_cpu`、`seq_lens_list`、`actual_seq_lengths_q`、`query_start_loc`、
  `max_query_len`、`block_tables`、`slot_mapping`、`causal`、`model_runner_type`、`reshape_cache_event`。
- 每层共享：`_build_attn_group_metadata:3138–3139` 把同一个 `attn_metadata_i` 赋给该 group 内所有 layer name。

### 3.7 采样与 LoRA

- `sampling_metadata`（core `vllm/v1/sample/metadata.py`）由
  `InputBatch._make_sampling_metadata()`（`834–938`）构造，**只在 `batch_update` 非空时重建**
  （`refresh_metadata:831–832`），因此**稳态 decode 不重建** —— 这是"数据内容影响控制流"的关键优化。
- `lora_mapping`：`request_lora_mapping (max_num_reqs,)` + `lora_id_to_request_ids: dict[int,set]`；
  `set_active_loras` 在 `_prepare_inputs:1307–1309` 调用，只在 `lora_config` 存在时。

---

## 4. Ascend 特有分支详解

### 4.1 `NPUModelRunner` 覆盖清单（与 prepare_input 相关部分）

| 方法 | ascend 行号 | 为什么需要 |
|---|---|---|
| `_init_device_properties` | `608–609` | 无 `num_sms` 概念，置 None |
| `_sync_device` | `611–612` | `torch.npu.synchronize()` |
| `_update_states` | `816–832` | 1) KV-load 失败重算的 rewind guard；2) `_apply_pp_sampled_tokens_from_scheduler_output` |
| `_pad_query_start_loc_for_fia` | `834–881` | FIA TND layout 要求 `hidden_states.shape[0] == actual_seq_lengths_q[-1]`（注释 `844–845`） |
| `_prepare_inputs` | `883–1318` | DCP / MTP / 图模式 / `_build_attn_state` |
| `_build_attn_state` | `1320–1347` | 用 CPU seq 状态判决 5 种 `AscendAttentionState` |
| `_calc_spec_decode_metadata` | `1366–1430` | NPU 侧 `pin_memory().to()` 直传 |
| `_correct_optimistic_seq_lens_cpu` | `1432–1459` | 异步 spec decode 免 D2H |
| `_build_attention_metadata` | `2875–3224` | 构造 `AscendCommonAttentionMetadata` + 逐 KV group/attn group build |
| `_dummy_run` | `3241–3556` | 自己实现，`skip_gdn_state_update` 参数 |
| `might` `_sanitize_placeholder_input_ids_for_forward` | `1349–1365` | 清掉 spec 调度占位符 |

### 4.2 为什么 `_build_attention_metadata` 落在 prepare_input 内（ascend）

core 把它也放在 `"preprocess"` 里（`4322`），ascend 保持同层但**换了一个 scope 名**并**排除了 `_preprocess`**。
机制上的原因是：`AscendMetadata` 的构造**必须发生在 CPU 上**（`seq_lens_list`、`actual_seq_lengths_q`
是 **Python list**，FIA 算子直接吃 list 形参），因此它天然属于 CPU 侧阶段。

### 4.3 `synchronize_input_prep` 的作用

定义在 core `3809–3822`，ascend **未覆盖**：

```python
@contextmanager
def synchronize_input_prep(self):
    if self.prepare_inputs_event is None:
        yield; return
    self.prepare_inputs_event.synchronize()   # 等上拍 H2D 读完 CPU pinned 缓冲
    try:
        yield
    finally:
        self.prepare_inputs_event.record()    # 记录"我写完 CPU 缓冲了"
```

- `prepare_inputs_event` 只在 `use_async_scheduling` 时创建（core `737–741`），
  且是 **blocking=True**（`741`，注释 `739–740`：避免 busy-poll CUDA/NPU driver lock 导致 TP 下 rank 掉队）。
- **ascend 的 record 点在 `1883` 之后**（即 `_update_states` 结束处），**不包含** `_prepare_inputs`。
  也就是说，下一拍等待的是"上拍的 `_update_states` 期间的 H2D 已完成"，而不是 `_prepare_inputs` 期间的。
  **[待验证]** 这是否为上游有意为之（因为 `_prepare_inputs` 复用同一批 pinned 缓冲：
  `query_start_loc`、`req_indices`、`query_pos`、`num_scheduled_tokens` 等）。
  验证方法：在 `_prepare_inputs` 末尾也插一个 event 做对照实验，测 tail latency 与 TP 下 straggler 率。

### 4.4 图模式（ACLGraph）对 prepare_input 的影响

- `FULL_DECODE_ONLY` / `FULL` 会把 `num_tokens_padded > num_tokens_unpadded`：
  触发 `_pad_query_start_loc_for_fia`（`2048–2063`）与
  `_build_attention_metadata` 里的 padding 补齐（`2951–2968`, `3197–3223`）。
- `builder.build_for_cudagraph_capture` vs `builder.build`（`3107–3119`）走不同分支，
  capture 期会跳过部分 DSA 结构（`3095–3098`）。
- `pad_attn = cudagraph_mode == CUDAGraphMode.FULL`（`1988`）决定 slot_mapping 用 padded 还是 unpadded 维度。

### 4.5 NPU graph 下的 `_dummy_run`

- 空 batch（`1906`）与 DP 空转都走 `_dummy_run`；它有**自己的** `synchronize_input_prep()`（`3359`），
  注释 `3354–3358` 明说它共享同一批 pinned CPU 缓冲，必须参与同一 event 协议。
- `skip_gdn_state_update=True` 时 `gdn_query_start_loc.np.fill(0)`（`3395–3397`），
  并由 `_should_build_dummy_attn_metadata`（`3226–3238`）决定是否建 dummy attn metadata。

---

## 5. prefill vs decode 差异

| 子步骤 | 纯 decode（每请求 1 token） | 纯 prefill（长 chunk） | chunked prefill（混合） |
|---|---|---|---|
| `commit_block_table` | O(B·K) H2D，B 大 | O(B·K) H2D | 同 |
| `req_indices`/`cumsum`/`positions` | O(T)=O(B) | O(T)（T 可达 8k） | O(T) |
| `index_select` input_ids | O(B) | **O(T)，主导** | O(T) |
| `_compute_prev_positions` | O(B) Python | O(B) Python | O(B) Python |
| `_prepare_input_ids` | 同步路径 O(B)；异步路径 O(B) Python + 多次 H2D | 同步路径 O(T) memcpy | 视调度模式 |
| `_build_attn_state` | `np.all(num_scheduled==1)` → DecodeOnly | `np.all(num_computed==0)` → PrefillNoCache | `np.all(num_valid==1)`/chunked |
| `_build_attention_metadata` | `seq_lens.tolist()` O(B)；`actual_seq_lengths_q.tolist()` O(B)；mask 复用 | **同样 O(B) 的 list 转换，与 T 无关**；但 `block_table` 行更长 | 同 prefill |
| `compute_slot_mapping` | kernel launch，正比 B | kernel launch，正比 T | 同 |
| `spec metadata` | 若开 MTP：`_calc_spec_decode_metadata` 的 5 次 pin+H2D | 通常无 | 部分请求有 |
| `logits_indices` 生成 | `query_start_loc[1:]-1`（device op） | 同 | 同 |

**要点**：

1. `_build_attention_metadata` 的 CPU 成本主要随 **B**（`.tolist()`、list 构造）而非 T 增长。
   因此**长 prefill 时 CPU 段相对 device 段反而变便宜**（device 时间 ∝ T）。
2. **纯 decode + 大 B** 是 `prepare_input` 占比最危险的区间：CPU 成本 ∝ B，而 device 时间（decode attention + MLP）
   在 MTP / 小模型 / 短序列下也 ∝ B，二者同阶，CPU 极易露头。
3. chunked prefill 会同时抬高 T 与 B，且 `discard_request_mask` 必然非空（`_prepare_inputs:1091`），
   使 S16 的两次 H2D 不再是零长度。

---

## 6. 同步点全清单

| # | 位置 | 代码 | 触发条件 | 是否会掩盖 device 时间 |
|---|---|---|---|---|
| 1 | ascend `1104` / core `2093` | `self.num_accepted_tokens_event.synchronize()` | `num_spec_tokens > 0`（创建）+ **spec 且 hybrid**（才会被 record，见 §2/S17） | **会**。等的是上拍 `sample_tokens` 中 record 的 device 完成；spec+非 hybrid 时是空操作 |
| 2 | core `3818` | `self.prepare_inputs_event.synchronize()` | `use_async_scheduling` | **会**（设计如此，摊销异步 H2D） |
| 3 | ascend `1809` | `self._sync_device()`（`torch.npu.synchronize()`） | `profiling_chunk_config.need_timing` 为真 | **会**，且是**全局同步**，只在 timing 校准窗口内 |
| 4 | ascend `611–612` | `_sync_device` 定义 | 被 `1809` 等调用 | 同上 |
| 5 | ascend `2930/2931/2942` | `_get_dcp_metadata` 里的 `.numpy()` / DCP 生成 | `use_dcp` | 不直接同步，但 `.numpy()` 要求 CPU 数据 |
| 6 | core `1809` | `cu_num_tokens[cur_index].item()` | 异步 `_prepare_input_ids` | 不阻塞（CPU 数组） |
| 7 | ascend `333/383` | `seq_lens.tolist()`、`actual_seq_lengths_q` | 每拍 attention build | **不阻塞**（数据已在 CPU），是纯 CPU 转换开销 |
| 8 | ascend `2964` | `blk_table.get_device_tensor()` | 每 KV group | 不阻塞 |

**全局搜索结论**：`_prepare_inputs`（ascend `883–1318`）内部**没有** `.item()`/`.cpu()`/`synchronize()`；
唯一的同步点是 S17（`1103–1104`）。`_build_attention_metadata` 内部同样没有强制同步
（它刻意用 CPU 侧 `optimistic_seq_lens_cpu` 规避 D2H，见注释 `2990–3005` 与 `1232–1241`）。

> **profiling 含义**：如果用 wall-clock 打点测 `prepare input`，**遇到 hybrid 模型会把 device 时间算进来**。
> 必须同时看 `perf`（纯 CPU 采样，不感知阻塞）与 phase wall 两者才能分离。

---

## 7. 完整调用图

```
NPUModelRunner.execute_model(scheduler_output)                      # ascend:1792
│
├─ profiling_chunk_config 校准窗口                                    # 1801–1810  ← _sync_device
├─ ngram_gpu deepcopy / async-scheduling deepcopy                    # 1814–1849
├─ kv_transfer handle_preemptions                                    # 1851–1856
│
└─ record_function("prepare input")                                  # 1859  ┐
   │                                                                        │
   ├─ with synchronize_input_prep():                                 # 1860  │
   │  ├─ async-spec prev_req_id 修补（O(B) Python）                  # 1867  │
   │  └─ _update_states(scheduler_output)                            # 1881  │
   │     ├─ [core 1169] finished_req_ids → requests.pop / remove_request     │
   │     ├─ _zero_block_ids(new_block_ids_to_zero)                   # 1197  │
   │     ├─ copy_kv_cache_blocks_inplace(kv_cache_block_copies)      # 1199  │
   │     ├─ 逐 new_req 构造 CachedRequestState                        # 1243  │
   │     ├─ 逐 cached_req 更新 num_computed/block_ids/output_token_ids# 1332 │
   │     ├─ input_batch.add_request(...)                             # 1489  │
   │     ├─ input_batch.condense()                                   # 1493  │
   │     ├─ _may_reorder_batch()                                     # 1495  │
   │     └─ input_batch.refresh_metadata()          # 1497 → 条件性 _make_sampling_metadata
   │                                                                        │
   ├─ [early return: EC transfer]                                    # 1885  │
   ├─ [early return: 空 batch ± _dummy_run(1)]                       # 1895  │
   ├─ tokens list + np.array                                         # 1918  │
   │                                                                        │
   ├─ _prepare_inputs(...)                                           # 1933  │
   │  └─ 见 §2 的 S1..S26                                            # 883   │
   │                                                                        │
   ├─ _compute_cascade_attn_prefix_lens                              # 1939  │
   ├─ _determine_batch_execution_and_padding                         # 1949  │
   │  └─ cudagraph_dispatcher.dispatch + (DP) _sync_metadata_across_dp      │
   ├─ maybe_create_ubatch_slices → (None,None)                       # 1977  │
   ├─ mamba_utils.preprocess_mamba (+ stage_postprocess_inputs)      # 1994  │
   ├─ DSA dsa_positions (use_compress)                               # 2032  │
   ├─ _pad_query_start_loc_for_fia                                   # 2056  │
   ├─ _build_attention_metadata                                      # 2065  │
   │  ├─ AscendCommonAttentionMetadata(...)                          # 2996  │
   │  └─ per kv_cache_gid / attn_gid: _build_attn_group_metadata     # 3045  │
   │     └─ AscendAttentionMetadataBuilder.build(...)                # attention_v1.py:291
   │        ├─ split_decodes_and_prefills                            # utils.py:360
   │        ├─ attn_mask_builder.get_attention_mask  (singleton 缓存) # :327 │
   │        ├─ query_start_loc_cpu.pin_memory().to(device)  ← 每拍新建 pinned │
   │        ├─ actual_seq_lengths_q = qsl[1:].tolist()               # :332 │
   │        ├─ seq_lens_list = seq_lens.tolist()                     # :333 │
   │        └─ metadata_cls(...)                                     # :376 │
   └─ _sanitize_placeholder_input_ids_for_forward                    # 2079  ┘
   
   ├─ _preprocess(...)                                               # 2091  ← 不在 scope
   ├─ update_cos_sin(positions)                                      # 2098  ← 不在 scope
   └─ record_function("forward") + _model_forward                    # 2119–2146
```

---

## 8. `_update_states` 成本模型（prepare_input 的另一半）

core 实现 `1169–1543`。主循环三块：

1. **删除阶段**（`1179–1230`）：`finished_req_ids` 两轮遍历（`1180`、`1192`）+ `unscheduled_req_ids` 集合差
   （`1224`）+ 逐 req `remove_request`（`1229–1230`）。复杂度 **O(#finished + #unscheduled)**，
   但 `remove_request` 内部会更新 `batch_update_builder.removed`，
   为后续 `condense()` 提供降序空位表。
2. **新增/更新阶段**（`1243–1484`）：
   - 新请求：构造 `CachedRequestState`、`torch.Generator(device).manual_seed()`（**仅 RANDOM_SEED 采样**，`1258–1259`）、
     `_init_mrope_positions`（仅 M-RoPE）；
   - 缓存请求：逐 req 更新 `num_computed_tokens`、块表 `append_row`（`1451`）、
     `output_token_ids.extend/delete`、`update_req_spec_token_ids`（`1479`）。
   - **async spec decode 下**：`req_state.output_token_ids.extend([-1]*k)`（`1361`）先**乐观膨胀**，
     并挂一个 `deferred_spec_decode_corrections` 闭包（`1363–1365`，`1511–1541`）留到 forward 之后再修正。
     这是"数据内容影响控制流"的另一处：只有 `prev_num_draft_len > 0` 的请求才走这条。
3. **收尾阶段**（`1486–1509`）：
   - `input_batch.add_request(request)`（`1489`）→ core `338–484`：写 `token_ids_cpu` 切片、
     写全部采样参数标量、可能**惰性分配 `allowed_token_ids_mask_cpu_tensor (max_num_reqs, vocab_size) bool`**
     （core `431–450`，vocab 151k 时 = `max_num_reqs × 151k` 字节，**一次大分配**）。
   - `input_batch.condense()`（`1493`）→ core `686–812`：**Python while 循环**做空位下沉，
     每次搬运 `token_ids_cpu[empty,:num_tokens] = token_ids_cpu[last,:num_tokens]`
     （**O(active_tokens) 内存拷贝，最坏 O(B·S)**）+ `block_table.move_row` +
     ~10 个标量数组搬运 + `batch_update_builder.moved.append(...)`。
     **这是 `_update_states` 里最重的、且强烈依赖请求生命周期模式的一段**。
   - `_may_reorder_batch()`（`1495`）→ core `1106–1129`：`reorder_batch_to_split_decodes_and_prefills`，
     cost ∝ B（ascend 的 builder `reorder_batch` 恒返回 False，见 `attention_v1.py:261–262`，
     但 `reorder_batch_threshold=decode_threshold` 已设（`:245`），由通用函数按 query len 重排）。
   - `refresh_metadata()`（`1497`）→ **仅 `batch_update` 非空时才重建 `SamplingMetadata`**（`831–832`）。

> **预测**：稳态 decode（无请求进出、无抢占）时 `_update_states` ≈ O(B) 的轻量更新；
> **请求快速进出 / 抢占**时 `condense()` 的 O(B·S) 拷贝会明显放大。
> 这个假设必须在 profiling 中用"稳态 vs 高波动"两组 workload 验证。

---

## 9. 与前后阶段的接口契约（对接模块）

| 方向 | 接口 | 契约 |
|---|---|---|
| 上游 | `Scheduler.schedule() → SchedulerOutput` | 跨进程 pickle；worker 侧对其**只读**（除 ngram_gpu/async 需 `deepcopy`/`replace`，见 `1814–1849`） |
| 上游 | `Executor.execute_model` RPC | worker 是**单线程事件循环**，prepare_input 阻塞即整体阻塞 |
| 下游 | `set_ascend_forward_context(...)`（`2121`） | 消费 `attn_metadata` + `num_tokens_padded` + `num_tokens_across_dp` + `cudagraph_mode` + `batch_desc` |
| 下游 | `_model_forward`（`2144`） | 消费 `input_ids` / `positions` / `inputs_embeds` / `**model_kwargs` |
| 下游 | `compute_logits` + `hidden_states[logits_indices]`（`2171–2172`） | 消费 `logits_indices` |
| 下游 | `sample_tokens()`（`2218`） | 消费 `ExecuteModelState`（含 `spec_decode_metadata`、`attn_metadata`、`batch_desc`） |
| 旁路 | `dynamic_eplb` / `eplb_updator.forward_before()`（`2100–2101`） | 依赖 `update_eplb_heat_collection_status` 在 `1986` 设的状态 |
| 旁路 | KV connector（`kv_connector_metadata`、`maybe_get_kv_connector_output`） | `handle_preemptions` 必须在 `_update_states` **之前**（注释 `1854–1855`） |

---

## 10. 不确定点（`[待验证]`）汇总

| # | 待验证内容 | 验证方法 |
|---|---|---|
| 1 | `_compute_prev_positions` 的 O(B) Python 循环是否进 perf top-N | `perf record -g` 采样 30s，看 `_compute_prev_positions` 帧占比；B=1 与 B=256 对比 |
| 2 | `query_start_loc.copy_to_gpu()` 全量拷贝（`1039`）的固定开销 | 微基准：改 `copy_to_gpu(num_reqs+1)` 做 A/B，测 prepare_input delta |
| 3 | `_calc_spec_decode_metadata` 的 5 次 `pin_memory().to()`（`1412–1416`）贡献 | MTP on/off 两组 `perf` 对比 + 火焰图里 `at::native::copy_`/`cudaMemcpy` 帧 |
| 4 | `synchronize_input_prep` 的 record 点（`1883`）是否覆盖不足 | 在 `_prepare_inputs` 末尾追加 event 做对照实验 |
| 5 | `condense()` 在请求高波动下的真实成本 | 三类 workload（稳态 / 高 churn / 频繁抢占）对比 `_update_states` phase 时长 |
| 6 | Triton kernel 首次 JIT 编译是否污染 warmup | 检查 `perf` 中 `triton`/`compile` 帧是否只在启动后前几步出现 |
| 7 | `_build_attn_state` 的 `np.all` 是 O(B) 还是 O(B·S) | 读 numpy 源码语义 + 微基准；`num_computed_tokens_cpu` 是 int32 数组，`== 0` 是 O(B) |
| 8 | `AscendAttentionMetadataBuilder.build` 的 `pin_memory()` 每拍新建 pinned 页 | 火焰图看 `c10::cuda::CUDACachingAllocator` 或 `pin_memory` 分配帧；`torch.npu` 下的等价实现 |
