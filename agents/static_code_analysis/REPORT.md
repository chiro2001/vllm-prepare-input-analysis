# 静态代码逻辑分析 —— 交接报告

**Agent**: `/root/static_code_analysis`
**状态**: 完成
**产出**:

| 文件 | 内容 |
|---|---|
| `docs/01-prepare-input-code-logic.md` | 边界、逐子步骤、数据结构 IO 语义、Ascend 特有分支、prefill/decode 差异、同步点清单、调用图 |
| `docs/02-complexity-and-factors.md` | 一阶复杂度模型、24 项影响因素表、数据依赖控制流清单、临界条件推导、优化机会清单 |
| `data/static/prepare-input-steps.json` | 57 个步骤 + 4 个同步点 + 10 条数据依赖分支（机器可读） |
| `agents/static_code_analysis/verify_line_anchors.sh` | 行号锚点校验脚本（29 项，全部 PASS） |
| `agents/static_code_analysis/nl.sh` | 带行号打印源码的小工具 |

---

## 1. 结论摘要（给上层 agent 直接引用）

### 1.1 边界（最容易踩坑的一点）

**不存在一个跨 core/ascend 统一的 "prepare_input"。**

| | core (vLLM 0.26.0) | ascend |
|---|---|---|
| scope 名 | `"gpu_model_runner: preprocess"` | `"prepare input"` |
| 行号 | `gpu_model_runner.py:4148–4347` | `model_runner_v1.py:1859–2082` |
| 含 `_update_states` | ✅ `4153` | ✅ `1881` |
| 含 `_prepare_inputs` | ✅ `4195` | ✅ `1933` |
| 含 `_determine_batch_execution_and_padding` | ✅ `4216` | ✅ `1955` |
| 含 `_get_slot_mappings` | ✅ `4311` | ❌（ascend 不调用该函数，slot_mapping 由 block_table 内部产出） |
| 含 `_build_attention_metadata` | ✅ `4322` | ✅ `2065` |
| 含 `_preprocess` | ✅ `4345` | ❌（`2091`，在 scope 外） |
| 含 `update_cos_sin(positions)` | — | ❌（`2098`） |
| 空 batch 路径 | 在 scope 内（`4163`） | 在 scope 内（`1895`，含 `_dummy_run`） |

**给 profiling agent 的硬性提醒**：
1. ascend 的 `prepare input` **不含** embedding 查表（`_preprocess`）—— 如果拿 core 的
   `preprocess` 数据去解释 ascend，会高估 ascend 的 prepare_input 占比。
2. `prepare input` 的 wall 会被**空 batch step**稀释（`1895–1925` 会提前返回）。
   分层统计时必须按 `total_num_scheduled_tokens > 0` 过滤。
3. `synchronize_input_prep()` 的 `record()` 点在 ascend `1883`（`_update_states` 之后），
   **不覆盖 `_prepare_inputs` 的 H2D**。见 §3.2 的待验证项。

### 1.2 CPU 负载的四个真实来源（按预期贡献排序）

> **注意**：下表的"预期排序"是静态推理，**尚未有实测证据**，必须由 profiling agent 证伪。

| 排序 | 来源 | 代表位置 | 阶 |
|---|---|---|---|
| 1 | **`_build_attention_metadata` + `AscendAttentionMetadataBuilder.build`** | `model_runner_v1.py:2875–3224`、`attention_v1.py:291–395` | O(B·G)，与 T 解耦 |
| 2 | **H2D 调用次数（不是字节数）** | 每 step ≥10 次小拷贝 + ≥3 次 kernel launch | O(#调用) 固定底噪 |
| 3 | **`_update_states` 的 churn 敏感段** | `condense()` `gpu_input_batch.py:686–812`、`swap_states()` `569–679` | 稳态 O(B)，churn 时 O(B·S) |
| 4 | **Python 逐请求循环** | `_compute_prev_positions`（core `1758`）、`_prepare_input_ids` 异步分支（core `1799`）、`num_tokens=[...]`（ascend `1087`） | O(B) 但常数大 |

**明确排除的伪热点**：
- `slot_mapping` 计算 **不是** CPU 热点：core 与 ascend 都改成 Triton kernel
  （`vllm/v1/worker/block_table.py:166`、`vllm_ascend/ops/triton/compute_slot_mapping.py:12`），
  CPU 侧只剩 launch。
- `attn_mask` 生成 **不是** 每拍成本：`AttentionMaskBuilder` 是 `@singleton` 且带缓存
  （`attention_mask.py:33/42–49`）。
- `sampling_metadata` 重建 **不是** 每拍发生：只在 `batch_update` 非空时
  （`gpu_input_batch.py:831–832`）。

### 1.3 唯一强同步点

`self.num_accepted_tokens_event.synchronize()`（ascend `1104`）。分三档触发，**必须区分**：

| 场景 | event 存在 | 是否 record | 是否真阻塞 |
|---|---|---|---|
| `num_spec_tokens == 0` | ❌ | — | 否 |
| spec decode，模型**非** hybrid | ✅ | ❌（`_update_states_after_model_execute` 早退，core `1556–1557`） | **否**（空操作） |
| spec decode + hybrid（MTP/EAGLE + 线性注意力） | ✅ | ✅（core `1584`/`1590`，在 `sample_tokens` 里） | **是** |

> 这条对 profiling 极其关键：如果用 **MTP + hybrid 模型**做实验，`prepare input` 的 wall
> 会把 device 时间吸进来，看起来 CPU 很忙，其实在等 device。

---

## 2. 给 profiling agent 的建议

### 2.1 必须分层的维度

1. `total_num_scheduled_tokens == 0` vs `> 0`（空 batch 稀释）
2. `use_async_scheduling` on/off（决定 `_prepare_input_ids` 走哪条实现，两张实现差异极大）
3. `num_spec_tokens == 0` vs `> 0`（`_calc_spec_decode_metadata` 的 5 次 pin+H2D）
4. `cudagraph_mode == NONE` vs `FULL_DECODE_ONLY` vs `FULL`（padding 引入额外 `.tolist()`/`cat`）
5. `use_compress` / `_has_gdn` / `use_dcp` / `lora_config` / `enable_prompt_embeds`（各自独立的额外分支）
6. 请求 churn 强度（稳态 vs 高完成率 vs 频繁抢占）—— 区分 `condense()`/`swap_states` 的贡献

### 2.2 需要重点观察的函数（profiling 目标清单）

按预期 CPU 占比排序，建议直接进 `perf report` 的 top-N 核对：

```
_prepare_inputs                                    (ascend model_runner_v1.py:883)
_build_attention_metadata                          (ascend model_runner_v1.py:2875)
AscendAttentionMetadataBuilder.build               (attention_v1.py:291)
_update_states                                     (core gpu_model_runner.py:1169)
InputBatch.condense                                (core gpu_input_batch.py:686)
InputBatch.swap_states                             (core gpu_input_batch.py:569)
InputBatch._make_sampling_metadata                 (core gpu_input_batch.py:834)
_prepare_input_ids                                 (core gpu_model_runner.py:1761)
_compute_prev_positions                            (core gpu_model_runner.py:1746)
_calc_spec_decode_metadata                         (ascend model_runner_v1.py:1366)
split_decodes_and_prefills                         (attention/utils.py:360)
torch.index_select / np.repeat / np.cumsum         (ATen/NumPy)
torch.Tensor.pin_memory / copy_ / cudaMemcpy       (每拍新建 pinned 的证据)
```

**"每拍新建 pinned 缓冲"的三个可证伪信号**（建议单独 grep 火焰图）：
1. `attention_v1.py:330` 的 `query_start_loc_cpu.pin_memory()`
2. `_calc_spec_decode_metadata:1412–1416` 的 5 次 `.pin_memory().to()`
3. `_prepare_input_ids:1856–1880` 的 `torch.tensor(list, pin_memory=True).to(device)`

如果火焰图里出现稳定的 `c10::cuda::CUDACachingAllocator` / `pin_memory` / `munmap` 帧，
就证明这条路径值得优化。

### 2.3 建议的 workload 矩阵（覆盖因素表的边界）

| 维度 | 取值 | 目的 |
|---|---|---|
| 模型 | Qwen3.5-0.8B（BF16, TP1） | 让 device 尽量快，逼 CPU 露头 |
| B | 1, 8, 32, 128, 256 | 找 O(B) 斜率与固定底噪 |
| ISL | 128, 2k, 32k | 区分 O(T) 项 |
| OSL | 1（decode 主导） vs 长输出 | 影响 S 与 churn |
| chunked prefill | off / on（不同 chunk size） | 混合批次，`discard_mask` 非空 |
| MTP | off / n=1 / n=3 | spec 路径 |
| graph 模式 | NONE / FULL_DECODE_ONLY / FULL | padding 影响 |
| churn | 低（固定并发） / 高（短请求快速完成） | `condense`/`swap_states` |

**最小可行集**（若时间受限先做这 6 格）：
`(B=1, ISL=128, decode)`、`(B=64, ISL=128, decode)`、`(B=256, ISL=128, decode)`、
`(B=64, ISL=2k, chunked)`、`(B=64, ISL=128, decode, MTP n=3)`、
`(B=64, ISL=128, decode, graph=FULL_DECODE_ONLY)`。

---

## 3. 给无卡复刻 agent 的结论与提示

### 3.1 结论：**需要 record & replay**

理由（均给到行号，见 `docs/02` §2.1）：

1. 有 **10 处控制流由数据内容决定**（JSON 的 `data_dependent_branches` 数组），
   其中 6 处依赖请求级 token/length 语义：
   - `_build_attn_state`（`1320–1347`）的三重 `np.all` 判决 5 种 attn_state；
   - `is_kv_consumer and req_id in new_schedule_reqs`（`1286`）；
   - `prev_sampled_token_ids is not None`（core `1778`）切换 `_prepare_input_ids` 实现；
   - `num_accepted_tokens_event is not None`（`1103`）切换是否阻塞；
   - `batch_update_builder.removed`（core `698`）决定 `condense()`；
   - `batch_update`（core `831`）决定 `SamplingMetadata` 重建。
2. `_update_states` 的决策依赖**跨 step 粘性状态**（`batch_update_builder`、`prev_req_id_to_index`、
   `prev_sampled_token_ids`），纯参数矩阵无法表达。

纯参数化合成只适合**单因素对照实验**（如只改 B、只改 T），不适合作为"与真机一致"的主证据。

### 3.2 replay 载荷设计（已在 `docs/02` §6 给出 JSON schema）

**最小充分集**（按 step 记录）：

1. `SchedulerOutput` 全量（pickle 或结构化）—— 注意 `num_scheduled_tokens` 是 dict、
   `scheduled_spec_decode_tokens` 是 dict[str, list[int]]，**顺序会影响 `scheduled_new_reqs` 的处理顺序**。
2. **进入 `_prepare_inputs` 之前**的 `input_batch` CPU 数组：`num_computed_tokens_cpu`、
   `num_prompt_tokens_cpu`、`num_accepted_tokens_cpu`、`num_tokens_no_spec`、
   `token_ids_cpu` 的各请求**活跃前缀**（`_get_active_token_count` 长度）、`block_table` 各行、
   `prev_req_id_to_index`、`prev_sampled_token_ids`、`spec_token_ids`。
3. flags：`use_async_scheduling`、`num_spec_tokens`、`cudagraph_mode`、`_has_gdn`、`use_dcp`、
   `use_compress`、`lora_config is not None`、`enable_prompt_embeds`、`cudagraph_batch_sizes`。

**实现提示（不带卡）**：

- `torch_npu` 缺失 → 需要 shim 出 `torch.npu` 的 `Event`/`Stream`/`synchronize`，
  但**必须保持"异步语义"**：即 `Event.synchronize()` 在 replay 中不能变成空操作，
  否则 §1.3 的同步点会消失，topdown 分布会失真。
- `CpuGpuBuffer.copy_to_gpu()` 在无卡时是整个阶段最需要小心的地方：
  它是真实 DMA（CPU 周期 + 内存带宽），若用 `.clone()` 代替会显著改变
  cycles/memory 特征。建议**保留真实的内存拷贝**（`torch` CPU→CPU 或用 `numa` 感知的 buf），
  只把"提交给 device"这一步改成 no-op。
- Triton kernel launch（`compute_slot_mapping`）在无卡时应替换为**等价的 CPU 实现**
  （`block_table.py:222–239` 的 numpy 分支就是现成的等价实现，可直接借用）。
  **但要注意**：真机上是 kernel launch（CPU 只花 ~µs），CPU 实现会引入 O(T) 计算，
  反而**高估** CPU 负载。建议默认 no-op + 可选 O(T) 模式做敏感性分析。

### 3.3 一致性判据（建议）

| 指标 | 判据 | 测量方式 |
|---|---|---|
| topdown 各分量 | ±2–3 pp | libkperfx 920B preset（topdown） |
| IPC | ±10% | `perf stat -e instructions,cycles` |
| 热点函数 top-20 重合 | ≥80% | flamegraph-rs 折叠栈对比 |
| per-step `prepare input` wall 分布 | 同形 | 分位数包络 / KS 检验 |

**特别注意 topdown 口径**：无卡环境没有真实 NPU 的 DMA 引擎，
`copy_to_gpu` 的 CPU 侧行为（pin 页写、`madvise`、驱动的 `ioctl` 提交）
是 topdown 里 `backend_bound`/`bad_speculation` 的重要来源。若这部分被 shim 掉，
topdown 一定对不上。**这是"无卡一致性"的最大风险点，建议优先验证。**

---

## 4. 未解疑点（`[待验证]`）与验证方法

| # | 疑点 | 验证方法 | 归属 agent |
|---|---|---|---|
| 1 | `_compute_prev_positions` 的 O(B) Python 循环是否进 perf top-N | `perf record -g`，对比 B=1 / B=256 | profiling |
| 2 | `query_start_loc.copy_to_gpu()` 全量拷贝（`1039`）的固定开销 | A/B：改 `copy_to_gpu(num_reqs+1)` | profiling |
| 3 | `_calc_spec_decode_metadata` 的 5 次 pin+H2D（`1412–1416`）贡献 | MTP on/off 火焰图对比 | profiling |
| 4 | `synchronize_input_prep` 的 record 点（`1883`）是否覆盖不足 | 在 `_prepare_inputs` 末尾追加 event 做对照 | profiling |
| 5 | `condense()` 在高 churn 下的真实占比 | 三类 churn workload 对比 | profiling |
| 6 | Triton kernel 首次 JIT 是否污染 warmup | 检查前 100 step 是否出现 `triton`/`compile` 帧 | profiling |
| 7 | replay 中 `copy_to_gpu` 的 shim 方式是否破坏 topdown | 先做"仅把 device 提交 no-op、保留内存拷贝"的版本，与真机对比 topdown | 复刻 |
| 8 | `_build_attn_state` 的 `np.all` 是 O(B) 还是 O(B·S) | 微基准（纯 CPU，可立即做） | 复刻 |
| 9 | vLLM 0.27.1（另一镜像）在上述边界上是否有实质差异 | diff `gpu_model_runner.py` / `model_runner_v1.py` 的相关函数 | 待指派 |

---

## 5. 复现方式

```bash
# 行号锚点校验（30 项，应全部 OK）
cd /home/chiro/projects/vllm/preparing-input-phase
./agents/static_code_analysis/verify_line_anchors.sh

# JSON 结构校验
python3 -c "import json;d=json.load(open('data/static/prepare-input-steps.json'));print(len(d['steps']))"

# 带行号查看源码
./agents/static_code_analysis/nl.sh \
  /home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend/vllm_ascend/worker/model_runner_v1.py 883 1318
```

**源码口径（不得混池）**：

| 组件 | commit | 本地路径 |
|---|---|---|
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` | `refs/vllm`（亦可用 `HIST_PROJECT/vllm`，`git status` 干净） |
| vllm-ascend | `f2f74a16c3c50a76f4349d807918e83edec1e35c` | `HIST_PROJECT/vllm-ascend`（`git status` 干净） |

> `refs/vllm` 在本次分析开始时其 `vllm/` 子目录尚未 rsync 完成（后台 rsync 仍在跑），
> 因此行号校验以 `HIST_PROJECT/vllm` 为准 —— 两者为同一 commit 且工作树干净。

---

## 6. 我**没有**做的事（避免越界）

- 未运行任何 GPU/NPU 实验（chip3 归 profiling agent）。
- 未修改 `refs/` 下任何文件。
- 未修改 `HIST_PROJECT/` 下任何文件（只读）。
- 未在 `docs/`、`data/`、`agents/` 之外创建文件（除 `figures/` 未产出，留给图表 agent）。

## 7. 建议给图表与文档 agent 的输入

以下三张图最有价值，`docs/01` 已提供可直接改写的 ASCII 源：

1. **scope 边界对比图**：core `preprocess`（4148–4347）vs ascend `prepare input`（1859–2082）
   的区间条带，标出 `_preprocess` / `_build_attention_metadata` 的归属差异。
2. **调用图/火焰结构图**：`docs/01` §7 的 ASCII 树 → SVG。
3. **临界条件示意图**：`docs/02` §4 的 `T_prepare ≈ a·B + c` vs `T_forward(B)` 交点，
   横轴 B、纵轴时间，标出"更危险"的方向（TP↑、模型↓、graph↑）。
