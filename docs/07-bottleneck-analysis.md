# prepare_input 何时成为瓶颈：成本模型、判据与验证设计

> 版本 v1（2026-09-24，drivers_analysis）。配套：
> `01-prepare-input-code-logic.md`（边界与逐步骤）、`02-complexity-and-factors.md`（影响因素矩阵）、
> `plan/experiment-matrix-v2.md`（实验清单）、`data/model/complexity-model.json`（机器可读模型）。
>
> **证据等级约定**（全文强制）
>
> - `[已证实]`：有源码行号，或本仓库 `data/model/microbench-*.csv` 的实测数据支撑。
> - `[推断]`：由设计结构推出的结论，必须给出验证方法。
> - `[待实测]`：数字只能由 chip3 真机 / replay harness 产生。
>
> 本文所有公式的**显式假设**集中在 §1.2；任何一条不成立时结论要重新推导。
> 正文中带 `⛳` 的表格是给 measurement / replay 代理**回填实测值**的位置。

---

## 0. 结论速览（TL;DR）

1. **"prepare_input 成为瓶颈"不是"它耗时大"，而是三组比较**（§4）：
   - 小 batch 区：CPU 固定底噪 `c`（每步与 B、T 无关的启动/调用开销）vs device 每步底噪 `e`；
   - 大 batch 区：CPU 斜率 `a = ∂T_prepare/∂B` vs device 斜率 `d = ∂T_dev/∂B`；
   - token 区：CPU 斜率 `a_T = ∂T_prepare/∂T` vs device 斜率 `∂T_dev/∂T`。
   只有 `a > d`（decode）或 `c > e`（极低负载）时，增大负载才会让 prepare_input 从"被掩盖"转为"瓶颈"。
2. **异步调度默认开**（`[已证实]`，`vllm/config/vllm.py:1059–1107` +
   `UniProc/MultiprocExecutor.supports_async_scheduling()=True`），于是引擎走
   `step_with_batch_queue`、`max_concurrent_batches=2`（`vllm/config/vllm.py:512–522`），
   worker 侧 D2H 被搬到**独立线程**（`vllm/v1/executor/multiproc_executor.py:956–988`）。
   ⇒ 稳态下单步时间由 `max(T_cpu_engine_step, T_dev_step)` 决定，**CPU 有"一整拍 device 时间"的预算**。
3. **异步关闭时（如 `--async-scheduling false`、pooling、旧执行后端）流水退化为串行**，
   `T_step ≈ T_cpu + T_dev`，此时 prepare_input 的**任何**增量都 1:1 打到达吐与 TPOT 上 —— 判据完全不同（§4.5）。
4. **首选瓶颈工况**（`[推断]`，由 §3 的一阶模型推出）：纯 decode + 大 B + 短上下文
   + 小模型（0.8B）+ 图模式（FULL_DECODE_ONLY）+ MTP 开。此时 device 每拍只做 `B` 个 token
   的权重搬运与短 KV attention，而 prepare_input 的 **O(B) Python 段（实测 0.21 µs/req 纯 Python
   部分、0.272 µs/req 整段）与固定 H2D 次数**都不随 m 摊薄。
5. **长 prefill / 大 chunk 几乎不可能让 prepare_input 成为瓶颈**（`[推断]`）：device 时间 ∝ T·(S+T) 增长，
   而 CPU 侧整段 O(T) 斜率只有 **7.8–8.8 ns/token**（合成负载实测，§6.1；单算子更低：
   `np.subtract(out=)` 0.13 ns/el、`np.take` 0.76 ns/el、`np.repeat` 1.25 ns/el @T=32768）。
   真正的风险区在**小 chunk（512–2048 tokens）+ 大 B**：device 侧 GEMM 还没起量，CPU 底噪已经付完。
6. **prepare_input 内部只有 1 个会把 device 时间"吸"进 CPU 计时的点**：S17
   `num_accepted_tokens_event.synchronize()`（仅 spec decode **且** hybrid 模型；
   `vllm/v1/worker/gpu_model_runner.py:2088–2093`、创建点 `914–917`、record 点 `1583–1590`）。
   profiling 时必须把它的 wall 与 CPU 真忙时间分开（§2.6）。
7. **TP 会放大 CPU 瓶颈**（`[推断]`）：每个 rank 各自跑一份 prepare_input，步时间由**最慢 rank** 决定，
   `E[T_step] ≈ μ + σ·E[max_N Z]`；CPU 段的抖动会以 straggler 形式直接吃掉吞吐（§4.4）。
8. **是否需要 record & replay：需要**，且理由比"参数化难度"更硬 —— 见 §7.1 的 9 条数据依赖控制流；
   但 **replay 的粒度可以分层**（只重放改变分支的语义字段 + 规模参数化），不必逐 step 全量录（§7.2）。

---

## 1. 记号、假设与证据来源

### 1.1 记号

| 符号 | 含义 | 取值来源 |
|---|---|---|
| `B` | 本拍 `num_reqs` | `input_batch.num_reqs` |
| `T` | 本拍 `total_num_scheduled_tokens` | `SchedulerOutput.total_num_scheduled_tokens` |
| `m = T/B` | 平均每请求调度 token 数（decode=1，chunked prefill>1） | — |
| `S` | 单请求已算 token 数（≈上下文长度） | `num_computed_tokens_cpu` |
| `P` | prompt 长度 | `num_prompt_tokens_cpu` |
| `W` | `max_num_reqs`（缓冲区宽度，决定切片与尾部清零成本） | `scheduler_config.max_num_seqs` |
| `K` | `max_num_blocks_per_req = cdiv(max_model_len, block_size)` | `block_table.py:70–90` |
| `G` | KV cache group 数（非 hybrid=1） | `kv_cache_config.kv_cache_groups` |
| `L` | 模型层数 | `model_config` |
| `n` | `num_spec_tokens`（MTP/EAGLE draft 数） | `speculative_config` |
| `R` | TP 规模（rank 数） | `parallel_config.tensor_parallel_size` |

### 1.2 显式假设（不成立则结论需重推）

| # | 假设 | 影响 | 验证方式 |
|---|---|---|---|
| H1 | 稳态、预热完成（Triton JIT、cudagraph capture、pinned 缓冲已分配） | 排除一次性开销 | 丢掉前 100 step 再统计；`perf` 中不应出现 triton compile 帧 |
| H2 | worker 侧 RPC 串行处理（一次一个 `execute_model`/`sample_tokens`） | 决定流水形状 | `multiproc_executor.py` `worker_busy_loop`（单 deque 循环） |
| H3 | 异步调度默认开、`max_concurrent_batches=2` | 决定 `max()` vs 串行 | 启动日志 "Asynchronous scheduling is enabled"；`engine_core.step is step_with_batch_queue` |
| H4 | H2D 与计算同流、`non_blocking=True`；pinned 缓冲复用 | 允许重叠 | `CpuGpuBuffer.copy_to_gpu`（`vllm/v1/utils.py:139–142`） |
| H5 | `_prepare_inputs` 内除 S17 外无 `.item()/.cpu()/synchronize()` | CPU 段不吸收 device 时间 | `01-...md §6` 全局搜索结论 + 火焰图不应出现 `aclrtSynchronize*` |
| H6 | 采样与反序列化不计入 prepare_input（属 `sample_tokens`/engine core） | 分母口径 | `prepare input` scope 边界（`01-...md §1.2`） |
| H7 | 单 rank 视角；TP 只在 §4.4 讨论 | — | — |

### 1.3 数据来源

| 来源 | 用途 | 位置 |
|---|---|---|
| 源码（vLLM 0.26.0 = `refs/vllm`；vllm-ascend 0.26.0rc1） | 控制流、调用次数、阶 | 行号内联 |
| a3-22 CPU 微基准（核 200–203，实测） | 一阶系数 | `data/model/microbench-*.csv` |
| `[待实测]` 项 | 由 chip3 真机/replay 回填 | §3.4、§6 的 ⛳ 表 |

---

## 2. 执行流水的真实形状（决定"掩盖"能否发生）

### 2.1 engine core 的两种 step 实现

`[已证实]` `vllm/v1/engine/core.py`：

```
self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue   # :224-226
batch_queue_size  = vllm_config.max_concurrent_batches                                 # :199
# 主循环调用点：outputs, model_executed = self.step_fn()                               # :1403
```

- `max_concurrent_batches`：`async_scheduling=True` 且 PP≤1 → **2**；否则 `pp_size`（=1）。（`config/vllm.py:512–522`）
- **异步开 → 走 `step_with_batch_queue`（:617）**：schedule → `execute_model(non_block=True)` →
  `sample_tokens(non_block=True)` → 入队；队列未满就**立刻返回**，不等 device（:672–678）。
- **异步关 → 走 `step()`（:576）**：`execute_model` 后**立刻 `future.result()`**（:594），
  再 `sample_tokens`（其结果需 D2H，被 `update_from_output` 使用）。

### 2.2 worker 侧：谁在等 device

`[已证实]` `vllm/v1/executor/multiproc_executor.py`：

```
worker_busy_loop():                    # 主线程，串行 dequeue RPC
    method, args, ... = rpc_broadcast_mq.dequeue()
    output = func(*args, **kwargs)     # execute_model / sample_tokens
    self.handle_output(output)

handle_output():     if use_async_scheduling: async_output_queue.put(output)   # :956–965
async_output_busy_loop():  output.get_output()   ← D2H + device 同步在这里     # :968–988
```

`AsyncGPUModelRunnerOutput.__init__`（`gpu_model_runner.py:258–300`）在**独立 copy stream** 上发起
`sampled_token_ids.to("cpu", non_blocking=True)` 并 record 一个 event，**不在主线程同步**；
真正的 synchronize 发生在 `get_output()`（由 `async_output_busy_loop` 线程调用）。

> **结论（关键结构事实）**：异步调度下，worker 主线程在稳态**几乎不等 device**；
> 它的每步工作是纯 CPU 的"准备 + 下发"。因此
> `T_step ≈ max(T_cpu_engine_step, T_dev_step)`，其中
> `T_cpu_engine_step = T_sched + T_rpc + T_prepare + T_fwd_launch + T_sample_launch + T_post`。
> **prepare_input 不是单独与 device 赛跑，而是与 engine 的其它 CPU 成本一起，共享"一拍 device 时间"的预算。**

### 2.3 `prepare_inputs_event` 的节流效应（易被忽略）

`[已证实]` `gpu_model_runner.py:3808–3822`（实现）+ `737–741`（创建）+ ascend record 点
`vllm_ascend/worker/model_runner_v1.py:1883`：

```
@contextmanager
def synchronize_input_prep(self):
    if self.prepare_inputs_event is None: yield; return
    self.prepare_inputs_event.synchronize()      # 入口：等"上一拍 record 点"之前的流上工作
    try: yield
    finally: self.prepare_inputs_event.record()  # 出口：记录当前流位置
```

ascend 的 with 体只包住 `_update_states`（`1860–1883`），即 **record 点落在 `_prepare_inputs` 之前**。
记 `event_N` = 第 N 拍在 `_update_states` 结束时 record 的事件。第 N+1 拍入口的 `synchronize()` 等的是 `event_N`；
由于同一 stream 顺序入队，`event_N` 之前的流上工作包含 `forward(N-1)` 与 `_update_states(N)` 的 H2D
（**不包含** `_prepare_inputs(N)` 之后的 H2D 与 `forward(N)`）。
⇒ 该同步把 **CPU 的领先量限制在"约一拍 device 时间"**：CPU 想进入第 N+1 拍，必须等第 N-1 拍的 forward 落地。
这正是 §4.1 判据（"CPU 只有一拍 device 时间的预算"）可操作的原因，而不是乐观假设。
`[待实测]` 精确边界建议用 A/B 实验确认（在 `_prepare_inputs` 末尾补记一个 event，观察是否产生更深的流水）。

`[推断]` 推论：device 变慢会让该同步点等待变长，使 CPU 段 wall **看起来**变长；
反之 CPU 很快时该点会让 CPU 空转等待。profiling 必须能把这两种情况与"真忙"区分（§2.6）。

### 2.4 prepare_input 在时间轴上的落点

```
engine core (async on, depth 2):
  t0  schedule(N)             ─┐
  t1  RPC execute_model(N)    ─┤→ worker: prepare_input(N) → forward launch(N) → post(N)
  t2  RPC sample_tokens(N)    ─┤→ worker: logits+sample launch(N) → AsyncOutput(copy stream)
  t3  schedule(N+1)            │   ← 与 device(N) 并行
  t4  RPC execute_model(N+1)  ─┘   ← worker 主线程：prepare_input(N+1)【与 device(N) 重叠】
  t?  future.result()              ← 只在队列满（每 2 拍）时才等更早那拍的 output

device:  [==== forward(N) ====][== sample(N) ==][==== forward(N+1) ====] ...
```

**prepare_input 的可掩盖窗口 = `T_dev_step`**（不是整个 step wall，也不是端到端）。
当 `T_prepare > T_dev − (T_sched + T_rpc + T_launch_fwd + T_post)` 时，engine 主循环开始被 CPU 拖住。

### 2.5 `record_function("prepare input")` 的 wall 到底测了什么

- 包含：`synchronize_input_prep` 的等待（§2.3）、`_update_states`、`_prepare_inputs`、
  `_determine_batch_execution_and_padding`、`_build_attention_metadata`、`_sanitize_placeholder...`。
- 不包含：`_preprocess`、`update_cos_sin`、`forward`、`post process`、`sample_tokens`。
- **两类必须剔除/分层的 step**：空 batch step（`num_scheduled_tokens==0`，ascend `1895–1910`）
  与 timing 校准窗口（`_sync_device()`，ascend `1801–1810`）。

### 2.6 区分"CPU 真忙"与"被阻塞"（profiling 判读规则）

| 现象 | phase wall | `perf`（on-CPU） | cycles | 判读 |
|---|---|---|---|---|
| CPU 真忙 | 长 | 有栈、采样密 | 高 | 真 CPU 瓶颈 |
| 被 event 阻塞 | 长 | 无采样 / 落在 `sched_*`、`futex`、`poll` | 低 | device 或同步问题 |
| H2D 入队（驱动路径） | 中 | 落在 `libascendcl` / `aclrtMemcpyAsync` / 驱动 | 中 | "调用开销"型 CPU 成本 |

`[推断]` 判读方法：`perf record -g` 的**样本量 × 采样周期**应与 phase wall 同量级；
若 wall ≫ on-CPU 时间（>3×），先怀疑 event/同步，而不是 CPU 计算。

---

## 3. 成本模型

### 3.1 一阶模型

```
T_prepare(B, T, S, G, n, W, flags)
  =  c₀                                          # 固定底噪
  +  a_B · B                                      # 每请求线性
  +  a_T · T                                      # 每 token 线性
  +  a_K · B · K · G                              # 块表 H2D 字节
  +  a_L · L                                      # 逐 layer metadata 赋值
  +  a_n · [n>0] · B · n                          # spec 元数据（pin+H2D）
  +  a_c · churn(B, S, #removed, #preempted)      # _update_states 突变项（二阶）
  +  a_dp · [dp_size>1] · B                       # DP all_reduce（每步）
  +  T_sync                                       # 仅在 S17 非零
```

| 项 | 阶 | 数据内容依赖 | 主要来源（行号见 01 文档） |
|---|---|---|---|
| `c₀` | O(1) | 否 | ≥10 次 H2D/device op 的调用开销 + 8–12 次 numpy 调用开销 + 2×`.tolist()` + 若干 ATen 调用 |
| `a_B·B` | O(B) | 否 | `_compute_prev_positions`(S7)、`num_tokens` list(S16)、`np.array(list)`(S16/1918)、两次 `.tolist()`(S11/S21 的 metadata)、`optimistic_seq_lens`(S13)、`_build_attn_state`(S3)、`_split_decodes_and_prefills` |
| `a_T·T` | O(T) | 否 | `np.repeat`×2(S2)、`cumsum+subtract(out=)`(S4)、`positions` gather+add(S5)、`index_select`(S9)、3–4 次 O(T) H2D(S9/S19/S20) |
| `a_K·B·K·G` | O(B·K·G) | 否 | S1 `commit_block_table`：`copy_to_gpu(num_reqs)` → 形状 `(B,K)` int32 × G |
| `a_L·L` | O(L) | 否 | `_build_attn_group_metadata` 逐 layer `for layer_name in attn_group.layer_names`（ascend `3138–3139`） |
| `a_n·B·n` | O(B·n) | **是** | `_calc_spec_decode_metadata` 5×`pin_memory().to()`（ascend `1366–1430`）+ 异步 `_prepare_input_ids` 的 4 个 list（core `1799–1826`） |
| `a_c` | 稳态 O(B)；churn O(B·S) | **是** | `InputBatch.condense()`（core `686–812`）、`swap_states()`（`569–679`） |
| `a_dp` | O(B)+1 次 all_reduce | **是**（dp_size==1 短路） | `_sync_metadata_across_dp`（ascend `702–740`） |
| `T_sync` | device 时间 | **是** | S17（ascend `1101–1127`） |

### 3.2 固定底噪 `c₀` 的**实测**分解（920B，a3-22 核 200–203，单线程绑核）

全部来自 `data/model/microbench-*.csv`（已扣空循环开销）：

| 原语 | 单次调用 µs | 规模项 | 备注 |
|---|---:|---|---|
| `np.add(buf,buf,out=)` | 0.42（B=1）–0.93（B=1 with big buffer） | — | 调用开销主导 |
| `np.nonzero(mask)` | 0.54 | 与 B 无关 | |
| `np.subtract(arange,offs,out=)` | 0.80 → 0.97（T=512）→ 4.35（T=32768） | 0.13 ns/el | |
| `np.take(flat,idx)` | 1.02 → 25.0（T=32768） | 0.76 ns/el | |
| `np.repeat(arange,ones)` | 0.97 → 40.9（T=32768） | 1.25 ns/el | |
| `np.gather(buf[idx])` | 0.19 → 35.2（T=32768） | 1.07 ns/el | |
| `np.cumsum(counts)` | **2.63（B=1）→3.58（B=256）** | ≈3.8 ns/el | 固定成本 2.6 µs |
| `np.all(counts==1)` | **3.57（B=1）→3.60（B=256）** | 0 | 纯固定；`_build_attn_state` 最多 3 次 |
| `_get_cumsum_and_arange`（S4 全序列 / 3 次 numpy） | **8.08（B=1）→10.0（B=256）** | — | |
| `np.array(list,int32)` | 0.40（B=1）→6.91（B=256）→15.0（B=512） | ≈27–31 ns/el | |
| `np.ndarray.tolist()` | 0.09（B=1）→2.03（B=256） | 7.9 ns/el | |
| `arr[1:].tolist()` | 0.18（B=1）→2.16（B=256） | 8.4 ns/el | |
| `[requests[r].num_tokens for r in req_ids]` | 0.20（B=1）→32.7（B=256）→67.9（B=512） | **≈128–133 ns/req** | 单项最大 |
| `_compute_prev_positions`：for + `dict.get` | 0.22（B=1）→9.9（B=256）→20.0（B=512） | ≈39 ns/req | |
| `dict` 构建 `{rid: i}` | 0.24（B=1）→11.9（B=256） | ≈47–51 ns/req | 参考值（`req_id_to_index` 类） |
| `set(finished_req_ids)` | 0.18–0.35 | ≈1.3 ns/req | 可忽略 |

**由此得到的底噪分解**：`_prepare_inputs` 内 numpy 调用约 8–12 次、Python 固定开销约 5 处、
`.tolist()` 2 次 ⇒ **numpy+Python 部分 ≈ 20–30 µs/step**；再加上同数量的 ATen 调用
（实测 1.6–2.3 µs/次 × 约 8–12 次 ≈ 15–25 µs），与
**表 6-1 的 CPU-only replica 实测 `c₀` = 48.0–48.7 µs 一致** `[已证实]`。
完整 `c₀` 还要叠加 ≥10 次 H2D 的入队开销与 device op launch（§6.3 的 `k_h2d_small`、`k_launch`，`[待实测]`）。

### 3.3 斜率 `a_B` 的**实测**分解（每请求，decode 稳态）

| 贡献项 | ns/req | 来源 |
|---|---:|---|
| `[requests[r].num_tokens for r in req_ids]`（S16） | ≈128 | 实测 |
| `_compute_prev_positions` for + `dict.get`（S7） | ≈39 | 实测 |
| `np.array(num_tokens, int32)`（S16） | ≈27–31 | 实测 |
| 两次 `.tolist()`（metadata build，`attention_v1.py:332–333`） | ≈16 | 实测（7.9 + 8.4 ns/el） |
| `optimistic_seq_lens` 的 ATen `add(out=)` + `fill_`（S13） | `[待实测]`（ATen，估计 20–60） | — |
| `_build_attn_state` 分支判定（S3） | ≈0（已是固定 `c₀`） | 实测 |
| `discard_*` 两次 H2D 的**入队**开销 | `[待实测]` | — |
| 块表 H2D 字节（`B·K·4` 字节，DMA 并行，非 CPU 计算） | `[待实测]` | 推算 |
| **合计（CPU-only 实测下界）** | **≈210–240 ns/req** | 实测 |

> **可证伪预测 P1**：真机 decode 稳态下，`T_prepare` 对 B 的中位斜率落在 **0.20–0.60 µs/req**。
> 若实测 >1 µs/req，说明存在本模型未识别的 O(B) 项 —— 首要嫌疑：
> `_update_states` 的逐请求属性更新、`input_batch` 的标量数组搬运、`attn_group.layer_names` 循环。

### 3.4 二阶项（非线性来源，每条都要能单独开关）

| # | 机制 | 表达 | 触发条件 | 测量设计 |
|---|---|---|---|---|
| E1 | `condense()` 空位下沉 | `O(#removed·active_tokens)`，最坏 O(B·S) | 请求完成/被移除 | 高 churn vs 稳态的 `_update_states` 时长差 |
| E2 | `swap_states()` | O(B·S) | 混合 batch 需重排 | decode-only vs 混合 batch |
| E3 | `add_request()` 整行写入 | 首拍 O(P) | 新请求进入 | 冷启动首拍 vs 稳态 |
| E4 | spec 元数据 5×pin+H2D | 固定 5 次 + O(B·n) | `scheduled_spec_decode_tokens` 非空 | MTP on/off |
| E5 | 图模式 padding | 多 `.tolist()` 元素 + `torch.cat` + device `fill_` | `cudagraph_mode ∈ {FULL, FULL_DECODE_ONLY}` | graph on/off |
| E6 | DP all_reduce | 每步 1 次通信 + O(dp) | `dp_size>1` | DP=1 vs 2 |
| E7 | hybrid S17 同步 | wall 吸收 device 时间 | spec + hybrid（Mamba/GDN） | GDN vs 纯 attention 模型 |
| E8 | `allowed_token_ids` 惰性大分配 | 首次 O(B·V)（V≈151k，B=256 → 38 MB） | 请求带 `allowed_token_ids` 且 batch_update 非空 | 首次 vs 后续 step 的 P99 |
| E9 | GIL/线程竞争 | ≥2 线程争 GIL | 异步调度 + 主线程与 output 线程 | `perf sched` / py-spy 看主线程 off-CPU |

---

## 4. 瓶颈判据

### 4.1 通用判据（异步调度默认开）

单拍各量都发生在同一个 worker 进程的主线程上，串行：

```
T_cpu  = T_sched + T_rpc + T_prepare + T_launch_fwd + T_post          (1)
T_step ≥ max( T_cpu , T_dev )                                        (2)
```

**"prepare_input 成为瓶颈"的可操作定义**：

```
T_prepare > T_dev − (T_sched + T_rpc + T_launch_fwd + T_post)  :=  B_cpu      (3)
```

`B_cpu` 是"允许 prepare_input 消耗的 CPU 预算"，**不等于 `T_dev`**：调度、RPC 序列化、
kernel launch 已经先扣掉。`[待实测]` 各分项量级（预期 `T_sched+T_rpc` ≈ 50–150 µs、
`T_launch_*` ≈ 30–100 µs），必须在真机同时打点，否则会高估 `B_cpu`。

**占比口径**（对应 `COORDINATION.md §6` 的强制要求）：

```
share_cpu  = T_prepare / T_cpu     # CPU 预算占用率（分子分母同为 CPU 时间）
share_step = T_prepare / T_step    # 单步 wall 占比（被掩盖时 < share_cpu）
share_e2e  = 吞吐/时延损失归因（仅当 T_cpu > T_dev 时才有非零贡献）
```

### 4.2 decode 稳态的临界条件（用 B 表达）

decode（`m=1`、短上下文）下把 (1)(2) 线性化：

```
T_prepare ≈ c_p + a·B        # a = 每请求边际 CPU 成本（P1 预测 0.2–0.6 µs/req）
T_dev     ≈ e   + d·B        # d = 每请求边际 device 成本
T_cpu_other ≈ c_o            # schedule+RPC+launch+post（与 B 弱相关）            (4)
```

```
CPU 受限 ⟺ c_p + a·B + c_o > e + d·B
        ⟺ B > (e − c_p − c_o) / (a − d)          （仅在 a > d 时成立）
```

三种形态（**这是本文最希望被实测证伪/证实的部分**）：

1. **`a > d`**：device 每请求边际时间比 CPU 小 ⇒ 存在临界 `B*`，B 越大 prepare_input 占比越高。
   典型：小模型、短上下文、图模式（kernel 变快但 CPU 逻辑不变）。
2. **`a ≤ d` 但 `c_p + c_o > e`**：CPU 固定底噪已超过 device 固定底噪 ⇒ **任何 B 都是 CPU 瓶颈**（低负载也慢）。
   典型：小 batch + TP 分片 + 图模式 + 极短序列 + 高固定 launch/H2D 成本。
3. **`c_p + c_o < e` 且 `a < d`**：prepare_input 全程被掩盖 ⇒ 优化它不会提升吞吐（**只降 CPU 占用**）。
   这是"看到占比高但没有优化收益"的常见误判来源，必须用 `share_e2e` 而不是 `share_step` 判断。

`[待实测]` 至少需要测出 `a, d, c_p+c_o, e` 四个数才能给出 `B*`。建议直接拟合：
`T_prepare(B)` 与 `forward(B)` 各 ≥6 点，报斜率与置信区间，而不是先验假设。

### 4.3 chunked prefill 的临界条件（用 chunk 大小表达）

device 侧（单请求、chunk = T token、已有上下文 S）：

```
T_dev(T) ≈ e' + β·T + γ·T·(S + T/2)      # 线性项=权重/MLP 与 KV 写；二次项=attention 打分
```

CPU 侧：`T_prepare ≈ c_p + a_T·T`，其中合成负载实测 `a_T ≈ 7.8–8.8 ns/token`（§6.1，
由 6–8 个 O(T) 算子叠加；单算子系数见 §3.2 的 ns/元素）。于是
`T_prepare/T_dev` 随 T **单调下降**。结论：

1. **chunk 越小越危险**：`T→0` 时比值 → `c_p/e'`；小 chunk 下 GEMM 严重欠占用使 `e'` 不退让，
   比值可能 >1 ⇒ **CPU 主导**（"chunked prefill 用 512 反而更慢"的机制）。
2. **chunk 越大越安全**：二次项把 device 时间拉爆，CPU 项线性 ⇒ 比值 → 0。
3. 判据落点：解 `c_p + a_T·T = e' + β·T + γ·T(S + T/2)` 得最小可行 chunk `T_min`。
   `[待实测]` 用 `max_num_batched_tokens ∈ {512, 2048, 8192}`（EXECUTION §2 组 C）三点拟合 `β, γ`。

> **可证伪预测 P2**：当 `max_num_batched_tokens < T_min` 时，`prepare input` 的 phase wall
> **不随 chunk 变小而下降**（它是 CPU 固定成本），即在小区间出现"平台"，平台高度 ≈ `c_p`。

### 4.4 TP / 多 rank 的 straggler 放大

`[已证实]` 每个 rank 是独立 worker 进程，各自执行**同一份 CPU 逻辑**
（`multiproc_executor.py:311–322` 广播 `execute_model`），且 rank 间在 collective 处必须等齐：

```
T_step = max_{r=1..R} (T_cpu,r) + T_dev + T_comm
```

若 `T_cpu` 近似正态 `N(μ, σ²)`，则 `E[max] ≈ μ + σ·√(2 ln R)`。
TP=8、`σ=0.1μ` 时每步多付 ≈0.2μ；而 CPU 段的抖动来源正是 §5 的 A5/A7/A8（pinned 分配、GIL、NUMA）。

`[推断]` 验证方法：

1. 同机 TP=2/4 跑同一 workload，比较**单 rank CPU 时间分布**与**跨 rank 最大值**；
2. 代码里两处 `blocking=True` 的 event（`gpu_model_runner.py:737–741` 注释）已提示
   "busy-poll 驱动锁会让 rank 变 straggler" —— 用 rank 间 phase 起始时间偏移量 + `preempt/ms` 验证。

### 4.5 异步关闭（串行流水）时的判据

`[已证实]` `step()`（`core.py:576–615`）在 `execute_model` 后立即 `future.result()`，
`sample_tokens` 的 D2H 结果被 `update_from_output` 使用 ⇒ 同拍内 CPU/device 无法重叠：

```
T_step ≈ T_sched + T_rpc + T_prepare + T_dev + T_post + T_sample + T_sched_update
```

⇒ 判据退化为 **只要 `∂T_prepare/∂B > 0` 就影响 TTFT/TPOT**；
此时 `share_step = T_prepare/T_step` 直接等于优化收益上限，"被掩盖"这一现象不存在。
**建议实验矩阵保留一组 `--async-scheduling false` 作为对照组**，用它把"真实 CPU 成本"
从"被掩盖后的可见成本"中剥离出来。

### 4.6 预登记的可证伪预测（供实验直接判真伪）

| # | 预测 | 现状 | 证伪条件 |
|---|---|---|---|
| P1 | 真机 decode 稳态下 `T_prepare` 对 B 的中位斜率 ∈ [0.20, 0.60] µs/req | **CPU-only 部分已被微基准支持（0.268 µs/req）** | 实测斜率 >1 µs/req（须列出 perf top 中新增的 O(B) 帧） |
| P2 | 存在 `T_min`：`max_num_batched_tokens < T_min` 时 `prepare input` 的 phase wall 不随 chunk 变小而下降（平台高度 = `c_p`） | 待验证 | chunk=512 的 phase wall 显著低于 chunk=2048（>15 %） |
| P3 | `T_prepare` 中"内核外开销"（H2D 入队 + pin + 固定调用）在 B=T=1 时占 35–75 % 的 prepare wall | 待验证 | 真机 B=T=1 的 phase wall < 65 µs（即 CPU-only 的 48.7 µs 之外不足 16 µs） |
| P4 | TP 增大不降低 `T_prepare`（每 rank 一份），因此 TP≥4 时 0.8B 模型的 `B*` 明显小于 TP=1 | 待验证 | TP=4 的单 rank `T_prepare` 显著低于 TP=1（>15 %） |

---

## 5. 放大 / 掩盖机制总表

方向：**掩盖** = 让 `prepare_input` 的占比/影响变小；**放大** = 让它的占比/影响变大。

| # | 机制 | 方向 | 机理（证据） | 验证方法 |
|---|---|---|---|---|
| H1 | async scheduling + depth-2 流水 | 掩盖 | §2.1/2.2（`core.py:225/617`、`max_concurrent_batches=2`） | `--async-scheduling true/false` 对照同一 workload |
| H2 | D2H 移到独立线程 | 掩盖 | `multiproc_executor.py:956–988` | 关异步后同点 TPOT 对比；`py-spy dump` 看主线程状态 |
| H3 | 大 `T_dev`（长上下文 / 大模型 / 大 chunk） | 掩盖 | `T_dev ∝ S` 或 `∝ T²` | ISL/OSL 扫描，看 `share_step` 下降 |
| H4 | 图模式（FULL_DECODE_ONLY） | 掩盖 device / 放大 CPU | 减少 launch 次数；但引入 padding 与额外 `.tolist()` | graph on/off 两组 phase wall + `forward` wall |
| H5 | H2D 先发起再算（S1 注释 `904–905`） | 掩盖 | 拷贝与 CPU 计算重叠 | 火焰图里 `commit_block_table` 应几乎不可见 |
| H6 | `prepare_inputs_event`（§2.3） | 掩盖（限流） | 把 CPU 领先量压到 ~1 拍 | 在 `_prepare_inputs` 末尾补 event 的 A/B 实验 |
| A1 | 大 B（decode） | 放大 | O(B) Python 段 + 固定 H2D 次数 | 组 A 并发扫描 |
| A2 | 请求 churn（完成/新进/抢占） | 放大 | `condense()` O(B·S) | 稳态 vs 高 churn（短 OSL、高并发） |
| A3 | MTP / spec decode | 放大 | 5×pin+H2D（`1366–1430`）+ async 路径 O(B) list | 组 E on/off |
| A4 | hybrid 模型（GDN/Mamba） | 放大 + **吸收 device** | S17 同步（`1101–1127`） | GDN 模型 vs 纯 attention 模型 |
| A5 | 每步新建 pinned 缓冲 | 放大 | `attention_v1.py:330`、`_calc_spec_decode_metadata:1412–1416` | 火焰图看 `pin_memory` / `cudaHostAlloc` 帧占比 |
| A6 | 小 T 的固定启动开销 | 放大（相对） | 实测：`np.all`≈3.6 µs、`np.cumsum`≈2.6 µs/次 | B=1/T=1 单点的 wall 与预测 `c₀` 对比 |
| A7 | GIL / 线程竞争 | 放大 | 主线程 + async output 线程 | `perf sched` 的 off-CPU 时间、py-spy 采样 |
| A8 | NUMA 跨节点访问 | 放大 | 大数组 gather/copy 带宽下降 | 微基准：`(cpu node × mem node)` 四组合（`data/model/microbench-numa.csv`） |
| A9 | TP 最大者决定 | 放大 | §4.4 | 同机 TP=2/4 的 rank 间 phase 分布 |
| A10 | DP all_reduce（每步） | 放大 | ascend `702–740` | DP=1 vs 2 |
| A11 | 缓冲区宽度 `W=max_num_reqs` 过大 | 放大 | 全量 `copy_to_gpu()`（S11/S12/S15）+ 尾部 `fill_` | 改 `--max-num-seqs` 的 A/B |

---

## 6. 反事实估算（实测微基准 + 显式假设；供真机对照）

### 6.1 合成负载（replica）实测

`scripts/microbench_prepare_input.py::PrepareInputReplica` 按 `_prepare_inputs` 的**代码顺序**
复刻了 CPU-only 子集（numpy/ATen/逐请求循环/`.tolist()`/固定次数的 H2D **计数**），
不含真 H2D、device op 与 churn 突变路径。

**表 6-1 合成负载耗时（实测，`data/model/microbench-composite.csv`）**

CPU-only replica（含 ATen、逐请求循环、`.tolist()`；**不含真 H2D / device op**）：

| B | m=1（decode） | m=4 | m=16 | m=64 | m=256 |
|---:|---:|---:|---:|---:|---:|
| 1 | **48.7** | 57.5 | 56.5 | 56.6 | 59.4 |
| 8 | **49.0** | 59.8 | 59.3 | 65.1 | 75.2 |
| 32 | **55.6** | 65.0 | 69.9 | 81.5 | 131.5 |
| 64 | **64.0** | 74.9 | 82.3 | 110.6 | 214.4 |
| 128 | **82.1** | 97.5 | 114.4 | 168.6 | 377.5 |
| 256 | **117.1** | 139.4 | 168.8 | 283.7 | 710.5 |

（单位 µs/step；中位数，绑核 200–203、单线程、T = B·m）

> 图：`figures/07-replica-cost-curve.svg`（decode/chunked/MTP 的 B 曲线 + 拟合线）。

**MTP（spec n=2，`m=1`）**：B=1 → 51.9；B=8 → 53.2；B=32 → 62.8；B=64 → 76.1；B=128 → 102.7 µs。
相对无 spec 的增量：B=32 时 +7.2 µs、B=128 时 +20.6 µs —— 即 **≈0.16 µs/req 的额外 CPU 成本**，
外加 5 次"每次固定"开销（replica 只计了次数，真实 pin+H2D 另计，见 §6.3）。
（按回归斜率差算：`0.406 − 0.272 = 0.134 µs/req`；按端点差算 `20.6/128 = 0.161 µs/req`，
两者给出的区间是 **0.13–0.16 µs/req**。）

**最小二乘拟合**（n=6）：

```
decode:  T_prepare_cpu_only ≈ 47.25 µs + 0.2721 µs/req · B      (R² = 0.99937)
MTP n=2: T_prepare_cpu_only ≈ 50.42 µs + 0.4060 µs/req · B
chunked: T_prepare_cpu_only ≈ c + 7.8–8.8 ns/token · T          (B=8 与 B=256 两族分别拟合)
```

⇒ **`c_p`(CPU-only) ≈ 47–49 µs**、**`a`(CPU-only) ≈ 0.27 µs/req**、
`a_T`(CPU-only, replica) ≈ **7.8–8.8 ns/token**。
`a` 落在预测 P1 的 0.20–0.60 µs/req 区间内（P1 得到支持）；MTP 使 per-req 斜率升到 0.41 µs/req。
完整 `c_p`/`a` 还要加 H2D 与 device op 入队成本（§6.3 的 `k_h2d_small`、`k_launch`）。

> **口径提醒**：`a_T = 7.8–8.8 ns/token` 是**合成负载的整段斜率**，比 §3.2 的
> **单算子**系数（`np.subtract(out=)` 0.13 ns/el、`np.take` 0.76 ns/el、`index_select` 1.29 ns/el）
> 大一个量级，因为每 token 要经过 **6–8 个 O(T) 算子**（两次 `repeat`、`subtract`、`gather`、
> `add`、`token_indices` 的乘加、`index_select`）。报数时必须写清是**单算子**还是**整段**。
> 另见 D7：replica 为防止缓冲越界多做了一次 `%`（约 +1.5–2 ns/token），真机应更小。

⛳ **表 6-1b 待回填：真机同工况 `prepare input` phase wall**

| 工况 | B | T | replica（CPU-only） | 真机 phase wall | 差值 = H2D/launch/sync |
|---|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 48.7 µs | ⛳ | ⛳ |
| decode | 64 | 64 | 64.0 µs | ⛳ | ⛳ |
| decode | 256 | 256 | 117.1 µs | ⛳ | ⛳ |
| chunked | 8 | 8192 | 131.5 µs | ⛳ | ⛳ |
| decode+MTP2 | 64 | 64 | 76.1 µs | ⛳ | ⛳ |

> 预期差值（`[推断]`）：**8–14 次 H2D/device op × 3–10 µs ≈ 25–140 µs**。
> 若实测差值远小于此，说明 H2D 入队在 Ascend 上比 CUDA 便宜；远大于此则说明
> 每次都触发了额外同步或 pinned 分配（A5）。

### 6.1b 合成负载的 CPU 负载特征（IPC / 分支 / cache）

`data/model/ipc-summary.jsonl`（`perf stat` 包裹 replica 循环 6 s，绑核 200–203，单线程）：

> 图：`figures/07-ipc-profile.svg`（IPC 与分支失败率的双轴对照）。

| 工况 | µs/step | IPC | 分支指令 M/s | 分支失败率 | cache miss 率 | 频率 |
|---|---:|---:|---:|---:|---:|---:|
| B=1, T=1 | 48.0 | **1.57** | 871 | **3.73 %** | 2.33 % | 2.84 GHz |
| B=64, T=64 | 64.3 | **1.95** | 1039 | 2.51 % | 2.21 % | 2.84 GHz |
| B=256, T=256 | 118.9 | **2.39** | 1237 | 1.51 % | 1.95 % | 2.84 GHz |
| B=8, T=32768 | 322.4 | **2.93** | 1303 | 0.82 % | 2.07 % | 2.84 GHz |

> **这是无卡负载的"指纹"**：小 batch 区是**解释器主导**（低 IPC 1.6、高分支失败 3.7 %），
> 大 token 区是**内存/向量主导**（IPC 2.9、分支失败 0.8 %）。
> 真机 `prepare input` 的 topdown/IPC 应当落在这条趋势线上；若真机 IPC 明显更低
> （<1.2），说明额外的时间花在**驱动/系统调用路径**（H2D 入队、event wait）上。

### 6.1c NUMA 局部性

`data/model/microbench-numa-large.csv`（源数组 512 MB 强制 DRAM 访问；CPU 固定 node 2 的核 200–203）：

> 对照实验说明：同目录的 `microbench-numa.csv` 是 **256 KB 工作集**的版本（落在 L2/L3 内），
> 三种内存放置的差异被噪声淹没（`np.gather` 88–120 µs 无单调性，**不可用于结论**）；
> 有效证据是下表的大工作集版本。这条对照本身也说明：
> **prepare_input 的热数据（数 MB）在 NUMA 上通常不敏感，除非跨 socket 的 DMA/首次触页**。

| 内存 node | `np.gather`(1M 元素) | `torch.index_select` | 相对本地 |
|---|---:|---:|---:|
| node 2（本地 / 同 socket） | 10.83 ms | 9.72 ms | 1.00× |
| node 3（同 socket 远端） | 11.34 ms | 10.59 ms | 1.05–1.09× |
| node 5（跨 socket） | **21.56 ms** | **17.52 ms** | **1.80–1.99×** |

> `[推断]` 注意口径：这是**512 MB 工作集**的极端情形。真实 prepare_input 的热数据
> （`token_ids_cpu` 约 `W×max_model_len×4 B` = 256×8192×4 ≈ 8 MB）多数落在 L3，
> 因此 NUMA 惩罚在有卡环境下应远小于 2×；**但跨 socket 的 pinned 缓冲 DMA
> 与跨 socket 首次触页（`add_request` 的整行写入）仍可能显著**。
> 验证方法：replay harness 上加 `numactl --membind` 的 A/B（§5 A8）。

### 6.2 关键量级对照

⛳ **表 6-2 反事实估算 vs 真机实测**

| 量 | 反事实估计（本文，实测部分已加粗） | 真机实测（待填） | 允许偏差 / 归因 |
|---|---|---|---|
| `c₀`（B=T=1，CPU-only replica） | **48.0–48.7 µs**（实测） | ⛳ | replica 的偏差源见 §6.4 |
| `c₀`（含 H2D/launch 的完整值） | 75–190 µs（外推） | ⛳ | 差值 = `k_h2d_small`×次数 + `k_pin` |
| `a_B`（µs/req，decode，CPU-only） | **0.268**（实测） | ⛳ | 偏大 → 查 `_update_states`/metadata 的逐 req 成本 |
| `a_B`（含 H2D 入队） | 0.30–0.60 | ⛳ | 单次 H2D 入队 3–10 µs 时 |
| `a_T`（ns/token，chunked，CPU-only） | **7.8–8.8**（replica 整段斜率，含 D7 的 `%`）；去 `%` 后 ≈6–7 | ⛳ | 含 `index_select` 与约 6–8 个 O(T) 算子 |
| `T_prepare` @B=64, ISL=128, decode | 64 µs（CPU-only）→ 90–200（含 H2D） | ⛳ | |
| `T_prepare` @B=256, ISL=128, decode | 117 µs（CPU-only）→ 140–260（含 H2D） | ⛳ | |
| `share_step` @B=256（0.8B、短上下文） | 20–60 % | ⛳ | 与 `T_dev` 强相关，须同时报 device 值 |

### 6.3 无卡环境**无法**得到、必须真机采集的系数

| 系数 | 含义 | 采集方式 |
|---|---|---|
| `k_h2d_small` | 单次小 H2D 的 CPU 入队开销（µs/次） | 真机 micro：循环 1000 次 `copy_to_gpu(n)`；`perf` 看 `aclrtMemcpyAsync` 帧 |
| `k_dma_per_B` | 块表 DMA 的等效带宽（GB/s） | B×K 扫描（组 F） |
| `k_launch` | 单次 NPU kernel launch 的 CPU 时间 | graph off vs on 的差；`perf` 看 `aclrtLaunchKernel` 占比 |
| `k_pin` | `Tensor.pin_memory()` 的真实成本（NPU 下） | 复现 `attention_v1.py:330` 的调用并计时 |
| `e, d` | device 侧固定项与每请求斜率 | 组 A 的 `forward` phase wall 拟合 |
| `β, γ` | chunk 的线性/二次项 | 组 C 的 `forward` phase wall 拟合 |
| `σ_rank` | TP 下 rank 间 CPU 时间标准差 | 组 A 的 TP=2/4 对照 |

### 6.4 replica 与真机的**已知结构性偏差**（必须写进保真度报告）

| # | 偏差 | 方向 | 量级 | 补偿方式 |
|---|---|---|---|---|
| D1 | 不含真 H2D DMA 与 device op | 低估 | 25–140 µs/step（`[推断]`） | 用 `k_h2d_small` 实测值回填 |
| D2 | 不含 churn 突变（`condense`/`swap_states`/`add_request`） | 低估 | 高 churn 时可能翻倍 | replay 需要真 `SchedulerOutput` 序列 |
| D3 | `_update_states` 用 8 次数组自拷贝近似 | 量级正确、形状简化 | ±10 µs | replay 用真实现 |
| D4 | 不含 `torch_npu` 的 `Tensor.npu()`/`aclrtMemcpyAsync` 调用栈 | 低估 syscall/驱动占比 | topdown 的 backend/frontend 分布会变 | 真机对比 topdown 各分量 |
| D5 | numpy/torch 版本与容器内可能不同（本文 numpy 2.4.6 / torch 2.9.0+cpu） | ±5–20 % | — | 真机在容器内重跑同一脚本 |
| D6 | 逐 layer 的 metadata 赋值用 dict 覆盖近似（真实现是逐 group × 逐 layer） | 低估（G>1 时更明显） | 小（µs 级） | replay 用真 `_build_attention_metadata` |
| D7 | 为防缓冲越界，`token_indices` 上多做了一次 `%`（约 +1.5–2 ns/token） | **高估** O(T) 项 | ~1.5–2 ns/token | 真机/精确 replica 去掉该取模 |

---

## 7. 无卡复刻：数据依赖判定与 replay 粒度

### 7.1 判定：**需要** record & replay

其中 6 处依赖**请求级 token/length 语义**，无法用"规模参数 + 随机数据"覆盖：

1. `scheduled_spec_decode_tokens` 是否非空 → `num_valid_tokens` 走零拷贝视图还是 O(B) list comp；
2. `_build_attn_state` 的 3 个 `np.all` → 5 种 `AscendAttentionState`，改变 attention kernel 与 mask；
3. `use_async_scheduling and prev_req_id_to_index` → 是否多一次 H2D（S7）；
4. `num_accepted_tokens_event` 是否存在 → 是否**阻塞同步**（S17）；
5. `valid_sampled_token_count_gpu` 路径 → kernel 修正 vs 直接 H2D（S18）；
6. `scheduled_spec_decode_tokens` 长度 → spec 元数据 5×pin+H2D vs 单个 device op（S24）；
7. `is_kv_consumer and req_id in new_schedule_reqs` → draft token 计数分支（S24）；
8. `batch_update_builder.removed` / `batch_update` 非空 → `condense()` 与 `SamplingMetadata` 重建；
9. `_needs_seq_lens_cpu_sync and async_spec_decode_active` → CPU 乐观值修正（S22）。

**量化理由**（而不是"更保险"）：

- 第 2 条决定 attention 分支，进而决定 `_build_attention_metadata` 的**构造路径**与后续 device 时间，
  是"CPU 占比"二阶导数的来源；
- 第 4 条会把 device 时间**吸收进 prepare_input 的 wall**（§2.6 的归因陷阱）；
- 第 8 条是 O(B·S) 与 O(B) 的量级差（§3.4 E1/E2）。

**频率估计**（`[推断]`，待真机验证）：稳态 decode 下这 9 条多数落在同一分支；
但**请求进出**（完成/新进/抢占）每拍都可能改变第 8 条，长跑 1000 步里通常有 100–300 拍是非稳态。
⇒ 只跑"稳态合成"会系统性低估 `_update_states`，也会漏掉 `condense` 的尾延迟。

### 7.2 replay 粒度（分层，避免"全量录"的存储与维护成本）

| 层 | 内容 | 是否必需 | 理由 |
|---|---|---|---|
| L1 | `SchedulerOutput` 的**语义字段**：`num_scheduled_tokens`、`scheduled_spec_decode_tokens`、`finished_req_ids`、`scheduled_new_reqs`（prompt_token_ids / block_ids / sampling_params）、`scheduled_cached_reqs`（new_block_ids / num_computed_tokens）、`num_common_prefix_blocks` | **必需** | 决定 9 条分支中的 6 条 |
| L2 | `InputBatch` 进入前的**粘性数组**：`num_computed_tokens_cpu`、`num_prompt_tokens_cpu`、`num_accepted_tokens_cpu`、`prev_req_id_to_index`、`prev_sampled_token_ids`、活跃 `token_ids_cpu` 前缀、块表行 | **必需** | 决定 `_build_attn_state`、`condense`、`_compute_prev_positions` |
| L3 | flags：`use_async_scheduling`、`num_spec_tokens`、`cudagraph_mode`、`has_gdn`、`use_dcp`、`lora`、`enable_prompt_embeds`、`max_num_reqs`/`max_model_len`/`block_size` | **必需** | 决定编译期分支与缓冲宽度（决定 `c0`、`a_K`） |
| L4 | 逐步的 phase 计时与 device 时间 | 可选 | 用于保真度对照，可独立采集 |

**存储量估算**（`[推断]`）：L1+L2 若按"每步只存活跃前缀 + 增量块表"编码，
1000 步 × B=256 约 **数十 MB~数百 MB** 量级（远小于 perf.data），可以完整进交付包。

### 7.3 "规模参数化"的正确做法

把 B 从 8 提到 256 时，应**复制已有请求的语义**（保持 `attn_state`、spec 结构、
computed/prompt 比例），而不是随机生成 —— 否则 `_build_attn_state` 会跳到另一分支，
导致"参数化"变成了"换了 workload"。同理，ISL 扫描应沿用同一条 prompt 前缀，
只截断/延长 prompt 长度，并同步调整 `num_computed_tokens` 分布。

### 7.4 replay 能否不带卡执行：**能，但有 3 个必须打桩的边界**

| 边界 | 真实现 | 打桩方案 | 风险 |
|---|---|---|---|
| NPU 张量 | `self.x.gpu`、`copy_to_gpu()`、`pin_memory().to()` | 换成 CPU 影子缓冲 + **计数**；把标定好的 `k_h2d_small` 以"忙等/累加"方式加回 | 若只计数不折算，会低估 25–140 µs/step（D1） |
| device op | `reshape_and_cache`、`compute_slot_mapping` kernel、`fill_` | 只保留 launch 计数 | 同上 |
| `torch.npu` 专属 | `torch.npu.Event/Stream/synchronize` | 用 CPU `threading.Event` 或空实现 | S17 的"吸收 device 时间"现象无法复刻 ⇒ 必须靠真机对照 |

**保真度判据**（与 `EXECUTION.md §4` 一致）：topdown 分量差 ≤3 pp、IPC 相对差 ≤10 %、
热点函数 top-20 交集 ≥16/20、单步耗时分布相对差 ≤15 %；
不达标时必须给出**偏差来源归因**（例："缺少 ACL launch，占差值 x%"），而不是只标"失败"。

---

## 8. 预登记的优化假设（供后续验证，不作为结论）

| # | 位置 | 假设 | 预期收益 | 验证 / 风险 |
|---|---|---|---|---|
| O1 | `attention_v1.py:330` | 每拍新建 pinned 缓冲 → 复用 | 省 1 次 `pin_memory` + 1 次 H2D 分配 | 大 B 才显著；需与 `prepare_inputs_event` 协议兼容 |
| O2 | `_calc_spec_decode_metadata:1412–1416` | 5 次 `pin_memory().to()` → 合并/复用 | MTP 下省 5 次小 DMA（本模型估 15–50 µs） | 需确认 5 个张量形状可合并 |
| O3 | `model_runner_v1.py:1039/1049/1073` | 无参 `copy_to_gpu()` 全量 → 按 B/T 切片 | 小 B 省大量 DMA 字节（W=256 时省 250×） | 与 event 覆盖范围耦合（§2.3） |
| O4 | core `1758–1759` | `_compute_prev_positions` O(B) Python → 向量化 | 39 ns/req → ~2 ns/req（B=256 省 10 µs） | 收益小，除非 B>512 |
| O5 | S16 `[requests[r].num_tokens ...]` | 用 `num_tokens_no_spec` 数组替代属性访问 | 128 ns/req → ~0（B=256 省 33 µs，**最大单项**） | 需处理 spec 语义差异 |
| O6 | `attention_v1.py:332–333` | 每拍 `.tolist()` → 预分配 list 复用 | 16 ns/req（B=256 省 4 µs） | 取决于 FIA 是否接受 tensor 形参 |

> **优化收益上限的算法**：若 `T_cpu > T_dev`（CPU 受限），收益 ≈ `ΔT_prepare` 直接映射到吞吐；
> 若 `T_cpu < T_dev`（被掩盖），收益 ≈ 0（只降 CPU 占用率）。
> 因此每条优化都必须先回答："目标工况下，`T_cpu` 是否已越过 `B_cpu` 阈值？"

---

## 9. 未决问题与风险

1. `[待实测]` `T_sched + T_rpc` 的量级。若它在 100 µs 以上，会显著压缩 §4.1 的 `B_cpu`，
   使"prepare_input 不是最大项、但整体 CPU 受限"成为常见现象 —— **engine core 侧必须同时打点**。
2. `[待实测]` Ascend 上 `torch.npu.Event` 的 `record/synchronize` 语义是否与 CUDA 一致，
   特别是 `prepare_inputs_event` 的 record 点覆盖范围（§2.3）。判据：CPU 领先量是否被限制在 ~1 拍。
3. `[推断]` 空 batch step（`total_num_scheduled_tokens==0`）在 DP/EPLB 场景下会以 `_dummy_run(1)`
   形式进入 prepare_input scope（ascend `1895–1910`）—— 统计必须分层，否则稀释占比。
4. `[推断]` `max_num_reqs` 很大而实际 B 很小时存在"缓冲宽度税"（尾部 `fill_`、全量 `copy_to_gpu`）。
   本文把它归入 `c₀`，但严格说它随 `W` 线性 ⇒ **建议增加一维实验**（改 `--max-num-seqs` 的 A/B）。
5. `[已证实-部分]` 本文 CPU 斜率来自 a3-22 微基准；容器内 python/numpy/torch 版本不同
   （本文 numpy 2.4.6、torch 2.9.0+cpu），偏差 5–20 %，真机回填时以容器内数值为准。
6. `[推断]` TP 下每个 rank 都跑同一份 CPU 逻辑 ⇒ CPU 时间随 TP **不下降**，
   而 device 时间随 TP 下降 ⇒ **TP 越大，CPU 越容易成为瓶颈**（与"加卡更快"的直觉相反，需实验证实）。
