# prepare_input 数据层（record & replay / shape 合成）交接报告

> owner: `trace_schema`（父任务 `replay_harness`）｜更新：2026-09-24（Asia/Shanghai）
> 运行环境：`a3-22` + `harness/scripts/pi-docker.sh`（无卡容器，python 3.12.13 /
> torch 2.10.0+cpu / vLLM 0.26.0 commit `568afb3` / vllm-ascend 0.26.0rc1
> commit `f2f74a1`），CPU 绑核 `200-215`，未使用 120-159，未挂 NPU。

## 0. 结论速览

1. **数据通路已交付且已跑通**：record 钩子（环境变量开关、纯 stdlib、不改 vLLM
   源码）→ JSONL（`StepRecord` schema v1）→ `load_jsonl` → **真实**
   `vllm.v1.core.sched.output.SchedulerOutput` 对象 → 统一喂给
   `pi_harness.runner.replay.PrepareInputReplay._one_step_with`
   （即真实 `NPUModelRunner._update_states` + `_prepare_inputs`）。
2. **形状合成优先方案成立**：`synth_trace()` 默认用**真实的
   `vllm.v1.core.sched.scheduler.Scheduler` + `KVCacheManager`** 在 CPU 上驱动，
   不是自己写调度器；形状（准入、chunked prefill 切分、prefix 命中、
   block 分配、MTP 草稿位）天然与真机同源。
3. **A/B（回答"是否必须 record&replay"）**：在 4 个代表性负载上，把
   token id / block id / 请求 ID 做**同形状重排**（`shuffled`）后，与未重排的
   `synth` 相比，每步 `prepare_input` 的线程 CPU 时间配对中位数比为
   **1.0004–1.0098（≤1.0%）**，p90 偏差 **1.3%–3.2%**，同组测量噪声中位
   0.03%–1.9%（详见 §6）→ **数据内容（取值）对 prepare_input 成本无影响；
   shape 级合成足够，不必为"数据内容"做真机 record&replay**。
   但 `real` 组（真机 trace）**尚未到位**：它检验的是"合成形状 vs 真机形状"，
   这条链路已用"用 synth trace 冒充 real"的 0 差异对照验证过接口（§6.4）。
4. **判定统计量说明**：本机是共享宿主，逐步 wall 时间的 p90 噪声可达 16%，
   因此判定用 **per-step 线程 CPU 时间**（`time.thread_time`，免疫抢占）+
   **配对比值**（每一步自己做对照）。2% 阈值在噪声边缘，故额外要求
   "观测偏差 > 同组 even/odd 半集噪声地板"，否则判 `inconclusive_noise`（§6.3）。

## 1. 交付物清单

| 文件 | 行数 | 作用 |
|---|---|---|
| `harness/pi_harness/trace/capture.py` | 440 | record 钩子：monkeypatch `NPUModelRunner.execute_model/sample_tokens/_update_states/_prepare_inputs`，env 开关，内存缓冲 + 定期 flush |
| `harness/pi_harness/trace/sitecustomize.py` | 44 | 零改动注入：把本目录放 `PYTHONPATH`，`PI_CAPTURE` 设置时自动装钩子（worker 进程同样生效） |
| `harness/pi_harness/trace/replay.py` | 292 | `load_jsonl` → 真实 `SchedulerOutput` / `ModelRunnerOutput`（set/tuple/None 还原，类型对齐），附 `--show` 校验 CLI |
| `harness/pi_harness/workload/schema.py` | 706 | `StepRecord` v1、JSONL 读写、字段级编码/解码、phase 判定、shape 签名 |
| `harness/pi_harness/workload/generate.py` | 509 | `synth_trace(cfg)` + `--` CLI（全部形状参数）、`shuffle_trace()`（同形状改值） |
| `harness/pi_harness/workload/synth_real.py` | 386 | 用真 Scheduler + KVCacheManager 在 CPU 上生成 trace（默认路径） |
| `harness/pi_harness/workload/synth_pure.py` | 363 | 纯 Python 退化合成（vLLM 起不来时；`meta.engine="pure"`） |
| `harness/pi_harness/workload/worker_state.py` | 274 | 仿真 worker 侧 `InputBatch` 行序/计数/block 列表（真实 trace 里 `input_batch` 快照的来源） |
| `harness/pi_harness/workload/runner_exec.py` | 261 | **A/B 统一执行器**：逐 step 走 `PrepareInputReplay._one_step_with`，用 `SubStepTimer` + 线程 CPU 探针计时 |
| `harness/pi_harness/workload/ab.py` | 1050 | A/B 主程序：三组同形状对比、配对统计、硬编码判定阈值、CSV/JSON + manifest |
| `harness/pi_harness/workload/local_exec.py` | 366 | 纯 numpy 的 CPU 镜像（**仅调试/交叉验证，不作为结论依据**） |
| `harness/pi_harness/workload/crosscheck_worker_state.py` | 137 | 把 trace 喂给真实 `NPUModelRunner._update_states`，逐字段比对仿真 `input_batch` |
| `harness/pi_harness/workload/selftest_local.py` | 327 | 无卡自检（合成/schema/打乱/执行器/捕获钩子/解码） |
| `harness/scripts/ab_run.sh` | 123 | 一键入口：造 trace + 跑 A/B + 落盘 |

产物（本地与 `a3-22` 同名）：`data/harness/ab_*.csv|json`、
`data/harness/trace_synth_*_manifest.json`、
`data/harness/xcheck_*.json`；trace jsonl 留在 `a3-22`（可由 manifest 里的参数 +
脚本 sha256 复现）。

目录里 `ab_legacy_*.csv|json`、`trace_synth_{v1,r32,r32b,r32c,big,prefill8k,mtp}*`
是**早期用 `local_exec.py` 镜像/早期无 drain step 的版本**跑出来的，仅作留档，
**不作为结论依据**（`ab_<tag>.json` 里 `executor.executor == "local"`）。

## 2. Schema v1

一行一个 `StepRecord`（`harness/pi_harness/workload/schema.py`）：

| 字段 | 类型 | 内容 |
|---|---|---|
| `step_idx` | int | step 序号（0 起） |
| `phase` | str | `prefill` / `decode` / `mixed` / `spec-decode` / `drain` |
| `scheduler_output` | dict | `SchedulerOutput` 字段级 JSON 表示（下表） |
| `req_states` | list[dict] | worker 侧每请求：`req_id / prompt_token_ids / num_prompt_tokens / num_computed_tokens / block_ids / num_output_tokens / spec_len` |
| `model_output` | dict\|None | 采样结果（`req_ids / req_id_to_index / sampled_token_ids`），可直接喂 `scheduler.update_from_output` |
| `t_prepare_input_us` | float\|None | 真机捕获时 = `_update_states` + `_prepare_inputs` 之和（µs） |
| `input_batch` | dict\|None | `req_ids / num_computed_tokens_cpu / num_prompt_tokens_cpu / num_tokens_no_spec / block_table[...] / num_blocks_per_row / token_ids_cpu_shape / spec_token_ids` |
| `timings` | dict\|None | 捕获时的子步骤耗时（`update_states_us` / `prepare_inputs_us` / `execute_model_us`） |
| `meta` | dict | `source`(capture/synth/shuffled) / `synth` / `engine`(real/pure) / `params` / `schema_version` / `drain` |

`scheduler_output` 覆盖的真实字段（与 `vllm/v1/core/sched/output.py` 一一对应）：
`scheduled_new_reqs[]`（`req_id / prompt_token_ids / mm_features / sampling_params /
pooling_params / block_ids / num_computed_tokens / lora_request / prompt_embeds_shape /
prompt_is_token_ids / prefill_token_ids`）、`scheduled_cached_reqs`（`req_ids /
resumed_req_ids / new_token_ids / all_token_ids / new_block_ids /
num_computed_tokens / num_output_tokens`）、`num_scheduled_tokens`、
`total_num_scheduled_tokens`、`scheduled_spec_decode_tokens`、
`scheduled_encoder_inputs`、`num_common_prefix_blocks`、`finished_req_ids`、
`free_encoder_mm_hashes`、`scheduled_encoder_input_stats`、`preempted_req_ids`、
`has_structured_output_requests`、`pending_structured_output_tokens`、
`num_invalid_spec_tokens`、`new_block_ids_to_zero`、`kv_cache_block_copies`
（`KVCacheBlockCopy.src/dst_block_id`）、`num_spec_tokens_to_schedule`。

两条刻意约定（`schema.py` 顶部有说明）：

* `req_states[i].prompt_token_ids` 只在请求**进入 worker batch 的那一步**写全，
  之后为 `null`（完整 prompt 始终存在于该步
  `scheduled_new_reqs[].prompt_token_ids`）——避免 2k 长 prompt 在几百步里重复几百次。
* `input_batch.token_ids_cpu_shape` 只记形状；取值可由 prompt + 采样序列重建。

`drain` step（`total_num_scheduled_tokens == 0` 且 `finished_req_ids` 非空）是
真实引擎在"还有请求要上报结束"时必然执行的一步（`Scheduler.has_requests()`
在 `finished_req_ids` 非空时为真），worker 侧只跑 `_update_states` 就返回。
没有它，replay 结束后 worker batch 里会残留已结束请求，多个 run 无法复用同一
runner（§4/§6.1）。

## 3. record 侧（真机捕获）

```bash
# 真机（chip3）启动服务时加两个环境变量即可，不改任何源码：
PYTHONPATH=/work/harness:/work/harness/pi_harness/trace \
PI_CAPTURE=/work/data/harness/trace_real.jsonl \
PI_CAPTURE_MAX_STEPS=2000 \
python -m vllm.entrypoints.openai.api_server ...     # 或你们的启动脚本
```

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `PI_CAPTURE` | 未设 | 输出 JSONL 路径；**未设置时钩子完全不生效** |
| `PI_CAPTURE_TARGET` | `vllm_ascend.worker.model_runner_v1:NPUModelRunner` | 要 patch 的类 |
| `PI_CAPTURE_MAX_STEPS` | 0（不限） | 最多捕获多少步，达到后钩子变透明 |
| `PI_CAPTURE_FLUSH_EVERY` | 64 | 每多少步落盘一次（另有 `atexit` 兜底 flush） |
| `PI_CAPTURE_TIMING` | 1 | 是否再 patch `_update_states` / `_prepare_inputs` 计时 |
| `PI_CAPTURE_INPUT_BATCH` | 1 | 是否记录 `input_batch` 快照 |
| `PI_CAPTURE_NOTE` | 空 | 写进每行 `meta.note`（比如实验编号） |

设计要点：

* **零源码改动**：`sitecustomize.py` 在解释器启动时自动 `install()`，vLLM 起的
  worker 子进程同样命中；import 期只用 stdlib，不 import torch/vllm。
* **低开销**：先累积到内存 list，`flush_every` 步或进程退出时追加写 JSONL；
  记完 `MAX_STEPS` 后钩子直接旁路（`if not enabled: return orig(...)`）。
* **不依赖真机也能自检**：`selftest_local.py` 用一个 stub runner 验证钩子
  （缓冲、`MAX_STEPS`、`sample_tokens` 拆分采样路径、子步骤计时、req_states 顺序），
  无需 vLLM/NPU。
* 未支持的字段（`prompt_embeds` 的体数据、connector metadata）以
  `__unsupported__` / `*_shape` 明确标注，不会静默丢。

## 4. replay 侧

* `load_jsonl()` → `list[StepRecord]`；`to_scheduler_output()` →
  **真实** `SchedulerOutput`：`finished_req_ids` 还原成 `set`、
  `CachedRequestData.resumed_req_ids` 还原成 `set`、
  `NewRequestData.block_ids` / `new_block_ids` 还原成 `tuple[list[int], ...]`、
  `kv_cache_block_copies` 还原成 `KVCacheBlockCopy`、
  `sampling_params` 用 `inspect.signature` 过滤后构造真实 `SamplingParams`；
  版本不支持的字段名会被过滤并写进 notes（不静默丢失）。
* 校验：容器内 `python -m pi_harness.trace.replay <trace.jsonl> --show 5` 打印每步
  `new/cached/tok/finished/spec/common_prefix`，并汇总 decode notes
  （当前 synth/捕获 trace 的 notes 数为 0）。
* `triton_launch_us` 按父任务要求**恒为 0**（未注入 Triton Python launch 开销）。

## 5. synth 侧（形状合成）

`synth_trace(cfg)` → `list[StepRecord]`，与 record 完全同 schema。默认
`engine="real"`：CPU 上构造 stand-in `VllmConfig` + `FullAttentionSpec` 的
`KVCacheConfig`，实例化**真实** `Scheduler`（+ 真 `KVCacheManager`、
真 prefix-cache 哈希 `sha256`），用 `Scheduler.schedule()` 出 `SchedulerOutput`，
用确定性的假"模型"（每请求 `1 + 接受草稿数` 个采样 token，取值按请求区分，
避免不同请求生成同 token 造成假共享前缀）驱动
`Scheduler.update_from_output()` / `update_draft_token_ids()`；请求结束的
drain step 也照实记录。

参数（CLI 同名）：`--batch --isl --osl --block-size --max-model-len --spec-k
--prefix-hit-ratio --chunk-size --arrival {simultaneous,poisson,staircase}
--steps --seed [--engine real|pure] [--shuffle-values] [--max-num-reqs]
[--num-blocks] [--vocab-size] [--token-id-base] [--prefix-len]
[--arrival-rate/--arrival-group/--arrival-period] [--accept-ratio]`。

形状真实性证据（都是真 scheduler 算出来的，不是拟合）：

| 负载 | 步数 | phase 分布 | 总 token | 步均请求数 | 最大 `num_common_prefix_blocks` |
|---|---|---|---|---|---|
| batch32/isl2048/osl128/poisson/prefix0.5/chunk2048 | 262 | prefill 1 / mixed 39 / decode 221 / drain 1 | 49120 | 15.6 | 17（prefix=0 时 0） |
| batch8/isl4096/osl64/simultaneous/prefix0.25/chunk2048 | 79 | prefill 3 / mixed 14 / decode 61 / drain 1 | 33272 | 6.5 | 33（2048 block 对齐）|
| batch96/isl128/osl192/poisson/prefix0.5/chunk8192 | 516 | prefill 1 / mixed 90 / decode 424 / drain 1 | 30624 | 35.7 | 3 |
| batch16/isl2048/osl96/**spec-k 2**/poisson/chunk4096 | 106 | prefill 1 / **spec-decode 104** / drain 1 | 27958 | 7.6 | 17 |

`engine="pure"` 只是 vLLM 起不来时的退化路径（`meta.engine="pure"`），报告里
所有 A/B 数字都不来自它。

可复现性：上述 4 条 trace 用同一命令重跑，逐字节一致
（`sha256`：mix `b042036b…`、prefill `cd3ed584…`、decodeb128 `9a699804…`、
mtp2big `f8683fdb…`），说明形状合成是确定性的（`--seed` 固定）。

## 6. A/B：是否必须 record&replay

### 6.1 口径（三组必须走同一条下游）

* 三组输入都先落到**同一 schema** 的 JSONL，再 `load_jsonl` → 真实
  `SchedulerOutput`，逐 step 调用
  `PrepareInputReplay._one_step_with()`（内部就是真实
  `NPUModelRunner._update_states` + `_prepare_inputs`）。drain step 只跑
  `_update_states`（与真实 worker 行为一致）。
* 计时：`pi_harness.runner.timer.SubStepTimer`（由 `PrepareInputReplay.build()`
  装到 worker 类上）读每步 `_prepare_inputs` / `_update_states` 的 inclusive 时间；
  另外在**实例**上加了线程 CPU 探针（不改类、不改源码），得到
  `prepare_inputs_cpu_us`。
* 统计：每步在 R 次重复里取 **min**（CPU 抢占只会加时间，min 最接近真实成本），
  三组之间做**逐步配对比值**（同一步三组形状完全相同，各步自成对照）；
  组顺序每轮旋转，避免 cache/频率漂移偏向某一组。
* 判定阈值（写死在 `ab.py`）：

```
DATA_CONTENT_TOL   = 0.02   # 配对中位比值偏离 1 超过 2% 才算“内容相关”
DIST_P90_TOL       = 0.05   # 配对 |dev| 的 p90 落在 5% 内算“分布同形”
DIST_TOL           = 0.10   # KS 仅作并列报告（非判定）
附加条件: 观测偏差必须 > 同组 even/odd 半集噪声地板，否则 inconclusive_noise
```

### 6.2 结果（全部为真实 worker 代码路径）

| 负载（`--tag`） | step/run | repeats | shuffled p50 / p90 (µs) | synth p50 / p90 (µs) | 配对中位比 | p90 偏差 | 同组噪声(中位/p90) | 判定 |
|---|---|---|---|---|---|---|---|---|
| `mix` batch32 isl2048 osl128 poisson prefix0.5 | 262 | 5 | 499.6 / 576.2 | 495.9 / 568.5 | **1.0098** | 2.53% | 0.77% / 4.79% | `content_insensitive` |
| `prefill` batch8 isl4096 osl64 simultaneous prefix0.25 | 79 | 5 | 500.1 / 633.9 | 493.1 / 634.3 | **1.0030** | 2.90% | 1.13% / 5.57% | `content_insensitive` |
| `decodeb128` batch96 isl128 osl192 poisson prefix0.5 | 516 | 5 | 529.9 / 578.2 | 527.8 / 580.3 | **1.0014** | 3.15% | 1.85% / 7.09% | `content_insensitive` |
| `mtp2big` batch16 isl2048 osl96 **spec-k=2** | 106 | 8 | 664.1 / 758.3 | 660.8 / 755.4 | **1.0004** | 1.33% | 0.03% / 2.20% | `content_insensitive` |
| `mtp2_short`（反例：步少+重复少） | 71 | 4 | 667.1 / 818.3 | 759.3 / 1166.5 | 0.9611 | 6.96% | 7.61% / 15.95% | `inconclusive_noise`（噪声 > 效应） |

按 phase 分开的 p50/p90（`mix`，单位 µs）：

| phase | shuffled p50 / p90 | synth p50 / p90 | 步数/run |
|---|---|---|---|
| prefill（chunked 首块） | 710.9 / 789.4 | 704.2 / 757.4 | 1 |
| mixed（chunked prefill + decode 混批） | 584.3 / 627.0 | 578.1 / 609.7 | 39 |
| decode | 496.4 / 524.0 | 492.6 / 515.0 | 221 |
| drain | 0（不跑 `_prepare_inputs`） | 0 | 1 |

（其余 tag 的同结构数字在 `data/harness/ab_<tag>.json` 的
`groups.*.stats_by_phase` 与 `comparisons_by_phase` 里。）

### 6.3 噪声地板与 0 差异对照

* `nullrunner`：把**同一条 synth trace** 同时当 `real` 和 `synth` 跑三组
  （`--real data/harness/trace_synth_nullrunner.jsonl`），实测
  `shuffled vs real` 配对中位比 **1.0168**、`synth vs real` **1.0155**、
  该组噪声地板 **6.2%**（quality=poor）。结论：本 harness 的自一致性在
  **1–2%** 量级，2% 判定阈值处在噪声边缘 ⇒ 每次 A/B 必须同时报告
  noise floor，并用 ≥5 次重复（否则会像 `mtp2_short` 那样判 `inconclusive`）。
* 该对照同时证明：`real`/`shuffled`/`synth` 三组接线正确、形状零差异
  （`mismatch_steps = 0`），真机 trace 到位后只需把
  `--real <trace_real.jsonl>` 指向它即可。

### 6.4 结论

> **对已测的 4 个负载：数据内容（token id / block id / 请求 ID 取值）对
> prepare_input 每步成本的影响 ≤1.0%（配对中位比 1.0004–1.0098），p90 偏差
> ≤3.2%，且都落在同组测量噪声量级 ⇒ shape 级合成足够，不需要为"数据内容"
> 做真机 record&replay。**
>
> 仍未回答的一半：`synth` 的**形状**是否等于**真机**的形状（需要 service_bringup
> 的 trace_real.jsonl）。形状完全由真 Scheduler 生成（§5），加上
> `crosscheck_worker_state` 的一致性（§7），预期差异很小，但必须以
> `real` 组的 p50/p90 与 `shape_checks` 为准，未验证前不下结论。

### 6.5 真机 trace 到位后补 `real` 组

```bash
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash harness/scripts/pi-docker.sh "bash harness/scripts/ab_run.sh \
    --tag real --real data/harness/trace_real.jsonl \
    --synth data/harness/trace_synth_mix.jsonl \
    --batch 32 --isl 2048 --osl 128 --max-num-reqs 64 --chunk-size 2048 \
    --repeats 5 --warmup 1"'
```

注意：若真机 trace 是**截断**的（`PI_CAPTURE_MAX_STEPS` 提前停），它不会以
drain step 结束，`ab.py` 会自动切到 `--runner-reuse never`（每次 run 重建 runner，
慢但状态不串），并在 `executor.runner_reuse` 里标注。

## 7. 保真度与已知偏差

| 项 | 状态 | 方向/影响 |
|---|---|---|
| 下游代码路径 | **真实** `_update_states`/`_prepare_inputs`（runner 包，`AttrAudit` 未装配属性为空） | 无偏差 |
| 合成形状来源 | **真** `Scheduler` + `KVCacheManager` | 无偏差（除下述 stand-in） |
| `synth` 的模型输出 | 假的采样 token（每个请求取值不同，`1 + 接受草稿数`） | 只影响被调度形状，与 CPU 成本无关 |
| Triton Python launch 开销 | **未注入**（`triton_launch_us = 0`） | 低估真实每步固定开销（真机有 ~µs 级 launch）；已按要求登记 |
| `pin_memory()` | 无卡容器里被置为 no-op（`shim` 未覆盖 `Tensor.pin_memory` → `torch_npu` 钩子触发 `aclInit` 507008，MTP 路径 `_calc_spec_decode_metadata` 会踩到） | 只影响真实 H2D 传输，不影响本次测的 CPU 侧；`PI_PIN_MEMORY=1` 可恢复真实行为 |
| `input_batch` 快照（synth） | 由 `worker_state.py` 仿真 | 已用 `crosscheck_worker_state` 对真实 `NPUInputBatch` 逐字段校验：`mix` trace **262/262 步全等**（req 顺序、`num_computed_tokens_cpu`、`num_prompt_tokens`、block 行宽），`r32` 80/80、`mtp` 60/60 |
| drain step | 记录并只跑 `_update_states` | 与真实 worker 行为一致 |
| 设备/传输时间 | 完全不建模 | 本任务只关心 CPU 侧 prepare_input |
| `local_exec.py` | 纯 numpy 镜像，**不作结论** | 早期用它跑的数已改名 `ab_legacy_*` 保留 |

## 8. 尚未验证 / 下一步

1. `real` 组未跑（等待真机 trace）。届时需要：形状对比
   （`shape_checks.real_vs_synth.mismatch_steps`）+ 配对 p50/p90。
2. `shim` 的 `pin_memory` 缺口建议由 probe_devdeps 补（现在是本模块自己绕开）。
3. Triton launch 成本未注入，真机对照时需在 `docs/06` 的偏差表里保留。
4. 多模态（`mm_features`/`prompt_embeds`）与 connector metadata 只做了结构记录，
   本任务负载未覆盖，未做端到端验证。
5. `arrival=poisson/staircase` 的到达过程是"引擎步"粒度，不是真实毫秒级到达；
   A/B 里三组用的是同一份到达计划，因此不影响对比结论。

## 9. 复现

```bash
# 0) 无卡自检（含捕获钩子 stub 测试）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash harness/scripts/pi-docker.sh "bash harness/scripts/ab_run.sh --tag smoke --selftest \
    --batch 8 --isl 1024 --osl 32 --chunk-size 2048"'

# 1) 合成 trace（真 scheduler）+ 三组 A/B（synth / shuffled / real[可选]）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash harness/scripts/pi-docker.sh "bash harness/scripts/ab_run.sh --tag mix \
    --batch 32 --isl 2048 --osl 128 --arrival poisson --prefix-hit-ratio 0.5 \
    --chunk-size 2048 --max-num-reqs 64 --repeats 5 --warmup 1"'

# 2) 只校验某条 trace 能否解码成真实 SchedulerOutput
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash harness/scripts/pi-docker.sh "USER=pi LOGNAME=pi \
    python -m pi_harness.trace.replay data/harness/trace_real.jsonl --show 5"'

# 3) 仿真 worker 状态 vs 真实 _update_states 的一致性
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash harness/scripts/pi-docker.sh "USER=pi LOGNAME=pi \
    python -m pi_harness.workload.crosscheck_worker_state \
      --trace data/harness/trace_synth_mix.jsonl --out data/harness/xcheck_mix.json"'
```

产出 manifest：每条 trace 有 `trace_synth_<tag>_manifest.json`（参数、时间戳、
`generate.py/schema.py/synth_real.py/synth_pure.py/worker_state.py` 的 sha256、
形状汇总），每个 A/B 结果 JSON 内含 `manifest`（参数、重复次数、脚本 sha256、
平台、vLLM 版本）与 `executor`（执行器/复用策略/triton 说明）。本报告引用的
全部脚本（含 `pi_harness/runner/**` 下游路径、`pi_harness/shim/**`）与 trace 的
sha256 汇总在 `data/harness/ab_scripts_sha256.json`（21 个文件 + 15 条 trace）。
