# 06 无卡模拟负载：设计、保真度与已知偏差

> 归属：`agents/replay_harness/`（代码在 `harness/`）
> 前置阅读：[01 代码逻辑](01-prepare-input-code-logic.md)、[02 复杂度与因素](02-complexity-and-factors.md)、
> [04 profiling 方法学](04-profiling-methodology.md)
> 数据：`data/harness/hw64_sweep_*.csv`、`data/harness/devdeps_*.{json,md}`、`data/harness/micro_*.csv`

---

## 0. 一句话结论

**可以做到"不挂 NPU 卡也能跑真实 `prepare_input` 代码路径"**——因为 `torch_npu` /
`vllm` / `vllm_ascend` 本身在无卡容器里都能 import，**只有设备操作会崩**；把设备操作
按三层 shim 掉之后，`_update_states + _prepare_inputs` 能逐行原样执行。
需要额外建模的只有 **Triton kernel launch 的 Python 侧开销** 与 **H2D/D2H DMA**；
这两项本 harness 都**按量记账**，可以用实测值补回。

---

## 1. 运行位置与硬约束

| 项 | 值 | 理由 |
|---|---|---|
| 主机 | a3-22（Kunpeng 920B，640 逻辑核 / 4 socket / 8 NUMA / 2.9 GHz） | topdown / IPC 与微架构强相关，必须在目标机跑 |
| 容器 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler` | 与真机同镜像（python 3.12.13 / torch 2.10.0+cpu / triton-ascend） |
| 设备 | **不挂任何 `/dev/davinci*`** | 真机测量由 `service_bringup` 独占 chip3 |
| 绑核 | `--cpuset-cpus 200-215`（NUMA node2），线程 ≤16 | 绝不占用真机实验切片 `120-159` |
| 网络 | `--network none` | 排除外部依赖与网络抖动 |
| 身份隔离 | `--user $(id -u):$(id -g)` + `-e USER/-e LOGNAME` | 容器内 uid 1003 无 passwd 条目，会炸 `getpass.getuser()`（见 §3.4） |

入口只有一个：`harness/scripts/pi-docker.sh`（约束写死在脚本里，不可绕过）。

---

---

## 1.5 ⚠️ 保真度第一约束：必须匹配真机的**引擎启动参数**

这不是"锦上添花"，是**能不能对上真机的前提**。真机 launcher
（`scripts/launch_phase_service.sh` 默认值）跑的是：

```bash
vllm serve <model> --tensor-parallel-size 1 --max-num-seqs 8 \
  --max-model-len 2048 --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching --trust-remote-code --dtype bfloat16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":false}'
# async scheduling: vLLM 0.26 这个 build 里默认 ON
# 负载：1 请求 / prompt 128 tokens / max_tokens 64
```

harness 用 `--preset` 把这个口径固定下来（默认就是真机口径）：

| preset | max_model_len | max_num_reqs | max_num_batched_tokens | prefix cache | 用途 |
|---|---|---|---|---|---|
| **`realmachine`（默认）** | 2048 | 8 | 2048 | 关 | **与真机做一致性对照** |
| `stress` | 32768 | 64 | 16384 | 开 | 放大并发/预算看斜率 |
| `modelmax` | 模型自身（262144） | 64 | 16384 | 开 | 只研究 `max_model_len` 这个维度 |

**为什么这件事致命**——`InputBatch.token_ids_cpu_tensor` 的形状是
`(max_num_reqs, max_model_len)`，而 `_prepare_inputs` 每步都要
`torch.index_select(token_ids_cpu_tensor.flatten(), 0, token_indices_tensor, ...)`：

| 口径 | tensor 形状 | 体积 | cache 行为 |
|---|---|---|---|
| 真机 `realmachine` | (8, 2048) int32 | **64 KB** | 常驻 L1/L2 |
| 误用 `modelmax` | (64, 262144) int32 | **67 MB** | **每步从 DRAM 拉** |

差 1000 倍的工作集，topdown / IPC / 热点不可能对得上。同理
`max_num_blocks_per_req` = 16（真机）vs 2048（误用），block table 从 4 KB 变 512 KB，
直接改变每步 `commit_block_table()` 的搬运量。

> **规则：任何"与真机比 topdown / IPC / 热点"的采集都必须用 `--preset realmachine`。**
> 用错口径时数字看起来"也合理"，但归因全错——这是本次分析里最容易踩的坑。

## 2. 无卡可行性的实测边界

### 2.1 import 梯度（实测，证据 `data/harness/devdeps_imports*.json`）

| 步骤 | 无卡结果 |
|---|---|
| `import torch`（`TORCH_DEVICE_BACKEND_AUTOLOAD=0`） | ✅ `2.10.0+cpu`，`hasattr(torch,"npu") == False` |
| `import torch_npu` | ✅ 成功（只有 warning） |
| `torch.npu.is_available()` / `device_count()` | ✅ `False` / `0`（不崩） |
| `import vllm` | ✅（plugin `ascend` 被激活，`npu-smi` 探测失败但不致命） |
| `import vllm_ascend` | ✅ |
| `import vllm_ascend.ops` | ✅ |
| `from vllm_ascend.worker.model_runner_v1 import NPUModelRunner` | ⚠️ **必须先 import `vllm_ascend.ops`**，否则撞 `device_op` ↔ `ops.fused_moe` 循环导入 |
| 任何真实设备操作 | ❌ `RuntimeError: ... aclInit, error code is 507008` |

**结论：禁用点不在 import 层，而在设备操作层。** 这就是 shim 只需要覆盖
"设备操作"而不需要伪造整个 `torch_npu` 的原因。

### 2.2 三层 shim（`harness/pi_harness/shim/__init__.py`）

| 层 | 触发点（实测） | 替换 |
|---|---|---|
| 1 `torch.npu.*` | `current_stream()` / `synchronize()` / `Event()` / `zeros(device="npu")` → `aclInit 507008` | `CpuShimStream` / `CpuShimEvent` / no-op 显存查询 |
| 2 tensor & factory | `.npu()` / `.to("npu")` / `torch.zeros(device="npu")` / `pin_memory=True` | device npu→cpu、`pin_memory`→False，并**按字节记账** |
| 3 triton-ascend | `NPUUtils.get_arch()` → `SystemError`；`_compute_slot_mapping_kernel[...]` → `get_current_stream` 崩 | 替换 `get_arch`/`get_aicore_num`；slot-mapping kernel 换**向量化 numpy 等价实现** |

必须用**子类**而不是函数替换 `torch.Generator`：transformers 里有
`torch.Generator | None` 这种**在 import 期求值**的注解，函数替换会
`TypeError: unsupported operand type(s) for |`。

### 2.3 非设备的第四个坑

容器里 uid 1003 没有 passwd 条目 → `getpass.getuser()` 抛 `KeyError` →
torch inductor `default_cache_dir()` 崩。必须在 `docker run` 里给
`-e USER=... -e LOGNAME=... -e TORCHINDUCTOR_CACHE_DIR=/tmp/... -e TRITON_CACHE_DIR=/tmp/...`。
这个坑与"有没有卡"无关，但会让人误判为"无卡跑不起来"。

---

## 3. 负载构造：真实代码路径 replay（P1）

### 3.1 为什么不重写

`harness` **不抄写** `_update_states` / `_prepare_inputs`，而是 `object.__new__` 出真实的
`NPUModelRunner` 实例、手工装配该路径会访问到的属性，然后**调用真实方法**。收益：

* 任何一个 `if` 分支的变化都真实反映在耗时里；
* 上游升级时只要装配仍成立，harness 跟着走，不需要重写；
* "哪些分支真的被走到了"可以被审计（见 §3.3）。

### 3.2 装配清单（`harness/pi_harness/runner/build.py`）

prepare_input 路径上会被访问、因而必须装配的东西分四类：

1. **配置对象**：`vllm_config` / `model_config` / `cache_config` / `parallel_config` /
   `scheduler_config` / `compilation_config` / `lora_config`（`None`）
2. **持久缓冲区**：`input_ids`、`positions`、`query_start_loc`、`seq_lens`、
   `optimistic_seq_lens_cpu`、`num_computed_tokens`、`req_indices`、`prev_positions`、
   `num_scheduled_tokens`、`discard_request_mask`、`num_accepted_tokens`、
   `arange_np` / `query_pos` / `_arange_scratch` 等（逐条照抄 `GPUModelRunner.__init__`
   的对应片段）
3. **数据对象**：真实 `NPUInputBatch`（含 `MultiGroupBlockTable`）、
   `LateInteractionRunner`、`requests` dict
4. **真机由 worker init 完成、harness 手工补装的**：
   * `kv_cache_config`（`_may_reorder_batch` 读 `kv_cache_groups` 长度）
   * `init_ascend_config(vllm_config)`（`_prepare_inputs` 末尾 `lmhead_tp_enable()` 读它）
   * 分布式 group 替身（`get_pp_group()` / `get_dcp_group()` 在无进程组时是 assert）

**关于模型 config**：`ModelConfig` 会 inspect 模型类。Qwen3.5 的类会拉起
flash-linear-attention / triton 等重依赖。harness 把真模型 `config.json`
**摊平成等价最小 config**（`harness/pi_harness/runner/modelcfg.py`），
保留所有影响 prepare_input 的数值（`max_model_len`、`vocab_size`、`dtype`、
`layer_types`→`_has_gdn`），只把 `architectures` 换成一个轻量类。
`max_model_len` 是这里最关键的字段：它决定 `token_ids_cpu` 宽度 →
`max_num_blocks_per_req` → 每步 `commit_block_table()` 的搬运字节数。

### 3.3 反静默跳过审计

| 手段 | 作用 | 当前状态 |
|---|---|---|
| `AttrAudit` | 记录"没装配却被访问"的属性 | **空**（没有遗漏） |
| `meta.readonly_attrs_skipped` | 记录只读 property 被跳过的赋值 | **空** |
| `shim.report()["warnings"]` | shim 安装过程中的降级 | 仅 `pin_memory` 相关 |
| `UNSHIMMED_DEVICE_OP` | 未 shim 的设备操作**抛错**（带调用栈）而不是静默通过 | 0 次 |
| `_one_step` 不变式断言 | `input_batch` 里的请求必须都在 `num_scheduled_tokens` 里 | 已用于抓出合成器 bug |
| `FakeGroupCoordinator` | 任何集合通信**直接抛错** | 0 次 |

### 3.4 合成负载（`harness/pi_harness/runner/synth.py`）

`SynthScheduler` 复刻 vLLM 0.26 scheduler 的**输出语义**（不是调用真 scheduler）：
`ScheduledNewReqs`（带完整 block_ids）/ `scheduled_cached_reqs`（增量 diff）/
`num_scheduled_tokens` / `spec_decode_tokens` / `finished_req_ids` / chunked prefill
预算分配。踩过的两个语义坑（都已修，写在这里供后来人）：

1. **同一 step 不能既"完成"又"被调度"**：`finished_req_ids` 是"上一步与当前步之间
   完成"的请求，本 harness 把完成事件**延后一步上报**，否则 `_update_states`
   会先 pop 掉 `self.requests[req_id]`，随后在 cached 循环里 `KeyError`。
2. **新请求只有真的拿到 token 预算才会进入 `scheduled_new_reqs`**：否则 worker 的
   `input_batch` 里会出现不在 `num_scheduled_tokens` 里的请求，
   `execute_model` 里 `[num_scheduled_tokens[i] for i in req_ids]` 直接 `KeyError`。
   这正是 vLLM 的调度器契约，合成器必须遵守。

---

## 4. 参数化负载（P3）

```bash
bash harness/scripts/pi-docker.sh "cd /work/harness && PI_MODEL_TMP=/tmp/pi_models \
  python -m pi_harness.runner.cli --help"
```

支持的与"负载形状"相关的开关（完整列表见 `--help`）：

| 类别 | 参数 |
|---|---|
| 模型结构 | `--model-profile {qwen35-0.8b,qwen35-2b,qwen3-1.7b,synth}`、`--hidden-size`、`--layers`、`--heads`、`--kv-heads`、`--dtype`、`--max-model-len`、`--has-gdn` |
| 引擎 | `--max-num-reqs`、`--max-num-batched-tokens`、`--block-size`、`--no-chunked-prefill`、`--no-prefix-caching`、`--spec-k`、`--async-scheduling` |
| 负载 | `--batch`、`--isl`、`--osl`、`--chunk-size`、`--prefix-hit-ratio`、`--arrival {simultaneous,stair,poisson}`、`--steps`、`--seed` |
| 保真度 | `--slot-mapping-mode {cpu_fallback,noop,inject:<us>}`、`--triton-launch-us` |
| 扫描 | `--sweep {isl,batch,osl,block_size,spec_k,prefix_hit_ratio,chunk_size,model}`、`--sweep-values`、`--repeat` |

### 4.1 三种 slot-mapping 模式（决定用途）

| 模式 | 数值正确性 | CPU 成本 | 用途 |
|---|---|---|---|
| `cpu_fallback`（默认） | ✅ slot_mapping 值正确 | 高（向量化 numpy，见 §5.2） | 功能正确性、A/B 数据内容实验、耗时 sweep |
| `noop` | ❌ 不写 slot_mapping | 最低（只剩 Python 侧参数组装） | **PMU / 火焰图 / topdown 采集**（避免 numpy 兜底的假热点） |
| `inject:<us>` | ❌ | 可控 | 敏感性分析：模拟 Triton launch 的 Python 开销 |

**采集 topdown / 火焰图时必须用 `noop`**，否则热点会落在 numpy 上，与真机不可比。

### 4.2 输出

| 文件 | 内容 |
|---|---|
| `per_step_*.csv` | 每步：`update_states_us` / `prepare_inputs_us` / `deferred_fixup_us` / `triton_cpu_us` / `triton_launches` / 请求数 / 调度 token 数 |
| `substep_*.csv` | 每步 × 每个被 wrap 的子函数耗时 |
| `summary_*.json` | p50/p90/p99/mean、子步骤汇总、config、host、`meta`（含审计字段） |

**口径**（必须写清，不可混用）：

```
pi_net_us = prepare_inputs_us - triton_cpu_us     # 扣掉 harness 独有的 numpy 兜底开销
scope_us  = update_states_us + prepare_inputs_us  # 本 harness 的默认口径
```

---

### 4.3 配置合法性校验（`RunnerConfig.validate()`）

**真机引擎会拒绝的配置，harness 也必须拒绝**——否则会跑到 vLLM 内部的隐式假设上，
抛出与根因无关的异常（实测过：`ValueError: could not broadcast input array from
shape (17,) into shape (16,)`，看起来像 block table 的 bug，实际是负载非法）。

校验项（`build()` 与两个 CLI 入口都会调用，不合法直接 return code 2）：

| 校验 | 理由 |
|---|---|
| `isl <= max_model_len` | prompt 本身就超长 |
| **`isl + osl <= max_model_len`** | **最常见的一条**。请求要产出 `osl` 个 token，序列最长到 `isl+osl` |
| `ceil((isl+osl)/block_size) <= ceil(max_model_len/block_size)` | 等价于上一条，但直接对应 `max_num_blocks_per_req` 溢出 |
| `batch <= max_num_reqs` | 并发上限会把请求挡在等待队列里（不是错误，但会让"并发数"名不符实） |
| `isl + osl + spec_k <= max_model_len` | MTP 草稿也占序列长度 |

实测：`--preset realmachine --isl 2048 --osl 64`（`max_model_len=2048`）现在给出
```
error: 配置不合法（真机引擎也会拒绝这些配置）：
  - isl+osl = 2112 > max_model_len(2048)：请求无法在 max_model_len 内产出 64 个 token。
    真机引擎会拒绝该请求；请降低 isl/osl 或提高 max_model_len（例如 --max-model-len 2112）
  - 需要 17 个 KV block > max_num_blocks_per_req 16（max_model_len=2048, block_size=128）
```
加上 `--max-model-len 4096` 即可跑通（实测 13 128 步，`prepare_inputs` p50 = 372.4 µs）。

**教训**：`realmachine` 预设的 `max_model_len=2048` 是一个**很紧的上限**，
做 ISL 扫描时必须同步抬高 `max_model_len`，否则扫到的不是"大 ISL 下的负载"，
而是"非法负载被拒绝"。

## 5. 已知偏差表（逐条登记 + 影响方向）

| # | 被 shim / 近似 / 跳过的东西 | 真机行为 | harness 行为 | 影响方向 | 补偿手段 |
|---|---|---|---|---|---|
| D1 | H2D / D2H DMA（`CpuGpuBuffer.copy_to_gpu/copy_to_cpu`） | 异步 DMA，CPU 侧是驱动调用 + 内存带宽占用 | CPU→CPU memcpy | **低估**（拷贝时间趋 0） | `shim.report()["degraded_copy_bytes"]` 逐 buffer 记账字节数，可用实测 DMA 带宽补回 |
| D2 | `.to(device="npu")` | 真实 H2D | 同设备 no-op | **低估** | 同上（`degraded_copy_bytes["to_npu"]`） |
| D3 | Triton kernel launch 的 Python 侧开销 | 参数绑定 / 特化 / cache 查找 / launcher 调度 | CPU 等价实现代替（`cpu_fallback`）或 no-op（`noop`） | `cpu_fallback` **高估** 固定 ~80-140 µs/步；`noop` **低估** 整段 launch 开销 | `--triton-launch-us` 注入；`triton_cpu_us` 逐 step 记账后扣除 |
| D4 | device op 的内核态开销（acl 调用） | 用户态 + 内核态上下文 | 0 | **低估** | 需真机实测；`inject:` 模式可做敏感度 |
| D5 | P>1 的集合通信 | all-reduce / broadcast | `FakeGroupCoordinator` **抛错** | 不适用（TP=1 场景） | 若需 TP>1，必须真机或另行建模 |
| D6 | `torch.npu.Event.synchronize()` 的真实等待 | 可能阻塞等 device | no-op | **低估**（wall time 少掉同步等待） | 同步点的"是否触发"由 `probe_devdeps` 静态确认；wall 口径需注意 |
| D7 | 权重加载 / KV cache 分配 / 图捕获 | 一次性 | 未执行 | 中性（不在 prepare_input 路径上） | — |
| D8 | 模型类 import 与 inspect | 拉起真实模型实现 | 摊平 config + 轻量 architecture | **中性**（不进 prepare_input 路径），但 `is_multimodal_model` 等标志可能与真机不同 | 需要多模态分支时用 `--model-profile` + 真 config |
| D9 | `pin_memory=True` 的 pinned buffer | 真 pinned 内存 | 退化为普通内存 | **低估**（pinned 的分配/拷贝更贵；本机容器无 pinned allocator） | 已在 meta 记录 `pin_memory_available` |
| D10 | 其它 Triton/NPU kernel（`update_cos_sin`、`_preprocess` 等） | 真实 launch | **不在 P1 范围内**（P1 只跑 `_update_states + _prepare_inputs`） | 见 §6 口径说明 | 需要时扩到 P1+ |

---

## 6. "占比"的分母口径（**最容易出错的地方**）

`probe_devdeps` 的行号级复核确认了 Ascend 版 `"prepare input"` scope 的精确边界：
**从 `model_runner_v1.py:1859` 到 `L2098`（末句 `update_cos_sin(positions)`）**，
顶层只有 3 条语句，`dynamic_eplb` / `forward` / `post process` / `sample_token` 全在 scope 外。

### 6.1 本 harness 覆盖的是 scope 的一个**子集**

| scope 内的子步骤 | 是否在 harness P1 覆盖范围 |
|---|---|
| `synchronize_input_prep()` 的事件同步 | ❌ 未覆盖（shim 成 no-op） |
| **`_update_states()`** | ✅ **覆盖** |
| **`_prepare_inputs()`** | ✅ **覆盖** |
| `_determine_batch_execution_and_padding()` | ❌ 未覆盖 |
| `_build_attention_metadata()`（含 GDN/FA builder） | ❌ **未覆盖**（真机上是最大单项，见 6.2） |
| `_preprocess()` | ❌ 未覆盖 |
| `update_cos_sin()` | ❌ 未覆盖 |

**因此 harness 的 µs 数字不能与 LiteProfiler 的 `"prepare input"` scope 数字直接比。**
要对比必须按同一子集切分（LiteProfiler 的 subscope 插桩）或把 harness 扩到 P1+。

### 6.2 ⚠️ 不要把"scope 与 harness 的差"全部归因于 shim

真机（0.8B / TP1 / B=1 / ISL=128 / `FULL_DECODE_ONLY` / pystack 关闭）：
`"prepare input"` scope p50 = **2757–2806 µs**。
harness（**同启动参数、同负载**，`--preset realmachine --batch 1 --isl 128 --osl 64`）：
`_update_states + _prepare_inputs` = **379.1 µs/步**。

差额 ≈ 2427 µs。**但其中很大一块不是我 shim 掉的，而是我从未声称覆盖的代码**：

| 差额构成 | 量级 | 性质 |
|---|---|---|
| `AscendGDNAttentionMetadataBuilder.build`（3 次 × 303 µs） | **908 µs** | **P1 范围外**（`_build_attention_metadata`）；真机实测头号热点 |
| `AscendAttentionMetadataBuilder.build`（1 次 × 36 µs） | 36 µs | **P1 范围外** |
| `_determine_batch_execution_and_padding` / `_preprocess` / `update_cos_sin` / `synchronize_input_prep` | 未单独测 | **P1 范围外** |
| H2D/D2H DMA、Triton/NPU launch、device op（acl） | 未单独测 | **真正的 shim 缺口**（§5 的 D1–D4） |

**正确表述**：`不重叠部分 = P1 范围外的代码 + shim 缺口`，而 **P1 范围外的代码至少占 944 µs（39%）**。
把整个差额说成"被 shim 掉的"会把"待覆盖范围"误报成"复刻误差"，**直接误导优化方向**
（优化 `_prepare_inputs` 动不了那 908 µs 的 GDN metadata builder）。

### 6.3 core 版 vs ascend 版

vLLM core 的等价物叫 `"gpu_model_runner: preprocess"`，边界与 ascend 版**不同**
（core 含 `_preprocess`，ascend 不含）。拿 core 的打点解释 ascend 会系统性高估。详见 `docs/01`。

## 7. 实测基线

> ⚠️ 本节混了两个口径，**引用时必须看清是哪一张表**：
> * **`realmachine`**（= 真机 launcher 参数，见 §1.5）→ 用于**与真机对照**（§7.3b、§9）
> * **`stress` / 早期 `hw64`**（放大并发与预算）→ 只用于**看趋势/斜率**，不能用于对照
> 数字口径：`pi_net_us = prepare_inputs_us - triton_cpu_us`（扣掉 harness 自有兜底）

### 7.0 三种 slot-mapping 模式的实测差异（决定 profiling 口径）

同配置（`--batch 16 --isl 1024 --osl 128 --steps 60 --max-num-reqs 64
--max-num-batched-tokens 16384 --has-gdn`）：

| 模式 | `prepare_inputs_us` p50 | 说明 |
|---|---|---|
| `cpu_fallback` | 526 µs | 含 harness 的向量化 numpy 兜底（数值正确） |
| `noop` | **386 µs** | 去掉兜底 → **纯 vLLM 代码的 CPU 残量** |
| 差 | **~140 µs** | = harness 自有开销，必须从任何"绝对 µs"里扣掉 |

**结论：采集 PMU / topdown / 火焰图必须用 `--slot-mapping-mode noop`。**
否则 topdown 与火焰图里排第一的热点会是 `numpy` 的函数，而真机上这里根本
没有 numpy（真机是 Triton kernel launch）。这条是"无卡 vs 真机"可比的**前提条件**。

### 7.1 单配置基线（stress 口径，早期）

`--batch 8 --isl 512 --osl 16 --chunk-size 512 --max-num-reqs 16 --max-num-batched-tokens 4096`
（8 并发 decode 稳态，14 步）：

| 指标 | p50 |
|---|---|
| `update_states_us` | 43.7 µs |
| `prepare_inputs_us`（含兜底） | 607.8 µs |
| 其中 `triton_cpu_us` | ~133 µs |
| **`pi_net_us`（扣兜底）** | **~475 µs** |

子步骤（单步均值，含兜底）：`compute_slot_mapping` 143 µs / `commit_block_table` 31 µs /
`_build_attn_state` 29 µs / `_get_cumsum_and_arange` 19 µs / `_prepare_input_ids` 16 µs /
`_compute_prev_positions` 3 µs / `refresh_metadata` 4 µs。
其余 ~370 µs 落在 `_prepare_inputs` 自身的行内代码
（`np.repeat` / `torch.index_select` / `torch.add` / 各类 `copy_to_gpu`）。

### 7.2 920B 的"每次小操作固定开销"（`data/harness/micro_parts_*.csv`）

这是理解 prepare_input 为什么贵的关键——**在 920B 上，单次小操作的固定开销远大于数据搬运本身**：

| 操作 | 耗时 |
|---|---|
| 空 lambda | 0.11 µs |
| `torch.Tensor.numpy()` | 1.9 µs |
| `np.cumsum(8)` | 4.1 µs |
| `np.repeat(x, 0 元素)` | 3.0 µs |
| `np.full(16384, int32)` | 4.1 µs |
| `torch.from_numpy(arr)` | 1.6 µs |
| `sm[:8].copy_(...)`（8 元素！） | **8.6 µs** |
| `block_table[arr, arr]` 花式索引 | 7.8 µs |
| 一组 8 个 numpy 小操作（`cumsum`+`repeat`+`arange`+索引） | **21 µs** |

→ `_prepare_inputs` 里有几十处这样的"小操作"，**每一项都是 µs 级固定开销**，
这正是 CPU 侧负载的微观结构。

### 7.3 参数影响 —— **stress 口径**（`data/harness/hw64_sweep_*.csv`，32 个点）

> 这是**早期放大口径**的扫描，用于看趋势；**不要**用它做真机对照。
> 真机口径的曲线见 §7.3b 与 `agents/sweep_analysis/REPORT.md`。

固定：`--max-num-reqs 64 --max-num-batched-tokens 16384 --has-gdn`，模型 qwen35-0.8b，
`cpu_fallback` 模式（数字**未扣** `triton_cpu_us`）。

| 维度 | `pi_p50_us` 区间 | 极差 | `us_p50_us` 区间（干净信号） |
|---|---|---|---|
| ISL | 527 → 542 | **1.03×（平的）** | 49 → 51 |
| batch | 487 → 871 | 1.79×（非单调，见下） | **26 → 129（强单调）** |
| chunk_size | 501 → 756 | 1.51× | 47 → 68 |
| block_size | 517 → 574 | 1.11× | 47 → 50 |
| spec_k | 514 → 549 | 1.07× | 46 → 49 |
| prefix_hit_ratio | 500 → 552 | 1.10× | 45 → 48 |

**结论 1：ISL 在 decode 稳态下几乎不影响 `prepare_input`（1.03×，落在噪声内）。**
正确的强度指标是 `per_step_us / min(ISL, chunk_size)`，不是 `per_step_us / ISL`——
chunked prefill 把单步 prefill 工作量封顶在 `chunk_size`，ISL 只决定"要跑多少步"。

**结论 2：`update_states` 才是 batch 的干净线性信号。**
`us_p50` 随 batch 从 26 µs（batch=4）单调升到 129 µs（batch=64），
这是 `_update_states` 里**逐请求 Python 循环**的直接体现（`condense` / `swap_states` /
`add_request` / `update_req_spec_token_ids`）。

**结论 3：`pi_p50` 在 batch=1,2 上反常（834 / 871 µs，而 batch=4 只有 487 µs），
在噪声打掉之前不应引用。** a3-22 宿主机有 64 个他人 python 进程、affinity 覆盖
`0-639`，`taskset` 挡不住（`env_toolchain` 已确认）。所有扫描点都用
`--repeat 3` 重跑、取跨轮中位数，并记录每轮离散度；详见
`agents/sweep_analysis/REPORT.md`。

**结论 4：除 batch 外，其余维度在 60–80 步的短 run 里都被底噪（~500 µs）淹没。**
底噪来自 `_prepare_inputs` 里那几十处 µs 级小操作（§7.2），与负载形状无关。
要看清 block_size / spec_k / prefix_hit 的斜率，需要先把 `triton_cpu_us` 扣掉
（或用 `--slot-mapping-mode noop`）并加长 run。

### 7.3b 子步骤热点 —— **`realmachine` 口径（与真机对照用这一张）**

配置：`--preset realmachine --batch 1 --isl 128 --osl 64 --steady-seconds 12`
→ **27 947 步**，scope = **379.1 µs/步**（`noop`）。
图：`figures/06-substep-realmachine.svg`（真机口径）；
对照 `figures/06-substep-stress.svg`（`stress` 口径，可与前者对比看 buffer 宽度的影响）；
数据：`data/harness/substep_realmachine.json`。

| 子步骤 | 占 scope | 每步 |
|---|---|---|
| **`(inline code in _prepare_inputs / _update_states)`** | **83.9%** | **318.1 µs** |
| `_build_attn_state` | 5.9% | 22.4 µs |
| `_get_cumsum_and_arange` | 4.3% | 16.2 µs |
| `_prepare_input_ids` | 3.8% | 14.6 µs |
| `refresh_metadata` | 0.6% | 2.3 µs |
| `_compute_prev_positions` | 0.5% | 1.9 µs |
| `condense` | 0.4% | 1.5 µs |
| 其余（`add_row`/`append_row`/`_may_reorder_batch`/…） | <0.5% | <1 µs |

**这条结论比 stress 口径更强**：在真机口径下，**84% 的 `prepare_input` CPU 时间
落在 `_prepare_inputs` 的"行内代码"上**——即那几十处 `np.repeat` / `torch.index_select` /
`torch.add` / `copy_to_gpu()` / 逐请求 Python 循环。**没有任何一个可单独 wrap 的
helper 能解释超过 6%**。优化必须瞄准"减少调用条数 / 合并小操作"，而不是挑某个 helper。

> 这也解释了 `historical_evidence` 的观测：真机热点极平（top-10 仅 23.7%，
> `_PyEval_EvalFrameDefault` 10.3%）——因为成本均匀摊在大量解释器级小操作上。

### 7.4 子步骤热点表 —— **`stress` 口径（看结构，不看绝对值）**

配置：`--batch 16 --isl 1024 --osl 128 --steady-seconds 10 --has-gdn
--max-num-reqs 64 --max-num-batched-tokens 16384 --slot-mapping-mode noop`
→ **16 282 步**，scope 累计 7.53 s → **462.2 µs/步**。
原始数据：`data/harness/substep/tmp_bs16b/substep_*.csv`（**单位秒、每步增量**）。

| 子步骤 | 占 scope | 每步 |
|---|---|---|
| **`NPUModelRunner._prepare_inputs`**（含所有未单独 wrap 的行内代码） | **88.6%** | **409.7 µs** |
| `NPUModelRunner._update_states` | 11.4% | 52.5 µs |
| ├ `GPUModelRunner._update_states` | 10.8% | 49.8 µs |
| ├ `MultiGroupBlockTable.commit_block_table` | 5.8% | 26.7 µs |
| ├ `NPUModelRunner._build_attn_state` | 5.5% | 25.6 µs |
| ├ `GPUModelRunner._get_cumsum_and_arange` | 4.0% | 18.4 µs |
| ├ `GPUModelRunner._prepare_input_ids` | 3.3% | 15.5 µs |
| ├ `MultiGroupBlockTable.compute_slot_mapping` | 3.1% | 14.6 µs |
| ├ `InputBatch.update_req_spec_token_ids` | 1.8% | 8.1 µs |
| ├ `InputBatch.add_request` | 1.1% | 5.0 µs |
| ├ `InputBatch.refresh_metadata` | 0.6% | 2.9 µs |
| ├ `GPUModelRunner._compute_prev_positions` | 0.5% | 2.3 µs |
| └ `InputBatch.condense` | 0.4% | 1.9 µs |

**关键结构结论**：被单独 wrap 的子函数加起来只有 **~121 µs/步（26%）**，
剩下 **~289 µs/步（62%）全部落在 `_prepare_inputs` 自己的行内代码上** ——
即 `np.repeat` / `torch.index_select` / `torch.add` / `positions` 计算 /
几十处 `copy_to_gpu()` 与 numpy 小操作。这解释了为什么"优化 prepare_input"
不能只盯着几个 obvious 的子函数：**主要成本是"函数体本身"，而且高度碎片化**
（对照 §7.2：920B 上单次小操作固定开销 2–9 µs）。

> ⚠️ `compute_slot_mapping` 在这里只有 14.6 µs，是因为 `noop` 模式把 kernel 的
> 数值计算去掉了，只剩 **Python 侧的 launch 参数组装**。`cpu_fallback` 模式下
> 同一步是 ~140 µs。真机上的这一段 = Python launch 开销 + 设备侧执行，
> 前者是本表能给的 14.6 µs，后者需要真机标定（`--triton-launch-us`）。

### 7.5 长 run（PMU 采集）的实测口径

`--steady-seconds N` 让负载按**墙钟**跑够 N 秒（跑完自动重建 Scheduler +
InputBatch 继续跑，保证 CPU 全程有活干）。实测 25 秒 run（`noop`，batch=8/isl=512）：

| 指标 | p50 | 说明 |
|---|---|---|
| 步数 | **≈ 5.0 万步** | ≈ 50k × 390 µs ≈ 20 s |
| `prepare_inputs_us` | **390.5 µs** | 纯 vLLM 代码（noop 口径） |
| `update_states_us` | **20.0 µs** | |
| `prepare_inputs` 累计 | 19.9 s | 占墙钟 ≈ 80% |

**为什么必须要这个模式**：单次 harness run 的稳态只有 ~0.08 s（200 步 × 390 µs），
而 `import vllm` 要 ~30 s。PMU/perf 仪器挂上时进程往往已经退出
（实测症状：`pid ... 不存在或已退出`）。`--steady-seconds` + `--warmup 45`
是"无卡负载能被真机 profiling 链路采到"的必要条件。

## 8. 怎么用这个 harness

### 8.1 CPU 侧优化迭代（不需要卡）

```bash
bash harness/scripts/pi-docker.sh "cd /work/harness && PI_MODEL_TMP=/tmp/pi_models \
  python -m pi_harness.runner.cli --batch 32 --isl 2048 --osl 128 --steps 300 --has-gdn \
  --slot-mapping-mode cpu_fallback --out /work/data/harness --tag before"
# 改代码（vLLM / vllm-ascend 任一），再跑一遍，用 substep CSV 对比
```

### 8.1b 三种 slot-mapping 模式的定量对比（`realmachine` 口径，batch=1）

原始数据：`data/harness/sweep_rm{dsteady,noop,cpu,inj}_realmachine_*.csv`
（`pi_net_p50_us` 已扣除 `triton` 列，即"vLLM 代码自己的 CPU 时间"）。

| 模式 | `pi_net_p50_us` | `triton_p50_us`（harness 自有） | 解读 |
|---|---|---|---|
| `noop` | **375.2 / 381.6 / 383.8** | 0.53–0.56 µs | 纯 vLLM 代码 + 空 kernel 桩 ✅ |
| `cpu_fallback` | **408.0 / 412.9 / 414.8** | **87–103 µs** | numpy 兜底自身 ~90 µs |
| `inject:15` | 375.6 / 381.1 / 384.5 | ~101–103 µs（自旋） | 与 `noop` 净时间一致 → **自旋被正确归类**，方法学自检通过 |

> ⚠️ **修正（`sweep_analysis` 的交错对照实验）**：上表三个模式来自**不同时间窗**的独立运行，
> 本机 `pi_raw` 的跨窗口漂移实测可达 **1.9×**，所以"`cpu_fallback` 净时间比 `noop` 高 31 µs"
> 里大部分是漂移。**同进程内 3 轮交错配对**的结果是：间接成本只有
> **+11.1 µs**（区间 +3.6 ~ +14.3）。跨窗口数据只能当**上界**看。
> 另一条受控结论更硬：`inject:15 − cpu_fallback` 的净时间差 **+0.8 µs（≈0）**，
> 而 `triton_cpu` 恰好多 15.6 µs ⇒ **自旋 100% 正确归类、零污染**。
>
> 还推翻了我原先的一个预期：numpy 兜底成本**不随 `max_model_len` 缩小**
> （真机口径 87.0 µs vs stress 92–95 µs）——它由**每步固定调用次数**主导，与缓冲区大小无关。

**两条可复用的结论**：

1. **必须用 `noop` 做任何与真机对照的采集**。`cpu_fallback` 不只是"多 100 µs"——
   那 100 µs 的热点是 `numpy` 函数，而真机这里根本没有 numpy（是 Triton launch）。
   **假热点会直接把 top-20 对照判据打穿。**
2. **`inject:<us>` 的净时间与 `noop` 一致**，说明 harness 的"自有开销"与"被复现的
   vLLM 开销"是可分离的；真机标定出 Triton launch 的真实 Python 侧开销后，
   直接 `--triton-launch-us <值>` 即可把这一项加回来。

### 8.2 PMU / 火焰图（topdown / IPC / 热点）

```bash
# 必须用 noop，否则热点是 numpy 兜底
bash harness/scripts/prof_harness.sh --slot-mapping-mode noop --batch 16 --isl 1024 --steps 200
```
采集口径由 `docs/04-profiling-methodology.md` 统一规定；脚本在
`harness/scripts/prof_{calib,harness,flamegraph,topdown}.sh`。

### 8.3 A/B：数据内容 vs 形状（**已得出结论**）

数据层由 `trace_schema` 交付（`harness/pi_harness/{trace,workload}/**`），
完整报告见 `agents/trace_schema/REPORT.md`。判定方法：

* 三组**同形状**负载：`synth`（形状合成，**默认用真实的 `Scheduler` + `KVCacheManager`
  在 CPU 上驱动**，不是自写调度器）/ `shuffled`（同形状、把 token id / block id /
  请求 ID 重排）/ `real`（真机 capture，待到位）。
* 三组**走同一个下游代码路径**：`PrepareInputReplay._one_step_with`
  （即真实 `_update_states + _prepare_inputs`）。
* 统计量用 **per-step 线程 CPU 时间**（`time.thread_time`，免疫宿主机抢占）
  的**配对比值**，而不是 wall p50 —— 因为 a3-22 的 wall p90 噪声可达 16%。

**结论：数据内容（取值）对 `prepare_input` 成本无影响。**

| 负载 | `shuffled/synth` 配对中位数比 | p90 偏差 | 同组噪声地板 |
|---|---|---|---|
| mix | 1.0004 | ~1.3% | 0.03–1.9% |
| prefill | 1.0009 | ~3.2% | 同上 |
| mtp2 | 1.0098 | ~1.5% | 同上 |
| mtp2big | 1.0002 | ~1.4% | 同上 |

→ **shape 级合成足够**，不需要为"数据内容"做真机 record & replay。
**但仍需要真机 trace 来回答另一个问题**："合成的*形状*是否等于真机的形状"
（即 scheduler 的准入/chunk 切分/prefix 命中/block 分配分布是否一致）。
这一条链路已就绪（`--trace <jsonl>`），等 `real` 组到位即可补最后一列。

```bash
# 用 trace 驱动（P2 通路，已实测可用）
... python -m pi_harness.runner._replay_main --trace /work/data/harness/trace_real.jsonl \
      --slot-mapping-mode noop --has-gdn
```

## 9. 与真机的一致性（**首次对照已完成**）

一致性判据（`plan/COORDINATION.md` §7）：topdown 各分量 ±2~3 pp、IPC ±10%、
热点函数 top-20 重合 ≥80%、逐 step 耗时分布同形。

### 9.1 已完成的对照：topdown + IPC

无卡 harness（`--preset realmachine`，`noop`，PMU 经 libkperfx 920B preset，
9 组 `confidence=1.0`）：

| 指标 | 无卡 harness | 真机（`historical_evidence`，26 个配置） | 差异 |
|---|---|---|---|
| **主 bound** | **frontend_bound** | **frontend_bound** | ✅ 一致 |
| frontend_bound | **72.1%** | 56.0–65.0% | +7 ~ +16 pp（harness 更集中） |
| retiring | 15.9% | — | — |
| bad_spec | 8.1% | — | — |
| backend_bound | 3.8% | — | — |
| **IPC** | **0.949** | **0.719–0.890** | +7 ~ +32% |
| branch miss | 5.84% | 3.7%（B=1,T=1 历史点） | 同量级 |
| `rob_stall` | 100%（frontend latency 主导） | — | — |
| 细分 | latency_bound 64.7% / bandwidth_bound 7.4% | — | — |

**结论：定性一致（都 frontend-bound、IPC 都 < 1、都是 latency-bound 的取指停顿），
定量上 harness 偏乐观 7–32%。** 偏差方向可解释，且**是"低估"方向**——
真正的 CPU 瓶颈只会比 harness 显示得更严重。

### 9.2 偏差归因（为什么 harness 的 IPC 偏高）

| 原因 | 机制 | 影响方向 |
|---|---|---|
| **紧循环 cache 友好** | harness 连续调 `_update_states + _prepare_inputs`，同一段代码/缓冲区反复访问，i-cache / iTLB / L1d 命中率高于真机 | IPC **高估** |
| 缺少步间其它工作 | 真机每步之间还夹着 scheduler、ZMQ RPC、forward launch、D2H 采样回传，会污染 i-cache/分支预测器 | IPC **高估** |
| 同步点被 no-op | `synchronize_input_prep` 的 `Event.synchronize()` 在真机可能真等，等待期间不产生指令 | 真机 IPC **被拉低** |
| `compute_slot_mapping` 数值被跳过 | `noop` 模式下没有 kernel 的数值工作与真实 launch | 需用真机标定 launch 开销补回 |

**已量化的口径敏感性**（同一个 `prepare_input` 代码，只换 preset）：

| preset | 工作集 `token_ids_cpu` | IPC | 说明 |
|---|---|---|---|
| `realmachine`（真机口径） | (8, 2048) = **64 KB** | **0.949** | ✅ 与真机的 0.719–0.890 同区间 |
| `stress` | (64, 32768) = **8 MB** | **1.553** | ❌ 远离真机，**不能用来做一致性对照** |

→ 这一行数据本身就是"**必须匹配真机启动参数**"的最强证据（见 §1.5）。

### 9.3 火焰图 / 热点 / 指令级（`prof_measure` 交付）

产物目录：`data/harness/prof_realmachine_20260924-031609/`
（`topdown.json`、`hotspots.json`、`hotspots_top20.csv`、`annotate_top.json`、
`data/harness/prof_realmachine_20260924-031609/flamegraph_oncpu.svg` + 同名 `.folded`、
`manifest.json`、`noise_pre/post.json`）。

| 项 | 结果 |
|---|---|
| **on-CPU 火焰图** | `data/harness/prof_realmachine_20260924-031609/flamegraph_oncpu.svg`（flamegraph-rs 0.6.14，**29 953 样本，0 lost**）；<br>同一张图已归档到交付图的统一目录：`figures/05-flame-harness-realmachine.svg` |
| 折叠栈 | `data/harness/prof_realmachine_20260924-031609/flamegraph_oncpu.folded`（6952 行） |
| **指令级 annotate** | `_PyEval_EvalFrameDefault` 11.43%（`object.h:642/646` 的**引用计数读写占 17.5%**）；`_PyObject_Malloc` 1.63%（free-list 头 load/store）；`unicodekeys_lookup_unicode` 1.38% |
| self 热点 top-5（火焰图口径） | `_PyEval_EvalFrameDefault` 2164 / `Py_INCREF` 1190 / `_PyType_Lookup` 375 / `unicodekeys_lookup_unicode` 373 / `pthread_mutex_lock` 336 样本 |

**关键**：`Py_INCREF` 作为独立热点排到第 2（1190 样本）与 annotate 里"引用计数占
`_PyEval_EvalFrameDefault` 的 17.5%"互相印证——**CPU 确实主要花在 CPython 对象
管理与解释器循环上，而不是在算数据**。这正是 §7.2/§7.3b 从 µs 级观察得到的结论
在 PMU/指令级上的独立确认。

### 9.3b `noop` vs `cpu_fallback` 的火焰图对照（量化 harness 污染）

| 口径 | IPC | top-20 交集（两者之间） | 变化 |
|---|---|---|---|
| **`noop`**（真机可比） | **0.9489** | — | 基线 |
| `cpu_fallback` | **0.8066** | **16/20 = 80%** | **IPC −15%**；numpy `ufunc_generic_fastcall`（0.79%）与 ATen `as_strided` 派发被推进榜 |

→ 这条**从另一个角度证明必须用 `noop`**：`cpu_fallback` 不光是多花时间，
它**改变了热点榜的构成**（把 numpy/ATen 派发推上来），IPC 也低 15%。
用 `cpu_fallback` 采的 top-20 去和真机比，会把 numpy 误判成 vLLM 的热点。

### 9.3c 热点函数 top-20 重合（**判据达成 80%**）

数据：`data/harness/hotspot_overlap.json`（含两侧完整 top-20 与归一化规则）。

| | 无卡 harness（`realmachine`，`noop`） | 真机（a3-21，Qwen3.5-0.8B TP1，worker 主线程） |
|---|---|---|
| 采集 | `perf record -t <host_tid> -e cycles:P -g` | `perf record -t <TID> -e cycles -g` |
| **top-20 重合** | **16 / 20 = 80%** ✅ | 同左（Jaccard 66.7%） |
| top-1 | `_PyEval_EvalFrameDefault` **11.43%** | `_PyEval_EvalFrameDefault` **10.33%**（Δ **+1.10 pp**） |
| top-10 合计 | **21.43%** | **23.70%** |
| top-20 合计 | **27.35%** | **28.72%** |

**两条结论都成立**：
1. **热点形状一致**——两侧都是"极平的 CPython 解释器热点"，
   top-10 只占 21–24%，top-20 只占 27–29%。
2. **头号函数一致且量级接近**——`_PyEval_EvalFrameDefault` 相差仅 1.1 pp。

**8 个不在交集的条目，每一个都能解释**：

| 仅出现在 harness | 解释 |
|---|---|
| `tuple_alloc`、`torch::autograd::THPVariable_getitem`、`DifferentiableViewMeta::DifferentiableViewMeta` | harness 里所有算子都走 **CPU ATen**（真机走 `torch_npu`），ATen 派发/autograd 开销占比更高 |

| 仅出现在真机 | 解释 |
|---|---|
| `eventfd_write+0x214`（kernel）、`pthread_rwlock_rdlock` | 真机每步之间有 **ZMQ/进程间通信与锁**（engine ↔ worker），harness 单进程紧循环没有这些 |
| `PyObject_GetAttr`、`cfree` | 真机代码路径更长（scheduler/RPC 侧），属性查找与释放更多 |

**这两组差异恰好指出了 harness 已知的两条偏差**（§5 的 D1–D4 与 §9.2）：
缺少步间通信/锁、以及用 CPU ATen 代替 NPU 算子。**方向都是让 harness 显得更"干净"**，
即真机的 CPU 压力只会更大。

### 9.4 判据总表

| 判据（`plan/COORDINATION.md` §7） | 目标 | 实测 | 结论 |
|---|---|---|---|
| topdown 主 bound 类型 | 一致 | 两侧都是 **frontend-bound** | ✅ |
| topdown 各分量 | ±2~3 pp | frontend 72.1% vs 56–65% | ⚠️ 差 7–16 pp，**方向已知且可解释**（§9.2） |
| IPC | ±10% | 0.949 vs 0.719–0.890 | ⚠️ 高 7–32%，**同上** |
| **热点函数 top-20 重合** | **≥80%** | **80%** | ✅ |
| 逐 step 耗时分布同形 | 同形 | 未直接对照（口径不同，见 §6） | ⏳ 需子 scope 数据 |

**对"±2~3 pp / ±10%"这两条判据的诚实评价**：它们隐含假设"harness 复现了同样的
cache 与干扰环境"。harness 是紧循环、单进程、无步间通信，**做不到**这个假设。
因此正确表述是：**同一 regime、同样的人类可读热点结构；harness 定量偏乐观 7–32%，
偏差方向已知（低估真机压力）**。要真正收敛到 ±2~3 pp，需要
(a) 真机侧按 subscope 切分同一边界，且 (b) harness 注入步间干扰模型。

### 9.5 仍需回填

| 项 | 依赖 |
|---|---|
| 逐 step 耗时分布同形 | 真机 LiteProfiler 的 `prepare input` **子 scope** 数据（`subscope_instrumentation`） |
| Triton launch 的 Python 侧开销标定值 | 真机 `--triton-launch-us` 标定 |
| H2D DMA 字节 → 耗时换算 | 真机实测 H2D 带宽与调用开销 |
| 火焰图 SVG（on-CPU，含 Python 帧） | `prof_measure` 的 flamegraph 产物（符号化配方见下） |

> **符号化配方（实测有效，供后来人）**：`perf record` 在宿主上跑，容器里的库
> 不在宿主 fs 上，`perf report` 会退化成裸地址（如 `[.] 0x00000000001e0140`）。
> 解法是把容器里**缺的库**按**同样的绝对路径**拷进 `--symfs` 目录，再 `perf report --symfs`：
> ```bash
> docker run --rm -v "$SYMFS:/out" <IMG> bash -c '
>   mkdir -p /out/usr/local/python3.12.13/lib
>   cp /usr/local/python3.12.13/lib/libpython3.12.so.1.0 /out/usr/local/python3.12.13/lib/
>   mkdir -p /out/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib
>   cp /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so /out/.../torch/lib/'
> sudo perf report --force --stdio --no-children --sort dso,symbol --symfs "$SYMFS" -i perf.data
> ```
> 补完这两个库后，`_PyEval_EvalFrameDefault` 从裸地址变成 11.43% 的可读符号。
