# probe_devdeps 交接报告：无卡设备依赖清单 + `prepare_input` 可执行性矩阵

> 2026-09-24（Asia/Shanghai）｜owner：`probe_devdeps`｜范围：`harness/probes/**`、`data/harness/devdeps_*`、`agents/probe_devdeps/**`
> 运行入口（唯一合法）：`ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "<cmd>"'`

## 0. 一句话结论

**vLLM 0.26.0 + vllm-ascend 0.26.0rc1 的 `prepare_input` 真实代码路径可以在完全没有 NPU 卡的情况下跑起来**，
但需要**四层**处理：① device 字符串 npu→cpu 重映射（工厂 + Tensor/Module + `torch.accelerator`）；
② `torch_npu` 明确**不伪造**（无卡下 import 本身是安全的）；③ Triton kernel launch 的 CPU 等价替换
（`_compute_slot_mapping_kernel`，这是 prepare_input 主路径上唯一的"必然失败"点）；④ 手工补装
真机由 worker init 建立的对象（`init_ascend_config` / `KVCacheConfig` / 分布式 group 桩）。
被 shim 抹掉的设备交互中，**`Event.synchronize`（scope 入口）与 Triton launch 是偏差最大的两处**，方向都是**低估**。

## 1. 做到哪一步 / 哪一步做不到

| 项 | 状态 | 证据 |
|---|---|---|
| A. 导入梯度实验（16 步，全部落盘） | **完成** | `data/harness/devdeps_imports*.json` + `data/harness/devdeps_stacks/A*.txt` |
| A. 设备 API 逐条运行时探针（30 条，独立子进程） | **完成** | `data/harness/devdeps_stacks/A8_device_runtime_probes_devapi.txt` |
| B. shim 可行性 | **完成**（tensor/device 层由 replay_harness 落地，本 agent 提供审计与参考实现） | `data/harness/devdeps_shim_smoke.json` |
| C. API 清单（静态 × 动态） | **完成** | `data/harness/devdeps_apilist.{md,json}` |
| D. 子步骤矩阵 | **完成** | `data/harness/devdeps_matrix.{md,json}` |
| E. 冒烟验证 | **完成**（17 用例 16 通过，1 条为已知"只写属性"限制） | `data/harness/devdeps_shim_smoke.json` |
| E′. scope 边界行号级证据 | **完成** | `data/harness/devdeps_scope_boundary.md` + `data/harness/devdeps_scope.json` |
| **做不到**：真机 `aclInit` 之后的任何行为 | 无卡环境下 `aclInit` 返回 507008，**任何真实设备调用都必然失败**；这是物理限制，不是实现缺陷 | `A8_device_runtime_probes_devapi.txt` |
| **做不到**：`torch_npu._inductor` 导入 | 无卡下 `triton ... get_arch` 抛 `SystemError: <built-in function get_arch> returned NULL` | `A9_torch_npu_surface_order.txt` |
| **做不到**：真实 pinned memory | 容器无 pinned allocator，`torch.zeros(4, pin_memory=True)` 直接 RuntimeError → 必须降级为 False | `A1_torch_basic.txt` |
| **近似而非复现**：H2D/D2H 与 kernel 执行 | `CpuGpuBuffer.cpu`/`.gpu` 两块都变 CPU tensor，`copy_to_gpu()` 退化为 CPU→CPU memcpy | `devdeps_shim_smoke.json` C6/C7 |

## 2. A. 导入梯度实验（要点）

完整栈见 `data/harness/devdeps_stacks/`。每条都带：命令、返回码、前 40 行栈、结论一行。

### 2.1 关键事实

1. **容器内 uid 1003 没有 passwd 条目** → `getpass.getuser()` 抛 `KeyError` → torch inductor
   `default_cache_dir()` 崩。**任何 `import vllm` / `import torch_npu` 都会中招**。
   修法（已由 replay_harness 落到 `pi-docker.sh`）：`-e USER=REMOTE_USER -e LOGNAME=REMOTE_USER
   -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti_cache -e TRITON_CACHE_DIR=/tmp/triton_cache`。
   *这是本轮最容易被误判成"vllm 装坏了"的环境坑。*
2. **`import torch_npu` 在无卡容器里成功**（`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 与 `=1` 都成功），
   只有 `[LOG_WARNING] can not create directory, directory: /ascend/log`。
   `torch_npu.__version__ = 2.10.0.post4`，`torch_npu.utils.get_cann_version() = '9.1.0'`。
   **结论：不要伪造 torch_npu 模块**——真模块存在且可用，伪造只会引入偏差。
3. `import torch` 后 `hasattr(torch,'npu') = False`；`torch.device("npu:0")` 报
   `RuntimeError: Expected one of cpu, cuda, ... device type`。
   **即"npu"这个 device 名在 `import torch_npu` 之前根本不存在**（privateuse1 的 name 是 `privateuseone`）。
4. `import vllm` / `import vllm_ascend` / `import vllm_ascend.platform` **都成功**（exit=0）。
   `VLLM_TARGET_DEVICE` 设与不设都解析出 `NPUPlatform`（因为 platform plugin `ascend` 被 entry point 激活）。
   `VLLM_USE_V1` 在 vLLM 0.26 里已不是有效开关（`envs.VLLM_USE_V1 = <missing>`）。
5. **`import vllm_ascend.worker.model_runner_v1` 直接 import 会循环导入失败**：
   ```
   model_runner_v1 → vllm_ascend.patch.worker → patch_triton →
   vllm.third_party.flash_linear_attention.ops.utils:123
     device = "cuda" if current_platform.is_cuda_alike() else get_available_device()
   → triton/backends/ascend/driver.py:151 get_arch()
   → SystemError: <built-in function get_arch> returned NULL without setting an exception
   ```
   另有一条更早的分支（`attention.attention_v1 → device.device_op → ops.__init__ → ...`）
   会报 `ImportError: cannot import name 'DeviceOperator' from partially initialized module`。
   **正确顺序：先 `import vllm`（触发 platform plugin → `vllm_ascend.ops` 完成初始化）+
   修复 `triton NPUUtils.get_arch`，再 import worker 模块。** 见 `A4d_*_order.txt`。
6. `VLLM_PLUGINS=""` 与 unset 的区别（实测）：
   - `""` → `envs.VLLM_PLUGINS == ['']` → allowlist 匹配不到任何名字 → **general plugins 一个都不加载**，
     但 **platform plugin 仍加载**（`load_plugins_by_group` 对 platform group 也用同一 allowlist，
     而 `ascend` 条目照样被激活，因为 platform 解析走的是 `resolve_current_platform_cls_qualname`）。
   - unset → `None` → 8 个 general plugins 全加载（含 `ascend_kv_connector`/`ascend_model` 等）。
   **对 harness 的含义：用 `VLLM_PLUGINS=""` 可以关掉不必要的 general plugin 副作用，且不影响 NPUPlatform 解析。**
7. 无卡容器里 `npu-smi` **不存在**（`command -v npu-smi` → not found；直接调用 rc=127），
   `/dev/davinci*`、`/dev/devmm_svm`、`/dev/hisi_hdc` 全部不存在；
   但 `/usr/local/Ascend/cann-9.1.0` 与 `msprof` 存在，`import acl` 成功、`acl.init()` 返回 **500000**。
8. 容器事实（全部写进 `devdeps_manifest_container.json`）：
   - python **3.12.13**；openEuler **24.03 LTS-SP3**；aarch64；内核 `6.6.0-159.4.3.154.oe2403sp4.aarch64`
   - torch **2.10.0+cpu**（`torch.version.cuda = None`）、torchaudio 2.10.0+cpu、torchvision 0.25.0+cpu
   - torch_npu **2.10.0.post4**、CANN **9.1.0**（`/usr/local/Ascend/cann-9.1.0`）
   - **numpy 1.26.4**、vllm **0.26.0+empty**、vllm_ascend **0.26.0rc1**
   - **triton 版本有三个口径，必须同时记录**：`pip list` → `triton==3.5.0` + `triton_ascend==3.2.2`；
     而 `triton.__version__` 在运行期解析为 **3.2.0**（被 triton_ascend 的 path manager 覆盖）。
     **引用版本号时必须写清是哪个口径**，否则无法复现。
   - 640 逻辑核、8 NUMA、L3 560 MiB；cgroup cpuset **`200-215`**；`torch.get_num_threads()=1`、
     `torch.get_num_interop_threads()=640`（**interop 线程数未受限，跑多线程 op 前要设**）
   - 镜像内 vllm / vllm-ascend **没有 `.git`**（`git rev-parse HEAD` → `<no git>`），
     所以镜像内的 commit 身份**无法自证**；本报告一律用**文件 sha256** 锚定
     （`model_runner_v1.py` = `4699af75…`、`block_table.py` = `6c17cb4c…`、
     `gpu_model_runner.py` = `81b7627f…`、`v1/utils.py` = `10eda8e6…`），
     并与 a3-22 `refs/` 逐字节比对通过。

### 2.2 运行时设备 API 探针（30 条，逐条独立子进程）

| 结果 | API |
|---|---|
| **可调用（无卡不崩）** | `is_available()`→False、`device_count()`→0、`Event()`、`Event(enable_timing=True)`、`empty_cache()`、`memory_allocated()`→0、`is_current_stream_capturing()`→False、`config.allow_internal_format=True`、`torch_npu.npu.is_available()`→False、`torch_npu.utils.get_cann_version()`、`import acl`、`import torch_npu.profiler.dynamic_profile`、`import torch_npu.op_plugin.atb._atb_ops` |
| **失败：`aclInit` 507008**（`torch_npu/npu/__init__.py:273 _lazy_init`） | `current_device()`、`current_stream()`、`default_stream()`、`synchronize()`、`mem_get_info()`、`set_device(0)`、`torch.zeros(device='npu')`、`torch.zeros(device='npu:0')`、`Tensor.npu()`、`Tensor.to('npu')`、`pin_memory=True`、`torch_npu.npu.current_device()` |
| **失败：其它** | `get_device_name(0)`/`get_device_properties(0)` → `AssertionError: Invalid device id`（**不是** aclInit）；`Stream()` → `TypeError: '<' not supported between instances of 'NoneType' and 'int'`（`npu/utils.py:112`）；`ExternalEvent()` → `LazySetDevice 107001` |
| **失败：`torch.accelerator`** | `current_device_index()` / `synchronize()` / `current_stream()` / `set_device_index(0)` → aclInit 507008；`empty_cache()` → `INTERNAL ASSERT FAILED ... Allocator for npu is not a DeviceAllocator`；`memory_info` 不存在（真名是 `get_memory_info`） |

> 注意 `get_device_name`/`get_device_properties` 与 `Stream()` 的失败**不是** aclInit，
> 说明 shim 若只拦 aclInit 类路径，仍会在这两处静默崩。**它们必须被显式替换。**

## 3. B/D. shim 审计结论（被测对象：`harness/pi_harness/shim`，owner=replay_harness）

### 3.1 冒烟结果（`data/harness/devdeps_shim_smoke.json`）

**16 passed / 1 failed**。全部 17 条命令与输出见 `data/harness/devdeps_stacks/shim_smoke_C*.txt`。

| 用例 | 结果 |
|---|---|
| C0 `install(device='cpu')` 幂等 + `import torch_npu` 命中 | OK |
| C1 `import torch_npu` | OK |
| C2 `torch.npu.*` 命名空间（`is_available`/`current_stream`/`Event`/`Stream`/`synchronize`/`mem_get_info`） | OK |
| C3 `import vllm` → `0.26.0` | OK |
| C4 `import vllm_ascend.ops` | OK |
| **C5 `from vllm_ascend.worker.model_runner_v1 import NPUModelRunner`** | **OK**（MRO：NPUModelRunner → GPUModelRunner → …） |
| **C6 `CpuGpuBuffer(1024, dtype=int32, device=torch.device("npu"))` + `copy_to_gpu()`** | **OK**（`cpu.device=cpu`、`gpu.device=cpu`、值一致） |
| C7 `CpuGpuBuffer(device='npu:0', pin_memory=True)` | OK（`cpu.is_pinned=False`，已降级并记账） |
| C8/C9/C10 `Tensor.npu()` / `.to('npu')` / `.to(device('npu:0'))` | OK（全部映射到 cpu） |
| C11/C12 `torch.zeros(device='npu'/'npu:0')` | OK |
| C13 `torch.zeros(pin_memory=True)` | OK（降级，记账 `pin_memory->False:zeros`） |
| C14 `with torch.device('npu'): torch.zeros(2)` | OK |
| C15 `torch.npu` Stream/Event 语义（record/synchronize/stream ctx） | OK |
| **C16 `torch.npu.config.allow_internal_format` 读写** | **FAIL**：`AttributeError: '_npuConfig' object has no attribute 'allow_internal_format'` |

**关于 C16**：这**不是** shim 缺陷。真实 `_npuConfig` 对象（torch_npu 自带）就是"可写不可读"的：
写入不创建属性，读取必然 AttributeError。全仓 grep 确认 vllm/vllm-ascend 里
`torch.npu.config.allow_internal_format` **只有一处写入**（`model_runner_v1.py:224`），无读取点。
→ **列为已知限制，不要求修复**；但 shim 若替换 `npu.config` 对象，须保证可写。

### 3.2 shim 的 API 命中计数（`shim.report()`，见 `devdeps_shim_report.json` / C0）

```
device.map_npu_to_cpu / map_npu_to_cpu_str   ← device 字符串重映射（两层：对象/字符串）
pin_memory->False:<factory>                  ← pin_memory 降级（按 factory 分桶）
CpuGpuBuffer.copy_to_gpu                     ← 拷贝退化
Tensor.npu / Tensor.to(npu)                  ← tensor 方法层
npu.stream / Event.record / Event.synchronize / Stream.synchronize
degraded_copy_bytes{copy_to_gpu, to_npu}     ← **降级搬运的字节数**（最重要的偏差量化指标）
UNSHIMMED_DEVICE_OP                          ← 未覆盖路径的审计钩子（必须为 0）
```

**审计要求**：`UNSHIMMED_DEVICE_OP` 必须恒为 0。若不为 0，说明有路径真的去碰设备了，
shim 的 `report().warnings` 会给出调用栈；这时**不允许静默通过**，必须登记。

### 3.3 与我的参考实现的差异（`harness/probes/shim_patch_proposal.py`）

我在 shim 只有命名空间替换、冒烟 10/17 失败时写了参考实现（factory 包装 + `Tensor.to/npu/npu_` +
`nn.Module.to/npu` + `torch.accelerator.*` 接管），自测 **13/13 通过**
（`data/harness/devdeps_shim_patch_selftest.json`）。replay_harness 随后在 `shim/**` 里独立落地了同类三层，
并额外解决了两个我未覆盖的点：**`torch.Generator` 必须用子类替换**（函数对象会让 import 期的
`torch.Generator | None` 注解求值抛 `TypeError`）、以及 `nn.Module.to` 的 device 归一化。
`shim_patch_proposal.py` 保留作为**独立复现路径**（用它可以不依赖 shim 的当前状态复现"三层 patch 才够"这一结论）。

## 4. C. API 清单（`data/harness/devdeps_apilist.md`）

37 行表格（API / 出现位置 / 静态命中(链内/包内) / 运行时命中 / 类别 / shim 行为 / 风险）+ 动态计数明细。
**静态与动态的差异说明**（任务要求）：

1. **静态命中在"链内"这一列的取值普遍小于"包内"**：`torch.npu.current_stream` 包内 61 处、
   **链内只有 1 处**（`model_runner_v1.py:2674` 的 `torch.npu.current_stream().synchronize()`，
   在 `_update_full_graph_params` 里，`enable_enpu` 才进入）。
   → 说明**绝大多数 `torch.npu.*` 调用不在 prepare_input 链上**（在 init / forward / sample / graph capture 路径）。
   把全仓计数当成"prepare_input 的设备依赖"会**高估**一个数量级。
2. **运行时命中为 0 ≠ 真机不调用**：`probe_dynamic_hits.py` 只跑到"构造 + 最小调用"层面，
   没跑完整 step。**prepare_input 主路径的逐步计数需要 runner（replay_harness）产出后回填同一张表**
   （表格已预留该列，`runtime_keys` 在 `devdeps_apilist.json` 里可机器回填）。
3. **链内真正的设备交互以"间接"形式出现**：不是直接写 `torch.npu.xxx`，而是经
   `CpuGpuBuffer.copy_to_gpu`（链内 **29** 处调用点）、`.to(self.device)`、`pin_memory()`、
   `Event.synchronize`、triton launch。**只 grep `torch.npu.` 会漏掉几乎全部主路径成本。**
4. **静态 0 命中但运行时必需的三项**：`torch.Generator`（注解）、`init_ascend_config`、
   `KVCacheConfig` —— 它们是"装配依赖"而非"设备依赖"，但缺一个就跑不起来。

## 5. E. `prepare input` scope 边界（占比分母的唯一定义）

**文件**：`vllm_ascend/worker/model_runner_v1.py`
（sha256 `4699af75ac8b4b55283d55190365e93d13820a05888bfa65aea8746deabc4d3a`，与 a3-22 `refs/` 逐字节一致）

| 项 | 值 |
|---|---|
| 宿主函数 | `NPUModelRunner.execute_model()` |
| **scope 起点** | **L1859** `with record_function_or_nullcontext("prepare input"):` |
| **scope 终点** | **L2098** `update_cos_sin(positions)`（`with` 块的最后一条语句） |
| 顶层语句数 | **3**（L1860 `with self.synchronize_input_prep():` / L2084 `_preprocess` 赋值 / L2098 `update_cos_sin`） |
| scope 之外（不得计入） | L2100 `if self.dynamic_eplb:` 起；L2120 `forward`；L2147 `post process`；L2266 `sample_token` |

**行号级证据**（`data/harness/devdeps_scope_boundary.md`，含末三行原文与缩进判据）：
```python
 2097:             # update global cos, sin
 2098:             update_cos_sin(positions)  # <-- scope 末行
 2100:         if self.dynamic_eplb:  # <-- 缩进回到 8 空格 → scope 之外
```

### 5.1 scope 内包含的子步骤（`devdeps_scope.json` 的 1932 条拍平步骤 / 262 个设备交互点）

1. `synchronize_input_prep()`（L1860，定义 `vllm/v1/worker/gpu_model_runner.py:3810`）
   —— 入口即 `prepare_inputs_event.synchronize()`，**async scheduling 下的真同步点**
2. `_update_states`（L1881 → ascend L816 → 父类 `gpu_model_runner.py:1169`）
3. `_start_dump_data` / `_execute_mm_encoder` / `_finalize_dump_data`（仅 EC transfer 时）
4. 空 batch 早退（`EMPTY_MODEL_RUNNER_OUTPUT` / `kv_connector_no_forward`）
5. **`_prepare_inputs`**（L1929 → L883）：`commit_block_table` → `req_indices` → `_build_attn_state` →
   `_get_cumsum_and_arange` → `positions` → `_compute_prev_positions` → `token_indices` + `index_select` →
   `query_start_loc` → `optimistic_seq_lens_cpu` → `discard_mask` → `num_accepted_tokens` →
   `num_computed_tokens` → `req_indices/query_pos/num_scheduled_tokens` 拷贝 →
   `positions`（device 版）→ `seq_lens` → `compute_slot_mapping` → spec metadata → `logits_indices` → lora/pad
6. `_compute_cascade_attn_prefix_lens`（仅 cascade 时）
7. `_determine_batch_execution_and_padding`（L1949 → L2776）
8. `maybe_create_ubatch_slices` / `dynamic_eplb` / mamba-align / compress 分支（默认配置不触发）
9. `_pad_query_start_loc_for_fia`（L2056 → L834，仅 FULL graph / SP）
10. `_build_attention_metadata`（L2065 → L2875）
11. `_sanitize_placeholder_input_ids_for_forward`（L2079）
12. `_preprocess`（L2084 → `gpu_model_runner.py:3490`）
13. **`update_cos_sin(positions)`**（L2098 → `vllm_ascend/ops/rotary_embedding.py:144`）

> 归因提醒：`update_cos_sin` 是**设备侧**热点（3 个 kernel：index_select/repeat/copy 到全局 rope cache），
> 但它在 scope **之内**，所以做"prepare_input 占比"时必须说明它是被算进去的。

## 6. D. 子步骤矩阵（`data/harness/devdeps_matrix.md`，35 行）

每行含：`无卡可执行? / 需要的 shim / 真机 CPU 侧行为 / shim 后偏差方向 / 风险`。
统计：**32 行可执行（含 4 行"手工补装"）/ 1 行不可执行（Triton launch，需 CPU 替换）/ 11 行偏差方向含"低估"**。

### 6.1 偏差方向汇总（对 CPU 负载特征的影响）

| 方向 | 条目 | 原因 |
|---|---|---|
| **低估**（harness 比真机轻） | `synchronize_input_prep`（event 等待）、`torch.npu.synchronize`、`empty_cache`、`num_accepted_tokens_event.synchronize`、async spec decode 的 D2H、spec metadata 的 pinned 异步拷贝、`_prepare_input_ids` 的异步拷贝、`_sync_device`、block table 异步 H2D 的**重叠**部分 | shim 把所有 event/stream 同步变成 no-op；真机上这些是**阻塞等待**，无卡下没有任何东西可等 |
| **高估**（harness 比真机重） | `CpuGpuBuffer.copy_to_gpu`（大 buffer）、device 侧 `positions`/`seq_lens`/`logits pad`/`masked_fill` 等表达式 | 真机这些是**设备 kernel**（CPU 只付 launch，~µs），harness 里变成 CPU 真的搬内存 |
| **不确定** | `compute_slot_mapping`（Triton launch）、`zero_block_ids`（Triton launch） | 真机 = Python launch 开销 + 设备并行；harness = numpy 向量化。**必须用 `set_launch_cost_us()` 标定后再比** |
| **中性** | `req_indices`/`positions_np`/`token_indices`/`_build_attn_state`/`_compute_prev_positions`/plan 决策等纯 numpy/Python 记账 | 本来就是 CPU 工作，shim 不变 |
| **分支敏感（不是方向问题，是可比性问题）** | `_determine_batch_execution_and_padding`、`lmhead_tp_enable()`、`_may_reorder_batch` | shim 必须伪造 `is_available=True` / `device_count=1` / 设备属性，并补装 ascend config 与 KVCacheConfig；**填错会导致走与真机不同的分支**，此时占比数字不可比 |

### 6.2 已知无法复现的三类（必须在 docs/06 登记）

1. **真实同步等待**（event/stream synchronize）→ 一致性地低估 CPU 时间，且**不改变热点形状**（热点函数仍是同一批）。
2. **H2D/D2H 的 DMA 语义**（pinned + `non_blocking`）→ 方向取决于 buffer 大小与是否 pinned，见上表。
3. **设备 kernel 的 CPU 侧 launch 成本** → 目前为 0（`launch_cost_us` 默认值），需真机实测回填。

## 7. 被 shim / CPU 近似 / 跳过的东西（逐条登记，纪律要求）

| # | 被替换 / 近似的对象 | 替换成什么 | 影响方向 | 登记位置 |
|---|---|---|---|---|
| 1 | device 字符串 `npu` / `npu:N` | `cpu` | 中性（结构不变，只是数据落点） | apilist `device 重映射` |
| 2 | `pin_memory=True` | `False` | 大 buffer 时**高估**；pageable 场景**低估** | apilist `pin_memory=True` |
| 3 | `Tensor.npu/npu_/to/cuda` | `.to('cpu')` / no-op | 中性 | apilist `Tensor.npu()/.to('npu')` |
| 4 | `torch.npu.Event` | `CpuShimEvent`（record/wait/synchronize 全 no-op） | **低估**（真机同步点消失） | apilist `torch.npu.Event` |
| 5 | `torch.npu.Stream` / `torch.npu.stream` | `CpuShimStream` / `nullcontext` | 轻度**低估** | apilist `torch.npu.Stream/stream` |
| 6 | `torch.npu.synchronize` | no-op | **显著低估** | apilist `torch.npu.synchronize` |
| 7 | `torch.npu.is_available()` / `device_count()` | `True` / `1`（**与真机无卡事实相反**） | 分支敏感：不伪造会走 fallback 分支 | apilist `device_count/is_available` |
| 8 | `torch.npu.get_device_name/properties` | 假字符串 / `_DummyDeviceProps` | 分支敏感（影响 `num_sms`、显存常量） | apilist 同名条目 |
| 9 | `torch.npu.mem_get_info` | 常量 `(1<<40, 1<<40)` | 中性 | apilist 同名条目 |
| 10 | `torch.accelerator.*` | no-op / 常量 | **低估**（`_sync_device` 等） | apilist `torch.accelerator.*` |
| 11 | `triton.backends.ascend.driver.NPUUtils.get_arch/get_aicore_num/get_device_properties` | 常量 `Ascend910B3` / 24 / 24 | 中性（只在 import 期问架构） | apilist `triton...get_current_target` |
| 12 | **`_compute_slot_mapping_kernel`（Triton launch）** | `runner/triton_cpu.py` 的向量化 numpy 等价实现 | **不确定**（需 `launch_cost_us` 标定） | apilist + matrix 重点条目 |
| 13 | `torch_npu` 模块本身 | **不替换**（真模块，无卡下 import 安全） | 无 | apilist `import torch_npu` |
| 14 | `torch_generator` | `torch.Generator` 的**子类**（非函数包装） | 无（纯 import 期约束） | apilist `torch.Generator` |
| 15 | 分布式 group | 手工桩（world_size=1 / is_last_rank=True） | 分支敏感 | matrix `【装配】分布式 group 桩` |
| 16 | `init_ascend_config` | 手工补装 | 分支敏感（`need_timing`/`enable_enpu`） | matrix `【装配】init_ascend_config` |
| 17 | `KVCacheConfig` | 手工构造最小对象 | 分支敏感 + **可能静默污染 slot_mapping 数值** | matrix `【装配】KVCacheConfig` |
| 18 | `acl.*` | **不拦截**（`import acl` OK，`acl.init()` 返回 500000） | 无（prepare_input 链不调用它） | apilist `acl.*` |

## 8. 交付物清单与复现命令

### 8.1 脚本（`harness/probes/`，全部带 `--help`，可重跑）

| 脚本 | 作用 |
|---|---|
| `import_gradient.py` | A 组导入梯度（16 step），`--list` / `--only` / `--set-env` / `--suffix` |
| `scope_prepare_input.py` | E 组 scope 边界（AST），输出 `devdeps_scope.json` + `devdeps_scope_boundary.md` |
| `smoke_shim.py` | D 组冒烟（17 用例），输出 `devdeps_shim_smoke.json` |
| `probe_dynamic_hits.py` | C 组动态计数（含 triton_cpu 接入与 `--no-triton` 对照） |
| `shim_patch_proposal.py` | 独立参考实现 + 13 用例自测（证明"三层 patch 才够"） |
| `gen_devdeps_tables.py` | 生成 apilist / matrix / manifest（静态扫描 + 动态合并） |
| `collect_manifest.py` / `collect_host_manifest.sh` | 容器内 / 宿主机 manifest |
| `push.sh` / `pull_data.sh` | **只推自己的路径**、只拉小文件（`--stacks` 才拉全量日志） |

### 8.2 复现（一条命令序列）

```bash
# 从开发机
bash agents/replay_harness/harness/probes/push.sh push
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/probes/collect_host_manifest.sh'
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "python harness/probes/collect_manifest.py"'
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "python harness/probes/import_gradient.py"'
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "python harness/probes/smoke_shim.py"'
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "python harness/probes/probe_dynamic_hits.py"'
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "python harness/probes/scope_prepare_input.py --out /work/data/harness/devdeps_scope.json --boundary-out /work/data/harness/devdeps_scope_boundary.md"'
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "python harness/probes/gen_devdeps_tables.py"'
bash agents/replay_harness/harness/probes/pull_data.sh --stacks
```

### 8.3 产物

| 文件 | 说明 |
|---|---|
| `data/harness/devdeps_apilist.{md,json}` | 37 行 API 清单（静态 × 动态）+ 动态计数明细 |
| `data/harness/devdeps_matrix.{md,json}` | 35 行子步骤矩阵 |
| `data/harness/devdeps_scope_boundary.md` | scope 起止行、顶层语句、全部被调函数、分母定义 |
| `data/harness/devdeps_scope.json` | 1932 条拍平步骤 + 262 个设备交互点（机器可读） |
| `data/harness/devdeps_stacks/*.txt` | A/B/D 全部完整 stdout/stderr（含 40 行栈） |
| `data/harness/devdeps_imports{,_envfix,_order,_devapi}.json` | A 组结构化结果 |
| `data/harness/devdeps_shim_smoke.json` / `devdeps_shim_report.json` / `devdeps_shim_patch_selftest.json` | D 组与 shim 计数 |
| `data/harness/devdeps_dynamic_hits.json` | C 组动态命中 + `triton_cpu.stats()` |
| `data/harness/devdeps_manifest{,_host,_container}.json` | 合并 manifest（镜像 digest/ID/created、命令、时间戳、脚本 sha256、源码 sha256、环境） |

**Manifest 关键字段**（`data/harness/devdeps_manifest.json`，18 个 artifact + 67 条日志索引）：

| 字段 | 值 |
|---|---|
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler` |
| RepoDigest | `sha256:24ae7427b6cad5ee29e0665e6f69a4d51c9f6178035e38f2ed161bb3d29fe81c` |
| Image Id | `sha256:dc9a31b8330d399ad8e91dabaca25798c3d838c36851897bf9a1f77f793072ec` |
| Created | `2026-09-03T17:17:10Z` |
| 宿主 | `host22`（a3-22） |
| 绑核 | `--cpuset-cpus 200-215`（**不使用 120-159**） |
| 网络 | `--network none` |
| NPU 设备 | **不挂任何 `/dev/davinci*`**（`pi-docker.sh` 从不传 `--device`） |
| `pi-docker.sh` sha256 | `fb429117052b…` |
| `model_runner_v1.py` sha256 | `4699af75ac8b4b55283d55190365e93d13820a05888bfa65aea8746deabc4d3a` |
| shim sha256（审计快照） | `2072409af9ce8a854e64f687ee8cfec8b1e60b1ffbab8fa08701f1e9d16a79ad` |
| 脚本 sha256 | `script_sha256` 段（7 个探针脚本逐一记录） |
| 时间戳 | manifest `generated_at` + 每个 step 的 `devdeps_stacks/*.txt` 头部 |

## 9. 给下游的三条硬要求

1. **占比分母**必须用 `devdeps_scope_boundary.md` 的 L1859–L2098 定义；
   `dynamic_eplb` / `forward` / `post process` / `sample_token` 不得计入。
2. 引用 `prepare_input` 的设备依赖时，**必须区分"链内"与"包内"**（`torch.npu.current_stream` 是 1 vs 61）。
3. 与真机比时间前，必须先用真机数据回填 `triton_cpu.set_launch_cost_us()`
   与 `pin_memory` 偏差，否则 `compute_slot_mapping` 与 `copy_to_gpu` 两类会引入系统性偏差。

## 10. 未解决 / 待他人接手

| 项 | 说明 | 建议 owner |
|---|---|---|
| `launch_cost_us` 真机标定 | 需要真机 `perf` 出 `_compute_slot_mapping_kernel[(grid,)](...)` 那一行的 CPU 时间 | prof_measure / measurement |
| `pin_memory` 偏差量化 | 需要真机对比 pinned 与非 pinned 的 `copy_to_gpu` CPU 时间 | prof_measure |
| `compute_slot_mapping` 数值逐值对拍 | harness 的 numpy 实现 vs 真机 triton kernel 输出（含 `BLOCKS_PER_KV_BLOCK>1`、CP world_size>1 分支） | replay_harness + measurement |
| prepare_input 主路径的运行时 API 逐步计数 | 表格已预留 `runtime_keys` 列，runner 稳定后回填 `devdeps_apilist.json` | replay_harness |
| `_build_attention_metadata` 内部（scope 内代码量最大的一段）的逐分支审计 | 本次只验证到"可构造 + scope 正确"；MLA/DSA/GDN 各 builder 的设备交互未逐条展开 | 后续 agent |
