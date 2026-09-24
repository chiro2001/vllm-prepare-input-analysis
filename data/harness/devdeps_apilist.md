# prepare_input 设备依赖 API 清单（静态 × 动态交叉验证）

> 生成时间：2026-09-23T18:15:06+0000　|　脚本：`harness/probes/gen_devdeps_tables.py`　|　scope：`model_runner_v1.py` L1859-2098 (`execute_model`)
>
> 静态扫描范围：**链内** = prepare_input scope 的直连语句 + 其可达被调函数（AST 展开 ≤2 层，覆盖 20 个文件）；**包内** = `/work/refs/vllm-ascend/vllm_ascend` + `/work/refs/vllm/vllm/v1` 全量。
>
> 运行时命中：`harness/probes/probe_dynamic_hits.py`（计数代理）+ `probes/shim_patch_proposal.py` 自测；**prepare_input 主路径的逐步计数需要 runner（replay_harness）产出后再回填**。

| API | 出现位置 (file:line) | 静态命中 (链内/包内) | 运行时命中 | 类别 | shim 行为 | 风险 |
|---|---|---|---|---|---|---|
| `torch.npu.current_stream` | model_runner_v1.py:2674 | 1 / 61 | 2 (probe_dynamic_hits.py, shim.report()) | 设备交互 | 返回单例 CpuShimStream；`with torch.npu.stream(s)` 退化为 nullcontext（计数） | 真机只是取 stream 对象（亚 µs，无同步）；shim≈0 → 中性。若代码依赖 stream 之间的可见性/同步语义，会丢失顺序（无卡下无在飞 kernel，可接受） |
| `torch.npu.Event` | fused_moe.py×9, model_runner_v1.py×7, cpu_npu.py×6 | 0 / 58 | 3 (probe_dynamic_hits.py, shim.report()) | 设备交互 | CpuShimEvent：record/wait/synchronize/query 全 no-op（计数） | `synchronize_input_prep` 每步 `prepare_inputs_event.synchronize()` 真机可能阻塞（async scheduling 下是主要隐藏同步点）；shim=0 → **低估** |
| `torch.npu.Stream` | utils.py×6, model_runner_v1.py×6, worker.py×4 | 0 / 40 | 2 (probe_dynamic_hits.py, shim.report()) | 设备交互 | CpuShimStream（计数） | 真机构造 stream 有一次性 ctx 成本；shim≈0 → 轻度低估 |
| `torch.npu.stream` | model_runner_v1.py×7, attention_v1.py×3, packed_tensor.py×3 | 0 / 25 | 2 (probe_dynamic_hits.py, shim.report()) | 设备交互 | contextlib.nullcontext（计数） | 真机 stream 切换 + 事件依赖 ~µs 级；shim=0 → 低估（prepare_input 主路径不在 stream 内，影响主要在后处理/采样） |
| `torch.npu.synchronize` | worker.py×3, model_runner_v1.py×3, npu_ipc_engine.py×2 | 0 / 14 | 1 (probe_dynamic_hits.py) | 设备交互 | no-op（计数） | 真机是整设备同步，通常是大幅 CPU 阻塞；shim=0 → **显著低估**，必须靠 PI_OP_COST_* 或真机数值补 |
| `torch.npu.current_device` | sequence_parallelism_moe.py×4, comm_utils.py×4, sequence_parallelism.py×3 | 0 / 24 | 0 | 设备交互 | 常量 0（计数） | 真机第一次会触发 aclInit（一次性百 ms 级）；稳态≈0 → 中性 |
| `torch.npu.set_device` | worker.py×2, worker_310p.py×2, mooncake_backend.py×2 | 0 / 16 | 0 | 设备交互 | no-op（计数） | 真机首次 set_device 触发 aclInit/ctx 建立（一次性），后续接近 0；shim 把一次性成本抹掉（对单步稳态影响可忽略） |
| `torch.npu.empty_cache` | sfa_v1.py×2, rfork_loader.py×2, camem.py×1 | 0 / 13 | 1 (probe_dynamic_hits.py) | 设备交互 | no-op（计数） | 真机若真的回收会阻塞（十几 ms 级）；shim=0 → 低估，但只在显存压力路径上 |
| `torch.npu.mem_get_info` | sleep_mem_optimized.py×4, worker.py×2, model_runner_v1.py×2 | 0 / 12 | 1 (probe_dynamic_hits.py) | 设备交互 | 返回常量 (1<<40, 1<<40)（计数） | 真机是驱动查询（µs 级）；shim 常数返回，**不影响 CPU 负载形状** |
| `torch.npu.memory_stats/allocated/reserved/reset_peak` | worker.py×5, patch_torch_accelerator.py×3, platform.py×2 | 0 / 11 | 0 | 设备交互 | 常量 / no-op（计数） | 低风险（统计口径，不进 prepare_input 主路径） |
| `torch.npu.device_count / is_available` | worker.py×4, worker_310p.py×2, modelslim_config.py×1 | 0 / 8 | 0 | 设备交互 | shim 返回 1 / True（**与真机无卡事实相反**，为的是让上层走正常分支） | 重要：真机 is_available=True、device_count>0；shim 必须伪造为可用，否则 vLLM 会走 fallback 分支，改变代码路径。必须在报告里显式登记 |
| `torch.npu.get_device_name / get_device_properties` | platform.py×4 | 0 / 4 | 0 | 设备交互 | 返回假名字字符串 / _DummyDeviceProps | 真机返回值会决定 num_sms、显存容量等；**shim 必须给出与目标卡一致的常量**，否则 cudagraph/batch padding 决策会变 |
| `torch.npu.is_current_stream_capturing` | model_runner_v1.py×2, aclgraph_utils.py×1 | 0 / 3 | 0 | 设备交互 | False（计数） | 无卡 harness 不做 graph capture；真机 eager 步也返回 False → 中性 |
| `torch.npu.ExternalEvent / NPUGraph / graph_*` | attention_v1.py×15, mla_v1.py×5, encoder_acl_graph.py×5 | 0 / 49 | 0 | 设备交互 | CpuShimEvent / 空类 / no-op | 只在 aclgraph 采集路径使用（本 harness 不开 graph）→ 低风险 |
| `torch.npu.config.allow_internal_format` | model_runner_v1.py×1 | 0 / 1 | 0 | 纯CPU | `model_runner_v1.py:224` 是**只写**（`torch.npu.config.allow_internal_format = True`），写入能过；真实 `_npuConfig` 对象**可写不可读**（读会 AttributeError，实测 C16） | **低**：源码里只有这一处写入、没有读取点（已全仓 grep 确认）；shim 若替换 config 对象仍须保证 `allow_internal_format` 可写、不要引入读取依赖 |
| `torch.npu.*（模块级属性读取）` | camem.py×4, pyhccl.py×3, platform.py×2 | 0 / 14 | 0 | 设备交互 | shim 覆盖不到的属性读取会 AttributeError（见 report().warnings） | **审计钩子**：这条用于发现新出现的、未被 shim 登记的 API |
| `torch.accelerator.synchronize` | gpu_model_runner.py:7811, gpu_model_runner.py:7817 | 2 / 23 | 1 (probe_dynamic_hits.py) | 设备交互 | no-op（计数）；参考补丁也接管 | `GPUModelRunner._sync_device()` 走它（execute_model 的 profiling 分支，在 scope 之前）；shim=0 → 低估 |
| `torch.accelerator.* 其它` | gpu_model_runner.py×11, gpu_worker.py×9, patch_torch_accelerator.py×7 | 0 / 58 | 0 | 设备交互 | 参考补丁把 current_device_index/set_device_index/device_count/current_accelerator/empty_cache/memory_* 全部接管 | 未接管时 `torch.accelerator.current_device_index()` 直接 aclInit 507008（实测） |
| `import torch_npu` | worker.py×2, netloader_pg.py×1, packed_tensor.py×1 | 0 / 5 | 0 | 纯CPU | **不伪造**，直接用真实模块（无卡下 import 成功，仅 warning） | 真实 import 有 ~2-4 s 一次性开销；不影响单步负载 |
| `torch_npu.npu / torch_npu.utils` | elastic_load.py×6, utils.py×5, sparse_kv_offload_manager.py×5 | 0 / 31 | 0 | 纯CPU | 保留真实模块；`torch_npu.utils.get_cann_version()` 实测可返回 '9.1.0' | 低（导入型依赖） |
| `torch_npu.profiler.dynamic_profile` | — | 0 / 0 | 0 | 设备交互 | 真实模块可 import；`dp.start/stop` 需 devprof，未在无卡下调用 | vllm_ascend.worker.worker 顶层 import 它（会加载 profiler 库，一次性 ~百 ms） |
| `torch_npu.op_plugin.atb._atb_ops` | worker.py×1 | 0 / 1 | 0 | 设备交互 | 真实模块可 import（实测 OK） | 注册 ATB 算子扩展；import 期无设备操作 |
| `torch_npu._inductor` | worker.py×1 | 0 / 1 | 0 | 设备交互 | **不可 import**（无卡下 `triton ... get_arch` SystemError） | 只在 inductor 编译路径用；本 harness 必须不走 torch.compile |
| `torch_npu.multiprocessing.reductions` | packed_tensor.py×1, npu_ipc_engine.py×1 | 0 / 2 | 0 | 纯CPU | 真实模块可 import | 低（只有权重传输/ipc 路径用） |
| `acl.*` | camem.py×1 | 0 / 1 | 0 | 设备交互 | 不拦截；实测 `import acl` OK，`acl.init()` 返回 500000（无设备） | vllm_ascend 的 prepare_input 链**不直接调用 acl**（静态命中 0）；若后续引入，需要显式登记 |
| `CANN 环境变量 / ASCEND_HOME_PATH` | dispatch_ffn_combine.py×4, dispatch_ffn_combine_w4_a8.py×4, dispatch_ffn_combine_bf16.py×4 | 0 / 132 | 0 | 纯CPU | 保留真实 CANN 目录（镜像自带 cann-9.1.0） | 低 |
| `triton launch（_compute_slot_mapping_kernel）` | ascend: `vllm_ascend/worker/block_table.py:179`（kernel 定义 `ops/triton/compute_slot_mapping.py:12`）；core 同名实现: `vllm/v1/worker/block_table.py:166`（定义同文件:347） | 1 / 174 | 1 (runner/triton_cpu.stats()) | 设备交互 | **必需**：`pi_harness/runner/triton_cpu.py::install()` 把 `_compute_slot_mapping_kernel` 换成 numpy 等价实现（同时替换 `block_table` 模块内的名字绑定），并记账 `launches`/`launch_ns`/`fallback_ms`；`set_launch_cost_us()` 可注入真机 Python launch 开销。**仅靠 shim 的 triton proxy 不够**（它只截 `get_current_target`）：launch 仍走 ascend driver → `LazySetDevice 107001` 崩（实测对照见 `devdeps_dynamic_hits.errors`） | 真机 CPU 侧成本 = Python launch 开销（参数绑定/特化/cache 查找/launcher 调度，需真机标定，初值 10-50 µs/次），设备侧执行与之并行；本替换额外付出 numpy 向量化成本 → 偏差方向**不确定**，必须用 `launch_cost_us` 标定后再比 |
| `triton.runtime.driver.active.get_current_stream（未拦截时的失败点）` | 无直接 vllm 调用点（经 triton runtime 进入）；证据: `data/harness/devdeps_dynamic_hits.json` 的 errors 字段 | 0 / 0 | 0 | 设备交互 | **未拦截**（shim 只截 `get_current_target`） | 实测栈：`triton/runtime/jit.py:572 run()` → `driver.active.get_current_stream(device)` → `triton/backends/ascend/driver.py:259` → `backend_register.py:283 _npu_getCurrentRawStreamNoWait` → `RuntimeError LazySetDevice 107001`。要通用覆盖（不止 slot mapping）必须在这一层做 driver proxy |
| `torch.Generator（**类型注解**依赖）` | vllm-ascend / transformers 侧（`torch.Generator` 注解）；证据：replay_harness 实测（P1 冒烟） | 0 / 32 | 0 | 纯CPU | shim 必须用**子类**替换 `torch.Generator`，**不能**用函数替换：库在 import 期对 `torch.Generator | None` 求值，函数对象没有 `__or__` → `TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'` | **高**：纯 import 期类型系统约束，与设备无关，但会让整条 import 链崩；属于 shim 自身引入的新风险 |
| `init_ascend_config(vllm_config)` | `vllm_ascend/ascend_config.py`；消费点 `vllm_ascend/worker/model_runner_v1.py`（`lmhead_tp_enable()` / `self.ascend_config`） | 1 / 193 | 0 | 纯CPU | 无卡 harness **手工补装**：`_prepare_inputs` 末尾 `lmhead_tp_enable()` 会读 ascend config；真机由 `NPUWorker.init_device()` 建立 | **中**：漏装直接 AttributeError；补装后要确认读到的字段（`scheduler_config.profiling_chunk_config.need_timing`、`enable_enpu`）与真机一致，否则 `execute_model` 顶部的 profiling 分支会走不同路径 |
| `KVCacheConfig（kv_cache_groups）` | `vllm/v1/kv_cache_interface.py`；消费点 `vllm_ascend/worker/model_runner_v1.py::_may_reorder_batch` | 7 / 1026 | 0 | 纯CPU | 无卡 harness **手工构造**最小 `KVCacheConfig`：`_may_reorder_batch` 读 `kv_cache_config.kv_cache_groups`；真机由 `initialize_kv_cache()` 产出 | **中**：`kv_cache_groups[].kv_cache_spec.block_size` 决定 `BlockTable.blocks_per_phys_block` 与 physical/logical 映射；填错会让 slot_mapping 数值错（**静默**） |
| `triton.runtime.driver.active.get_current_target` | — | 0 / 0 | 1 (shim.report()) | 设备交互 | 替换为 cpu target（`backend='cpu'`） | 这是 import 期唯一必须拦截的 triton API（flash_linear_attention 的 ops/utils.py:123） |
| `torch.zeros(..., device='npu') 类 device 重映射` | gpu_model_runner.py:1058, gpu_model_runner.py:1102, gpu_model_runner.py:1700, gpu_model_runner.py:3088, gpu_model_runner.py:3121, gpu_model_runner.py:3585 …(+9) | 15 / 436 | 11 (shim_patch_proposal.py selftest) | 两者都有 | **当前 shim 未覆盖**（实测 10/17 冒烟失败）；`probes/shim_patch_proposal.py` 给出参考实现（factory + Tensor.to/npu + accelerator） | 不覆盖 → prepare_input 第一步 `CpuGpuBuffer(device=npu)` 就崩；覆盖后所有 npu tensor 变 CPU tensor（**有意近似**） |
| `pin_memory=True` | gpu_model_runner.py:1098, gpu_model_runner.py:1701, gpu_model_runner.py:1857, gpu_model_runner.py:1860, gpu_model_runner.py:1876, gpu_model_runner.py:1879 …(+9) | 15 / 322 | 3 (shim_patch_proposal.py selftest) | 两者都有 | 容器无 pinned allocator（实测 `torch.zeros(4, pin_memory=True)` RuntimeError）→ 必须降级为 False 并登记 | 真机 pin 页让 H2D 走 DMA（CPU 侧便宜）；shim 的 CPU→CPU memcpy 由 CPU 真搬数据 → 大 buffer 时**高估**，pageable 场景反而**低估** |
| `CpuGpuBuffer.copy_to_gpu / copy_to_cpu` | gpu_model_runner.py:1780, gpu_model_runner.py:1782, gpu_model_runner.py:1783, gpu_model_runner.py:1834, gpu_model_runner.py:1838, gpu_model_runner.py:1840 …(+23) | 29 / 109 | 0 | 两者都有 | 两块都变 CPU tensor，退化为 CPU→CPU memcpy | 见上条；这是 prepare_input 里**调用次数最多**的设备交互（链内静态 29 次调用点） |
| `Tensor.npu() / .to('npu')` | w4a8.py×5, worker.py×3, fa3_v1.py×2 | 0 / 14 | 11 (shim_patch_proposal.py selftest) | 设备交互 | 需要 device 重映射层；否则穿透到 privateuse1 → aclInit/guard | prepare_input 主路径本身不用（静态 0 命中），但 runner 构造期大量使用 |
| `.cpu / .gpu / .np 缓冲三元组（CpuGpuBuffer）` | gpu_model_runner.py:1026, gpu_model_runner.py:1028, gpu_model_runner.py:1032, gpu_model_runner.py:1034, gpu_model_runner.py:1749, gpu_model_runner.py:1752 …(+137) | 144 / 773 | 0 | 两者都有 | shim 下三者的差异只剩「哪个 Python 对象」；`_np` 与 `cpu` 共享内存的语义必须保持 | 若把 `.np` 与 `.cpu` 解耦（例如复制而非共享），会静默改变正确性（写 np 不生效） |

## 动态实验计数明细（原始 key，未做任何加权）

| key | 次数 | 来源 |
|---|---|---|
| `triton_cpu.launch_ns` | 290519 | runner/triton_cpu.stats() |
| `patch:device_remapped` | 11 | shim_patch_proposal.py selftest |
| `patch:zeros` | 10 | shim_patch_proposal.py selftest |
| `patch:pin_memory_downgraded` | 3 | shim_patch_proposal.py selftest |
| `patch:zeros_like` | 3 | shim_patch_proposal.py selftest |
| `patch:Tensor.to` | 3 | shim_patch_proposal.py selftest |
| `shim:pin_memory->False:zeros` | 3 | shim.report() |
| `shim:CpuGpuBuffer.copy_to_gpu` | 2 | shim.report() |
| `shim:device.map_npu_to_cpu` | 2 | shim.report() |
| `shim:Tensor.to(npu)` | 2 | shim.report() |
| `shim:device.map_npu_to_cpu_str` | 2 | shim.report() |
| `torch.npu.current_stream` | 1 | probe_dynamic_hits.py |
| `torch.npu.Event()` | 1 | probe_dynamic_hits.py |
| `torch.npu.synchronize` | 1 | probe_dynamic_hits.py |
| `torch.npu.mem_get_info` | 1 | probe_dynamic_hits.py |
| `torch.npu.empty_cache` | 1 | probe_dynamic_hits.py |
| `torch.accelerator.synchronize` | 1 | probe_dynamic_hits.py |
| `torch.npu.Stream()` | 1 | probe_dynamic_hits.py |
| `torch.npu.stream` | 1 | probe_dynamic_hits.py |
| `patch:Tensor.npu` | 1 | shim_patch_proposal.py selftest |
| `patch:accelerator.synchronize` | 1 | shim_patch_proposal.py selftest |
| `patch:accelerator.current_device_index` | 1 | shim_patch_proposal.py selftest |
| `patch:accelerator.device_count` | 1 | shim_patch_proposal.py selftest |
| `patch:accelerator.is_available` | 1 | shim_patch_proposal.py selftest |
| `patch:accelerator.current_accelerator` | 1 | shim_patch_proposal.py selftest |
| `patch:accelerator.empty_cache` | 1 | shim_patch_proposal.py selftest |
| `patch:empty` | 1 | shim_patch_proposal.py selftest |
| `patch:tensor` | 1 | shim_patch_proposal.py selftest |
| `patch:arange` | 1 | shim_patch_proposal.py selftest |
| `patch:ones` | 1 | shim_patch_proposal.py selftest |
| `shim:npu.synchronize` | 1 | shim.report() |
| `shim:npu.empty_cache` | 1 | shim.report() |
| `shim:triton.get_current_target` | 1 | shim.report() |
| `shim:Tensor.npu` | 1 | shim.report() |
| `shim:npu.stream` | 1 | shim.report() |
| `shim:Event.record` | 1 | shim.report() |
| `shim:Event.synchronize` | 1 | shim.report() |
| `shim:Stream.synchronize` | 1 | shim.report() |
| `triton_cpu.launches` | 1 | runner/triton_cpu.stats() |
| `triton_cpu.launch_cost_us` | 0 | runner/triton_cpu.stats() |
| `triton_cpu.fallback_ms` | 0 | runner/triton_cpu.stats() |

计数口径：`probe_dynamic_hits.py` 在同一次运行里给 `torch.npu.*` / `torch.accelerator.*` / `torch_npu.*` 全部套了计数代理，所以「运行时命中 = 0」只说明**本轮动态实验没走到**，不代表真机不调用；prepare_input 主路径的逐步计数等 runner（replay_harness）跑起来后回填同一张表。
