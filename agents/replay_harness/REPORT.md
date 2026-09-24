# replay_harness 交接报告

> 任务：构造**不挂 NPU 卡**也能跑的负载，复现 vLLM 0.26.0 + vllm-ascend 0.26.0rc1
> `prepare_input` 在真机上的 CPU 侧特征。
> 完成时间：2026-09-24（Asia/Shanghai）

---

## 1. 结论先行

**做到了。** 不挂任何 `/dev/davinci*` 的情况下，真实的
`NPUModelRunner._update_states()` + `NPUModelRunner._prepare_inputs()`
可以逐行原样执行，`AttrAudit` 为空、无未 shim 的设备操作、无静默跳过。

核心发现（反直觉）：**无卡可行性的障碍不在 import 层。**
`torch_npu` / `vllm` / `vllm_ascend` 在无卡容器里都能 import 成功；
只有**设备操作**会崩（`aclInit 507008`）。因此 shim 只需覆盖三类设备操作，
而不需要伪造整个 `torch_npu`。

---

## 2. 交付物

| 路径 | 内容 |
|---|---|
| `harness/README.md` | 使用说明、三层 shim 原理、保真度边界 |
| `harness/scripts/pi-docker.sh` | **唯一运行入口**：无设备 + `--network none` + 绑核 200-215 + 身份修复 |
| `harness/scripts/sync.sh` | 本地 ↔ a3-22 同步 |
| `harness/pi_harness/env.py` | bootstrap（shim → import vllm → 打桩分布式 → import runner） |
| `harness/pi_harness/shim/__init__.py` | **三层 shim** + `report()` 记账 |
| `harness/pi_harness/runner/build.py` | `object.__new__(NPUModelRunner)` + 手工装配 + `AttrAudit` |
| `harness/pi_harness/runner/synth.py` | 合成 `SchedulerOutput`（语义对齐真 scheduler） |
| `harness/pi_harness/runner/triton_cpu.py` | slot-mapping kernel 的向量化 numpy 等价实现 + 三种模式 |
| `harness/pi_harness/runner/timer.py` | 子步骤计时（类上 wrap，不改 vLLM 源码） |
| `harness/pi_harness/runner/replay.py` | step 驱动（`_one_step` / `_one_step_with`） |
| `harness/pi_harness/runner/cli.py` | `python -m pi_harness.runner.cli`（含 `--sweep`） |
| `docs/06-synthetic-load.md` | 设计 + 保真度证据 + **D1–D10 已知偏差表** |
| `data/harness/micro_parts.json` | 920B 单次小操作固定开销（理解 CPU 负载的微观基础） |
| `data/harness/hw64_sweep_*.csv` | 6 个维度的参数扫描原始数据 |
| `agents/probe_devdeps/*` | 设备依赖清单（子代理，见下） |
| `agents/sweep_analysis/*` | 扫描分析（子代理，见下） |
| `agents/trace_schema/*` | record & replay 数据层 A/B（子代理，见下） |
| `agents/prof_measure/*` | PMU 采集链路（子代理，见下） |

---

## 3. 关键工程发现（踩过的坑，都已解决并留档）

1. **容器内 uid 1003 无 passwd 条目** → `getpass.getuser()` 抛 `KeyError` →
   torch inductor `default_cache_dir()` 崩。**必须在 `docker run` 里给
   `-e USER/-e LOGNAME/-e TORCHINDUCTOR_CACHE_DIR/-e TRITON_CACHE_DIR`。**
   这个坑与"有没有卡"无关，很容易被误判成"无卡跑不起来"。
2. **Vllm-ascend 内部循环导入**：直接 `import vllm_ascend.worker.model_runner_v1`
   → `device_op` ↔ `ops.fused_moe` 循环。修法：先 `import vllm_ascend.ops`。
3. **`torch.Generator` 必须用子类替换**：transformers 里有
   `torch.Generator | None` 这种 **import 期求值**的注解，函数替换会
   `TypeError: unsupported operand type(s) for |`。
4. **triton-ascend 在 import 期查设备 arch**（`NPUUtils.get_arch()` → `SystemError`），
   且 `BlockTable.compute_slot_mapping()` **每步**都要发一次 Triton kernel launch。
   这是 prepare_input 主路径上最重要的设备交互点之一。
5. **`execute_model` 里 `"prepare input"` scope ≠ `_update_states + _prepare_inputs`**：
   真 scope 还含 `_determine_batch_execution_and_padding`、`_build_attention_metadata`、
   `_preprocess`、`synchronize_input_prep`。**harness 的 P1 数字不能直接与
   LiteProfiler 的 scope 数字比**（`docs/06` §6 有明确口径说明）。
6. **合成负载必须遵守 vLLM 的调度器契约**，否则会撞 vLLM 的隐式断言：
   (a) 同一 step 不能既 `finished` 又 `scheduled`（完成事件要延后一步上报）；
   (b) 新请求只有真的拿到 token 预算才能进 `scheduled_new_reqs`。
   两条都写进了 `synth.py` 的注释。

---

## 3.5 交付后新增的两条工程发现（补记）

7. **`substep_*.csv` 曾经写的是"累计值"而不是"每步增量"**（我自己的 bug，已修）。
   症状：对列求和会得到 `80083 s` 这种荒谬数字，而 `summary.json` 里的占比却是对的
   （因为占比用的是累计值之比）。修法：`run()` 里记录 `timer.inclusive` 的**逐步差分**
   再写 CSV。**所有引用 `substep_*.csv` 的下游分析都必须用修复后的数据。**
   已修复版本的数据在 `data/harness/substep/tmp_bs16b/`。

8. **无卡容器起法的环境变量清单必须单一真源**。`pi_harness/profiling/collect.py`
   自己拼 `docker run` 时漏了 `USER/LOGNAME/TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR`，
   导致进程在 ~3 秒内 `exit=1`（`KeyError: 'getpwuid(): uid not found: 1003'`），
   而症状表现成"PMU 挂不上、pid 不存在"——**极易误判成负载太短**。
   修法：新增 `harness/scripts/pi_env.sh` 作为唯一真源，
   `PI_DOCKER_ENVS[@]` + `pi_env_selfcheck()`，`pi-docker.sh` 与 profiling collector 共用。

## 3.6 ⚠️ 交付后补记：真机口径必须先对齐（最关键的一条）

**发现的重大保真度缺口**：真机 launcher（`scripts/launch_phase_service.sh` 默认值）
跑的是 `--max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048
--no-enable-prefix-caching`，而我最初给 sweep/profiling 的口径是
`max_num_reqs=64 + max_model_len=模型自身 262144`。

后果（`InputBatch.token_ids_cpu_tensor` 形状 = `(max_num_reqs, max_model_len)`）：

| 口径 | 形状 | 体积 | `max_num_blocks_per_req` | block table |
|---|---|---|---|---|
| 真机 | (8, 2048) int32 | **64 KB**（L1/L2 常驻） | **16** | 4 KB |
| 误用 | (64, 262144) int32 | **67 MB**（每步 DRAM） | 2048 | 512 KB |

而 `_prepare_inputs` **每步**都要 `torch.index_select(token_ids_cpu_tensor.flatten(), ...)`
+ `commit_block_table()`，所以这两套配置的 cache 行为完全不同，
topdown / IPC / 热点**不可能对上**。

**修法**：`RunnerConfig.preset` + `--preset {realmachine,stress,modelmax}`，
默认 `realmachine`（= 真机 launcher 口径）。`realmachine` 实测基线：
`--batch 1 --isl 128 --osl 64 --steady-seconds 12` → 27 947 步，
`prepare_inputs_us` p50 = **364.8 µs**、`update_states_us` p50 = **17.1 µs**、scope = **379.1 µs/步**。

**教训**：做 CPU 侧负载复刻时，"引擎启动参数"必须**逐字对齐真机**，
尤其是任何决定持久 buffer **形状**的参数（`max_model_len` / `max_num_reqs` /
`max_num_batched_tokens` / `block_size`）。这些参数的影响不是线性的，而是
**cache 层级的跃变**。

## 4. 保真度：已复现 vs 未复现

**已复现**：真实 Python 代码路径、真实 `NPUInputBatch`/`MultiGroupBlockTable`/
`CpuGpuBuffer` 数据结构、真实尺寸参数链（`max_model_len` → `token_ids_cpu` 宽度 →
`max_num_blocks_per_req` → `commit_block_table()` 每步搬运字节数）、真实的
chunked-prefill / decode / spec-decode 分支选择。

**未复现（逐条在 `docs/06` §5 登记，含影响方向）**：H2D/D2H DMA（**低估**，
但按字节记账）、Triton launch 的 Python 开销（可注入）、device op 内核态开销
（**低估**）、P>1 集合通信（**抛错**，不静默）、`Event.synchronize()` 的真实等待
（**低估**）、权重加载/KV 分配/图捕获（中性）。

**反静默跳过审计**：`AttrAudit`（空）、`readonly_attrs_skipped`（空）、
`UNSHIMMED_DEVICE_OP`（0 次）、`FakeGroupCoordinator`（0 次）、
`_one_step` 不变式断言（已抓出 1 个合成器 bug）。

---

## 4.5 最终一致性结论（**全部判据已结算**）

详见 `docs/06-synthetic-load.md` §9。摘要：

| 判据 | 目标 | 实测 | 结论 |
|---|---|---|---|
| topdown 主 bound | 一致 | 两侧都 **frontend-bound** | ✅ |
| topdown 分量 | ±2~3 pp | frontend **72.1%** vs 56–65% | ⚠️ 偏乐观 7–16 pp |
| IPC | ±10% | **0.949** vs 0.719–0.890 | ⚠️ 偏乐观 7–32% |
| **热点 top-20 重合** | **≥80%** | **16/20 = 80%** | ✅ |
| top-1 一致 | — | `_PyEval_EvalFrameDefault` **11.43%** vs **10.33%** | ✅（Δ 1.1 pp） |
| 扁平度 | 同形 | top-10 **21.4%** vs **23.7%** | ✅ |
| 逐 step 分布 | 同形 | 口径不同，待真机子 scope | ⏳ |

**推荐给使用者的表述**：
> 无卡 harness 与真机处于**同一性能 regime**（frontend-bound、IPC<1、
> CPython 解释器主导的极平热点、top-20 重合 80%）。
> **定量上 harness 偏乐观 7–32%**，偏差方向已量化并归因
> （紧循环 cache 更友好 / 缺步间 ZMQ 与锁 / 同步点 no-op），
> **全部指向"真机 CPU 压力只会更大"**。
> 因此用 harness 做"优化前后的相对比较"是可靠的；用它的绝对 µs 预测真机时需要乘一个修正系数。

## 5. 实测基线（供后续引用）

`--batch 8 --isl 512 --osl 16 --chunk-size 512 --max-num-reqs 16 --max-num-batched-tokens 4096`，
14 个稳态 decode step：

| 指标 | p50 |
|---|---|
| `update_states_us` | 43.7 µs |
| `prepare_inputs_us`（含 numpy 兜底） | 607.8 µs |
| 其中 harness 独有 `triton_cpu_us` | ~133 µs |
| **`pi_net_us`** | **~475 µs** |

920B 微观特征（`data/harness/micro_parts.json`）：**单次小操作固定开销 2–9 µs**
（`sm[:8].copy_()` = 8.6 µs、`np.cumsum(8)` = 4.1 µs、`Tensor.numpy()` = 1.9 µs），
一组 8 个 numpy 小操作 ≈ 21 µs。`_prepare_inputs` 里有几十处这种小操作
→ **CPU 侧负载的微观结构是"固定调用开销 × 操作条数"，而不是数据量**。

---

## 6. 怎么跑

```bash
# 同步
bash agents/replay_harness/harness/scripts/sync.sh push

# 一次负载
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh \
  "cd /work/harness && PI_MODEL_TMP=/tmp/pi_models python -m pi_harness.runner.cli \
     --batch 32 --isl 2048 --osl 128 --steps 300 --has-gdn --out /work/data/harness --tag demo"'

# 参数扫描
... --sweep isl --steps 80 --max-num-reqs 64 --max-num-batched-tokens 16384 --has-gdn

# PMU / 火焰图（必须 noop 模式，避免 numpy 兜底变成假热点）
... --slot-mapping-mode noop --no-timing --batch 16 --isl 1024 --steps 200
```

---

## 7. 做不到 / 待办

| 项 | 状态 | 说明 |
|---|---|---|
| P1+：把 `_build_attention_metadata` / `_preprocess` 也纳入 | **未做**（超出 P1 定义） | 需要真 attention backend 的 metadata builder；`docs/06` §6 已写明口径差异 |
| 真机 record & replay（P2 的 `real` 组） | 交给 `trace_schema` | 需要 `service_bringup` 的 trace |
| 与真机 topdown/IPC/热点 top-20 的一致性对照 | **待回填** | 依赖 `prof_measure`（无卡侧）与 `measurement_profiling`（真机侧） |
| `--triton-launch-us` 的标定值 | **待回填** | 需要真机实测 Triton launch 的 Python 侧开销 |
| DMA 字节数 → 耗时的换算 | **待回填** | 需要真机实测 H2D 带宽与调用开销 |
| 宿主机噪声控制 | 部分 | a3-22 有 64 个他人 python 进程，affinity 覆盖 0-639；`taskset` 挡不住。必须 `--repeat ≥3` + 前后跑 `hostnoise_gate.sh` |

**已知的实测噪声问题（重要）**：初版 `--sweep batch` 数据非单调
（batch=1 → 834 µs，batch=8 → 487 µs），量级差异超出软件原因，判定为宿主机噪声。
在引用扫描数字前必须用多轮重复的统计量（`sweep_analysis` 正在做）。

---

## 8. 上传打包建议（本 agent 视角）

按 `plan/COORDINATION.md` §5「原始 perf.data 只在 a3-22 留档，不上传」，
打包上传前建议排除以下内容（我已处理本地副本）：

| 路径 | 大小（处理前） | 处理 | 理由 |
|---|---|---|---|
| `data/harness/prof_runs/*/passes/perf/perf.data` | 419 MiB（8 份） | ✅ 已从本地删除（a3-22 保留） | 原始 perf 数据，约定不上传 |
| `data/harness/prof_runs/*/passes/perf/symfs/` | ~? | ✅ 已从本地删除（a3-22 保留） | 符号化用的库副本，可重建 |
| `agents/replay_harness/src_snapshot/` | 81 MiB | ✅ 已删除 | 与 `refs/{vllm,vllm-ascend}` 完全重复 |
| `data/harness/trace_synth_*.jsonl`（10 个） | ~37 MiB | ⚠️ 建议只留 2–3 个代表性样本 | 可由 `agents/trace_schema/` 的脚本 + manifest 复现 |
| `agents/drivers_analysis/.conda/`（如存在） | 1.1 GB | ⚠️ 确认已排除 | conda 环境，可重建 |
| `data/toolchain/raw/`、`data/profiles/` | — | 保留 | 是证据链的一部分（不含 perf.data） |

处理后本地交付包：`data/` 113 MiB、`figures/` 3.7 MiB、`docs/` 248 KiB、`agents/` 15 MiB。
