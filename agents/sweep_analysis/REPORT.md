# sweep 分析报告：prepare_input 参数影响曲线（无卡 harness，preset 分口径）

> 作者：`sweep_analysis`（2026-09-24，Asia/Shanghai）
> 数据 `data/harness/sweep_*`｜图 `figures/06-sweep-*`｜机器可读 `data/harness/sweep_fits.json`
> 详版摘要同步到 `docs/03-share-in-inference.md` §7.1
> 运行入口：`ssh a3-22` + `harness/scripts/pi-docker.sh`（无卡、`--network none`、cpuset 200-215）

---

## 1. 口径（所有数字必须带这一节的定义）

```
prepare_inputs_us  = NPUModelRunner._prepare_inputs 单函数耗时（harness 原始上报）
update_states_us   = NPUModelRunner._update_states
scope_total_us     = update_states_us + prepare_inputs_us        # harness 默认口径
triton_cpu_us      = harness 独有开销（slot-mapping kernel 的 numpy 等价实现 + 注入的 launch 自旋）

pi_net_us          = prepare_inputs_us - triton_cpu_us    ★ 本报告所有结论用这个
scope_net_us       = scope_total_us   - triton_cpu_us
```

**为什么扣**：真机上这一步是 `_compute_slot_mapping_kernel[(num_reqs+1,)](...)` 的一次 Triton
launch（Python 侧只做参数绑定/特化/cache 查找，然后交给 device）。无卡容器跑不了 kernel，
harness 用 numpy 复刻其数值语义，于是把 device 侧等价时间混进了 `prepare_inputs_us`。
**不变式（已逐 step 确认）**：每个被测 step `triton_launches == 1`（每步 1 次 launch）。

> 注：老版 harness 每步 1 次 launch；新 `preset` 默认 `slot_mapping_mode=noop`，
> 此时 `triton_cpu_us ≈ 0.5 µs`（就是 noop 记账本身的成本）。

## 2. preset 是结论的一部分

| preset | max_model_len | max_num_reqs | max_num_batched_tokens | prefix cache | async | slot_mapping | 用途 |
|---|---|---|---|---|---|---|---|
| `realmachine` | 2048 | 8（扫大 batch 时放开 64） | 2048 | **off** | **on** | **noop** | 与真机 launcher 对照 |
| `stress` | 32768 | 64 | 16384 | on | off | **noop** | 放大斜率 |

`max_model_len` 决定 `InputBatch.token_ids_cpu_tensor` 形状 `(max_num_reqs, max_model_len) int32`：
真机 `(8, 2048)` = 64 KB（常驻 L1/L2）vs 旧误用 `(64, 262144)` = 67 MB（每步从 DRAM 拉）。
同理 `max_num_blocks_per_req`：真机 16，旧误用 2048（block table 4 KB vs 512 KB）。

**旧口径（`max_num_reqs=64 / max_num_batched_tokens=16384 / MML=模型自身`）的原始数据已按
要求重命名为 `data/harness/hw64_sweep_LEGACY_*.csv`（未删除），其结论作废。**

## 3. 数据来源

| 代号 | 文件 | 内容 |
|---|---|---|
| A | `sweep_<tag>_realmachine_*.csv`、`sweep_stress_stress_*.csv` | `preset_grid.py` 产出：含 raw/net/triton/meta，跨轮中位数 |
| B | `sweep_steps_<preset>_<tag>.csv` | 逐 step 池（整体建模用；realmachine 29 648 步） |
| C | `sweep_modecmp_rm_*.csv` + `sweep_modecmp_rm_r*_<mode>_{per_step,substep,summary}` | 交错三模式对照 |
| D | `sweep_mode_*`、`sweep_d_*`、`sweep_grid_*`、`sweep_probe_*` | 旧口径的细节/探针（仅作方法与噪声引用） |
| E | `sweep_noise_*.json` | hostnoise gate 留档 |

固定身份：镜像 `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler` @
`sha256:dc9a31b8330d399ad8e91dabaca25798c3d838c36851897bf9a1f77f793072ec`；
模型 `qwen35-0.8b`；host22；torch 2.10.0+cpu；numpy 1.26.4。

## 4. 结果（全部为 `pi_net_us` p50，已扣 `triton_cpu_us`）

### 4.1 拟合表

| preset | 维度 | `c` (µs) | `a` | R² | n | 判断 |
|---|---|---|---|---|---|---|
| realmachine | batch B=1→64（max_num_reqs=64） | 380.75 | **0.5375 µs/req** | 0.983 | 5 | 底噪主导（B=64 时常数仍占 92%） |
| realmachine | batch B=1→8（max_num_reqs=8） | 371.02 | 0.8468 µs/req | 0.778 | 4 | 底噪主导 |
| realmachine | isl 64→1024（warmup=3 ⇒ decode 窗口） | 377.49 | 0.0055 µs/token | 0.080 | 5 | 平（窗口内无 prefill 步） |
| realmachine | isl 128→1024（warmup=0 ⇒ 含 prefill） | 373.40 | 0.0040 µs/token | 0.979 | 3 | 平（chunk=2048 ≥ ISL） |
| realmachine | `_update_states` vs batch（B=1→64） | 18.00 | **1.6750 µs/req** | 1.000 | 5 | 斜率主导 |
| stress | isl 64→2048 | 388.62 | 0.0065 µs/token | 0.511 | 6 | 平（chunked prefill 摊平） |
| stress | batch 1→64 | 377.20 | 0.7310 µs/req | 0.971 | 7 | 底噪主导 |
| stress | chunk_size 128→8192 | 393.45 | 0.0003 µs/token | 0.031 | 6 | 平 |
| stress | block_size 32→256 | 390.79 | 0.0068 µs/(token/block) | 0.304 | 4 | 平（32→256 净 −13 µs） |
| stress | spec_k 0→3 | 449.21 | **49.08 µs/draft** | 0.578 | 4 | 斜率主导 |
| stress | prefix_hit_ratio 0→1 | 414.85 | **−11.08 µs/单位** | 0.438 | 5 | 弱下行 |
| stress | `_update_states` vs batch | 29.39 | 1.6287 µs/req | 1.000 | 7 | 斜率主导 |

**底噪主导 vs 斜率主导的分界**：除 `spec_k` 外，所有维度在其覆盖区间内都是**常数项主导**
（常数 371–394 µs，占 B=1 时 ~100%，占 B=64 时仍 ~92%）。
`spec_k` 是唯一在区间内就出现斜率主导的维度。

### 4.2 逐 step 池化模型（比逐配置拟合更稳）

对同 preset 的**所有 decode 步**做最小二乘（不先按配置聚合，避免聚合掩盖方差）：

| 池 | n_steps | 因变量 | 自变量 | `c` (µs) | 斜率 | R² |
|---|---|---|---|---|---|---|
| realmachine（B≤8） | 29 648 | `pi_net` | num_reqs | 377.88 | **676 ns/req** | 0.057 |
| realmachine（B≤8） | 29 648 | `_update_states` | num_reqs | 17.22 | **1814 ns/req** | 0.642 |
| realmachine（B≤64） | 942 | `pi_net` | num_reqs | 384.51 | **537 ns/req** | 0.126 |
| realmachine（B≤64） | 942 | `_update_states` | num_reqs | 23.75 | **1670 ns/req** | 0.500 |
| stress（B≤64） | 2 408 | `pi_net` | tokens/step | 349.47 | 1878 ns/token | 0.309 |
| stress（B≤64） | 2 408 | `_update_states` | tokens/step | 11.67 | 2093 ns/token | 0.343 |

**结论（回答"谁的斜率更大"）**：
* realmachine，B≤64：`_update_states` **1670 ns/req** vs `pi_net` **537 ns/req** ⇒ **3.1×**
* realmachine，B≤8：1814 vs 676 ns/req ⇒ **2.7×**
* stress：`_update_states` 2093 ns/token vs `pi_net` 1878 ns/token ⇒ 1.11×
  （stress 下两者都随 token 数走，差距缩小）

⇒ **请求数增长时，"逐请求 Python 循环"（`_update_states`）比"随总 token 数的
`np.repeat`/`index_select`"（`_prepare_inputs`）涨得更快。** 高并发下的优化优先级应给
`_update_states`。R² 偏低是因为逐 step 噪声大，但两组斜率差异远大于噪声带。

### 4.3 chunked prefill 把 ISL 摊平（必须用 min(ISL, chunk) 归一）

* stress：ISL 64→2048 时每步调度 token 恒为 32 ⇒ `pi_net` 斜率仅 6.5 ns/token，
  全是"每步固定 chunk"的底噪；用 `per_step_us / min(ISL, chunk_size)` 读才对：
  `388.6 / 32 = 12.1 ns/token`。
* realmachine（chunk=2048 ≥ ISL≤1024）：ISL 只改变 `token_ids_cpu_tensor` 的**宽度**，
  而 decode 步每请求只取 1 个 token → 斜率 4–5.5 ns/token（本质是 cache footprint 效应，
  不是逐 token 计算）。
* 小 chunk 端有真实上行：stress chunk_size 128 → `pi_net` 402.0 µs，
  该点 13/80 步是 prefill（`tokens/step` 仍显示 32 是中位数掩盖了分布）；chunk≥512 后转平。

### 4.4 block_size 维度：`max_num_blocks_per_req` 与 `commit_block_table`

**stress 口径（MML=32768, block_size 扫描, `sweep_stresssub_stress_20260923-192036.csv`）**：

| block_size | `max_num_blocks_per_req` | 每行字节 | `pi_net` p50 (µs) |
|---|---|---|---|
| 32 | 1024 | 4.0 KB | 393.6 / 402.3（2 轮） |
| 64 | 512 | 2.0 KB | 404.5 / 403.1 |
| 128 | 256 | 1.0 KB | 404.2 / 404.1 |
| 256 | 128 | 0.5 KB | 405.7 / 404.9 |

`block_size` 32→256 让 `max_num_blocks_per_req` 降 **8×**（1024→128，每行字节 4.0 KB→0.5 KB），
但 `pi_net` **没有下降**（393.6–405.7 µs，轮间噪声 8.7 µs）。两个窗口分别拟合为
`a`=0.0068 / R²=0.304（`sweep_stress_stress_*`）与 `a`=0.0247 / R²=0.542（`sweep_stresssub_stress_*`），
**斜率符号两窗一致、量级都在噪声内** ⇒ block_size 是弱因子。

**为什么"搬运字节↑"没兑现成耗时↑**：`commit_block_table` 调 `copy_to_gpu(num_reqs)`，
只搬 **num_reqs 行**（stress 下 32 行 × 1024 int32 = 128 KB→ 32 行 × 128 int32 = 16 KB），
而这里的实测 `commit_block_table` 只有 **24.0 µs/步**（block_size=32、blk/req=1024，
`sweep_stresssub_stress_*` 第一行的 `substeps_us_per_step`）。
即：**它是可测的真实效应，但绝对量级只占 `pi_net` 的 ~6%，对端到端不构成杠杆。**

旧口径（MML=262144）下同样关系的干净证据（`sweep_d_blocksize_*_substep_*.csv`，
该口径已被判作废，仅用于展示"字节→时间"的单调性）：

| block_size | `max_num_blocks_per_req` | 每行字节 | `commit_block_table` µs/步 | `compute_slot_mapping` µs/步 |
|---|---|---|---|---|
| 32 | 8192 | 32.0 KB | **40.89** | 105.34 |
| 64 | 4096 | 16.0 KB | **29.71** | 99.79 |
| 256 | 1024 | 4.0 KB | **25.10** | 119.62 |

⇒ `max_num_blocks_per_req` 与 `commit_block_table` 的耗时**在该口径下确实单调对上**
（32768→4096 行字节 ⇒ 40.89→25.10 µs）；但在真机口径（MML=2048 ⇒ blk/req=16）下
这一项只有几 µs，**不再是杠杆**。

> 方法学限制：harness 的 `SubStepTimer` 只在**每个进程的第一次** replay 上生效
> （`wrap_prepare_input_path` 用 `_pi_timer_wrapped` 去重，而该标志挂在类上、不会随新
> `SubStepTimer` 复位）。因此 `sweep_stresssub_stress_*.csv` 只有第一行有 `substeps_us_per_step`，
> 其余行为空 —— 这是 harness 的既有行为，未修改。

### 4.5 spec_k / prefix_hit_ratio

* `spec_k`：0→1 时 `pi_net` 398.8 → 558.2 µs（**+159 µs**），1→2→3 稳定在 555–579 µs；
  `tokens/step` 32→64→96→128，`_update_states` 71.4 → 226–232 µs。
  拟合 49.08 µs/draft、R²=0.578（0→1 的跳变 + 1/2/3 的平台使一阶拟合不是最佳形状；
  建议读"0 与 >0 的台阶"而不是斜率）。
* `prefix_hit_ratio`：−11.08 µs/单位比例（R²=0.438），即 0→1.0 净省 **48 µs（11.6%）**；
  `_update_states` 基本不动（−0.17 µs/单位）。方向与语义一致：命中的请求无需 prefill 调度。

### 4.6 子步骤归因（观测到的量级，stress/旧口径下测得）

`compute_slot_mapping`（含 harness 的 numpy 兜底）与 `_update_states` 是两个最大的被计时子步骤；
stress 口径下 `NPUModelRunner._prepare_inputs` 的包裹时间 ≈ 537 µs，而 substep 之和只有 ~215 µs，
说明**大部分 `_prepare_inputs` 时间落在未被 timer 覆盖的语句里**（逐语句明细见
`sweep_probe_ccall_base_*.json` 的 `c_us_per_step`，最大项是 `dict.get` / `Tensor.copy_` /
`np.repeat` / `_get_cumsum_and_arange`）。这是一条**方法学缺口**，不是 harness bug。

### 4.7 设备交互点占 prepare_input 多少（交错三模式，realmachine）

同配置（B=1, ISL=128, OSL=64）交错跑 3 轮、每轮轮换顺序（消除整段窗口漂移）。
图：`figures/06-slot-mapping-modes.svg`。左 = 堆叠柱（`pi_net` + `triton_cpu` = `pi_raw`），
右 = 同 round 配对差值。

| 模式 | `pi_raw` p50 (µs) | 轮间极差 | `triton_cpu` p50 | `pi_net` p50 | 每轮配对 Δnet |
|---|---|---|---|---|---|
| `noop`（下界） | 379.4 | 9.9 | 0.5 | **378.9** | — |
| `cpu_fallback` | 470.9 | 7.0 | 87.0 | 383.9 | +14.3 / +3.6 / +11.1 → **中位 +11.1** |
| `inject:15` | 487.9 | 3.3 | 102.6 | 386.3 | +15.1 / +6.8 / +9.8 → **中位 +9.8** |
| `inject:15 − cpu_fallback` | — | — | +15.6 | — | +0.8 / +3.2 / −1.4 → **中位 +0.8** |

* **下界**：`noop` ⇒ 设备交互点在 CPU 侧的占用 ≈ 0（唯一 1 次 launch/步被 shim 摘掉）。
* **上界**：若按 Triton Python launch 开销 15 µs/步计，`triton_cpu` = **102.6 µs/步**，
  占 `noop` 的 `pi_raw` **27.0%**。即"设备交互点最多占 prepare_input 约 1/4"。
* **numpy 兜底的间接成本**：直接成本 87.0 µs 被 `triton_cpu` 完整扣掉，
  残留间接成本 **+11.1 µs（中位；区间 +3.6~+14.3）= `noop` `pi_net` 的 2.9%**
  （额外 buffer 写 + cache 效应）。
* **记账自检通过**：`inject:15 − cpu_fallback` 的 `pi_net` 只差 **+0.8 µs（≈0）**，
  而 `triton_cpu` 恰好多 15.6 µs（102.6 − 87.0）⇒ **注入的自旋 100% 进入 `triton_cpu` 列，
  零污染净时间**。

#### ⚠️ 必须纠正的一条（跨窗口 vs 同窗口）

**不要把跨窗口的 "+31.2 µs 间接成本" 当作结论。** 分开跑的
`sweep_rm{noop,cpu,inj}_realmachine_*.csv`（19:14:25 / 19:15:08 / 19:15:52，3 个不同窗口）给出
`cpu − noop` = **+31.2 µs**；但同一进程内交错跑（同窗口、轮换顺序）的配对中位数只有
**+11.1 µs**。同一台机器上 `pi_raw` p50 的窗口漂移实测可达 1.9×（§5），
所以 **3× 的差距是漂移，不是效应**。跨窗口数据只能当上界。

* **不受 preset 影响的发现**：`cpu_fallback` 的 numpy 兜底成本**不随 `max_model_len` 缩小**——
  真机口径（预算 2048、每步 1 token）下 87.0 µs，与 stress 口径（预算 16384、16 token）的
  92–95 µs 同量级。该成本由**每步固定调用次数**主导，不是 token 数。（原先"真机口径下会
  显著变小"的预期，实测**不支持**。）

### 4.8 与真机基线核对

`--preset realmachine --batch 1 --isl 128 --osl 64 --steady-seconds 12` ⇒ **26 869 步**：

| 量 | 本次实测 | owner 基线 | 偏差 |
|---|---|---|---|
| `prepare_inputs_us` p50 | 376.5 µs | 364.8 µs | +3.2% |
| `update_states_us` p50 | 18.5 µs | 17.1 µs | +8.2% |
| `scope_total` p50 | 394.8 µs | 379.1 µs | +4.1% |

（`triton_cpu` 0.6 µs，扣后 `pi_net` = 375.9 µs。）偏差在宿主噪声带内。
文件：`data/harness/sweep_rmsteady_realmachine_20260923-190652.csv`。

## 5. 噪声与可信度

`scripts/hostnoise_gate.sh --cpus 200-215 --sample-s 3` 三次留档：

| gate | slice mean busy | hot CPUs | 与切片重叠的无关进程数 |
|---|---|---|---|
| pre | 25.15% | 200, 208, 210, 214 | 43 |
| post | 7.23% | 209 | 41 |
| final | 6.63% | 210, 211 | 41 |
| post2（03:18 补测） | 6.27% | 212 | 41 |

三次都判 **NOISY**（阈值 mean≤2.0% 且无 hot cpu）。

相同配置跨时间窗的 `pi_raw` p50 漂移实测可达 **1.9×**：
`--batch 16 --isl 1024` 在 4 个窗口分别为 506.4 / 535.2 / 763.0 / 517.1 µs
（见 `hw64_sweep_LEGACY_batch_*`、`sweep_d_base_*`、`sweep_r3_batch_*`、`sweep_grid_*`）。

**因此本次所有主结论用同窗口单进程网格 + `--repeat 3` 取跨轮中位数**；
绝对 µs 只给区间，形状（`a`、R²）与相对比较才是稳健结论。

## 6. 异常点与解释

| 现象 | 判断 | 证据 |
|---|---|---|
| 旧口径 batch=1,2 的 `pi_raw` 高达 834/871 µs，batch=4 掉到 487 | **真实效应 + 旧口径放大**：真机口径下同批次（B=1..8）只有 364–379 µs，无该反常。旧口径 67 MB `token_ids_cpu_tensor` 的 cache 行为把固定成本抬到 ~800 µs | `hw64_sweep_LEGACY_batch_*` vs `sweep_rm_realmachine_*` |
| realmachine isl 256/512 某一轮跳到 399/404 µs | **噪声**（同点另两轮 376/377；轮间极差 21.6/26.6 µs） | `sweep_rm_realmachine_20260923-190512.csv` |
| `--sweep spec_k` 的 4 行几乎一样 | **harness bug**：`spec_k` 未透传 ⇒ 4 行同配置重复，不是真实效应 | `hw64_sweep_LEGACY_spec_k_*`（`spec_k` 列全 0） |
| stress `prefix_hit_ratio` 单调下行而 realmachine 无此维度 | **真实效应**：stress 每步 32 token 且允许 prefix caching；realmachine 是 `--no-enable-prefix-caching`（真机口径） | `sweep_stress_stress_*` vs `preset` 定义 |
| realmachine isl=2048 崩 | **harness/synth 缺口**：`could not broadcast input array from shape (17,) into shape (16,)`（block 表宽度 = `max_num_blocks_per_req` = 16 不够） | `agents/sweep_analysis/logs/phase_rmw0.log` |
| `_prepare_inputs` 计时里 ~320 µs 无子步骤归属 | **方法学缺口**：timer 只 wrap 了少数方法，逐语句归因见 `sweep_probe_ccall_*` | `sweep_d_base_substep_*.csv` |

## 7. harness 发现的问题（未改 harness，仅报告 + 分析侧 workaround）

约束要求不改 harness（除非在 `synth.py` / `triton_cpu.py` 各改一处）；以下三处都不在那两个文件里，
因此**保持 harness 原样**，仅在分析侧绕开，并把建议补丁回传 harness owner。

### 7.1 `--sweep spec_k` 静默失效

`harness/pi_harness/runner/cli.py::run_sweep`：
```python
field = {"model": "model_profile"}.get(a.sweep, a.sweep)   # 缺 "spec_k": "num_spec_tokens"
```
`setattr(cfg, "spec_k", v)` 只挂新属性；真正决定分支的 `RunnerConfig.num_spec_tokens` 恒为 0。
**证据**：`data/harness/hw64_sweep_LEGACY_spec_k_20260923-181500.csv` 4 行的 `spec_k` 列全 `0`。
**建议补丁**：`{"model": "model_profile", "spec_k": "num_spec_tokens"}`。

### 7.2 spec 路径缺 `Tensor.pin_memory()` shim → 无卡容器 aclInit 507008

真跑 spec 时 `vllm_ascend/worker/model_runner_v1.py:1412` 执行
`torch.from_numpy(cu_num_draft_tokens).pin_memory().to(...)`。shim 只把 `pin_memory=` **关键字**
改 False，没拦**方法**形式 ⇒ 真实 aclInit：
```
RuntimeError: ... NPU function error: aclInit, error code is 507008
  File ".../model_runner_v1.py", line 1293, in _prepare_inputs
  File ".../model_runner_v1.py", line 1412, in _calc_spec_decode_metadata
```
**复现**：`... pi_harness.runner.cli --batch 16 --isl 1024 --spec-k 1`（日志
`agents/sweep_analysis/logs/sweep_d_spec_1.log`）。
**建议补丁**：`_patch_tensor_and_factories()` 里加 `T.pin_memory = lambda self, device=None: self`。
本报告 §4.5 的 spec_k 曲线由 `agents/sweep_analysis/preset_grid.py` 产出（进程内补该 no-op，
与既有 "pin_memory→False" 策略等价；`has_spec_decode=True` / `tokens/step = 1+k` 可核对）。

### 7.3 CLI `a.num_spec_tokens` 回归（02:55 引入，至今未修）

```python
if a.num_spec_tokens is not None:      # argparse 的 dest 是 spec_k
    cfg.num_spec_tokens = a.spec_k
```
`build_parser()` 只有 `--spec-k`（dest `spec_k`）⇒ 任何 CLI 调用都
`AttributeError: 'Namespace' object has no attribute 'num_spec_tokens'`。
**影响**：02:55 之后无法用 CLI 跑；本报告 02:55 之后全部改用分析侧驱动
（`preset_grid.py` / `mode_compare.py`，直接构造 `RunnerConfig`）。
**建议补丁**：`if a.spec_k is not None: cfg.num_spec_tokens = a.spec_k`。

## 8. 已知偏差（方向已标注）

1. **H2D/D2H 未复现**：真机 `copy_to_gpu` 是 async DMA，无卡下退化为 CPU→CPU
   （`shim` 有 `degraded_copy_bytes` 记账）。⇒ 真机实际成本 **≥** 本报告数字。
2. **Triton launch 的 Python 侧开销**：默认 `--triton-launch-us 0`，即 `pi_net` **不含**该开销；
   `inject:15` 给敏感性上界（§4.7）。
3. **device op 的内核态开销**（acl 调用）未复现 ⇒ **低估**。
4. **`warmup=3` 吃掉 prefill 步**：除小 chunk / 大 ISL 外，测量窗口几乎全是 decode。
   这解释了"ISL 曲线为什么平"；`warmup=0` 对照见 §4.1。
5. **P>1（TP/DP）未复现**：`FakeGroupCoordinator` 在集合通信处抛错。
6. 绝对 µs 携带 **1.2–1.9×** 的窗口漂移（§5）。
7. `_prepare_inputs` 的子步骤归因不完整（§4.6），约 60% 时间无归属。

## 9. 复现命令

```bash
# 真机口径：batch / isl
bash agents/sweep_analysis/run_remote.sh pgrid realmachine batch \
  "--values 1,8,16,32,64 --max-num-reqs 64 --steps 200 --repeat 3 --dump-steps --tag rm64"
bash agents/sweep_analysis/run_remote.sh pgrid realmachine isl \
  "--values 64,128,256,512,1024 --steps 200 --repeat 3 --dump-steps --tag rm"
# stress 全维度
bash agents/sweep_analysis/run_remote.sh phase_stress
# 交错三模式（realmachine）
# → mode_compare.py --preset realmachine --rounds 3
# 分析 + 出图（开发机）
python3 agents/sweep_analysis/analyze.py
python3 agents/sweep_analysis/plot_sweep.py
```

所有产物的写入范围：`data/harness/sweep_*`、`figures/06-sweep-*`、本文件、
`docs/03-share-in-inference.md` §7.1。**未触碰 harness 代码、未使用 CPU 120-159、未挂 NPU。**
