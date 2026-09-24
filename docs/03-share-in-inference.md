# prepare_input 在推理中的占比与瓶颈判据

> 本章程由 drivers_analysis 起草（2026-09-24），**"判据与公式"一节已定稿**；
> 其余章节（占比曲线、分场景数据表）由 measurement 代理用真机数据补全。
> 配套：`07-bottleneck-analysis.md`（完整模型与推导）、`data/model/complexity-model.json`（机器可读）。

---

## 判据与公式

### 1. 先明确口径（三个"占比"不可混用）

```
share_cpu  = T_prepare / T_cpu_engine_step    # CPU 预算占用率
share_step = T_prepare / T_step               # 单步 wall 占比
share_e2e  = 端到端损失归因（仅当 T_cpu > T_dev 时非零）
```

`T_cpu_engine_step = T_sched + T_rpc + T_prepare + T_launch_fwd + T_post`（worker 主线程串行）。
**报占比必须同时给出：分子、分母、是否被掩盖（`T_cpu > T_dev` 的真假）。**

### 2. 主判据（异步调度开，0.26.0 默认）

```
T_step ≥ max( T_cpu_engine_step , T_dev_step )
T_prepare 成为瓶颈  ⟺  T_prepare > T_dev_step − (T_sched + T_rpc + T_launch_fwd + T_post) := B_cpu
```

### 3. decode 的线性化临界条件

```
T_prepare  ≈ c_p + a·B
T_dev_step ≈ e   + d·B
T_cpu_other ≈ c_o

CPU 受限 ⟺ B > (e − c_p − c_o)/(a − d)      （仅在 a > d 时存在有限临界点）
```

三种形态：

| 形态 | 条件 | 现象 | 优化收益 |
|---|---|---|---|
| 斜率受限 | `a > d` | 大 B 时占比随 B 升高 | 与 `a` 成正比 |
| 底噪受限 | `c_p + c_o > e` | **任何 B 都 CPU 受限**（低负载也慢） | 减 `c_p`（H2D 次数、固定调用） |
| 被掩盖 | `a < d` 且 `c_p + c_o < e` | 占比可能很高但**无吞吐收益** | ≈0（只降 CPU 占用） |

### 4. chunked prefill 的临界 chunk 大小

```
T_dev(T)    ≈ e' + β·T + γ·T·(S + T/2)
T_prepare(T) ≈ c_p + a_T·T            （a_T ≈ 7.8–8.8 ns/token = 整段实测；单算子 0.13–1.29 ns/el）
解 c_p + a_T·T = e' + β·T + γ·T(S + T/2)  得最小可行 chunk T_min
```

小 chunk 区 `T_prepare` 出现"不随 T 下降的**平台**"，平台高度 = `c_p`。

### 5. TP 下的放大

```
T_step = max_{r=1..R} T_cpu,r + T_dev + T_comm
E[T_cpu,max] ≈ μ + σ·√(2 ln R)          （T_cpu ~ N(μ, σ²) 的近似）
```

每个 rank 都执行同一份 CPU 逻辑 ⇒ `T_cpu` 不随 TP 下降，而 `T_dev` 随 TP 下降
⇒ **TP 越大，CPU 越容易成为瓶颈**。

### 6. 异步关闭（串行流水）

```
T_step ≈ T_sched + T_rpc + T_prepare + T_dev + T_post + T_sample + T_sched_update
```

此时 `share_step` 直接等于优化收益上限，"掩盖"不存在。**必须保留一组对照实验。**

### 7. 真机占比（chip3，0.8B / TP1 / ISL=128 / decode / `FULL_DECODE_ONLY` / pystack off）

> 数据源：`data/subscope/ladder-s{1,2}.json`，由 `scripts/extract_ladder.py` 用
> `--step-scope Step:Schedule --window-mode next` 口径生成。**这是本项目唯一的真机占比数据。**

| 场景 | B | `prepare input` p50 | 单步 p50 | **占比** | `forward` |
|---|---:|---:|---:|---:|---:|
| S1 | 1 | **2757.6 µs** | 4915 µs | **55.3%** | 830.6 µs |
| S2 | 64 | **3873.6 µs** | 10010 µs | **38.7%** | — |

**反直觉但关键**：B 从 1 涨到 64（×64），`prepare input` 只涨 **×1.40**，
所以**占比反而从 55.3% 降到 38.7%**。原因见 §7.2(c)：`prepare input` 的
固定底噪 `c_p ≈ 371–381 µs` 与每请求斜率 `a ≈ 0.54–0.85 µs/req` 相比，
device 侧的单步时间随 batch 增长得更快。

⇒ **低并发（B 小）才是 `prepare_input` 最该被优化的区域**：
那里它的绝对成本不低（2.8 ms），而分母（device 时间）最小。

#### 7.1 A/B/C/D 阶梯：把"插桩代价"与"采样污染"分开

| 臂 | 配置 | `prepare input` p50 | 单步 p50 | 占比 |
|---|---|---:|---:|---:|
| A | 基线（不挂载） | 2757.6 µs | 4915 µs | 55.3% |
| B | 挂载 + `PI_SUBSCOPE=off` | 2806.2 µs | 4979 µs | 55.6% |
| C | 挂载 + 探针 on（49 个） | 3072.6 µs | 5440 µs | 57.2% |
| D1 | 挂载 + pystack 默认（1 ms） | 3484.8 µs | 6187 µs | 56.2% |
| D2 | 挂载 + pystack off（B 的重复臂） | 2909.5 µs | 5089 µs | 56.4% |

| 差值 | `prepare input` | 单步 | 含义 |
|---|---:|---:|---|
| **挂载代价 B−A** | +48.7 µs（**+1.77%**） | +63.4 µs | 多 import 一个模块——可接受 |
| **探针代价 C−B** | +266.4 µs（**+9.49%**） | +422.5 µs | 49 探针 × 2.2 µs 地板；**超 5% 预算，引用子 scope 数值时必须扣除** |
| **pystack 污染 D1−B** | +678.5 µs（**+24.2%**） | +919.2 µs | 历史数据就带着这个 |
| **噪声底 D2−B** | +103.3 µs（+3.68%） | +110 µs | 宿主噪声；**所有 <3.7% 的差异不可解释** |

> S2（B=64）的同一阶梯见 `data/subscope/overhead-s2-20260923T184335Z.json`：
> 挂载 −0.32%、探针 +6.64%、pystack +16.66%、噪声底 +2.5%~+3.5%。
> 两场景一致 ⇒ 上述四项是**系统性代价**，不是偶然。

### 8. 拟合参数表

#### 8.1 无卡 harness 实测（真实代码路径 + shim，见 `06-synthetic-load.md`）

| 参数 | 含义 | 实测值 | R² | n | 口径 |
|---|---|---|---:|---:|---|
| `c_p` | `prepare_input` 固定底噪 | **380.75 µs** | — | — | realmachine, batch 1→64 |
| `a` | 每请求斜率 | **0.5375 µs/req** | **0.983** | 5 | 同上（`max_num_reqs=64`） |
| `c_p,a` | 同上（`max_num_reqs=8`） | 371.02 µs / 0.8468 µs/req | 0.778 | 4 | 同上（`max_num_reqs=8`） |
| `c_u, a_u` | `_update_states` 固定/每请求 | 18.00 µs / **1.6750 µs/req** | **1.000** | 5 | 同上 |
| `a_T` | 每 token 斜率（decode 窗口） | 0.0055 µs/token = 5.5 ns/token | 0.080 | 5 | realmachine isl 64→1024 |
| `a_T` | 每 token 斜率（含 prefill） | 0.0040 µs/token = 4.0 ns/token | 0.979 | 3 | realmachine isl 128→1024 |
| `c_s, a_s` | spec decode | 449.21 µs / **49.08 µs/draft** | 0.578 | 4 | stress spec_k 0→3 |
| `k_h2d_small` | 单次小 NPU 算子派发 | **3–12 µs** | — | — | 真机微基准（`scripts/npu_dispatch_bench.py`） |
| `k_aten` | 单次 ATen CPU 调用 | 1.6–2.3 µs | — | — | 微基准 |

#### 8.2 真机对照（chip3）

| 参数 | 无卡 harness | **真机（含 device 交互）** | 倍率 |
|---|---:|---:|---:|
| `prepare input` scope 总时长（B=1） | 379.1 µs | **2806 µs** | **7.4×** |
| `_update_states`（B=1） | 18.5 µs | **100.7 µs** | 5.4× |
| `AscendGDNAttentionMetadataBuilder.build` | 不在覆盖范围 | **3 × 303 µs = 908 µs** | — |

> **这个 7.4× 不能被简单读成"shim 误差"**：其中 **≥944 µs 是 harness P1 范围外的代码**
> （GDN builder 908 + full-attn builder 36），其余才是 H2D/DMA、Triton launch、
> acl device op 等 shim 缺口。完整归因见 `06-synthetic-load.md` §6.2。

> ⚠️ `e, d`（device 固定/每请求）与 `β, γ`（chunk 线性/二次）**尚未真机拟合**：
> 需要 B×ISL 网格的 phase 数据。按 `plan/EXECUTION.md` §2 的 A/B/C 组补齐即可，
> 已有脚本（`scripts/measure/point_run.py` + `matrix_run.py` + 42 点计划）。

---

### 7.1 参数影响曲线（无卡 harness sweep；口径=`pi_net_p50_us`；`sweep_analysis` 负责，2026-09-24）

> 本节由 `sweep_analysis` 追加，与上表（measurement 代理的真机曲线）**并存不覆盖**：
> 上表是**真机** `T_prepare/T_dev` 占比；本节是**无卡 harness**的单函数 CPU 负载曲线，
> 只回答"哪些参数以多大斜率影响 `_prepare_inputs` / `_update_states`"，不回答"占 device 多少"。
> 数据：`data/harness/sweep_*`；机器可读：`data/harness/sweep_fits.json`；详版：`agents/sweep_analysis/REPORT.md`。

#### 口径（**每次引用数字都必须带这行**）

```
pi_net_us = prepare_inputs_us − triton_cpu_us          ★ 本节所有结论用这个
scope_net_us = update_states_us + prepare_inputs_us − triton_cpu_us
```

`triton_cpu_us` 是 harness 独有开销（无卡环境里 `_compute_slot_mapping_kernel` 的 numpy 等价
实现 + 注入的 launch 自旋）；真机上对应的是 device kernel，不在 CPU 侧 prepare_input 里。
**不变式已确认**：每个被测 step 的 `triton_launches == 1`（逐 step 见
`data/harness/sweep_*_per_step_*.csv` 的 `triton_launches` 列）。

#### preset 是结论的一部分（不可混池）

| preset | max_model_len | max_num_reqs | max_num_batched_tokens | prefix cache | async | slot_mapping | 用途 |
|---|---|---|---|---|---|---|---|
| `realmachine` | 2048 | 8（扫大 batch 时放开到 64） | 2048 | off | on | **noop** | 与真机 launcher 对照 |
| `stress` | 32768 | 64 | 16384 | on | off | **noop** | 放大斜率 |

`max_model_len` 决定 `InputBatch.token_ids_cpu_tensor` 形状：真机 `(8, 2048) int32 = 64 KB`
（常驻 L1/L2）vs 旧误用 `(64, 262144) = 67 MB`（每步从 DRAM 拉）。
**旧口径的原始数据保留为 `data/harness/hw64_sweep_LEGACY_*.csv`，其结论作废。**

#### 图

| 图 | 口径 | 内容 |
|---|---|---|
| `figures/06-sweep-rm-batch.svg` | realmachine | `pi_net` / `scope_net` / `_update_states` vs B（1→64） |
| `figures/06-sweep-rm-isl.svg` | realmachine | `pi_net` vs ISL（64→1024，decode 窗口） |
| `figures/06-sweep-{isl,batch,chunk,blocksize,spec,prefix}.svg` | stress | 六个维度 |
| `figures/06-sweep-summary.svg` | stress | 2×3 汇总 |

#### 拟合表（`pi_net_us` p50，已扣 `triton_cpu_us`；单位 µs/步）

| preset | 维度 | `c`(µs) | `a` | R² | n | 底噪/斜率主导 |
|---|---|---|---|---|---|---|
| realmachine | batch（B=1→64, max_num_reqs=64） | 380.75 | **0.5375 µs/req** | 0.983 | 5 | 全区间底噪主导（B=64 时常数仍占 92%） |
| realmachine | batch（B=1→8, max_num_reqs=8） | 371.02 | 0.8468 µs/req | 0.778 | 4 | 同上 |
| realmachine | isl（64→1024, decode 窗口） | 377.49 | 0.0055 µs/token | 0.080 | 5 | 平（窗口内无 prefill） |
| realmachine | isl（warmup=0, 含 prefill 步） | 373.40 | 0.0040 µs/token | 0.979 | 3 | 平（chunk=2048 ≥ ISL） |
| realmachine | `_update_states` vs batch | 18.00 | **1.6750 µs/req** | 1.000 | 5 | 斜率主导 |
| stress | batch（1→64） | 377.20 | 0.7310 µs/req | 0.971 | 7 | 底噪主导 |
| stress | isl（64→2048） | 388.62 | 0.0065 µs/token | 0.511 | 6 | 平（chunked prefill 摊平） |
| stress | chunk_size（128→8192） | 393.45 | 0.0003 µs/token | 0.031 | 6 | 平 |
| stress | block_size（32→256） | 390.79 | 0.0068 µs/(token/block) | 0.304 | 4 | 平（最大 −13 µs） |
| stress | spec_k（0→3） | 449.21 | **49.08 µs/draft** | 0.578 | 4 | 斜率主导 |
| stress | prefix_hit_ratio（0→1） | 414.85 | **−11.08 µs/单位比例** | 0.438 | 5 | 弱下行 |

#### 三个必须记住的结论

1. **`_update_states` 的每请求斜率是 `_prepare_inputs` 的 2.7–3.1 倍。**
   realmachine 逐 step 池化（29 648 decode 步）：`_update_states` **1814 ns/req** vs
   `pi_net` **676 ns/req**（B=64 口径 1670 vs 537 ns/req）。请求数增加时**逐请求 Python
   循环**（`_update_states`）比随总 token 数走的 `np.repeat`/`index_select`（`_prepare_inputs`）
   涨得更快。⇒ 高并发下优化优先级应给 `_update_states`。

2. **chunked prefill 把 ISL 摊平成"每步固定 chunk"。**
   stress 下 ISL 64→2048 的 `pi_net` 斜率仅 6.5 ns/token（R²=0.51），
   原因是每步实际 token 数被 `chunk_size` 与预算钉死在 32（`tokens_per_step` 全程 = 32）。
   用 `per_step_us / min(ISL, chunk_size)` 读才对：`= 388.6/32 = 12.1 ns/token`。

3. **MTP（spec_k）是唯一有量级斜率的一维：+49 µs/步 per draft token**
   （`tokens/step` 从 32 涨到 128，`_update_states` 同步 +47.9 µs/draft）。
   其余维度在各自覆盖区间内都是**底噪主导**（常数项 ~377–394 µs）。

#### 设备交互点占 `prepare_input` 多少（交错三模式对照，realmachine）

同配置（B=1, ISL=128, OSL=64）× 3 轮轮换顺序互相对照（消除整段窗口漂移）：

| 模式 | `pi_raw` p50 | `triton_cpu` p50 | `pi_net` p50 | Δnet vs noop |
|---|---|---|---|---|
| `noop`（最小 CPU 干扰） | 379.4 µs | 0.5 µs | **378.9 µs** | +0.0 |
| `cpu_fallback`（数值正确） | 470.9 µs | 87.0 µs | 383.9 µs | +5.0 µs |
| `inject:15`（+15 µs/launch 自旋） | 487.9 µs | 102.6 µs | 386.3 µs | +7.4 µs |

* **下界**：`noop` ⇒ 设备交互点在 CPU 侧的真实占用 ≈ 0（1 次 launch/步已被 shim 摘掉）。
* **上界（若按 Triton 实测 Python launch 开销 15 µs/步计）**：`inject:15` 的
  `triton_cpu` = 102.6 µs，占 `noop` 的 `pi_raw` 的 **27.0%**、占 `pi_net` 的 **27.1%**。
* `cpu_fallback` 的 numpy 兜底 87.0 µs 里有 82.0 µs 被 `triton_cpu_us` 记账、**5.0 µs 漏在
  `pi_net` 里**（缓存污染/窗口外的分配），即扣减后仍残留约 1.3% 的偏差。
* 注意：`cpu_fallback` 的 ~87 µs **不随 `max_model_len` 变小而变小** —— 真机口径
  （`max_num_tokens=2048`、每步 1 token）下仍是 87 µs，与 stress 口径（16384 预算、16 token）
  的 92–95 µs 同量级。**该成本由每步固定调用次数主导，而非 token 数。**

#### 与真机基线的核对

`--preset realmachine --batch 1 --isl 128 --osl 64 --steady-seconds 12` ⇒ **26 869 步**：
`prepare_inputs_us` p50 = **376.5 µs**、`update_states_us` p50 = **18.5 µs**、
`scope_total` = **394.8 µs/步**（`triton_cpu` 0.6 µs，扣后 `pi_net` = 375.9 µs）。
harness owner 的独立基线为 364.8 / 17.1 / 379.1 µs ⇒ **偏差 +3.2% / +8% / +4.1%**，
在宿主噪声带内（见下）。

#### 噪声下限（任何绝对 µs 都要带）

`hostnoise_gate.sh --cpus 200-215` 三次留档（pre/post/final）：slice mean busy
**6.6%–25.2%**，`hot_cpus` 非空、41–46 个他人进程与 200-215 亲和性重叠 ⇒ **三次都判 NOISY**。
相同配置跨时间窗的 `pi_raw` p50 漂移可达 1.9×（如 batch=16/isl=1024：506 → 535 → 763 → 517 µs）。
**因此：形状（a、R²）可信；绝对 µs 只给区间。**

#### 已知缺口（未在本节结论中使用）

| 缺口 | 现象 | 证据 |
|---|---|---|
| `--sweep spec_k` 静默失效 | `spec_k` 列恒为 0（4 行是同配置重复） | `data/harness/hw64_sweep_LEGACY_spec_k_*.csv` |
| spec 路径缺 `Tensor.pin_memory` shim | 无卡容器 `aclInit 507008` 崩 | `agents/sweep_analysis/logs/sweep_d_spec_1.log` |
| CLI `a.num_spec_tokens` 回归 | 任何 CLI 调用 AttributeError | `harness/pi_harness/runner/cli.py:121` |
| realmachine + isl=2048 | `could not broadcast (17,) into (16,)` | `agents/sweep_analysis/logs/phase_rmw0.log` |

前三项是 harness 自身的 bug，本节数据由分析侧驱动
（`agents/sweep_analysis/{preset_grid,sweep_grid,mode_compare}.py`）绕开产生；
详细复现步骤与建议补丁见 `agents/sweep_analysis/REPORT.md` §3。

---

### 7.2 参数影响曲线（无卡 harness，preset=realmachine / stress，口径=`pi_net_p50_us`）

> 追加：2026-09-24（`sweep_analysis`）。**本小节所有数字都标 preset。**
> `stress` 口径只用于**斜率/趋势**，**不能**用于与真机对照。
> 机器可读：`data/harness/sweep_fits.json:formal_fits`（字段
> `{preset, dim, x_values, y_values, c_us, a_us_per_unit, r2, n_points, noise_band_pct, caliber}`）。

#### （a）preset 敏感性：同一个 `prepare_input`，IPC 差 64%

来源：`docs/06-synthetic-load.md` §9.2（PMU/libkperfx 920B preset，9 组 `confidence=1.0`）。

| preset | `token_ids_cpu` 形状 | 工作集 | **IPC** | 能否用于真机对照 |
|---|---|---|---|---|
| `realmachine`（真机 launcher 口径） | (8, 2048) int32 | **64 KB** | **0.949** | ✅ 与真机历史区间 0.719–0.890 同区间 |
| `stress` | (64, 32768) int32 | **8 MB** | **1.553** | ❌ 远离真机 |

**同一个 `prepare_input` 代码，只换引擎启动参数，IPC 差 63.6%（0.949 → 1.553）。**
机制：`max_model_len` 决定 `InputBatch.token_ids_cpu_tensor` 的宽度，而 `_prepare_inputs`
每步都 `torch.index_select(token_ids_cpu_tensor.flatten(), ...)`；
64 KB 常驻 L1/L2 vs 8 MB 每步从 DRAM 拉 —— **cache 行为完全不同**。
这是"必须匹配真机启动参数"的最强证据（也是旧 `--sweep` 结论作废的原因）。

#### （b）slot-mapping 模式对照（preset=realmachine，B=1/ISL=128/OSL=64，每步恰好 1 次 launch）

图：`figures/06-slot-mapping-modes.svg`。左侧堆叠柱 = `pi_net` + `triton_cpu` = `pi_raw`。

| 模式 | `pi_raw` p50 | `triton_cpu` p50 | `pi_net` p50（口径） | 说明 |
|---|---|---|---|---|
| `noop`（下界） | 379.4 µs | 0.5 µs | **378.9 µs** | 设备交互在 CPU 侧 ≈ 0 |
| `cpu_fallback` | 470.9 µs | 87.0 µs | **383.9 µs** | numpy 数值等价实现 |
| `inject:15` | 487.9 µs | 102.6 µs | **386.3 µs** | cpu_fallback + 15 µs/launch 自旋 |

**同 round 配对差值（唯一可信的模式间比较；跨窗口比较会被漂移污染）**：

| 配对 | 每轮差值（µs） | 中位数 |
|---|---|---|
| `cpu_fallback − noop` | +14.3 / +3.6 / +11.1 | **+11.1 µs** |
| `inject:15 − noop` | +15.1 / +6.8 / +9.8 | **+9.8 µs** |
| `inject:15 − cpu_fallback` | +0.8 / +3.2 / −1.4 | **+0.8 µs** |

**三条结论**：

1. **numpy 兜底的成本几乎全部落在 `triton_cpu` 列里**：87.0 µs 直接成本被完整扣掉，
   只留下 **+11.1 µs（中位数，区间 +3.6~+14.3）** 的间接成本（额外 buffer 写 + cache 效应），
   占 `noop` `pi_net` 的 **2.9%**。
   ⚠️ **不要用跨窗口差值（+31.2 µs）当间接成本** —— 那 3× 的膨胀来自时间窗漂移。
2. **注入的自旋被 100% 正确归类**：`inject:15 − cpu_fallback` = **+0.8 µs**（≈0），
   而 `triton_cpu` 恰好多了 15.6 µs（102.6 − 87.0）⇒ 自旋没有污染 `pi_net`。
   这是一次方法学自检通过。
3. **设备交互点占 prepare_input 的上界 ≈ 27%**：按 15 µs/launch 计，
   `triton_cpu` = 102.6 µs，占 `noop` 的 `pi_raw` 的 27.0%。
   （`stress` 口径下同一项是 0.5–0.7 µs，因为 `noop` 是默认值；见 `06-sweep-summary` 图注。）

#### （c）realmachine 核心曲线

图：`figures/06-sweep-realmachine-batch.svg`、`figures/06-sweep-rm-isl.svg`。
口径 `pi_net_p50_us`，`--repeat 3` 取跨轮中位数。

| preset | 维度 | `c_us` | `a_us_per_unit` | R² | n | `noise_band_pct` |
|---|---|---|---|---|---|---|
| realmachine | batch 1→8（max_num_reqs=8） | 371.02 | 0.8468 µs/req | 0.778 | 4 | 0.68% |
| realmachine | batch 1→64（max_num_reqs=64） | 380.75 | **0.5375 µs/req** | 0.983 | 5 | 0.37% |
| realmachine | `_update_states` vs batch 1→64 | 18.00 | **1.6750 µs/req** | **1.000** | 5 | — |
| realmachine | isl 64→1024（decode 窗口） | 377.49 | 0.0055 µs/token | 0.080 | 5 | 1.16% |
| realmachine | isl 128→1024（`--warmup 0`，含 prefill） | 373.40 | 0.0040 µs/token | 0.979 | 3 | 0.56% |

> `noise_band_pct` = 同配置重复轮 `pi_raw` p50 的 (max−min)/median。`stress` 各维为单轮，
> 故该字段为 0；`stress` 的轮间离散度另见 `sweep_stresssub_stress_*`（block_size 复测两轮）。

**`_update_states` 的每请求斜率是 `_prepare_inputs` 的 3.1 倍**（1.6750 vs 0.5375 µs/req，
两者 R² 分别为 1.000 / 0.983）⇒ 高并发下的优化优先级应给 `_update_states`。
真机口径 `batch=1` 基线：`pi_net` = **375.9 µs**、`_update_states` = 18.5 µs、
`scope_net` = 394.8 µs（`--steady-seconds 12`，26 869 步）。

#### （d）stress 六维（**仅斜率/趋势，不可作真机对照**）

图：`figures/06-sweep-{isl,batch,chunk,blocksize,spec,prefix}.svg`
 + `figures/06-sweep-summary.svg`（2×3 汇总）+ `figures/06-sweep-panels.svg`（同源的拼版）。

| dim | `c_us` | `a_us_per_unit` | R² | n | 判断 |
|---|---|---|---|---|---|
| isl 64→2048 | 388.62 | 0.0065 µs/token | 0.511 | 6 | 平（chunked prefill 摊平） |
| batch 1→64 | 377.20 | 0.7310 µs/req | 0.971 | 7 | 底噪主导 |
| chunk_size 128→8192 | 393.45 | 0.0003 µs/token | 0.031 | 6 | 平 |
| block_size 32→256 | 390.79 | 0.0068 µs/(token/block) | 0.304 | 4 | 平（净 −13 µs） |
| spec_k 0→3 | 449.21 | **49.08 µs/draft** | 0.578 | 4 | 斜率主导 |
| prefix_hit_ratio 0→1 | 414.85 | **−11.08 µs/单位** | 0.438 | 5 | 弱下行（净 −48 µs） |

`stress` 口径 `batch` 的 `_update_states` 斜率 = **1.6287 µs/req**（R²=1.000），
与 `realmachine` 的 1.6750 一致 ⇒ **`_update_states` 的逐请求成本对 preset 不敏感**，
而 `_prepare_inputs` 的斜率会随 preset 变（0.5375 vs 0.7310）——因为它吃 `max_model_len`
决定的 cache footprint。

### 7.3 历史基线图（既有项目的实测复刻，口径见 `docs/09`）

本节三张图来自对既有 `HIST_PROJECT` 项目 26 个 decode 配置的挖掘，
**用于说明"占比随模式跃变"这件事不是本次采集的偶然**（数据：`data/historical/`）。

| 图 | 内容 | 一句话结论 |
|---|---|---|
| `figures/hist-01-phase-decomposition.svg` | 各配置的 phase 分解 | `prepare input` 绝对耗时 1.6–6.4 ms，**与模型规模基本无关** |
| `figures/hist-02-prepare-vs-step-period.svg` | `prepare input` vs 单步周期（按模式分组） | eager→graph 时**分母塌了 7.5×**，占比从 6.7% 跳到 55.3%，而分子只 +10% |
| `figures/hist-03-topdown-ipc.svg` | worker 主线程 topdown 四桶 + IPC | frontend-bound 55.99–65.00%、IPC 0.719–0.890，跨 0.8B→80B / TP1–4 稳定 |

> ⚠️ 引用这三张图的数字时**必须带 pystack 污染声明**：历史 phase 数据由
> `POST /start_profile` 采集，该接口会无条件启动 1 ms 采样器（实测对 `prepare input`
> 污染 **+24.2%**、对 ITL **+60%**）。判据与修正见 `docs/09-historical-data-caveat.md`。
