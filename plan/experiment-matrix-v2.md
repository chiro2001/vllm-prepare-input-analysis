# 实验矩阵 v2（成本模型标定版）

> 基于 `plan/EXECUTION.md §2`（v1）。**保留 v1 全部组**，按 §1 的理由**增补 6 组**，
> 并把每个实验点与 `data/model/complexity-model.json` 的系数/预测一一挂钩。
> 目的：让"何时成为瓶颈"从定性判断变成**可拟合、可证伪**的四个数 `(c_p, a, e, d)` + `(β, γ)`。

---

## 0. 与 v1 的差异汇总（含理由）

| 组 | v1 | v2 | 理由 |
|---|---|---|---|
| A 并发扫描 | ISL=128, OSL=1024, 9 点 | **不变**，但要求同时报 `T_prepare` 与 `T_dev` 两条曲线 | 拟合 `c_p, a, e, d` 需要两侧数据；只报占比无法解出临界 B |
| B ISL 扫描 | 6 点，OSL=1 | **不变**，补 `T_dev` | 验证 `a_T` 与 P2 的"平台" |
| C chunked prefill | 3 点 | **加密到 5 点**：`512, 1024, 2048, 4096, 8192` | §4.3 的 `β, γ` 是二次拟合，3 点无法给置信区间 |
| D 混合负载 | 3 点 | **不变** | 真实锚点 |
| E MTP | D ± spec | **不变**，加采"5 次 pin+H2D 的火焰图占比" | 对应 `A_N` 与优化 O2 |
| F 块大小/前缀 | block_size × prefix | **不变** | 对应 `A_K` |
| **G 异步开关对照** | — | **新增**：D 的两个代表点 × `--async-scheduling {true,false}` | 决定 `max()` 还是串行；给出"被掩盖程度"（§4.5） |
| **H churn 对照** | — | **新增**：稳态 vs 高频进出（短 OSL + 高并发 + 抢占） | 唯一能量化 `A_C` 的 O(B·S) 突变项（E1/E2） |
| **I TP 放大** | — | **新增**：A 的 2 个点 × TP∈{1,2,4} | 验证 §4.4 straggler 与 P4；TP 是"CPU 不降、device 降"的关键场景 |
| **J 缓冲宽度税** | — | **新增**：固定 B=8/64，扫 `--max-num-seqs ∈ {64, 256, 1024}` | 检验 `c₀` 中随 `W` 的项（全量 `copy_to_gpu`、尾部 `fill_`、惰性大分配） |
| **K 系数标定（无模型）** | — | **新增**：`k_h2d_small`、`k_pin`、`k_launch`、`k_dma_per_B` 微基准 | §6.3 的 4 个未知系数只能这样拿，否则模型不闭环 |
| **L replica↔真机逐项对齐** | — | **新增**：B∈{1,64,256}、chunk∈{2048,8192}、MTP on/off | 支撑 §7.4 保真度判据与 D1–D6 偏差归因 |

---

## 1. 每个实验点的标准记录模板

```
目的：        （要拟合/证伪哪个系数或预测）
自变量：      （本组唯一变化的量）
固定量：       镜像 digest / TP / 绑核 / 模型 revision / cudagraph 模式 / VLLM_USE_V1 / 是否 MTP
观测：        prepare input / forward / post process / sample_token 的 phase wall；
              engine-core 主线程 CPU 时间；端到端 TTFT/TPOT/吞吐；
              num_scheduled_tokens 序列；显存/KV 占用；host noise 快照
预期判别：    （什么结果算支持，什么结果算证伪）
```

---

## 2. 主矩阵（逐组）

### 组 A｜并发扫描（decode 重）— 拟合 `c_p, a, e, d`

- 目的：解出 decode 的 `(c_p, a)` 与 `(e, d)`，给出临界 `B*`。
- 自变量：`concurrency ∈ {1,2,4,8,16,32,64,128,256}`（9 点，≥6 达标）。
- 固定：`Qwen3.5-0.8B` BF16、TP1、chip3、CPU 120-159、
  `numactl --cpunodebind=1 --membind=1`、`VLLM_USE_V1=1`、ISL=128、OSL=1024。
- 观测：模板 + `forward` phase wall（`T_dev` 代理）。
- 预期判别：
  - 支持 P1 → `T_prepare` 斜率落在 0.20–0.60 µs/req；
  - 证伪 → 斜率 >1 µs/req（报告里须列出 perf top 中新增的 O(B) 帧）；
  - `T_prepare/T_dev` 随 B 单调上升 ⇒ `a > d` 形态成立。

### 组 B｜输入长度扫描（prefill 重）— 拟合 `a_T`、验证 P2

- 目的：确认 prepare 的 O(T) 系数是 ns/token 级，且长 prefill 不会让 prepare 露头。
- 自变量：`ISL ∈ {128,512,2048,8192,16384,32768}`，`OSL=1`，`bs=1`。
- 观测：模板 + `T_dev`（prefill forward）。
- 预期判别：`T_prepare` 随 ISL 增长 <10 µs/8k token ⇒ 支持；
  若超过 50 µs/8k token，说明存在未识别的 O(T) 项（候选：H2D 字节、`.tolist()`、Triton 参数准备）。

### 组 C｜chunked prefill（加密到 5 点）— 拟合 `β, γ`、验证 P2

- 目的：找到 `T_min`（chunk 小到 CPU 主导的阈值）。
- 自变量：`--max-num-batched-tokens ∈ {512, 1024, 2048, 4096, 8192}`，ISL=32768，bs=1。
- 观测：模板 + TTFT。
- 预期判别：若 512 与 1024 的 `prepare input` phase wall 相同（±5%）而 forward 时间显著不同
  ⇒ **平台存在，P2 成立**；若 phase wall 随 chunk 单调下降 ⇒ P2 被证伪。

### 组 D｜混合负载（真实服务形态）

- 目的：给"真实占比"一个锚点（外推校验，非拟合用）。
- 参数：ISL=2048、OSL=256、`concurrency ∈ {8,32,128}`。
- 预期判别：`share_step` 应落在组 A 与组 B 的包络之间；偏离则说明混合 batch 有额外分支（E2/E5）。

### 组 E｜MTP 开关

- 目的：量化 `A_N`（5×pin+H2D + O(B·n)）与优化 O2 的收益上限。
- 自变量：D 的配置 ± `--speculative-config`（k=1、k=2）。
- 观测：模板 + 火焰图中 `pin_memory`/`aclrtMemcpyAsync` 帧占比。
- 预期判别：支持 → MTP 的 prepare 增量 ≈ 5×(3–10 µs) + 0.16 µs/req×B；
  证伪 → 增量 ≪15 µs（说明 Ascend 上 pin/H2D 比 CUDA 便宜，需回写 §6.3）。

### 组 F｜块大小 / 前缀命中（块表规模）

- 目的：标定 `a_K·B·K·G`（DMA 字节项）。
- 自变量：`block_size ∈ {64,128}`、`prefix hit ∈ {0,0.5,0.9}`、`max_model_len ∈ {8192, 32768}`。
- 观测：模板 + 火焰图里 `copy_to_gpu`/`memcpy` 帧占比。
- 预期判别：`T_prepare` 随 `K` 的斜率 ≈ `B·4·G/k_dma`；若 K 翻倍而 prepare 不变，
  说明拷贝被完全重叠（H5 成立），应把该项从 `c₀` 降权。

### 组 G｜异步开关对照（新增）— 决定"掩盖"是否存在

- 目的：量化 `T_prepare` 的真实增量与"可见增量"的差；给出串行流的判据。
- 自变量：`--async-scheduling {true,false}` × B ∈ {1, 64}（D 组配置）。
- 观测：`T_step`、TPOT、phase wall 各分项。
- 预期判别：async off 时 `T_step − T_step(async on) ≈ T_prepare`（若此前被掩盖）
  ⇒ 直接给出 §4.1 中 `B_cpu` 的实测值。

### 组 H｜请求波动对照（新增）— 标定 `A_C`

- 目的：量化 `condense()`/`swap_states()` 的 O(B·S) 突变（E1/E2）。
- 自变量：
  - 稳态：B=128，全部请求长时间运行（OSL 大、无新进）；
  - 高 churn：B=128，OSL=8 且持续灌入新请求（每步约 12% 请求被替换）；
  - 抢占：并发 > KV 容量（触发 preemption/resume）。
- 观测：`prepare input` phase wall 的 P50/P95/P99，以及 `batch_update_builder.removed` 非空比例
  （若能有卡环境插桩）。
- 预期判别：高 churn 的 P95/P50 > 1.3 ⇒ E1 成立；否则该机器上 condense 被摊销。

### 组 I｜TP 放大（新增）— 验证 §4.4 与 P4

- 目的：证明"TP 不降低 CPU 时间、但降低 device 时间"，从而更早进入 CPU 受限。
- 自变量：A 组的 B ∈ {1, 32} × TP ∈ {1, 2, 4}（同机多卡；注意锁与"不许碰别人的卡"）。
- 观测：每 rank 的 phase wall 分布（至少 min/med/max），`T_step`。
- 预期判别：支持 P4 → 单 rank `T_prepare` 与 TP=1 相当（±15%），而 `T_dev` 随 TP 下降，
  交叉点 `B*` 左移；同时 `max_rank(T_prepare) − median_rank(T_prepare)` 应 >5%。

### 组 J｜缓冲宽度税（新增）— 拆 `c₀` 中随 W 的部分

- 目的：验证 S11/S12/S15 的无参 `copy_to_gpu()` 与尾部 `fill_` 随 `max_num_reqs` 线性。
- 自变量：`--max-num-seqs ∈ {64, 256, 1024}`，固定 B=8 与 B=64（远小于 W）。
- 预期判别：支持 → `T_prepare` 随 W 显著上升（B=8 时也应可见）；证伪 → 曲线平坦，
  说明这些拷贝被 H2D 异步彻底掩盖（对应优化 O3 收益很小）。

### 组 K｜系数标定（无模型，新增）

| 子实验 | 目的（系数） | 方法 | 预期判别 |
|---|---|---|---|
| K1 小块 H2D | `k_h2d_small` | 1000 次 `CpuGpuBuffer.copy_to_gpu(n)`，n∈{1,64,4096}；`perf` 看驱动帧 | 若 <2 µs/次，则 §6.2 的 75–190 µs 上界偏高 |
| K2 `pin_memory()` | `k_pin` | 循环 1000 次 `torch.from_numpy(arr).pin_memory()` | 与 `torch.npu` 行为对照 |
| K3 kernel launch | `k_launch` | 同一 op 在 graph on/off 下的 CPU 侧差值 | 给出 §4.1 中 `T_launch_fwd` 的量级 |
| K4 块表 DMA | `k_dma_per_B` | B×K 矩阵，单独计时 `commit_block_table` | 得到 GB/s 等效带宽 |

### 组 L｜replica ↔ 真机逐项对齐（新增）

- 目的：完成 §7.4 保真度判据与 D1–D6 归因。
- 自变量：B ∈ {1, 64, 256}（decode）、chunk ∈ {2048, 8192}（prefill）、MTP on/off。
- 观测：replica（无卡，核 200–239）与真机（chip3）在 5 个同名点上各采：
  单步耗时分布、`perf stat` IPC/topdown、perf 火焰图 top-20。
- 预期判别（=交付判据）：topdown ±3 pp、IPC ±10%、top-20 交集 ≥16/20、中位耗时 ±15%。
  未达标 → 输出偏差归因表（每条给量化占比）。

---

## 3. 采集纪律（v1 §2.3 + 本文补充）

1. 每个点写 manifest：镜像 digest、模型 revision、workload 参数、绑核、时间戳、脚本哈希。
2. `prepare input` 与 `forward` 必须**同时**记录；只报占比无法拟合判据公式。
3. 统计前剔除：启动后前 100 step、空 batch step、timing 校准窗口。
4. `perf`/flamegraph 与 msprof/DevKit 不共存；一次只跑一种。
5. 无卡 replica 只用核 200–239；真机只用 120–159 并持 `locks/chip3.lock`。
6. 结论回填 `data/model/complexity-model.json` 的 `status` 字段
   （`pending` → `measured`/`falsified`），保持模型与数据同源。

---

## 4. 优先级（时间受限时的最小充分集）

```
P0: 组 A（9 点）      → 拟合 c_p, a, e, d；给出 B*
P0: 组 G（2 点）      → 判定"掩盖"是否存在
P0: 组 K1/K3          → 把 CPU-only replica 校正到真机
P1: 组 C（5 点）      → 给出 T_min（chunked 场景的核心交付）
P1: 组 L（5 点）      → 保真度判据与归因
P2: 组 E/F/H/I/J      → 覆盖 MTP / churn / TP / W 的放大机制
P3: 组 B/D            → 占比包络与真实锚点
```

> 只做 P0 也能回答"何时成为瓶颈"（`B*` 与掩盖判定），但拿不到 chunked 的 `T_min`；
> 因此建议 C 组一并完成。
