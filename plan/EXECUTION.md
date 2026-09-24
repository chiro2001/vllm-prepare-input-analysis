# prepare_input CPU 侧分析 —— 执行计划

> 版本 1.0 / 2026-09-24。配合 `COORDINATION.md` 使用。

## 0. 交付物清单

| 编号 | 文档 | 核心内容 |
|---|---|---|
| 00 | `docs/00-INDEX.md` | 导航、结论速览、名词表 |
| 01 | `docs/01-prepare-input-code-logic.md` | 边界、调用链、逐步骤逻辑、数据结构 IO 语义 |
| 02 | `docs/02-complexity-and-factors.md` | 复杂度模型 + 影响因素矩阵（含实测） |
| 03 | `docs/03-share-in-inference.md` | 在推理中的占比曲线与瓶颈判据 |
| 04 | `docs/04-profiling-methodology.md` | perf / flamegraph-rs / libkperfx 采集口径与复现命令 |
| 05 | `docs/05-hotspots.md` | 热点函数 / 热点源码行 / 热点指令 + 火焰图 |
| 06 | `docs/06-synthetic-load.md` | 无卡负载构造与保真度验证 |
| 07 | `docs/07-bottleneck-analysis.md` | 何时成为瓶颈、优化建议、外推 |
| 08 | `docs/08-reproduction-manual.md` | 从零复现全流程 |

数据：`data/`（CSV/JSON + 汇总）；图：`figures/`；原始 `perf.data` 留在 a3-22，不进交付包。

## 1. 阶段划分

```
P0 环境/工具链   env_toolchain ───────────┐
P1 静态分析      static_code_analysis ────┤
P2 真机 bring-up service_bringup ─────────┼─→ P3 占比矩阵 (measurement)
P3'无卡 harness  replay_harness ──────────┤   P3'' CPU profiling (perf/flamegraph/libkperfx)
P4 保真度对照（真机 vs 无卡） ─────────────┤
P5 文档/图/打包/上传/links-server ─────────┘
```

## 2. 实验矩阵（P3 占比 + profiling 共用）

固定：镜像 `vllm-ascend:v0.26.0rc1-a3-openeuler`，TP1，chip3，CPU `120-159`，
`numactl --cpunodebind=1 --membind=1`，`VLLM_USE_V1=1`（默认），不使用 msprof/DevKit。

### 2.1 主矩阵（主模型 `Qwen3.5-0.8B` BF16）

| 组 | 目的 | 参数 |
|---|---|---|
| A 并发扫描（decode 重） | 找 CPU/device 交叉点 | ISL=128, OSL=1024, concurrency ∈ {1,2,4,8,16,32,64,128,256} |
| B 输入长度扫描（prefill 重） | ISL 对 prepare_input 的阶 | ISL ∈ {128,512,2048,8192,16384,32768}, OSL=1, bs=1 |
| C chunked prefill | 单步 token 数的影响 | ISL=32768, `--max-num-batched-tokens` ∈ {512,2048,8192}, bs=1 |
| D 混合负载 | 真实服务形态 | ISL=2048, OSL=256, concurrency ∈ {8,32,128} |
| E MTP 开关 | draft token 路径 | D 的配置 ± `--speculative-config`(k=1) |
| F 块大小/前缀命中 | block table 规模 | `--block-size ∈ {64,128}`, prefix hit ∈ {0,0.5,0.9}（同一前缀复用） |

### 2.2 模型结构轴（`Qwen3.5-2B` / `Qwen3.5-27B-w8a8-mtp`）

只在两个点做对照：组 A 的 concurrency=1 与 32，验证"模型结构对 prepare_input 的影响
主要经由 `max_model_len`/`max_num_reqs`（缓冲区宽度）而非权重规模"。

### 2.3 每个实验点必须采

1. `prepare input` / `forward` / `post process` / `sample_token` 的每步耗时（LiteProfiler 或等价）；
2. 端到端：TTFT、TPOT/ITL、吞吐；
3. engine-core 主线程 CPU 时间（`/proc/<pid>/task/<tid>/stat` 增量 + `perf stat`）；
4. 显存/KV cache 占用、实际 `num_scheduled_tokens` 序列。

## 3. Profiling 采集矩阵（P3''）

在 A/B/C/D 各挑 1-2 个代表点，做**四层**证据：

| 层 | 工具 | 产物 |
|---|---|---|
| 时间占比 | LiteProfiler / torch profiler | 每步 phase 表 |
| 函数热点 | `perf record -g -F 999 -t <tid>` | 火焰图 SVG（flamegraph-rs） |
| 顶层分解 | `libkperfx` topdown preset（920B） | frontend/backend/retiring/bad-spec + 子项 |
| 指令/源码热点 | `perf annotate` / `perf report --stdio` + 源码对照 | 热点源码行、热点指令 |

## 4. 无卡复刻的验收判据（P4）

| 维度 | 目标 | 判定方式 |
|---|---|---|
| topdown 分量 | 各分量差 ≤ 3 pp（绝对值） | libkperfx 同 preset 对比 |
| IPC | 相对差 ≤ 10% | libkperfx `cycles,instructions` |
| 热点函数 top-20 | 交集 ≥ 16/20，且首位热点相同 | perf 火焰图/`perf report` 对比 |
| 单步耗时形状 | 分布（中位数/p99）相对差 ≤ 15%，且随 batch 的斜率一致 | 逐步计时 CSV |
| 归因结论 | 关键结论（谁是大头、随什么增长）一致 | 人工核对 |

不达标时要给出**偏差来源的归因**（例如"缺少 ACL enqueue 的 ~x%"），而不是简单标注失败。

## 5. 需要回答的核心问题（交付验收）

1. `prepare input` scope 的精确边界（含/不含 `_update_states`、`synchronize_input_prep`）。
2. 逐子步骤复杂度 + 实测斜率（µs per req / per token / per block）。
3. prefill/decode/chunked/MTP/高并发下的占比曲线。
4. 瓶颈判据：`T_cpu_prepare` 与 `T_device` 的交叉条件，给出可操作的经验公式。
5. 无卡负载：怎么构造、保真到什么程度、哪些特征无法复现及原因。
6. 热点函数/源码/指令 top-N，以及可优化点（含预期收益与风险）。

## 6. 协作与锁

- chip3 串行：任何占卡动作先取锁 `a3-22:~/projects/vllm/prepare-input-phase/locks/chip3.lock`。
- 无卡 harness 不许用 CPU `120-159`；建议 `200-239` / `360-399`。
- 每个 agent 报告写 `agents/<name>/REPORT.md`，结论回填 `docs/`。
- 采集前跑 `scripts/hostnoise_gate.sh`；采集前后记录 host noise 快照。
