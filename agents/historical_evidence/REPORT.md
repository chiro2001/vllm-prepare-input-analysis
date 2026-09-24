# historical_evidence —— 交接报告

> 时间：2026-09-24（Asia/Shanghai）｜范围：**只做历史数据挖掘**，未跑 benchmark、
> 未占 NPU、未起服务、未改 `HIST_PROJECT`（全程只读）。
>
> 主要交付：`docs/03-share-in-inference.md`（前半草稿）、`data/historical/*.csv`、
> `figures/hist-0{1,2,3}-*.{svg,png}`、本报告。

## 0. 一分钟结论

1. **`prepare input` 的绝对成本与模型规模基本无关**：历史 26 个 decode 配置里，
   只要不是 graph+MTP-on，它都落在 **1.6–6.4 ms/步**（跨 0.8B→27B、TP1→TP4、graph/eager）。
2. **占比由分母决定**：小模型 graph 是 **48–59%**（真瓶颈候选）；大模型 graph+MTP-off
   只有 **9–14%**；eager 只有 **2.7–7.3%**；**大模型 graph+MTP-on 的 74% 是自旋假象**。
3. **CPU 侧画像很稳定**：worker 主线程 **frontend-bound 56–65%**、**IPC 0.72–0.89**，
   热点摊在 CPython 解释器（`_PyEval_EvalFrameDefault` 10.33%）、对象分配、
   属性/字典查找与锁上；**算子下发接口只占 0.055%**。
4. **历史数据能支撑「占比量级」与「CPU 画像」，不能支撑「`prepare input` scope 内的热点归因」
   「输入长度/并发斜率」「L1–L3/DRAM」「端到端占比」**（见 §3）。

## 1. 关键数字（都带出处）

### 1.1 每步 phase（`data/historical/phase_per_step_by_tag.csv`）

| 配置（rank0） | step 周期 | prepare input | 占比 | forward | 来源 tag |
|---|---:|---:|---:|---:|---|
| 0.8B TP1 graph MTP-off | 4.89–4.93 ms | **2.91–2.95 ms** | **59.6%** | 0.89–0.91 ms | e1a / m1a |
| 0.8B TP1 graph MTP-on | 10.05–10.10 ms | **5.55–5.61 ms** | 55% | 0.90 ms | e1a / m1a |
| 0.8B TP4 graph MTP-off | 5.33 ms | **3.12 ms** | 58% | 0.98 ms | m4 |
| 2B TP1 graph MTP-off | 5.78–5.80 ms | **2.76–2.85 ms** | 48–49% | 1.89–1.99 ms | e1b / m1b |
| 2B TP1 graph MTP-on | 10.74–10.82 ms | **6.34–6.43 ms** | 59% | 0.88 ms | e1b / m1b |
| 17B TP1 graph MTP-off | 5.49–5.64 ms | **1.60–1.65 ms** | 29% | 2.95–3.00 ms | e1a / m1a |
| 27B TP1 graph MTP-off | 34.87 ms | **3.09 ms** | 8.8% | 30.64 ms | m1c |
| 38-27B TP1 graph MTP-off | 31.20–31.29 ms | **2.83–2.93 ms** | 9.1% | 27.25 ms | e1b / m1b |
| 27B TP1 graph **MTP-on** | 46.20–46.30 ms | **35.27 ms** | 76% | 5.55 ms | e1c / m1c |
| 38-27B TP1 graph **MTP-on** | 42.36–42.42 ms | **31.37–31.45 ms** | 74% | 5.53 ms | e1b / m1b |
| 0.8B TP1 eager MTP-off | 36.05–37.16 ms | **2.60–2.71 ms** | 7.2% | 32.3–33.3 ms | e1a / m1a |
| 27B TP1 eager MTP-off | 94.56 ms | **2.67 ms** | 2.8% | 90.44 ms | m1c |

**交叉校验**：抽取值与既有文档独立吻合 —— 38-27B graph MTP-off e1b 抽出
**2.831 / 27.246 ms**，`MTP_PREPARE_INPUT_SPIN.md` §0.4 记的是 **2.83 / 27.28 ms**；
27B graph MTP-on 抽出 **35.27 / 5.55 ms**，该文档记 **35.30 / 5.50 ms**。

### 1.2 top-down / IPC（`data/historical/worker_thread_topdown_configs.csv`）

- graph 模式 **MTP-off 的全部 17 个臂**：frontend **55.99–65.00%**、IPC **0.719–0.890**；
  跨 0.8B / 2B / 17B / 27B / 38-27B / 80B、TP1/2/4 都成立。
- **graph + MTP-on 的 27B / 80B 臂**（backend 38.95–56.66%、IPC 0.975–1.117）是自旋污染；
  扣自旋后与 MTP-off 逐项差 ≤0.2 pp。
- **eager 臂**是天然无自旋对照（在核 99.9%），四桶 MTP on/off 差 ≤1.5 pp。
- **置信度**：a3-22 上 43 个 run 的 182 份 topdown `raw.csv`、724 行事件
  **全部 `percent_enabled = 100.00`**（无 multiplexing）；`time_enabled` 4.48–24.65 s。
  `perf stat` 的 3 个有效 run 同样 100.00。

### 1.3 热点（`data/historical/hotspots_*.csv`）

- 主线程自采样（0.8B / TP1 / graph / MTP-off / chip15 / 4 kHz / 31.9 s）：
  top-1 `_PyEval_EvalFrameDefault` **10.33%**；top-10 合计 23.70%、top-30 合计 32.19%
  —— **剖面极平，无单点尖峰**。按 tag：python_eval 54.17%、allocator 14.43%、
  kernel_syscall 8.73%、lock_or_sync 7.67%、torch_dispatch 3.86%（分母 = top-30 之和）。
- DSO 归并（另一轮，libtorch_npu 带全量调试符号）：**CPython 39.97% + libtorch 族 35.55%**，
  CANN 只有 6.24%；inclusive 首位是 `THPVariable_getitem`（14.00%）。
- **算子下发接口全族 0.055%**，`aclmdlRIExecuteAsync` 0.002%。

### 1.4 每 token 的 CPU 成本

| 口径 | 数值 | 来源 |
|---|---|---|
| 每输出 token 指令数（离线 generate） | **9.2 M**（MTP-off）/ 11.6 M（MTP-on） | `vllm-slice-insn-opt/results/SUMMARY.md` |
| 每输出 token 周期数（服务路径） | **11.56 / 12.35 / 11.87 M** | `worker_thread_perfstat.csv` |
| IPC（服务路径） | **0.742–0.787** | 同上 |
| task-clock / 窗口 | **0.92–0.98 CPU** | 同上 |
| 上下文切换 | 2.7–2.9 K/s | 同上 |

## 2. 数据产物清单

```
data/historical/
  phase_per_step_windows.csv (+.meta.json)    2424 行：52 run x 489 窗口 x 作用域
  phase_per_step_configs.csv (+.meta.json)      26 行：配置级每步均值
  phase_per_step_by_tag.csv                     61 行：按波次分
  worker_thread_topdown_windows.csv            362 行：窗口级四桶 + IPC
  worker_thread_topdown_configs.csv (+.meta)    71 行：配置 x rank
  worker_thread_mode_eager_vs_graph.csv         60 行
  worker_thread_perfstat.csv (+.meta.json)        3 行
  hotspots_perf_record_top.csv (+.meta.json)     30 行
  hotspots_perf_record_tags.csv                   8 行
  hotspots_perf_record_slice_insn_mtp_off.csv   670 行
  hotspots_perf_record_slice_insn_mtp_on.csv    773 行
  throughput_baseline.csv (+.meta.json)          60 行
  tokens_per_step.csv                           110 行
  mtp_attribution.csv                             4 行
  spin_attribution_arms.csv (+.meta.json)         5 行
figures/
  hist-01-phase-decomposition.{svg,png}
  hist-02-prepare-vs-step-period.{svg,png}
  hist-03-topdown-ipc.{svg,png}
agents/historical_evidence/
  scripts/extract_lite_phases.py     lite trace -> 每窗口 phase 表
  scripts/assemble_historical.py     汇总成配置级表 + 元数据
  scripts/plot_historical.py         三张图
  raw/a321_perf_analysis/            从 a3-21 复制的 perf 分析产物（含 folded 调用栈 2.07 MB）
  raw/a321_perfstat/                 从 a3-21 复制的 perf stat 原始 csv/json
```

## 3. 数据空洞清单（新实验必须补）

| # | 空洞 | 为什么必须补 | 具体动作 |
|---:|---|---|---|
| G1 | **热点无法归到 `prepare input` scope 内** | 历史 perf 按 TID 采；「prepare input 里在算什么」目前只能外推 | 在 `prepare input` 内打 S1–S26 子 scope，再做 scope 对齐采样（或在 scope 边界打时间戳与 perf 样本对齐） |
| G2 | **没有 L1/L2/L3/LLC/DRAM 计数器** | task 明确要的「L1/L2/L3/DRAM 相关指标」历史完全缺失 | 用已定好的 5 组（`core`/`l1l2`/`l3llc-mem`/`tlb-d`/`tlb-i`）采；确认 `min(running/enabled) >= 95` |
| G3 | **没有 TTFT / TPOT** | `share_of_e2e` 没有分母 | 新实验的端到端基准（含 `metrics.delta.json`） |
| G4 | **workload 维度单一**（ISL=128、C=1） | per-token / per-req 斜率全未实测 | A/B/C/D 组 |
| G5 | **没有 prefill / chunked-prefill 的 phase** | prefill 是 `prepare_input` 的另一半工作量（prompt embeds、`_prepare_input_ids`） | B/C 组 |
| G6 | **MTP 残余自旋约 9 ms/步未归因** | 26 个历史 lite 配置里最反直觉的一条 | 子 scope 定位；或按 §8.1 的「换流」实验 |
| G7 | **没有现成火焰图文件**（本地那份 `callgraph.folded.*` 不是 folded 格式，实测 0 行含 `;`） | task 明确要求火焰图 | 原始 `perf.data`（30.7 MiB，在 a3-21 的 `runs/20260914T1014Z-...-perf-record-r1/perf/`）仍在，现场 `perf script \| stackcollapse-perf.pl \| flamegraph-rs` 即可；`raw/.../record-rank0.stacks.json` 的 `callchains.top_stacks` 也能直接画 top-30 栈 |
| G8 | **跨波次不可比** | 同一配置 w* 与 e*/m* 的 step 周期差 15–30% | 新实验固定镜像 digest + 绑核 + 每轮 manifest；比值只用同轮内的 |

补充说明：

- **G2 的细节**：920B 每组最多 8 个硬件事件可 100% running，第 9 个整组失效（fail-closed）。
  推荐 5 组方案在 `PERF_EVENT_SET_OPTIMIZATION.md` §4.4 已定稿，两机实测 21/21 组有效，
  **但历史上一组都没采**（`runs/*/perf/` 里只有 `stat-ipc.csv` 与 `stat-software.csv`）。
- **G3 的细节**：`reports/dataset/long-table.csv` 只有 7 行、来自 2026-09-14 最早的 run；
  `summary/summary.json` 的 `per_config_rank` 有 6 组但都不含 TTFT/TPOT 有效值。
  历史里唯一接近的是 lite trace 派生的 `throughput_mean`（步数 × tokens/步 ÷ 窗口秒），
  它是单请求 decode 速率，**不是端到端指标**。

## 4. 建议的新实验点（按性价比排序）

1. **`prepare input` 子 scope 化**（最高优先级）。没有这一步，G1/G6 永远无解，
   而且它同时服务「热点归因」「残余自旋定位」「per-token 斜率」三件事。
2. **并发扫描 A 组**（1→256，0.8B graph）：历史完全空白，
   而这是「`prepare input` 何时成为瓶颈」最直接的曲线。
3. **ISL 扫描 B 组 + chunked prefill C 组**：验证 `docs/02` 的 per-token 复杂度，补 G4/G5。
4. **cache 计数器（G2）**：只有它能回答「CPU 侧是前端受限还是访存受限」；
   历史证据（frontend 56–65%）指向前端，但**没有 cache 数据佐证**。
5. **火焰图（G7）**：需要一次现场转换（`perf.data` → folded → SVG），
   但 `perf.data` 与 `record-rank0.stacks.json` 都在，成本很低，且是交付物硬要求。

## 5. 踩过的坑（写给后续 agent）

1. `reports/` 只存在于**本地** `~/projects/vllm/HIST_PROJECT/`；
   a3-22 的 `HIST_PROJECT` 里**没有** `reports/`，a3-21 的里面有。
   一手 run 数据在 **a3-22**（约 175 个 run 条目）与 **a3-21**（222 个 run）。
2. `reports/lite-traces/<run>/lite.trace.json.gz` 的 TP4 trace **一个文件含 4 个 rank 泳道**，
   且 phase 泳道在另一个 `tid=1073741xxx` 上 —— 按「一个文件一个线程」处理会把 4 个 rank 混池。
3. 步数锚点不能只用 `Step:Schedule`：`e1*/m1*` 波次的镜像**没有** schedule 插桩。
   回退到 `prepare input`（与 `forward` 计数差 ≤1，差的是窗口边界上的半步）。
4. 求和覆盖率时**不要把嵌套作用域加进去**：`schedule:*` 在 `Step:Schedule` 内、
   `output:*` 在 `Step:Output` 内、`model: mtp:*` 在 `draft_token` 内。
   否则覆盖率会算成 102–106%。
5. `perf stat -x,` 的输出里，**第 5 列是 `percent_enabled`**（不是 `time_running`）；
   `stat-*.csv` 比 topdown `raw.csv` 多一个前置 task 列，字段偏移不同。
6. 历史 `perf.data` 与 `perf-analysis/` 在 **a3-21**（不在 a3-22）；
   a3-22 的 run 目录里 `perf/` 一个都没有（只有 `topdown-l1/`、`lite-profiler/`、`mappings/` 等）。
7. 本机（开发机）没有 `perf`；a3-22 上普通用户被拒（`perf_event_paranoid=2`），
   但 `sudo -n perf` 可用。
8. 跨机墙钟不可混用：a3-22 的墙钟比 a3-21 快约 8.11 h
   （`analysis/README.md` §口径纪律 1），本文所有跨机数字都只用时长/比值，不用墙钟排序。
