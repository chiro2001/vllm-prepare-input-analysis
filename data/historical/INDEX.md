# data/historical —— 文件索引与溯源

来源层级：**T0 一手产物**（a3-21 / a3-22 的 `runs/`）→ **T1 归约产物**
（`HIST_PROJECT/reports/`，由既有工具链生成）→ **T2 本目录**（本 agent 抽出）。
本目录全部是 **T2**；每个文件的 T0/T1 出处见下表与同名 `.meta.json`。

生成方式（可复现）：

```bash
python3 agents/historical_evidence/scripts/extract_lite_phases.py \
  --traces-root ~/projects/vllm/HIST_PROJECT/reports/lite-traces \
  --out-dir data/historical
python3 agents/historical_evidence/scripts/assemble_historical.py
python3 agents/historical_evidence/scripts/plot_historical.py
```

| 文件 | 行数 | 内容 | T1 出处 | meta |
|---|---:|---|---|---|
| `phase_per_step_windows.csv` | 2424 | 每个 lite trace 的每个 25 s 窗口 × 每个作用域的 count/mean/p50/p90/p99 | `reports/lite-traces/<run>/lite.trace.json.gz`（52 份） | `.meta.json` |
| `phase_per_step_configs.csv` | 26 | 配置级每步 phase 均值与占比 | 同上 | `.meta.json` |
| `phase_per_step_by_tag.csv` | 61 | 同上，但按波次 tag（w1/m1a/e1b…）分开 | 同上 | `phase_per_step_configs.meta.json` |
| `worker_thread_topdown_windows.csv` | 362 | 每个 25 s 窗口的四桶 + IPC + 原始计数 | `reports/topdown-campaign/topdown-l1-windows.csv` | `worker_thread_topdown.meta.json` |
| `worker_thread_topdown_configs.csv` | 71 | 配置 × rank 的四桶与 IPC 均值/极差 | `reports/topdown-campaign/per-config-rank.csv` | 同上 |
| `worker_thread_mode_eager_vs_graph.csv` | 60 | graph / eager 对照（含 lite 派生吞吐） | `reports/mode-campaign-815/per-config.csv` | 同上 |
| `worker_thread_perfstat.csv` | 3 | cycles / instructions / IPC / task-clock / ctx-sw / page-fault | `a3-21:runs/2026091*Z-...-perf-stat-r1/perf/stat-{ipc,software}.*` | `.meta.json` |
| `hotspots_perf_record_top.csv` | 30 | worker 主线程自采样 top-30（period 加权） | `a3-21:perf-analysis/20260914T1014Z-perf-record-r1/record-rank0.stacks.json` | `.meta.json` |
| `hotspots_perf_record_tags.csv` | 8 | 上述 top-30 按 tag 归并（**分母是 top-30，不是全剖面**） | 同上 | 同上 |
| `hotspots_perf_record_slice_insn_mtp_off.csv` | 670 | 带 IPC 列的主线程剖面（离线 generate，带全量调试符号） | `vllm-slice-insn-opt/results/perf-prof-08b-mtp-off.flat.txt` | `hotspots_perf_record_top.meta.json` |
| `hotspots_perf_record_slice_insn_mtp_on.csv` | 773 | 同上（MTP-on） | `...-mtp-on.flat.txt` | 同上 |
| `throughput_baseline.csv` | 60 | 每配置吞吐（lite 派生）+ IPC + 四桶 | `reports/mode-campaign-815/per-config.csv` | `.meta.json` |
| `tokens_per_step.csv` | 110 | 每 run 每 replay 的步数 / 吞吐 / tokens per step | `reports/mtp-attribution/tokens-per-step.csv` | `throughput_baseline.meta.json` |
| `mtp_attribution.csv` | 4 | MTP 归因：K、接受长度、四桶偏移分解、tokens/step | `reports/mtp-attribution/attribution.csv` | 同上 |
| `spin_attribution_arms.csv` | 5 | 自旋五臂（A/B/C1/C2/C）的 cycles 与四桶 | `reports/spin-attribution/windows.csv` | `.meta.json` |

## 单位与口径速查

| 字段族 | 单位 | 说明 |
|---|---|---|
| `*_us` / `step_period_us` | 微秒 | lite trace 的原始时间基就是微秒 |
| `*_ms` | 毫秒 | 由 `*_us` 换算，便于阅读 |
| `*_share_pct` / `*_percent` | 百分数（0–100） | 除 `share_of_top30_percent` 外，分母见表头说明 |
| `cycles` / `instructions` | 次数 | 主线程（`perf stat -t <worker main TID>`） |
| `ipc` | inst/cycle | `inst_retired / cycles` |

## 已知坑（引用前必读）

1. **`prepare input` / `forward` 的墙钟含 device 等待**：这些 lite 波次没有 wait scope，
   graph+MTP-on 的 27B 臂尤其不能当作 CPU 侧耗时（见 `docs/03` §3）。
2. **跨波次不可比**：`batch_tag` 不同的行不要直接比绝对值（同配置可差 15–30%）。
3. **TP4 trace 一文件含 4 个 rank**：`rank` 列已拆开，勿按 run 聚合而忽略 rank。
4. **topdown 无 multiplexing**：a3-22 的 182 份 raw.csv / 724 行事件全部
   `percent_enabled = 100.00`；`time_enabled` 4.48–24.65 s。
5. **`hotspots_perf_record_tags.csv` 的分母是 top-30 之和**，不是整份 profile（86.02 G cycles）。
   要全剖面口径请用 `share_of_full_profile_percent` 列。
