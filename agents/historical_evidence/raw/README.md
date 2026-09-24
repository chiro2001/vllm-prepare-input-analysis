# raw/ —— 从远程只读复制的原始制品

复制时间：2026-09-24。来源机器：**a3-21**（`HIST_PROJECT` 在 a3-21 上有 `perf-analysis/`，
在 a3-22 上没有）。复制过程只读，未修改任何远端文件。

## a321_perf_analysis/  ← `a3-21:HIST_PROJECT/perf-analysis/20260914T1014Z-perf-record-r1/`

| 文件 | 大小 | 内容 | 是否可用 |
|---|---:|---|---|
| `record-rank0.stacks.json` | 63 KB | 结构化热点报告：top-30 符号（带 tag）、top-30 调用栈、完整性自证、categories/totals | 可用（`data/historical/hotspots_perf_record_top.csv` 从这里生成） |
| `record-rank0.stacks.md` | 36 KB | 同一报告的人类可读版 | 可用 |
| `flat-symbols.final.txt` | 3.2 MB | `perf report --sort symbol` 的 flat 报告 | **注意：符号未解析**（大面积 `0x…` 地址），只有 DSO 与地址 |
| `callgraph.folded.final.txt` | 2.1 MB | 文件名像 folded，**实测 0 行含 `;`，是 flat 报告**，不是折叠栈 | **不可直接喂 flamegraph** |

原始 `perf.data`（30.7 MiB，125,703 样本，4 kHz，31.9 s）**没有复制**，仍在
`a3-21:HIST_PROJECT/runs/20260914T1014Z-qwen35-08b-tp1-mtp-off-long-decode-b1-perf-record-r1/perf/record-rank0.perf.data`。
需要火焰图时在那台机器上 `perf script -i <perf.data> | stackcollapse-perf.pl > folded` 再渲染。

## a321_perfstat/  ← `a3-21:HIST_PROJECT/runs/2026091{4}T09xxZ-...-perf-stat-r1/perf/`

| 文件 | 内容 |
|---|---|
| `*.stat-ipc.csv` | `perf stat` 原始输出（cycles / instructions） |
| `*.stat-ipc.normalized.json` | 归一化后的结构化结果（含 `time_running_percent`、per-decode-second、per-output-token） |
| `*.stat-ipc.window.txt` | 采集窗口与屏障语义（`barrier_semantics=client_read_barrier`、`server_pause_guaranteed=false`） |
| `*.stat-software.csv` | task-clock / context-switches / cpu-migrations / page-faults |

`20260914T0946Z` 那一轮是 `<not counted>`（`time_enabled=0`），**已从
`data/historical/worker_thread_perfstat.csv` 剔除**，对应文件不在本目录。

## 引用一致性

`a3-22` 上的 topdown `raw.csv`（182 份 / 724 行）**没有**复制到这里，
但已在分析中用它们的 `percent_enabled` 列做过置信度核对（全部 100.00），
结论写在 `data/historical/worker_thread_topdown.meta.json` 与
`agents/historical_evidence/REPORT.md` §1.2。
