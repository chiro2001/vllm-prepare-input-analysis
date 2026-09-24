# 03 章图与数据表（plot_measure.py 生成）

- 输入：`data/measure/_analyzer_selftest/points.json`（19 条可用 point 记录）
- 数据目录：`data/measure/_analyzer_selftest`；图目录：`data/measure/_analyzer_selftest/figures`
- 前缀：`03`；标题后缀：`' (synthetic self-test)'`

## 口径

- 每个点的指标取自 `point.json` → `phases.summary`（`pi-lite-summary-v1`）。
  **默认用 `decode` 波段**（engine 稳态）；decode 的 prepare 步数 < 5 的 prefill-only 点退化为 `all`。每个点实际用的波段见下表。
- `prepare *_us` / `forward *_us` / `step_p50_us` 都来自该波段的 `p50_us`；
  `step_p50_us` = `by_phase[phase].step_dur.p50_us`（engine 单步 wall）。
- `prepare_share_p50` = `share[phase].prepare_over_step_p50`，单位 **%**；
  分子 = 单步 `prepare input` scope，分母 = 同一步的 step scope（async 下是
  `Step:Model` dispatch，不覆盖整轮 engine loop；要整轮分母请用
  `analyze_lite.py --step-scope Step:Schedule --window-mode next`）。
- `ttft_ms` / `itl_ms` / `tps` = 该点所有 round 的 `ttft_s_mean` /
  `mean_itl_ms` / `output_tps_aggregate` 的算术平均（单位分别 ms / ms / token/s）。
- `on_cpu_ratio` 来自 `point.json → cpu.on_cpu_ratio`（engine-core 主线程
  `/proc/<pid>/task/<tid>/schedstat` 的 on-CPU 占比）。
- 图表标签一律英文：容器/开发机 matplotlib 无中文字体，中文会渲染成方框。

## 生成的点表

| tag | group | B | ISL | OSL | chunk | MTP | 波段 | prepare n | prepare p50 us | share p50 % | step p50 us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a-b1 | A | 1 | 128 | 1024 | 2048 | off | decode | 64 | 934.000 | 31.530 | 2962.000 |
| a-b2 | A | 2 | 128 | 1024 | 2048 | off | decode | 64 | 968.000 | 32.220 | 3004.000 |
| a-b4 | A | 4 | 128 | 1024 | 2048 | off | decode | 64 | 1036.000 | 33.550 | 3088.000 |
| a-b8 | A | 8 | 128 | 1024 | 2048 | off | decode | 64 | 1172.000 | 36.000 | 3256.000 |
| a-b16 | A | 16 | 128 | 1024 | 2048 | off | decode | 64 | 1444.000 | 40.200 | 3592.000 |
| a-b32 | A | 32 | 128 | 1024 | 2048 | off | decode | 64 | 1988.000 | 46.620 | 4264.000 |
| a-b64 | A | 64 | 128 | 1024 | 2048 | off | decode | 64 | 3076.000 | 54.850 | 5608.000 |
| a-b128 | A | 128 | 128 | 1024 | 2048 | off | decode | 64 | 5252.000 | 63.310 | 8296.000 |
| a-b256 | A | 256 | 128 | 1024 | 2048 | off | decode | 64 | 9604.000 | 70.250 | 13672.000 |
| b-isl128 | B | 1 | 128 | 1 | 8192 | off | decode | 64 | 2600.269 | 66.540 | 3908.077 |
| b-isl512 | B | 1 | 512 | 1 | 8192 | off | decode | 64 | 2601.075 | 66.150 | 3932.307 |
| b-isl2048 | B | 1 | 2048 | 1 | 8192 | off | decode | 64 | 2604.301 | 64.640 | 4029.229 |
| b-isl8192 | B | 1 | 8192 | 1 | 8192 | off | decode | 64 | 2617.203 | 59.250 | 4416.915 |
| c-chunk2048 | C | 1 | 32768 | 1 | 2048 | off | decode | 64 | 2656.144 | 61.380 | 4327.664 |
| c-chunk512 | C | 1 | 32768 | 1 | 512 | off | decode | 64 | 2656.144 | 67.090 | 3959.024 |
| c-chunk8192 | C | 1 | 32768 | 1 | 8192 | off | decode | 64 | 2656.144 | 45.780 | 5802.224 |
| e-mtp-off | E | 64 | 2048 | 256 | 2048 | off | decode | 64 | 3076.000 | 54.660 | 5628.000 |
| e-mtp-on | E | 64 | 2048 | 256 | 2048 | on | decode | 64 | 3106.000 | 45.640 | 6805.200 |
| r-no-analyzer | R | 8 | 128 | 256 | 2048 | off | - | - | - | - | - |

## 图

- `data/measure/_analyzer_selftest/figures/03-prepare-vs-concurrency.svg`
- `data/measure/_analyzer_selftest/figures/03-prepare-share-vs-concurrency.svg`
- `data/measure/_analyzer_selftest/figures/03-prepare-vs-device-cross.svg`
- `data/measure/_analyzer_selftest/figures/03-isl-scan.svg`
- `data/measure/_analyzer_selftest/figures/03-chunked-prefill.svg`
- `data/measure/_analyzer_selftest/figures/03-mtp-onoff.svg`

## 未生成的图 / 缺失数据

- 无：6 张图全部生成。

## 解析告警

- r-no-analyzer: no phases.summary (analyzer not run?), performance columns stay empty
- r-no-analyzer: analyzer summary has no usable prepare-input block; performance columns stay empty
- record 19: unexpected schema 'not-a-point', skipped
- 9 of 19 points are in the concurrency groups (A/G/I/J) used for the three concurrency figures
- figure subsets: concurrency figures use 9 point(s) from groups A/G/I/J; isl-scan uses 4 group-B points; chunked-prefill uses 3 group-C points

