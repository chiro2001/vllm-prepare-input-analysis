# 03 章图与数据表（plot_measure.py 生成）

- 输入：`data/measure/_pipeline_test/points_flat.json`（1 条可用 point 记录）
- 数据目录：`data/measure/_pipeline_test`；图目录：`/tmp/fig_test2`
- 前缀：`03`；标题后缀：`''`

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
| _test | R | 1 | 128 | 64 | - | - | decode | 63 | 4090.980 | 73.600 | 5570.470 |

## 图


## 未生成的图 / 缺失数据

- `03-prepare-vs-concurrency.svg`：数据不足（该轴上少于 2 个有效点）
- `03-prepare-share-vs-concurrency.svg`：数据不足（该轴上少于 2 个有效点）
- `03-prepare-vs-device-cross.svg`：数据不足（该轴上少于 2 个有效点）
- `03-isl-scan.svg`：数据不足（该轴上少于 2 个有效点）
- `03-chunked-prefill.svg`：数据不足（该轴上少于 2 个有效点）
- `03-mtp-onoff.svg`：数据不足（该轴上少于 2 个有效点）

## 解析告警

- figure subsets: concurrency figures use 1 point(s) from groups A/G/I/J; isl-scan uses all points; chunked-prefill uses all points

