# LiteProfiler 解析 + 绘图（`measurement_profiling/lite_analyzer` 子代理笔记）

> 交付时间：2026-09-24。父任务：`/root/measurement_profiling`。
> 本节记录 4 个交付物的接口、**与现有解析器的逐项对比**、以及两条必须知道的
> 口径事实（async 双 `Step:Model`、分母选择）。**没有碰 chip3、没有拿锁、
> 没有跑任何 benchmark**：全部结论来自只读拷贝的 smoke 切片。

---

## 0. 交付物与运行方式

| # | 路径 | 作用 |
|---|---|---|
| 1 | `scripts/measure/analyze_lite.py` | LiteProfiler 切片 → `steps.csv` + `summary.json`（`pi-lite-summary-v1`）+ 文本报告；**纯 stdlib** |
| 2 | `scripts/measure/plot_measure.py` | `point.json` 数组 → 6 张 SVG + `matrix.csv` + `README.md`；仅依赖 matplotlib |
| 3 | `agents/measurement_profiling/ANALYZER_NOTES.md` | 本文件 |
| 4 | `data/measure/_analyzer_selftest/**` | 自测产物（真实切片解析结果 + 合成 points.json 出的 6 张图） |

```bash
# 契约 A（point_run.py 的调用方式，逐字兼容）
python3 scripts/measure/analyze_lite.py LITE_LOG [--csv OUT.csv] [--json OUT.json] \
        [--txt OUT.txt] [--tid TID] [--step-scope "Step:Model"] [--warmup-steps N]

# 契约 B
python3 scripts/measure/plot_measure.py --points-json POINTS.json --outdir figures \
        --data-outdir data/measure/<run> [--prefix 03] [--title-suffix ""]
```

**两个附加（不影响契约的）可选项**，仅在需要时用：

```bash
# 异步模式下拿"整轮 engine loop"分母（见 §5）
python3 scripts/measure/analyze_lite.py LITE_LOG --step-scope Step:Schedule --window-mode next
```

* `--step-scope`：默认 `Step:Model`；没有 `Step:*` 插桩的镜像退回 `prepare input`（带 warning）。
* `--window-mode scope|next`：`scope`（默认）= 契约规定的 `[start, start+dur)`；
  `next` = `[start_i, start_{i+1})`。

---

## 1. 自测输入：smoke 切片事实

`runs/smoke-qwen3508b-tp1-async-on-20260923T1740Z/lite-profiler/lite.log`
（Qwen3.5-0.8B / TP1 / async scheduling **on** / chip3 / ISL=128 / OSL=64）：

| 事实 | 值 |
|---|---|
| 日志行 | 982 行全部可解析，malformed = 0 |
| 写日志的 tid | `1`（66 行：http/输入处理）、`131`（915 行：**engine core**）、`1131`（1 行：tokenizer） |
| 选中的 engine 线程 | `tid=131`（自动选 `prepare input` 行最多的 tid） |
| engine step 数 | 66（= 66 个 `prepare input` = 66 个 `Step:Model` **dispatch** 行） |
| phase 波段 | `prefill`(102.8 ms) → `decode`(436.1 ms) → `idle`(19.2 ms) |
| 稳态（剔除前 2 步） | `decode` 63 步、`idle` 1 步；**prefill 只有 2 步，且都在 warmup 里** |

> 注意：该切片是 1 个请求的 prefill+64 token decode，所以 **`decode` 波段里
> 只有 1 个请求在跑**（B=1）。它验证的是工具正确性，不是并发结论。

---

## 2. ⚠️ 关键发现一：async 模式下每个 engine-core 迭代有**两个** `Step:Model`

切片里 `Step:Model` 有 **132 行**，而 `prepare input` 只有 **66 行**。原因在
镜像里的 `vllm/v1/engine/core.py`（`step_with_batch_queue()`，async 路径）：

```python
with record_function_or_nullcontext("Step:Schedule"):      # 1) schedule
    scheduler_output = self.scheduler.schedule(...)
with record_function_or_nullcontext("Step:Model"):         # 2) dispatch（内含 prepare input/forward/post process）
    exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
...
with record_function_or_nullcontext("Step:Model"):         # 3) wait：batch_queue.pop() + future.result()
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    model_output = future.result()
with record_function_or_nullcontext("Step:Output"):        # 4) update_from_output
    engine_core_outputs = self.scheduler.update_from_output(...)
```

实测（tid=131）：66 行 dispatch（p50 = **5569.7 µs**，含 `prepare input`/
`forward`/`post process`）+ 66 行 wait（p50 = **36.4 µs**，范围 9.8–945 µs，无子 scope）。

**因此 `analyze_lite.py` 的 step 定义做了"主锚点"过滤**：只把
*包含至少一个 `prepare input` 行*的 `Step:Model` 当作一个 step，另外 66 行在
warning 里报出来并（按契约的 orphan 规则）折进最近的前一步。不做这个过滤，
`n_steps` 会翻倍、`step_dur` 的 p50 会掉到 ~35 µs（66 个 5.5 ms + 66 个
35 µs 混在一起），下游算占比直接错。

> 附带发现（供父任务判断口径）：**该 `wait`/`Step:Output`/`sample_token`/
> `draft_token`/`async_state_update` 属于"上一拍的输出处理"**（流水线错位），
> 它们的时间戳落在本拍 dispatch 窗口之外的间隙里，见 §3。

---

## 3. ⚠️ 关键发现二：契约窗口 `[start, start+dur)` 在 async 下必然产生大量 orphan

契约规定：窗口 = `[start_us, start_us+dur_us)`，窗口外的行折算到"最近的前一步"
并计入 `orphans`。在 async 切片上，两个 dispatch 之间的间隙约 **1.2 ms**，
里面正好装着下一轮的 schedule 和上一轮的输出处理，于是：

| 指标 | 值 | 说明 |
|---|---|---|
| `orphans` | **652**（= tid=131 的 912 条非 phase 行的 71.5%） | 全部落在 66 个窗口之间/之后 |
| orphan 构成 | `Step:Schedule` 66、`schedule:*` 261、`Step:Model`(wait) 66、`Step:Output` 66、`sample_token` 64、`draft_token` 64、`async_state_update` 64、`Input:Process` 1 | 即"非模型执行段"整体 |
| 影响 | 62/66 步的 measured scope 之和 > 该步 step scope | `coverage.ratio_of_step_dur` = **105.9%**（decode），`steps.csv` 的 `other_us` 因此被 clamp 到 0 |

这不是工具错，而是"`Step:Model` 只覆盖模型执行段、不覆盖整轮循环"的直接后果。
两种用法都对，取决于你要分母是什么：

* 要 `T_prepare / T_model_phase` → 默认（`--step-scope Step:Model`），`other_us` 无意义；
* 要 `T_prepare / T_iteration` → `--step-scope Step:Schedule --window-mode next`，
  此时 `orphans` 只剩 **7** 行、`other_us` 恢复可解释、coverage = 83.9%。

---

## 4. 与 `scripts/parse_phase_timing.py` 的逐项对比（同一个 smoke 切片）

命令（两侧相同输入）：

```bash
python3 scripts/measure/analyze_lite.py data/measure/_analyzer_selftest/lite.log \
        --csv data/measure/_analyzer_selftest/steps.csv \
        --json data/measure/_analyzer_selftest/summary.json \
        --txt data/measure/_analyzer_selftest/summary.txt
```

### 4.1 `scope_inventory`（**全 66 步**、含 warmup，逐项对拍）

| scope | 旧 `mean_us` | 新 `mean_us` | 偏差 | 旧 `p50_us` | 新 `p50_us` | 偏差 |
|---|---|---|---|---|---|---|
| **prepare input** | **3967.576** | **3967.576** | **0.000%** | **4088.565** | **4088.565** | **0.000%** |
| Step:Model | 6876.200 | 6876.200 | −0.000% | 5609.575 | 5605.670 | −0.070% |
| forward | 2500.855 | 2500.855 | 0.000% | 1099.430 | 1099.430 | 0.000% |
| Step:Schedule | 334.914 | 334.914 | 0.000% | 323.090 | 323.090 | 0.000% |
| post process | 166.312 | 166.312 | 0.000% | 117.405 | 117.405 | 0.000% |
| sample_token | 153.595 | 153.595 | 0.000% | 110.520 | 110.520 | 0.000% |
| async_state_update | 81.347 | 81.347 | 0.000% | 48.475 | 48.475 | 0.000% |
| Step:Output | 40.854 | 40.854 | −0.000% | 39.050 | 39.050 | 0.000% |
| schedule: allocate_slots | 39.805 | 39.805 | 0.000% | 39.160 | 39.160 | 0.000% |
| schedule: make_cached_request_data | 11.180 | 11.180 | −0.004% | 10.700 | 10.700 | 0.000% |
| schedule: update_after_schedule | 7.401 | 7.401 | +0.003% | 7.170 | 7.170 | 0.000% |
| draft_token | 2.684 | 2.684 | +0.015% | 2.555 | 2.555 | 0.000% |
| schedule: get_num_common_prefix_blocks | 2.348 | 2.348 | −0.007% | 2.270 | 2.270 | 0.000% |
| Input:Process | 28.790 | 28.790 | 0.000% | 28.790 | 28.790 | 0.000% |

**验收要求（`prepare input` 均值/p50 与旧解析器 ±1%）已满足，实际是 0.000%。**
稳态值也同样对得上：旧文本报告 `prepare input` 稳态均值 **3874.8 µs**，
新的 `by_phase.all["prepare input"].mean_us` = **3874.8 µs**，
`by_phase.decode` = **3932.8 µs**（旧文本的 decode 列相同）。

### 4.2 其它字段

| 字段 | 旧解析器 | 新解析器 | 说明 |
|---|---|---|---|
| `rows` / malformed | 982 / 0 | 982 / 0 | 一致 |
| 选中 tid | 131 | 131 | 一致（都按 `prepare input` 行数） |
| `n_steps` | 66 | 66 | 一致（旧工具退回 `prepare input` 锚点） |
| 波段步数 | decode 63 / idle 1 | decode 63 / idle 1 | 一致（prefill 2 步都在 warmup） |
| `tids` | `{1:0, 1131:0, 131:66}`（**该字典统计的是各 tid 的 `prepare input` 行数**） | `{1:66, 131:915, 1131:1}`（**schema 要求：各 tid 的总行数**） | 口径不同，非冲突 |
| `orphans` | 6（只统计"第一个窗口之前"的行，窗口后的行不计数） | **652**（契约定义：所有落在窗口外的行） | 见 §3 |
| `prepare/step` | 均值比 **100.4%**（= 期望 1.004） | decode **p50 73.6% / mean 72.5%** | 见 4.3 |

### 4.3 为什么旧的"占比 100.4%"不能用

旧解析器在 `Step:Model` 行数(132)≠ `prepare input` 行数(66) 时退回
`prepare input` 锚点，窗口是 `[prep_i.start, prep_{i+1}.start)`。而每一拍的
dispatch `Step:Model` 行**开始时间比同一拍的 `prepare input` 早 25–56 µs
（p50 27 µs）**，
于是被算进**上一个**窗口；窗口内于是只留下"下一拍的 dispatch + 本拍的 wait"。
实测（旧 `phase_timing.csv`）：step 0 的 `Step:Model` = 107717.72 µs
= dispatch₀(102242.91) + dispatch₁(5474.81)，是两次 dispatch 之和；
而 step 63 的 `Step:Model` = 397.25 µs（几乎是纯 wait），其
`prepare/Step:Model` 高达 ~1000%。64 个稳态比值里有若干这种分母塌陷的点，
把"均值的比值"抬到 100.4%（p50 其实是 72.5%）。

新解析器的 `step_dur` 直接取**同一步的 dispatch `Step:Model`**，所以
`prepare/step_dur` 的 p50 = 73.6%、mean = 72.5%，与 CSV 里逐行看到的
`6832/9280` 一类数值一致，不再有配对错位。

> 给父任务的结论：**旧工具这一列（`prepare_input_share_of_step_model`）不要
> 再引用**；需要占比请用新 `summary.json → share.*`。

---

## 5. 两个分母怎么选（写文档时必须二选一并声明）

同一个切片，同一份 `prepare input` 数据，只换分母：

| 分母 | 命令 | `step_dur` p50 | `share` decode (p50 / mean) | coverage |
|---|---|---|---|---|
| 模型执行段 `Step:Model`（默认） | 默认 | **5569.7 µs** | **73.6% / 72.5%** | 105.9% |
| 整轮 engine loop（`Step:Schedule` 间隔） | `--step-scope Step:Schedule --window-mode next` | **6991.0 µs** | **58.5% / 56.9%** | 83.9% |

两者相差 ~15 个百分点，**都属于契约 §1.1 的 `share_step`，但分母不同**：

* 默认分母 = "`prepare input` 占**模型相位**的比例"，回答"CPU 在模型执行段里占多少"；
* 整轮分母 = "`prepare input` 占**engine 单步 wall** 的比例"，才是 §1.1 里
  `T_step` 的语义（也是对外讲"每步 6.99 ms 里 3.9 ms 花在 CPU"时该用的数）。

建议：文档正文用**整轮分母**，图表/表格同时给出 `step_dur`（模型段）以便对齐
`forward`。`point_run.py → point.json` 目前记的是默认口径（`Step:Model`），
父任务若要整轮口径，可在 `maybe_analyze()` 里追加一次
`--step-scope Step:Schedule --window-mode next`（成本：再解析一次 50 KB 日志，<0.1 s）。

---

## 6. `plot_measure.py`：约定 + 自测结果

### 6.1 约定（写进每份数据目录的 `README.md`）

* 指标来自 `point.json → phases.summary`（`pi-lite-summary-v1`）；
  **默认 `decode` 波段**，decode 的 `prepare input` 步数 < 5（prefill-only 点，
  如 B/C 组）自动退化为 `all`；每个点实际用的波段写在数据目录 README 的表里。
* `prepare_*` / `forward_*` / `step_p50_us` 取该波段 `p50_us`；
  `step_p50_us = by_phase[phase].step_dur.p50_us`。
* `prepare_share_p50` = `share[phase].prepare_over_step_p50`，**单位 %**
  （矩阵表里也乘 100）。
* `ttft_ms` / `itl_ms` / `tps` = 各 round `ttft_s_mean` / `mean_itl_ms` /
  `output_tps_aggregate` 的算术平均。
* `on_cpu_ratio` = `point.json → cpu.on_cpu_ratio`（engine 主线程 on-CPU 占比）。
* 标签**一律英文**：容器/开发机 matplotlib 无 CJK 字体，中文会变方框；
  配色用 Okabe-Ito 蓝/橙/灰，二态柱状图再加斜线 hatch（灰度/色盲可辨）。
* 组选择（避免把不同配置的点画进一条曲线）：
  - 并发图：`group ∈ {A,G,I,J}` 的点（≥2 个不同并发）；
  - `isl-scan`：`group == B` 的点（否则全部点）；
  - `chunked-prefill`：`group == C` 的点（否则全部点）；
  - `mtp-onoff`：优先 `group == E`；否则用"除 MTP 外配置相同"的配对；
    否则全部带 MTP 状态的点。图上左上角标注实际用的子集。
* 每条曲线在图例里带点数（`(n pts)`）。

### 6.2 自测（`data/measure/_analyzer_selftest/`）

`make_fake_points.py` 合成 **20 条记录**（A 组 9 点、B 组 4 点、C 组 3 点、
E 组 2 点、1 条无 `phases.summary`、1 条外来 schema），跑：

```bash
python3 data/measure/_analyzer_selftest/make_fake_points.py
python3 scripts/measure/plot_measure.py \
  --points-json data/measure/_analyzer_selftest/points.json \
  --outdir data/measure/_analyzer_selftest/figures \
  --data-outdir data/measure/_analyzer_selftest \
  --prefix 03 --title-suffix " (synthetic self-test)"
```

结果：**6/6 图 + `matrix.csv` + `README.md` 全部生成**，20 条读入 / 19 条可用，
两条坏记录按预期分别"保留行但性能列留空"与"整条跳过 + warning"：

| 产物 | 断言 |
|---|---|
| `figures/03-prepare-vs-concurrency.svg` | prepare p50 曲线 + 右轴 step_dur 曲线，各标 9 pts |
| `figures/03-prepare-share-vs-concurrency.svg` | p50 与 mean-of-ratios 两条，y 轴 0–70% |
| `figures/03-prepare-vs-device-cross.svg` | prepare/forward 两曲线 + **交点标注 B≈19.2**（合成数据） |
| `figures/03-isl-scan.svg` | 上：µs/step（4 个 B 组点）；下：µs/token = µs/`min(ISL, chunk)` |
| `figures/03-chunked-prefill.svg` | x=chunk，3 个 C 组点；曲线平（≤5%）时自动标注"flat" |
| `figures/03-mtp-onoff.svg` | 分组柱状（MTP off/on × prepare/forward/step_dur），标注子集与 n |
| `matrix.csv` | 固定 16 列，19 行（坏点行性能列为空，`tag=r-no-analyzer`） |

同目录另存了 PNG（`rsvg-convert` 渲染）供快速肉眼检查：6 张 SVG 与 6 张 PNG 同名。

---

## 7. 环境与可执行性

| 项 | 结论 |
|---|---|
| a3-22 系统 python3 | 3.11.6（`/usr/bin/python3`）；`analyze_lite.py` 在 a3-22 上直接跑通，rc=0、无 stderr、无需 numpy/pandas |
| 一致性 | a3-22(3.11.6) 与开发机 conda(3.12.10) 输出的 `summary.json` **逐字段完全相同**（仅 `lite_log` 路径不同） |
| a3-22 matplotlib | **没有** → `plot_measure.py` 必须在开发机/容器（matplotlib 3.11.2）里跑 |
| 只读交叉验证副本 | a3-22 上仅写了 `scratch/lite_analyzer/`（脚本副本 + 解析产物），未动 `runs/`、未碰 chip3、未拿锁 |

---

## 8. 遗留风险 / 局限（父任务写文档时请照抄声明）

1. **只看单线程**：summary 只统计选中 tid（engine core 主线程）的行；
   `acl_thread` / `release_thread` / ZMQ 等线程的 CPU 时间不在其中，
   与 `cpu.on_cpu_ratio`（同样只跟 engine 主线程）口径一致，但都不能代表
   "整机 CPU 负载"。要全机口径必须用 `perf stat -a` / libkperf PMU。
2. **`other_us` 在 async 默认口径下不可用**（62/66 步被 clamp）；
   要步内分解请用 §5 的整轮口径。
3. **`schedule_us` 的两种来源**：默认取 `Step:Schedule` 包装 scope；
   若镜像只有 `schedule: *` 子 scope，则退化为它们的和并给出 warning，
   **两者不可直接比较**。
4. **phase 只有 3 个波段**（prefill/decode/idle）：本切片 prefill 仅 2 步且都在
   warmup，故 `by_phase` 里没有 `prefill` 块（`steps_by_phase` 仍给 0）。
   B/C 组 prefill-only 点会走 `all` 口径，父任务统计时注意区分。
5. **step_dur 的定义依赖插桩镜像**：`Step:Model`/`Step:Schedule` 出现在本项目的
   插桩镜像里；若换成 stock 镜像，需要 `--step-scope "prepare input"`（此时
   `step_dur` 即 `prepare input` 自身，占比恒为 100%，只能看绝对耗时）。
6. **合成自测数据不是测量值**：`make_fake_points.py` 的斜率是为了让"交点标注"
   等代码路径被真正执行，**不得**被引用为任何结论。
7. `plot_measure.py` 的 `mtp-onoff` 需要至少两个 MTP 状态；
   若 E 组只跑了一个状态，该图会跳过并在数据目录 `README.md` 里说明原因。

---

## 9. 校验和（sha256）

见交付回报；生成命令：

```bash
sha256sum scripts/measure/analyze_lite.py scripts/measure/plot_measure.py \
          agents/measurement_profiling/ANALYZER_NOTES.md \
          data/measure/_analyzer_selftest/{steps.csv,summary.json,summary.txt, \
          summary_itersched.json,matrix.csv,README.md,points.json,make_fake_points.py} \
          data/measure/_analyzer_selftest/figures/*.svg
```
