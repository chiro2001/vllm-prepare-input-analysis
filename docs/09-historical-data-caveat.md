# 历史数据口径警告：pystack 采样器污染（重要）

> 建立：2026-09-24。**在引用 `HIST_PROJECT` 项目的任何 phase 耗时数字前必须读本文。**

## 1. 事实链

1. `docs/03-share-in-inference.md` 里的"prepare input 占 48–59%"来自历史项目
   `a3-22:~/projects/vllm/HIST_PROJECT`，其 phase 数据由 **LiteProfiler** 采集。
2. LiteProfiler 的激活方式是 `POST /start_profile`（见
   `HIST_PROJECT/scripts/measure_existing_services.py:60`：
   "POST /start_profile before each window and /stop_profile after"）。
3. 在 LiteProfiler 镜像的 patch（`~/liteprofiler-vllm.patch`）里，
   `EngineCore.profile()` 的实现**无条件**调用
   `from vllm.utils.python_stack_sampler import start_stack_sampler`（见该 patch
   中 `vllm/v1/engine/core.py` 的 hunk），失败才静默忽略。
4. `python_stack_sampler` 的默认参数：`VLLM_PYSTACK_INTERVAL_US=1000`（1 ms），
   采样**进程内所有 Python 线程**，每次采样抓取并格式化调用栈后写 CSV。
5. `service_bringup` 在 a3-22 上的实测（0.8B TP1 graph decode，decode 稳态）：

   | 配置 | ITL | 输出 tok/s |
   |---|---|---|
   | pystack 1 ms 开启（`/start_profile` 默认行为） | 6.84 ms | 117.0 |
   | pystack 关闭 | **4.27 ms** | **185.1** |

   ⇒ 单开采样器就让**单步时间 +60%**。

## 2. 结论与影响

- 历史 phase 表里的 `prepare input = 3.13 ms`（0.8B TP1 graph decode）**很可能包含
  采样器贡献**，不能直接当作"vLLM 自身的 prepare_input CPU 成本"。
- 历史数字仍可用于**跨配置相对比较**（同一污染对所有臂近似一致），但
  **不能**用于回答"prepare_input 占端到端多少"这类绝对值问题。
- 因此本项目所有新实验都必须**显式声明 pystack 状态**，并至少给出
  `pystack=off` 的一组数据作为"真值"。

## 3. 新实验的强制口径

| 要求 | 做法 |
|---|---|
| 显式关闭 | 启动时设 `VLLM_PYSTACK_INTERVAL_US=0`（或 launcher 的 `--pystack-interval-us 0`），并把 `VLLM_PYSTACK_LOG` 指到 run 目录 |
| 记录状态 | `run_manifest.json` 里记录最终生效的 interval 与实际是否产生 pystack.csv |
| A/B 对照 | 同一 workload 至少跑 `pystack=off` 与 `pystack=on` 各一次，差值即污染量 |
| 引用历史 | 凡引用历史数字，必须注明"pystack 状态未受控（默认开启）" |

## 4. 待补证据

- `subscope_instrumentation` 的 D 臂（`PI_SUBSCOPE=on` + pystack 默认值）会给
  "pystack 对 prepare_input **本身**（而非 ITL）的污染量"。
- 若 D 臂显示污染量与 service_bringup 观察到的 ITL 差一致，则历史 prepare_input
  数字需要整体下修；下修幅度写进本文 §5。

## 5. 修正后的数值（D1/D2 实测回填，2026-09-24）

数据源：`subscope_instrumentation` 的 D1/D2 臂（`runs/sub-s1-mounted-off-pystack-*`、
`runs/sub-s2-mounted-off-pystack-*`），同 workload、同 chip3、同 CPU 切片、
唯一变量是 `VLLM_PYSTACK_INTERVAL_US`（1000 = 上游默认 / 0 = 关闭）。
step 分母口径：`analyze_lite.py --step-scope Step:Schedule --window-mode next`
（理由见 §5.3）。汇总见 `data/subscope/ladder-s1.json` / `ladder-s2.json`。

### 5.1 pystack 对 `prepare input` 本身的污染

| 场景 | D2（pystack off） | D1（pystack 1 ms） | **污染量** | 污染比例 |
|---|---:|---:|---:|---:|
| B=1 ISL=128 decode，prepare_input | 2909.5 µs | 3484.8 µs | **+575.3 µs** | **+19.8%** |
| B=1 ISL=128 decode，step | 5089.0 µs | 6187.0 µs | +1098.0 µs | +21.6% |
| B=64 ISL=128 decode，prepare_input | 3960.6 µs | 4806.6 µs | **+846.0 µs** | **+21.4%** |
| B=64 ISL=128 decode，step | 10255.0 µs | 12489.0 µs | +2234.0 µs | +21.8% |

**结论：pystack 对 `prepare input` 的污染是 +20~21%（两个场景一致）**，
而 `service_bringup` 观察到的是 ITL +60%（4.27 → 6.84 ms）。
⇒ **采样器的主要伤害不落在 `prepare_input` 内部**，而是在它之外
（每步 `sys._current_frames()` + 栈格式化 + CSV 落盘 + 与主线程抢 GIL，
主要集中在 forward/sample 的 Python 段）。

### 5.2 历史数字的修正系数

历史 phase 数据是**在 pystack 默认开启**的条件下采集的。按实测污染比例回推：

| 配置（历史口径） | 历史记录（含污染） | 推定无污染真值 | 修正系数 | 说明 |
|---|---:|---:|---:|---|
| 0.8B TP1 graph decode | prepare **3.132 ms** / step 5.661 ms / 55.3% | prepare ≈ **2.61 ms** / step ≈ 4.65 ms | **×0.834** | 以 prepare 的 +19.8% 回转 |
| 0.8B TP1 eager decode | prepare **2.843 ms** / step 42.459 ms / 6.7% | prepare ≈ **2.37 ms** / step ≈ 34.9 ms | **×0.834** | eager 的 step 由 device 主导，比例另算 |

> **注意**：上表两行是**外推**（历史那一轮没有同条件的 pystack-off 对照，
> 而本轮的 workload/模型配置也不完全相同：历史用 ISL=128/OSL=1024/并发 1，
> 本轮用 ISL=128/OSL=64/并发 1 或 64）。所以给出的是**量级修正**，不是精确重测。
> 若需要精确数字，必须用历史那份 workload（`HIST_PROJECT` 的 wave1-v2
> 配置）重跑一对 D1/D2。
>
> **本节同批次的直接实测（同条件对照，可直接引用）**：
> B=1 decode 下 `prepare_input` 真值 ≈ **2.91 ms**（D2），含污染值 3.48 ms（D1）；
> 这与历史记录的 3.132 ms 处于同一量级，且**新的真值比历史值低约 7%**。

### 5.3 另一个必须记住的分母口径

async scheduling 下每个 engine 迭代会写**两行 `Step:Model`**（dispatch +
batch_queue wait），因此以 `Step:Model` 为分母会让 `n_steps` 翻倍、
step p50 掉到 ~35 µs（历史报告里 `prepare_input_share_of_step_model = 100.4%`
即由此而来）。**正确做法**：用
`analyze_lite.py LOG --step-scope Step:Schedule --window-mode next`，
或直接数 `prepare input` 行（每步一行）。
`scripts/extract_ladder.py` 已把该口径固化，本文所有 step 数字均用它计算。

### 5.4 引用历史数字的强制标注

凡引用 `HIST_PROJECT` 的 phase 耗时，必须写明：

> 「pystack 状态未受控（`/start_profile` 默认启动 1 ms 采样器）；
> 实测其对 `prepare_input` 的污染约 +20%，对 ITL 的污染约 +60%。」
