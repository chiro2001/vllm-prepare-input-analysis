# drivers_analysis 报告（prepare_input 瓶颈判据与成本模型）

> 2026-09-24 / 状态：**完成**（本文档即交付报告；实测数据由后续 agent 回填 ⛳ 位置）

## 1. 任务与范围

把"prepare_input 什么时候成为瓶颈"建成**可计算、可证伪**的模型，并设计验证它的实验清单。
硬约束遵守情况：

- ✅ 未跑任何 NPU 实验、未起服务、未碰 chip0/2/4-15、未动 120-159 核。
- ✅ 全部 CPU 微基准在 a3-22 的 **核 200–203**（NUMA node 2）执行，`taskset` 绑核、单线程。
- ✅ 环境隔离：在项目目录内建独立 conda 环境 `~/.conda/drivers`（Python 3.11 + numpy 2.4.6 + torch 2.9.0+cpu），
  未污染 `pinput` 环境与镜像。
- ✅ 未产生 `perf.data`（只用 `perf stat` 计数，输出在 `ipc-summary.jsonl` 与报告表格中）。

## 2. 交付物

| 产物 | 说明 |
|---|---|
| `docs/07-bottleneck-analysis.md` | **主交付**：流水结构、一阶/二阶成本模型、判据、放大/掩盖机制、反事实估算（含实测）、replay 判定、优化假设、未决问题（635+ 行） |
| `docs/03-share-in-inference.md` | "判据与公式"章节定稿（含 ⛳ 待回填的曲线与拟合参数表） |
| `data/model/complexity-model.json` | 机器可读模型：9 个成本项 × 依赖/阶/系数/来源/状态 + 9 个二阶项 + 6 掩盖/11 放大机制 + 4 条预测 + replay 规格 |
| `plan/experiment-matrix-v2.md` | 实验矩阵 v2：保留 v1 六组，**新增 6 组**（G 异步开关、H churn、I TP、J 缓冲宽度、K 系数标定、L replica↔真机对齐），每点含目的/自变量/固定量/观测/预期判别 |
| `scripts/microbench_prepare_input.py` | numpy/python/ATen 原语 + `_prepare_inputs` CPU 序列 replica（按代码顺序复刻），可跑单因素与规模扫描 |
| `scripts/microbench_numa_ipc.py` | NUMA 局部性探针 + 稳定 IPC 负载窗口（供 perf/libkperfx 外部包裹） |
| `data/model/microbench-*.csv`、`ipc-summary.jsonl`、`microbench-meta.json` | 原始数据（8 个文件） |

## 3. 关键结论（按重要性）

### 3.1 流水结构决定"掩盖"是否可能（新发现，v1 文档未覆盖）

1. **异步调度在 0.26.0 默认开**（`config/vllm.py:1059–1107`；UniProc/Multiproc 两个 executor 都
   `supports_async_scheduling()=True`；vllm-ascend 不覆盖 executor）⇒ `max_concurrent_batches=2`
   ⇒ engine 走 `step_with_batch_queue`：**CPU 与 device 可以重叠**。
2. worker 侧 **D2H 被搬到独立线程**（`multiproc_executor.py:956–988` + `AsyncGPUModelRunnerOutput`），
   主线程稳态不等 device ⇒ `T_step ≥ max(T_cpu, T_dev)`。
3. `prepare_inputs_event`（`gpu_model_runner.py:3808–3822`，ascend record 点 `1883`）把
   **CPU 领先量限制在约 1 拍 device 时间** ⇒ "CPU 只有一拍预算"是可操作判据。
4. 结论：**prepare_input 与 engine 侧调度/RPC/launch 共享同一份 CPU 预算**，
   单看 prepare_input 占比会高估它的收益上限。

### 3.2 判据（可拟合的四参数模型）

```
T_prepare ≈ c_p + a·B ;  T_dev ≈ e + d·B ;  T_cpu_other ≈ c_o
CPU 受限 ⟺ B > (e − c_p − c_o)/(a − d)     （仅当 a > d）
另有"底噪受限"形态：c_p + c_o > e 时任何 B 都 CPU 受限。
chunked：c_p + a_T·T = e' + β·T + γ·T(S+T/2) → T_min
TP：T_step = max_r T_cpu,r + T_dev；E[max] ≈ μ + σ√(2 ln R)
```

### 3.3 无卡微基准的实测系数（核 200–203，单线程）

| 量 | 实测值 | 文件 |
|---|---|---|
| `c_p`（CPU-only，B=T=1） | **47.25 µs（拟合截距）/ 48.7 µs（实测点）** | `microbench-composite.csv` |
| `a`（CPU-only，decode） | **0.2721 µs/req（R²=0.99937，n=6）** | 同上 |
| MTP n=2 | 截距 50.42 µs、斜率 **0.406 µs/req** | 同上 |
| `a_T`（CPU-only，chunked，整段斜率） | **7.8–8.8 ns/token**（含防越界的 `%`，去后 ≈6–7） | 同上 |
| MTP n=2 额外成本 | ≈0.16 µs/req + 5 次固定 H2D | 同上 |
| numpy 调用固定开销 | 0.42–3.57 µs/次（`np.all` 3.57、`np.cumsum` 2.63） | `microbench-numpy.csv` |
| ATen 调用固定开销 | **1.61–2.30 µs/次** | `microbench-torch.csv` |
| Python 每请求成本 | `num_tokens` list-comp **128 ns/req**；`dict.get` 循环 39 ns/req | `microbench-python.csv` |

**P1 得到支持**：CPU-only 斜率 0.272 µs/req（R²=0.9994）落在预测区间 [0.20, 0.60] 内（含 H2D 后预计 0.3–0.6）。

> 口径提醒：`a_T` 报的是**整段**斜率（每 token 经过 6–8 个 O(T) 算子），
> 不要与 §3.2 的**单算子**系数（0.13–1.29 ns/el）混用。

### 3.4 合成负载的 CPU 特征（这是"无卡负载指纹"，供保真度对照）

| 工况 | µs/step | IPC | 分支失败率 | cache miss |
|---|---:|---:|---:|---:|
| B=1, T=1 | 48.0 | 1.57 | 3.73 % | 2.33 % |
| B=64, T=64 | 64.3 | 1.95 | 2.51 % | 2.21 % |
| B=256, T=256 | 118.9 | 2.39 | 1.51 % | 1.95 % |
| B=8, T=32768 | 322.4 | 2.93 | 0.82 % | 2.07 % |

**解读**：小 batch 区是**解释器主导**（低 IPC、高分支失败），大 token 区是**内存/向量主导**。
真机若 IPC 明显低于该趋势（<1.2），说明时间花在**驱动/系统调用路径**（H2D 入队、event wait）。

### 3.5 NUMA

512 MB 工作集、CPU 固定 node 2：跨 socket（mem node 5）相对本地 **1.80–1.99×** 惩罚；
同 socket 远端 1.05–1.09×。但 prepare_input 的热数据（≈8 MB 的 `token_ids_cpu` + 块表）
多数落 L3，**真实惩罚应远小于 2×**，需在 replay 上用 `--membind` A/B 确认（§5 A8）。
（小工作集的 `microbench-numa.csv` 差异不可辨，**不要用**；只有 `microbench-numa-large.csv` 有效。）

### 3.6 环境残留（给打包/清理）

- 无卡微基准环境：`<远程项目>/.conda/drivers`（**1.1 GB**，Python 3.11 + numpy/pandas/matplotlib + torch 2.9.0+cpu）。
  打了标签仍在项目目录内；**打包交付时应排除 `.conda/`**，如需复现按 §6 的命令重建即可。
- 进程：所有微基准为前台运行、已退出；`pip` 安装进程已结束（pypi.org 那次卡死的进程已 `pkill`）。

### 3.6 是否需要 record & replay：**需要**（结论明确）

9 处控制流由数据内容决定，其中 6 处依赖请求级 token/length 语义；
第 2 条（`_build_attn_state` 5 分类）改变 metadata 构造路径与后续 device 时间，
第 4 条（S17 同步）会把 device 时间**吸收进** prepare_input 的 wall，
第 8 条（`condense`）是 O(B) 与 O(B·S) 的量级差。
**粒度分层**：L1 `SchedulerOutput` 语义字段 + L2 `InputBatch` 粘性数组 + L3 flags 为必需，
L4 计时可选；存储量估数十 MB~数百 MB（远小于 perf.data，可进交付包）。

## 4. 给其他 agent 的交接（按角色）

### 给 measurement（真机占比 + profiling）

1. **必须同时采 `prepare input` 与 `forward` 的 phase wall** —— 只有占比曲线无法拟合判据。
2. 按 `plan/experiment-matrix-v2.md` 的 P0 顺序：A（9 点）→ G（async on/off）→ K1/K3 → C（5 点）→ L。
3. 统计前剔除：前 100 step、空 batch step、`_sync_device()` 校准窗口。
4. 回填位置：`docs/07-bottleneck-analysis.md` §6.1b（表 6-1b）、§6.2（表 6-2）、
   `docs/03-share-in-inference.md` §7/§8；并把 `complexity-model.json` 的 `status` 从 `pending` 改 `measured`。
5. 关键判读规则（§2.6）：wall ≫ perf on-CPU（>3×）⇒ 先怀疑 S17 同步而不是 CPU 计算。

### 给 replay harness（无卡复刻）

1. `scripts/microbench_prepare_input.py::PrepareInputReplica` 可作为骨架：
   它已按代码顺序复刻 S2/S4/S5/S7/S9/S11/S13/S16/S19/S23 的 CPU 部分，**并且是已知偏差的来源清单**（§6.4 D1–D6）。
2. 打桩三处边界：`.gpu/copy_to_gpu/pin_memory().to()`（计数 + 用 `k_h2d_small` 折算回时间）、
   device op（只计数）、`torch.npu.Event/Stream`。
3. 复刻件当前**缺**的是：真 H2D DMA、device op、churn 突变路径（`condense`/`swap_states`）、
   逐 group×layer 的 metadata 赋值 —— 这些必须在 replay 里用真实实现补上，否则 §7.4 的 top-20 判据会不达标。
4. 保真度对照点建议直接用组 L（B∈{1,64,256}、chunk∈{2048,8192}、MTP on/off）。

## 5. 未决与风险（需要下游决策）

1. `T_sched + T_rpc` 的量级未知（`[待实测]`）；若 >100 µs，会显著压缩 CPU 预算 `B_cpu`。
2. Ascend 上 `torch.npu.Event` 语义是否与 CUDA 一致（影响 §2.3 的"1 拍领先量"结论）。
3. 本文 CPU 数据来自 a3-22 的 **conda CPU-only torch**；容器内为 torch_npu 构建，
   `pin_memory()`/`to()` 的成本必须重测（`microbench-meta.json` 记录了本文环境）。
4. `perf stat` 只覆盖用户态（`perf_event_paranoid=2`，`:u` 后缀），
   因此"驱动/内核侧"占比在本文中只能由差值推断，真机需 root + libkperfx 才能定量。

## 6. 复现命令

```bash
# 环境（项目目录内，不污染他人）
ssh a3-22 '~/miniforge3/bin/conda create -y -p <proj>/.conda/drivers python=3.11 && \
  <proj>/.conda/drivers/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy pandas matplotlib torch==2.9.0'

# 微基准（只用核 200–203）
cd <proj> && OMP_NUM_THREADS=1 taskset -c 200-203 .conda/drivers/bin/python \
  scripts/microbench_prepare_input.py --only all --out-dir data/model

# IPC（外部 perf 包裹）
for cfg in "1 1" "64 64" "256 256" "8 32768"; do set -- $cfg; \
  perf stat -e cycles,instructions,task-clock,branches,branch-misses \
  taskset -c 200-203 .conda/drivers/bin/python scripts/microbench_numa_ipc.py ipc \
  --seconds 6 --batch $1 --tokens $2 --label "B$1-T$2" --out data/model/ipc-summary.jsonl; done

# NUMA（CPU 固定 node2，内存跨 node）
for mem in 2 3 5; do numactl --physcpubind=200-203 --membind=$mem \
  .conda/drivers/bin/python scripts/microbench_numa_ipc.py numa \
  --label "cpu2-mem$mem-src512MB" --tokens 1048576 --src-mult 64 \
  --out data/model/microbench-numa-large.csv; done
```

> 注：pypi.org 在 a3-22 上极慢（实测 ~66 B/s 的停滞），**必须用清华镜像**。
