# cpython_build_exp —— 交接报告

**Agent**：`/root/cpython_build_exp`
**任务**：CPython 构建配置对解释器性能影响的**真机量化验收实验**（a3-22）
**状态**：✅ 完成（A/B/C 三臂全部建成并测完；真实负载 harness 的 stock/A/B/C 四组均已采到）
**写入范围**：本地 `/home/chiro/projects/vllm/preparing-input-phase` 与
`a3-22:/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython/`（+ `/tmp`）
**未触碰**：`/dev/davinci*`、chip3、CPU `120-159`、宿主机系统配置、他人容器、`~/projects/vllm/HIST_PROJECT/*`

---

## 0. 最重要的一条（会改变行动项）

**镜像里实际被加载的 `libpython3.12.so.1.0` 已经是 computed gotos 构建，
尽管它自己的 `pyconfig.h` / `sysconfig` 说不是。**

⇒ **"给镜像加 `--with-computed-gotos`"这项行动应当直接取消**（收益已经是 0，已在线上兑现）。
⇒ **真正该做的是 `--enable-optimizations --with-lto`**：本负载形状 **−22.6%**（C vs B），
真实 `prepare_input_us` **−15.6%**（464.7 → 392.3 µs），构建时间 27 s → 813 s，**运行期零代价、零代码改动**。
⇒ 该文（`docs/10-cpython-directions/00-runtime-build-config.md`）里"走 switch 分派 + PGO/LTO 都没开，
这是最高性价比方向"的判定，前半段**被推翻**；后半段（PGO/LTO 没开）**成立，是本实验剩下的唯一构建杠杆**，
且其实测量级**高于**该文原先假设的 10~20%。
⇒ 该文已按本实验结论**就地修正**（§9 表）。

### 三重静态证据 + 一重行为证据（互相独立）

| 证据 | 镜像 stock | arm A（`--without-computed-gotos`） | arm B（`--with-computed-gotos`） |
|---|---|---|---|
| `opcode_targets` 指针表符号（256×8 B） | **有**（0x800 @ `.data.rel.ro`） | **无** | **有**（0x800） |
| `_PyEval_EvalFrameDefault` 内 `br xN` | **253** | **1** | **253** |
| `_PyEval_EvalFrameDefault` 体积 / 指令数 | **46268 B / 11567** | 44072 B / 11018 | **46268 B / 11567** |
| 整库 `.text` 体积 | **0x24dfcc** | 0x24d30c | **0x24dfcc** |
| 分派指令形态 | `ldr x0,[x1,w8,sxtw#3]; br x0`（每 opcode 一处） | `adr/add(...,sxth#2)/br x0`（全局 1 处，4 B 偏移 Jump Table） | 同镜像 |
| 行为：分派密集核 vs 镜像 | — | **慢 7.6–11.8%** | **差 0.2–1.6%** |
| 行为：`pi_sim_bench` 混合负载（同会话交错） | **242222 ns/step** | 257573 ns/step | 235489 ns/step |

`aarch64` 上 `sxth #2`（4 字节偏移）vs `sxtw #3`（8 字节指针）是**判据本身**：
`switch` 生成偏移跳转表，computed gotos 生成指针表。

---

## 1. 交付物

| 路径 | 内容 |
|---|---|
| `docs/10-cpython-directions/06-build-config-experiment.md` | **主文档**：结论优先；三臂定义 / 静态与行为证据 / 微基准 / PMU 归因 / 真实负载 / IPC 0.771 判定 / 验收判据 / 复现清单 |
| `docs/10-cpython-directions/00-runtime-build-config.md` | **已按本实验结论修正**（§9 表逐条对照） |
| `agents/cpython_build_exp/REPORT.md` | 本文件 |
| `data/cpython-build-exp/scripts/` | 全部脚本（见 §3） |
| `data/cpython-build-exp/data/` | 三臂 `sysconfig` 快照 / 基准原始 JSON / perf 解析 CSV / 汇总 |
| `data/cpython-build-exp/logs/` | configure / make 全量日志、`*.meta.txt` |
| `a3-22:.../exp-cpython/{build-armA,B,C,logs,data,refimg,pydeps}/` | 构建树、原始数据、**镜像原版 `libpython3.12.so.1.0` 参考副本** |

---

## 2. 构建命令与耗时（实测）

源码 `Python-3.12.13.tgz`（TUNA 镜像，`sha256=0816c476…500b0b`，与 python.org SPDX 摘要一致），
在**同一镜像**内用容器自带 gcc 12.3.1（`…-111.oe2403sp3`）构建，`make -j16`，**不做 `make install`**：

```bash
cd ~/projects/vllm/prepare-input-phase/exp-cpython
IMG=quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler
for a in A B C; do
  sudo -n docker run -d --rm --name cpbuild-$a --cpuset-cpus 200-239 -v $PWD:/work \
    --entrypoint bash $IMG -lc "WORK=/work JOBS=16 bash /work/scripts/build_arm.sh $a"
done
```

| arm | configure 参数（共同项 `--enable-shared`） | configure | make（`-j16`） | `USE_COMPUTED_GOTOS` | libpython | `.text` |
|---|---|---:|---:|---:|---:|---:|
| A | `--without-computed-gotos` | 17 s | **27 s** | 0 | 30,812,384 B | 0x24d30c |
| B | `--with-computed-gotos` | 17 s | **27 s** | 1 | 30,689,296 B | 0x24dfcc |
| C | `--enable-optimizations --with-lto --with-computed-gotos` | 44 s | **813 s（13.6 min）** | 1 | 30,209,248 B | **0x23124c** |

**结论**：A/B 两臂 30 秒可建 ⇒ 这类实验**没有理由省掉隔离臂**。

---

## 3. 工具与脚本（全部可重跑）

| 脚本 | 作用 |
|---|---|
| `build_arm.sh <A\|B\|C>` | 容器内构建一臂：configure → make → 写 `sysconfig` 快照 + `*.meta.txt` |
| `pi_sim_bench.py` | 本负载形状微基准（25 字段 dataclass / 属性链 / 短方法 / dict / 小 numpy / 对象 churn），输出 `work_digest` 证明三臂工作量相同 |
| `dispatch_micro_bench.py` | 5 个分派密集核（分派影响的**上界**对照） |
| `run_bench_arms.sh` / `run_all_benches.sh` | **逐轮交错**跑三臂（`A,B,C,A,B,C…`），`taskset -c 200-203`，`OMP_NUM_THREADS=1` |
| `run_bench_stock_image.sh` | 在**不换任何库**的镜像里跑同一基准（判定"生产到底走哪条分派"的关键对照） |
| `run_bench_mixed_stock.sh` | **把镜像与三臂放进同一会话逐轮交错**（`A,B,C,stock` × N）——镜像对照**必须**这么做，否则会话间漂移 3–5% 会毁掉结论 |
| `perf_stat_arms.sh` | `perf stat` 两组 4 事件（`cycles/instructions/branches/branch-misses`，`L1-icache/iTLB`） |
| `run_harness_arm.sh <stock\|A\|B\|C>` | 跑真实 `prepare_input` 无卡 harness；`PERF=1` 时对该容器进程采 PMU |
| `finish_after_C.sh` | 等 C 臂构建完成 → 自动接着跑全部需要 C 的测量（可 `nohup` 后台跑） |
| `run_harness_repeats.sh` / `summarize_*` | 重复 + 汇总 |
| `summarize_bench.py` | 原始 JSON → CSV + markdown 表 |

---

## 4. 核心实验数字

### 4.1 微基准（A/B/C 绝对 + 相对，7 轮交错中位数，`taskset -c 200-203`）

主口径 = **11 轮交错**（`data/bench-pi-clean/`）；括号内为 7 轮复现（`data/bench-pi/`）。

| 指标 | A（switch） | B（CG） | **C（PGO+LTO+CG）** | 镜像 stock |
|---|---:|---:|---:|---:|
| `pi_sim_bench` ns/step | 256266（263673） | 236414（251374） | **182984（192233）** | 242222 |
| 相对 A | 1.0000× | **0.9225×（−7.8%）** | **0.7140×（−28.6%）** | 0.9404× |
| **相对 B（= 镜像现状）** | 1.0840× | 1.0000× | **0.7740×（−22.6%）** | 1.0286× |
| `call_loop` ns/iter | 125.99 | 113.02 | **107.40** | 112.67 |
| `branch_loop` ns/iter | 101.02 | 90.25 | **67.04** | 89.61 |
| `int_loop` ns/iter | 112.61 | 100.89 | **86.86** | 107.64 |
| `global_loop` ns/iter | 63.31 | 58.17 | **49.66** | 58.05 |
| `attr_loop` ns/iter | 66.80 | 62.30 | **58.54** | 62.40 |

三臂 + 镜像的 `work_digest` **全部 = 77232000**（工作量相同的证明）。
以上全部来自**同一会话的逐轮交错**（`data/bench-dispatch-mixed/`、`data/bench-pi-mixed/`，7 轮中位）——
这一点很重要：本机**会话间**漂移可达 3–5%，镜像**必须**与各臂在同一会话里交错跑。
读数：**5 个分派核里 4 个，镜像与 B 臂相差 ≤ 0.7%**，同时比 A 臂快 6.6–11.3%
⇒ 镜像落在 computed gotos 一侧。

### 4.2 PMU：IPC 与 branch-misses（A vs B，同一负载/同一核）

主口径 = `pi_sim_bench --steps 4000 --rounds 5`（`data/perf-clean/`）。

| 计数 | A | B | C | B/A | C/B |
|---|---:|---:|---:|---:|---:|
| cycles | 15,153,701,127 | 14,145,882,614 | 10,874,288,079 | 0.9335 | **0.7687** |
| instructions | 35,904,059,214 | 34,062,602,480 | 29,908,817,262 | 0.9487 | **0.8781** |
| **IPC** | **2.3693** | **2.4080** | **2.7504** | **+1.6%** | **+14.2%** |
| branches | 7,849,095,216 | 7,152,758,930 | 5,826,018,326 | 0.9113 | 0.8145 |
| branch-misses | 92,413,390 | 81,140,060 | 63,921,788 | 0.8780 | **0.7878** |
| **branch-miss 率** | **1.1774%** | **1.1344%** | **1.0972%** | −0.043 pp | −0.037 pp |
| L1-icache miss 率 | 7.2390% | 8.0824% | 7.5167% | +0.84 pp（**变差**） | −0.57 pp |
| iTLB miss 率 | 1.4494% | 1.5612% | 0.7765% | +0.11 pp（**变差**） | **−0.78 pp** |

**"为什么有效"的答案有两套，且完全不同**：

* **computed gotos（A→B）**：`instructions −5.1%`（省指令），`IPC` 只 +1.6%、
  branch-miss 率只降 0.043 pp、i-cache/iTLB 失配率**反而变差**（代码更大）。
  ⇒ **不是**分支预测变好。这与"`switch` 单条间接跳转打数百目标 → BTB 冲突 → `ptag_stall`"
  的假设**不符**。
* **PGO+LTO（B→C）**：`instructions −12.2%` **且** `IPC +14.2%`，`branch-misses −21.2%`、
  `iTLB load-misses −63.6%`、`L1-icache load-misses −31.9%` ⇒ **前端压力全面下降**，
  方向与我们的 `frontend_bound 66%` 画像**一致**。**这才是应该投的那一项。**

### 4.3 真实负载（无卡 harness，`--preset realmachine --batch 1 --isl 128`）

| arm | n | `prepare_input_us` p50（中位） | `update_states_us` p50 | 相对 stock |
|---|---:|---:|---:|---:|
| stock | 5 | **464.7** | 18.7 | 1.0000× |
| A（switch） | 2 | 464.8 | 19.4 | +0.01% |
| B（CG） | 2 | 451.6 | 18.2 | −2.83% |
| **C（PGO+LTO+CG）** | **3** | **392.3** | **15.6** | **−15.59%** |

**读数**：① C 相对生产现状 **−15.6%**，且 `update_states_us` 同向 **−16.6%**（旁证，非单点噪声）；
② **A/B/stock 三者在这一判据上分辨不出来**（|Δ| ≤ 2.8%，而单轮 run-to-run 波动本身就有 ±1–2%）
⇒ **harness 的 `prepare_inputs_us` 只够验收 PGO+LTO（~15%）这一量级，验收不了 computed gotos（~5%）**。

---

## 5. 踩坑记录（重要，避免重走）

1. **`sysconfig` / `pyconfig.h` 会骗人**：它们描述**安装面**，不描述**当前这个 `.so`**。
   本任务的前提判定正是踩在这里。判分派必须看二进制（`objdump`）或行为（分派微基准）。
2. **`DT_RPATH` 优先于 `LD_LIBRARY_PATH`**：镜像的 `bin/python3.12` 带
   `DT_RPATH=/usr/local/python3.12.13/lib`（不是 RUNPATH），所以 `LD_LIBRARY_PATH` **无法**换 libpython。
   唯一可靠换法是**单文件只读 bind-mount** 覆盖 `libpython3.12.so.1.0`。
3. **docker 默认 seccomp 会挡 `perf_event_open`**：容器里 `perf stat` 报
   `No permission to enable cycles event`。解法是 `--security-opt seccomp=unconfined`。
   镜像自身没有 `perf`，把宿主的 `/usr/bin/perf` + **缺的 6 个与 Python 无关的 so**
   （`libopencsd*`、`libbabeltrace*`、`libpfm`、`libtraceevent`）单文件挂进去即可，**不要**整目录挂。
4. **`declare -A GROUPS` 会失败**：`GROUPS` 是登录环境里的标准变量（组 ID 数组），
   报 `cannot convert indexed to associative array`。改名即可。
5. **多行命令经 `ssh '...'` 会被拍平**：换行会被吞，三条命令粘成一条且静默错跑。
   可靠做法是**写脚本文件 + `tar | ssh` 分发**（本任务全程采用）。
6. **基准里 `np.empty` 会让 `work_digest` 随机**：任何"工作量相同"的证明都不能用未初始化内存。
   改用 `np.arange`。
7. **无卡 harness 的 `--out` 必须是容器内路径**（宿主树挂在 `/work`），传宿主绝对路径会报
   `PermissionError: '/home/REMOTE_USER'`，且**在算出全部结果之后**才失败，浪费一整轮。
8. **`docker update --cpuset-cpus`** 可以在不重启容器的情况下把构建移出基准核（本任务用它对
   C 臂做 `200-239 → 216-239`），比重建容器便宜。
9. **CPU 200-203 不是独占的**：hostnoise gate 实测该切片 mean 7.94% / max 31.33%，
   有 29 个他人 python 进程 affinity 覆盖它。所以**必须逐臂逐轮交错**，不能"一臂跑完再跑一臂"。
10. **ssh 下载大文件要限速/换镜像源**：python.org 直连在 a3-22 上极不稳定
    （4.4 MB 后停滞），TUNA 镜像 27 MB / 6 s。
11. **镜像对照必须与各臂在同一会话里交错跑**：本机**会话间**漂移 3–5%（A、B 两臂在不同会话里
    各自漂移），把镜像单独跑一个会话再与各臂比值，会得出"镜像比 B 慢 7%"这种假结论。
    第一次我就是这么错的，后来用 `run_bench_mixed_stock.sh`（`A,B,C,stock` 逐轮交错）纠正。
12. **别让后面一次重跑覆盖前面的原始文件**：`run_bench_stock_image.sh` 第二次调用把第一次的
    `data/bench-pi-stock/*.json` 覆盖了，导致文档里引用的数字一度无原始文件可查。
    交付前务必用 `SUMMARY.md` 里的数字对照一遍原始 JSON。

---

## 6. 完成度与未完成

| 项 | 状态 |
|---|---|
| A / B 臂构建（含 `sysconfig` 验证） | ✅ |
| C 臂构建（PGO+LTO，三阶段，813 s） | ✅ |
| 静态分派证据（三重） | ✅ |
| 行为分派证据（镜像 vs A/B，7 轮 × 6 负载） | ✅ |
| 微基准 A/B/C（11 轮主口径 + 7 轮复现 + 镜像对照） | ✅ |
| `perf stat` A/B/C（2 组 × 4 事件，5 轮窗口） | ✅ |
| 真实负载 harness（stock n=5、A n=2、B n=2、C n=3，逐次交错） | ✅ |
| `libkperfx` topdown（微基准侧） | ❌ 时间预算内未做（需 root + 9 组复用），已在主文档 §10 登记 |
| 端到端 TTFT/TPOT | ❌ 不做（无卡下无意义；需占 chip3） |
| 用 vLLM 自己的负载做 PGO 训练集 | ❌ 未做（需改构建流程）；当前 C 数字是**保守下界** |

**时间线（a3-22 本地时间，总耗时约 30 分钟）**：
11:34 起 → A/B 构建 11:40–11:41（各 ~45 s）→ C 构建 11:41–11:55（13.6 min 的 make）
→ 三臂微基准/PMU/harness 集中在 11:43–12:02（其中 11:52–12:00 与 C 构建并行）
→ 12:01 干净复核跑 → 12:03 数据同步完成。

**资源占用与回收**：`a3-22:.../exp-cpython/` 当前占 **1.4 GB**（三份源码树 + 三份构建树 +
`pydeps/` + 数据）。全部容器已 `--rm` 退出，进程已清零，chip3 未被触碰（收工时 `npu-smi` 显示
`No process in device`）。若需回收空间，可删 `build-armA/B/C/`（各约 360 MB）与
`build-arm*/Python-3.12.13/` 下的 `*.o`，**保留 `logs/` 与 `data/` 即可复现全部结论**；
`src/Python-3.12.13.tgz`（26 MB）与 `pydeps/`（60 MB）建议保留以便重跑。
