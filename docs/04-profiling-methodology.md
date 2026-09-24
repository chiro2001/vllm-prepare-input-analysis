# 04 · Profiling 方法与采集口径

> 状态：**草稿 v0.9**（env_toolchain 负责的 libkperfx / perf / flamegraph-rs /
> 环境隔离 / 锁与噪声门章节已完成并逐条实测；后续 agent 可在末尾「待补充」处续写）。
> 建立：2026-09-24。目标机 `a3-22`（host22，Kunpeng 920B，640 核 / 8 NUMA），
> 目标 NPU `chip3 = /dev/davinci3 = NPU1/Chip1 = Phy-ID 3`，
> CPU 切片 `120-159`（worker `122-157`），`sudo -n` 免密，允许 privileged docker。

本文件回答一件事：**在 a3-22 上，用什么命令、采到什么数据、可信度怎么判定**。
所有数字都带原始命令与产物路径；未实测的部分明确标注「未验证」。

| 章节 | 工具 | 产物 |
|---|---|---|
| §1 | 工具链总览 | —— |
| §2 | libkperfx（920B topdown / IPC / ROI 门控） | `data/toolchain/libkperfx-*.log`、`libkperfx-920b-event-table.md` |
| §3 | perf（stat / record / report / annotate / script） | `data/toolchain/perf-qualification-a3-22.json` + `data/toolchain/raw/` |
| §4 | flamegraph-rs（火焰图） | `figures/*.svg`、`data/profiles/<name>/folded.txt` |
| §5 | 环境隔离（conda / 容器） | `data/toolchain/conda-env-pinput.json`、`data/toolchain/images.json` |
| §6 | 锁、噪声门、采集纪律 | `scripts/chip3_lock.sh`、`scripts/hostnoise_gate.sh` |
| §7 | 采集前检查清单 | —— |

---

## 1. 工具链总览与就位方式

| 组件 | 版本 / 身份 | 位置 | 说明 |
|---|---|---|---|
| 内核 | `6.6.0-159.4.3.154.oe2403sp4.aarch64` | a3-22 | `uname -r` 实测 |
| perf | `perf version 6.6.0-159.4.13.167.oe2403sp4.aarch64` | a3-22 `/usr/bin/perf` | RPM 版本号与内核版本号不同名，属正常 |
| libkperfx | `1.0.0`（`920B(counting)` 后端） | a3-22 `~/projects/vllm/prepare-input-phase/tools/libkperfx` | 从 a3-21 整目录搬入后在 a3-22 重新构建 + 自测 |
| flamegraph | `flamegraph 0.6.14`（flamegraph-rs；crate 名是 **`flamegraph`** 不是 `cargo-flamegraph`） | 本机 `~/.cargo/bin`（x86_64）；a3-22 `tools/bin`（aarch64 musl 静态） | 见 §4 |
| inferno | `0.12.8`（`inferno-collapse-perf` / `inferno-flamegraph` / `inferno-diff-folded`） | 本机 `~/.cargo/bin`；a3-22 `tools/bin` | 折叠 + 渲染，跨架构渲染的关键 |
| conda env | `pinput`（python 3.11.16 / numpy 2.4.6 / pandas 3.0.6） | a3-22 `~/miniforge3/envs/pinput` | **只放分析侧依赖**，vLLM 一律走容器 |
| 基线镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler` | a3-22 docker | id `sha256:dc9a31b8…`，见 `data/toolchain/images.json` |
| 插桩镜像 | `local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler` | a3-22 docker | id `sha256:b2c6ff72…`，**本地构建、无 RepoDigest** |

`a3-21 → a3-22` 的搬运方式（a3-21 上没有 `rsync`，且 a3-22 解析不了 `a3-21`
主机名，所以必须经开发机中转）：

```bash
# 开发机上执行
ssh a3-21 'cd ~/projects/profiling && tar czf - libkperfx' > /tmp/libkperfx.tgz
tar xzf /tmp/libkperfx.tgz -C agents/env_toolchain/staging
tar czf /tmp/libkperfx-push.tgz -C agents/env_toolchain/staging libkperfx
cat /tmp/libkperfx-push.tgz | ssh a3-22 'tar xzf - -C ~/projects/vllm/prepare-input-phase/tools'
```

---

## 2. libkperfx：920B 的 topdown / IPC / ROI 门控

libkperfx 只走 `perf_event_open(2)`，不依赖 `perf` 命令行、不依赖 libpfm，
因此可以与 `perf record` 抢 PMU 之外互不干扰地使用（但**不要同时跑**，见 §6）。

### 2.1 在 a3-22 上重新构建与自测（必须做，不能沿用 a3-21 的结论）

```bash
cd ~/projects/vllm/prepare-input-phase/tools/libkperfx
make -j16 check                       # 普通用户：构建全部 test_* 后跑 40 项
sudo -n KPERFX_REPORT=build/test-report-root.txt bash tests/run_all.sh   # root：打开内核态计数
```

> **坑**：`make -j16`（默认目标）只构建库和 CLI，**不构建 `build/test_*`**，
> 于是 `tests/run_all.sh` 会静默跳过 14 项测试、只跑 26 项。
> 第一次就踩了这个坑（26/26），改成 `make check`（等价于 `make all $(TEST_BINS)`）后才是 40 项。

实测结果（原始日志 `data/toolchain/libkperfx-a3-22-selftest.log`、
`-user.log`、完整报告 `libkperfx-a3-22-test-report.txt`）：

| 运行身份 | 结果 |
|---|---|
| root（内核态计数打开） | **PASS: 40  FAIL: 0** |
| 普通用户（`paranoid=2`，自动降级 `exclude_kernel=1`） | **PASS: 40  FAIL: 0** |

逐项：`test_api test_cli test_count test_env test_event_matrix test_gate
test_multiplex test_output test_privilege test_sample test_tables test_threads
test_workload_loads test_zen4 python-binding cli-probe cli-probe-mux cli-info
cli-verify-events cli-presets load-{alu,fma,div,fsqrt,branch,indirect,l1,l2,l3,dram,
store,tlb,pagefault} load-branch-group3 load-dram-group7 sweep-split sweep-merge
iter-gate-group{1,2,3}` —— 全部 `[PASS]`。

### 2.2 硬件上限、置信度与复用（a3-22 实测）

```bash
sudo -n ./kperfx info     # 原始输出 data/toolchain/raw/kperfx-info.txt
sudo -n ./kperfx probe    # 单组上限探测
sudo -n ./kperfx probe-mux
```

```text
# kperfx probe  （每档把 CPU_CYCLES 重复 N 次）
EVENTS CONFIDENCE   CYCLES         VERDICT
1      1.0000       21000153       ok
...
8      1.0000       21000183       ok
9      0.0011       0              not scheduled
group limit = 8 counter(s) per group

# kperfx probe-mux （每组 8 个计数器）
GROUPS SIMULTANEOUS confidence      SERIAL confidence            RATIO
1      1.000                        1.000                        1.000
2      0.450 0.551                  1.000 1.000                  0.450
3      0.460 0.541 0.000            1.000 1.000 1.000            0.460
```

**结论（可直接引用）**

1. **920B 单组上限 = 8 个计数器**（第 9 个开始 `not scheduled`）。
2. 两组同跑时每组置信度 ≈ **8/16 = 0.5**（a3-22 实测 0.450/0.551），
   即 `confidence = time_running/time_enabled = 已排上/请求`，与事件数量成反比。
   第 3 组在本机直接 `0.000`（该组的 8 个计数器一个都没排上），
   所以**不要假设「N 组一定平分」**，每台机必须自己 `probe-mux` 一遍。
3. 因此正确用法是 **split–merge 串行采集**，而不是同时开多组：
   ```bash
   ./kperfx split --base 920b_topdown_full -n 9 --out-dir sweep -- ./your_binary
   ./kperfx merge --out sweep/metrics.txt sweep/split_*.json
   ```
   串行每组置信度恒为 `1.000`（上面 SERIAL 列）。
4. a3-21 与 a3-22 的 `probe-mux` 结论一致（0.473/0.528/0.000 vs 0.460/0.541/0.000），
   但与库自带 README 里 k147 的历史数据（0.242/0.550/0.209）**不同**，
   再次说明必须重新探。

### 2.3 事件可用性：`verify` 与「零计数」事件

```bash
sudo -n ./kperfx verify      # 逐事件探测，原始输出 data/toolchain/raw/kperfx-verify.txt
```

```text
a3-22: 46 counting, 46 zero, 1 unsupported, 0 failed
a3-21: 51 counting, 41 zero, 1 unsupported, 0 failed
```

两台机的**同名事件可用集合不同**（差 5 个）。已知在 a3-22 上恒零的典型：
`LD_RETIRED`、`ST_RETIRED`、`TOTAL_RESOURCE_STALL`、`FDIV_FSQRT_STALL`，
以及 `HW_BUS_CYCLES`（`not-supported`）。顶级 topdown 四件套与
`MEM_ACCESS / L1D_CACHE / L1D_CACHE_REFILL` 等**均正常计数**，
所以 `frontend_bound / retiring / bad_spec / backend_bound` 这一层仍然可用。
**任何依赖 `LD_RETIRED/ST_RETIRED` 的派生指标（例如 load/store 分解）在 a3-22 上要标注为不可用。**

### 2.4 topdown preset 名与事件码表

```bash
./kperfx presets                              # 家族列表
./kperfx presets 920b_topdown_full_1          # 组 1 的事件 + 全部指标公式
```

| 项 | 值 |
|---|---|
| 主 preset | `920b_topdown_full_1` … `920b_topdown_full_9` |
| 别名家族 | `950b_topdown_full_*`、`k147_topdown_full_*` |
| 小 preset | `basic` / `branch` / `memory` / `stalls` / `ports` / `software` |
| 组数 | 9（组 1–8 各 8 个硬件计数器 = 66 个事件；组 9 是 4 个软件事件，不占 PMU） |
| `issue_width` | 6 |
| 命名事件总数 | 93（`./kperfx events`，a3-22） |

**完整事件码表（NAME / TYPE / CONFIG / ALIASES）+ 9 个组的事件清单**已落盘：

- `data/toolchain/libkperfx-920b-event-table.md`（人读，含生成方式）
- `data/toolchain/libkperfx-920b-events.csv`（机读）
- `data/toolchain/libkperfx-920b-presets.json`（组 → 事件）
- 原始：`data/toolchain/raw/kperfx-events.txt`、`raw/preset-920b_topdown_full_*.txt`

核心指标公式（`kperfx presets 920b_topdown_full_1` 原文）：

```text
frontend_bound   = fetch_bubble / (6 * cpu_cycles) * 100
retiring         = inst_retired / (6 * cpu_cycles) * 100
bad_spec         = (inst_spec - inst_retired) / (6 * cpu_cycles) * 100
backend_bound    = 100 - frontend_bound - bad_spec - retiring
resource_bound   = total_resource_stall / exec_stall * backend_bound
core_bound       = (exec_stall - mem_stall_anystore - mem_stall_anyload
                    - total_resource_stall) / exec_stall * backend_bound
mem_bound        = (mem_stall_anyload + mem_stall_anystore) / exec_stall * backend_bound
mem_l1_bound     = (mem_stall_anyload - memstall_l1miss) / exec_stall * backend_bound
mem_l2_bound     = (memstall_l1miss - memstall_l2miss) / exec_stall * backend_bound
mem_l3_dram_bound= (memstall_l2miss - memstall_l3miss) / exec_stall * backend_bound
mem_mem_bound    = memstall_l3miss / exec_stall * backend_bound
frontend_latency_bound   = fetch_bubble_max / cpu_cycles * 100
frontend_bandwidth_bound = frontend_bound - frontend_latency_bound
```

> ⚠️ 这些公式里的 `mem_stall_*` / `exec_stall` 都在**组 2**，`total_resource_stall`
> 在**组 1**；`merge` 按 `entries[].config` 合并后统一套公式，所以必须 9 组齐全，
> 缺组会让 `backend_bound` 的分解静默变成 `NaN`/0。采集清单里要断言 `count == 66`。

### 2.5 三种使用姿势（prepare_input 分析会用到的）

**A. ROI / 迭代门控（推荐，不受段外代码污染）** —— 代码里只留两个标记：

```c
kperfx_start_env();
for (i = 0; i < N; i++) { kperfx_iter_start(); work(i); kperfx_iter_end(); }
kperfx_end_env();
```

```bash
KPERFX_EVENTS=cpu_cycles,inst_retired,inst_spec KPERFX_ITER_START=10 KPERFX_ITER_END=12 \
KPERFX_FORMAT=json KPERFX_OUTPUT=file KPERFX_FILE=split_1.json ./your_binary
```

门控只做快照差分，窗口外不累计；若一次标记都没调用，会**退化为整段**并在结果里
置 `gated=false`（不会得到一堆 0）—— 分析脚本必须检查这个标志。

**B. 测一条命令（最省事）**

```bash
KPERFX_PRESET=920b_topdown_full_1 KPERFX_FORMAT=text ./kperfx run -- CMD
```

**C. 只挂到某个线程**（引擎 core 线程分析用）

```bash
KPERFX_TARGET=tid KPERFX_PID=<tid> KPERFX_EVENTS=cpu_cycles,inst_retired ... ./your_binary
```

### 2.6 `useronly_forced` 必须每次都看

`perf_event_paranoid=2` 时普通用户会被内核拒绝内核态位，库自动降级并置
`useronly_forced=true`。**root 下实测 `useronly_forced=false`**（见
`data/toolchain/perf-qualification-a3-22.json` → `checks.libkperfx_mux`）。
prepare_input 里有 `malloc`/`mmap`/`ioctl` 这类内核时间，**必须用 root 采**，
否则 `cycles` 会系统性偏小。

---

## 3. perf 采集口径（a3-22 资格验证）

全部结论来自一次可重跑的资格验证，产物
`data/toolchain/perf-qualification-a3-22.json`（结构化）+ `data/toolchain/raw/01..14-*.log`（原始）。

复现命令：

```bash
cd ~/projects/vllm/prepare-input-phase
sudo -n python3 scripts/perf_qualify.py --cpus 122-157
```

被加载的合成受害者是本仓库的 `scripts/synth/probe_load.c`（C 侧，`-O2 -g
-fno-omit-frame-pointer`，三个热点核函数加 `noinline`）与
`scripts/synth/probe_load.py`（CPython 侧，模拟 prepare_input 的 list/dict/
小张量开销）。资格验证**不使用 NPU**。

### 3.1 主机与权限前提

| 项 | 值（原始） |
|---|---|
| `perf_event_paranoid` | `2` → **所有 PMU 采集必须 `sudo -n`** |
| `scaling_governor` | `performance` |
| 采集前残留 | `msprof / devkit / mindstudio`：`NONE`；`perf record/stat/top/trace`：`NONE` |

### 3.2 八项契约逐条实测结果

| # | 契约 | 判定 | 关键证据 |
|---|---|---|---|
| 1 | `perf stat` 计数 + 用户/内核拆分 | **PASS** | `cycles=7,252,895,665`，`IPC=1.8647`，`cycles:u` 与 `cycles:k` 都非零 |
| 2 | `perf stat` 复用 → running ratio | **PASS** | 9 个事件同开，每个都报 `(88.8x%)`；`8/9 = 88.9%` |
| 3 | `perf record -g --call-graph fp`（C） | **PASS** | `probe_hot_branch 57.73% / probe_hot_pointer_chase 31.41% / probe_hot_fp 10.65%` |
| 4 | `perf annotate --stdio` 源码/指令归因 | **PASS** | 22 行源码 + 22 行指令；热点指令示例 `23.21% : 400d64: cmp w10,#0x7` |
| 5 | `perf script` / `perf report --stdio` 契约 | **PASS** | 两者都可解析；`--header-only` 含 cmdline |
| 6 | `perf record -g --call-graph dwarf`（CPython） | **PASS** | top：`_PyEval_EvalFrameDefault 34.36%`、`PyObject_Malloc 14.49%`、`PyLong_FromLong 2.85%` |
| 7 | `perf record -p PID`（线程/进程定向） | **PASS** | 目标 pid 采样成功，top `worker 99.87%` |
| 8 | libkperfx 复用置信度（独立 cross-check） | **PASS** | 16 事件 → `instances=2`，`time_enabled=3002426300`，`time_running=1501213150`，`confidence=0.500000`，`useronly_forced=false` |

### 3.3 复用（multiplexing）的定量规律

**`confidence = time_running / time_enabled = min(1, 8 / n_events)`**，a3-22 上两套工具独立验证：

| 工具 | 事件数 | 实测 enabled/confidence | 理论 8/n |
|---|---|---|---|
| perf stat | 9 | 0.888 ~ 0.891 | 0.889 |
| libkperfx | 16 | 0.500 | 0.500 |

**实践含义**：任何「同一段负载里同时数 >8 个事件」的方案都会按比例丢时间，
topdown 必须走 `split`/`spur` 串行多轮；报告里每个数字都要带 `time_enabled/`
`time_running`（两个工具都直接给），否则无法判断是不是被复用稀释过。

### 3.4 `perf annotate` 能出源码行的前提（踩坑记录）

第一次 `perf annotate -s probe_hot_branch` 报 **`data has no samples!`**。
原因不是工具缺失（`objdump/addr2line/nm/readelf` 都在，`gdb` 缺但不需要），
而是 `-O2` 把三个热点核函数**内联进了 `worker`**，`perf report` 里
99.71% 都记在 `worker` 上，`probe_hot_branch` 符号存在但零样本。
给热点函数加 `noinline` 后，`perf report` 立刻分解为三个符号、`annotate` 出源码行。

**对 prepare_input 的映射**：真实的 `_prepare_inputs` 会被 CPython 解释器内联吗？
不会——但**同名/近名的小函数会被 Python 层的 C 扩展内联或成为 tail call**，
所以对 Python 负载要优先用 `--call-graph dwarf`（实测能把
`_PyEval_EvalFrameDefault / PyObject_Malloc / PyLong_FromLong` 分出来），
对 C/C++ 侧（`vllm/_C`、torch 算子）才用 `fp`。

### 3.5 标准采集命令模板（prepare_input / 引擎 core 线程）

```bash
TID=<engine core 线程 tid>          # 用 ps -L -o tid,comm -p <pid> 找
# 1) 时间线（每次只数 ≤8 个事件，串行多轮）
sudo -n perf stat -e cycles,instructions,inst_spec,branches,branch-misses \
                  -t $TID -- sleep 10
# 2) 火焰图（dwarf，Python 调用树完整）
sudo -n perf record -o /tmp/pi.data -F 999 -g --call-graph dwarf,32768 -t $TID -- sleep 10
# 3) 文本归因
perf report --stdio -i /tmp/pi.data --no-children --percent-limit 0.1 -g none
perf script -i /tmp/pi.data | gzip -9 > pi.script.gz      # 只有几十 KB，可进交付包
# 4) 指令级
perf annotate --stdio -i /tmp/pi.data -s <hot_symbol>
```

> `perf.data` **不上传**：实测 2.5s 的 `flamegraph` 默认采集就写出 **301 MB**。
> 交付包里只保留 `perf script` 文本（gzip 后 ~100 KB）、`perf report` 文本、
> 折叠栈与 SVG；原始 `perf.data` 留在 a3-22 `data/profiles/<name>/`。

---

## 4. flamegraph-rs（火焰图）

### 4.1 安装（开发机，x86_64）

```bash
cargo install flamegraph --locked \
  --config 'source.crates-io.replace-with="rsproxy"' \
  --config 'source.rsproxy.registry="sparse+https://rsproxy.cn/index/"'
cargo install inferno --locked --config ...   # 同上
flamegraph --version      # flamegraph 0.6.14
```

**踩坑**：`crates.io` 直连 403（`static.crates.io` 同样 403），必须换 rsproxy 镜像；
另外 crate 名是 **`flamegraph`**，`cargo install cargo-flamegraph` 会得到
`could not find cargo-flamegraph in registry crates-io`（这不是镜像故障，
`ca/rg/cargo-flamegraph` 在索引里本来就 404，而 `ca/rg/cargo-nextest` 是 200）。

### 4.2 a3-22 上放一份可用的 aarch64 二进制

开发机与 a3-22 **架构不同**（x86_64 vs aarch64），因此：

1. 先用 gnu target 交叉编译 → 装到 a3-22 → **失败**：
   `flamegraph: /usr/lib64/libc.so.6: version 'GLIBC_2.39' not found`（a3-22 是 glibc 2.38）。
2. 改用 **musl 静态**交叉编译 → 成功：
   ```bash
   rustup target add aarch64-unknown-linux-musl
   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld \
   RUSTFLAGS="-C link-self-contained=yes" \
   cargo install flamegraph inferno --locked --target aarch64-unknown-linux-musl --root /tmp/fg-musl
   scp /tmp/fg-musl/bin/{flamegraph,cargo-flamegraph} a3-22:.../tools/bin/
   ```
   实测 a3-22 上 `flamegraph --version` → `flamegraph 0.6.14`，`ldd` → `not a dynamic executable`。
3. a3-22 上端到端实证（`flamegraph` 自己驱动 `perf record`，需要 root）：
   ```text
   [ perf record: Captured and wrote 301.184 MB perf.data (4906 samples) ]
   writing flamegraph to "/tmp/end2end.svg"
   ```
   也验证过「渲染已有 perf.data」：`./flamegraph --perfdata perf.data -o out.svg` → 79 KB SVG。
4. 备选（未采用，但已验证可行）：a3-22 上原生装 rust ——
   `rsproxy.cn/rustup/` 只有 `rustup-init`，**没有 channel manifest（404）**；
   `mirrors.tuna.tsinghua.edu.cn/rustup/dist/channel-rust-stable.toml` 返回 200，
   因此 `RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup` 可行。
   采用交叉编译是为了不在共享机器上引入 ~1.5 GB 工具链。

### 4.3 两条渲染拓扑

| 拓扑 | 何时用 | 数据流 |
|---|---|---|
| **offload（默认）** | 保真对照、需要两边用同一渲染器 | a3-22 `perf script → gzip`（~100 KB）→ 开发机 `inferno-collapse-perf → inferno-flamegraph` → SVG |
| target-side | 想少一次回传 / 现场出图 | a3-22 上 `tools/bin/flamegraph` 或 `inferno-*` 直接出 SVG，只回传 SVG |

两种都实测通过（`scripts/flamegraph.sh selftest` 与 `... selftest --on-target`）。

### 4.4 用法

```bash
# 采集 + 渲染（离线拓扑）
scripts/flamegraph.sh record --name pi-decode-b32 --tid $TID --seconds 20 \
        --call-graph dwarf --title "prepare_input / decode / bs=32"
# 只要折叠栈（做定量对比）
scripts/flamegraph.sh folded --name pi-decode-b32
# 真机 vs 无卡 的差分火焰图（保真对照直接用）
scripts/flamegraph.sh diff --a pi-decode-b32 --b synth-decode-b32 \
        --title "real vs synthetic / decode / bs=32"
```

产物：`figures/<name>.svg`、`data/profiles/<name>/{perf-report.txt,perf-header.txt,folded.txt,perf.script.gz}`；
原始 `perf.data` 只留在 a3-22 的 `data/profiles/<name>/perf.data`。

> `inferno-collapse-perf` 的折叠行**末列是该栈的权重（perf 的 period 之和，
> 默认即 cycles）**，不是样本条数；做占比对比时分子分母都用同一列。

---

## 5. 环境隔离

| 用途 | 隔离手段 | 理由 |
|---|---|---|
| vLLM / vllm-ascend / torch_npu | **容器**（基线或 liteprofiler 镜像） | 版本必须钉死在 `data/toolchain/images.json` 里的 commit |
| 分析脚本（pandas/matplotlib/统计） | conda env **`pinput`** | 不污染容器、不被 CANN 的 python 影响 |
| PMU 采集 | 宿主机 root | `paranoid=2`；容器内采宿主机核线程需要 privileged（本任务允许） |

```bash
~/miniforge3/bin/conda create -y -n pinput python=3.11 numpy pandas matplotlib psutil
~/miniforge3/bin/conda run -n pinput python -c "import numpy,pandas;print(numpy.__version__,pandas.__version__)"
# 3.11.16 2.4.6 3.0.6   —— 详见 data/toolchain/conda-env-pinput.json
```

**py-spy 未安装**：PyPI 上没有 aarch64 wheel，本机装它要拉一整套 rust。
引擎 core 线程的 Python 栈改用 `perf record -t <tid> --call-graph dwarf` +
解释器符号（实测能出 `_PyEval_EvalFrameDefault` 等），见 §3.2 #6。

---

## 6. 锁、噪声门与采集纪律

### 6.1 chip3 独占锁

`scripts/chip3_lock.sh`（`mkdir` 原子语义 + `locks/chip3.lease.json` 租约）：

```bash
scripts/chip3_lock.sh acquire --owner <agent> --purpose "<what>" [--wait S]
scripts/chip3_lock.sh status
scripts/chip3_lock.sh release --owner <agent>       # 只允许 holder 释放
scripts/chip3_lock.sh run --owner <agent> --purpose "..." -- CMD
```

实测（2026-09-24 01:4x，此时 `service_bringup` 正持锁跑 A/B）：

| 操作 | 结果 |
|---|---|
| `status` | `HELD`，租约显示 `owner=service_bringup purpose="chip3 A/B: profiler overhead control (phase timing OFF)"` |
| 第二方 `acquire --wait 2` | 退出码 **75**（`EX_TEMPFAIL`），未抢锁 |
| `release --owner other` | 退出码 **1**，`refusing to release: held by owner=service_bringup, not other` |
| `run`（持锁中） | 按设计等待 `--wait`（默认 1800 s）——**不要在别人持锁时调 `run`**，会静默挂住 |

> `npu-smi` 寻址坑：`npu-smi info -t proc-mem -i 3` 指的是 **NPU id 3**
> （Phy-ID 6/7，正在跑别人的 `VLLMEngineCor`），**不要用它判断 chip3**。
> chip3 必须 `npu-smi info -t proc-mem -i 1 -c 1`（= 物理 chip 3 / `/dev/davinci3`）。

### 6.2 噪声门

```bash
scripts/hostnoise_gate.sh --cpus 120-159 --json data/toolchain/hostnoise-gate-<n>.json
```

三件事：① chip3 是否 `No process in device`；② CPU 切片是否被无关进程占用；
③ 是否有 `msprof/DevKit/perf` 残留。退出码 0=放行、1=拦截。

**2026-09-24 01:40 a3-22 实测（重要）**：

```text
[chip3]   FREE   (npu-smi info -t proc-mem -i 1 -c 1)
[residue] none
[cpu]     slice 120-159: mean 5.54%  max 100.00%  hot(>20%)=[131, 151, 154]
[cpu]     64 unrelated process(es) with affinity on the slice
          pid=1018517 cpu=198.65% threads=59 uid=1002 cpus_allowed=[0,639] ... python
[cpu]     verdict: NOISY
```

即：**chip3 是独占的，但 CPU `120-159` 不是**。宿主机上还有 64 个他人
`python` 进程（uid=1002，59 线程/个，约 200% CPU），它们的 `Cpus_allowed_list`
是 `0-639`，**随时可能落进我们的切片**。`taskset` 只能约束我们自己，不能把别人赶走。

由此得到三条采集纪律：

1. **每次采集前后都跑 `hostnoise_gate.sh` 并留档**（JSON 进 `data/toolchain/`），
   报告里给出当时的 `per_cpu_busy_pct`。
2. 优先用 **per-thread PMU 计数**（`perf stat -t TID`、libkperfx `KPERFX_TARGET=tid`）：
   计数器只统计我们线程，别人的执行不会进分子；受影响的只有 wall time 与
   LLC/内存带宽引起的 IPC 轻微漂移。**wall（phase 计时）必须报噪声区间**。
3. gate 会输出 `<5% busy` 的「安静 CPU」清单与 `taskset` 建议（本次：
   `120-121,123-130,132-139,141-145,147,149-150,152-153,155,158-159`，31/40）。
   **每轮实验现算现用**，不要硬编码。

### 6.3 互斥原则

- **msprof / DevKit 与 perf / libkperfx 不共存**：一次只跑一种（两者都抓 PMU/中断）。
- libkperfx 与 `perf record` 也不要同时挂同一个线程。
- 任何占 NPU / benchmark / perf 的动作先拿 chip3 锁。
- 无卡 harness **不要用 `120-159`**（`plan/EXECUTION.md` §6），
  建议 `200-239` / `360-399`；但这两个切片同样会被他人进程命中，
  所以同样要过 `hostnoise_gate.sh`。

---

## 7. 采集前检查清单（每个实验点照抄）

```bash
cd ~/projects/vllm/prepare-input-phase
scripts/hostnoise_gate.sh --cpus 120-159 --json data/toolchain/hostnoise-gate-$RUN.json || echo "GATE CLOSED"
scripts/chip3_lock.sh acquire --owner $AGENT --purpose "$RUN" --wait 1800 || echo "LOCK BUSY"
#   ... 容器内 liteProfiler 取 phase 时间；宿主机 perf/libkperfx 取 CPU 侧 ...
sudo -n perf stat -e cycles,instructions,branches,branch-misses -t $TID -- sleep $DUR
scripts/chip3_lock.sh release --owner $AGENT
scripts/hostnoise_gate.sh --cpus 120-159 --json data/toolchain/hostnoise-gate-$RUN.after.json
```

每个 manifest 必填：镜像 digest、vLLM/vllm-ascend commit、模型 revision、
workload 参数、绑核、`time_enabled/time_running`、gate 快照、脚本 sha256。

---

## 8. 产物索引（env_toolchain 交付）

| 路径（a3-22 与本地一致） | 内容 |
|---|---|
| `data/toolchain/libkperfx-a3-22-selftest.log` | root 自测完整日志（40/40 PASS） |
| `data/toolchain/libkperfx-a3-22-selftest-user.log` | 普通用户自测（40/40 PASS） |
| `data/toolchain/libkperfx-a3-22-test-report{,-root}.txt` | 逐项断言报告（含每项原始输出） |
| `data/toolchain/libkperfx-920b-event-table.md` / `-events.csv` / `-presets.json` | 事件码表 + 9 组清单 |
| `data/toolchain/perf-qualification-a3-22.json` | perf 八项契约 + 复用率 + 切片占用 |
| `data/toolchain/raw/01..14-*.log` | 上述每一项的原始命令与输出 |
| `data/toolchain/conda-env-pinput.json` | conda 环境与 144 个包版本 |
| `data/toolchain/images.json` | 镜像 id / digest / patch-id 一致性判定 |
| `data/toolchain/hostnoise-gate-0{1,2,3}.json` | 噪声门快照（含 120-159 被他人占用证据） |
| `data/toolchain/raw/15-chip3-lock-semantics.log` | chip3 锁语义实测（75 / 1 退出码、租约内容） |
| `scripts/synth/probe_load.{c,py}` | 无卡合成负载（C 与 CPython 两个口径） |
| `scripts/perf_qualify.py` | 一键资格验证（可重跑，产出上面的 JSON） |
| `scripts/flamegraph.sh` | 火焰图驱动（record / render / folded / diff / selftest） |
| `scripts/hostnoise_gate.sh` | 噪声门 |
| `scripts/chip3_lock.sh` | chip3 独占锁（service_bringup 提供，env_toolchain 已实测验收） |

---

## 9. 待补充（留给后续 agent）

- [ ] 真机 `prepare_input` 的火焰图（§4.4 的 `record --tid`）与热点函数 top-20。
- [ ] 真机 vs 无卡 的 `diff` 火焰图与 topdown 分量对照（§4.4 `diff`）。
- [ ] libkperfx 9 组 `split/merge` 在真实 phase 上的 topdown 数值。
- [ ] `perf annotate` 落到 vllm-ascend / torch_npu 符号后的热点指令。
- [ ] 若启用 MTP：`model: mtp: propose / first_pass / draft_forward` 三个 phase 的独立占比。
