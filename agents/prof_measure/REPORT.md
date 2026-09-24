# 无卡 profiling 采集与分析方法（prof_measure 交接报告）

> 口径版本 **prof-v1**（2026-09-24 建立）。写这份报告时 `docs/04-profiling-methodology.md`
> 还没出现（env_toolchain 未产出），因此本文按「通用命令 + 显式口径」自洽定义；一旦 04 定稿，
> 以 04 为准，本报告里的差异点见第 9 节。
>
> 机器：a3-22（Kunpeng 920B，2.9GHz，4 socket × 80 core / SMT2 / 8 NUMA，640 逻辑核）。
> 全部测量在 **200-239 / 360-399** 里选核，**绝不使用 120-159**（真机实验切片）。

## 0. 结论速览

1. **链路通了**：被测负载跑在无卡容器里（pi-docker.sh 同款约束），PMU / perf 一律在宿主侧
   以 root 采集（`perf -p <宿主 PID>` 与 `libkperfx target=process,pid=<宿主 PID>`）。
   实测 920B 计数 **time_running/time_enabled = 1.0000**（无 multiplex），
   perf 采样 0 lost samples。
2. **920B topdown 可采**：libkperfx 9 组 preset（每组 ≤8 计数器，硬件上限 8）跑通，
   能给出 level1（retiring / bad-spec / frontend-bound / backend-bound）+ 前端细分 +
   backend 的 resource/core/mem 细分 + mem 的 L1/L2/L3/DRAM/store 细分 + OOO stall 细分，
   每项都带 `time_enabled` / `time_running` / `confidence`。
3. **火焰图两条腿**：`flamegraph-rs 0.6.14` 已装好并产出 on-CPU SVG；容器 DSO 在宿主上
   不存在（`/usr/local/python3.12.13/...`），必须抽 DSO 做 `--symfs` 才有人名（已验证
   `_PyEval_EvalFrameDefault` / `_PyObject_Malloc` 等出现在 SVG 里）。Python 帧另可用
   `py-spy --native`（同时给 Python 函数名与 C 帧）。
4. **热点指令可读**：`perf annotate --stdio -l` 给出「源码行热度 + 指令热度」，
   例如 arith 相位 `_PyObject_Free` 里 `ubfx x4, x1, #34, #15` 占该函数 28.79%，
   对应 `obmalloc.c:1029`。
5. **标定锚点已建立**（纯 CPU microbench，固定工作量）：arith IPC 3.89 / memcpy 2.66 /
   hash 1.17，置信度均 1.0；arith 的全套 9 组 topdown 见第 7 节。这三个相位覆盖
   「解释器前端/退休」「L2 访存」「C 实现（释放 GIL）」三类负载，可作为
   「无卡 vs 真机」对照的锚点。
6. **必须知道的两件事**：a3-22 是**多租户共享机**，别的租户（uid 1002 的 python 进程，
   affinity 0-639）会随机落到我们的核上，导致个别 pass 变慢、IPC 失真——已经做了
   「每 pass 记录 + 一致性检查 + 自动重采」；另外**容器内不能采 PMU**（默认 seccomp 直接
   拒绝 `perf_event_open`，容器里也没有 perf），所以别在容器里装采集器。

## 1. 交付物与入口

| 交付物 | 路径 | 说明 |
|---|---|---|
| 采集总控 | `harness/pi_harness/profiling/collect.py` | 起容器 + 挂仪器 + 汇总 + manifest |
| PMU 计数 | `harness/pi_harness/profiling/pmu.py` | libkperfx 外部 PID 计数窗口（宿主 root） |
| topdown | `harness/pi_harness/profiling/topdown.py` | 9 组 split JSON 合并 + 四层分量 |
| 火焰图 | `harness/pi_harness/profiling/flamegraph.py` | flamegraph-rs 优先，内置渲染器兜底 |
| 热点函数/指令 | `harness/pi_harness/profiling/annotate.py` | perf report / perf annotate 解析 |
| 标定负载 | `harness/pi_harness/profiling/microbench.py` | 纯 CPU 固定工作量 microbench |
| 环境/manifest | `harness/pi_harness/profiling/envcap.py` | 核/频率/SMT/NUMA/镜像 digest/哈希 |
| 噪声门 | `harness/pi_harness/profiling/hostnoise.py` | 逐核忙度 + 选静默核 (`--pick N`) |
| 脚本 | `harness/scripts/prof_calib.sh` | 标定一键跑（产出 `data/harness/prof_calib_*`） |
| 脚本 | `harness/scripts/prof_harness.sh` | **通用采集：给一个 python 入口 + 参数即可** |
| 脚本 | `harness/scripts/prof_topdown.sh` | 只采 topdown（l1 / full，split / mux） |
| 脚本 | `harness/scripts/prof_flamegraph.sh` | perf.data→SVG；或现场采 PID；或 py-spy |
| 脚本 | `harness/scripts/prof_perf_wrapper.sh` | 给 flamegraph-rs 注入 `--symfs` 的 perf 包装器 |
| 标定数据 | `data/harness/prof_calib*_*.{json,csv}` | 见第 7 节表格 |
| 示例火焰图 | `data/harness/prof_calib_flamegraph_sample.svg` | on-CPU（cycles 加权） |
| C 标定锚（可选） | `libkperfx examples/workload_gen` | `prof_calib.sh --workload-gen alu` 接入 |

## 2. 环境与安装（一次性，全部在**自己**的用户目录）

```bash
# 1) libkperfx（只读来源 a3-21，装到 ~/tools/libkperfx；cmake 缺失，用 make）
mkdir -p ~/tools && cd ~/tools
#   a3-22 无法解析 a3-21 主机名 → 经由本机中转（或直接 tar 管道）
cd ~/tools/libkperfx && make -j16            # 产出 kperfx / libkperfx.so / libkperfx.a
./kperfx info                                # pmu=920B backend=920B(counting) group limit 8
sudo ./kperfx probe | head                   # 逐组置信度表

# 2) flamegraph-rs（国内镜像；static.rust-lang.org 只有 ~25KB/s，USTC 是 ~1.4MB/s）
curl -sSfL -o ~/tools/rustup-init \
  https://mirrors.ustc.edu.cn/rust-static/rustup/dist/aarch64-unknown-linux-gnu/rustup-init
RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static \
RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup \
RUSTUP_HOME=$HOME/tools/rustup CARGO_HOME=$HOME/tools/cargo \
  ~/tools/rustup-init -y --profile minimal --default-toolchain stable --no-modify-path
# crates 走 sparse 镜像（否则 crates.io 拉索引极慢）
printf '[source.crates-io]\nreplace-with = "mirror"\n[source.mirror]\nregistry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"\n' \
  > ~/tools/cargo/config.toml
CARGO_HOME=$HOME/tools/cargo ~/tools/cargo/bin/cargo install flamegraph --locked
# → ~/tools/cargo/bin/flamegraph (0.6.14)

# 3) py-spy（Python 帧火焰图）
python3 -m pip install --user py-spy        # → ~/.local/bin/py-spy (0.4.2)
```

实测版本：`perf 6.6.0-159.4.13`、`libkperfx 1.0.0`、`flamegraph 0.6.14`、`py-spy 0.4.2`、
内核 `6.6.0-159.4.3.154.oe2403sp4.aarch64`、`perf_event_paranoid=2`、`sudo -n` 免密可用。

## 3. A1 采集链路：容器内跑负载 + 容器外 root 采集

### 3.1 为什么这么选

| 方案 | 结果 | 结论 |
|---|---|---|
| 容器内 `perf` | 镜像里**没有 perf**（`which perf` 空） | 不可用 |
| 容器内 `libkperfx`（默认 seccomp / 默认 caps） | `kperfx probe` 全部 `open failed (perf_event_open failed)` | **被 seccomp 拒** |
| 容器加 `--cap-add=SYS_ADMIN --security-opt seccomp=unconfined` | 容器内可用，group limit 8、置信度 1.0 | 可用，但改了容器配置 |
| 宿主 root `perf -p <容器进程>` / `libkperfx target=pid` | 全量可用（内核态计数、调用栈、annotate 都在） | **本方案默认** |

容器保持 pi-docker.sh 同款（不挂 NPU、`--network none`、绑核、非 root 用户）就够，
**不需要给容器加任何 capability**：采集器全在宿主 sidecar 里。

### 3.2 起法与 PID 约定（给 replay_harness / 别人的接口）

```bash
# 起法（等价于 prof_harness.sh 内部做的事）
docker run -d --cidfile /tmp/x.cid \
  --network none --cpuset-cpus 200-215 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PYTHONPATH=/work/harness -v <repo>:/work -w /work <IMAGE> \
  bash -c 'kill -STOP $$; exec python3 -m pi_harness.runner ...'

# PID 怎么找（唯一约定）
CID=$(cat /tmp/x.cid)
PID=$(docker inspect -f '{{.State.Pid}}' "$CID")   # 容器 init 的宿主 PID
# 命令用 exec 接在 bash 后面 → PID 不变；kperfx 用 target=process 覆盖全部线程/子进程
sudo kill -CONT "$PID"                             # 仪器挂好后放行
```

* **STOP 门控**：容器以 `kill -STOP $$; exec <cmd>` 启动，宿主先把 perf / kperfx 挂好
  （kperfx 挂好后写 ready 文件），再 `SIGCONT`。短负载也不会丢头，且不需要负载配合。
* **长跑进程**：用 `prof_harness.sh --attach <宿主PID> --duration 20`（例如常驻服务的 worker）。
* **cgroup 方案**：`perf record --cgroup <path>` 也可以，但 docker 的 cgroup 路径随 cgroup v2
  层级变化、还要额外处理，单进程/单进程树时 `-p PID` 更精确，所以默认用 `-p`。

### 3.3 可复现命令

```bash
# ① 计数（PMU，宿主 root）：IPC + 置信度
sudo env PYTHONPATH=$PWD/harness PI_LIBKPERFX=$HOME/tools/libkperfx \
  python3 -m pi_harness.profiling.pmu count --pid "$PID" --wait-exit \
  --events cpu_cycles,inst_retired,inst_spec,br_pred,br_mis_pred,l1d_cache_refill \
  --ready-file /tmp/pmu.ready --out pmu/count.json

# ② 采样（perf，宿主 root）：on-CPU 火焰图 + 热点
sudo perf record -o perf.data -F 999 -g --call-graph dwarf -p "$PID"   # 用 timeout -s INT 或 SIGINT 收
sudo perf report --force --stdio -i perf.data --no-children --sort dso,symbol -t '|'
sudo perf annotate --force --stdio -l -i perf.data --symbol <热点符号> --symfs passes/perf/symfs

# ③ 一体化（推荐）：容器 + 三件套 + manifest 一次做完
bash harness/scripts/prof_harness.sh --tag replay_bs8 --stages pmu,topdown,perf \
  --topdown-level l1 --cpus auto --pyspy --warmup 40 --timeout 900 -- \
  python3 -m pi_harness.runner --model-profile qwen35-0.8b --batch 8 --isl 512 --osl 8 --steps 200
```

### 3.4 链路自证（原始证据在 `data/harness/prof_runs/smoke*/`）

| 证据 | 值 | 位置 |
|---|---|---|
| libkperfx 计数置信度 | `time_enabled == time_running`，confidence **1.0000** | `pmu/count.json` |
| 是否被迫只测用户态 | `useronly_forced=false`（内核态计数保留） | 同上 |
| perf 采样 | `Woken up 250 times`、`Captured and wrote 62.5MB (7666 samples)`、0 lost | `passes/perf/perf.log` |
| 交叉验证 | `perf stat -p PID -e cycles,instructions` IPC 3.31 vs libkperfx IPC 3.7~3.9（不同相位/窗口） | `tests/t_perf_attach.sh` |

## 4. A2 libkperfx 920B topdown

920B 单组最多 8 个计数器，全套 topdown 需要 62 个 → libkperfx 拆成 **9 组**，
语义是「**同一份负载跑 9 遍，每遍只开一组**」，最后按事件 config 合并：

| 组 | 内容 | 用途 |
|---|---|---|
| 1 | cpu_cycles, inst_retired, inst_spec, fetch_bubble, fetch_bubble_max, total_resource_stall, exec_stall, fdiv_fsqrt_stall | level1 四件套 + 前端细分 |
| 2 | exe_stall_div, fsu_stall, mem_stall_anyload/anystore, memstall_l1/l2/l3miss, rob_stall | backend 的 mem/core 细分 |
| 3 | br_mis_pred, o3_flush, nuke_flush, bp_misp_* , if_sp_flush | 分支/冲刷 |
| 4 | l2i_tlb(_refill), l2i_cache(_refill), pcbuf_stall, *_ptag_stall | 前端/PTAG |
| 5-6 | ptag/mpq stall, ports_0..6, ld_cancel_* | 端口与取消 |
| 7-8 | l2_bound_*, dram_local/remote(_cache), st_* | 存储侧 |
| 9 | context_switches, cpu_migrations, page_faults, cpu_clock | 软件事件 |

公式（`issue_width=6`，见 `kperfx presets 920b_topdown_full`）：

```
retiring       = inst_retired / (6*cpu_cycles) * 100
frontend_bound = fetch_bubble / (6*cpu_cycles) * 100
bad_spec       = (inst_spec - inst_retired) / (6*cpu_cycles) * 100
backend_bound  = 100 - frontend - bad_spec - retiring        # 推出来的一项
```

命令：

```bash
# 单组（l1，快）
sudo ~/tools/libkperfx/kperfx --preset 920b_topdown_full_1 run -- <cmd>
# 全套 9 组（每组重启一次负载）
sudo ~/tools/libkperfx/kperfx split --base 920b_topdown_full -n 9 --out-dir D -- <cmd>
sudo ~/tools/libkperfx/kperfx merge --out D/topdown.txt D/split_*.json
# 本仓库的等价实现（推荐，直接出结构化 JSON + 置信度）
PYTHONPATH=harness python3 -m pi_harness.profiling.topdown --split-dir D --out D/topdown.json
```

`topdown.json` 结构：`level1`（4 项，和恒为 100%）、`level2.frontend.{latency,bandwidth}`、
`level2.backend.{resource,core,mem}`、`level3_mem.{l1,l2,l3_dram,mem,store}`、
`ooo_stall.{rob,ptag,mapq,pcbuf,other}`、`raw.{cpu_cycles,inst_retired,inst_spec}`、
`confidence.{per_group,min,time_enabled_sum,time_running_sum,ratio}`、`notes`。

**坑（实测）**：`kperfx info` 在 `perf_event_paranoid=2` 下会打印
`kernel bits unavailable -> events fall back to user only`，但那是**静态提示**；
以 root 运行时实际打开的是内核态计数（结果里 `useronly_forced=false`）。判断口径要看结果字段，
不要看这行提示。另外 `FDIV_FSQRT_STALL` 在整数负载上基本恒为 20（≈0），属正常。

## 5. A3 火焰图（on-CPU）

### 5.1 工具链与降级路径

| 优先级 | 工具 | 状态 | 差异 |
|---|---|---|---|
| 1 | **flamegraph-rs 0.6.14**（`cargo install flamegraph`） | ✅ 已装并跑通 | 直接吃 perf.data，内部 `perf script` + inferno；支持 zoom/search |
| 2a | `perf script` + **内置渲染器**（`flamegraph.py` 自带，无第三方依赖） | ✅ 已实现 | 只有静态 SVG（无 JS 交互），宽度=样本占比，语义与 inferno 一致 |
| 2b | `perf script \| inferno-flamegraph` | 未安装（`cargo install inferno` 即可） | 与 flamegraph-rs 同引擎 |
| 2c | `stackcollapse-perf.pl \| flamegraph.pl`（Brendan Gregg） | 未安装（需 perl 脚本） | 经典实现，输出最"标准" |
| 3 | **py-spy 0.4.2**（Python 帧） | ✅ 已装 | `--native` 同时给 Python 函数名 + C 帧；`--nonblocking` 与 `--native` 互斥 |

```bash
# 首选（collect.py 自动走这条）
python3 -m pi_harness.profiling.flamegraph from-perfdata perf.data -o fg.svg \
    --collapsed fg.folded --tool flamegraph-rs --symfs passes/perf/symfs
# 降级（flamegraph-rs 不在时自动 fallback）
python3 -m pi_harness.profiling.flamegraph from-perfdata perf.data -o fg.svg \
    --collapsed fg.folded --tool builtin --symfs passes/perf/symfs
# Python 帧（pid = 宿主 PID）
python3 -m pi_harness.profiling.flamegraph pyspy --pid 12345 -o py.svg --duration 10
```

### 5.2 符号解析：这一步不做就白采（重要坑）

容器里的 DSO 路径（`/usr/local/python3.12.13/lib/libpython3.12.so.1.0`、torch 的 `.so`）
**在宿主上不存在**，perf 直接采样时热点会退化成裸地址（实测 top-3 全是
`[.] 0x00000000001c95dc` 这种）。三条实测结论：

1. `perf script --symfs DIR` 能把容器 DSO 解析成人名，**但同时会让内核 DSO 全变
   `[unknown]`**（因为内核也改去 symfs 里找）。所以符号化为两遍：
   一遍不带 `--symfs` 拿内核 `地址->符号` 映射，一遍带 `--symfs` 拿用户态符号，
   在折叠栈阶段把 `[unknown]` 还原（`flamegraph.kernel_symbol_map`）。
2. `--symfs` 必须配合 `--force`（perf.data 属 root，普通用户读会被拒）。
3. **DSO 必须从容器里抽，哪怕宿主上同名文件存在**：`/usr/lib64/libc.so.6` 的宿主版本
   与镜像版本（2026-08-14 构建）不同，用宿主版本会出现「符号列出来了但地址对不上」
   ——实测 `memcpy`/`hash` 相位 top-1 变成裸地址、`_PyEval_EvalFrameDefault` 掉出榜单。
   修法：`build_symfs()` 对每个 mmap 路径都 `docker cp` 出来，并按镜像 ID 缓存到
   `~/tools/symfs_cache/<image_id>/`（首个 pass 慢，后续秒级复用）。
4. flamegraph-rs 内部固定用 `perf script`，不暴露 `--symfs`；用 `PERF` 环境变量指向
   `harness/scripts/prof_perf_wrapper.sh` 注入（collect.py 已自动设置）。

每份采集的 `summary.json.symfs` 记录 `n_dsos / copied / cached / existing / failed`，
抽取失败会被写成 caveat（`/lib/modules/**/*.ko.xz` 属内核模块，宿主机上，跳过是对的）。

## 6. A4 热点指令（perf annotate）

```bash
# 热点函数列表（self overhead，按 DSO+符号）
python3 -m pi_harness.profiling.annotate hotspots -i perf.data -o hotspots.json --top 40 \
    --symfs passes/perf/symfs
# 指令级注释（--stdio -l 带行号）
python3 -m pi_harness.profiling.annotate annotate -i perf.data \
    --symbol _PyObject_Free -o annotate.txt --json annotate.json --symfs passes/perf/symfs
# 解析（也可以先跑 annotate 再离线 parse）
python3 -m pi_harness.profiling.annotate parse annotate.txt --json annotate.json
```

`annotate.json` 给两段：`hot_source_lines`（源码行热度，来自 perf 的
`Sorted summary` 段）与 `top_instructions`（指令热度，含所属源码行）。
**实测样例**（arith 相位 `_PyObject_Free`，1422 样本）：

| 指令 | 占比 | 源码位置 | 含义 |
|---|---|---|---|
| `ubfx x4, x1, #34, #15` | 28.79% | `obmalloc.c:1029` `arena_map_get()` | 从池指针里抽 arena 索引位 |
| `ubfx x2, x1, #20, #14` | 17.64% | `obmalloc.c:1126` `arena_map_is_used()` | 抽 arena 编号 |
| `ldr w4, [x6, #4]` | 10.41% | `obmalloc.c:1129` | 读 arena map 位图 |
| `ldr x2, [x3, #8]` | 7.30% | `obmalloc.c:1808` `pymalloc_free()` | 读 pool 头 |
| `ldr x20, [x20, #3192]` | 7.29% | `pycore_pystate.h:27` `_PyInterpreterState_Main()` | 取解释器态 |

注意：`--symbol` 要传**去掉 `[.]` 前缀**的符号名（collect.py 已自动处理，
`annotate.symbol_used` 记录了实际用的名字）。

## 7. C 标定：纯 CPU microbenchmark（无卡 vs 真机的锚点）

```bash
bash harness/scripts/prof_calib.sh --phases arith,memcpy,hash --cpus auto --threads 1
bash harness/scripts/prof_calib.sh --phases arith --full-topdown --cpus auto   # 9 组全套
```

三个相位**固定迭代数**（`--iters`，确定性：9 组 topdown 各 pass 做完全一样的活）：
`arith` = 2000 万次整数乘加+移位，`memcpy` = 300 万次 64KiB 块拷贝（超 L1 落 L2），
`hash` = 35 万次 sha256(16KiB+32B)（CPython 的 C 实现，缓冲 >2KB 会释放 GIL，可扩核）。
数值在 a3-22 容器里标定到「每相位约 4~5 秒」。

标定矩阵：`data/harness/prof_calib_matrix_<ts>.csv`（跨相位 IPC/topdown/热点一览）。

| 相位 | IPC | retiring | bad-spec | frontend-bound | backend-bound | 头号热点 |
|---|---|---|---|---|---|---|
| arith | 3.89 | 37.8% | 1.5% | 13.3% | 47.4% | `_PyObject_Free` |
| memcpy | 2.66 | 40.7% | 0.4% | 7.9% | 51.0% | `_PyEval_EvalFrameDefault` |
| hash | 1.06 | 19.6% | 0.2% | 5.9% | 74.2% | `_PyEval_EvalFrameDefault` |

（数值来自 `prof-calib2` 批次；每个值的 `confidence` 都是 1.0000。逐次运行的原始
`count.json/topdown.json/hotspots.json` 在 `data/harness/prof_runs/prof_calib*_*/`。）

**为什么它能当锚点**：这三个相位覆盖了 topdown 的不同象限（解释器前端/退休、L2 访存、
C 计算），且**不依赖 NPU / torch / 模型**，真机容器里跑同一命令即可得到同口径数字；
真机 vs 无卡的 IPC/topdown 差值就是「平台差异」，可以用来校准后续 harness 结果。

`--threads N`（N≤16）+ hash 相位可以验证**多核扩展**；单线程相位用 `--cpus auto`
挑核（见 §8.3）。

## 8. B 口径统一（每个数字都带上下文）

### 8.1 manifest

每次采集写 `manifest.json`，含：`caliber_version`、`tag`、时间戳、完整起法
（`cmd` / `docker run` 原文）、**镜像 digest**（`image.image_id` + `repo_digests`）、
核列表与每个核的 `scaling_cur_freq` / `cpuinfo_max_freq` / `thread_siblings_list` /
`numa_node`、SMT 是否竞争（`smt_engaged` + `smt_sibling_groups`）、
`governor`、`perf_event_paranoid`、采集前后的 `loadavg` 与 top-12 进程快照、
脚本 sha256（`script_sha256`）。PMU 的每个 pass 另记 `time_enabled/time_running/
confidence/useronly_forced`。

### 8.2 噪声门（采前 + 采后各一次）

```bash
python3 -m pi_harness.profiling.hostnoise --cpus 200-215 --seconds 5 --json noise.json
# 返回码 0=静默 1=超阈；--allow-noisy 只报不拦
```

判定：目标核 `busy% = 1 - (idle+iowait)/total` 两次快照（阈值 5%），
整机 `load1 / 可见逻辑核数`（阈值 0.5），并留 top-12 进程快照。
`collect.py` 把结果写成 `noise_pre.json` / `noise_post.json`，超阈会在
`summary.caveats` 里点名。**只读，不改 governor/irqbalance/sysctl。**

另外会调用（若存在）`scripts/hostnoise_gate.sh --cpus <我们的 cpuset>` 留档；
那个脚本还会检查 chip3 占用与残留 msprof/perf，对无卡任务只是参考信息。

### 8.3 选核：`--cpus auto`

a3-22 是共享机，常驻 40~60 个别人的 python 进程（affinity 覆盖 0-639）。
`hostnoise --pick N` 在 `200-239,360-399` 里按 6 秒忙度挑最安静的 N 个核，
默认避开同一物理核的 SMT 兄弟；`prof_harness.sh --cpus auto` / `prof_calib.sh --cpus auto`
都用它。实测选出过 `223`（单线程标定）与 `216,218,221-222,225,227,229,231`（8 核）。

### 8.4 wall time 占比 vs CPU busy cycles 占比（口径红线）

**两者不可混用**，交付里必须分别标注：

| 口径 | 定义 | 来源 | 用在哪 |
|---|---|---|---|
| wall time 占比 | `T_prepare_input / T_step_wall` | harness 每步计时（`--no-timing` 时必须关掉） | 「何时成为瓶颈」类结论 |
| CPU busy cycles 占比 | `cycles_in_ROI / cycles_in_step` | libkperfx 计数窗口 | 「CPU 花在哪」类结论（topdown 分子分母） |
| CPU busy time 占比 | `cycles / (freq × n_cores)` | 同上 | 把 CPU 负载折算回时间 |

pmu.py 同时输出 `wall_s`、`time_enabled`、`time_running`、`cycles_per_sec`，
就是为了让这两种占比可以互相换算而不混淆。**prepare_input 里含同步等待**，
`wall_s` 明显大于 `cycles/2.9GHz` 时说明有等待（例如 GIL 释放、stub 的 launch 延迟）。

### 8.5 一次只开一种 instrument

perf 采样与 PMU 计数不共存（都抢通用计数器；perf 采样还会因 PMI 中断抬高 cycles）。
所以 collect.py 把 pmu / topdown / perf 拆成**多个 pass**，每个 pass 一份容器、
一份 manifest 记录；同一 pass 内只开一个 instrument。

## 9. 与 `docs/04-profiling-methodology.md` 的口径差异

写这份报告时 `docs/04` 还不存在（由 env_toolchain 负责）。如果 04 定稿后与本报告冲突，
以 04 为准；预期需要对齐的点：

| 项 | 本报告（prof-v1） | 备注 |
|---|---|---|
| issue width | 6（来自 libkperfx preset 公式） | 920B 若换 SKU 需复核 |
| topdown 拆分 | split 9 组（每组一次负载） | mux 模式可用但置信度可能 <1 |
| 采样频率 | `-F 999`（低于 `perf_event_max_sample_rate`） | 需要更高精度可 1999，但丢样本风险上升 |
| 调用栈 | `--call-graph dwarf`（默认） | 可切 `fp`（更省）或 `dwarf,fp` |
| 火焰图口径 | on-CPU（cycles 加权，py-spy 只采 running 线程） | 与「wall time 火焰图」不同 |
| 口径版本 | `prof-v1` | 写进每份 summary.json 的 `caliber_version` |

## 10. 采得到 / 采不到（诚实清单）

**采得到**

* 920B 的 62 个 PMU 计数器（9 组 preset）+ 4 个软件事件；cycles / instructions / IPC /
  分支 / L1D·L2·L3 访存 / DRAM local·remote / OOO stall 细分 / 前端细分；
* topdown 四层 + 细分，且每组 `time_running/time_enabled` 与 confidence；
* on-CPU 火焰图（含 Python 函数名的 py-spy 版本 + C 帧的 perf 版本）；
* 热点函数 self overhead 排名、热点源码行、热点指令（地址/助记符/操作数/源码行）；
* 核频率、SMT 竞争、NUMA 位置、镜像 digest、脚本哈希、噪声快照。

**采不到 / 需要外部输入**

* **容器内 PMU**：默认 seccomp 直接拒 `perf_event_open`（除非给容器
  `--cap-add=SYS_ADMIN --security-opt seccomp=unconfined`，本方案刻意不给）。
* **内核态符号化 + `--symfs` 同时用**：perf 的已知行为，本仓库用两遍法绕过（§5.2）。
* **tracepoint / 内核栈业务归因**：`perf record -e` 只够 cycles；要做 `sched_switch`、
  `sys_enter` 之类需要额外的 tracepoint 事件与更高权限，本任务未做。
* **微架构带宽（emc/带宽计）**：920B 的 uncore 计数（`kperfx uncore`）能看到设备，
  但没有 preset 与公式，本任务未纳入。
* **NPU 侧时间**：无卡环境没有 NPU，`prepare_input` 与 device 的交叉点只能靠
  真机数据（measurement_profiling）拼。
* **真机对照所需的历史值**：必须由真机侧提供（本报告只提供无卡侧数字与区间对照工具
  `compare.py`）。

## 11. 已知坑（按踩到的顺序）

1. **容器内没有 perf**，且默认 seccomp 拒 `perf_event_open` → 采集器必须放宿主。
2. **宿主 DSO 与镜像不一致** → 必须 `docker cp` 出容器 DSO（§5.2）。
3. **`perf script -F dso` 在 6.6 上输出空行** → 改从 `perf report -D` 的
   `PERF_RECORD_MMAP` 抓路径。
4. **`--symfs` 让内核帧变 `[unknown]`** → 两遍法还原（§5.2）。
5. **perf.data 属 root**：后续 `perf report/annotate/script` 都要 `--force`；
   采集完 `chown -R` 归位给本人（collect.py 自动做）。
6. **`kperfx info` 的 "useronly_forced" 提示与 root 实际行为不符**：
   `perf_event_paranoid=2` 下它提示会降级，但 root 实际保留了内核位，
   以结果字段 `useronly_forced=false` 为准。
7. **`perf stat -e br_retired` 在 920B 上报错**（命名事件表里没有，
   用 `br_pred`(0x12) / `br_mis_pred`(0x10)）。
8. **同组事件数 >8 会 multiplex**：920B 硬件上限 8（`kperfx probe` 确认）；
   超过就必须拆组或接受 confidence <1。
9. **py-spy 的 `--duration` 只吃整数秒**：传浮点会 panic
   （`invalid duration: ParseIntError`）。
10. **共享机的"单次干净"不存在**：必须靠 `pass 墙钟一致性检查 + 自动重采`
    （collect.py 会在某组明显偏慢时重采该组，最多 2 轮），并把不一致写进 caveats。
11. **`--warmup` 必须大于 import 时间**（torch/vllm import ≈ 30~45 s）；
    否则仪器挂上时进程已经跑完，症状是 `pid ... 不存在或已退出`。
    更稳的做法是让负载提供稳态时长（`--steady-seconds`），并用 `--window` 只量一段。
12. **容器缺 uid 的 passwd 条目**：`getpass.getuser()` 会 KeyError，
    导致 `import torch` 直接失败（症状伪装成"负载跑太快"，实际是 exit=1）。
    必须传 `USER`/`LOGNAME` 等环境变量 —— 唯一真源是 `harness/scripts/pi_env.sh`，
    collect.py 已改为读它（容器的 `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` 也来自它）。

## 12. 给 replay_harness / measurement_profiling 的接口

```bash
# 1) 采集（noop 口径：与真机可比；负载必须提供稳态时长）
bash harness/scripts/prof_harness.sh --tag replay_b8 --stages pmu,topdown,perf \
  --topdown-level full --cpus auto --warmup 45 --window 25 --pyspy -- \
  python3 -m pi_harness.runner._replay_main \
    --batch 8 --isl 512 --osl 8 --has-gdn --max-num-reqs 64 \
    --max-num-batched-tokens 16384 --slot-mapping-mode noop --no-timing \
    --steady-seconds 180

# 2) 只用已有 perf.data 出图（不重跑负载）
bash harness/scripts/prof_flamegraph.sh --perfdata perf.data -o fg.svg \
  --collapsed fg.folded --symfs DIR

# 3) 只重算 topdown（不重跑负载）
bash harness/scripts/prof_topdown.sh --split-dir DIR --out topdown.json

# 4) 与真机历史区间对照 / 与 cpu_fallback 对照
python3 -m pi_harness.profiling.compare --a summary_noop.json --b summary_fallback.json \
  --out data/harness/prof_cmp_noop_vs_callback.json
python3 -m pi_harness.profiling.compare --a summary_noop.json \
  --ref-ipc 0.719:0.890 --ref-frontend 56:65
```

**PID 约定（唯一）**：`docker inspect -f '{{.State.Pid}}' <cid>`；命令用 `exec` 接在
bash 后面 → PID 不变；kperfx 用 `target=process` 覆盖全部线程与子进程。
**STOP 门控**：容器用 `bash -c 'kill -STOP $$; exec <cmd>'` 启动，仪器就绪后宿主
`kill -CONT`。负载不需要任何配合；想更稳就再加 `--warmup`（大于 import 时间）。

## 13. 正式结果一：`--preset realmachine`（与真机可比口径）

这是本轮**唯一可用来与真机对照**的一组。命令（`pi_env.sh` 为环境变量唯一真源）：

```bash
source harness/scripts/pi_env.sh
bash harness/scripts/prof_harness.sh --tag prof_realmachine \
  --stages pmu,topdown,perf --topdown-level full --cpus auto \
  --warmup 50 --window 30 --pyspy --duration 25 --timeout 900 -- \
  python3 -m pi_harness.runner._replay_main --preset realmachine \
    --batch 1 --isl 128 --osl 64 --steady-seconds 240
```

* run 目录：`data/harness/prof_runs/prof_realmachine_20260924-031609/`（含 250MB perf.data，留档）
* 交付目录：`data/harness/prof_realmachine_20260924-031609/`
* cpuset `201-202,205,207-208,210,212,214`（`--cpus auto` 自动挑的 8 个静默核，NUMA node2）
* 负载：`SlotMappingMode=noop`、`no_timing`、单请求 128 prompt / 64 输出

### 13.1 PMU 计数（`count.json`，30 s 窗口，与 topdown 分开采）

| 计数器 | 值 | 备注 |
|---|---|---|
| `time_enabled` | 29 999 639 890 ns | **等于 time_running** → 无 multiplex |
| `time_running` | 29 999 639 890 ns | confidence **1.0000** |
| `CPU_CYCLES` | 86 997 939 180 | 86.998e9 / 30 s = 2.90e9 cycles/s = **满频 2.9 GHz** |
| `INST_RETIRED` | 82 550 834 074 | |
| **IPC** | **0.9489** | = inst_retired / cycles |
| `INST_SPEC` | 125 616 294 018 | bad-spec 空间 = spec - retired |
| `BR_PRED` / `BR_MIS_PRED` | 20 165 215 374 / 1 176 860 231 | 误预测率 **5.84%** |
| `L1D_CACHE_REFILL` | 1 171 364 138 | 每千条退休指令 14.2 次 L1D 重填 |

按你给的 p50（`prepare_inputs` 364.8 µs + `update_states` 17.1 µs ≈ 382 µs/步）折算，
30 s 窗口约 **7.85 万步**，每步约 **1.108 M cycles**（382 µs × 2.9 GHz ≈ 1.108 M）——
与 PMU 完全对得上，说明窗口里确实是稳态在跑。

### 13.2 topdown 全树（9 组 split，每组独立重跑负载；置信度全 1.0）

| 层级 | 分量 | 值 |
|---|---|---|
| **L1** | **frontend_bound** | **72.11%** |
| L1 | retiring | 15.94% |
| L1 | bad_spec | 8.14% |
| L1 | backend_bound | 3.81% |
| L1 合计 | — | 100.00%（自检通过，`notes: []`） |
| L2 前端 | frontend_latency_bound | **64.75%** |
| L2 前端 | frontend_bandwidth_bound | 7.36% |
| L2 后端 | core_bound | 2.87% |
| L2 后端 | mem_bound | 0.88% |
| L2 后端 | resource_bound | 0.06% |
| L3 访存 | mem_l1_bound | 0.45% |
| L3 访存 | mem_l2_bound | 0.19% |
| L3 访存 | **mem_l3_dram_bound** | **0.24%（本机不可测，见下）** |
| L3 访存 | mem_mem_bound | **0%（本机不可测，见下）** |
| L3 访存 | mem_store_bound | 0% |
| OOO stall | mapq_stall / ptag_stall / rob_stall / pcbuf | 13.47% / 81.69% / 4.23% / 0.62% |

置信度：9 组全部 `confidence = 1.0000`，`time_enabled_sum == time_running_sum ==
269 995 063 690 ns`（9 × 30 s）。

**L3/DRAM 的读法（重要）**：本机 `memstall_l3miss` 与 `dram_*` 计数器恒为 0
（`pmu_sweep` 已确认是这台机器的 PMU 特性，不是负载真的没有 L3/DRAM 访问）。
所以正确写法是「**L3/DRAM 合并槽 0.24%，DRAM 分量本机不可测**」，
不能写成「DRAM bound = 0%」。

### 13.3 与真机历史值对照

```bash
python3 -m pi_harness.profiling.compare --a data/harness/prof_realmachine_20260924-031609/summary.json \
    --ref-ipc 0.719:0.890 --ref-frontend 56:65 \
    --out data/harness/prof_realmachine_vs_histref.json
```

| 维度 | 无卡 `realmachine` | 真机历史 26 配置 | 判定 |
|---|---|---|---|
| 主 bound | **frontend** | **frontend** | 定性一致 ✅ |
| frontend_bound | 72.11% | 56.0–65.0% | 偏乐观 +7.1~+16.1 pp |
| frontend_latency_bound | 64.75% | —（真机同为 latency 主导） | 定性一致 |
| IPC | 0.9489 | 0.719–0.890 | 偏乐观 +0.059~+0.230 |
| 误预测率 | 5.84% | — | 供真机对照 |
| 对照旧口径（`stress`） | IPC 1.553 | — | **远离真机 → 证明 preset 必选** |

**偏差方向已登记**：无卡少了步间 ZMQ 通信/锁，且算子走 CPU ATen（真机走 `torch_npu`），
两条都让无卡侧显得更"干净"——真机压力只会更大（见 `docs/06` §5/§9.3）。

### 13.4 on-CPU 火焰图

* SVG（flamegraph-rs 0.6.14，含 Python 帧）：`data/harness/prof_realmachine_20260924-031609/flamegraph_oncpu.svg`
  （1.30 MB，5339 个帧，其中 69 个 `_PyEval_*` 帧、23 个 numpy 帧）
* 折叠栈（可换渲染器重画）：同目录 `flamegraph_oncpu.folded`
* 采集参数：`perf record -F 999 -g --call-graph dwarf`，30 001 样本，**0 lost samples**
* 符号化：容器 DSO → `--symfs`（见 §5.2），`libpython3.12.so.1.0` 解析出 11.43% 的
  `_PyEval_EvalFrameDefault`
* py-spy 那条路本轮**没成功**：`py-spy 0.4.2 --native` 报 `UNW_EBADREG: bad register number`
  （native 栈展开在这个 CPython 3.12.13 + aarch64 组合上失败）。Python 帧改由
  perf + dwarf 调用栈 + flamegraph-rs 提供，**不影响交付**；`pyspy.log` 留档。

### 13.5 热点函数 top-20（`hotspots.json` / `hotspots_top20.csv`）

| # | overhead | DSO | 符号 |
|---|---|---|---|
| 1 | **11.43%** | libpython3.12 | `_PyEval_EvalFrameDefault` |
| 2 | 1.63% | libpython3.12 | `_PyObject_Malloc` |
| 3 | 1.38% | libpython3.12 | `unicodekeys_lookup_unicode` |
| 4 | 1.37% | libpython3.12 | `_PyType_Lookup` |
| 5 | 1.23% | libpython3.12 | `_PyObject_Free` |
| 6 | 1.12% | libc | `pthread_mutex_lock` |
| 7 | 0.90% | libpython3.12 | `_PyObject_GenericGetAttrWithDict` |
| 8 | 0.82% | libpython3.12 | `tupledealloc` |
| 9 | 0.79% | libpython3.12 | `PyType_IsSubtype` |
| 10 | 0.76% | libpython3.12 | `_Py_dict_lookup` |
| 11 | 0.71% | libtorch_cpu | `c10::impl::OperatorEntry::lookup(DispatchKeySet)` |
| 12 | 0.68% | libc | `0x83e78`（symfs 版本差异，未解析） |
| 13 | 0.64% | libc | `malloc` |
| 14 | 0.64% | libpython3.12 | `tuple_alloc` |
| 15 | 0.59% | libpython3.12 | `_PyFrame_ClearExceptCode` |
| 16 | 0.58% | ld-linux | `0x127a0`（同上） |
| 17 | 0.54% | libtorch_cpu | `torch::autograd::DifferentiableViewMeta::DifferentiableViewMeta(...)` |
| 18 | 0.53% | libtorch_python | `torch::autograd::THPVariable_getitem` |
| 19 | 0.51% | libpython3.12 | `initialize_locals` |
| 20 | 0.50% | libc | `0x83e70`（同上） |

**形状**：top-1 独占 11.43%，**top-10 合计 21.43%**，top-20 合计约 27.4%——
热点非常扁平（解释器 dispatch + 对象分配 + 字典查找 + ATen 派发），
没有单一"凶手函数"。这与真机的分布形状一致（真机 top-10 23.70% / top-20 28.72%）。

### 13.6 热点指令（`perf annotate --stdio -l`，top-3 函数）

**① `_PyEval_EvalFrameDefault`（11.43%，11567 条指令）**

| 指令 | 局部占比 | 源码 | 含义 |
|---|---|---|---|
| `str w0, [x25]` | 2.45% | `object.h:646` `Py_INCREF` | 引用计数自增（**store**） |
| `ldrb w28, [x0, #3]` | 2.39% | `ceval.c:772` | 读对象 GC 位 |
| `str w0, [x1]` | 2.16% | `object.h:646` | 引用计数自增 |
| `ldr w0, [x25]` | 2.04% | `object.h:642` | 读引用计数 |

源码行聚合：`object.h:642` **9.47%**、`object.h:646` **8.07%**、`ceval.c:772` 2.39%
→ **≈17.5% 的解释器时间花在引用计数读写上**，这正是 frontend-latency-bound +
`_PyEval_EvalFrameDefault` 独占 11.43% 的微观解释，也是"能否用
`Py_GIL_DISABLED`/free-threading 或减少对象流转收益"的直接证据。

**② `_PyObject_Malloc`（1.63%，73 条指令）**

| 指令 | 局部占比 | 源码 |
|---|---|---|
| `ldr x0, [x2, #8]` | 24.68% | `obmalloc.c:1541`（`pymalloc_alloc`） |
| `str x3, [x2, #8]` | 19.48% | `obmalloc.c:1544` |
| `ldr x4, [x2, #16]` | 13.54% | `obmalloc.c:1535` |
| `stp x29, x30, [sp, #-32]!` | 9.45% | `obmalloc.c:1562`（函数序言） |

→ 分配器热点是 **free-list 头指针的 load/store**（L1D 延迟敏感），不是算法分支。

**③ `unicodekeys_lookup_unicode`（1.38%，156 条指令）**

| 指令 | 局部占比 | 源码 |
|---|---|---|
| `ldp x21, x22, [sp, #32]` | 29.88% | `dictobject.c:968` |
| `lsl x0, x20, #4` | 23.13% | `dictobject.c:940` |
| `add x19, x19, x19, lsl #2` | 9.15% | `dictobject.c:949` |
| `ldr x1, [x0, #24]` | 8.71% | `dictobject.c:940` |

→ 字典查找里 `index = (size_t)hash & mask` 与 entry 跨步访问（`<< 4` 即 entry 16 B）
是热点；行聚合 `dictobject.c:968` 35.17% + `:940` 32.57%。

（完整 JSON：`annotate_top.json`；每个函数另有 `.txt` 原文在 run 目录的
`passes/perf/annotate/`。）

## 14. 正式结果二：`noop` vs `cpu_fallback`（harness 污染量化）

```bash
# noop（= 与真机可比口径）
bash harness/scripts/prof_harness.sh --tag prof_realmachine ... -- \
  python3 -m pi_harness.runner._replay_main --preset realmachine \
  --batch 1 --isl 128 --osl 64 --slot-mapping-mode noop --no-timing --steady-seconds 240
# cpu_fallback（= 让 numpy 兜底暴露出来，用于量化 harness 自有开销）
bash harness/scripts/prof_harness.sh --tag prof_cpufallback --stages pmu,perf ... -- \
  python3 -m pi_harness.runner._replay_main --preset realmachine \
  --batch 1 --isl 128 --osl 64 --slot-mapping-mode cpu_fallback --no-timing --steady-seconds 180
# 对照
python3 -m pi_harness.profiling.compare \
  --a data/harness/prof_realmachine_20260924-031609/summary.json \
  --b data/harness/prof_cpufallback_20260924-033625/summary.json --top 20 \
  --out data/harness/prof_cmp_noop_vs_cpufallback.json
```

| 指标 | `noop` | `cpu_fallback` | 差 |
|---|---|---|---|
| IPC（30 s 窗口） | **0.9489** | **0.8066** | −0.142（**−15%**） |
| `pthread_mutex_lock` | 1.12% | 1.11% | 持平 |
| `_PyEval_EvalFrameDefault` | 11.43% | 9.80% | −1.63 pp |
| numpy 相关 | 无 | **`ufunc_generic_fastcall` 0.79%**、`PyArray_NewFromDescr_int` 0.42% | 新出现 |
| ATen `as_strided` 路径 | 无 | `at::_ops::as_strided::call`、`at::native::as_strided_tensorimpl` | 新出现 |
| top-20 交集 | — | — | **16/20 = 80%** |

**结论**：`cpu_fallback` 确实把 **numpy 与其驱动的 ATen `as_strided` 派发**推进了热点榜
（正是预期的污染），并且把 IPC 拉低 15%（`ufunc_generic_fastcall` 与 as_strided 派发
在纯 Python 负载之外额外吃 cycles）。所以：

* **对真机口径**：只能用 `noop`（真机这里是 Triton launch，不是 numpy）；
* **量化 harness 自有开销**：用这条差值（IPC −15%，numpy 0.79% + as_strided 派发）；
* 两组 top-20 交集仍是 80%，说明主结构没被改写，污染是**叠加**而非**替换**。

（`cpu_fallback` 那组没跑 topdown，只跑 `pmu,perf`：污染量化用不到 topdown 全树。）

## 15. 交付文件索引（a3-22）

| 路径 | 内容 |
|---|---|
| `data/harness/prof_realmachine_20260924-031609/` | **主交付**：count.json / topdown.json / hotspots.json / hotspots_top20.csv / annotate_top.json / flamegraph_oncpu.svg / flamegraph_oncpu.folded / summary.{json,csv} / manifest.json / noise_{pre,post}.json / cmd.sh / FILES.txt |
| `data/harness/prof_cpufallback_20260924-033625/` | 对照组同上（无 topdown） |
| `data/harness/prof_cmp_noop_vs_cpufallback.json` | 两组 top-20 交集、独占项、IPC 对照 |
| `data/harness/prof_realmachine_vs_histref.json` | 与真机历史区间（IPC 0.719–0.890 / frontend 56–65%）的区间判定 |
| `data/harness/prof_calib*_*.{json,csv}` | 纯 CPU 标定（§7） |
| `data/harness/prof_calib_flamegraph_sample.svg` | 标定火焰图示例 |
| `data/harness/prof_runs/` | 原始 run（含 250 MB perf.data，**不上传**，仅 a3-22 留档） |

**复现**：每个交付目录里的 `cmd.sh` 是当次采集的完整命令（含 `PI_DOCKER_ENVS`）；
`manifest.json` 记了镜像 digest / cpuset / 频率 / SMT / NUMA / 脚本 sha256 / 噪声快照。
只想重算产物（不重跑负载）：

```bash
python3 -m pi_harness.profiling.collect --post-only <run_dir> \
    --cpus <cpuset> --tag <tag> --annotate-top 3
bash harness/scripts/prof_collect_artifacts.sh <run_dir>
```

## 16. 与 replay_harness 的口径约定（2026-09-24 对齐，已生效）

1. **profiling 一律用 `--slot-mapping-mode noop`**（`_replay_main` 默认值）。
   `cpu_fallback` 只用于量化 harness 自有开销：实测把 numpy `ufunc_generic_fastcall`
   （0.79%）与 ATen `as_strided` 派发推进热点榜，并把 IPC 从 0.9489 拉到 0.8066（−15%）。
   **包含 `cpu_fallback` 的火焰图不能拿去跟真机比**——真机这一步是 Triton launch。
2. **`--preset realmachine` 是唯一可对照口径**（`max_model_len=2048 / max_num_reqs=8 /
   max_num_batched_tokens=2048 / no-prefix-caching / has-gdn`）。
   `InputBatch.token_ids_cpu_tensor` 的尺寸从 `(64, 262144)` = 67 MB 降到 `(8, 2048)` = 64 KB，
   cache 行为完全不同：旧 `stress` 口径 IPC 1.553，`realmachine` 口径 IPC 0.9489。
3. **窗口要落在稳态**：负载必须提供 `--steady-seconds`，采集用 `--warmup`（> import 时间）
   + `--window`（只量一段），否则会量到 import / 收尾阶段。
4. **每组 confidence 必须为 1.0**（920B 单组 ≤8 计数器），并前后跑 `hostnoise_gate.sh`
   留档（`noise_pre.json` / `noise_post.json`）。
5. **`_replay_main` 的 `--no-timing` 不影响火焰图**：`_prepare_inputs` /
   `compute_slot_mapping` 是真实调用，perf 采样看得到（`--no-timing` 只关掉 SubStepTimer
   的 wrap 开销，约 0.5 µs/调用）。
6. **`pin_memory`/`aclInit` 缺口**：`realmachine` 预设不走 MTP，本轮两组采集
   （noop / cpu_fallback）日志中**没有** `UNSHIMMED_DEVICE_OP`、也没有 `aclInit` 报错；
   唯一见到的 torch_npu 相关告警是容器启动时的
   `/usr/local/Ascend/cann-9.1.0 owner does not match`（非功能性）。
