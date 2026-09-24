# env_toolchain —— 交接报告

**Agent**: `/root/env_toolchain`
**状态**: 完成（8 个交付项全部实测通过；3 处「卡住」均已记录失败证据并给出替代路线）
**时间**: 2026-09-24 01:20–01:55 CST
**目标机**: `a3-22` = host22，Kunpeng 920B，640 核 / 8 NUMA，`sudo -n` 免密，可跑 privileged docker
**写入范围**: 仅本地 `/home/chiro/projects/vllm/preparing-input-phase` 与
`a3-22:/home/REMOTE_USER/projects/vllm/prepare-input-phase`（临时物在 `/tmp`）。

---

## 1. 一句话结论

a3-22 上的 **perf / flamegraph-rs / libkperfx topdown 三条采集链路全部打通并逐条验证**；
工具链数字口径、复用率规律、跨架构火焰图方案都已固化在脚本里。
**唯一需要全项目知悉的风险：chip3 是独占的，但 CPU `120-159` 不是**——
宿主机还有 64 个他人 python 进程（uid=1002）的 affinity 覆盖 `0-639`，
`taskset` 挡不住它们，必须靠 `hostnoise_gate.sh` + per-thread PMU 计数来控噪。

---

## 2. 交付物（绝对路径）

### 2.1 本地（= 最终交付树，已与远端同步）

| 路径 | 内容 |
|---|---|
| `/home/chiro/projects/vllm/preparing-input-phase/docs/04-profiling-methodology.md` | 方法论文档（§2 libkperfx / §3 perf / §4 flamegraph-rs / §5 环境隔离 / §6 锁与噪声门 / §7 清单 / §9 待补充） |
| `.../data/toolchain/libkperfx-a3-22-selftest.log` | root 自测（**40/40 PASS**） |
| `.../data/toolchain/libkperfx-a3-22-selftest-user.log` | 普通用户自测（**40/40 PASS**） |
| `.../data/toolchain/libkperfx-a3-22-test-report{,-root}.txt` | 逐项断言原始输出 |
| `.../data/toolchain/libkperfx-a3-22-build.log` | `make check` 构建日志 |
| `.../data/toolchain/libkperfx-920b-event-table.md` | **920B 事件码表 + 9 个 preset 组**（93 个命名事件） |
| `.../data/toolchain/libkperfx-920b-events.csv` / `-presets.json` | 同上的机读版 |
| `.../data/toolchain/perf-qualification-a3-22.json` | perf **8 项契约全 PASS** + 复用率 + 切片占用 |
| `.../data/toolchain/conda-env-pinput.json` | `pinput` env，144 个包 |
| `.../data/toolchain/images.json` | 两个镜像的身份 + patch-id 一致性判定 |
| `.../data/toolchain/hostnoise-gate-0{1,2}.json` | 噪声门快照 |
| `.../data/toolchain/raw/` | 上面每一项的**原始命令 + 原始输出**（kperfx info/probe/probe-mux/presets/events/verify、9 个 preset 明细、perf 01..14 日志、镜像 inspect） |
| `.../scripts/perf_qualify.py` | 一键重跑资格验证（产出上面的 JSON） |
| `.../scripts/flamegraph.sh` | 火焰图驱动：`record / render / folded / diff / selftest`（offload 与 `--on-target` 两条拓扑） |
| `.../scripts/hostnoise_gate.sh` | 噪声门（chip3 + CPU 切片 + 残留采集器） |
| `.../scripts/synth/probe_load.{c,py}` | 无卡合成负载（C 口径 + CPython 口径） |
| `.../agents/env_toolchain/staging/libkperfx/` | 从 a3-21 搬来的原始快照（本地中转副本，7 MB） |
| `.../agents/env_toolchain/make_event_table.py` | 事件表生成器 |

### 2.2 a3-22

与本地同名同路径（`/home/REMOTE_USER/projects/vllm/prepare-input-phase/...`），另外：

| 路径 | 内容 |
|---|---|
| `.../tools/libkperfx/` | 已重建 + 自测通过的库/CLI/python binding（含 `build/test-report{,-root}.txt`） |
| `.../tools/bin/{flamegraph,cargo-flamegraph,inferno-*}` | **aarch64 musl 静态**二进制（开发机交叉编译，13 个） |
| `.../data/profiles/toolchain-qualification/perf-{c,py,pid}.data` | 资格验证的原始 perf.data（97 MB，**留在目标机、不上传**） |
| `.../data/profiles/selftest-*/` | 火焰图自测的 perf.data/script/folded/SVG |
| `.../locks/chip3.holder`, `locks/chip3.lease.json` | chip3 锁目录（当前由 `service_bringup` 持有） |

---

## 3. 复现命令（每一步都能重跑）

```bash
# 0) 前置：噪声门（任何采集前）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash scripts/hostnoise_gate.sh \
  --cpus 120-159 --json data/toolchain/hostnoise-gate-$RUN.json'

# 1) libkperfx 自测（root 与普通用户各一遍）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase/tools/libkperfx && make -j16 check'
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase/tools/libkperfx && \
  KPERFX_REPORT=build/test-report-root.txt sudo -n bash tests/run_all.sh'
# 注意：`make -j16` 只建库+CLI（会静默少跑 14 项），必须用 `make check`

# 2) 事件表 / 上限 / 复用
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase/tools/libkperfx && \
  sudo -n ./kperfx info && sudo -n ./kperfx probe && sudo -n ./kperfx probe-mux && \
  sudo -n ./kperfx verify && ./kperfx events && ./kperfx presets 920b_topdown_full_1'

# 3) perf 资格验证（8 项契约，~40 s）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && sudo -n python3 scripts/perf_qualify.py --cpus 122-157'

# 4) 火焰图（离线渲染拓扑；record 用 --tid 采引擎 core 线程）
scripts/flamegraph.sh record --name <n> --tid <tid> --seconds 20 --call-graph dwarf
scripts/flamegraph.sh folded --name <n>
scripts/flamegraph.sh diff --a <real> --b <synthetic>
# 自测（会自建 probe_load 并采 4 s）：
scripts/flamegraph.sh selftest            # 开发机渲染
scripts/flamegraph.sh selftest --on-target   # a3-22 渲染

# 5) 锁（实测语义）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash scripts/chip3_lock.sh status && \
  bash scripts/chip3_lock.sh acquire --owner X --purpose Y --wait 60 && \
  bash scripts/chip3_lock.sh release --owner X'
```

---

## 4. 关键实测数字（可直接进正文）

| 项 | 值 | 证据 |
|---|---|---|
| 920B PMU 单组上限 | **8 个计数器**（第 9 个 `not scheduled`） | `raw/kperfx-probe.txt` |
| 全量 topdown 组数 | **9**（组 1–8 × 8 个硬件计数器 + 组 9 = 4 个软件事件），命名事件 93 个 | `raw/kperfx-events.txt`、`libkperfx-920b-event-table.md` |
| 发射宽度 | `issue_width = 6` | `raw/preset-920b_topdown_full_1.txt` |
| 复用率规律 | `confidence = time_running/time_enabled = min(1, 8/n)`；perf 9 事件 → `(88.8x%)`；libkperfx 16 事件 → `0.500000` | `perf-qualification-a3-22.json` |
| root 下 `useronly_forced` | **false**（内核态位有效） | 同上 `checks.libkperfx_mux` |
| perf stat IPC（合成混合负载） | `1.8647`（cycles 7.25e9 / 2.5 s） | 同上 |
| `perf annotate` 源码/指令归因 | 22 行源码 + 22 行指令，如 `23.21% : 400d64: cmp w10,#0x7` | `raw/08-perf-annotate-c.log` |
| dwarf 采 CPython 的 top | `_PyEval_EvalFrameDefault 34.36%` / `PyObject_Malloc 14.49%` / `PyLong_FromLong 2.85%` | `raw/10-perf-report-py.log` |
| 火焰图自测 | folded 242–281 栈；SVG 60–80 KB；`perf script.gz` ~110 KB | `data/profiles/selftest-*/` |
| 原始 perf.data 体量 | dwarf 2.5 s ≈ **301 MB**（gnu target `flamegraph` 实测） | 本报告 §5.4 |

---

## 5. 踩坑与失败证据（明确记录，不静默跳过）

### 5.1 `cargo install cargo-flamegraph` → 不存在

```text
error: could not find `cargo-flamegraph` in registry `crates-io` with version `*`
404  https://rsproxy.cn/index/ca/rg/cargo-flamegraph
200  https://rsproxy.cn/index/ca/rg/cargo-nextest     # 镜像没坏
200  https://rsproxy.cn/index/fl/am/flamegraph        # 正确的 crate 名
```

**结论**：flamegraph-rs 的 crate 名是 **`flamegraph`**（装出来的二进制才叫
`cargo-flamegraph` / `flamegraph`）。另外 `crates.io` 与 `static.crates.io` 直连
**403**，必须走 `sparse+https://rsproxy.cn/index/`。

### 5.2 `sudo` 改变 `$HOME` → 误写到 `/root`

第一版 `perf_qualify.py` 用 `os.path.expanduser("~/projects/...")` 推导项目根，
被 `sudo -n` 重定向到 `/root/projects/vllm/prepare-input-phase`（**越界写**）。
已 **完整清理**：先取证（`/tmp/pinput-salvage/failed-run-01.json` + 原始日志），
再 `sudo -n rm -rf /root/projects`（时间戳 01:38:29 三级同秒，确认是本进程一次性创建），
`/root` 已恢复原状；脚本改为**从 `__file__` 推导项目根**。

### 5.3 `~` 在 `${VAR:-~/...}` 里会被 bash 展开 → scp 找错机器

```text
TDIR=[/home/chiro/projects/vllm/prepare-input-phase/...]   # 被展开成本机家目录
scp: /home/chiro/.../perf.script.gz: No such file or directory
```

`scripts/flamegraph.sh` 里另有第二个同源 bug：`cd` 之后仍用相对 `$rdir`，
报 `perf-record.log: No such file or directory`。
**修法**：先 `ssh … 'echo $HOME'` 探到远端绝对家目录，之后 ssh 与 scp 全用绝对路径。

### 5.4 交叉编译的 gnu 二进制在 a3-22 跑不起来

```text
./flamegraph: /usr/lib64/libc.so.6: version `GLIBC_2.39' not found
a3-22 glibc = 2.38
```

**替代路线（已采用）**：`aarch64-unknown-linux-musl` + `rust-lld` +
`-C link-self-contained=yes` → **静态**二进制，`ldd` 输出 `not a dynamic executable`，
在 a3-22 上 `flamegraph --version` → `flamegraph 0.6.14`，端到端出图成功。
（另一条可行路线也已验证：a3-22 原生装 rust —— `rsproxy.cn/rustup/` 缺 channel
manifest（404），`mirrors.tuna.tsinghua.edu.cn/rustup/dist/channel-rust-stable.toml`
返回 200；未采用是为了不给共享机器塞 1.5 GB 工具链。）

### 5.5 `perf annotate` 报 `data has no samples!`

不是工具缺失（`objdump/addr2line/nm/readelf` 都在），而是 `-O2` 把三个热点核函数
内联进了 `worker`（`perf report` 里 99.71% 记在 `worker`）。
给热点函数加 `noinline` 后分解正常。**对 prepare_input 的启示**：Python 负载要用
`--call-graph dwarf`，`fp` 只能看到解释器那几个叶子帧。

### 5.6 越界写与清理（再次强调）

除 §5.2 外，所有临时物都在 `/tmp`；芯片/别人的容器、`dsv41-*`、`cann910`、
他人进程、宿主机系统配置（governor / irqbalance / sysctl / npu-smi set）**一律未动**。
唯一非本任务目录的写入是 `/root/projects`（已清理）。

---

## 6. 给其他 agent 的硬性提醒

1. **`npu-smi info -t proc-mem -i 3` 不是 chip3！** 那是 NPU id 3（Phy-ID 6/7，
   正在跑别人的 `VLLMEngineCor`）。chip3 必须 **`-i 1 -c 1`**（= `/dev/davinci3`）。
   本报告 §1 的判据：`No process in device`。
2. **CPU `120-159` 非独占**（实测 mean 5.54%、max 100%、64 个他人进程 affinity
   覆盖 `0-639`）。每次采集前后跑 `hostnoise_gate.sh` 并留档；wall/phase 计时必须报
   噪声区间，不要用单次数字下结论。gate 会给出当下 `<5% busy` 的 CPU 清单。
3. **PMU 与 wall 是两套证据**：per-thread PMU 计数（`perf stat -t TID`、
   libkperfx `KPERFX_TARGET=tid`）不受他人执行污染；wall 会被抢占，必须配噪声快照。
4. **一次只跑一种采集器**：msprof/DevKit 与 perf/libkperfx 互斥；libkperfx 与
   `perf record` 也不要同时挂同一线程。
5. **`perf.data` 不上传**（dwarf 2.5 s 就 301 MB）。交付包只放 `perf script`
   （gzip ~110 KB）+ `perf report` 文本 + folded + SVG；原始 data 留在 a3-22
   `data/profiles/<name>/`。
6. **topdown 必须 9 组串行 split/merge**，且断言 `count == 66`；缺组会让
   `backend_bound` 分解静默变 0。a3-22 上 `LD_RETIRED/ST_RETIRED/TOTAL_RESOURCE_STALL`
   **恒零**，依赖它们的派生指标要标注「不可用」。
7. 火焰图：`scripts/flamegraph.sh diff --a 真机 --b 无卡` 直接产出差分 SVG，
   这是 P4 保真度验收的现成工具；折叠行末列是 **period 权重**不是样本条数。

---

## 7. 未完成 / 留给后续

- 真机 `prepare_input` 的火焰图、topdown 数值、`annotate` 热点指令（依赖 P2/P3 容器
  跑起来）——已在 `docs/04-profiling-methodology.md` §9 列为待补充。
- `py-spy` 在 aarch64 不可用（PyPI 无 wheel，需整套 rust 工具链），已用
  `perf record -t <tid> --call-graph dwarf` 替代，未再尝试源码编译。
