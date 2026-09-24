# PMU / libkperfx 子代理笔记（920B topdown 采集口径与实测）

> owner: `/root/measurement_profiling/pmu_sweep`
> 日期：2026-09-23/24（Asia/Shanghai）｜机器：a3-22（Kunpeng 920B，640 核，kernel 6.6.0-159.4.3.154，`perf_event_paranoid=2`）
> 交付物：`scripts/measure/pmu_tid_sweep.py`、本文件、`data/profiles/pmu-selftest-20260923T180526Z/`

---

## 0. 一句话结论

`920b_topdown_full_1..9` 可以对着**已经在跑的宿主 TID**（vLLM engine-core 线程）逐组采集，
9 组全部 `confidence=1.0`，**一次完整 9 组扫描 = 9×window + ~1.4 s**（window=3 s → 实测 28.4 s）；
cycles/instructions 与 `perf stat` 的相对误差 **0.03 % / 0.21 %**（交错窗口复测 0.0x~0.3 %，判据 ≤5 %）。
但本机有两个必须先知道的坑：**① 目标必须是"忙线程"而不是"进程组长"**；
**② `memstall_l3miss` 与三个 `dram_*` 计数器在本机恒为 0**（256 MiB 流式负载下也是 0），
所以 920B 上"LLC/DRAM"只能通过 topdown 的 `(memstall_l2miss − memstall_l3miss)` 槽位近似。

---

## 1. 交付物与哈希

| 文件 | sha256 |
|---|---|
| `scripts/measure/pmu_tid_sweep.py` | `d984fd655a82bad90d6a90aa4f74df9d6a73a04cb35158c220458ba26826a073` |
| a3-22 同路径副本 | 同上（`sha256sum` 两侧一致，且等于每个 manifest 里的 `script_sha256`） |

自测证据目录（本地与 a3-22 各一份，64 个文件 / 348 KB，无 `perf.data`）：

```
data/profiles/pmu-selftest-20260923T180526Z/
├── list-groups.txt              # --list-groups 完整输出（§3 引用）
├── perf_reference.{txt,json}    # perf stat 参考值（5 s 窗口）
├── sweep-g12378-w5/             # 主自测：组 1-3,7,8，各 5 s
├── full9-w3/                    # 9 组全绿验证：window=3 s
├── kind-dram/                   # 内存型负载交叉验证（含独立 perf 参考）
├── interleaved-perf-kperfx.txt  # perf/kperfx 交错窗口复测（§4 表）
├── l3-dram-probe.txt            # "L3/DRAM 计数器是否可用"专项探针 + 原始输出
├── l3-dram-probe-verify.txt     # kperfx verify 全事件表（注意其 VERDICT 的含义）
└── selftest.{json,log}          # 汇总
```

---

## 2. 接口契约（与 `point_run.py` 逐字对接）

```bash
python3 scripts/measure/pmu_tid_sweep.py --tid HOST_TID --window S --groups 1-9 \
        --tag NAME --outdir DIR [--pid HOST_PID] [--list-groups] \
        [--preset-prefix 920b_topdown_full_]
```

约定与实现一一对应：

* 每个组 `g` 用 `kperfx.Session(preset=f"{prefix}{g}", target="tid", pid=HOST_TID,
  mode="count", format="json", output="stdout")` 开一个计数窗口，`window` 秒后 `stop()`；
* **组间严格串行**（libkperfx 的"整组不混跑"）；
* 每组记录 `--inner` 子进程的 UTC 起止时间与真实 wall（`inner_wall_s`），
  以及外层记录的 `outer_wall_s`；
* `DIR/pmu_group{g}.json` = **libkperfx 自己写的原始 JSON**
  （直接调 C 导出函数 `kperfx_ctx_write_output(ctx, res)`，即 `kperfx --format json
  --output stdout` 的那条路径；python 绑定本身不落盘，所以这里是 C writer 的字节原样输出）；
* `DIR/pmu_group{g}.meta.json` = `time_enabled / time_running / confidence /
  useronly_forced / counters(counters_map) / status / attempts / rc / 时间戳 / stderr tail`；
* `DIR/topdown.json` = `kperfx.topdown_920b(merged)` 的四桶 + 子项 + `per_group_confidence`
  + 每个字段的覆盖度 `coverage/coverage_incomplete`（哪些子项所依赖的组没跑）；
* `DIR/libkperfx.json` = 全部计数器原始值 + `per_counter`（含每个计数器继承的
  `time_enabled_ns/time_running_ns`）+ `derived`（IPC/CPI/GHz…）+ meta；
* `DIR/manifest.json` = 命令行、groups、window、脚本 sha256、最小/平均 confidence、
  `all_confident`、`low_confidence_groups`、`zero_counting_events`、`exit_code`；
* 退出码：**0** 全部组成功；**1** 部分组可用；**2** 完全失败或拒绝启动（失败也留
  `pmu_group{g}.json` 占位 + `manifest.refused/error`，不吞错）；
* 额外可选（默认值即契约行为）：`--min-confidence 0.99`、`--retries 1`
  （confidence < 0.99 或失败时**重跑该组**，最多 1 次）、`--force`（忽略 PMU 冲突检测）、
  `--no-sudo`。

PMU 权限：脚本自身可用普通用户调用，**PMU 部分自动 `sudo -n python3 <本脚本> --inner …`**
（`sudo -n` 免密已实测可用）；已在 root 下运行时不再套 sudo。
若 libkperfx 回退到 `useronly_forced=true`（非 root 会遇到），会写进 meta/manifest，
不会静默混入。

---

## 3. `--list-groups` 输出与缓存层级归属（**父代理要看的就是这一节**）

完整输出见 `data/profiles/pmu-selftest-20260923T180526Z/list-groups.txt`（含每组事件 + role + 公式）。

### 3.1 直接回答：L1/L2/L3/LLC/DRAM 在哪个组

| 层级 | 预置组 | 具体计数器 | 实测可用性 |
|---|---|---|---|
| **L1D（数据 L1）** | **组 2**（stall 口径） | `mem_stall_anyload`、`memstall_l1miss` | ✅ 可用（dram 负载下 5.19e9） |
| L1D 访问/缺失计数 | **不在任何组** | `l1d_cache`(0x4)、`l1d_cache_refill`(0x3) | ✅ 可用，但需自定义组 |
| **L1I / 取指层级** | **组 4** | `l2i_cache`、`l2i_tlb`（`l2i_*_refill` 恒 0）、`l1i_cache_refill`(0x1，仅自定义) | ⚠️ 计数但很小 |
| **L2（数据侧）** | **组 2** + **组 7** | 组 2：`memstall_l1miss`(→L2 访问)、`memstall_l2miss`；组 7：`l2_bound_buf` / `l2_bound_snp` / `l2_bound_arb`（L2 瓶颈原因） | ✅ 可用（组 7 在 dram 负载下 78.7e6 / 75.3e6） |
| **L2I（指令侧 L2）** | **组 4** | `l2i_cache`、`l2i_tlb` | ✅ 可用 |
| **L3 / LLC** | **组 2**（派生槽） | `(memstall_l2miss − memstall_l3miss)` | ⚠️ **因为 `memstall_l3miss`≡0，该槽 ≈ `memstall_l2miss`**；无独立 L3 access/fill 计数器，无 L3C uncore |
| **DRAM** | **组 2 / 7 / 8** | `memstall_l3miss`（组 2）、`dram_local`（组 7）、`dram_remote` / `dram_remote_cache`（组 8） | ❌ **本机恒为 0**（见 §5.2） |
| 软件事件 | 组 9 | `context_switches`、`cpu_migrations`、`page_faults`、`cpu_clock` | ✅ 可用 |

> 结论：做 prepare_input 的 CPU 侧分析，**组 1（L1 topdown+IPC）、组 2（访存 stall 层级）、
> 组 3（bad-spec 归因）、组 4（取指/L2I）、组 7（L2 瓶颈原因）**是主力；组 8 只能出 store 相关，
> DRAM 部分不可用；组 9 用来看抢占/迁移（对"绑核是否干净"很有用）。

### 3.2 分组清单（`--list-groups` 摘要）

```
group 1: 8 events  cpu_cycles, inst_retired, inst_spec, fetch_bubble,
                   fetch_bubble_max, total_resource_stall, exec_stall, fdiv_fsqrt_stall
          role: L1 topdown 四桶 + IPC 主链
group 2: 8 events  exe_stall_div, fsu_stall, mem_stall_anyload, mem_stall_anystore,
                   memstall_l1miss, memstall_l2miss, memstall_l3miss, rob_stall
          role: 访存 stall 层级（L1/L2/LLC/DRAM bound 分子）
group 3: 8 events  br_mis_pred, o3_flush, nuke_flush, bp_misp_br_ind/blr/bl/ret, if_sp_flush
          role: bad-spec 归因
group 4: 8 events  l2i_tlb, l2i_tlb_refill, l2i_cache, l2i_cache_refill,
                   pcbuf_stall, int_ptag_stall, cc_ptag_stall, vfp_single_ptag_stall
          role: 取指侧 cache/iTLB + ptag/pcbuf stall
group 5: 8 events  vfp_pair/vpd_ptag_stall, int/vfp/cc_mpq_stall,
                   ports_0_serialize, ports_0_nonserialize, ports_1
group 6: 8 events  ports_2..6, ld_cancel_dtlb_miss, ld_cancel_misalign, ld_cancel_lq_full
group 7: 8 events  ld_cancel_inst_type/fwd_hzd/struct_hzd/pipeline,
                   l2_bound_buf, l2_bound_snp, l2_bound_arb, dram_local
group 8: 6 events  dram_remote, dram_remote_cache, st_sca_full, st_head_no_pgen,
                   st_order_fail, st_bound_pipeline
group 9: 4 events  context_switches, cpu_migrations, page_faults, cpu_clock
```

公式（`kperfx.preset_formula`，issue_width=6）：

```
frontend_bound = fetch_bubble / (6*cpu_cycles)*100      retiring = inst_retired/(6*cpu_cycles)*100
bad_spec       = (inst_spec-inst_retired)/(6*cpu_cycles)*100
backend_bound  = 100 - frontend_bound - bad_spec - retiring
mem_l1_bound      = (mem_stall_anyload - memstall_l1miss)/exec_stall * backend_bound
mem_l2_bound      = (memstall_l1miss  - memstall_l2miss)/exec_stall * backend_bound
mem_l3_dram_bound = (memstall_l2miss  - memstall_l3miss)/exec_stall * backend_bound
mem_mem_bound     =  memstall_l3miss / exec_stall * backend_bound
core_bound     = (exec_stall - mem_stall_anyload/anystore - total_resource_stall)/exec_stall
                 * backend_bound
ipc = inst_retired / cpu_cycles
```

---

## 4. 自测：与 `perf stat` 的数值一致性（真做，判据 ≤5 %）

**负载**：`tools/libkperfx/build/workload_gen --kind alu --threads 1 --duration-ms 87000`，
`taskset -c 200`（**未使用 120-159 保留切片**，未碰任何 `/dev/davinci*`，未取 chip3 锁）。

**注意**：`workload_gen` 把负载放在 **pthread** 里，进程组长只在 `pthread_join` 睡觉。
直接测 `pid` 会得到 `perf: <not counted>` / kperfx 全 0（本子代理踩过这个坑）。
脚本用 `/proc/<tid>/schedstat` 选出真正在烧 CPU 的线程（`find_busy_tid`），
本次选中 `busy_tid=4160170`（leader 4160169 的 runtime 增量为 0；
**leader 的 `utime` 是"全进程合计"，用它挑线程会选错**，见 §5.1）。
证据目录：`data/profiles/pmu-selftest-20260923T180526Z/`（`selftest.json` 为汇总）。

| 指标 | `perf stat -t <tid> -e cycles,instructions`（5.001 s 窗口） | libkperfx 组 1（5 s 窗口） | 相对误差 | 判定 |
|---|---|---|---|---|
| cycles | 14,501,769,283 | `CPU_CYCLES` = 14,498,074,842 | **0.0255 %** | ✅ |
| instructions | 19,238,718,943 | `INST_RETIRED` = 19,278,219,558 | **0.2053 %** | ✅ |

派生指标：IPC 1.3297、CPI 0.7520、`cpu_ghz_over_group1_time_running` 2.8999 GHz
（≈ 920B 标称 2.9 GHz，可作为"时间轴自检"）。
内存型负载复核（`--kind dram`，独立 perf 参考，见 `kind-dram/selftest.json`）：
cycles 误差 **0.0220 %**、instructions 误差 **0.0147 %**，均通过。

**交错窗口复测**（`interleaved-perf-kperfx.txt`：perf → kperfx → perf → kperfx，各 5 s，同一 ALU 线程）：

| 窗口 | cycles | instructions |
|---|---|---|
| perf A | 14,502,094,609 | 19,222,550,565 |
| kperfx A | 14,501,295,894（+0.006 %） | 19,235,277,707（+0.07 %） |
| perf B | 14,478,853,575 | 19,218,507,615 |
| kperfx B | 14,499,353,103（+0.14 %） | 19,272,060,849（+0.28 %） |

→ 两个工具在**相邻窗口**上的一致性是 0.0x~0.3 % 量级；剩余波动来自 ALU 负载自身
（同一工具隔一段时间再测，inst 会在 ±0.4 % 内漂移、cycles 基本恒定），不是采集口径差异。
判据 ≤5 % 有 10 倍以上余量，可以放心把 PMU 数字与 perf 数字并排写进文档。

九组全绿验证（`--groups 1-9 --window 3`，ALU 负载）：

| 组 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| confidence | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| t_en (ns) | 3.00054e9 | 3.00020e9 | 2.99779e9 | 2.99832e9 | 3.00067e9 | 3.00065e9 | 3.00008e9 | 2.99956e9 | 3.00002e9 |
| t_run (ns) | 与 t_en 逐组相等（confidence=1.0） | | | | | | | | |
| 组 wall (s) | 3.1317 | 3.1374 | 3.1381 | 3.1364 | 3.1369 | 3.1382 | 3.1419 | 3.1332 | 3.1329 |
| attempts | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

`useronly_forced=false`（9/9），`min_confidence=1.0`，`all_confident=true`，
整棵 topdown 树 `complete_topdown_tree=true` / `coverage_incomplete=[]`；
`zero_counting_events` = ALU 负载下的 11 个事件
（`CPU_MIGRATIONS, DRAM_LOCAL, DRAM_REMOTE, DRAM_REMOTE_CACHE, LD_CANCEL_LQ_FULL,
MEMSTALL_L3MISS, PAGE_FAULTS, VFP_MPQ_STALL, VFP_PAIR_PTAG_STALL, VFP_SINGLE_PTAG_STALL,
VPD_PTAG_STALL`）——**这是"负载没触发"，不是"事件坏了"**，见 §5.3。

### 4.1 附：一段"已知形状"的参考剖面（以后验无卡负载保真度可直接对比）

同一个 ALU 负载、9 组全量 topdown：

```
L1 四桶:  frontend_bound 0.250%   bad_spec 0.009%   retiring 22.08%   backend_bound 77.66%
子项:     core_bound 71.65%   resource_bound 4.80%   mem_bound 1.21%
          mem_l1_bound 0.734%  mem_l2_bound 0.021%  mem_l3_dram_bound 0.459%  mem_mem_bound 0%
frontend: latency 0.236% / bandwidth 0.013%
OOO:      pcbuf 99.20% / mapq 0.58% / ptag 0.21%（组 4/5 的多数 ptag 事件为 0）
IPC 1.330, 2.8999 GHz（组 1 窗口 3.0005 s）
```

纯计算循环就应该是"backend/core_bound 高、mem_* 近 0"；无卡 prepare_input 负载若出现
显著 `mem_l1/l2_bound`，说明访存结构（dict/list/小张量的 cache 行为）确实不同，属预期差异，
需要在文档里说明差异来自哪一类访存模式。

---

## 5. 920B 上的坑（**采集前必读**）

### 5.1 目标必须是忙线程，而不是进程组长

`workload_gen`（以及任何 pthread 型负载）的组长在 `pthread_join`：`perf stat -t <pid>` 会输出
`<not counted>`，kperfx 会给出 `time_enabled=0 / 全 0`。**vLLM 侧同理：必须用 engine-core
线程的 TID**（`/proc/<pid>/task/*`），不能用服务进程 PID。脚本在自测里用
`/proc/<tid>/schedstat` 自动挑忙线程；生产路径应当由父代理传正确 `--tid`，
可以顺手传 `--pid`：脚本会校验该 TID 是否属于这个 PID，不一致时在 `pmu_sweep.log` 里
打 `warning: tid X is not a thread of pid Y`；TID 的身份（`comm/state/cpu/cpus_allowed`）
始终写进 `manifest.json:target`，可直接用来核对"测的是不是 engine-core 线程、跑在哪个核"。

踩坑细节：**不能用 `/proc/<tid>/stat` 的 utime 挑忙线程**——线程组 leader 的 `utime`
是"整个进程的合计"，所以睡觉的 leader 看起来和干活的 worker 一样忙（实测两者都是 40 ticks/0.4 s）。
`/proc/<tid>/schedstat` 第一个字段（runtime ns）是逐线程的，才是可用的判据
（实测 leader 0 ns、worker 399 ms）。

### 5.2 `memstall_l3miss` / `dram_local` / `dram_remote` / `dram_remote_cache` 在本机恒为 0

专项探针（原文见 `data/profiles/.../l3-dram-probe.txt`）：256 MiB 流式（每 64 B 取 1 B，
必miss L2/L3）跑 5 s ——

| 计数器 | `--kind dram` (256 MiB) | `--kind l3` (8 MiB) |
|---|---|---|
| `mem_access` | 2.02e9 | 3.16e9 |
| `l1d_cache_refill` | 6.56e8 | 5.95e8 |
| `memstall_l1miss` | **5.19e9** | 3.20e9 |
| `memstall_l2miss` | **5.12e9** | 2.99e9 |
| `memstall_l3miss` | **0** | **0** |
| `dram_local` / `dram_remote` | **0** | **0** |

→ 后果：`mem_l3_dram_bound` 槽位实际承载"L2 以下的全部 stall"，`mem_mem_bound` 恒 0。
**报告时不要写成"DRAM bound = 0 %"**，要写成"L3/DRAM 合并槽 X%，其中 DRAM 分量本机不可测"。
另：`kperfx uncore` 只列出 `hisi_pcie*`（无 L3C/DDRC），没有备用路径。

### 5.3 "计数为 0" ≠ "事件坏了"

`kperfx verify` 的 VERDICT 取决于它自己那点负载：本次两次 `verify` 分别是
`49 counting / 43 zero` 与 `46 counting / 46 zero`，而且它把 `MEMSTALL_L1MISS` 标成
`NOT-COUNTING`——但同一事件在 dram 负载下计数 5.19e9。
因此脚本只把 0 值事件列进 `manifest.zero_counting_events` 作为**提示**，
判断"死事件"必须换针对性负载（本目录的 probe 就是模板）。

### 5.4 PMU 争用会让 confidence 悄悄掉到 0.35

实测：另一个采集会话活跃时，同一 TID 上 `perf stat` 报 `cycles (35.27 %)`、
kperfx 报 `confidence=0.3522`（`t_en=5.000 s, t_run=1.761 s`），而**数值本身仍"看起来正常"** ——
只有 confidence 能暴露它。因此：

* 脚本对 `confidence < 0.99` 自动**重跑该组一次**（`--retries`），仍低则写进
  `low_confidence_groups`，`all_confident=false`；
* 启动前做冲突检测：另一个 `perf`/`kperfx`/本脚本**测同一个 TID**、或其被测 TID 当前就跑在
  目标 CPU 上、或它显式 `-C <目标 CPU>` / 无 scope（system-wide）时**拒绝启动**（`--force` 可越过）；
* 父代理的数据口径请照抄：**topdown/IPC 必须带 `min_confidence` 与 `per_group_confidence`**。

冲突检测的判定细节（踩过坑，已修）：只有**真正开 PMU 的子命令**才算采集者
（`perf stat|record|top|bench`、`kperfx run|split|probe|verify`）。
本机实测过一次误判：队友正在跑 `sudo -n perf report --stdio … -t | -i perf.data`
（纯读文件的离线分析，还带了被管道截断的 `-t`），旧版本把它当成"无 scope 的系统级采集"→ 拒绝启动。
现在 `perf report/annotate/script/diff`、`kperfx events/presets/merge/info` 一律忽略；
另外祖先进程链（本脚本自己被 `bash -c`/`sudo` 包着）也会排除，不会再自我误伤。

### 5.5 `time_running/time_enabled` 是"组"级别的

libkperfx 报的 `time_enabled/time_running` 是**该组 leader** 的一对值（内核不会给混跑组里的
每个计数器单独报 running 时间）。9 组串行、每组 ≤8 事件时实测 `t_en == t_run`（confidence 1.0），
所以 `libkperfx.json:per_counter` 里每个计数器标注的是"其所属组的 t_en/t_run"，
这一点在文件的 `meta.time_running_semantics` 也写明确了。**不要把它理解成逐计数器值。**

### 5.6 预置组不含任何 cache 访问/缺失计数

`l1d_cache`、`l1d_cache_refill`、`l1i_cache_refill`、`ld_retired`、`st_retired`、`mem_access`
都存在且可用，但**不在 9 组里**。若父代理需要"L1 miss rate / 访存条数"这类量，
需要另开自定义组（≤8 事件），例如
`--events l1d_cache,l1d_cache_refill,mem_access,inst_retired,cpu_cycles,memstall_l1miss,memstall_l2miss`。
本脚本的 `--inner --group` 走的是预置族；自定义组请直接用 `kperfx.Session(events=...)`
（模板见 `l3-dram-probe.txt` 里的复现片段）。

### 5.7 其他

* `perf stat` 默认会把 `sampled=false` 的计数事件做 scale；kperfx 本次未开启 `--scale`，
  所以两个工具比的是原始 delta（confidence=1 时二者等价，本自测已验证）。
* PMU 采集与 `perf record`（采样）**绝不并发**：`point_run.py` 已用 `--perf-s`/`--pmu-window`
  互斥表达，本脚本另有 5.4 的冲突检测兜底。
* 自测/本脚本**不访问 NPU**、不写别人目录、不动宿主 sysctl；采集只"观察"TID，
  不占 CPU（9 组扫描自身 CPU 占用 < 1 %，可以用 `--window 3` 与 vLLM 负载并行跑而几乎不干扰）。
* 只有当某组窗口内目标线程几乎被调度出去时才会出现 `t_run < t_en`；
  vLLM 负载的 engine-core 线程通常常驻，实测 1.0。

---

## 6. 时间预算（父代理排期用）

| 项 | 实测/公式 |
|---|---|
| 单组固定开销 | ≈ **0.15 s**（python 启动 + sudo + 建组 + 收尾），实测 3 s 窗口 → 3.131~3.139 s |
| 一次完整 9 组扫描 | `9 × window + ~1.4 s`；**window=3 → 28.41 s（实测，`full9-w3/manifest.json: wall_s`）**，window=5 → ≈46.4 s，window=10 → ≈91.4 s |
| 5 组（1-3,7,8）| `5 × window + ~0.8 s`；window=5 → 26 s（实测 25.9 s） |
| 合并/落盘 | 包含在上面 1.4 s 里（topdown+libkperfx+manifest 各几 KB） |

建议窗口：与 `point_run.py` 的 workload 窗口对齐，**≥5 s**（统计更稳），
`--pmu-delay` 让它落在稳态区间内（避开服务 warmup）。9 组×5 s 共 ~46 s，
比一轮 perf record 便宜，适合每个实验点都采。

---

## 7. 复现命令

```bash
# 0) 预设组清单（不需要 root，不需要 PMU 权限）
python3 scripts/measure/pmu_tid_sweep.py --list-groups

# 1) 对已有 TID 采集（vLLM engine-core 线程；PMU 部分自动 sudo -n）
python3 scripts/measure/pmu_tid_sweep.py --tid <HOST_TID> --pid <HOST_PID> \
        --window 5 --groups 1-9 --tag a-c64 --outdir data/measure/<run>/pmu

# 2) 自测（复现本文件全部数字；用 CPU 200，任务文档禁止 120-159）
python3 scripts/measure/pmu_tid_sweep.py --selftest --selftest-cpu 200 \
        --groups 1-3,7,8 --window 5 --selftest-window 5 --selftest-full-window 3

# 3) 内存型交叉验证（证明访存事件是活的）
python3 scripts/measure/pmu_tid_sweep.py --selftest --selftest-kind dram \
        --groups 1-2,7-8 --window 5 --no-selftest-full \
        --selftest-dir data/profiles/<stamp>/kind-dram
```

---

## 8. 遗留风险 / 交给父代理的判断点

1. **L3/LLC/DRAM 只能给合并槽位**（§5.2）。若文档需要"DRAM bound"结论，只能写
   "≤ mem_l3_dram_bound，且本机无法再细分"。
2. **OCI 容器内的 engine 线程**：`--tid` 用宿主 TID（`ps -eLo pid,tid,comm` / `nsenter` 视角），
   kperfx 已在宿主侧打开成功（本次自测即为宿主 TID）；容器内取 TID 会与宿主不一致，注意口径。
3. **PMU 是共享资源**：与任何 perf/kperfx 会话并发都会掉 confidence（§5.4）。
   `point_run.py` 里 perf 与 pmu 已互斥；跨 agent 的并发现象（本机同时有别人的 harness 在跑）
   靠冲突检测 + 重试 + `low_confidence_groups` 兜底，极端情况下重跑该点。
4. **`zero_counting_events` 需要按负载解释**：prepare_input 若以 Python 字典/小张量为主，
   `memstall_l3miss`、`dram_*`、`st_*` 大概率为 0，这属于正常，不要当成失败。
5. **未做**（如需可再排期）：自定义 cache 组（§5.6）、per-IRQ/uncore 侧（本机不可用）、
   `--multiplex` 同时开多组（会牺牲 confidence，与"整组不混跑"冲突，默认关闭）。
