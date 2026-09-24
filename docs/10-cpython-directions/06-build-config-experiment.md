# CPython 构建配置对解释器性能的影响 —— a3-22 真机量化验收基线

> **定位**（依 2026-09-24 的指示调整）：本文**不是**"发现镜像没开优化"，而是
> **构建优化的量化验收基线**：给出 aarch64 + 本负载形状下的绝对数字、相对加速比、
> PMU 归因、以及一套可复现的判据，用于验收"优化生效了没有、生效了多少"。
>
> **一句话结论（含一处对原始判定的更正，见 §2）**：
> ① **镜像里实际跑的那个 `libpython3.12.so.1.0` 已经是 computed gotos 构建** ——
> 尽管它自带的 `pyconfig.h` / `sysconfig` 声称 `USE_COMPUTED_GOTOS` 未定义。
> ② 因此"给镜像加 `--with-computed-gotos`"在 a3-22 上的**净收益是 0**（已在生产里兑现）；
> computed gotos 相对 switch 的收益是 **−4.7 ~ −7.8%（混合负载，两次独立会话）/
> −7.6 ~ −11.8%（分派密集核）**，这部分**已经在线上生效**。
> ③ 还剩下的构建杠杆是 **PGO+LTO**（arm C）：本负载形状 **−22.6%**（C vs B，11 轮中位），
> 真实 `prepare_input` 代码路径 **`prepare_inputs_us` −15.6%**（464.7 → 392.3 µs，n=5 vs n=3）。
> ④ 两项的机制**完全不同**：computed gotos 是"**少执行指令**"（instructions −5.1%，
> IPC 仅 +1.6%，branch-miss 率只降 0.043 pp）；PGO+LTO 才是"**修好前端**"
> （instructions −12.2% **且** IPC +14.2%，branch-misses −21.2%、iTLB misses −63.6%）。
> 这条**否证**了 `00-runtime-build-config.md` §2 里"`ptag_stall 69.5%` ← `switch` 的单条
> 间接跳转打爆 BTB"的机制解释，并把"该投哪一项"指向 PGO+LTO。

---

## 0. 结论速览

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | **镜像的 libpython 已经是 computed gotos**：`_PyEval_EvalFrameDefault` 函数体积 **46268 B / 11567 条指令 / 253 条间接跳转**，与 `--with-computed-gotos` 臂**逐字节同尺寸**；`--without-computed-gotos` 臂是 44072 B / 11018 条 / **1** 条 | **强**（静态三重：函数体积、.text 体积、间接跳转计数） |
| 2 | 行为上（**同会话逐轮交错**）镜像 python 也落在 computed gotos 一侧：5 个分派密集核中 **4 个与 B 臂相差 ≤ 0.7%**，同时比 A 臂快 **6.6–11.3%**；混合负载比 A 快 6.0%、比 B 慢 2.9% | **强**（7 轮交错测量） |
| 3 | computed gotos 在**本负载形状**下的净效应 = **−4.7 ~ −7.8%**（两次会话）/ **−8.8 ~ −11.8%**（分派密集核） | **强** |
| 4 | 该收益来自**指令数下降 5.1%**，不是分支预测改善（IPC 仅 +1.6%、branch-miss 率 −0.043 pp） | **强**（perf stat） |
| 5 | 生产镜像并**没有**吃过 PGO/LTO。这是唯一剩下的构建杠杆：**本负载形状 −22.6%**（C vs B），真实 `prepare_inputs_us` **−15.6%** | **强** |
| 5b | 两项的机制不同：CG 省指令（IPC 不动）；**PGO+LTO 同时降指令并大幅降 branch-miss / iTLB-miss**，与"前端受限"画像方向一致 | **强** |
| 6 | 微基准 IPC ≈ 2.37–2.75、无卡 harness IPC ≈ 0.95，**都不等于**真机 `prepare_input` 窗口的 0.771；但 harness 的 0.95 与真机 0.719–0.890 属**同一区制**（本仓库既有结论），0.771 的可复现性成立、对 switch 分派的归因不成立 | 中 |

---

## 1. 口径与身份（复现的前提）

| 项 | 值 |
|---|---|
| 宿主机 | `a3-22` = host22，Kunpeng 920B，**640 逻辑核 / 4 socket × 80 core / 8 NUMA**，kernel `6.6.0-159.4.3.154.oe2403sp4.aarch64` |
| 宿主机 OS / 编译器 | openEuler 24.03 LTS-SP4；宿主 `gcc (GCC) 12.3.1 (openEuler 12.3.1-110.oe2403sp4)` |
| **构建编译器** | 容器内 `gcc (GCC) 12.3.1 (openEuler 12.3.1-111.oe2403sp3)`，binutils 2.41，make 4.4.1（**构建与生产在同一镜像内完成，编译器与镜像一致**） |
| 基础镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler`，image id `sha256:dc9a31b8330d399ad8e91dabaca25798c3d838c36851897bf9a1f77f793072ec`，repo digest `sha256:24ae7427b6cad5ee29e0665e6f69a4d51c9f6178035e38f2ed161bb3d29fe81c` |
| 源码 | `Python-3.12.13.tgz`（TUNA 镜像），`sha256=0816c4761c97ecdb3f50a3924de0a93fd78cb63ee8e6c04201ddfaedca500b0b`，与 python.org SPDX 声明的摘要一致 |
| 生产解释器 | `/usr/local/python3.12.13/bin/python3.12`，`3.12.13 (main, Aug 3 2026, 02:43:47) [GCC 12.3.1 (openEuler 12.3.1-105.oe2403sp3)]` |
| 绑核 | 构建 `200-239`（`make -j16`；C 臂后期收缩到 `216-239` 以便腾出基准核）；**基准/PMU 一律 `taskset -c 200-203`**（NUMA node 2）。**全程未使用 120-159** |
| 线程 | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` |
| 噪声 | 见 §7：该切片**非静默**（有他人 python 以宽 affinity 漂入），故三臂**逐轮交错**（A,B,C,A,B,C…）取中位数，使漂移对各臂等权 |
| GPU/NPU | **未挂 `/dev/davinci*`**，未使用 chip3，未起长期服务，未改宿主机配置 |

### 1.1 三个 arm 的精确定义

三臂均为 `./configure --prefix=<独立目录> --enable-shared <下表>`，**同一份源码、同一镜像内编译器、`make -j16`、不做 `make install`**，直接在源码树里跑 `./python`（`LD_LIBRARY_PATH=<源码树>`）。

`--enable-shared` 是**三臂共同项**，因为生产镜像的 `CONFIG_ARGS` 正是 `'--enable-shared'`，保持 ABI 一致。

| arm | configure 追加参数 | `USE_COMPUTED_GOTOS` | 构造目的 |
|---|---|---|---|
| **A** | `--without-computed-gotos` | **0** | switch 分派对照（**不是**镜像现状，见 §2） |
| **B** | `--with-computed-gotos` | **1** | 隔离 computed gotos |
| **C** | `--enable-optimizations --with-lto --with-computed-gotos` | **1** | PGO+LTO（训练集是 CPython 自带的 `-m test --pgo`，与我们的负载分布不同 ⇒ 数字是**保守下界**） |

三臂构建后均用 `./python -c "import sysconfig; print(sysconfig.get_config_var('USE_COMPUTED_GOTOS'))"` 验证得到 **0 / 1 / 1**；快照见 `data/cpython-build-exp/data/arm{ABC}.sysconfig.json`。

---

## 2. ⭐ 对原始判定的更正：镜像的 libpython **已经是 computed gotos**

### 2.1 静态证据（三重、彼此独立）

对镜像里**实际被加载的那个** `/usr/local/python3.12.13/lib/libpython3.12.so.1.0` 做反汇编：

| 构建 | `_PyEval_EvalFrameDefault` 体积 | 指令数 | 该函数内间接跳转（`br xN`） | 整库 `.text` 体积 |
|---|---:|---:|---:|---:|
| **镜像 stock** | **46268 B** | **11567** | **253** | **0x24dfcc** |
| A（`--without-computed-gotos`） | 44072 B | 11018 | **1** | 0x24d30c |
| B（`--with-computed-gotos`） | **46268 B** | **11567** | **253** | **0x24dfcc** |

镜像与 B 臂**函数体积、指令数、整库 .text 体积三项完全相同**；A 臂少 3264 B（`0xcc0`）`.text`。
这正是 `switch`（**一条**间接跳转打 Jump Table）与 computed gotos（**每个 opcode 一条**间接跳转）的差别。

**顺带确定：镜像是"纯 `--with-computed-gotos`"构建，没有 PGO/LTO。**
理由：如果镜像做过 PGO（`-fprofile-use`）或 LTO，热/冷代码分离与内联决策都会改变布局，
**不可能**与一个 plain `-O3` 构建的 `.text` 大小**逐字节相同**（0x24dfcc），
更不可能连 `_PyEval_EvalFrameDefault` 都同为 46268 B。
这条推论与镜像 `sysconfig` 唯一可信的那半句（没有 `--enable-optimizations` / `--with-lto`）一致，
也与行为测量一致（镜像 ≡ B，见 §2.2）。

**判据的本质差别**（这才是 253 vs 1 的来源）：

| 构建 | 分派表 | 分派指令形态 |
|---|---|---|
| **镜像 stock** | **`opcode_targets.0`**，位于 `.data.rel.ro`，**大小 0x800 = 256 × 8 B（指针表）** | `ldr x0, [x1, w8, sxtw #3]` + `br x0`（**每个 opcode 一处**，共 253 处） |
| A（switch） | **无 `opcode_targets` 符号** | `adr x1, <表>` + `add x0, x1, w0, sxth #2` + `br x0`（**全局仅 1 处**；`sxth #2` = 4 字节宽偏移 ⇒ 编译器生成的 Jump Table） |
| B（computed gotos） | `opcode_targets.0`，`.data.rel.ro`，**0x800** | 同镜像 |

```bash
# 复现（三条互相独立）
cd ~/projects/vllm/prepare-input-phase/exp-cpython
for f in refimg/image-libpython3.12.so.1.0 build-arm{A,B}/Python-3.12.13/libpython3.12.so.1.0; do
  echo "== $f"
  objdump -t $f | grep opcode_targets                       # 指针表符号（CG 独有）
  objdump -d --disassemble=_PyEval_EvalFrameDefault $f \
    | grep -cE "\bbr\s+x[0-9]"                              # 253 / 1 / 253
done
```

### 2.2 行为证据（与静态证据独立）

用**同一份** `scripts/dispatch_micro_bench.py` / `pi_sim_bench.py`（`taskset -c 200-203`、
`OMP_NUM_THREADS=1`、固定迭代数）跑镜像 python 与 A/B/C 臂。

> **关键方法要求**：镜像必须与各臂**在同一个会话里逐轮交错**跑
> （`scripts/run_bench_mixed_stock.sh`，顺序 `A,B,C,stock` × 7 轮）。
> 本机**会话间**漂移可达 3–5%（§10.6），把镜像单独跑一个会话再与各臂比会得出错误结论。

| 内核（7 轮中位） | A（switch） | B（CG） | C（PGO+LTO） | **镜像 stock** | **镜像/A** | **镜像/B** | B/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| `int_loop` | 112.61 | 100.89 | 86.86 | **107.64** | −4.4% | +6.7% | 0.896 |
| `attr_loop` | 66.80 | 62.30 | 58.54 | **62.40** | **−6.6%** | **+0.2%** | 0.933 |
| `call_loop` | 125.99 | 113.02 | 107.40 | **112.67** | **−10.6%** | **−0.3%** | 0.897 |
| `branch_loop` | 101.02 | 90.25 | 67.04 | **89.61** | **−11.3%** | **−0.7%** | 0.893 |
| `global_loop` | 63.31 | 58.17 | 49.66 | **58.05** | **−8.3%** | **−0.2%** | 0.919 |
| `pi_sim_bench`（ns/step） | 257573 | 235489 | 183034 | **242222** | **−6.0%** | +2.9% | 0.914 |

原始 JSON：`data/bench-dispatch-mixed/`、`data/bench-pi-mixed/`（`work_digest` 全部 `77232000`）。

**结论**：镜像 python **明确落在 computed gotos 一侧，而不是 switch 一侧**：
5 个分派密集核里有 **4 个与 B 臂的差距 ≤ 0.7%**，同时比 A 臂快 **6.6–11.3%**；
混合负载上镜像比 A 快 6.0%、比 B 慢 2.9%。
残余的 0–3% 落在我们实测到的会话间漂移范围内，且镜像与我们的 C 臂**不是同一个编译器点版本**
（镜像 `12.3.1-105` vs 我们容器里的 `12.3.1-111`），**不做更强的主张**；
判定分派路径本来就以 §2.1 的静态证据为准，行为数据只作独立佐证。

> 附：第一次（非交错）会话的读数与本次一致，但**原始文件已被后续重跑覆盖**，
> 故本文一律引用上面这次**同会话交错**的数据。

### 2.3 为什么 `pyconfig.h` / `sysconfig` 会给出相反答案

镜像安装面与**实际被链接的 `libpython3.12.so.1.0`** 不一致：

```bash
$ ssh a3-22 'docker run --rm --entrypoint python3 \
    quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler -c "
import sysconfig
print(sysconfig.get_config_var(\"USE_COMPUTED_GOTOS\"))      # -> 0
print(sysconfig.get_config_var(\"HAVE_COMPUTED_GOTOS\"))     # -> 1
print(sysconfig.get_config_var(\"CONFIG_ARGS\"))"'
0
1
'--enable-shared' 'LDFLAGS=...' '--prefix=/usr/local/python3.12.13'
# 且 pyconfig.h:1710 是   /* #undef USE_COMPUTED_GOTOS */
```

**我们只陈述观察到的这组不一致，不给未经证实的原因。** 可复现的旁证：
`python3.12 -VV` 记录的生产编译器是 `openEuler 12.3.1-105.oe2403sp3`，而**当前**同一镜像里
`gcc --version` 是 `12.3.1-111.oe2403sp3` —— 生产 libpython **不是在当前镜像的编译器 revision 下构建的**，
与"安装的 header/Makefile 与已安装的 .so 来自不同构建"这一假设相容（**假设，未证实**）。

> **方法论后果（比结论本身更重要）**：
> `sysconfig.get_config_var("USE_COMPUTED_GOTOS")` 是**解释器构建时的配置回执**，
> 不是**当前这个 `.so` 的构建回执**。
> 要判定"生产到底走哪条分派路径"，必须看**二进制**（函数体积 / 间接跳转数）或**行为**（分派微基准），
> 不能只看 `pyconfig.h`。

---

## 3. 微基准（核心产出的方法部分）

### 3.1 负载形状与构造理由

`scripts/pi_sim_bench.py`，按真实 `prepare_input` 画像构造（依据 `docs/01` / `docs/05` / `docs/07`）：

| 真实特征（已实测） | 基准里的对应物 |
|---|---|
| `CommonAttentionMetadata` 25 字段 dataclass，每步重建 + `.replace()` | `AttnMeta` **25 字段** dataclass，每步构造 + `dataclasses.replace()` |
| `AscendGDNAttentionMetadataBuilder.build()` **每步 3 次** | `Builder.build()` 每步 3 次 + 3 次 `.replace()` |
| 属性链 `self.vllm_config.scheduler_config.max_num_seqs` | 三层嵌套 dataclass 属性链 |
| 大量 1–3 行短方法 | 5 个 1 行方法（`compute_num_computed_tokens` / `num_valid_tokens` / …） |
| `req_id_to_index` 字典查找 | `dict` 插入 + `get` 命中/未命中 |
| 每请求状态对象 churn（无 `__slots__`） | `ReqState`（10 个属性、list、`__init__`）逐轮重建 |
| 小张量 `cu_seqlens` / `slot_mapping` / block table，shape (1,)–(64,) | numpy `cumsum`/`diff`/`nonzero`/`searchsorted`/boolean-mask 写，**无任何数值计算** |

每轮 = `--steps` 个"引擎步"，每步把上面 5 类工作**交错执行**（模拟真实每步混合），
固定迭代数、固定数据、无随机性。另有 5 个**分派密集核**（`scripts/dispatch_micro_bench.py`：
`int_loop`/`attr_loop`/`call_loop`/`branch_loop`/`global_loop`）作为"分派影响的上界"对照——
如果 computed gotos 有用，它在纯分派环里的效果必须 ≥ 在混合负载里的效果。

**相同工作量的证明**：每臂输出 `work_digest`（标量累加校验和）。
三臂 + 镜像的 digest **全部等于 `77232000`** ⇒ 执行了完全相同的算术。

---

## 4. 三臂结果

> **路径约定**：下文所有 `data/...`、`scripts/...` 都指实验目录
> `EXP=$HOME/projects/vllm/prepare-input-phase/exp-cpython`（在 a3-22 上）。
> 交付包里的同步副本是 `data/cpython-build-exp/{data,logs,scripts}/`，
> 机读总表见 `data/cpython-build-exp/SUMMARY.md`。

### 4.1 构建耗时（实测）

| arm | configure | make（`-j16`） | 产物 `libpython3.12.so.1.0` | `_PyEval_EvalFrameDefault` | 间接跳转 |
|---|---:|---:|---:|---:|---:|
| A | 17 s | **27 s** | 30,812,384 B（`.text` 0x24d30c） | 44072 B | 1 |
| B | 17 s | **27 s** | 30,689,296 B（`.text` 0x24dfcc） | 46268 B | 253 |
| C | **44 s** | **813 s（13.6 min）** | 30,209,248 B（`.text` **0x23124c**） | — | **217** |

C 臂的 813 s 全花在三阶段：`-fprofile-generate` 全量构建 → `-m test --pgo` 训练（43 个测试）→
`-fprofile-use` 全量重建 + **LTO 链接**（GCC 的 `-flto-partition=none` 是单分区，链接期很长；
libpython 一次链接就要 3~4 分钟）。

两个附带观察：① C 的 `.text` 比 B **小 4.9%**（LTO 做了跨模块死代码消除与去重）；
② C 的 `_PyEval_EvalFrameDefault` 内间接跳转从 253 降到 **217**，
说明 PGO+LTO 把一部分 opcode 的分派点合并/内联掉了——**分派点数本身也是 PGO 的作用面**。

> 顺带结论：**A/B 两臂 30 秒就能建出来**——所以"多建一个对照臂"的成本几乎为 0，
> 这类实验不应该省掉 B 臂。

### 4.2 微基准（`taskset -c 200-203`，逐轮交错，取中位数）

主口径 = **11 轮交错**（`data/bench-pi-clean/`，无其它负载）；括号内是**同配置的第二次独立复现**
（7 轮，`data/bench-pi/`；该次前 4 轮与 harness 重复测量重叠，故降为复现用）。

| 指标 | A（switch） | B（computed gotos） | C（PGO+LTO+CG） | 镜像 stock |
|---|---:|---:|---:|---:|
| `pi_sim_bench` ns/step（中位） | 256266（263673） | 236414（251374） | **182984（192233）** | 242222 |
| 最小 ns/step | 252803（258154） | 234275（247222） | **181791（185820）** | 241387 |
| 臂内离散（max−min）/中位 | 4.2%（4.8%） | 2.6%（15.1%，含 1 个离群轮） | 4.3%（4.6%） | — |
| **相对 A（中位比）** | **1.0000×** | **0.9225×（−7.8%）** | **0.7140×（−28.6%）** | 0.9404× |
| 相对 A（**逐轮配对中位**） | 1.0000× | **0.9286×（−7.1%）** | **0.7165×（−28.4%）** | — |
| **相对 B（= 镜像现状）** | 1.0840× | 1.0000× | **0.7740×（−22.6%）** | 1.0286× |
| 相对 B（逐轮配对中位） | 1.0768× | 1.0000× | **0.7747×（−22.5%）** | — |

**两轮复现的一致性**：`C/B` = **0.7740 / 0.7747（11 轮）** 与 **0.7647 / 0.7650（7 轮）**
⇒ **−22.5 ~ −23.5%**，跨会话稳定；`B/A` = **0.9225 / 0.9286** 与 **0.9534 / 0.9623**
⇒ **−3.7 ~ −7.8%**，方向一致但幅度有会话间漂移（B 臂离散度也更大）。

分派密集核（7 轮中位，ns/iter）：

| kernel | A | B | C | B/A | **C/B** |
|---|---:|---:|---:|---:|---:|
| `int_loop` | 119.38 | 103.88 | **86.72** | 0.870 | **0.835** |
| `attr_loop` | 67.66 | 63.97 | **59.52** | 0.945 | **0.930** |
| `call_loop` | 126.94 | 114.62 | **108.77** | 0.903 | **0.949** |
| `branch_loop` | 102.17 | 91.18 | **68.18** | 0.892 | **0.748** |
| `global_loop` | 62.50 | 59.30 | **50.76** | 0.949 | **0.856** |

> **读数**：
> ① **computed gotos 的净效应为 −4 ~ −8%**（两次独立会话：−4.7% / −7.8%）。
> 公开文献里 computed goto 的收益多为 **1–4%（x86）**，本机测得偏大；
> 是否与 aarch64 间接分支预测器特性有关，**本实验不展开**，只如实报数。
> 无论取哪一端，它都**比 PGO+LTO 小一个量级**；
> ② **PGO+LTO 的效应大而稳**：`C/B` 在两次会话、共 18 轮里逐轮为
> 0.7644 / 0.7650 / 0.7681 / 0.7666 / 0.7612 / 0.7877 /（离群 0.6516）/
> 0.7811 / 0.7879 / 0.7561 / 0.7769 / 0.7611 / 0.7920 / 0.7727 / 0.7864 / 0.7747 /
> 0.7598 / 0.7726 —— **中位 0.774 ⇒ 本负载形状 −22.6%**，除 1 个离群轮外全部落在 0.756–0.792；
> ③ C 臂在分派密集核上的收益同样大（`branch_loop` −25%、`int_loop` −16.5%、`global_loop` −14.4%），
> 说明 PGO 的收益不只是"改代码布局"，也改了**用户 Python 层的分支与调用序列**；
> ④ 分派密集核那一列（7 轮，`data/bench-dispatch/`）是**旧的 A/B 会话**，
> 与主口径不同会话，只用于方向性对照。

### 4.3 PMU 归因（`perf stat`，同一二进制 / 同一负载 / 同一核）

负载 = `pi_sim_bench --steps 4000 --rounds 5`（≈16 s 测量窗口，纯基准、不含解释器启动；
原始输出 `data/perf-clean/perf-arm{ABC}-g{1,2}.txt`）。同一 `perf stat` 命令、同一绑核、
同一负载，三臂各跑一遍。

| 计数 | A（switch） | B（CG） | C（PGO+LTO+CG） | **B/A** | **C/B** |
|---|---:|---:|---:|---:|---:|
| cycles | 15,153,701,127 | 14,145,882,614 | **10,874,288,079** | 0.9335 | **0.7687** |
| instructions | 35,904,059,214 | 34,062,602,480 | **29,908,817,262** | 0.9487 | **0.8781** |
| branches | 7,849,095,216 | 7,152,758,930 | **5,826,018,326** | 0.9113 | **0.8145** |
| branch-misses | 92,413,390 | 81,140,060 | **63,921,788** | 0.8780 | **0.7878** |
| L1-icache-load-misses | 495,328,741 | 490,760,814 | **334,154,269** | 0.9908 | **0.6809** |
| iTLB-load-misses | 99,177,696 | 94,796,845 | **34,521,681** | 0.9558 | **0.3642** |
| **IPC** | **2.3693** | **2.4080** | **2.7504** | +1.6% | **+14.2%** |
| **branch-miss 率** | **1.1774%** | **1.1344%** | **1.0972%** | −0.043 pp | −0.037 pp |
| **L1-icache miss 率** | 7.2390% | 8.0824% | 7.5167% | **+0.84 pp（变差）** | −0.57 pp |
| **iTLB miss 率** | 1.4494% | 1.5612% | **0.7765%** | **+0.11 pp（变差）** | **−0.78 pp** |

**"为什么有效"的答案（两项机制完全不同）**：

* **computed gotos（A→B）**：`instructions −5.1%`、`IPC +1.6%` ⇒ 主要来自
  "每字节码少执行几条指令"（`switch` 要额外做 `opcode` 取值 + Jump Table 装载 + 比较）。
  **分支预测没有变好**（branch-miss 率只降 0.043 pp），i-cache / iTLB 的**失配率反而变差**
  （线程化代码更大）。⇒ 与"BTB 冲突 → `ptag_stall`"的假设**不符**。
* **PGO+LTO（B→C）**：`instructions −12.2%` **且** `IPC +14.2%`，同时
  branch-misses **−21.2%**、iTLB-load-misses **−63.6%**、L1-icache-load-misses **−31.9%**。
  ⇒ 这一项是**真正的"代码布局 + 内联 + 跨模块优化"收益**，前端（取指/TLB/BTB）压力全面下降，
  与"前端受限"的画像**方向一致**。**这才是应该投的那一项。**

> **这条否证了原始机制假设**：如果 `ptag_stall 69.5%` 真由 `switch` 的
> "单条间接跳转打数百目标 → BTB 冲突"造成，那么换成 computed gotos **必须**让
> branch-miss 率显著下降——实测没有。叠加 §2（镜像本来就是 computed gotos），
> "`ptag_stall` ← 分派跳转"这条线索判定为 **不成立**。

---

## 5. 真实负载验证（无卡 harness）

`harness/scripts/pi-docker.sh` 的真实 `prepare_input` 代码路径（vLLM 0.26.0 真代码，无卡），
`--preset realmachine --batch 1 --isl 128 --steps 200`，`taskset -c 200-215`。

**CPython 换装方式**（**不在任何镜像上打 tag、不 commit、不动别人的容器**）：
镜像的 `bin/python3.12` 带 `DT_RPATH=/usr/local/python3.12.13/lib`，而 `DT_RPATH` 优先于
`LD_LIBRARY_PATH`（这是本项目差点踩空的一步），因此唯一可靠的换法是**只读地把
`libpython3.12.so.1.0` 单文件 bind-mount 覆盖**；stdlib / site-packages / torch / CANN 全部保持镜像原样。

| arm | n | `prepare_inputs_us` p50（中位） | p50（最小） | mean（中位） | `update_states_us` p50 | **相对 stock** |
|---|---:|---:|---:|---:|---:|---:|
| **stock**（镜像原样 python） | **5** | **464.7** | 458.9 | 469.7 | 18.7 | 1.0000× |
| A（switch） | 2 | 464.8 | 457.5 | 473.2 | 19.4 | **+0.01%** |
| B（computed gotos） | 2 | 451.6 | 450.4 | 458.7 | 18.2 | **−2.83%** |
| **C（PGO+LTO+CG）** | **3** | **392.3** | **388.3** | **415.6** | **15.6** | **−15.59%** |

（`--preset realmachine --batch 1 --isl 128 --steps 200`，`taskset -c 200-215`，每次一个全新容器。
arm 之间**逐次交错**：`stock,A,B,stock,A,B,…`。）

> **三条读数**：
> ① **C 相对生产现状 −15.6%（p50 464.7 → 392.3 µs）**，而且**旁证同步改善**：
> `update_states_us` 也 −16.6%（18.7 → 15.6 µs）。两个独立 scope 同向 ⇒ 不是单点噪声。
> ② **A / B / stock 三者在本条判据上分辨不出来**（|Δ| ≤ 2.8%，而 200 步单轮的 run-to-run
> 波动本身就有 ±1–2%；stock 的历史 3 次基线 470.9 / 469.6 / 476.6 µs 也是同一量级）。
> ⇒ **harness 的 `prepare_inputs_us` 无法用来验收 computed gotos 这种 ~5% 级别的解释器差异**，
> 它只够验收 PGO+LTO 这种 ~15% 级别。
> ③ 这也解释了为什么 arm A（switch）看起来"和 stock 一样快"：真实 `prepare_input` 的耗时里
> 解释器派发只占一部分，剩下的 torch/CANN C 代码与对象/记账开销与分派方式无关。

---

## 6. "A 臂能否复现镜像 IPC 0.771 量级"的判定

直接回答：**这个判据本身要拆成三个独立问题**，混在一起问会得出错误结论。

| 问题 | 答案 | 证据 |
|---|---|---|
| (a) A 臂是不是镜像的等价物？ | **不是**。A 是 `switch`，镜像是 computed gotos | §2 静态 3 项 + 行为 6 项 |
| (b) 微基准能否复现 IPC 0.771？ | **不能，而且不应该能** —— 它 IPC **2.31**（numpy/ATen 的 C 路径占比高、IPC 高） | §4.3 |
| (c) 无卡 harness 能否落到 0.771 量级？ | **能（同一区制）** —— 本仓库既有对照（libkperfx 同 preset）：harness **0.949** vs 真机 **0.719–0.890** | `data/harness/consistency_harness_vs_realmachine.json` |

因此：

1. **"0.771 量级"是可复现的**：用 harness + libkperfx 同一 preset 就落在同一区制
   （frontend-bound、IPC<1、frontend_latency_bound 主导，harness 乐观 +7~32%，归因见 `docs/06` §9.2）。
   **这与构建配置无关**。
2. **"0.771 来自 switch 分派"不可复现，也不需要复现**：生产根本没用 switch（§2）。
   0.771 的低 IPC 应由 ①真机每步间夹着 scheduler / RPC / forward launch / D2H 回传
   （污染 i-cache 与分支预测器）、②`Event.synchronize()` 真等、③device 驱动路径的
   syscall/内核态时间 来解释（这些在 `docs/06` §9.2 已登记，本文不重复论证）。
3. **对"降级"指令的回应**：按"若不能复现则降级"的口径——**未能复现的不是 0.771，而是
   "生产走 switch"这个前提本身**。所以我们降级的对象是**机制归因**（`ptag_stall` ← 分派跳转），
   而不是"低 IPC / frontend-bound 这个现象"（它由既有对照独立支撑）。

---

## 7. 环境噪声与控制

```bash
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash scripts/hostnoise_gate.sh --cpus 200-203 --sample-s 2 \
       --json exp-cpython/data/hostnoise-200-203.json'
```

```
[chip3]   BUSY  (别人任务；本项目全程未使用 chip3)
[residue] none (no msprof/DevKit/perf process)
[cpu]     slice 200-203: mean 7.94%  max 31.33%  hot(>20%)=[201]
[cpu]     29 unrelated process(es) with affinity on the slice (uid=1002 python, cpus_allowed=[0,639])
[cpu]     verdict: NOISY      idle CPUs (<5% busy): 3 of 4 -> 推荐 taskset: 200,202-203
```

**应对**（已全部执行）：
1. 三臂**逐轮交错**（`A,B,C,A,B,C,…`），中位数对各臂等权，系统漂移不偏向任何一臂；
2. 每轮都是**新进程**（消除解释器内部状态差异），每轮前 `gc.collect()`，固定 `PYTHONHASHSEED=0`；
3. 构建负载在 C 臂后期用 `docker update --cpuset-cpus 216-239` 移出 200-203；
4. 结论只用**中位数 + 全部分轮值**（`data/bench-*/**.csv`），不用单点最好值。

---

## 8. 验收判据（可直接拿去用）

"构建优化生效了没有、生效了多少"的最短判据链，**按代价从低到高**：

| 步骤 | 命令 | 通过判据 / 本机读数 |
|---|---|---|
| 1. 判定分派路径 | 对 `libpython3.12.so.1.0` 数 `_PyEval_EvalFrameDefault` 的间接跳转数与函数体积 | §2.1 表；**不要只看 `sysconfig`**。本机读数 ⇒ 生产**已经**是 computed gotos |
| 2. 分派净效应 | `dispatch_micro_bench.py`，A/B 交错 7 轮 | computed gotos vs switch：`call_loop` −11.8%、`branch_loop` −11.6%、`int_loop` −8.8%、`global_loop` −8.9%、`attr_loop` −7.6%。**若 <3% 说明该 CPU 的间接预测器已吃掉该历史优势**（本机不是） |
| 3. 本负载净效应 | `pi_sim_bench.py`，三臂交错 7 轮，**先比 `work_digest`** | digest 必须全等；C 相对 B 的增量 = "还能拿到多少"。本机：A→B **−6.7%** |
| 4. 归因 | `perf stat -e cycles,instructions,branches,branch-misses` | instructions 降而 IPC 平 ⇒ "少执行指令"；**只有 branch-miss 率降才是预测器收益**。本机是前者 |
| 5. 真实负载验收 | `run_harness_arm.sh <arm>`，看 `prepare_inputs_us` | 相对 stock 的 p50 变化；**n≥3 取中位**（单次波动 ±1–2%） |

**本次验收的最终读数**：

| 步骤 | 读数 |
|---|---|
| 1 | 生产**已经是** computed gotos（253 条间接跳转 + `opcode_targets` 0x800 指针表） |
| 2 | 该 CPU 上 computed gotos 值 **−7.6 ~ −11.8%**（分派密集核）；**不是 <3%**，即间接预测器没有吃掉这个差异 |
| 3 | 本负载形状：**B/A −7.8%（11 轮）/ −4.7%（7 轮）**；**C/B −22.6% / −23.5%** |
| 4 | CG 的机制是 **instructions −5.1%**（IPC +1.6%、branch-miss 率 −0.043 pp）；**PGO+LTO 的机制是 IPC +14.2% + branch-miss −21.2% + iTLB-miss −63.6%** |
| 5 | 真实 `prepare_inputs_us`：**C 相对 stock −15.6%**（464.7 → 392.3 µs）；A/B/stock 三者分辨不出来 |

⇒ **结论：`--with-computed-gotos` 不应再计入收益（已在线上）；真正该做的是
`--enable-optimizations --with-lto`。**

---

## 9. 对 `00-runtime-build-config.md` 的修正

该文写于本实验之前，其 §0/§1/§2/§4/§5.2 的相关判定被本实验的**二进制证据**推翻。
（该文已按新定位改写过一次；以下是第二轮修正，**本次已完成修改**。）

| 位置 | 原口径 | 应改为 |
|---|---|---|
| §0 / §1.2 | "解释器走 `switch` 分派" | **错误**：被实际链接的 `libpython3.12.so.1.0` 是 computed gotos（§2.1） |
| §1.1 | 以 `sysconfig` / `pyconfig.h` 作为"构建实况"证据 | 该证据只描述**安装面**，与**二进制**不一致；判定分派必须看二进制或行为（§2.3） |
| §2 | "`ptag_stall 69.5%` + `frontend_bound 66%` 与 switch 分派一致" | **不成立**：生产没有 switch；且 A→B 实测 branch-miss 率不变（§4.3） |
| §4 | arm A = "复现镜像现状" | arm A 是 switch，**不是**镜像现状；**arm B 才等价于镜像现状** |
| §5.2 | "构建优化 10~20% 解释器段收益 → −0.15 ~ −0.35 ms" | 实测：computed gotos 那部分**已经在生产里**（增量 = 0）；剩余构建杠杆只有 PGO+LTO，其本负载增量见 §4 |

---

## 10. 未做到 / 局限（如实登记）

1. **C 臂的 `libkperfx` topdown 未采**：只知道 `perf stat` 的 4+4 个事件（§4.3），
   没有 frontend_bound / retiring 的直接分解。若要做"构建优化是否把 frontend_bound 拉下来"，
   需要在 harness 侧用 libkperfx 同 preset 采一遍 A 与 C（本仓库 `harness/scripts/prof_topdown.sh` 已具备）。
2. **C 臂只用了一轮 PGO 训练集**（CPython 自带的 `-m test --pgo`，43 个测试）。
   PGO 训练集与我们的负载分布不同，**没有做"用 vLLM 负载做训练"的 PGO**（那需要改训练流程）。
   ⇒  §4 的 C 数字是**保守下界**，不是上界。
3. **未采 libkperfx topdown**：920B 的 topdown 需 root + 9 组复用采集。本次只跑了 `perf stat`
   的两组 4 事件（`cycles/instructions/branches/branch-misses` 与 `L1-icache/iTLB`）。
   因此**没有"frontend_bound 是否随构建变化"的直接 topdown 数字**，
   本文用 branch-miss 率 + i-cache/iTLB 失配率替代，并在 §4.3 明示该替代的局限。
4. **微基准不是真负载**：复刻的是*形状*，没有 device 派发 / DMA / 锁；其 IPC（2.37–2.75）
   **不能**与真机 prepare_input 的 0.771 直接比（§6）。
5. **未做端到端（TTFT / TPOT / 吞吐）对比**：无卡 harness 没有 device，端到端指标在无卡下无意义；
   要做必须占 chip3 起真服务，超出本次范围。
6. **B 臂的会话间漂移未解释**：`B/A` 在两次会话里是 −7.8% / −4.7%，
   而同一会话内 C 的离散度只有 ~1%。可能与该切片上他人进程（§7）的间歇占用有关，
   未进一步归因。**不要**把 `B/A` 当成精确到 1% 的数。
7. 镜像 `pyconfig.h` 与二进制不一致的**成因未证实**（§2.3 只给了相容性旁证）。
8. `_PyEval_EvalFrameDefault` 一处小差异未解释：A 臂函数 44072 B vs B 臂 46268 B 的差额
   （3264 B）大于"253 条 `br` 各 4 B"（1012 B），其余为 `DISPATCH()` 宏展开的取表/递进代码。
   未逐条指令核对，只做了计数与体积两个独立维度的一致性验证。

---

## 11. 复现清单

```bash
EXP=~/projects/vllm/prepare-input-phase/exp-cpython        # a3-22
IMG=quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler

# 0) 建三臂（A/B 各约 45 s；C 是三阶段 PGO+LTO，实测 configure 44 s + make 813 s）
cd $EXP && for a in A B C; do
  sudo -n docker run --rm --cpuset-cpus 200-239 -v $PWD:/work --entrypoint bash $IMG \
    -c "WORK=/work JOBS=16 bash /work/scripts/build_arm.sh $a"; done
#     （C 臂耗时长，建议把它放到 216-239 单独跑，把 200-203 留给基准：
#      `sudo docker update --cpuset-cpus 216-239 <容器名>`）

# 1) 分派路径判定（§2.1）
for f in refimg/image-libpython3.12.so.1.0 build-arm{A,B}/Python-3.12.13/libpython3.12.so.1.0; do
  echo -n "$f: "; objdump -d --disassemble=_PyEval_EvalFrameDefault $f \
    | grep -E "^[[:space:]]+[0-9a-f]+:" | grep -cE "\bbr[[:space:]]+x[0-9]"; done

# 2) 微基准 + PMU（§3/§4）
ARMS="A B C" bash scripts/run_all_benches.sh
BENCH=dispatch ROUNDS=7 bash scripts/run_bench_stock_image.sh   # 镜像对照
BENCH=pi       ROUNDS=5 bash scripts/run_bench_stock_image.sh

# 3) 真实负载（§5）
ARMS="stock A B C" REPS=3 bash scripts/run_harness_repeats.sh
```

产物：

| 路径 | 内容 |
|---|---|
| `data/cpython-build-exp/scripts/` | 全部脚本（构建 / 两个基准 / PMU / harness / 汇总） |
| `data/cpython-build-exp/data/` | 三臂 `sysconfig` 快照、基准原始 JSON / CSV、perf 解析结果 |
| `data/cpython-build-exp/logs/` | configure / make 全量日志、`*.meta.txt`（编译器、时间、cpuset、镜像 digest） |
| `a3-22:.../exp-cpython/{build-armA,B,C,logs,data,refimg}/` | 构建树、原始数据、镜像原版 libpython 参考副本 |
