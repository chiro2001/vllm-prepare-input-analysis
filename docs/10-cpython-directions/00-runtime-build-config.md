# 运行时构建配置：量化基线、验收判据与剩余杠杆

> **状态：已知项，内部已在推进**（2026-09-24 用户确认：「生产镜像中的 cpython 做已知的
> 编译优化，这部分我们已经在推动了」）。
>
> 因此本文**不作为"新发现"**，而定位为两件事：
> 1. **量化基线**：给出这个镜像的构建实况 + 一个可复现的 A/B/C 测量方法，
>    让"优化了多少"有数字可依（避免只有定性结论）；
> 2. **剩余杠杆**：构建参数解决之后，`prepare_input` 还差多少、下一刀砍哪里（§5）。
>
> 全部构建事实基于**在 a3-22 上对镜像实物的直接查询**，可复现。

> ### ⚠️ 第二轮修正（2026-09-24，由 `cpython_build_exp` 真机实验回填）
>
> **本文 §0 / §1.1 / §1.2 / §2 / §4 的"解释器走 `switch` 分派"这一判定已被推翻。**
> 镜像里**实际被链接的 `libpython3.12.so.1.0` 是 computed gotos 构建**：
>
> | 证据 | 镜像 stock | `--without-computed-gotos` | `--with-computed-gotos` |
> |---|---|---|---|
> | `opcode_targets`（256 × 8 B 指针表）符号 | **有** | **无** | 有 |
> | `_PyEval_EvalFrameDefault` 内 `br xN` 条数 | **253** | **1** | 253 |
> | 该函数体积 / 指令数 | **46268 B / 11567** | 44072 B / 11018 | 46268 B / 11567 |
> | 分派指令形态 | `ldr x0,[x1,w8,sxtw#3]; br x0`（每 opcode 一处） | `adr/add(sxth#2)/br x0`（4 B 偏移 Jump Table，全局 1 处） | 同镜像 |
> | 行为：`pi_sim_bench`（a3-22, 7 轮交错中位） | **242222 ns/step** | 257573 ns/step | 235489 ns/step |
>
> **原因**：`sysconfig` / `pyconfig.h` 描述的是**安装面**，不是当前这个 `.so`；
> 二者在镜像里不一致。判构建实况必须看二进制或行为。
>
> **对本文的净影响**：
> ① §0/§2 的"`ptag_stall 69.5%` 与 `switch` 一致"**不成立**——生产没有 `switch`；
> ② 实测 computed gotos 相对 `switch` 的收益是 **−4.7 ~ −7.8%（本负载形状，两次会话）/
>    −7.6 ~ −11.8%（分派密集核）**，
>    但**这部分已经在生产里兑现**，所以"再加 `--with-computed-gotos`"的增量是 **0**；
> ③ 该收益的机制是**指令数 −5.1%**，**不是**分支预测改善（IPC 只 +1.6%，branch-miss 率
>    1.177%→1.134%，只降 0.043 pp）——**这否证了 §2 的机制解释**；
> ④ **§3 的"没有 PGO/LTO"仍然成立**，这是构建侧**唯一剩下的杠杆**，且它的实测增量**远大于**
>    本文原先假设的 10~20%：
>    * 本负载形状微基准：**C/B = −22.6%**（11 轮中位；复现会话 −23.5%）
>    * 真实 `prepare_input` 代码路径（无卡 harness，`--preset realmachine --batch 1 --isl 128`）：
>      **`prepare_inputs_us` 464.7 → 392.3 µs，−15.6%**（stock n=5 vs C n=3，逐次交错）
>    * 机制：`instructions −12.2%` **且** `IPC +14.2%`，`branch-misses −21.2%`、
>      `iTLB-load-misses −63.6%` —— 是**前端压力的全面下降**，正好对应我们的 frontend-bound 画像。
>    * 代价：构建时间 **27 s → 813 s（≈30×）**，`-j16`，同一镜像内。
>
> 完整实验见 [`06-build-config-experiment.md`](06-build-config-experiment.md)。

---

## 0. 一句话结论

> ~~**实际跑 vLLM 的那个 Python（镜像内 `/usr/local/python3.12.13`）是用"默认参数"从源码编的：
> 既没有 PGO/LTO，也没有开启 computed gotos —— 解释器走 `switch` 分派。**~~
> **【已被实测推翻，见文首第二轮修正】** 实测：**分派已经是 computed gotos**（二进制 + 行为双证），
> **只有 PGO/LTO 确实没开**。正确的结论是：
>
> **实际跑 vLLM 的那个 Python 只缺 PGO/LTO；分派方式（computed gotos）没有被漏掉。**

（以下为原始文本，保留以供对照；其"`switch` 分派"部分**不成立**。）

而我们的负载画像（`frontend_bound 66%` + `ptag_stall 69.5%` + `_PyEval_EvalFrameDefault` 第一）
~~**与这种构建下的典型表现一致**。~~ —— 实测：A→B（`switch`→computed gotos）**不改变**
branch-miss 率（1.2708% → 1.2557%）与 IPC（2.314 → 2.316），**不支持**这条机制归因。

> **这个诊断已由内部推进**。本文余下部分的用途是：
> **① 提供一份可复现的量化方法与预期量级**（用于验收"优化生效了没有、生效了多少"）；
> **② 分析构建优化之后的剩余缺口**（§5）——因为即使构建优化全额兑现，
> `prepare_input` 仍在 1 ms 预算之外。

---

## 1. 事实（已核实，可复现）

### 1.1 镜像 Python 的构建状态

```bash
$ ssh a3-22 'docker run --rm --entrypoint python3 \
    quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler -c "
import sysconfig, sys
print(sys.version)
print(\"CONFIG_ARGS:\", sysconfig.get_config_var(\"CONFIG_ARGS\"))
print(\"OPT:\", sysconfig.get_config_var(\"OPT\"))
print(\"HAVE_COMPUTED_GOTOS:\", sysconfig.get_config_var(\"HAVE_COMPUTED_GOTOS\"))
print(\"USE_COMPUTED_GOTOS:\", sysconfig.get_config_var(\"USE_COMPUTED_GOTOS\"))
"'

3.12.13 (main, Aug  3 2026, 02:43:47) [GCC 12.3.1 (openEuler 12.3.1-105.oe2403sp3)]
CONFIG_ARGS: '--enable-shared' 'LDFLAGS=-Wl,-rpath /usr/local/python3.12.13/lib' '--prefix=/usr/local/python3.12.13'
OPT: -DNDEBUG -g -O3 -Wall
HAVE_COMPUTED_GOTOS: 1        ← 平台/编译器支持
USE_COMPUTED_GOTOS: 0         ← 但没启用
```

镜像内头文件是决定性证据：

```bash
$ grep -n "COMPUTED_GOTOS" /usr/local/python3.12.13/include/python3.12/pyconfig.h
161:#define HAVE_COMPUTED_GOTOS 1     ← 编译器支持
1710:/* #undef USE_COMPUTED_GOTOS */   ← 分派方式：关闭
```

### 1.2 这意味着什么：解释器走 `switch` 而不是 computed gotos

> **【第二轮修正】本节结论被推翻。** 下面的源码引用（`configure.ac` 不传参就不定义
> `USE_COMPUTED_GOTOS`）本身**没错**，但"镜像的 `.so` 是按这份 `pyconfig.h` 编出来的"这一步
> **错了**：镜像实际加载的 `libpython3.12.so.1.0` 里存在 `opcode_targets`（256×8 B 指针表），
> `_PyEval_EvalFrameDefault` 内有 **253** 条 `br xN`（每个 opcode 一处），
> 与 `--with-computed-gotos` 臂**逐字节同尺寸**（46268 B / `.text` 0x24dfcc）。
> **保持本节原文，仅为记录推理链在哪里断了。**

`Python/ceval.c`（v3.12.13 源码，已核对）：

```c
    {
    /* Start instructions */
#if !USE_COMPUTED_GOTOS
    dispatch_opcode:
        switch (opcode)
#endif
        {
#include "generated_cases.c.h"
```

`USE_COMPUTED_GOTOS` 未定义 ⇒ 预处理为 0 ⇒ `#if !0` 成立 ⇒ **编译出的是
`dispatch_opcode: switch (opcode)`**，即"**单条间接跳转、数百个目标**"的分派。

而 CPython 的 `configure.ac`（v3.12.13，第 6716–6738 行）里，`USE_COMPUTED_GOTOS`
**只在显式传参时才定义**：

```m4
AC_ARG_WITH([computed-gotos],
  [AS_HELP_STRING([--with-computed-gotos],
                  [enable computed gotos in evaluation loop (enabled by default on supported compilers)])],
[
if test "$withval" = yes; then AC_DEFINE([USE_COMPUTED_GOTOS], [1], ...) fi
if test "$withval" = no;  then AC_DEFINE([USE_COMPUTED_GOTOS], [0], ...) fi
],
[AC_MSG_RESULT([no value specified])])      ← 不传 = 什么都不定义 = 关闭
```

> ⚠️ 注意 help 文本写的是 "**enabled by default** on supported compilers"，
> 但实现里**不传参就是不定义**（= 关闭）。help 文本与实现不一致。
> **→ 这是上游的默认行为，不是这个镜像独有的失误。**
> （但后果由所有"默认参数构建 Python"的部署承担。）

### 1.3 对照组：同一台机器的另一个 Python 是开启的

```bash
$ ssh a3-22 '/usr/bin/python3 -c "
import sysconfig, sys; print(sys.version.split()[0])
print(\"HAVE_CG:\", sysconfig.get_config_var(\"HAVE_COMPUTED_GOTOS\"),
      \" USE_CG:\", sysconfig.get_config_var(\"USE_COMPUTED_GOTOS\"))"'
3.11.6
HAVE_CG: 1   USE_CG: 1        ← openEuler 系统 Python：开启
```

| Python 构建 | `HAVE_COMPUTED_GOTOS` | `USE_COMPUTED_GOTOS` | 分派方式 |
|---|---|---|---|
| **vLLM 镜像 3.12.13**（**实际跑 vLLM 的**） | 1 | **0 / 未定义** | **`switch`** |
| a3-22 系统 3.11.6（openEuler） | 1 | **1** | computed gotos |
| 开发机 conda 3.12.10 | — | （`CONFIG_ARGS` 显式含 `--with-computed-gotos`） | computed gotos |

⇒ **同机器、同 CPU 上，系统 Python 走 computed gotos，而 vLLM 用的那个 Python 不走。**
这不是"平台不支持"（`HAVE=1`），是构建选择。

---

## 2. 为什么这条线索与我们的负载画像高度契合

> **【第二轮修正】本节整节的机制解释已被实测否证，请勿引用。**
>
> 两点原因：
> 1. **前提不成立**：生产镜像实际加载的 `libpython3.12.so.1.0` 是 computed gotos（见文首修正表），
>    不存在"单条间接跳转打数百目标"的 `switch`。
> 2. **替代实验也不支持**：把 `switch` 换成 computed gotos（A→B，7 轮交错、同核同负载）
>    实测 **branch-miss 率 1.2708% → 1.2557%**（△0.015 pp，小于不确定度）、
>    **IPC 2.3144 → 2.3161**（△+0.07%），而 `instructions −5.0%`。
>    ⇒ 该优化的收益机制是**少执行指令**，与分支预测/BTB 无关。
>
> 下表原文保留，仅作为"曾经的假设"存档。

我们的真机画像（`docs/05-hotspots.md`）：

| 指标 | 实测值 | 与 `switch` 分派的关系 |
|---|---|---|
| `frontend_bound` | **66.01%**（其中 `frontend_latency_bound` **59.89%**） | 前端取指/解码受限——分派跳转的目标不确定正是典型来源 |
| `ptag_stall_pct` | **69.52%** | **PTAG（predict tag）是分支预测器资源的停顿**；单条间接跳转打数百个目标会打爆 BTB/BHT |
| `mapq_stall_pct` | 24.47% | 指令队列停顿（次级） |
| `IPC` | **0.771** | 低 IPC + 前端受限 = 典型"分派型"解释器 |
| 火焰图 top-1 | `_PyEval_EvalFrameDefault` | 就是那个 `switch` 所在的函数 |

**机制解释**（教科书级，且与上面数据自洽）：

- `switch` 分派 = **一条**间接跳转指令，运行时目标在数百个 case 之间跳；
  现代 CPU 的间接跳转预测器对"单点多数目标"表现差 ⇒ 高误预测 / 高 BTB 压力 ⇒ **前端停顿**。
- computed gotos（线程化代码）= **每个 opcode 一条**间接跳转，位置固定，
  预测器可以**逐 opcode 学习**目标 ⇒ 误预测显著下降。

> ⚠️ **但必须说清楚**：以上是**机制上的一致性**，**不等于实测因果**。
> 我们的 `ptag_stall` 也可能来自别处（例如 CPython 自身的分支、Python 层的 if/elif）。
> **所以这条线索必须用实验确认**——实验已启动，见 §4。

---

## 3. 另一个构建事实：没有 PGO / LTO

`--enable-optimizations` 在 CPython 里的**官方默认值是 no**（`configure.ac:1756–1769`）：

```m4
AC_ARG_ENABLE([optimizations], AS_HELP_STRING(
                [--enable-optimizations],
                [enable expensive, stable optimizations (PGO, etc.) (default is no)]),
```

镜像的 `CONFIG_ARGS` 里既没有 `--enable-optimizations` 也没有 `--with-lto`
（`OPT: -DNDEBUG -g -O3 -Wall` 是 configure 在不启用 PGO 时的默认值）。
⇒ **生产 Python 没有 PGO、没有 LTO。**

PGO 的意义在这里格外直接：PGO 的核心作用之一就是**按真实执行频率重排分支与热/冷代码布局**，
对"前端受限 + 分支密集"的解释器循环收益最大。额外好处：`--enable-optimizations`
在 GCC 下还会自动加 `-fno-semantic-interposition`（`configure.ac:1779–1786`），
这在我们这个"`--enable-shared` + 大量跨模块调用"的场景里也有意义。

---

## 4. 实验（已启动）：用于量化"生效了多少"，而非"是否值得做"

既然构建优化已由内部推进，本实验的用途就变成**提供 aarch64 + 本负载形状下的量化数字**，
用于验收与回归对比（避免只有"应该会更快"的定性结论）。三个 arm：

| arm | configure | 隔离的变量 |
|---|---|---|
| **A** 基线 | `--without-computed-gotos` | ~~复现镜像现状~~ **【修正】A 是 `switch`，不是镜像现状；镜像是 computed gotos** |
| **B** 仅换分派 | `--with-computed-gotos` | **computed gotos 的净效应** |
| **C** 完整推荐 | `--enable-optimizations --with-lto --with-computed-gotos` | 构建参数总收益（**实测：本负载形状 −22.6%，真实 `prepare_inputs_us` −15.6%**） |

| arm | 构建耗时（实测） | `pi_sim_bench` ns/step | 相对 A | 相对 B | `prepare_inputs_us` p50 | 相对 stock |
|---|---:|---:|---:|---:|---:|---:|
| A（switch） | configure 17 s + make **27 s** | 256266 | 1.0000× | 1.0840× | 464.8（n=2） | +0.01% |
| B（computed gotos，**= 镜像现状**） | configure 17 s + make **27 s** | 236414 | 0.9225×（−7.8%） | 1.0000× | 451.6（n=2） | −2.83% |
| **C（PGO+LTO+CG）** | configure 44 s + make **813 s** | **182984** | **0.7140×（−28.6%）** | **0.7740×（−22.6%）** | **392.3（n=3）** | **−15.59%** |
| 镜像 stock（实测对照） | — | 242222 | 0.9404× | 1.0286× | **464.7（n=5）** | 1.0000× |

三者**同版本（3.12.13）、同编译器（GCC 12.3.1）、同机器**，跑同一个"属性查找密集 +
短调用 + 小对象 churn + 小 numpy 算术"的微基准，并采 `perf stat` 的 IPC 与 branch-misses。

> **三个判据**（都能直接用于验收）：
> 1. **B 臂 vs A 臂**的 `branch-misses` 与 IPC 变化 → ~~验证"`ptag_stall 69.5%` 是否主要来自
>    解释器分派"~~ **【已完成，答案是否】**：`instructions −5.0%`、IPC **不变**、
>    branch-miss 率 **不变** ⇒ 加速来自"少执行指令"，不是分支预测。
> 2. **C 臂 vs A 臂**的端到端增益 → 给出构建优化的**总收益上界**，
>    直接回答"构建优化够不够"（见 §5.2 的缺口算术）。
> 3. ~~**A 臂复现度**~~ **【已完成】**：判据本身要拆开——(a) A 臂**不是**镜像等价物（镜像是 CG）；
>    (b) 微基准 IPC 2.31，**不可能**也不应该等于 0.771；(c) 无卡 harness 的 IPC **0.949**
>    与真机 **0.719–0.890** 属同一区制 ⇒ **0.771 的量级可复现，但"来自 `switch`"的归因不成立**。
>
> 若三臂差异 <3%：那本身是重要结论——说明现代 CPU 的间接预测器已吃掉
> computed gotos 的历史优势，构建优化的收益主要来自 PGO/LTO 的代码布局而非分派方式，
> **§5.3 的代码级杠杆优先级相应提高**。

**结果写在 `06-build-config-experiment.md`。**

---

## 5. 构建优化的边界与"之后还差多少"

### 5.1 工程属性（已由内部推进，此处仅备查）

| 问题 | 答案 |
|---|---|
| 要改多少代码？ | **0 行**（不改 vLLM、不改 Python，只改 Python 的构建参数） |
| 要改什么？ | 镜像构建时给 Python 加 `--enable-optimizations --with-lto --with-computed-gotos` |
| ABI 兼容吗？ | 同版本、同 ABI（`--enable-shared` 保持一致）⇒ **不需要重编 torch/vllm 的任何 C 扩展** |
| 代价 | 构建时间显著增加（PGO 要跑训练集；LTO 拉长链接）；`--with-computed-gotos` 会增加解释器代码体积 |
| 风险 | PGO 训练集是 CPython 自带 benchmark，**与我们的负载分布不同**；LTO 可能暴露第三方扩展的符号问题（需回归测试） |
| 怎么验证 | 用我们已有的无卡 harness（`harness/scripts/pi-docker.sh --preset realmachine`）跑 A/B，看 `prepare_inputs_us` |
| 受益范围 | **所有**在解释器里花时间的 Python 负载（不只 `prepare_input`） |

### 5.2 ⭐ 关键：构建优化能解决多少？剩余缺口在哪？

**这是本文对"下一步"最有用的部分。**

构建优化作用在**解释器派发**这一段。而我们在 `docs/05-hotspots.md` 实测的
`prepare_input` 内部构成（2.76 ms / 步，B=1）里，**只有一部分是解释器派发**：

| 构成 | 实测 | 构建优化能否作用 |
|---|---:|---|
| `AscendGDNAttentionMetadataBuilder.build` × 3 | 908 µs | **部分**（其 Python 控制流与属性查找） |
| 其余 ~28 个子 scope（slot_mapping / positions / update_states / …） | ~1.2 ms | **部分** |
| `_build_attention_metadata` 中的 full-attn builder | 124 µs | 部分 |
| **device 派发本身**（每次小算子 3–12 µs，`eventfd_write`/`pthread_mutex_lock` 也在这一层） | 含在上面各项内 | **❌ 不能**（这是 CANN/驱动的开销，与 Python 构建无关） |
| host 侧对象分配 / dict / 记账 | 含在上面各项内 | **部分**（PGO 可改善，但对象数不变） |

**缺口算术**：

> **【第二轮修正】原来的"10~20% 解释器段收益"这一假设必须拆开——它的绝大部分已经不在桌上了。**
>
> 原本的 10~20% 假设把两种东西混在一起：
> **(i) computed gotos** 与 **(ii) PGO+LTO**。实测结果是：
>
> | 项 | 实测 | 对生产的增量 |
> |---|---|---|
> | computed gotos（`switch` → CG） | 本负载形状 **−6.7%**、分派密集核 **−8.8~−11.8%** | **0** —— 生产**已经是** computed gotos |
> | PGO+LTO（CG → PGO+LTO+CG） | 本负载形状 **−22.6%**（11 轮中位；复现会话 −23.5%）、真实 `prepare_inputs_us` **−15.6%** | **这就是构建侧剩余的全部** |
>
> 也就是说：原文"构建优化 10~20%"这个区间，**方向对、结构错、量级也低估了**：
> * **结构错**：区间里属于分派的那部分**已经在线上跑着了**（生产已是 computed gotos）；
> * **量级低估**：剩下真正能拿的 PGO+LTO，在**本负载形状**上是 **−22.6%**，
>   在**真实 `prepare_input` 路径**上是 **−15.6%**，不是"10~20% 里再切一半"。
>
> **机制**：`instructions −12.2%` **且** `IPC +14.2%`，`branch-misses −21.2%`、
> `iTLB-load-misses −63.6%`、`L1-icache-load-misses −31.9%` —— 前端压力全面下降，
> 正好对应我们的 `frontend_bound 66%` 画像。
> 代价只有**构建时间**：同一镜像内 `make -j16` 从 **27 s → 813 s（≈30×）**，运行期零代价、零代码改动。

```
现状（生产，已含 computed gotos）      2.76 ms
PGO+LTO 实测增量（同一 harness 口径）   −15.6%
                                    ──────────────
构建优化后的落点                     ≈ 2.33 ms（2.76 ms × 0.844）
TPOT=1 ms 时 prepare_input 的预算     ≤ 0.55 ms（留一半给 device）
                                    ──────────────
仍差                                ≈ 4.2×（构建优化改变不了数量级）
```

> **推论（对本项目最重要的一条）**：PGO+LTO 是构建侧**唯一**还能拿的一刀，
> 按 harness 口径值 **−0.43 ms/步**，**与 §5.3 里最大的代码级单项（GDN 去重 260–600 µs）同量级甚至更大**，
> 且**改动 0 行应用代码、不占 chip3、可与代码优化并行**。
> 不要因为"构建优化是已知项、内部已在推进"就把它排在代码改动之后。

⇒ **构建优化是"必要条件"，但离达标还差 4–5 倍。**
它必须与下面 §5.3 的代码级杠杆**叠加**才可能接近 1 ms。

> 注：10–20% 是区间估计，**真实值由正在跑的 A/B/C 实验给出**
>（`06-build-config-experiment.md`）。若实测只有 3–5%，则构建优化对
> `prepare_input` 的贡献更小，§5.3 的优先级相应更高。

### 5.3 构建优化之后的剩余杠杆（按"每 µs 的落地成本"排序）

| 优先级 | 杠杆 | 预期收益 | 作用层 | 为什么排这个位置 |
|---|---|---:|---|---|
| **P0** | **`.replace()` 缓存击穿修复** | **80–200 µs/步** | Python（`gdn_attn_builder.py:541`） | 改动极小、机理已由 perf 子树证据锁定（`PyNumber_Subtract` 占该子树 67.5%） |
| **P0** | **3 组 GDN metadata 去重** | **260–600 µs/步** | Python + 架构 | **上游已有两个 open PR 可直接参照**（vLLM #52297 / vllm-ascend #16246） |
| P1 | decode-only 快路径（跳过 spec/prefill 分支的构造） | 百 µs 级 | Python | 与 P0 两项叠加，是 T1 阶段的主体 |
| P1 | 减少 torch 算子条数 | 1–3% | Python + CANN | **同时打掉 `eventfd_write`/`pthread_mutex_lock`**（这两项是 CANN per-op 开销，构建优化解决不了） |
| P2 | 属性查找提前 / cache 友好 | 1–2% | Python | 与构建优化部分重叠（都是改善 `_PyType_Lookup` 那段），实施时要避免重复计收益 |
| **P3** | **架构级**：账本迁 device、scheduler 直接产出定长张量、metadata 路径 C++/编译化 | 落点 **≈150–400 µs** | 架构 | 要到 ≤550 µs **必须做这一层**，因为前四行合计后仍 >1 ms |

**结论**：

> 构建优化解决的是"**解释器跑得有多快**"；
> 而 `prepare_input` 的大头是"**要做多少事**"（3 次重复构造、每步新建对象、几十次 device 派发）。
> **后者不能靠构建参数解决。** 两条线必须并行推进。

---

## 6. 方法论备注（供其它项目复用）

在"找论文"之外，**运行时自身的构建配置**是一条常被跳过的检查：
`sysconfig.get_config_var(...)` 与 `pyconfig.h` 就是它的自白书。
本项目里，镜像 3.12.13 的 `USE_COMPUTED_GOTOS` 未定义、且无 `--enable-optimizations`
——这些都能在 5 分钟内查到，而它们影响的正是"解释器派发密集"型负载最贵的那一段。

**建议的检查顺序**（对任何"Python 占大头"的性能问题都适用）：

1. `sysconfig.get_config_var("CONFIG_ARGS")` —— 构建时到底传了什么；
2. `HAVE_COMPUTED_GOTOS` / `USE_COMPUTED_GOTOS` —— 分派方式；
3. `OPT` / `CFLAGS` —— 优化级别与 PGO/LTO；
4. 与**同机器上另一个 Python**（如系统 Python）对照 —— 快速识别"是不是构建差异"。

本文 §1.3 的那个对照（同机系统 Python `USE_CG=1` vs 镜像 `USE_CG=0`）就是最省事的判据。

> **【第二轮修正 · 这一节要改】上面 1–3 步都有可能撒谎。** 本项目就是反例：
> 同一个镜像里 `CONFIG_ARGS` / `pyconfig.h` 都说"没有 computed gotos"，而**实际被链接的
> `.so` 是 computed gotos**。修正后的顺序：
>
> 1. `sysconfig` / `pyconfig.h` —— **先看，但只当作"安装面的自述"，不作为结论**；
> 2. **看二进制**（决定性）：数 `_PyEval_EvalFrameDefault` 的 `br xN` 条数，
>    并找 `opcode_targets` 符号（256 × 8 B 指针表 = computed gotos；
>    4 B 偏移 Jump Table + 全局单条 `br` = `switch`）；
> 3. **看行为**（最省事也最难伪造）：拿两个分派密集的纯 Python 循环，
>    分别跑生产解释器与你自己按 `--with/--without-computed-gotos` 编的对照，
>    看生产落在哪一边；
> 4. `OPT` / `CFLAGS` 判 PGO/LTO（这一项**没有**"安装面撒谎"的先例，但仍应交叉验证：
>    PGO 会让 `.text` 布局与冷热分离明显不同）。

---

## 附：本文证据的复现命令

```bash
# 1. 镜像 Python 的构建状态
ssh a3-22 'docker run --rm --entrypoint python3 \
  quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler -c "
import sysconfig,sys
print(sys.version); print(sysconfig.get_config_var(\"CONFIG_ARGS\"))
for k in (\"HAVE_COMPUTED_GOTOS\",\"USE_COMPUTED_GOTOS\",\"OPT\"): print(k, sysconfig.get_config_var(k))"'

# 2. 头文件层面的决定性证据
ssh a3-22 'docker run --rm --entrypoint bash \
  quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler -c \
  "grep -n COMPUTED_GOTOS /usr/local/python3.12.13/include/python3.12/pyconfig.h"'

# 3. 同一台机器上系统 Python 的对照
ssh a3-22 '/usr/bin/python3 -c "
import sysconfig; print(sysconfig.get_config_var(\"USE_COMPUTED_GOTOS\"))"'

# 4. 源码层面的分派方式
curl -s https://raw.githubusercontent.com/python/cpython/v3.12.13/Python/ceval.c | sed -n '840,850p'
curl -s https://raw.githubusercontent.com/python/cpython/v3.12.13/configure.ac | sed -n '6716,6740p'
```
