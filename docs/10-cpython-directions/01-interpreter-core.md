# 01 · CPython 解释器核心方向调研：哪些能真正减少"派发/属性查找"开销

> 调研对象：vLLM 0.26.0（`568afb3a1`）+ vllm-ascend 0.26.0rc1（`f2f74a16c`），
> Kunpeng 920B（aarch64）+ Ascend A3，容器内 **CPython 3.12.13**。
> 调研方式：**纯文献 / 网络 + 上游源码核查**（未上机、未跑 NPU、未改任何远端）。
> 所有外链均给 URL + 标题 + 时间，**访问日期 2026-09-24**（Asia/Shanghai）。
> 上游源码引用一律给 tag/revision + 文件:行号；本项目数字引用一律给 `docs/xx` 出处。
>
> 姊妹篇：[`03-torch-dispatch.md`](03-torch-dispatch.md)（torch/LAL 层）、
> [`05-graph-dispatch-host-side.md`](05-graph-dispatch-host-side.md)（图下发）。
> 本篇只回答一个问题：**在"解释器派发密集"这个已定性的方向上，CPython 侧还有没有可拿的钱。**

---

## 0. 先说结论

**一句话**：在我们当前的 **3.12.13** 上，**没有** tier-2、**没有** JIT、**没有** tail-call 解释器可开
（这三样在 3.12 里根本不存在，见 §3 T-06）；唯一"零代码改动、可能立刻见效"的解释器核心杠杆是
**构建期的 PGO+LTO**；其余所有有公开收益数字的手段（tail-call 3–5%、3.15 JIT 4–12%、
superinstruction 1.7–2%）都要求**换 CPython 版本**，而换版本要重建 vllm-ascend / torch-npu
的 cp312 ABI 全链路 —— 因此它们的**性价比低于**我们已识别的 `.replace()` 缓存击穿修复（−130 µs/步）。

把结论拆成 6 条：

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | **专门化(IC) 是"站点级"的，不是"对象级"的。** 同一个字节码站点被同一类型反复执行就会命中；3.12 把触发阈值从 53 次降到 **2 次**，所以"每个对象只访问一次属性"**不构成障碍**；真正的杀手是**站点多态**（同站点多种类型）与**类型版本失效**。 | **强**（PEP 659 + 3.12.13 源码 + 本机复现） |
| 2 | **tier-2 / JIT 只从"热回边"进入**：`JUMP_BACKWARD`（循环）或 `RESUME`（递归），阈值是**第 4096 次**。**没有 Python 层热循环的直线代码永远不会被 trace，也就永远不会被 JIT**。这一条直接决定"我们的 load 能不能吃到 JIT"。 | **强**（上游设计文档 + issue 中的阈值常量） |
| 3 | 我们这个负载里**大量属性查找根本走不到 IC**：C 层 `getattr()`/`PyObject_GetAttr` 不参与字节码专门化，而 `dataclasses.replace()` **每个字段一次 `getattr(obj, f.name)`**（源码核实）。这给"`.replace()` 三连"又加了一层机制解释。 | **强**（机制，源码级）／收益量级**弱** |
| 4 | **tail-call 解释器在 aarch64 上有公开数字，但比最初宣传小一个数量级**：9.2% → 官方更正 **3–5%**（pyperformance geomean，Clang 19 + PGO + ThinLTO，Neoverse N1 类机器）。 | **中**（上游 PR + 作者更正 + 第三方独立复测） |
| 5 | **"换到 3.15" 的默认解释器收益在 aarch64 Linux 上只有 ~3.6%**（3.15.0a0 vs 3.12.0，pyperformance geomean，arminc aarch64）。JIT 在同类机器上到 2026-06 才 ~7.3%。 | **中**（faster-cpython 官方 benchmark 仓库 + PEP 836 附录） |
| 6 | **`object.h:646` 那 16% 不是"可以优化的低效实现"**：3.12 在 64 位上 `Py_INCREF` 已经是"饱和加法 + 单次 store"，line 646 就是那条 store 指令。GIL 构建下没有更便宜的实现；immortal objects(PEP 683) 实测 **≈ 中性**（曾 1.02–1.03× 变慢）。⇒ 唯一出路仍是**减少对象/引用流量**。 | **中**（源码确定 + PEP 683 实测数字） |

---

## 1. 判读框架：一件优化能不能落到我们这个负载上

### 1.1 我们的负载画像（全部引用本项目已交付数字）

| 事实 | 数值 | 出处 |
|---|---|---|
| `prepare_input` 占单步 | **55.3%**（2757.6 µs / 4915 µs，B=1，pystack off） | `docs/00-INDEX.md` §2.1 |
| topdown | frontend_bound **66.01%**、retiring 12.85%、IPC **0.771** | `docs/00-INDEX.md` §2 |
| 火焰图扁平度 | top-1 `_PyEval_EvalFrameDefault` **10.35%**，top-10 合计 22.83% | `docs/00-INDEX.md` §2 |
| 属性查找类符号 | `_PyType_Lookup` 1.67% + `_PyObject_GenericGetAttrWithDict` 1.36% ≈ **3.03%** | `docs/05-hotspots.md` §3 |
| 调用类符号 | `_PyFunction_Vectorcall` 0.95% + `initialize_locals` 0.89% ≈ **1.84%** | `docs/05-hotspots.md` §3 |
| 引用计数 | `_PyEval_EvalFrameDefault` self 中 **16.01%** 落在 `object.h:646` | `docs/05-hotspots.md` §3 |
| 头号热点 | `AscendGDNAttentionMetadataBuilder.build` ×3 = **908 µs/步（29.3% of prepare_input）** | `docs/00-INDEX.md` §3 |

> **由本项目数字派生（非新增实测）**：
> - 属性查找通用路径 ≈ 3.03% × 2757.6 µs ≈ **84 µs/步**；
> - 解释器逐条派发（`_PyEval_EvalFrameDefault` self）≈ 10.35% × 2757.6 µs ≈ **285 µs/步**。
> 这两条是"解释器核心能碰到的钱"的**总量上界**（含 C 层调用，不全是字节码派发）。

### 1.2 三条硬判据（后面每条技术都用它们筛）

| 判据 | 内容 | 为什么是硬判据 |
|---|---|---|
| **H1 站点级重复** | 该技术的作用单元是**字节码站点**（同一 `code object` + 同一 offset），不是"对象"；站点必须在稳态下被反复执行 | IC 与 tier-2/JIT 都是站点级的；"每个对象只读一次属性"不影响它们，但"每个站点全进程只执行一次"会 |
| **H2 Python 层热循环** | 该技术要求**同一 Python code object 内有被反复执行的 backedge**（循环）或 `RESUME`（递归） | tier-2/JIT 的入口就是这两条；纯直线、每步只调一次的 Python 函数**不吃 JIT** |
| **H3 不换 ABI** | 该技术是否要求换 CPython 版本（→ 重建 torch-npu / vllm-ascend 的 cp312 扩展） | 换 ABI 的代价落在容器镜像与 CANN 适配上，不是"改个 flag" |

### 1.3 速查表

| 技术 | 公开收益（最好证据） | H1 | H2 | H3 | 我们该做吗 | 证据强度 |
|---|---|---|---|---|---|---|
| 专门化 IC（3.12 已在跑） | 3.11 整体 1.25× vs 3.10；3.12 阈值收紧再 +~1% | ✅ | — | ✅（已在） | **已吃到；先量化别乱猜** | 强 |
| tier-2 微指令 | 3.13 起才有；解释模式下比 tier-1 **慢 ~20%**，JIT 才把它追平 | ✅ | ❌ 需热循环 | ❌ | 不做 | 强 |
| copy-and-patch JIT | 3.13 ≈ 与解释器持平；3.15 报告 4–12%（x86-64/arm64 不一） | ✅ | ❌ 需热循环 | ❌ | 观察，不做 | 中 |
| tail-call 解释器（3.14） | **+3–5%** geomean（aarch64 Python 负载最高 +24%） | ✅ | — | ❌ | 第二梯队候选 | 中 |
| computed goto | 现代 CPU 上只剩 **1–4%** | ✅ | — | ✅（已开） | 无剩余空间 | 强 |
| 静态 superinstructions | **2%**（2021）/ **1.7%**（2023） | ✅ | — | ❌（3.11+ 已有） | 已吃到 | 强 |
| JIT 侧动态 superinstruction | 实验版本**慢 2–5%** | ✅ | ❌ | ❌ | 不做 | 中 |
| `initialize_locals`/调用路径 | 无独立公开数字；3.11 已内联调用帧 | ✅ | — | ❌ | 只能靠减少调用条数 | 弱 |
| PEP 709 内联推导式 | 微基准 **最高 2×**，真实负载样本 **+11%** | ✅ | — | ❌（3.12 已有） | 已吃到 | 强 |
| `dataclasses.replace()` 的隐藏 `getattr` | 无公开数字（本文新提出的机制） | ✅ | — | ✅ | **顺势一起改** | 中（机制强／收益弱） |
| immortal objects (PEP 683) | **≈ 中性**（曾 1.02–1.03× 变慢） | — | — | ❌ | 不做 | 中 |
| free-threading 的延迟引用计数 | 单线程**慢 5–10%** | — | — | ❌ | 不做 | 中 |
| PGO + LTO | 官方"推荐以获得最佳性能"；核心开发者称 PGO、LTO 各 ~10% 量级 | ✅ | — | ✅（同版本重建） | **先查，没开就开** | 中 |

---

## 2. 专门化自适应解释器（PEP 659，Python 3.11+）

### T-01 · 站点级 inline cache：`LOAD_ATTR` 家族

- **来源**：
  [PEP 659 – Specializing Adaptive Interpreter](https://peps.python.org/pep-0659/)（Mark Shannon，2021-01-15 创建，Status: Final）；
  CPython `v3.12.13` 源码 [`Python/specialize.c#L721` `_Py_Specialize_LoadAttr`](https://github.com/python/cpython/blob/v3.12.13/Python/specialize.c#L721)、
  [`Python/bytecodes.c#L1808` `LOAD_ATTR_INSTANCE_VALUE` / `#L1840` `LOAD_ATTR_WITH_HINT`](https://github.com/python/cpython/blob/v3.12.13/Python/bytecodes.c#L1808)。
- **核心机制（一句话）**：每个可专门化指令后面跟一段 inline cache；解释器在第 N 次执行时检查**操作数类型**，
  把 opcode 原地改写成专用变体（如 `LOAD_ATTR_INSTANCE_VALUE`），并把"类型版本号 / 实例 dict 槽位"写进 cache，
  之后每次执行只做一次**守卫比较**就取到值，跳过 `_PyType_Lookup` 与 dict 查找。
- **公开实测收益**：
  Python 3.11 相对 3.10 在 pyperformance 上**平均 1.25×**（官方发布说明；
  [What's New In Python 3.11 §Faster CPython](https://docs.python.org/3.11/whatsnew/3.11.html)，行 61），
  其中专门化是多个机制之一（不是全部）。
- **对我们的适用性**：**已在生效**。3.12.13 上 `LOAD_ATTR` 的变体有 7 个
  （`INSTANCE_VALUE` / `WITH_HINT` / `SLOT` / `CLASS` / `MODULE` / `PROPERTY` / `GETATTRIBUTE_OVERRIDDEN`）
  外加 `LOAD_ATTR_METHOD_*` 家族。**证据强度：强** —— 依据：PEP + 一手源码 + 本文附录 A 的本机复现
  （同一个站点在第 2 次调用后由 `LOAD_ATTR` 变成 `LOAD_ATTR_INSTANCE_VALUE`）。

### T-02 · 触发阈值：3.11 要几十次，3.12 只要 2 次（这条直接回答"每个属性只访问一次"）

- **来源**：
  [gh-98686 "Quicken everything"](https://github.com/python/cpython/issues/98686)（Brandt Bucher，2022-10-25）；
  源码 `v3.12.13` [`Include/internal/pycore_code.h#L406`](https://github.com/python/cpython/blob/v3.12.13/Include/internal/pycore_code.h#L406)。
- **核心机制（一句话）**：解释器里有一个 16-bit 退避计数器，**值 1 表示"第 2 次执行时尝试专门化"**；
  gh-98686 之前（3.11 及 3.12 alpha）的门槛要高得多：code object 先要跑 8 次触发 quickening
  （`QUICKENING_WARMUP_DELAY 8`），之后指令级计数器的初始值是 **31**（`ADAPTIVE_BACKOFF_START 5`），
  即"站点要执行几十次才第一次尝试专门化"；gh-98686 把后者改成 1，官方理由是
  "执行两次比执行一次更能说明热点，而额外的预热只会推迟专门化"。
- **公开实测收益**：该改动（gh-98686/PR #99182，2022-11）自述**一致的 ~1% 提升**，
  并明确说明"我试了 0–4095 各种初始值，**值 1 最好**"。
- **对我们的适用性**：**关键**。我们的 `prepare_input` 每步都会重跑同一批函数，
  站点在第 2 个 step 就具备专门化条件 ⇒ **"每个对象只读一次属性"不是障碍**。
  真正要担心的是下面 T-03 的失效条件。**证据强度：强**（一手 issue + PR + 源码常量 + 本机复现）。

### T-03 · 命中率取决于什么（5 个失效源）

- **来源**：源码 `v3.12.13`
  [`Python/specialize.c`](https://github.com/python/cpython/blob/v3.12.13/Python/specialize.c#L721)
  的 `analyze_descriptor` / `specialize_dict_access` / `_Py_Specialize_LoadAttr` 与
  [`Python/bytecodes.c`](https://github.com/python/cpython/blob/v3.12.13/Python/bytecodes.c#L1840) 的 `DEOPT_IF` 守卫；
  阈值见 [gh-129386](https://github.com/python/cpython/issues/129386)。
- **核心机制（一句话）**：专门化靠"猜 + 守卫"，守卫失败就退回通用路径并进入**退避**（`ADAPTIVE_COOLDOWN_VALUE = 52`，
  即失败后要再执行 53 次才会重新尝试）。
- **命中率的 5 个决定因素**（源码级）：

  | # | 失效源 | 源码依据 | 对我们的含义 |
  |---|---|---|---|
  | 1 | **站点多态**：同一站点先后出现不同类型/布局，守卫失败后退避 52 次 | `DEOPT_IF(...)` → `adaptive_counter_cooldown()` | 同一行代码同时处理多种 vLLM 对象（如 `Union` 分支）时，IC 长期停在通用路径 |
  | 2 | **类型版本失效**：`tp_version_tag` 变化（改类属性 / 加方法 / 改 MRO / monkeypatch） | `DEOPT_IF(tp->tp_version_tag != type_version)` | **`torch.compile`/Dynamo 这类会往类上挂东西的工具会一次性打掉该类型所有 IC** |
  | 3 | **实例 dict 布局变了**：`LOAD_ATTR_WITH_HINT` 守卫是"该槽位的 key 指针等于这个名字" | `DEOPT_IF(ep->me_key != name)` | 仅当实例已从"共享键 values 布局"退化成**真 dict**（例如某处访问过 `obj.__dict__` / `vars()`）时才相关 |
  | 4 | **类型不可专门化**：非 managed-dict 的 C 扩展类型、可变类描述符、非对象槽位的 getset、被覆盖的 `__getattribute__` 等 | `SPEC_FAIL_ATTR_NOT_MANAGED_DICT` / `ATTR_MUTABLE_CLASS` / `GETSET_OVERRIDDEN` 等 | 部分 C 扩展对象（含一些 torch/Ascend 对象）属性访问永远走通用路径 |
  | 5 | **C 层属性访问根本不参与专门化**：`getattr()` 内建、`PyObject_GetAttr`、`operator.attrgetter` 等 | 专门化是**字节码级**机制（PEP 659 §"CALL"/"LOAD_ATTR" 只覆盖 opcode 站点） | 见 T-20：`dataclasses.replace()` 每字段一次 `getattr` |

- **公开实测收益**：**没有"IC 命中率是多少"的公开数字**——上游只通过 `--enable-pystats` 的
  `summarize_stats.py` 报告每个 opcode 的 success/failure **次数**与失败原因，不发布命中率百分比，
  也不按负载形态分类。最接近的可引用数字是 T-02 的 **~1%**（阈值 31/8 → 2）与
  T-13 的 **1.7–2%**（相邻指令融合）。⇒ 命中率只能**自己测**，不能引用。
- **对我们的适用性**：这 5 条是"我们为什么还有 3.03% 花在 `_PyType_Lookup`+`_PyObject_GenericGetAttrWithDict`"的**候选解释清单**，
  而不是结论。⇒ **建议动作**：在无卡 harness 里用 **pystats 构建**（`./configure --enable-pystats`、
  `Tools/scripts/summarize_stats.py`）跑一遍，直接拿到各站点的 `specialization success/failure` 与失败原因计数，
  把上面 5 条从"可能"变成"排好序的名单"。**证据强度：强**（机制与常量来自一手源码；
  "哪一条占我们的大头"**未测**，属待验证）。

### T-04 · 精确回答："每个属性只访问一次"的代码有效吗？

- **来源**：同 T-01/T-02（PEP 659 + `v3.12.13` 源码 + 附录 A 本机复现）。
- **核心机制（一句话）**：IC 的键是"**字节码站点 + 类型版本**"，生命周期与单个对象无关；
  因此判定标准是"**这个站点会不会被反复执行**"，而不是"这个对象会不会被反复访问"。
- **公开实测收益**：**没有**该问题的直接公开数字（§9）；可引用的相邻证据是 T-02 的 ~1%（阈值收紧）
  与 T-03 的五条失效源（机制级）。
- **结论**：**分两种"一次"**。
  - ✅ **每个对象只读一次、但同一站点反复执行**（我们的主形态，例如每步对新建的 metadata 对象取一次属性）：
    **有效**。IC 绑定的是"站点 + 类型版本"，对象可以每次都新建；
    3.12 阈值 = 2 次执行，站点在第 2 步就命中。附录 A 已在本机复现"500 个不同实例、每个只读一次、站点仍专门化"。
  - ❌ **每个站点全进程只执行一次**（模块级/初始化代码、只走一次的分支、`eval`/动态生成的一次性代码）：
    **无效**。它连"执行两次"都不满足，永远停在通用路径。
- **另一个容易踩的坑**：**站点多态**。同一个 `data.x` 站点如果第 1 步遇到 `A`、第 2 步遇到 `B`，
  守卫失败 → 退避 52 次 → 期间全走通用路径；在"属性访问 + 短调用"的负载里，
  多态站点的代价**比非多态站点高一个量级**（每次 miss 还要额外做一次专门化尝试的前置分析）。
- **公开实测收益**：无"我们这个负载"的直接数字（见 §9）。可引用的最接近证据是 gh-98686 的 ~1%（阈值从 52→1）
  与 3.11 整体的 1.25×。
- **对我们的适用性**：**高相关性**；先测多态站点占比，再决定是否值得改代码把 Union 型属性访问拆站。
  **证据强度：中**（机制强 + 本机复现；对我们的收益量级未测）。

### T-05 · `LOAD_ATTR`/`CALL` 的方法调用路径（bound method 被省掉了）

- **来源**：源码 `v3.12.13` [`Python/bytecodes.c#L2563`](https://github.com/python/cpython/blob/v3.12.13/Python/bytecodes.c#L2563)
  （`LOAD_ATTR_METHOD_WITH_VALUES` / `LOAD_ATTR_METHOD_NO_DICT` / `LOAD_ATTR_METHOD_LAZY_DICT`）。
- **核心机制（一句话）**：对 `oparg & 1` 的 `LOAD_ATTR`（即"马上要调用"的形态），
  专门化不构造 bound method 对象，而是压入"函数 + self-or-null"两个值，
  后续 `CALL` 再用 `CALL_NO_KW_METHOD_DESCRIPTOR_*` / `CALL_PY_EXACT_ARGS` 直接建帧。
- **公开实测收益**：无单独公布的百分比；机制收益是**省掉一个 `PyMethod` 对象分配 + 一次引用计数往返**。
- **对我们的适用性**：**已在生效**。本机复现（附录 A）显示 `x.bit_length()` 这种写法一站到位：
  `LOAD_ATTR_METHOD_NO_DICT` + `CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS`。
  但注意：**`getattr(obj, "name")()` 这种动态写法拿不到**（见 T-20）。
  **证据强度：强**（机制）/ **弱**（收益量级）。

---

## 3. Tier 2 / 微指令 / JIT

> 这一节先做一件事实澄清，因为它决定了"我们能不能在 3.12 上开 JIT"这个问题的答案是"没有这个东西"。

### T-06 · 事实澄清：**CPython 3.12 里没有 tier-2，也没有 JIT**

- **来源**：源码核查（本文执行，命令见附录 A）：
  `v3.12.13` 的 `Python/optimizer.c` **不存在**（HTTP 404）；
  `Include/opcode_ids.h` 中 **没有** `ENTER_EXECUTOR`；
  `Python/ceval.c` 中 `_PyOptimizer_Optimize` / `_Py_TIER2` / `GOTO_TIER_TWO` 出现 **0 次**；
  对照 `v3.13.7`：三者分别存在（`ENTER_EXECUTOR` 出现，`ceval.c` 出现 4 次引用）。
  官方口径见 [PEP 744](https://peps.python.org/pep-0744/)：
  "自从 Python **3.13** 周期的早期开始，所有 CPython 构建都包含这套微指令翻译/优化/执行机制。然而它默认是关闭的"。
- **核心机制（一句话）**：3.12 的"快"完全来自 **tier-1 专门化**；tier-2（uop）与 JIT 是 3.13 才进 main 的新管线。
- **公开实测收益**：不适用（不存在）。
- **对我们的适用性**：**排除项**。任何"在 3.12 上设置 `PYTHON_JIT=1` / `-X uops` 试试"的方案都是无效动作；
  3.12 上唯一的解释器核心开关是构建参数（PGO/LTO/BOLT）。**证据强度：强**（源码 + PEP 744 文字）。

### T-07 · 3.13 的 tier-2 与 copy-and-patch JIT（PEP 744）

- **来源**：
  [PEP 744 – JIT Compilation](https://peps.python.org/pep-0744/)（Brandt Bucher，2024-04-11）；
  [What's New In Python 3.13 §An experimental just-in-time (JIT) compiler](https://docs.python.org/3.13/whatsnew/3.13.html)；
  [LWN: Adding a JIT compiler to CPython](https://lwn.net/Articles/977855/)（2024-06-18）。
- **核心机制（一句话）**：tier-1 专门化后的字节码先被"投影"成一段**线性微指令 trace**，
  再做常量折叠/类型传播等优化；这条 trace 要么用 tier-2 解释器跑，要么用 copy-and-patch 编译成机器码跑。
- **公开实测收益（都是"持平"级）**：
  - PEP 744 原文：**"JIT 目前与现有的专门化解释器大致一样快"**，并给出"至少 5%"才算"有意义的提升"的门槛；
    内存多 **10–20%**（aarch64-apple-darwin 偏上界）。
  - 3.13 What's New：JIT **默认关闭**，`--enable-experimental-jit` 的取值 `no/yes/yes-off/interpreter`；
    性能"**modest**（有限）"，"期望后续版本改进"。
  - LWN 记录的 Brandt Bucher 演讲：**tier-2 解释器比字节码解释器慢约 20%**，
    JIT 的价值就是把这份 dispatch 开销追回来（"基本 0% 变慢"）。
  - 微指令本身的开销：Jin 在 PEP 讨论中承认 **1–4% 的性能损失**（来自 micro-op 化），
    再靠优化器找回（[LWN: Python JIT stabilization](https://lwn.net/Articles/970397/)，2024-04-25）。
- **对我们的适用性**：**低**。两个理由：
  (a) 3.13 默认不启用 JIT（连 tier-2 都要 `interpreter` 模式或运行时开关），
  (b) 即使启用，收益在 3.13 上基本为 0。**证据强度：强**（PEP 原文 + 官方发布说明 + LWN 现场记录）。

### T-08 · tier-2/JIT 的**触发条件**（本节最重要的一条）

- **来源**：
  [CPython `InternalDocs/jit.md`](https://github.com/python/cpython/blob/main/InternalDocs/jit.md)（main 分支）；
  [gh-129386](https://github.com/python/cpython/issues/129386) 列出的门槛常量。
- **核心机制（一句话）**：**"程序一直在自适应解释器上跑，直到某条 `JUMP_BACKWARD` 或 `RESUME` 指令通过它 inline cache 里的计数器判定自己'够热'"**，
  然后才进入 trace 记录 → 优化 → 执行/编译；executor 装回 code object，把那条 `JUMP_BACKWARD` 替换成 `ENTER_EXECUTOR`。
- **阈值（一手）**：

  | 计数器 | 值 | 含义 |
  |---|---|---|
  | `ADAPTIVE_WARMUP_VALUE` | 1 | 第 2 次执行尝试 tier-1 专门化 |
  | `ADAPTIVE_COOLDOWN_VALUE` | 52 | 守卫失败后第 53 次重试专门化 |
  | `JUMP_BACKWARD_INITIAL_VALUE` | **4095** | **第 4096 次回跳**才尝试建立一条 root trace |
  | `SIDE_EXIT_INITIAL_VALUE` | **4095** | 第 4096 次守卫失败才建立 side-exit trace |

- **公开实测收益**：**本条不是优化**，因此没有"收益数字"；它给出的是**门槛**。
  与之相关的收益数字见 T-07（3.13 ≈ 持平）、T-09（tail-call 3–5%）、T-10（3.15 JIT 4–12%）。
- **对我们的适用性**：**这是判死刑的一条**。tier-2/JIT 的入口是**热回边**，
  也就是"**同一个 Python code object 内的循环**"。我们的火焰图是"极平 + 大量短函数"，
  热点主体是**每步被调用 3 次**的 builder 与大量一次性小函数；
  它们**要么没有 Python 层循环，要么循环体只迭代 1–8 次**（B=1）。这意味着：
  1. 就算升级到 3.15 并打开 JIT，**能否覆盖 `prepare_input` 取决于那个进程里是否存在够热的 Python 层循环**
     （vLLM 的 engine/worker busy-loop 是候选），而不是取决于"负载是解释器密集的"；
  2. 4096 次回跳的门槛意味着**建立 trace 需要上千个 step 的预热**，短时压测（几十步）根本看不到 JIT 生效；
  3. 这一条同时解释了"为什么 CPython 官方 benchmark 的数字不能外推到我们"：
     pyperformance 的负载大量是纯 Python 循环，而我们是 **Python 调 C/C++ 的薄壳**。

  **证据强度：强**（机制与阈值来自一手设计文档/issue；"我们是否有够热的 Python 循环"**未测**，
  这正是下一节 §8 建议的 5 分钟实验）。

### T-09 · 3.14 的 tail-call 解释器（PEP 744 之后的第一个"换解释器"选项）

- **来源**：
  [CPython PR #128718 / gh-128563 "A new tail-calling interpreter"](https://github.com/python/cpython/pull/128718)（Ken Jin，2025-01）；
  [What's New In Python 3.14 §A new type of interpreter](https://docs.python.org/3.14/whatsnew/3.14.html)；
  作者更正博文 [I'm Sorry for Python's tail-calling Interpreter's Results](https://fidget-spinner.github.io/posts/apology-tail-call.html)（2025-03-08）；
  第三方独立复测 [Nelson Elhage: Performance of the Python 3.14 tail-call interpreter](https://blog.nelhage.com/post/cpython-tail-call/)（2025-03-09）。
- **核心机制（一句话）**：不再用一个巨大的 `switch`/computed goto 主循环，
  而是把**每个 opcode 编译成一个独立 C 函数**，用 `musttail` + `preserve_none` 调用约定做**保证尾调用**串起来；
  好处是每个 opcode 的函数体小、寄存器分配好、跳转目标可预测。
- **公开实测收益（注意数字的"考古"过程）**：

  | 时点/来源 | 数字 | 基线 |
  |---|---|---|
  | PR 最初（2025-01） | aarch64 Linux(N1) **9.2%**、x86-64 7.4%、macOS M1 14.7% | Clang 19（含 LLVM 19 tail-dup 回归） |
  | 同 PR 的**更正声明** | **3–5%** geomean；x86-64 Linux 约 2% | 同一 PR 内作者的 correction notice |
  | Ken Jin 博文（2025-03） | **3–5%** | 承认基线被编译器 bug 拖慢 |
  | Elhage 独立复测 | Raptor Lake：clang19.tc vs clang18 = **1.03×**；Apple M1 = **1.00×**；GCC 14.2 = 1.02× | 全 LTO+PGO |
  | 3.14 官方 What's New（最终） | **"geomean 3–5% 更快"**，且明确"**仅 Clang 19+、仅 x86-64 与 AArch64**"、"**opt-in**"、"**强烈建议 PGO**" | Python 3.14 + Clang 19（无 tail-call） |
  | LWN 记录 EuroPython 2025 | x86-64 Linux 约 **2%**；arm64 macOS **5–7%**（但怀疑该编译器也有同类 bug） | — |

- **对我们的适用性**：**第二梯队候选**（H1✅ H2 无关 H3❌）。
  它是**唯一一个直接针对"前端受限 + 派发密集"的手段**——因为它的作用点正是 `frontend_bound 66%` 指向的东西：
  每字节码的取指/派发代码形状。但：
  (a) 公开收益只有 3–5%（aarch64 上界约 +24% 仅在 Python 密集子基准）；
  (b) 需要 **Clang 19+ 且必须开 PGO**，还要重建整个 Python（→ H3 失败）；
  (c) 我们跑的是 3.12.13 容器镜像，Python 是镜像的一部分。
  **证据强度：中**（上游一手 + 独立第三方，平台是 Neoverse N1 / M1 / Raptor Lake，
  **没有 Kunpeng 920B 的公开数据**，见 §9）。

### T-10 · 3.15 的 JIT：终于有"可引用"的窗口

- **来源**：
  [What's New In Python 3.15](https://docs.python.org/3.15/whatsnew/3.15.html)（截至 2026-09 为 3.15.0rc2 文档）；
  [Python Insider: Python 3.15's JIT is now back on track](https://blog.python.org/2026/03/jit-on-track/)（2026-03-23）；
  [PEP 836 – JIT Go Brrr: The Path to a Supported JIT Compiler for CPython](https://peps.python.org/pep-0836/)；
  [Python 3.15.0 beta 4 公告](https://blog.python.org/2026/07/python-3150-beta-4/)（2026-07-18）。
- **核心机制（一句话）**：新的 tracing 前端 + 基础寄存器分配 + LLVM 21 生成 stencil；
  仍然是"**热 trace**"模型（入口条件同 T-08），只是 trace 质量与代码生成大幅改善。
- **公开实测收益（都是 pyperformance geomean）**：

  | 平台 | 配置 | 数字 | 来源 |
  |---|---|---|---|
  | x86-64 Linux | JIT vs 标准解释器（全优化） | **8–9%** | 3.15 What's New |
  | AArch64 macOS | JIT vs **tail-calling** 解释器 | **12–13%** | 3.15 What's New |
  | AmpereOne Linux aarch64 | JIT | **1.073×（+7.3%）**，窗口 2026-06-16~27 | PEP 836 附录 |
  | M3 Pro macOS | JIT+TAILCALL | 1.126×（+12.6%） | PEP 836 附录 |
  | i5-8400 Linux x86-64 | JIT | 1.069×（+6.9%） | PEP 836 附录 |
  | 官方 Windows 64-bit 二进制 | 默认改用 tail-calling 解释器 | 无 geomean 数字 | 3.15 What's New |

  官方同时给出**范围**：x86-64 Linux 与 AArch64 macOS 上"**约 15% 变慢 ~ 100%+ 变快**（忽略 `unpack_sequence` 微基准）"。
  PEP 836 的目标是到 **3.17** 达到 **≥20%**（JIT + free-threading vs 纯解释器）。
- **对我们的适用性**：**中（有希望但不划算）**。
  aarch64 Linux 的 7.3% 是**当前最好的公开 aarch64 数字**，但：
  (a) 仍受 T-08 的"必须有热 Python 回边"约束；
  (b) JIT 需要 **LLVM 21 构建期依赖**，且官方二进制里仍是 **built but disabled**（`PYTHON_JIT=1` 打开）；
  (c) 换 3.15 = 换 ABI（H3❌），要重建 vllm-ascend/torch-npu/cp312→cp315 扩展。
  **证据强度：中**（官方文档 + PEP 附录；机器是 AmpereOne/M3，**非 Kunpeng**；负载是 pyperformance，**非我们的形态**）。

### T-11 · 数字考古："同一个特性被报成 9–15% → 3–5% → 1–5%"

- **来源**：
  [Ken Jin 的更正博文](https://fidget-spinner.github.io/posts/apology-tail-call.html)（2025-03-08）、
  [Elhage 的复测](https://blog.nelhage.com/post/cpython-tail-call/)（2025-03-09）、
  [LWN: Python, tail calls, and performance](https://lwn.net/Articles/1033373/)（2025-08-20）、
  以及当时流传的二手数字 [Simon Willison（2025-02-13）](https://simonwillison.net/2025/Feb/13/python-3140a5/)（"20–30% improvement"）。
- **核心机制（一句话）**：LLVM 19 的 tail-duplication 阈值改动让**旧 computed-goto 基线**莫名变慢约 10%，
  于是"新解释器 vs 被拖慢的基线"看起来快了 10–15%；换成 GCC 14 / Clang 18 作基线后只剩 1–5%。
  **基础事实**：computed goto 与 switch 在现代 CPU 上的差距本来就只有 **2–4%**
  （Elhage 用 `.nocg` 构建实测 clang18.nocg 比 clang18 **快 1.01×**；参见 T-12）。
- **对我们的适用性**：**这是一条引用纪律**。凡看到"某特性 +10%"，必须先问：
  基线是哪个编译器/哪个版本？是在哪台机器上？负载形态是什么？
  本文 §附录 B 对每条数字都标了"基线"与"机器"，我们建议内部汇报沿用同样口径。
  **证据强度：强**（多方独立复现，且 CPython 作者自己发了更正）。

- **公开实测收益（本条要对照的三个版本）**：同一特性——
  **9–15%**（PR 最初，基线是被 LLVM 19 拖慢的 Clang 19 computed goto）、
  **3–5%**（官方更正与 3.14 What's New）、
  **1–5%**（Elhage 在 GCC 14 / Clang 18 基线上的独立复测）——
  三个数字都是"实测"，差别全部来自**基线**。**证据强度：强**。

---

## 4. Superinstructions / computed goto / 线程化解释器

### T-12 · computed goto（线程化派发）：现代 CPU 上已经没有多少剩余空间

- **来源**：
  Rohou, Swamy, Seznec,
  [Branch Prediction and the Performance of Interpreters – Don't Trust Folklore](https://inria.hal.science/hal-01100647/document)（HAL，2015）；
  [Nelson Elhage 的 `.nocg` 对照实验](https://blog.nelhage.com/post/cpython-tail-call/)（2025-03-09）。
- **核心机制（一句话）**：把"每条指令执行完跳回中心 `switch`"换成"每条指令末尾直接 `goto *dispatch_table[opcode]`"，
  省掉一次回跳与一次重复的间接跳转，同时让编译器把派发代码复制进每个 opcode 体（branch duplication）。
- **公开实测收益**：
  - 论文（Haswell 时代）：Python 解释器平均 **约 21 条指令/字节码**；派发间接跳转的误预测率
    从 Nehalem 的 **12–20 MPKI** 降到 Haswell 的 **0.5–2 MPKI**，
    结论是"**间接分支的可预测性不该再被当作解释器的主要问题**"。
  - Elhage（2025，LTO+PGO）：`clang18.nocg`（关闭 computed goto 的 switch 版）比 `clang18` **快 1.01×**；
    他给出的"复制派发逻辑"合理估计是 **2–4%**。
- **对我们的适用性**：**没有剩余空间**。GCC/Clang 构建默认启用 `USE_COMPUTED_GOTOS`，
  我们已经在这条路上；它的天花板在现代核上就是 1–4% 量级，且已经被 3.11+ 吃掉了。
  **证据强度：强**（第三方论文 + 独立复测 + 上游配置事实）。

### T-13 · 静态 superinstructions（3.11/3.12 已在生效）

- **来源**：
  [bpo-44900: Add five superinstructions（PR #27741）](https://github.com/python/cpython/pull/27741)（Mark Shannon，2021-08-16，merged）；
  [GH-105229: Replace some superinstructions with single instruction equivalent（PR #105230）](https://github.com/python/cpython/pull/105230)（2023-06-02，merged）。
- **核心机制（一句话）**：编译期把频繁相邻的指令对融合成**一条**指令
  （`LOAD_FAST+LOAD_FAST`、`STORE_FAST+LOAD_FAST`、`LOAD_FAST+LOAD_CONST`、`STORE_FAST+STORE_FAST`…），
  少一次派发、少一次 inline-cache 跳转。
  **插入时点随版本变化**（本文源码核查也一并确认）：3.11/3.12 在 **quickening（首次执行）时**融合
  （逻辑在 `Python/specialize.c`）；**3.13 起移到编译期**（`Python/flowgraph.c::insert_superinstructions`，
  该函数在 v3.12.13 中不存在、在 v3.13.7/v3.14.0 中存在）。
  含义：3.12 上"只执行一次"的代码连 superinstruction 都没有；3.13+ 无论执行几次都先融合。
- **公开实测收益**：
  - 2021 五个 superinstruction：**2%**（PR 自述，附 perf 对比 gist）。
  - 2023 用"单指令等价物"替换其中若干（减少代码尺寸与内存读取）：**1.7%**（PR 自述，附 faster-cpython 结果链接）。
- **对我们的适用性**：**已吃到**。本机复现（附录 A）在 3.12.10 上直接看到
  `LOAD_FAST__LOAD_FAST`、`STORE_FAST__LOAD_FAST` 出现在自适应字节码里。
  这一条同时给出一个重要**标尺**：在 CPython 里"减少派发/融合指令"这种方向的公开收益就是 **1–2%** 量级，
  不是 10%+。**证据强度：强**（上游 merged PR + 自述数字 + 本机复现）。

### T-14 · JIT 侧动态 superinstruction：公开的**负结果**

- **来源**：[faster-cpython/ideas#647 "Superinstructions for Copy & Patch JIT"](https://github.com/faster-cpython/ideas/issues/647)（2024-01）。
- **核心机制（一句话）**：在 uop 层统计高频相邻 pair，把它们融合成一条 stencil，减少 JIT 后代码的 uop 条数。
- **公开实测收益**：实验分支的 pyperformance 结果**整体慢约 2–5%**（个别基准变快，`bench_mp_pool` 异常慢 4.77×）；
  作者本人也预期"这些 superinstruction 大概帮助不大"。
- **对我们的适用性**：**不做**。它是"融合派发"思路在 JIT 侧的反例，说明**派发融合不是免费的**：
  更大的 stencil 会带来代码尺寸/I-cache 的代价（与我们的 frontend_bound 直接相关）。
  **证据强度：中**（实验分支，非上游合并；但与我们的 frontend 受限形状相关）。

### T-15 · `ENTER_EXECUTOR`（tier-2 的入口指令）

- **来源**：源码核查（本文执行）：`v3.12.13` 的 `Include/opcode_ids.h` **没有** `ENTER_EXECUTOR`，
  `v3.13.7` **有**；[PEP 744](https://peps.python.org/pep-0744/)、
  [InternalDocs/jit.md](https://github.com/python/cpython/blob/main/InternalDocs/jit.md)。
- **核心机制（一句话）**：当一个循环的 `JUMP_BACKWARD` 够热、且 trace 建立成功时，
  解释器把该循环头部替换成 `ENTER_EXECUTOR`，后续迭代直接进入 executor（uop 解释器或 JIT 机器码）。
- **公开实测收益**：它本身不是优化，而是 tier-2 的**载体**；收益归入 T-07/T-10。
- **对我们的适用性**：**3.12 不存在这条指令** ⇒ "用 `ENTER_EXECUTOR` 调优"在我们当前版本上无从谈起；
  且它替换的仍是**循环头部**，再次印证 T-08 的判据 H2。**证据强度：强**（源码存在性）。

---

## 5. 调用开销：`initialize_locals`、PEP 709、method cache、kwnames

### T-16 · `initialize_locals` / `_PyEvalFramePushAndInit`：为什么它出现在我们的火焰图里

- **来源**：
  源码 `v3.12.13` [`Python/ceval.c#L1317` `initialize_locals`](https://github.com/python/cpython/blob/v3.12.13/Python/ceval.c#L1317)、
  [`#L1585` `_PyEvalFramePushAndInit`](https://github.com/python/cpython/blob/v3.12.13/Python/ceval.c#L1585)；
  对照 `v3.13.7` 同一函数仍在 [`#L1421`](https://github.com/python/cpython/blob/v3.13.7/Python/ceval.c#L1421)；
  调用链的一手证据见 [CPython issue #123372](https://github.com/python/cpython/issues/123372)
  （栈：`_PyFunction_Vectorcall → _PyEval_Vector → _PyEvalFramePushAndInit → initialize_locals`）。
- **核心机制（一句话）**：**每一次 Python 函数调用**都要经过它，把实参"装配"进新 frame 的 `localsplus`：
  (1) 位置参数逐个拷贝（`for (j = 0; j < n; j++) localsplus[j] = args[j];`）；
  (2) 若函数有 `**kwargs`，新建一个 dict；
  (3) 若有 `kwnames`，**逐个关键字在 `co_varnames` 里做名字匹配**；
  (4) 处理 `*args` 打包（可能分配 tuple）与默认值填充。
- **公开实测收益**：**没有找到 `initialize_locals` 单独的公开性能数字**（见 §9）。
  可引用的是"整体函数调用"的历史改善：3.11 起 Python→Python 调用不再递归重入解释器主循环
  （第三方综述：[Recent Performance Improvements in Function Calls in CPython](https://blog.codingconfessions.com/p/are-function-calls-still-slow-in-python)，2024-08-08；
  证据强度弱，因为它不是官方数字）。
  **由本项目数字派生**：`initialize_locals` 0.89% + `_PyFunction_Vectorcall` 0.95%
  ≈ **1.84% × 2757.6 µs ≈ 51 µs/步**（self time 相加，属上界近似）。
- **对我们的适用性**：
  1. 这 0.89% 不是"某个可以修的实现 bug"，而是**每调用一次 Python 函数就要付一次的固定税**；
  2. 因此它只能通过**减少 Python 层调用条数**来下降——这正是"把 3 次 GDN builder 合成 1 次"那类改动能顺带拿到的收益；
  3. **kwargs 调用会更贵**（见 T-19）：要额外走关键字名字匹配。
  **证据强度：中**（机制与调用链来自一手源码；51 µs/步是由本项目 self time 派生，非新增实测）。

### T-17 · PEP 709 内联推导式（3.12 已在生效）

- **来源**：[PEP 709 – Inlined comprehensions](https://peps.python.org/pep-0709/)（Carl Meyer，2023-02-24，Python-Version 3.12）。
- **核心机制（一句话）**：把 list/dict/set 推导式**内联进外层函数**，不再为每个推导式创建一个嵌套函数与 frame。
- **公开实测收益**：PEP 摘要原文——"仅推导式的微基准**最高 2×**；
  一个**重度使用推导式的真实代码样本 +11%**"。
- **对我们的适用性**：**已吃到**（容器就是 3.12.13）。**但有一个仍然有效的推论**：
  PEP 709 **不覆盖生成器表达式**（genexp 仍然建 frame），也不覆盖 `map/filter + lambda`；
  如果 `prepare_input` 的热路径里还有这类写法，把它们改成 list comprehension 仍然值得
  （机制上与 PEP 709 同源，但**具体收益没有公开数字**，属自测项）。
  **证据强度：强**（PEP 官方数字）/ **弱**（"我们还有多少 genexp"未测）。

### T-18 · 方法描述符缓存（`LOAD_ATTR_METHOD_*`）

- **来源**：同 §2 **T-05**；另见源码 `v3.12.13`
  [`Python/bytecodes.c#L2661` 起](https://github.com/python/cpython/blob/v3.12.13/Python/bytecodes.c#L2661)。
- **核心机制（一句话）**：`LOAD_ATTR`（要调用形态）专门化成 `LOAD_ATTR_METHOD_*` 后**不构造 bound method**，
  后续 `CALL` 直接命中方法描述符家族（`CALL_NO_KW_METHOD_DESCRIPTOR_{NOARGS,O,FAST}`、
  `CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS`、`CALL_BOUND_METHOD_EXACT_ARGS`）。
- **公开实测收益**：无单独公布的百分比；机制收益是"省掉一个 `PyMethod` 对象分配 + 一次引用计数往返"。
- **对我们的适用性**：**已在生效**（本机复现见附录 A.2）。
  **证据强度：强**（源码 + 本机复现）／**弱**（收益量级无公开数字）。

### T-19 · `kwnames`：**带关键字的 Python 调用在 3.12 永远不专门化**（★）

- **来源**：
  源码 `v3.12.13` [`Python/specialize.c` `specialize_py_call`](https://github.com/python/cpython/blob/v3.12.13/Python/specialize.c#L256)：
  ```c
  if (kwnames) {
      SPECIALIZATION_FAIL(CALL, SPEC_FAIL_CALL_KWNAMES);
      return -1;
  }
  ```
  对照 `v3.14.0` [`Python/bytecodes.c`](https://github.com/python/cpython/blob/v3.14.0/Python/bytecodes.c)：
  ```c
  family(CALL_KW, INLINE_CACHE_ENTRIES_CALL_KW) = {
      CALL_KW_BOUND_METHOD, CALL_KW_PY, CALL_KW_NON_PY,
  };
  ```
- **核心机制（一句话）**：3.12 对 **Python 函数**的 `CALL` 专门化只覆盖**无关键字**形态
  （C 函数另有 `CALL_BUILTIN_FAST_WITH_KEYWORDS` / `CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS` 变体）；
  带关键字（`f(x=1)`、`f(**d)`）的 **Python 调用永远走通用 CALL**——
  即"逐关键字在 `co_varnames` 里做名字匹配 + `initialize_locals` 的 kwargs 分支"。
- **公开实测收益**：无公开百分比。
- **对我们的适用性**：**高**。
  - vLLM 的元数据对象大量用关键字构造（dataclass 默认构造、`**changes`、`**kwargs` 透传）；
  - 我们那条 `.replace()` 的最后一跳正是 `obj.__class__(**changes)`（25 个关键字），
    所以它**必然**走通用 CALL；这条成本与 `.replace()` 的属性读取成本是**两笔**独立开销；
  - 3.14 起有 `CALL_KW` 家族（这是"换版本"的真实收益之一），但换版本要付 ABI 代价（H3）。
  **证据强度：强（机制，源码级）；弱（我们的量级，未测）**。

### T-20 · `dataclasses.replace()` 的隐藏成本：每字段一次 `getattr`（★）

- **来源**：CPython 3.12 `Lib/dataclasses.py::replace`（本机 `inspect.getsource` 核对，函数末尾为
  `return obj.__class__(**changes)`，循环内为 `changes[f.name] = getattr(obj, f.name)`；
  源码链接 [v3.12.10 `Lib/dataclasses.py`](https://github.com/python/cpython/blob/v3.12.10/Lib/dataclasses.py)）。
- **核心机制（一句话）**：`replace()` 先用 `getattr(obj, '__dataclass_fields__')` 取字段表，
  再对**每个未显式覆盖的字段**执行 `getattr(obj, f.name)`——这里的属性名是**运行时字符串**，
  因此走的是 **C 层通用属性查找**（`PyObject_GetAttr` → `_PyObject_GenericGetAttrWithDict` → `_PyType_Lookup`），
  **完全不吃字节码 IC 的红利**；最后再用 `**changes` 触发一次不专门化的 kwargs 构造（T-19）。
- **公开实测收益**：无（本条是机制分析，不冒充实测）。
- **对我们的适用性**：**与我们已识别的 `.replace()` 击穿问题叠加**。
  `CommonAttentionMetadata` 有 **25 个字段**（`refs/vllm/vllm/v1/attention/backend.py:411-490`），
  每步 3 个 GDN builder 各触发一次 `.replace()` ⇒ 最坏情况下每步
  **~75 次通用 `getattr` + 3 次 25-kwargs 构造**。
  这给火焰图里 `_PyType_Lookup`(1.67%) + `_PyObject_GenericGetAttrWithDict`(1.36%) 提供了一条**可验证的候选解释**：
  用 `pystats` 或给 harness 加"只统计 dataclasses.replace 子树"的探针即可证实/证伪。
  **证据强度：强（机制，源码级）；弱（收益量级，必须实测）**。

---

## 6. 引用计数：`object.h:646` 那 16% 到底能怎么办

### T-21 · `object.h:646` 是什么（先把归因钉死）

- **来源**：源码 `v3.12.13` [`Include/object.h`](https://github.com/python/cpython/blob/v3.12.13/Include/object.h#L636)（本文逐行核对，行号见附录 A）。
- **核心机制（一句话）**：64 位平台上，3.12 的 `Py_INCREF` 实现是
  "读高 32 位 → 加 1 → 判零（饱和，给 immortal 用）→ **写回**"，
  而 **第 646 行正是那条写回**：
  ```c
  641: PY_UINT32_T cur_refcnt = op->ob_refcnt_split[PY_BIG_ENDIAN];
  642: PY_UINT32_T new_refcnt = cur_refcnt + 1;
  643: if (new_refcnt == 0) { return; }
  646: op->ob_refcnt_split[PY_BIG_ENDIAN] = new_refcnt;
  ```
  所以 `perf annotate` 显示"16.01% 在 `object.h:646`"，含义是
  **`_PyEval_EvalFrameDefault` 里 16% 的时间花在引用计数写回（及其内联上下文）上**，
  而不是"某条可以换掉的昂贵指令"。
- **公开实测收益**：无（这是现状描述）。
- **对我们的适用性**：**它把优化方向锁死在"减少 INCREF/DECREF 次数"上**：
  - 减少临时对象（tuple/list/dict、bound method、临时 metadata 对象）；
  - 减少容器读写与属性读写本身（每次属性读都要 INCREF 结果）；
  - 注意 `Py_INCREF` 的写回是**内存 RMW**：同一缓存行上的多个小对象会互相干扰
    （920B 上"大量小对象"的形态天然容易出现）。
  **证据强度：中**（行号与实现是 100% 确定的；"能省多少"取决于能砍掉多少对象，属实测项）。

### T-22 · immortal objects（PEP 683）不是我们想要的杠杆

- **来源**：
  [PEP 683 – Immortal Objects, Using a Fixed Refcount](https://peps.python.org/pep-0683/)（2022-02-10）；
  [PR gh-84436 Implement Immortal Objects](https://github.com/python/cpython/pull/19474)；
  [Discussions: PEP 683 Updates](https://discuss.python.org/t/pep-683-immortal-objects-updates/23382)（2023-01-31）。
- **核心机制（一句话）**：给"永不释放"的对象（`None`/`True`/`False`/小整数/静态类型/被 intern 的字符串等）
  一个固定哨兵引用计数，使 INCREF/DECREF 变成饱和加法/快速短路；它主要服务于 **free-threading 与 sub-interpreter 共享**，
  不是为了给 GIL 构建提速。
- **公开实测收益**：
  - PEP 原文：naive 实现 **慢 4%**（MSVC 3%）；
  - PR 自测（2023-03，GCC 11.1，pyperformance）：**1.02× 慢**；
  - 官方结论："在若干 mitigation 之后回到 **~性能中性**"；
  - 作者原话（讨论帖）："作为 3.12 的一部分，它**没有带来性能收益**，但也没带来不可接受的损失"。
- **对我们的适用性**：**3.12 已经包含它**；它不会让我们的 16% 变小。
  ⇒ 不要把"refcount 花得多"当成"缺少 immortal objects"，也不要去追这个方向。
  **证据强度：中**（PEP + 上游自测；非第三方、无 aarch64 数据）。

### T-23 · free-threaded 的偏置/延迟引用计数（排除项）

- **来源**：
  [What's New In Python 3.14 §Free-threaded mode improvements](https://docs.python.org/3.14/whatsnew/3.14.html)；
  [PEP 836](https://peps.python.org/pep-0836/)（free-threading 是 JIT 的性能重点方向）。
- **核心机制（一句话）**：free-threaded 构建用 **biased reference counting + deferred refcount**
  （把"已知单线程持有"的引用计数留在对象上、其它线程的引用进队列）+ mimalloc，
  确实减少了跨线程 refcount 争用。
- **公开实测收益**：官方口径——3.14 free-threaded 对**单线程**代码的惩罚"**大约 5–10%**，取决于平台与编译器"；
  3.13 时期惩罚更大。
- **对我们的适用性**：**排除**。我们要解决的是单线程热点的 refcount 写回，
  而 free-threaded 构建先用 5–10% 的全局单线程税去换并发能力；
  对我们这种"单线程解释器热点 + 多进程/多线程框架"的场景，净收益为负。
  **证据强度：中**（官方数字；但那是 pyperformance，不是我们的负载）。

---

## 7. ARM / aarch64 特有：编译器门槛、公开数字、以及"没有数据"的部分

### T-24 · tail-call 解释器在 aarch64 上的公开数字（按机器、按基线）

- **来源**：同 T-09 的 PR #128718 / 3.14 What's New / Elhage 复测 / PEP 836。
- **核心机制（一句话）**：tail-call 解释器把"每 opcode 一个巨大的 `switch`"换成
  "每 opcode 一个小函数 + 保证尾调用"，其收益高度依赖**编译器版本与是否开 PGO**，
  因此同一个特性在不同 aarch64 机器/编译器上给出完全不同的数字。

| 机器（aarch64） | 配置 | 数字 | 基线 |
|---|---|---|---|
| ARM Neoverse N1（Ubuntu 22.04） | 3.14a3 + tail-call | 原始 **9.2%** → 更正后 **3–5%** | Clang 19 + PGO + ThinLTO（同一构建、仅换派发方式） |
| AArch64 **macOS M1** | 3.14a3 + tail-call | 原始 14.7% / 11.7%；**Elhage 复测 ≈ 1.00×（无收益）** | Clang 18 |
| AmpereOne（Linux aarch64） | 3.15 JIT（**不是** tail-call） | **1.073×（+7.3%）** | 同版本非 JIT 构建 |
| 任意 aarch64 | 3.12.0 → 3.15.0a0 **默认构建** | **1.036×（+3.6%）** | 3.12.0（arminc aarch64，pyperformance geomean） |

- **公开实测收益**：上表即为本条目的公开数字集合；**共同点是它们都不是 Kunpeng 920B**，
  也都不保证对"薄 Python 壳 + C/C++ 主体"的负载成立。
- **对我们的适用性**：**没有 Kunpeng 920/920B 的任何公开数据**（§9）。
  从"体系结构家族"看，Neoverse N1/Arm v8 与 Kunpeng 920 同代但**微架构不同**
  （取指宽度、间接分支预测器、BTB 容量都不同），因此 3–5% 只能当**量级参考**，不能当预测。
  好消息是：**这个方向的作用点正好是我们的痛点**（前端受限 + 派发密集），
  所以它是"值得在换版本时顺便验证"的候选，而不是"现在就该动"的候选。
  **证据强度：中**（一手 + 第三方，但机器不同）。

### T-25 · 编译器门槛：`musttail` 与 `preserve_none`

- **来源**：
  [CPython PR #128718 讨论串](https://github.com/python/cpython/pull/128718)（Ken Jin 等，2025-01~03）；
  [GCC 16 Release Series – Changes, New Features, and Fixes（AArch64 一节）](https://gcc.sourceware.org/gcc-16/changes.html)；
  [LWN: Python, tail calls, and performance](https://lwn.net/Articles/1033373/)（2025-08-20）；
  [Arch Linux python 打包 MR "enable tail-call interpreter with GCC 16"](https://gitlab.archlinux.org/archlinux/packaging/packages/python/-/merge_requests/9)（2026-05-03）。
- **核心机制（一句话）**：tail-call 解释器要求编译器同时支持
  **`musttail`（保证尾调用，否则编译失败）** 与 **`preserve_none`（只保留必要寄存器的调用约定）**；
  只有前者而没有后者时性能很差。
- **公开事实（不是收益数字）**：
  - Clang **19** 起在 x86-64 与 AArch64 上支持 `preserve_none`；
  - GCC 15 的 `musttail` 语义与 Clang 不同（Clang 把参数标记为 dead，GCC 不），CPython 需要写 `#ifdef` 绕开；
  - GCC 16 的 AArch64 新增 `preserve_none`（GCC 16 发行说明原文："Support for the `preserve_none` calling convention has been added…"）；
  - LWN 记录的 Ken Jin 发言：tail-call 解释器只支持"最近两年发布的编译器"，**具体是 Clang 19 与 GCC 16**；
    并且 **GCC 16 在开 PGO 时有一个导致 tail-call 变慢的 bug**，他本人都还没在 GCC 16 上完成基准（只跑了玩具 BF 解释器）。
- **对我们的适用性**：如果哪天要迁移，**编译器选择与 PGO 可用性是一个必须先验证的前置条件**；
  在 Kunpeng + openEuler 环境下意味着"要么把 Clang 19+ 装进构建链，要么等 GCC 16 的 PGO 问题解决"。
  **证据强度：中**（一手讨论 + 发行说明；GCC 16 上的实际收益**无公开数字**）。

### T-26 · JIT 的 aarch64 专属 codegen（只有上了 JIT 才有意义）

- **来源**：[gh-119726 "JIT: improve AArch64 code generation"](https://github.com/python/cpython/issues/119726)、
  [PR #123872 "generate and patch AArch64 trampolines"](https://github.com/python/cpython/pull/123872)。
- **核心机制（一句话）**：AArch64 的跳转距离限制迫使 JIT 用 trampoline；
  把 trampoline 移到 trace 末尾/运行时按需生成，可省代码尺寸与重定位。
- **公开实测收益**：`aarch64-unknown-linux-gnu` **0.8% faster / 0.6% less memory**；
  `aarch64-apple-darwin` **1.5% faster / 0.8% less memory**。
- **对我们的适用性**：**1% 级**的 JIT 内部改进，只有"先吃下 JIT"之后才谈得上。
  **证据强度：中**（上游 PR 自述 benchmark，Linux aarch64 有数字）。

### T-27 · PGO + LTO：**我们当前版本上唯一"零代码、同 ABI"的杠杆**

- **来源**：
  CPython 3.12 文档 [`Doc/using/configure.rst`（`--enable-optimizations` 段落）](https://docs.python.org/3.12/using/configure.html#cmdoption-enable-optimizations)：
  "Configuring Python using `--enable-optimizations --with-lto` (PGO + LTO) is recommended for best performance.
  The experimental `--enable-bolt` flag can also be used to improve performance."；
  [3.14 What's New](https://docs.python.org/3.14/whatsnew/3.14.html)：
  "Enabling profile-guided optimization is highly recommended when using the new interpreter as it is the only
  configuration that has been tested and validated for improved performance."；
  [PR #128718 讨论串](https://github.com/python/cpython/pull/128718) 中 Ken Jin 的自述：
  "PGO gives this roughly another 10% speedup over just `-O3`. LTO roughly another 10% over PGO and `-O3`."
- **核心机制（一句话）**：PGO 让编译器按真实负载的分支/调用频率排布代码（对解释器这种"巨大 switch + 大量分支"收益尤其大）；
  LTO 让 libpython 与扩展之间做跨模块内联。
- **公开实测收益**：
  - 官方：**"推荐以获得最佳性能"**（无百分比）；
  - 核心开发者自述：PGO **~10%**、LTO 再 **~10%**（**未注明平台与基线**，属个人测量）；
  - **没有找到 aarch64 Linux 上 PGO/LTO 的公开量化数字**（§9）。
- **对我们的适用性**：**这是我们当前唯一能立刻评估的解释器核心杠杆**，理由：
  (a) 同版本重建 ⇒ **ABI 不变**（cp312 轮子继续可用），风险远低于换版本；
  (b) 作用范围是**全体解释器时间**（不止 GDN builder 子树）；
  (c) 5 分钟即可判定是否已经启用：
  ```bash
  python3.12 -c "import sysconfig; print(sysconfig.get_config_var('CONFIG_ARGS')); \
                 print(sysconfig.get_config_var('CFLAGS'))"
  # 期望能看到 --enable-optimizations 与 --with-lto
  ```
  **证据强度：中**（官方文档 = 强；具体百分比 = 核心开发者自述、无平台信息、无第三方复现 = 中）。

### T-28 · 分支预测与"派发受限"：为什么这个话题在我们这里仍然成立

- **来源**：同 T-12 的 Rohou 等（2015）论文；本项目
  `docs/00-INDEX.md`（topdown 口径纪律）与 `docs/05-hotspots.md`（topdown 结果）。
- **核心机制（一句话）**：现代 CPU 的间接分支预测器已经能把解释器派发预测得很好
  （Haswell 级别 0.5–2 MPKI），所以"派发受限"**不是**指"间接跳转总是猜错"，
  而是指**指令供给**：每条字节码要执行 ~20 条 C 指令，前端要持续供给这些指令。
- **公开实测收益**：论文给出的核心数字是误预测率 **12–20 MPKI（Nehalem）→ 0.5–2 MPKI（Haswell）**，
  并明确结论"间接分支的可预测性不再是解释器的主要问题"；结合 T-12，computed goto 的剩余价值只有 **1–4%**。
  （没有找到"减少指令数能省多少"的直接公开数字；本文不臆造。）
- **对我们的适用性（三个必须说清的限定）**：
  1. 我们的 `frontend_bound 66.01%` + `retiring 12.85%` + `IPC 0.771` **形状上**与"取指/派发受限"一致；
     但高 frontend 占比**也**意味着"没有别的瓶颈在挡路"（backend 只 10.11%），不能读成"66% 时间在等 I-cache"。
  2. 920B 的 topdown 事件映射与 Intel 不同名同义，本项目已记录 `memstall_l3miss`/`dram_*` 恒 0
     等反直觉现象 ⇒ 跨架构类比只能到"定性"为止。
  3. **没有公开的 Kunpeng 920B 解释器分支预测数据**（§9）。
  ⇒ 可操作结论只有两条：**减少每条字节码消耗的指令数**（superinstruction/更少的 Python 层操作），
  以及**让派发代码更小更可预测**（tail-call 解释器的目标，3–5%）。
  **证据强度：中**（论文强 + 我们 PMU 形状一致；但 920B 具体行为无数据）。

---

## 8. 如果只能做一件事

### 8.1 先给结论（一句话）

**先花 5 分钟查一件事：我们容器里的 CPython 3.12.13 是不是用 `--enable-optimizations --with-lto` 构建的。**

- **如果没开** ⇒ **这就是最高性价比的方向，优先级高于 `.replace()` 缓存修复**：它是我们当前版本上
  **唯一**不换 ABI、不改一行业务代码、作用范围覆盖**全体解释器时间**（而不是某个 67.5% 的子树）的杠杆。
- **如果已经开了** ⇒ **解释器核心方向没有比 `.replace()` 更值得做的事**：剩下能拿出公开数字的手段
  （tail-call 3–5%、3.15 JIT 4–12%）都要求把 Python 从 3.12 迁到 3.14/3.15（cp312 ABI 全链路重建），
  而收益量级与 −130 µs/步 同阶、风险高一个数量级。此时正确顺序是：
  `.replace()` → GDN 3 合 1（姊妹文档 03 记录的两个上游 PR，260–600 µs/步）→ 再谈换解释器。

### 8.2 量化对比（全部标注来源与"是否派生"）

| 方案 | 作用范围 | 预期收益（换算到 µs/步） | 风险 / 成本 | 证据 |
|---|---|---|---|---|
| **PGO+LTO 重建 3.12.13**（同 ABI） | 全体解释器时间 | **0 ~ 276 µs**（若未开：核心开发者自述 PGO≈+10%、LTO≈再+10%；按 2.7576 ms 上界折算 10% = 276 µs；打折到 3% = 83 µs；**若已开则为 0**） | 低（重建镜像 + 复验 import）/ 1–3 人日 | 官方"推荐"= 强；10%/10% = 中（无平台信息） |
| **`.replace()` 保留缓存**（已识别） | GDN builder 子树 | **−130 µs**（项目自估；≈ prepare_input 的 4.7%） | 中（缓存正确性需回归）/ 1–3 人日 | 项目内实测子树 + 预期值（原文即"预期"） |
| **GDN 3 组 builder 合并**（姊妹文档 03 记录的两个上游 PR） | 908 µs 的头号热点 | **−260 ~ −600 µs** | 中（抄上游未合并 PR） | 本文**未独立复核**，引用 `docs/10-cpython-directions/03-torch-dispatch.md` 的 B 级证据 |
| **迁到 3.15 + JIT**（aarch64 Linux） | 仅"被 trace 到"的代码 | 乐观上界 **≈ +7.3%**（AmpereOne，2026-06）≈ **200 µs**；**受 T-08 限制，实际可能接近 0** | **高**：换 ABI + LLVM 21 构建 + vllm-ascend/torch-npu 全链路重建 | 中（PEP 836 附录；机器与负载都不是我们） |
| **迁到 3.14 + tail-call**（Clang 19 + PGO） | 全体派发 | **3–5%** ≈ **83–138 µs**（乐观；aarch64 的 Python 密集子项可达 +24%） | **高**：换 ABI + Clang 依赖 | 中（官方 What's New = 3–5%） |
| **只换 3.15 默认解释器（不带 JIT）** | 全体解释器时间 | **+3.6%** ≈ **99 µs**（3.15.0a0 vs 3.12.0，arminc aarch64，pyperformance geomean） | **高**：换 ABI | 中（官方 benchmark 仓库，从表格反推） |

> **口径说明**：上表"µs/步"是把百分比直接乘 `prepare_input` p50 = **2757.6 µs**（`docs/00-INDEX.md` §2.1）。
> 这些百分比**来自 pyperformance**，而我们的负载是"薄 Python 壳 + C/C++ 主体"，
> 因此换算只能当**上界**；表里凡标"乐观上界"的都不应写进收益承诺。

### 8.3 为什么"换版本"排在 `.replace()` 之后（三条理由）

1. **机制上它救不了主体**。T-08：tier-2/JIT 只从热回边/`RESUME` 进入，阈值是第 **4096** 次；
   我们的热点是"每步调 3 次的 builder + 大量一次性小函数"，
   真正能被 JIT 覆盖的只有那个进程里**恰好存在的 Python 层循环**，而且要先预热上千步。
2. **量级上同阶，风险上高一个数量级**。3.14 tail-call 的公开收益 3–5%（≈83–138 µs）与 `.replace()` 的
   −130 µs 同阶；但前者要把 cp312 换成 cp314，牵动 torch-npu / vllm-ascend / triton-ascend 的扩展 ABI
   与 CANN 适配，而后者是**同一容器内的一处代码改动 + harness 回归**。
3. **顺序上我们有更便宜的确定性收益**。姊妹文档 [`03-torch-dispatch.md`](03-torch-dispatch.md) 记录的两个上游 PR
   （#52297 `900 µs → 300 µs`；#16246 3 组 `6.3 ms → 4.5 ms/步`）对应我们头号热点的 260–600 µs/步，
   是 `.replace()` 的 2–5 倍，也大于任何解释器核心手段。把解释器换掉而热点还在，等于花大钱买小钱。

### 8.4 推荐执行顺序（无论上面哪个分支都建议做）

| 步骤 | 动作 | 判定标准 | 预计成本 |
|---|---|---|---|
| 0 | `python3.12 -c "import sysconfig;print(sysconfig.get_config_var('CONFIG_ARGS'))"`，确认有没有 `--enable-optimizations` / `--with-lto` | 有/无决定 §8.1 的分支 | 5 分钟 |
| 1 | 用**无卡 harness**（`harness/`，`--preset realmachine --batch 1 --isl 128`）跑 A/B：当前构建 vs PGO+LTO 重建版（若适用） | 按 `docs/06` 的口径：harness 只做**相对比较**，绝对值一律回真机 | 1–3 人日 |
| 2 | 构建一个 **`--enable-pystats`** 的 3.12.13（只用于诊断，不进交付），在 harness 上跑 `Tools/scripts/summarize_stats.py`，导出 `LOAD_ATTR` 各变体的 success/failure 与失败原因、`CALL` 的 `SPEC_FAIL_CALL_KWNAMES` 计数、`LOAD_ATTR` 的 hit/miss | 把 T-03/T-19/T-20 从"机制"变成"我们的数字"；若 `SPEC_FAIL_CALL_KWNAMES` 很大 ⇒ `.replace()` 那条 kwargs 构造值得优先处理 | 1 人日 |
| 3 | 做 `.replace()` 缓存保留（或改成"只构造必要字段"） | 同 §8.2 | 1–3 人日 |
| 4 | 抄上游 GDN 合并 PR（姊妹文档 03） | 260–600 µs/步 | 各自工作量 |
| 5 | **最后**才评估 CPython 3.14/3.15 迁移：先查"aarch64 + 目标编译器（Clang 19+ 或 GCC 16）能否带 PGO 构建通过，且 torch-npu/vllm-ascend 有没有 cp314/cp315 轮子" | 若 ABI 链路不成立，直接终止 | 2 人日预研 |

> ⚠️ **注意**：步骤 1 的 A/B 必须遵守 `docs/06-synthetic-load.md` 的保真度纪律
> （`--preset` 逐字匹配真机启动参数），否则 IPC 会偏 64%（0.949 vs 1.553）。

---

## 9. 明确查不到的（宁缺勿编）

以下内容本文**没有找到**可引用来源，因此没有给出任何数字；请勿在后续汇报中把它们当作结论：

1. **Kunpeng 920/920B 的 CPython 解释器性能数据**：没有公开的 pyperformance / 派发微基准 /
   分支预测（MPKI）数据，也没有针对该 SoC 的 CPython 解释器优化工作。
   本文所有 aarch64 数字都来自 Neoverse N1 / Apple M1 / M2-M3 / AmpereOne，**都不是 Kunpeng**。
2. **aarch64 Linux 上 PGO/LTO 的公开量化收益**：只有官方"推荐以获得最佳性能"的定性说法，
   和核心开发者"PGO ~10%、LTO 再 ~10%"的自述（**未注明平台**）。没有找到第三方复现。
3. **GCC 16 + tail-call 解释器在 aarch64 上的公开 benchmark**：CPython 作者在 2025-08 明确说
   "还没在（当时未发布的）GCC 16 上跑基准"，并指出 GCC 16 开 PGO 时 tail-call 会回归；
   2026-05 的 Arch Linux 打包 MR 启用了 GCC 16 + tail-call，但其理由写的是 x86 的 `preserve_none`。
4. **"属性查找 + 短调用 + 无热点"这类负载上 tier-2/JIT 的公开 benchmark**：
   pyperformance 没有任何子项对应我们的形态；CPython 官方也从未承诺可外推。
   因此本文对"我们能不能吃到 JIT"只给机制判定（T-08）+ 待做实验，不给收益预测。
5. **CPython 3.13 相对 3.12 在 aarch64 Linux 上的独立专项数字**：
   官方表里只有"3.15.0a0 vs 3.12.0 = 1.036×"与"3.15.0a0 vs 3.13.0 = 1.029×"，
   **反推** 3.13.0 ≈ 3.12.0 的 **+0.7%**（算术派生，非原始实测），本文已标注为派生。
6. **`initialize_locals` / `_PyFunction_Vectorcall` 的单独公开开销数字**：
   没有找到任何上游或第三方给出这两个符号的独立 microbenchmark；
   §5 的 ~51 µs/步 是**由本项目 self time 派生**（0.89% + 0.95%），不是新增实测。
7. **vLLM / vllm-ascend 官方对 CPython 构建参数的声明**：没有找到它们"推荐/要求 PGO 构建"的文档；
   因此本容器是否已启用 PGO/LTO **必须真机确认**（§8.4 步骤 0）。
8. **一个未解释清楚的本地观察**（建议用 pystats 在目标版本上复现）：
   在本机 3.12.10 上，同一个 `LOAD_ATTR` 站点，**普通类**在第 2 次调用后即变为 `LOAD_ATTR_INSTANCE_VALUE`；
   而一个**带类级默认值的 dataclass** 在第 2 次调用后仍是 `LOAD_ATTR`，继续调用到 ~20 次后才是
   `LOAD_ATTR_INSTANCE_VALUE`。本文能确定的是"它最终会专门化"，但**没有**确定"为什么第一次尝试会失败/延迟"
   （候选：`SPEC_FAIL_ATTR_NOT_IN_KEYS` 这类瞬态失败后进入退避）。这条影响"我们把 `_PyType_Lookup`
   归因于谁"，建议在目标版本上用 pystats 的 `LOAD_ATTR` 失败原因分布一次性证实/证伪。
9. **920B 上 pystats 的可用性**：`--enable-pystats` 需要重建解释器；
   没有在目标机上验证该构建能否与 CANN/torch-npu 共存（只用于诊断，预期可行但未验证）。

---

## 附录 A · 本机机制复现（**不是**收益测量）

**环境声明（重要）**：以下全部在**本地开发机**上完成，用的是 **CPython 3.12.10**（`python3.12 -V`），
**不是**目标机、**不是**容器里的 3.12.13。目标版本与本地只差 patch 版本，
下列机制性结论（阈值、指令变体、源码行号、`dataclasses.replace` 行为）在 3.12.13 上应完全一致
（相关源码文件在 3.12.x 内没有变化），但**这些运行不构成任何性能数字**，本文也从未把它们当收益引用。

### A.1 上游源码/存在性核查（curl + grep，全部在 2026-09-24 执行）

| 检查项 | v3.12.13 | v3.13.7 | 结论 |
|---|---|---|---|
| `Include/opcode_ids.h` 中 `ENTER_EXECUTOR` | **0 次** | 1 次 | 3.12 无 tier-2 入口指令 |
| `Python/ceval.c` 中 `_PyOptimizer_Optimize`/`_Py_TIER2`/`GOTO_TIER_TWO` | **0 次** | 4 处引用 | 3.12 无 tier-2 管线 |
| `Python/optimizer.c` 是否存在 | **HTTP 404** | 存在 | 同上 |
| `Include/internal/pycore_code.h` 的 `ADAPTIVE_WARMUP_VALUE` | **1** | 1 | 第 2 次执行即尝试专门化 |
| 同文件 `ADAPTIVE_COOLDOWN_VALUE` | **52** | 52 | 守卫失败后退避 53 次 |
| `Python/specialize.c::specialize_py_call` 对 `kwnames` 的处理 | **直接 `return -1`（不专门化）** | —（3.14 起有 `family(CALL_KW, …)`） | 带关键字的 Python 调用在 3.12 不专门化 |
| `Python/flowgraph.c::insert_superinstructions`（编译期融合 superinstruction） | **0 次**（融合仍在 quickening 时做） | **2 次**（编译期） | 3.12 只执行一次/两次的代码拿不到编译期融合 |

### A.2 专门化行为（脚本 + 观察）

```python
import dis
def names(fn): return [i.opname for i in dis.get_instructions(fn, adaptive=True) if 'LOAD_ATTR' in i.opname]
def g(o): return o.a
class P:
    def __init__(self): self.a = 1
p = P()
print(names(g))        # ['LOAD_ATTR']                 ← 0 次执行（未专门化）
g(p); print(names(g))  # ['LOAD_ATTR']                 ← 1 次执行
g(p); print(names(g))  # ['LOAD_ATTR_INSTANCE_VALUE']   ← 第 2 次执行即专门化
```

观察结果（3.12.10，本地）：

| 场景 | `LOAD_ATTR` 站点的变体 |
|---|---|
| 0 / 1 次调用 | `LOAD_ATTR`（未专门化） |
| 2 次调用 | `LOAD_ATTR_INSTANCE_VALUE` |
| 500 个**不同实例**、每个只读一次同一属性、同一站点 | `LOAD_ATTR_INSTANCE_VALUE`（**保持专门化**） |
| 方法调用 `x.bit_length()` | `LOAD_ATTR_METHOD_NO_DICT` + `CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS` |
| 自适应字节码里的 superinstruction（3.12 在 quickening 时插入） | `LOAD_FAST__LOAD_FAST`、`STORE_FAST__LOAD_FAST` |

⇒ 直接支撑 T-01/T-02/T-04/T-05/T-13 的结论：**站点级重复就够，对象级"只读一次"不是问题**。

### A.3 `object.h:646` 行号核对

`curl -sL .../v3.12.13/Include/object.h | sed -n '640,652p'`，逐行对照后确认：
**646 行 = `op->ob_refcnt_split[PY_BIG_ENDIAN] = new_refcnt;`**（即 `Py_INCREF` 的写回），
与 `perf annotate` 报的 `object.h:646` 完全对应。

### A.4 `dataclasses.replace` 源码核对

本机 `inspect.getsource(dataclasses.replace)`：循环体为
`changes[f.name] = getattr(obj, f.name)`，返回值为 `obj.__class__(**changes)`；
即**每字段一次 C 层 `getattr`** + **一次 kwargs 构造**（T-20/T-19 的来源）。

---

## 附录 B · 来源清单（访问日期均为 2026-09-24）

### B.1 官方规范 / 发布说明

| # | 标题 | URL | 时间 |
|---|---|---|---|
| 1 | PEP 659 – Specializing Adaptive Interpreter | https://peps.python.org/pep-0659/ | 创建 2021-01-15（Final） |
| 2 | What's New In Python 3.11（Faster CPython） | https://docs.python.org/3.11/whatsnew/3.11.html | 3.11 系列 |
| 3 | PEP 709 – Inlined comprehensions | https://peps.python.org/pep-0709/ | 创建 2023-02-24（Final，3.12） |
| 4 | PEP 683 – Immortal Objects, Using a Fixed Refcount | https://peps.python.org/pep-0683/ | 创建 2022-02-10（Final） |
| 5 | PEP 744 – JIT Compilation | https://peps.python.org/pep-0744/ | 创建 2024-04-11（3.13） |
| 6 | What's New In Python 3.13（An experimental just-in-time (JIT) compiler） | https://docs.python.org/3.13/whatsnew/3.13.html | 3.13.0，2024-10-07 |
| 7 | What's New In Python 3.14（A new type of interpreter；Free-threaded mode improvements） | https://docs.python.org/3.14/whatsnew/3.14.html | 3.14.0，2025-10-07（本文访问的是 3.14.6 文档页） |
| 8 | What's New In Python 3.15 | https://docs.python.org/3.15/whatsnew/3.15.html | 3.15.0rc2（2026-08 起） |
| 9 | PEP 836 – JIT Go Brrr: The Path to a Supported JIT Compiler for CPython | https://peps.python.org/pep-0836/ | 2026 |
| 10 | CPython 3.12 文档：`--enable-optimizations`（PGO+LTO 建议） | https://docs.python.org/3.12/using/configure.html#cmdoption-enable-optimizations | 3.12 系列 |
| 11 | Python Insider: Python 3.15's JIT is now back on track | https://blog.python.org/2026/03/jit-on-track/ | 2026-03-23 |
| 12 | Python Insider: Python 3.15.0 beta 4 is here! | https://blog.python.org/2026/07/python-3150-beta-4/ | 2026-07-18 |

### B.2 上游源码 / 设计文档（本文按 tag 核对）

| # | 内容 | URL |
|---|---|---|
| 13 | `v3.12.13 Include/internal/pycore_code.h`（warmup 1 / cooldown 52） | https://github.com/python/cpython/blob/v3.12.13/Include/internal/pycore_code.h#L402 |
| 14 | `v3.11.13 Include/internal/pycore_code.h`（`QUICKENING_WARMUP_DELAY 8`、初始计数 31） | https://github.com/python/cpython/blob/v3.11.13/Include/internal/pycore_code.h#L95 |
| 15 | `v3.12.13 Include/object.h`（`Py_INCREF` 第 646 行） | https://github.com/python/cpython/blob/v3.12.13/Include/object.h#L636 |
| 16 | `v3.12.13 Python/specialize.c`（`_Py_Specialize_LoadAttr` / `specialize_dict_access` / `specialize_py_call`） | https://github.com/python/cpython/blob/v3.12.13/Python/specialize.c#L721 |
| 17 | `v3.12.13 Python/bytecodes.c`（`LOAD_ATTR_*` / `CALL_*` 家族） | https://github.com/python/cpython/blob/v3.12.13/Python/bytecodes.c#L1760 |
| 18 | `v3.12.13 Python/ceval.c`（`initialize_locals` L1317、`_PyEvalFramePushAndInit` L1585） | https://github.com/python/cpython/blob/v3.12.13/Python/ceval.c#L1317 |
| 19 | `v3.13.7 Python/ceval.c`（`initialize_locals` L1421，确认 3.13 仍在） | https://github.com/python/cpython/blob/v3.13.7/Python/ceval.c#L1421 |
| 20 | `v3.14.0 Python/bytecodes.c`（`family(CALL_KW, …)`） | https://github.com/python/cpython/blob/v3.14.0/Python/bytecodes.c |
| 21 | `main InternalDocs/jit.md`（trace 入口：热 `JUMP_BACKWARD`/`RESUME`） | https://github.com/python/cpython/blob/main/InternalDocs/jit.md |
| 22 | `v3.12.10 Lib/dataclasses.py`（`replace()` 的 `getattr` 与 `**changes`） | https://github.com/python/cpython/blob/v3.12.10/Lib/dataclasses.py |

### B.3 上游 issue / PR（自述数字，本文按"自述"标注）

| # | 内容 | URL | 时间 |
|---|---|---|---|
| 23 | gh-98686 "Quicken everything"（阈值 8+31 → 2；~1%） | https://github.com/python/cpython/issues/98686 | 2022-10-25 |
| 24 | gh-129386（当前计数器常量：warmup 1 / cooldown 52 / JIT 4096） | https://github.com/python/cpython/issues/129386 | 2025 |
| 25 | bpo-44900 五个 superinstructions（**2%**） | https://github.com/python/cpython/pull/27741 | 2021-08-16 |
| 26 | gh-105229 单指令 superinstruction（**1.7%**） | https://github.com/python/cpython/pull/105230 | 2023-06-02 |
| 27 | gh-128563 / PR #128718 tail-call 解释器（含 **CORRECTION: 3–5%**） | https://github.com/python/cpython/pull/128718 | 2025-01 起 |
| 28 | gh-84436 / PR #19474 Immortal Objects（naive 慢 4%；最终 ≈ 中性） | https://github.com/python/cpython/pull/19474 | 2022–2023 |
| 29 | gh-119726 / PR #123872 AArch64 JIT trampoline（Linux aarch64 **+0.8%**） | https://github.com/python/cpython/pull/123872 | 2024 |
| 30 | CPython issue #123372（`_PyFunction_Vectorcall → … → initialize_locals` 调用链） | https://github.com/python/cpython/issues/123372 | 2024 |
| 31 | faster-cpython/ideas#647（JIT 动态 superinstruction：**慢 2–5%**） | https://github.com/faster-cpython/ideas/issues/647 | 2024-01 |
| 32 | faster-cpython/ideas 3.14 页（tail-call "1–5% geomean"） | https://github.com/faster-cpython/ideas/tree/main/3.14 | 2025 |

### B.4 第三方测量 / 学术 / 二手（用于交叉验证）

| # | 标题 | URL | 时间 | 用途 |
|---|---|---|---|---|
| 33 | Nelson Elhage: Performance of the Python 3.14 tail-call interpreter | https://blog.nelhage.com/post/cpython-tail-call/ | 2025-03-09 | tail-call 独立复测（1.00–1.03×）、computed goto 2–4% |
| 34 | Ken Jin: I'm Sorry for Python's tail-calling Interpreter's Results | https://fidget-spinner.github.io/posts/apology-tail-call.html | 2025-03-08 | 作者更正（真实收益 3–5%） |
| 35 | LWN: Python, tail calls, and performance | https://lwn.net/Articles/1033373/ | 2025-08-20 | EuroPython 记录：2%（x86-64 Linux）/5–7%（arm64 macOS）、GCC 16 问题 |
| 36 | LWN: Python JIT stabilization | https://lwn.net/Articles/970397/ | 2024-04-25 | micro-op 1–4% 损失；tier-2 只作调试 |
| 37 | LWN: Adding a JIT compiler to CPython | https://lwn.net/Articles/977855/ | 2024-06-18 | tier-2 解释器比字节码慢 ~20% |
| 38 | Rohou, Swamy, Seznec: Branch Prediction and the Performance of Interpreters – Don't Trust Folklore | https://inria.hal.science/hal-01100647/document | 2015 | 派发间接跳转 MPKI 12–20 → 0.5–2；~21 条指令/字节码 |
| 39 | faster-cpython/benchmarking-public（arminc aarch64 表） | https://github.com/faster-cpython/benchmarking-public | 持续更新（本表取 2025-08-23 行） | 3.15.0a0 vs 3.12.0 = 1.036×；JIT 行 vs base = 1.011×（慢） |
| 40 | GCC 16 Release Series – Changes（AArch64 `preserve_none`） | https://gcc.sourceware.org/gcc-16/changes.html | GCC 16（2026） | 编译器门槛 |
| 41 | Arch Linux python 打包 MR：enable tail-call interpreter with GCC 16 | https://gitlab.archlinux.org/archlinux/packaging/packages/python/-/merge_requests/9 | 2026-05-03 | GCC 16 实际启用案例 |
| 42 | Abhinav Upadhyay: Recent Performance Improvements in Function Calls in CPython | https://blog.codingconfessions.com/p/are-function-calls-still-slow-in-python | 2024-08-08 | 函数调用历史改善（二手，仅作背景） |
| 43 | Simon Willison: python-build-standalone now has Python 3.14.0a5 | https://simonwillison.net/2025/Feb/13/python-3140a5/ | 2025-02-13 | "20–30%" 早期误传的实例（T-11 反例） |

### B.5 本项目内部引用

| # | 内容 | 出处 |
|---|---|---|
| 44 | `prepare_input` p50 = 2757.6 µs、占单步 55.3% | `docs/00-INDEX.md` §2.1 |
| 45 | topdown frontend 66.01% / retiring 12.85% / IPC 0.771 | `docs/00-INDEX.md` §2 |
| 46 | 符号 self time（含 `object.h:646` 16.01%） | `docs/05-hotspots.md` §3 |
| 47 | GDN builder 908 µs/步、`.replace()` 预期 −130 µs | `docs/00-INDEX.md` §3 |
| 48 | GDN 3→1 的上游 PR（#52297、#16246）与其数字 | `docs/10-cpython-directions/03-torch-dispatch.md` §0（本文未独立复核） |
| 49 | harness 保真度纪律（`--preset` 逐字匹配；IPC 0.949 vs 1.553） | `docs/06-synthetic-load.md` §1.5 与 §9 |
| 50 | `CommonAttentionMetadata`（25 字段、`_num_computed_tokens_cache`） | `refs/vllm/vllm/v1/attention/backend.py:411-535` |

---

## 附录 C · 证据强度分级定义（本文统一口径）

| 级别 | 含义 | 典型例子（本文） |
|---|---|---|
| **强** | 一手来源（PEP / 官方文档 / 上游 merged 源码）+ 可复核的公开数字；或机制来自源码且已被本地复现 | `ADAPTIVE_WARMUP_VALUE=1`；`specialize_py_call` 对 `kwnames` 直接拒绝；`LOAD_ATTR` 第 2 次执行专门化 |
| **中** | 有一手来源与公开数字，但**平台/负载形态与我们不同**，或数字来自个人自述且无第三方复现 | tail-call 3–5%（Neoverse N1 / M1）；PGO ≈10%+10%（无平台）；3.15 JIT 7.3%（AmpereOne + pyperformance） |
| **弱** | 只有机制推断、或只有二手转述、或收益量级完全没有公开数字 | `initialize_locals` 的 ~51 µs/步（本项目派生）；`.replace()` 的 `getattr` 成本量级；genexp 改写收益 |

**三条引用纪律**（与本文所有数字绑定）：

1. 凡"µs/步"必须写清是**引用**还是**由本项目 self time / 百分比派生**；
2. 凡 pyperformance 的百分比，必须同时给出**机器、编译器、基线**三要素（T-11 的教训）；
3. 凡本文标"未测 / 查不到"的，一律不得改写成结论。
