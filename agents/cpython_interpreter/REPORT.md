# cpython_interpreter —— 交接报告

**Agent**：`/root/cpython_interpreter`
**任务**：CPython 解释器核心方向的文献/网络调研（纯调研，不碰目标机）
**状态**：✅ 完成（1 个交付文档；0 次上机；0 次跑 NPU）
**时间**：2026-09-24（Asia/Shanghai）
**写入范围**：仅本工作区
`docs/10-cpython-directions/01-interpreter-core.md`（869 行）与 `agents/cpython_interpreter/REPORT.md`

---

## 1. 交付物

| 路径 | 内容 |
|---|---|
| `docs/10-cpython-directions/01-interpreter-core.md` | 主文档：28 条技术（T-01~T-28），每条给"名称 / 来源(URL+标题+时间) / 核心机制一句话 / 公开实测收益 / 对我们负载的适用性 + 证据强度"；含 §8"如果只能做一件事"、§9"查不到的"、附录 A（本机机制复现）、附录 B（50 条来源清单）、附录 C（证据强度定义） |
| `agents/cpython_interpreter/REPORT.md` | 本文件 |

主文档结构：

| 节 | 内容 |
|---|---|
| §0–§1 | 结论速览 6 条 + 三条硬判据（H1 站点级重复 / H2 Python 层热循环 / H3 不换 ABI）+ 全技术速查表 |
| §2 | PEP 659 专门化：IC 机制、3.11→3.12 阈值（几十次 → 2 次）、5 个失效源、bound-method 路径 |
| §3 | tier-2 / 微指令 / JIT：**3.12 根本没有**、3.13 的 JIT（≈持平）、3.14 tail-call（3–5%）、3.15 JIT（4–12% 但按平台）、**触发条件是热回边（第 4096 次）**、数字考古 |
| §4 | computed goto（1–4%）、静态 superinstruction（2% / 1.7%）、JIT 动态融合的负结果、`ENTER_EXECUTOR` |
| §5 | `initialize_locals` / PEP 709 / method cache / **3.12 的 kwargs 调用永不专门化** / **`dataclasses.replace()` 每字段一次通用 `getattr`** |
| §6 | `object.h:646` = refcount 写回；immortal objects ≈ 中性；free-threaded 单线程 −5~10% |
| §7 | aarch64：tail-call 按机器的公开数字、`musttail`/`preserve_none` 门槛（Clang 19 / GCC 16）、AArch64 JIT codegen（0.8–1.5%）、**PGO+LTO 是唯一同 ABI 杠杆**、分支预测论文 |
| §8 | 如果只能做一件事（决策树 + 量化对比 + 6 步执行顺序） |
| §9 | 查不到的 9 条（宁缺勿编） |

---

## 2. 五条最重要的结论（供协调者快速判断）

1. **事实澄清：我们的 3.12.13 里没有 tier-2、没有 JIT、没有 tail-call 解释器。**
   源码核查：`v3.12.13` 无 `Python/optimizer.c`（404）、`Include/opcode_ids.h` 无 `ENTER_EXECUTOR`、
   `ceval.c` 无 `_PyOptimizer_Optimize`；对照 `v3.13.7` 全部存在。
   ⇒ 任何"在 3.12 上开 `PYTHON_JIT=1`/`-X uops`"的方案都是无效动作。

2. **"每个属性只访问一次"不是 IC 的障碍，站点级重复才是关键。**
   3.12 的门槛是"站点第 2 次执行就尝试专门化"（`ADAPTIVE_WARMUP_VALUE = 1`；3.11 是 quickening 8 次 + 计数 31）。
   本机复现：同一站点 0/1 次调用 = `LOAD_ATTR`，2 次 = `LOAD_ATTR_INSTANCE_VALUE`；
   500 个不同实例、每个只读一次同一属性，站点**保持专门化**。
   真正的杀手是**站点多态**（失败后退避 53 次）与**类型版本失效**。

3. **JIT/tier-2 只从热回边进入，阈值第 4096 次**（`JUMP_BACKWARD_INITIAL_VALUE`）。
   我们是"薄 Python 壳 + C/C++ 主体 + 每步只跑几次的 builder"，
   因此**不能假设换了 3.15 就能吃到 JIT**；必须先确认那个进程里有没有够热的 Python 层循环。

4. **两个"我们负载专属"的新发现（源码级，本文首次提出）**：
   - **3.12 的 `specialize_py_call` 对 `kwnames != NULL` 直接 `return -1`** ⇒
     带关键字的 Python 调用（`f(x=1)`、`f(**d)`）在 3.12 **永远不专门化**；
     而 `dataclasses.replace()` 的最后一跳 `obj.__class__(**changes)`（我们这里是 25 个关键字）正是这种调用。
   - **`dataclasses.replace()` 对每个未覆盖字段执行一次 `getattr(obj, f.name)`**（运行时字符串名）⇒
     走 C 层通用属性查找（`_PyObject_GenericGetAttrWithDict` / `_PyType_Lookup`），
     **完全不吃 IC**。我们的 `CommonAttentionMetadata` 有 25 字段、每步 3 次 `.replace()`
     ⇒ 最坏 ~75 次通用 getattr + 3 次 25-kwargs 构造/步。
     这给火焰图里 `_PyType_Lookup`(1.67%) + `_PyObject_GenericGetAttrWithDict`(1.36%) 提供了候选解释。

5. **唯一"同 ABI、零代码"的解释器核心杠杆是构建期的 PGO + LTO。**
   官方文档："`--enable-optimizations --with-lto` is recommended for best performance"；
   核心开发者在 PR 讨论里自述 "PGO ≈ 再 +10% over -O3，LTO ≈ 再 +10%"（**未注明平台**）。
   **没有找到 aarch64 Linux 的公开 PGO/LTO 数字**（§9）。

---

## 3. §8"如果只能做一件事"的判定（直接可引用）

**先花 5 分钟查容器里的 CPython 是否已 `--enable-optimizations --with-lto`：**

```bash
python3.12 -c "import sysconfig; print(sysconfig.get_config_var('CONFIG_ARGS')); \
               print(sysconfig.get_config_var('CFLAGS'))"
```

- **没开** ⇒ 这就是最高性价比方向，**优先于 `.replace()` 修复**：
  同版本重建 ⇒ ABI 不变（cp312 轮子继续可用）、作用范围是**全体解释器时间**（≈ 2757.6 µs）
  而不是某个 67.5% 的子树；量级 3%–10% ⇒ **83–276 µs/步**（上界按 10% 折算，**若已开则为 0**）。
- **已开** ⇒ **解释器核心没有比 `.replace()`（−130 µs）更值得做的事**：
  其余手段都要换 CPython 版本（3.14 tail-call 3–5% ≈ 83–138 µs；只换 3.15 默认解释器 +3.6% ≈ 99 µs；
  3.15 JIT 乐观 +7.3% ≈ 200 µs 但受"热回边"限制可能接近 0），
  量级与 −130 µs 同阶，而风险/工期高一个数量级（cp312 → cp314/cp315 全链路重建）。
- 无论哪个分支，都建议做**诊断**：构建一个 `--enable-pystats` 的 3.12.13，用 `summarize_stats.py`
  看 `LOAD_ATTR` 失败原因分布与 `SPEC_FAIL_CALL_KWNAMES` 计数（1 人日，风险≈0，信息量最大）。
- **顺带说明**：如果放宽到"任意方向"，本文认为最高性价比不是解释器核心，而是姊妹文档
  `docs/10-cpython-directions/03-torch-dispatch.md` 记录的两个上游 GDN 合并 PR（260–600 µs/步）。

---

## 4. 证据强度与引用纪律（本文的口径）

- **强**：PEP / 官方文档 / 上游 merged 源码 + 可复核公开数字，或源码机制已被本地复现。
- **中**：有一手来源与公开数字，但**平台或负载形态与我们不同**，或个人自述无第三方复现。
- **弱**：只有机制推断 / 二手转述 / 收益量级无公开数字。

**三条引用纪律**：
1. 凡"µs/步"必须写清是引用还是**由本项目 self time / 百分比派生**（本文的 84 µs / 285 µs / 51 µs 都标了派生）；
2. 凡 pyperformance 百分比必须同时给出**机器 + 编译器 + 基线**（T-11 记录了同一个 tail-call 特性被报成
   9–15% → 3–5% → 1–5% 的全过程，根因是 LLVM 19 的 tail-dup 回归拖慢了旧基线）；
3. 标"未测 / 查不到"的一律不得改写成结论。

---

## 5. 本文用到的检查手段（可复现，均为只读）

| 手段 | 说明 |
|---|---|
| 上游源码按 tag 核查 | `curl -sL https://raw.githubusercontent.com/python/cpython/<tag>/<path>` + `grep/sed`；覆盖 `v3.11.13 / v3.12.13 / v3.13.7 / v3.14.0 / main`，比对 `ENTER_EXECUTOR`、`optimizer.c`、`ADAPTIVE_*`、`specialize_py_call`、`insert_superinstructions` |
| 行号核对 | `v3.12.13/Include/object.h` L640–652 ⇒ 确认 **646 行 = `op->ob_refcnt_split[...] = new_refcnt;`**（`Py_INCREF` 写回），与真机 `perf annotate` 的 `object.h:646` 对应 |
| 本机机制复现 | 本机 CPython **3.12.10**（非目标机、非容器）用 `dis.get_instructions(..., adaptive=True)` 观察站点专门化与 superinstruction；`inspect.getsource(dataclasses.replace)` 核对 `getattr`/`**changes`。**仅为机制验证，不构成任何性能数字** |
| 网络检索 | Exa 搜索 + 直接抓取 PEP / docs.python.org / GitHub / LWN / 学术文档；所有外链访问日期均为 **2026-09-24** |

---

## 6. 未决 / 需要真机或后续 agent 接手的事项

完整列表见主文档 **§9**，这里列最关键的 4 条：

1. **本容器 CPython 的构建参数**（是否 PGO/LTO）—— 未查（本文的原则是不碰目标机）。一条命令即可判定，见 §3。
2. **`LOAD_ATTR` 失败原因的实测分布**—— 需要 `--enable-pystats` 构建 + `Tools/scripts/summarize_stats.py`；
   本文只给了 5 个候选失效源（多态 / 类型版本 / dict 布局 / 不可专门化类型 / C 层查找）。
3. **一个未解释清楚的本地观察**：3.12.10 上"带类级默认值的 dataclass"站点在第 2 次调用时仍是
   `LOAD_ATTR`，到 ~20 次才变 `LOAD_ATTR_INSTANCE_VALUE`（普通类第 2 次就变）。
   已确认"最终会专门化"，未确认"第一次为何失败/延迟"（候选：`SPEC_FAIL_ATTR_NOT_IN_KEYS` 类瞬态失败后退避）。
   **不要**把它当成"dataclass 无法专门化"的结论——本文没有这样说，也不建议这样用。
4. **Kunpeng 920B 的公开数据完全空白**：没有 CPython 解释器 benchmark、没有分支预测 MPKI、
   没有针对该 SoC 的解释器优化工作。本文所有 aarch64 数字来自 Neoverse N1 / M1 / M2-M3 / AmpereOne。

---

## 7. 与其它子任务的接口

| 相关文档 | 关系 |
|---|---|
| `docs/10-cpython-directions/03-torch-dispatch.md` | 姊妹篇；本文 §8 引用其 GDN 3→1 PR 数字（#52297 `900→300 µs`、#16246 `6.3→4.5 ms/步`），**本文未独立复核**，引用时请保留该标注 |
| `docs/10-cpython-directions/05-graph-dispatch-host-side.md` | 姊妹篇；本文 §3 T-08 的"热回边"判据可作为"图化 vs JIT"的补充论据 |
| `docs/05-hotspots.md` | 本文引用其符号 self time（含 `object.h:646` 16.01%）与 topdown 数字 |
| `docs/06-synthetic-load.md` | 本文 §8 的 A/B 方案依赖其 harness 保真度纪律（`--preset` 逐字匹配） |
| `docs/00-INDEX.md` §3 | `.replace()` 修复（−130 µs）的来源；本文给出了它的两条额外机制解释（T-19/T-20） |
