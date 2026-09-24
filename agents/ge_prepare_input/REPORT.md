# 子代理报告：GE 整图下发能否用于 `prepare_input`

**任务**：判定 GE（Graph Engine）整图下发能否抹除 `prepare_input` 的 host 开销，
并把开销严格分成 L1/L2/L3/L4 四层逐一论证。

**产出**：`docs/11-ge-whole-graph/03-ge-for-prepare-input.md`（792 行，本文档为摘要）。

**约束遵守**：未登录远程主机、未跑 NPU、未动 chip3、未起容器；
只读参考了 `docs/01`/`docs/02`/`docs/05`/`docs/10`、`data/static/prepare-input-steps.json`、
`vllm-ascend` 0.26.0rc1 与 `refs/vllm` 源码；联网仅查官方文档（hiascend.com / gitcode / PyTorch / vLLM）。
**所有写入均在 `/home/chiro/projects/vllm/preparing-input-phase/` 内。**

---

## 1. 结论（一句话 + 分层）

> **GE 能抹掉 `prepare_input` 的「派发税」（≈400–600 µs / 步），
> 抹不掉「记账与决策税」——因为 `prepare_input` 的输入是 host 侧的调度决策（Python 对象），
> 输出才是 GE 图能接受的张量。GE 只覆盖"从张量到张量"的那一段。**

| 层 | GE 能否省 | 可省量级 | 决定性依据 |
|---|---|---:|---|
| L1 kernel launch | 能，但这里几乎没得省 | ≈0–140 µs | `prepare_input` 里只有 `slot_mapping` 等极少 kernel launch |
| L2 算子派发 | **部分能** | **≈400–500 µs（上限）** | 仅限"可表达为张量运算"的调用；GDN builder 的 device 段 ≈494 µs 是主标的 |
| L3 元数据构造 | 只省算术，不省值 | ≈150–300 µs（与 L2 重叠） | 三个前提：固定 shape（部分满足）/ 固定地址（**缺口**）/ 参数 tensor 化（部分满足） |
| L4 host 对象与决策 | **完全不能** | **0** | AscendIR 无 dict/deque/dataclass 类型；capture 期禁止同步与查询 |

---

## 2. 五条最关键的新证据（本文件补上的，`00-source-evidence.md` 未覆盖）

1. **图输入的类型学铁证**（关键判断点 1）
   TorchAir 官方《动/静态图展示》给出的 GE 图 dump 里，图输入一律是
   `ge.Data(index, dtype, shape, placement)`；**Python 标量要么被折成 `ge.Const`（编译期固定），
   要么被升格成 `ge.Data(shape=[], placement="CPU")`（每步由 host 重喂）**。
   ⇒ **list/dict 无法成为图输入**，这正是 `docs/10` 把"FIA 用 tensor 传参"当作前提 3 证据的根因，
   也解释了为什么 `seq_lens_list`（list 形态）意味着"这段 Python 必须每步在 host 跑"。

2. **GE 图内可以包含 host 计算**（对 `docs/10` 的补丁）
   官方《动态 Shape 图调度》明确：GE 会把**小 shape、输入在 Host 的算子刻意保留在 Host 侧执行**
   （默认启用，LLaMA2 上刷 650+ 个算子）。⇒ 正确的结论不是"host 工作不可入图"，
   而是"**入图的 host 工作仍每步在 CPU 上跑，只是省掉了 InferShape/Tiling/内存分配/派发转换**"。
   这条既支撑了 L2 的收益，也守住了 L4 的结论。

3. **官方承认了 GE 化的障碍**
   `vllm-ascend/docs/.../ACL_Graph.md`："**some attention operators need runtime metadata updates
   even when the overall graph is static**" + "**Without that hook, capture alone is not enough**"。
   ⇒ 固定地址是**必要不充分**条件；图的静态性不消除每步 host 参数更新。

4. **"预分配 buffer ≠ 入图"**
   `_pad_non_spec_decode_graph_inputs`（`gdn_attn_builder.py:332–369`）**已经在写预分配 buffer**，
   但每步仍发起约 12 次 device op host 调用，实测 **285.9 µs/步**（`prepare_input` 第 3 大子步骤）。
   ⇒ `docs/10` §4"已有雏形"需要补这一刀：**形态对了，但仍在图外，照收派发税**。

5. **图化的前提正在由 `prepare_input` 买单**（我认为这是本次分析最有价值的洞察）
   `FULL_DECODE_ONLY` 要靠"每步 batch 长得一样"才能捕获，而把 batch 摆成这个形状的工作落在
   `prepare_input` 里（`_pad_query_start_loc_for_fia` + `_pad_non_spec_decode_graph_inputs`）。
   ⇒ **不能用"图化让 forward 变快"来论证"图化能优化 `prepare_input`"**——
   恰恰相反，`prepare_input` 的 55.3% 占比有一部分就是为图化付出的代价。

---

## 3. GDN builder 的 908 µs 逐段判定（关键判断点 5）

| 段 | µs/步 | 判定 |
|---|---:|---|
| `pad_graph_inputs`（294） | 285.9 | **可进图**（形态已对），但**组间不可去重**；可省的是 12 次 op 的派发税（≈100–200），不是全部 294 |
| `treat_single_token`（168） | 169.3 | **原理上不可下沉**（`torch.any(...).item()` → host 必须知道布尔值）；**但纯 decode 下可整体短路，不需要 GE** |
| `compute_num_computed_tokens`（129） | 129.5 | **不该下沉，该修 bug**：`.replace()` 清空 `_num_computed_tokens_cache`（perf 证据：子树 67.5% 在 `PyNumber_Subtract`） |
| `split_decodes`（118） | 121.1 | 返回 Python int，必须回 host；纯 decode 下可短路 |
| dataclass 构造（残差） | 21.6 + 172.9 残差 | 不可下沉；**可减少 kwargs/`__slots__`** 拿到一部分（`unicodekeys_lookup_unicode`=1.68% 是它的指纹） |

**汇总**：可达 GE 图 ≈400–500 µs；靠**非 GE 手段**（短路 + 修缓存）可拿 ≈300–430 µs；
结构性不可省 ≈100–200 µs。⇒ **"908 µs 全靠 GE 抹掉"是错的。**

---

## 4. 对 `docs/10` 的审查结论（关键判断点 6 与产出要求）

| `docs/10` 的判断 | 我的结论 | 说明 |
|---|---|---|
| 判据 1：成本与 device 图化无关 | **支持** | 同步调用仅 0.27%，topdown 呈 frontend-latency 主导 |
| 判据 2：三前提（#1 接近满足 / #2 缺口 / #3 已满足） | **部分挑战** | #2 是必要不充分（官方"even when the graph is static"）；#1 是被 padding 强行钉住的；#3 的覆盖面很小（只有图捕获那一小段用 tensor） |
| 判据 3：决定不可图化 | **支持并强化** | 量化成下限 150–400 µs（由调度决策形态决定） |
| §5.2 估计 "2.76 → 1.2–1.5 ms" | **推翻** | 三处重叠计收益（GDN 去重 600 应为 ≈400；静态缓冲与去重重叠；`pad_graph_inputs` 不可去重）。**只靠 GE/图化的落点是 ≈1.6–1.9 ms**；1.2–1.5 ms 是"图化 + C++/账本级改造"的联合估计 |

**修正后的收益阶梯**（起点 2757.6 µs）：

```
S1 修 .replace() 缓存（非 GE）                        −130        -> 2628
S2 decode-only 快路径 + GDN 三组去重（非 GE）      −430 ~ −500    -> 2100–2200
S3 full-attn builder 去 tolist/pin_memory            −40 ~ −100   -> 2000–2160
S4 GE/kernel 化：device 段入图、slot_mapping 入图  −300 ~ −450   -> 1600–1850
S5 到 1.2–1.5 ms 需 C++ 化 metadata / 账本迁 device（架构改造）
```

**间接收益单独记账**（关键判断点 6）：GE 让 forward 变成 1 次 `aclmdlRIExecuteAsync`
确实能省 host，但**不计入 `prepare_input` scope**；且当 CPU 已是瓶颈（本例即如此）时，
它**不提高吞吐**。上限 ≈800 µs 的 CPU 预算（forward p50 830.6 µs），兑现条件苛刻。

---

## 5. 理论下限

| 层级 | 下限 | 由什么决定 |
|---|---:|---|
| T0 现状 | 2757.6 µs | 实测 |
| T1 非 GE 优化 | ≈1.9–2.2 ms | Python 记账与翻译的条数 |
| T2 + GE/kernel 化 | **≈1.6–1.9 ms** | 逐请求循环 + 图输入准备 |
| T3 架构级（账本迁 device / C++ 化） | **≈150–400 µs** | **调度决策的形态**（Python 对象 vs 定长张量） |

**原理上不可下沉的六项**已逐条列出（`_update_states` 的 dict/condense/swap、
`.item()` 决策、`_build_attn_state` 五路选择、图分派、`event.synchronize()`、
`SamplingMetadata`），并额外论证了"GE 支持图内 If/While 为什么在本场景仍不成立"。

---

## 6. 交付与遗留

**交付**：`docs/11-ge-whole-graph/03-ge-for-prepare-input.md`
（含：开头分层结论 → 四层逐层论证 → 15 行组件适用性分解表 → 与 ACLGraph 对比 →
理论下限 → 对 `docs/10` 的审查 → 9 条待验证项 + 2 条可行性待验证 → 证据索引 → 一页结论）。

**与同目录兄弟文档的一致性**（写完后交叉核对）：

- `01-ge-capability.md` §0 结论 3「整图下发省掉的不是 host 开销，而是**按算子计费**的 host 开销」
  与本文 L2 的判定一致；其结论 7「图外的 host Python 代码 GE 一个字都省不掉」与本文 L4 一致；
  其 §E3.3（TorchAir 为 FIA 提供收 Tensor 的 `npu_fused_infer_attention_score`，仅图模式可用）
  已被本文 §2.3 引用为"改算子接口是硬前提"的交叉证据。
- `02-why-not-enabled.md` §0.1 挖出 **v0.9.x–v0.11.0 曾有 `TorchAirGraph`（官方原话 "the GE graph mode"）
  于 v0.12.0rc1 被移除**（理由 "aclgraph is stable and fast now"），本文 §4.5 引用它作为
  "整图 vs 分段图"在本栈的一次真实工程对照，并明确标注它**不能**单独证明"GE 对 `prepare_input` 无用"。

**口径校正（提请注意）**：根任务书写的 `prepare_input` scope `1859–2098` 偏大，
实际结束于 **2082**（`_preprocess` 与 `update_cos_sin` 在 scope 外，
`[源码]` `model_runner_v1.py:2084–2098`）。引用占比时请按 1859–2082。

**未做的事（诚实声明）**：

1. 未跑任何实验，所有收益数字要么标 `[实测]`（引用 `docs/05`），要么标 `[推断]` 并给了验证方法（§7 V1–V11）；
2. `docs/05` 已标注的 GDN builder 残差 ≈123 µs/步 与未归因 207 µs 仍未解决，
   这两块是本文 T2/T3 估计的主要误差源；
3. 未评估 GE 的**编译/启动成本**（官方静态 kernel 编译为分钟~数十分钟级）对在线部署的影响——
   这是"要不要上 GE"的独立决策维度，建议由 `01-ge-capability.md` 覆盖。

**最高优先级的两个动作建议**（都不需要 GE，可立即验证）：

1. 修 `.replace()` 的缓存击穿（V2，预期 −130 µs，改动极小）；
2. 加 decode-only 快路径短路 `treat_single_token` + `split_decodes` + GDN 三组去重
   （V1/V3，预期 −430 ~ −500 µs）。
