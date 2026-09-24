# REPORT · torch / LAL host 开销公开调研（子任务：/root/torch_dispatch）

> 日期：2026-09-24（Asia/Shanghai）｜执行者：`torch_dispatch` agent
> 产出：`docs/10-cpython-directions/03-torch-dispatch.md`（579 行）＋本报告
> 约束遵守：只写 `/home/chiro/projects/vllm/preparing-input-phase/`；未登录远程、未跑 NPU、
> 未动 chip3、未起容器；`refs/vllm` 与 `HIST_PROJECT/vllm-ascend` 只读。

---

## 1. 一句话结论

**我们的两个热点，上游都已经独立发现了，而且都有一个 open 的修复 PR；同时我们手上还握着
一个上游没有发现的小 bug（`CommonAttentionMetadata.replace()` 让派生缓存永远落空）。**

| 结论 | 编号 | 状态（核对 2026-09-24） |
|---|---|---|
| "GDN builder 每步被调 3 次、~900 µs" | vLLM **#52297**（GPU 版，900 µs → 300 µs，BS=1 e2e +61%）<br>vllm-ascend **#16246**（Ascend 版，6.3 → 4.5 ms/步，−28.6%） | 两个都 **open 未合并** |
| 同一个问题的另一种解法 | vLLM #57962、#49730 | #57962 **closed 未合并**（与 #52297 撞车） |
| `.replace()` → 缓存落空（我们的独有发现） | 无 | vLLM main 今日仍是裸 `dataclasses.replace` |
| "要不要 torch.compile 元数据" | — | **有 4 条独立证据说明是负收益**，别再试 |

---

## 2. 交付物结构（`docs/10-cpython-directions/03-torch-dispatch.md`）

| 章节 | 内容 | 对应用户问题 |
|---|---|---|
| §0 结论速览 | 8 条可引用结论 | 全部 |
| §1 口径 | 证据分级 A/B/C + 本项目单次成本基准表 | — |
| §2 dispatcher | 调用链、µs/op 汇总表、官方快路径 API、ARM 是否异常的拆解 | 问题 1 |
| §3 torch.compile | guard/前图字节码/AOT wrapper 成本 + 6 条量化 + graph break + 结论 | 问题 2 |
| §4 图下发 | NVIDIA 官方天花板、vLLM/SGLang/Fireworks/TRT-LLM、Ascend ACLGraph/npugraph_ex | 问题 3 |
| §5 工程案例 | vLLM V1/V0.6/MRV2、SGLang（含 BCG）、TRT-LLM、Dynamo + 横向对比表 | 问题 4 |
| §6 上游动态 | 12 条直接命中 + 13 条相邻 + "我们重复了什么" + cherry-pick 判定 | 问题 5 |
| §7 可借鉴清单 | A1/A2/A3/B1/B2/C1/C2，每条给改动位置、µs 换算、风险、证据强度 | 产出要求 |
| §8 反面清单 | 7 条"不要再试" | — |
| §9 引用清单 | 72 条 URL，全部标注访问日期 | — |

---

## 3. 方法（供后人复用）

### 3.1 工具

- `gh` CLI（已登录 `chiro2001`，有 `repo` scope）：`gh api -X GET search/issues -f q='...'`
  检索 vLLM / vllm-ascend issue+PR，`gh api repos/<org>/<repo>/pulls/<n>` 查真实合并状态。
- `tool_search` → `mcp__exa__web_search_exa` / `web_fetch_exa`：非 GitHub 的官方文档/博客。
- `curl` 直取 `raw.githubusercontent.com`：核对"上游 main 现在到底长什么样"。
- 本地源码 grep：`refs/vllm`（0.26.0）、`HIST_PROJECT/vllm-ascend`（0.26.0rc1）、本项目 `data/model/*.csv`。

### 3.2 有效关键词（英文命中率远高于中文）

```
repo:vllm-project/vllm     "CPU overhead" in:title
repo:vllm-project/vllm     metadata construction in:title,body
repo:vllm-project/vllm     num_computed_tokens in:title,body
repo:vllm-project/vllm     GDN in:title
repo:vllm-project/vllm     "prepare_inputs" performance in:title
repo:vllm-project/vllm     "model runner v2" in:title
repo:vllm-project/vllm-ascend  GDN metadata in:title
repo:vllm-project/vllm-ascend  "CPU" in:title
```

**最有价值的两条**：`GDN in:title`（直接撞到 #52297/#57962/#49730）与
`repo:vllm-project/vllm-ascend GDN metadata in:title`（直接撞到 #16246）。

### 3.3 踩过的坑（避免重复）

1. **`gh api .../issues/<n>` 上的 `.pull_request.merged_at` 不存在**（issues 端点只给
   `url/html_url/diff_url/patch_url`）。我第一轮据此把 #57962/#46112/#36868 误标成"状态未知"，
   后来改用 `.../pulls/<n>` 的 `.merged` 才拿到真相：**#57962、#46112、#36868 都是 closed 未合并**。
   ⇒ 判"能不能 cherry-pick"必须查 `pulls` 端点。
2. **搜索 API 有 rate limit**（认证后 30 search/min）⇒ 一轮别超过 4 个 search 调用；
   `search/code` 更严，且不打 `.pyi.in` 之外的部分内容。
3. **很多"看起来是官方文档"的域名是镜像**（`vllm.website.cncfstack.com`、
   `sgl-project-sglang-93.mintlify.app`）。凡引用数字，我都尽量回到一手域名
   （`docs.vllm.ai`、`docs.sglang.io`、`sglang.io/blog`、`developer.nvidia.com`、`vllm.ai/blog`）。
4. 本机 **没有 torch**（`python3 -c "import torch"` → ModuleNotFoundError）⇒ 不能在本机复现微基准；
   但本项目已有 920B 上的 aarch64 微基准 CSV，直接用它们的数字更准（见报告 §4）。
5. 工作区是**多 agent 共享**的：期间出现了 `docs/10-prepare-input-graphification.md`（别人的产出）。
   写 §4.4 时我与之对齐了口径（"图化不减少 host 计算"），并在文档里做了交叉引用，避免两篇打架。
   后来又出现 `docs/10-cpython-directions/05-graph-dispatch-host-side.md`（同目录、另一路调研，
   76 KB，专攻"能不能图化"）。我的 §4 保持"只给公开数字"，并在开头加了指向 05 的交叉引用；
   若最终合稿，**§4 与 05 的第 4 章有重叠，建议以 05 为主线、本篇 §4 只保留 NVIDIA 官方天花板数字**。

---

## 4. 本项目数据的复用点（别人可能不知道它们在这）

| 数字 | 位置 |
|---|---|
| 920B aarch64，纯 ATen 派发 1.606 µs/call；`empty` 1.609；`zeros(3,8)` 2.301；`index_select` 1.02 | `data/model/microbench-torch.csv`（含 p10/p90/repeats，可直接引用） |
| numpy 小算子 0.40–1.20 µs、`_get_cumsum_and_arange` 8.08 µs | `data/model/microbench-numpy.csv` |
| 机器口径：aarch64 / Python 3.11.16 / torch 2.9.0+cpu / 单线程 / affinities | `data/model/microbench-meta.json` |
| 图化前后 `prepare input` 2.843 → 3.132 ms（+10%），单步 −87% | `docs/10-prepare-input-graphification.md` §1 |
| 单次小 NPU 算子 3–12 µs | `docs/03-share-in-inference.md:127`、`docs/05-hotspots.md` §5bis.5 |

> ⚠️ 引用时务必区分：**1.6 µs/pc 是纯 host、不含 device 下发**；**3–12 µs 是整链含下发**。
> 这两组数字被混用的话会得出完全错误的优先级。

---

## 5. 未完成 / 待验证（交给后续 agent）

1. **A1、A2 的收益没有真机验证**：本文给的是"派发次数 × 单次成本"的换算 + 上游 PR 的比例折算，
   **必须在 chip3 上做 A/B**（关图/开图两个口径，`--pystack-interval-us 0`）。
2. **`.replace()` 缓存问题的上游 PR 措辞**：缓存字段带 `WARNING: Deprecated`（注释说 v0.15.0 移除），
   提 PR 前要先问 maintainer 是"保留缓存"还是"直接删字段 + device 端一次算好"。
3. **Ascend 侧 `--async-scheduling` 的兼容矩阵**：公开材料只到"vLLM V1 的 async scheduling 与
   spec decode/结构化输出/PP 的历史限制"，**vllm-ascend 0.26.0rc1 上能否开、开完有没有隐藏 D2H，
   没有任何公开数字**，必须实测。
4. **MRV2 在 Ascend 的现状**：vllm-ascend 有 MRV2 适配 PR（#16432 等，open），但 MRV2 是 GPU-first 设计，
   是否/何时支持 GDN 混合模型未知。
5. **文中 B/C 级收益（B1/B2）需要 A/B 才有意义**：目前只有成本基准，没有端到端测量。

---

## 6. 三条"别再浪费时间"的提醒

1. 不要再花时间证明"`torch.compile` 能不能加速元数据构造"——§3 已有 4 条独立证据 + Ascend 禁用 Inductor。
2. 不要再花时间找"dispatcher 层的快路径"——那一层是 **100 ns** 量级，`no_dispatch` 之类在我们的
   eager 基线下收益为 0。
3. 不要用 #8985 的 diff 去 cherry-pick —— 该 PR closed 未合并，但**同样形态的代码已经在
   0.26.0rc1 里**了（`_copy_sequence_indices_to_device`）；直接改本地代码即可。
