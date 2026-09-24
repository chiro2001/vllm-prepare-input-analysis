# host 侧元数据构造能否图化 —— 调研工作簿（agent: graph_dispatch）

> 角色：**"host 侧元数据构造能否图化"** 支线的调研负责人（纯文献/网络 + 本地只读源码）。
> 交付物：`docs/10-cpython-directions/05-graph-dispatch-host-side.md`。
> 本文件是同一份调研的**工作簿**：做了哪些检索、结论怎么来的、哪些没查到、下一步怎么验证。
> 时间：2026-09-24（Asia/Shanghai）。所有 URL 访问日期均为 2026-09-24。
> 约束遵守：只写 `preparing-input-phase/`；未登录远程主机、未跑 NPU、未动 chip3、未起容器。

---

## 1. 一句话结论

**"图下发"解决的是"设备任务的下发"，而 `prepare_input` 的成本在"host 侧的构造"——
两者交集为空。** 在昇腾上，"图化省 host"这条路**被官方文档直接否定**：
ACL Graph 的 task update 官方原话是 *"updating tasks is more time-consuming than delivering tasks separately"*。
真正能省 host 的是"**静态缓冲 + 参数 tensor 化 + 纯算术下沉 device + 剩余账本编译化/C++ 化**"，
图捕获只是这条链末端的一小步。

## 2. 支撑这个结论的 5 条关键证据

| # | 证据 | 来源 | 强度 |
|---|---|---|---|
| E1 | 捕获对象是 **Stream 上的 device task**，host Python 不在 Stream 上 | CANN「ACL Graph 简介」；torch_npu NPUGraph 指南 | 官方（强） |
| E2 | **task update 比单独下发更耗时**；且 task 数量/类型必须与捕获时一致、不支持并发更新 | CANN「Building a Model Running Instance Based on the Capture Mode」 | 官方（强） |
| E3 | 捕获区内做 `query_start_loc.tolist()` → `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned` | vLLM issue #40807 | 上游 issue（强） |
| E4 | 有人真的把 slot-mapping 折进 decode cudagraph：字节级一致、无回归，**e2e 收益 = 0**（被 async scheduling 掩盖），作者结论"win 在把 `_prepare_inputs`/`_build_attention_metadata` tensor 化" | vLLM PR #47924（draft，已关闭） | 上游原型（强） |
| E5 | 上游 V2 runner 的官方解法是**用 Triton kernel 在 device 上准备 `input_ids`/`positions`/`query_start_loc`/`seq_lens`**，理由是 "avoids Python bottlenecks"；本地 `refs/vllm/vllm/v1/worker/gpu/input_batch.py` 可直接看到这些 kernel | vLLM「Model Runner V2 Design Document」+ 本地源码 | 官方 + 源码（强） |

## 3. Ascend 专项的两个澄清（原本容易误读）

### 3.1 `graph_task_update` 能传 Python list —— 但这不等于"builder 可以图化"

- CANN/torch_npu 的语义是"在 update stream 上**把同一个算子带新实参再下发一遍**"
  （torch_npu `_GraphDispatchMode.update_capture_record` 里就是 `op_cache_entry(*args, **kwargs)`）。
- vllm-ascend 的 FIA 更新路径确实在传 Python list：
  `attention_v1.py:611-640`（`seq_lens_list` / `actual_seq_lengths_q`）。
- **但**：① 官方说更新比下发更贵；② list 仍要在 host 上构造；③ 若 list 依赖 device 精确值，
  就必须做一次 device→host 读值（与捕获语义冲突）。
- ⇒ **可以"传 list"，不能"靠传 list 省 host"。**

### 3.2 `enable_enpu` 不是图能力，是 NPU 软切分虚拟化

- `ENPU` = **Enhanced NPU** = **vCANN-RT 软切分**（openEuler `ubs-virt`；`libvruntime.so` 预加载拦截 runtime，
  算力按 100 ms 时间片轮转），启动成功后设置进程级 `ENPU_ENABLE=True`。
- vllm-ascend 只是据此调整 **"图参数更新"与"模型前向"的顺序**（ENPU 要求 "record first, wait later"，
  否则会卡死），并在更新前加一次当前流同步（`model_runner_v1.py:2669-2725`、`acl_graph.py:257-263`）。
- ⇒ `enable_enpu=True` 换来正确性，代价是**每步多一次 sync**；与"更强的图更新能力"无关。

## 4. 检索清单（做过什么）

| 检索主题 | 主要落地来源 |
|---|---|
| CUDA Graph 节点参数更新边界 | CUDA C Programming Guide §4.2；CUDA Runtime API Graph 组；`oxicuda-graph`（二次源，仅用于 memcpy 字节数不可变一条的交叉核对） |
| Ascend ACL Graph / task group / task update 约束 | CANN 官方文档 4 篇（ACL Graph 简介、Building a Model Running Instance、Single-Stream Capture、`aclmdlRIExecuteAsync`）+ CANN runtime 仓库 api_docs + op-plugin/torch_npu 文档与源码 |
| ENPU 是什么 | openEuler ubs-virt `vcann-rt/README.md`、昇腾「NPU 虚拟化软切分参考实践」、vllm-ascend PR #8456 |
| 其他引擎 | TensorRT-LLM（C++ GPT Runtime / Architecture Overview / Piecewise CUDA Graph）、SGLang（v0.4 博客 + issue #19347/#27186 + PR #16194 + `overlap_utils.py`）、NVIDIA Dynamo（Overall Architecture / Router Design / Architecture Flow） |
| vLLM 上游动态 | PR #47924、issue #29134（+PR #29624）、issue #40807、PR #32815、PR #17866、RFC #20727、DBO 系列 #20448/#23693/#24845、V2 DBO RFC #50738 与 PR #51700、PR #28579、Model Runner V2 设计文档 |
| device 侧 metadata 的先例 | vLLM `block_table.py` 的 slot-mapping Triton kernel、`spec_decode/utils.py` 的 5 个 kernel、V2 `input_batch.py` 的 `prepare_*` kernel；FlashInfer `plan()` 与 CUDA Graph 的关系（issue #187） |
| 负面证据 | E2、E4、TRT-LLM "attention op substantial host-side overhead"、FlashInfer "plan() cannot be used in Cuda Graph"、SGLang #19347/#27186、vLLM #40807/#17866/#29134 评论、DBO 单机变慢 |

## 5. 本地源码核对点（本文引用到的行号）

- `vllm-ascend/vllm_ascend/attention/attention_v1.py`
  - 捕获：`:992/1016`、`:1121/1142`、`:1195/1208`（`graph_task_group_begin/end`）
  - 更新：`:476`（`update_graph_params`）、`:525/538`（paged）、`:619/640`（FIA）、`:809/842`
  - list 形参：`:332-333`（`tolist()`）、`:611-640`（更新传 list）、`:1024-1060`（捕获传 tensor kv len）
- `vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py`：`:226`（`_cudagraph_support=UNIFORM_BATCH`）、
  `:332-369`（写预分配缓冲 = 正确的"近图"形态）、`:371-376`（spec 分支清零）、`:531`（`build` 主体）
  ；配套 `ops/triton/fla/utils.py:22-37`（host 侧 Python 循环构造 chunk 索引）
- `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`：`:469-472`（ENPU）、`:1246-1250`（slot_mapping 调用）、
  `:2669-2725`（update vs forward 的顺序）
- `vllm-ascend/vllm_ascend/compilation/acl_graph.py`：`:93-122`、`:230-290`
- `refs/vllm/vllm/v1/worker/block_table.py`：`:153-190`（`compute_slot_mapping` + `commit_block_table`）、
  `:346-380`（Triton kernel 与"Pad remaining slots for CUDA graph compatibility"）
- `refs/vllm/vllm/v1/worker/gpu_model_runner.py`：`:1937+`（`_prepare_inputs`）、`:1956`（先拷块表以重叠）、
  `:1761+`（`_prepare_input_ids` async 分支）、`:2073-2095`（async 的事件同步）
- `refs/vllm/vllm/v1/attention/backend.py`：`:600-627`（`AttentionCGSupport` 四档 + `supports_update_block_table`）
- `refs/vllm/vllm/v1/worker/gpu/model_runner.py:874` + `refs/vllm/vllm/v1/worker/gpu/input_batch.py`（V2 device 侧准备）
- `refs/vllm/vllm/config/vllm.py:512`（`max_concurrent_batches`：async → 2）、
  `refs/vllm/vllm/config/scheduler.py:158`（`async_scheduling`）

## 6. 没查到的（与正文 §9 对应，摘录）

1. ENPU 在昇腾官方文档中**无独立词条**（只有 vCANN-RT 软切分语境下的 `ENPU_ENABLE`）。
2. `aclmdlRICaptureTaskUpdate*` 的**定量**开销（只有"比单独下发更耗时"的定性）。
3. CUDA `cudaGraphExecKernelNodeSetParams` 与普通 launch 的**成本对比基准**（官方只说比重新 instantiate 轻）。
4. 昇腾上"图更新改变形状"是否可行（官方只约束"任务数量与类型一致"）。
5. vLLM 上游**没有**"attention metadata builder 整体下沉 device"的 RFC/PR；只有 PR #47924 的 B.2 规划
   （且该 PR 已关闭）与 MRV2 的 `prepare_*` kernel。
6. GDN/linear-attention 的 metadata 图化实践：公开资料为零，本文 §5.4 的判断是"源码 + 通用约束外推"。
7. `graph_task_update` 传 list 的底层代价（是否每步变 device tensor / 有无隐式 H2D / 是否重算 tiling）。
8. SGLang 是否有"metadata 在 device 构造"的完整方案（只有 `needs_cpu_seq_lens` 这类按需 D2H）。
9. Dynamo 是否干预引擎内 per-step 元数据（文档未涉及；正文按 `[推断]` 标注）。
10. Npugraph_ex 静态 kernel 编译对 host 元数据路径的收益（官方未涉及）。

## 7. 建议的下一步验证（3 条，与正文附录一致）

1. **V3（最高价值）**：在 `update_graph_params` 加探针，测 6 层 FIA 的 update 总时长。
   判据：>100 µs ⇒ "图化省 host"在本平台**证伪**，可从主线划掉。
2. **去冗余 + 缓存修复**：3 个 GDN builder 合成 1 个；修 `_treat_single_token_prefills_with_state_as_decodes`
   的 `.replace()` 导致 `_num_computed_tokens_cache` 失效。预期 ~730 µs（占 `prepare_input` 26%）。
3. **形态改造**：把 GDN/full-attn 给图用的量全部写进持久缓冲；把 `positions`/`token_indices`/`seq_lens`
   这类纯算术换成 device kernel（语义可抄 `input_batch.py::prepare_pos_seq_lens`）。

## 8. 交付状态

- ✅ `docs/10-cpython-directions/05-graph-dispatch-host-side.md`（正文，含 §7 可图化分解表、§8 1 ms 临界判断、§9 查不到的）
- ✅ 本文件 `agents/graph_dispatch/REPORT.md`（工作簿）
- ⛔ 未做（超出授权/环境）：任何 NPU 实测、远程登录、容器操作
