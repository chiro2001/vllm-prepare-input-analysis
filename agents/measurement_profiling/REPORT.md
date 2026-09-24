# 真机 perf / libkperfx 采集 —— 交接报告

**Agent**: `/root/measurement_profiling`
**状态**: P0 完成（真机火焰图 + children 分解 + topdown/IPC）；P1/P2 未做（见 §6）
**工况**: a3-22 chip3 `/dev/davinci3`，Qwen3.5-0.8B BF16 TP1，
**B=1 / ISL=128 / decode 稳态 / graph `FULL_DECODE_ONLY` / MTP off / async on**
**engine-core 宿主 TID**: `704908`（`npu-smi info -t proc-mem -i 1 -c 1`）

---

## 0. 结论速览（三问的答案）

1. **`AscendGDNAttentionMetadataBuilder.build` 的 children 分解与插桩对得上**：
   perf 独立测得它占该线程样本的 **20.10%**，其中
   `_pad_non_spec_decode_graph_inputs` **5.94%**、`_attach_non_spec_decode_metadata`
   **4.04%**；而全注意力侧 `AscendAttentionMetadataBuilder.build` 只有 **1.87%**。
   ⇒ **9.6 倍差距**，与插桩"3 个 GDN group × 371 µs/步"的结论一致。
2. **`compute_num_computed_tokens` 的 129 µs（43 µs/call）是 CPU 真忙，不是 device 等待**：
   其子树 67.50% 是 `PyNumber_Subtract`、66.15% 是 `THPVariable_sub`、
   11.54% 是 `slice_Tensor`，而 D2H 同步占 **0.00%**。
   ⇒ **缓存被 `.replace()` 击穿，每次重算**（详见 §3）。
3. **topdown/IPC**：`frontend_bound 66.01%`（`frontend_latency_bound 59.89%`）、
   `bad_spec 11.03%`、`retiring 12.85%`、`backend_bound 10.11%`、**IPC 0.771**，
   9/9 组 `confidence = 1.0000`。历史基线 55.99–65.00% / 0.719–0.890 → **复现成功**。

---

## 1. 采集口径（可复现）

| 项 | 值 |
|---|---|
| 镜像 | `local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler` |
| 启动 | `launch_subscope_service.sh --mode baseline --serve-only --extra-docker-args "--pid=host -v /tmp:/tmp -e PYTHONPERFSUPPORT=1" ...` |
| 容器 | `pi-phase-chip3-prof-real-b1`（无探针、无 pystack） |
| 流量 | `scripts/measure/drive_traffic.sh`（4–6 轮 × 1700 output tokens，绑核 120,121） |
| perf | `perf record -F 999 -g --call-graph fp -t 704908`，20 s，**19 569 样本 / 0 lost** |
| PMU | `pmu_tid_sweep.py --tid 704908 --groups 1-9 --window 3`，**9/9 confidence=1.0** |
| 互斥 | perf 与 PMU 串行执行，无重叠；结束后 `docker rm -f` + 释放 chip3 锁 |

**为什么用 `fp` 而不是 `dwarf`**：`PERF_SYMBOLS.md` 实测 dwarf 同窗口 128.7 MB
且 **Python 帧覆盖率 0%**；fp + trampoline 只有 5.0 MB 且 100%。
trampoline 的 +7.4% 解释器开销用独立来源规避：占比取自 PMU/LiteProfiler/探针，
火焰图只用于调用链归因（`docs/05` §0 已显式声明）。

**一个会静默毁掉采集的坑（已固化进工具）**：`max_model_len=2048` 会以 HTTP 400
拒绝 `max_tokens=8192`，引擎随即空转 → `perf.data` **0 样本**。第一次采集就是这样
失败的；`drive_traffic.sh` 改用多轮 ≤1700 token 的请求。

---

## 2. Q1：`AscendGDNAttentionMetadataBuilder.build` 的 children 分解

数据：`data/profiles/real-b1/fp/breakdown/AscendGDNAttentionMetadataBuilder.children.txt`。
单位 = **该 TID 全部样本的 %**（不是占 `prepare input` 的 %）。

| 符号 | children % | self % |
|---|---:|---:|
| `py::AscendGDNAttentionMetadataBuilder.build` | **20.10** | 0.00 |
| ├ `._pad_non_spec_decode_graph_inputs` | **5.94** | 0.00 |
| ├ `._attach_non_spec_decode_metadata` | **4.04** | 0.00 |
| └ `._attach_spec_decode_metadata` | 0.03 | 0.00 |

对照：`py::AscendAttentionMetadataBuilder.build` = **1.87%**。

### 2.1 与插桩六段的对照

| 插桩段（µs/步，3 次调用合计） | 插桩占 prepare_input | perf 对应帧 | perf 占线程 |
|---|---:|---|---:|
| `pad_graph_inputs` 285.9 | 8.7% | `_pad_non_spec_decode_graph_inputs` | **5.94%** ✅ 同为最大子步 |
| `attach_decode` 40.2 + `attach_prefill` 2.8 | 1.3% | `_attach_non_spec_decode_metadata` | **4.04%** |
| `compute_num_computed_tokens` 129.5 | 3.9% | 同名（core `CommonAttentionMetadata`） | **2.66%** |
| `split_decodes` 121.1 | 3.7% | `split_decodes_and_prefills` | 2.55% |
| `build_actual_seq_lengths` 163.0 | 4.9% | `_build_actual_seq_lengths` | 见 `subtree-gdn` |
| `treat_single_token` 169.3 | 5.1% | `_treat_single_token_prefills_with_state_as_decodes` | 见 `subtree-gdn` |
| **GDN build 合计** | 33.8% | **`AscendGDNAttentionMetadataBuilder.build`** | **20.10%** |

**换算自洽**：若 20.10% 对应 1113.8 µs，则步墙钟 ≈ 5541 µs，
与历史 0.8B graph 步周期 4.9 ms、本轮 `mean_itl ≈ 4.5 ms` 一致。

**唯一显著分歧**：`_attach_non_spec_decode_metadata` perf 测 **4.04%**（≈224 µs/步），
插桩只记到 **45 µs/步**。差值 ≈ 180 µs/步落在**未被探针覆盖的 attach 内部代码**
（`_copy_sequence_indices_to_device` / `spec_state_indices_tensor.copy_` 等），
比插桩自报 residual（123 µs/步）更大。**建议按本文符号列表补探针。**

---

## 3. Q2：`compute_num_computed_tokens` 的 43 µs/call 到底是什么

**答：纯 CPU 忙，且原因是"缓存被击穿"——不是 device 等待，也不是单纯"缓存返回"。**

数据：`data/profiles/real-b1/fp/subtree-cnct/summary.json`（520 样本 = 2.66%）。

### 3.1 判定"忙 vs 等"

| 叶帧类别 | % of 该子树 |
|---|---:|
| **`d2h_sync`**（`aclrtSynchronize` / `_local_scalar_dense`） | **0.00** |
| `ascend_driver` | 1.35 |
| `other`（主要为 `libtorch_npu` 未解析叶） | 48.46 |
| `libc` | 15.58 |
| `cpython_runtime` | 12.31 |
| `aten_op_dispatch` | 9.23 |
| `kernel` | 7.50 |
| `allocator` | 3.85 |

⇒ **没有等待**。对照 `AscendGDNAttentionMetadataBuilder.build` 整体也只有
`d2h_sync 0.15%` / `ascend_driver 1.16%`：
**GDN 路径的 371 µs/call 是派发密集，不是等卡**（同时回答了插桩的另一个疑问）。

### 3.2 真正的原因：缓存未命中

| 子树符号 | % of 子树 | 含义 |
|---|---:|---|
| `PyNumber_Subtract` | **67.50** | Python `-` 运算符 |
| `torch::autograd::THPVariable_sub` | **66.15** | torch 张量 `__sub__` |
| `at::_ops::slice_Tensor::call` | 11.54 | `qsl[1:]` / `qsl[:-1]` |
| `at::_ops::sub_Tensor::call` | 10.97 | 张量相减 |

`vllm/v1/attention/backend.py:530`：

```python
def compute_num_computed_tokens(self) -> torch.Tensor:
    if self._num_computed_tokens_cache is None:
        query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
        self._num_computed_tokens_cache = self.seq_lens - query_lens
    return self._num_computed_tokens_cache
```

perf 证明每步走的是**未命中分支**：`gdn_attn_builder.py:541` 的
`m = _treat_single_token_prefills_with_state_as_decodes(common_attn_metadata)`
经 `.replace()` 构造新对象，`_num_computed_tokens_cache` 随之回到 `None`，
于是 3 个 GDN builder 各重算一次（4+ 次 torch 算子派发/次）。

### 3.3 可直接落地的优化

* 让 `CommonAttentionMetadata.replace()` 保留 `_num_computed_tokens_cache`
  （或让 GDN builder 复用传入对象而不新建）；
* 或用 `np.diff(query_start_loc_cpu)` 等**宿主侧**运算取代 torch 张量相减。

预期收益：**≈43 µs/call × 3 call/步 ≈ 130 µs/步**（约 2.3% 步墙钟）。

---

## 4. Q3：topdown + IPC

数据：`data/profiles/real-b1/pmu/{topdown.json,libkperfx.json,manifest.json}`。
`min_confidence = 1.0`、`all_confident = true`、
`time_enabled == time_running`（2 850 546 620 ns，组 1）、`useronly_forced = false`。

| 分量 | 真机（本轮） | 历史 a3-21（graph MTP-off） |
|---|---:|---|
| frontend_bound | **66.01** | 55.99–65.00 |
| bad_spec | 11.03 | 9.29–10.69 |
| retiring | 12.85 | 11.98–14.83 |
| backend_bound | 10.11 | 11.73–21.11 |
| **IPC** | **0.771** | 0.719–0.890 |

子项（历史从未采集，本轮填补空洞 G2）：
`frontend_latency_bound 59.89`、`frontend_bandwidth_bound 6.12`、
`core_bound 6.64`、`mem_bound 2.97`、`mem_l1_bound 1.15`、`mem_l2_bound 0.48`、
`mem_l3_dram_bound 1.33`、`ptag_stall 69.52`、`mapq_stall 24.47`。

> ⚠️ **L3/DRAM 计数器本机恒为 0**（`zero_counting_events` 含 `MEMSTALL_L3MISS`、
> `DRAM_LOCAL`、`DRAM_REMOTE`、`DRAM_REMOTE_CACHE`），所以 `mem_l3_dram_bound`
> 实际承载"L2 miss 及以下"的全部 stall，**不能读成"DRAM bound = 1.33%"**。

**判读**：frontend-latency 主导 + IPC 0.77（远低于 6 宽发射）+ 访存 < 3%
⇒ `prepare input` 是 **CPython 解释器/派发密集**，不是计算密集、不是访存密集、不是等卡。

---

## 5. 附带收获：历史空洞 G1 闭合

`perf script --ns`（启动后纳秒）经 `/proc/stat` 的 `btime` 换算成 epoch，
与 LiteProfiler 的 epoch 微秒对齐后，可把样本归入**最内层 scope**
（`scripts/measure/scope_attribution.py`；产物 `data/profiles/real-b1/fp-scope/scope-attr/`）：

| 通道 | 结果 |
|---|---|
| 采样归因 | `prepare input` 内样本 **10 761 / 19 564 = 55.00%** |
| 墙钟区间（并集，裁剪到采集窗口） | **54.88%** |

两者差 **0.12 pp**，互相验证了 scope 归属与时钟换算；也与
`subscope_instrumentation` 用探针 + `Step:Schedule` 分母独立测得的
**55.3%（A 臂）/ 57.2%（C 臂）** 一致。

**被证伪的预测**（`docs/01`/`docs/02` 列为前三大成本来源）：
`pin_memory` **0.12%**、`tolist` **0.07%**、`index_select` **0.07%**、
`aclrtSynchronize` **0.18%** —— 合计 **< 0.5%**。

**被证实的结构**：`_build_attention_metadata` **31.60%** >
`_prepare_inputs` **20.26%**；`_update_states` 仅 **2.28%**（稳态无 churn）。

**热点源码行/指令**（`hotspots_srcline.csv` / `hotspots_instr.csv`）：
前三行都落在 `unicodekeys_lookup_unicode`（`dictobject.c:940/968/949`），
即 CPython **字符串键字典查找**；第 4/5 行是 `object.h:642/646`（`Py_INCREF`）。
⇒ 热点落在解释器对象模型，不在业务算法。

---

## 6. 未完成 / 限制（如实列出）

| 项 | 状态 | 原因 |
|---|---|---|
| B=64 decode 点 | **未做** | 属 P1；单点采集 + 四层分析已用掉主要预算 |
| chunked prefill 点 | **未做** | 同上 |
| `_update_states` 高 churn 点 | **未做** | 同上；稳态下仅 2.28% |
| 无 trampoline 的对照点 | **未做** | 火焰图只用于归因；占比已由 PMU / 探针 / 墙钟三来源独立给出 |
| `libopenblas` 等的 annotate | 部分 | 该 DSO 无 debug info，CSV 留空并在 manifest 标注（未伪造） |
| `perf.data` | **留在 a3-22** | 按 `plan/COORDINATION.md` §5 不进交付包 |

**口径精确说明**：本文的 20.10% / 5.94% / 2.66% 是**该线程 on-CPU 样本占比**，
插桩与 phase 表给的是 **µs/步**。换算依赖"线程几乎全程在核"这一假设，
本轮 `d2h_sync` 极低、`frontend_bound` 高，支持该假设；换算已在 §2.1 标注。

---

## 7. 产物清单

| 路径 | 内容 |
|---|---|
| `data/profiles/real-b1/fp/hotspots.csv` | self top-30（19 569 样本） |
| `data/profiles/real-b1/fp/hotspots_srcline.csv` | 热点**源码行** top-20 |
| `data/profiles/real-b1/fp/hotspots_instr.csv` | 热点**指令** top-20 |
| `data/profiles/real-b1/fp/breakdown/*.children.txt` | 27 个符号的 children 分解 |
| `data/profiles/real-b1/fp/subtree-{gdn,pad,attach,cnct,attn}/summary.json` | 子树叶帧分类 + 直接 callee + 子树 |
| `data/profiles/real-b1/fp-scope/scope-attr/summary.json` | G1：scope 归因 55.00% |
| `data/profiles/real-b1/pmu/{topdown,libkperfx,manifest}.json` | 9 组 topdown + IPC + 置信度 |
| `data/profiles/real-b1/{topdown.csv,profile_index.json}` | 机器可读汇总 |
| `figures/05-flame-real-b1.svg` | 真机全线程火焰图（含 Python 帧） |
| `figures/05-flame-real-b1-scope.svg` | **只含 `prepare input` scope 内样本**的火焰图 |
| `runs/prof-real-b1/run_manifest.json` | 容器/镜像/TID/绑核/端口 |
| `scripts/measure/*` | 可复现脚本（见 §8） |

## 8. 复现命令

```bash
# 1) 起环境（a3-22，chip3）
bash scripts/launch_subscope_service.sh --run-id prof-real-b1 --mode baseline \
  --serve-only --extra-docker-args "--pid=host -v /tmp:/tmp -e PYTHONPERFSUPPORT=1" \
  --model-key qwen35-08b --chip 3 --pystack-interval-us 0 --max-num-seqs 1 \
  --max-model-len 2048 --max-num-batched-tokens 2048 --cudagraph-mode FULL_DECODE_ONLY
TID=$(npu-smi info -t proc-mem -i 1 -c 1 | awk -F: '/Process id/{gsub(/[^0-9]/,"",$2);print $2}')

# 2) 流量 + perf（流量必须覆盖整个采集窗口）
bash scripts/measure/drive_traffic.sh --base-url http://127.0.0.1:18100 \
  --model qwen35-08b --outdir runs/prof-real-b1/requests/x --rounds 6 \
  --prompt-tokens 128 --max-tokens 1700 --cpus 120,121 &
sleep 8
bash scripts/measure/perf_capture.sh --tid "$TID" --seconds 20 --freq 999 \
  --call-graph fp --name real-b1 --outdir data/profiles/real-b1/fp

# 3) children / 子树（必须在容器销毁前做：Python 帧靠容器内 /tmp 的 perf map）
python3 scripts/measure/build_breakdown.py --perf-data data/profiles/real-b1/fp/perf.data \
  --outdir data/profiles/real-b1/fp/breakdown
python3 scripts/measure/subtree_breakdown.py --perf-data data/profiles/real-b1/fp/perf.data \
  --match AscendGDNAttentionMetadataBuilder.build --outdir data/profiles/real-b1/fp/subtree-gdn

# 4) topdown（与 perf 串行，不共存）
python3 scripts/measure/pmu_tid_sweep.py --tid "$TID" --groups 1-9 --window 3 \
  --outdir data/profiles/real-b1/pmu

# 5) scope 归因（先 /start_profile，并让流量覆盖采集窗口）
python3 scripts/measure/scope_attribution.py --perf-data data/profiles/real-b1/fp-scope/perf.data \
  --lite-log data/profiles/real-b1/lite.log --scope "prepare input" \
  --outdir data/profiles/real-b1/fp-scope/scope-attr
```
