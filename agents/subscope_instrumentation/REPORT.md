# subscope_instrumentation — 子阶段插桩报告

> 状态：**已完成**。S1 5 臂 + S2 6 臂 + 3 轮第二层插桩（C2b/C2c/C2d）+ NPU 派发微基准。
> chip3 锁已释放（2026-09-24 19:30Z），交给 `measurement_profiling`。
> Owner: `subscope_instrumentation`。所有数字都有 `runs/<run_id>/` 原始日志与
> `data/subscope/<run_id>/` 派生表支撑。

## 结论速览（可直接引用）

1. **`prepare input` 内部最大的结构性热点是 attention metadata 构建 = 1.22 ms/步
   （占 `prepare input` 37%）**，其中 **GDN（线性注意力）路径 1.11 ms、占 33.8%**，
   full-attention 路径只有 124 µs（3.8%）。
2. **Qwen3.5-0.8B 每步调用 4 次 `builder.build()`：3 次 GDN（各 6 层，3 个
   Mamba KV group）+ 1 次 full-attn（6 层）**。GDN 的 3 次调用是 3 个不同
   `kv_cache_gid`，不是同一个组重复。
3. **GDN build 的 371 µs/call 不是计算，是下发密集**：三个只有 3 个 NPU 算子的
   段合计 193 µs/步；而 `compute_num_computed_tokens` **只返回一个已缓存的
   tensor** 却要 43.2 µs/call。微基准给出关键常数：**920B 上一次小 NPU 算子
   3–12 µs**。
4. **`attn_mask` 不是 O(S²) 重建**：实测走缓存分支，该子步 1.24 µs。已排除该 bug。
5. **pystack 采样器对 `prepare_input` 的污染 = +20~21%**（两场景一致），
   远小于它对 ITL 的 +60%。见 `docs/09-historical-data-caveat.md` §5。
6. **占比随 batch 反直觉**：B=1 时 `prepare input` 占 decode step **55.3%**，
   B=64 时降到 **38.7%**——它的固定成本占比高、随 batch 增长慢
   （batch ×64，它只涨 ×1.40）。**低并发才是最需要优化的区域。**

## 0. 交付物

| 路径 | 内容 |
|---|---|
| `instrument/pi_subscope.py` | 探针运行时（`pi_scope` / `pi_prepare_input` / `pi_step_info` / `pi_note`） |
| `instrument/patched/model_runner_v1.py` | 插桩版 runner（49 探针），AST 等价性已验证 |
| `instrument/patched/attention_v1.py` | 基类 attention builder 探针（8 探针） |
| `instrument/patched/sfa_v1.py` | **SFA builder 探针**（6 探针，覆盖 `build()` 的另一条实现路径） |
| `instrument/baseline/*.image` | 三个文件的镜像内基线副本（sha256 见 `apply.sh`） |
| `instrument/verify_patch.py` | AST 等价性证明器（剥掉探针后与基线逐节点相同） |
| `instrument/probe_overhead.py` | 探针开销实测 |
| `instrument/test_pi_subscope.py`, `instrument/test_parse_subscope.py` | 自测 |
| `instrument/apply.sh`, `instrument/revert.sh`, `instrument/README.md` | 应用/还原/说明 |
| `scripts/launch_subscope_service.sh` | **公共启动器**（公共复用入口，见 §6） |
| `scripts/run_subscope_matrix.sh` | 6 臂实验驱动器 |
| `scripts/compare_overhead.py` | A/B/C/D1/D2 阶梯汇总 |
| `scripts/parse_subscope.py` | 原始日志 → per-step/summary/group/attn 分解表 |
| `scripts/plot_subscope.py` | 堆叠柱状图 |
| `scripts/run_manifest.py` | `runs/<id>/run_manifest.json`（含 engine-core PID/TID） |

## 1. 插桩手法与代价（实测）

**方式**：不可侵入挂载覆盖。三个 `.py` 用 `-v ...:ro` 覆盖镜像内副本，不重建镜像、
不写只读参考目录。`verify_patch.py` 用 AST 证明"剥掉探针后与基线逐节点相同"——
比"能编译"强得多，能挡住缩进/控制流被误改。

**关闭态零开销**：`PI_SUBSCOPE` 未设为 `on` 时 `pi_prepare_input()` 直接返回调用方
的 context manager（不包一层），`pi_scope()` 返回共享 `nullcontext`。

**探针自身开销**（a3-22，Kunpeng 920B，`taskset -c 200-203`，`PI_SUBSCOPE=on`）：

| 量 | 值 |
|---|---|
| 单探针（enter+exit，含两次 `perf_counter_ns` 与栈维护） | **2.20 µs** |
| 空循环基线 | 0.76 µs |
| 单探针边际成本 | 1.44 µs |
| 整步（`_Step` 包装 + 22 探针 + emit） | 63.0 µs |
| 某轮实测每步探针数 | 49 |
| 纯探针地板估计（49 × 2.20） | **108 µs/step** |

**实测代价**（`prepare input` 中位数差，不是估计）：

数值用 `scripts/extract_ladder.py` 的**独立 p50**（锚在 `prepare input` 行上，
免疫 `Step:Model` 双行问题）：

| 场景 | B−A 挂载/import | **C−B 探针** | D1−B 采样器污染 | D2−B 噪声底 |
|---|---|---|---|---|
| S1（B=1, ISL=128） | +48.6 µs (+1.76%) | **+266.4 µs (+9.49%)** | +678.6 µs (+24.18%) | +103.3 µs (+3.68%) |
| S2（B=64, ISL=128） | −11.4 µs (−0.29%) | **+259.1 µs (+6.71%)** | +944.4 µs (+24.45%) | +98.4 µs (+2.55%) |

**结论**：C−B 在 S1 上 9.49%、S2 上 6.71%，**超过任务设定的 5% 预算**。
原因是探针**数量**（第一层 49 个、C2d 全量 66 个）而非单探针成本：
微基准给出单探针边际成本 1.44 µs、整步包装 + emit 63.0 µs，49×2.20 ≈ 108 µs 是地板。
报告引用子 scope 数字时必须扣除该地板，`parse_subscope.py --probe-floor-us 2.2`
会把低于地板的 scope 列进 `meta.json.scopes_below_probe_floor`。

**噪声底说明**：D2−B 为 +103 µs（S1, +3.68%）/ +98 µs（S2, +2.55%），
**未超过 5%**，因此不需要再加重复臂；但它解释了为什么单次 A/B 差在 100 µs 量级上
不可靠——凡引用小于 ~100 µs 的差值，都必须说明它在噪声底附近。

## 2. 子阶段分解（S1: B=1, ISL=128, decode, graph, pystack off）

`prepare input` 墙钟 p50 = **3072.6 µs**（C 臂），未归因 219 µs（7.0%）。

| 子 scope | p50 µs | 占 prepare input |
|---|---:|---:|
| `pi: am.builder_build`（4 次/步） | 939.6 | 30.9% |
| `pi: build_attention_metadata` | 280.6 | 9.2% |
| `pi: in.slot_mapping` | 134.6 | 4.4% |
| `pi: in.positions_assembly` | 109.8 | 3.6% |
| `pi: sync_input_prep` | 100.0 | 3.3% |
| `pi: update_states` | 99.6 | 3.3% |
| `pi: in.block_table_commit` | 86.2 | 2.8% |
| `pi: am.cm_base_pre` | 74.9 | 2.5% |
| `pi: in.prepare_input_ids` | 66.5 | 2.2% |
| `pi: am.group_loop` | 65.9 | 2.2% |
| `pi: in.tokens_to_gpu` | 61.1 | 2.0% |

**一级发现**：`attn metadata` 组（`builder_build` + `group_loop` + `cm_base_pre` + …）
合计约 **1.4 ms/step，占 prepare_input 的 ~45%**，是 prepare_input 内部最大项，
且**在 120 步窗口内是固定成本**（first-20% vs last-20% 无变化）。

## 3. 第二层：`pi: am.builder_build` 内部（S1-C2，4 次调用/步）

`pi: am.builder_build`：4 次/步，**总 self 939.6 µs**，单次最大 319.5 µs
→ 量级是"4 组各 200–320 µs"，不是"1 大 3 小"。

只挂 `attention_v1.py` 探针时，其 8 个子步只解释 82 µs，**残余 940 µs（92%）**：

| `pi: am.build.*` 子步 | p50 µs | 占 builder |
|---|---:|---:|
| `qsl_h2d`（`query_start_loc_cpu.pin_memory().to(device)`） | 31.5 | 3.1% |
| `backend_metadata` | 14.5 | 1.4% |
| `tolist`（`actual_seq_lengths_q` / `seq_lens_list`） | 9.1 | 0.9% |
| `split_decodes` | 8.8 | 0.9% |
| `seq_lens_select` | 8.2 | 0.8% |
| `metadata_ctor` | 6.6 | 0.6% |
| `fia_pad` | 1.9 | 0.2% |
| `attn_mask` | 1.2 | 0.1% |
| **未归因残余** | **940.3** | **92.0%** |

**残余的来源已定位（静态 + 待 C2b 实测确认）**：`pi: am.builder_build` 包的是
`builder.build(...)`，而 `AscendSFAMetadataBuilder.build()`（`sfa_v1.py:300`）
是**另一条实现路径**，它调用 `self._build(...)` 而**不会**调用基类
`AscendAttentionMetadataBuilder.build()`。因此基类里那 8 个探针对 SFA 组根本不触发，
残余就落在 SFA 的 `_build()` 里。`patched/sfa_v1.py` 已为此加了 6 个探针
（`pi: sfa.{build,slice_inputs,seq_lens_select,get_cos_sin,store_kv_block_metadata,metadata_ctor}`）。

**attn_mask 路径（回答"是不是 O(S²) 每步重建"）**：不是。
`attention_mask.py:get_attention_mask(causal, model_config)`：

```python
if not causal:               return None
if runner_type == "pooling": return self.get_attn_mask(2048, torch.bool)   # 缓存 attn_mask_cache
return self.get_splitfuse_attn_mask()                                      # 缓存 chunked_prefill_attn_mask
```

我们的场景是 causal + `runner_type=generate` → 命中**缓存的
`get_splitfuse_attn_mask()`**。实测该子步 **1.18 µs**（= 一次属性读取 + 返回），
与"缓存命中"完全一致；若走未缓存的 `get_attn_mask`，会看到 O(S²) 的
`_generate_attn_mask`（2048×2048）每步重建，量级应为百微秒到毫秒。**排除该 bug。**

## 4. 历史数据口径修正（pystack 污染）

| 场景 | 历史（含污染） | 本次新测 | 污染量 | 修正后 | 修正比例 |
|---|---|---|---|---|---|
| S1 B=1 decode，prepare_input | 3.132 ms | D1 3.485 ms / B 2.806 ms | **+678 µs (+24.2%)** | ≈ **2.81 ms** | −10.3% |
| S1 B=1 decode，step | 5.661 ms | D1 4.948 / B 4.029 | +919 µs | ≈ 4.03 ms | −28.8% |
| S2 B=64 decode，prepare_input | — | D1 4.807 / B 3.862 | **+944 µs (+24.5%)** | ≈ **3.86 ms** | −19.6% |
| S2 B=64 decode，step | — | D1 6.621 / B 5.339 | +1281 µs | ≈ 5.34 ms | −19.4% |

要点：**pystack 对 prepare_input 的污染是 +24%（S1/S2 一致），
远小于 service_bringup 对 ITL 观察到的 +60%**。说明采样器的更大代价落在
prepare_input 之外（很可能在 forward/sample 的 Python 段：每步
`sys._current_frames()` + 栈格式化 + CSV 落盘，且它采样所有线程并与主线程抢 GIL）。

## 5. 插桩开销自查（AST 等价性）

```
$ python3 instrument/verify_patch.py instrument/baseline/model_runner_v1.py.image \
      instrument/patched/model_runner_v1.py
EQUIVALENT: patched/model_runner_v1.py == baseline/model_runner_v1.py.image + probes
$ python3 instrument/verify_patch.py instrument/baseline/attention_v1.py.image \
      instrument/patched/attention_v1.py
EQUIVALENT: patched/attention_v1.py == baseline/attention_v1.py.image + probes
$ python3 instrument/verify_patch.py instrument/baseline/sfa_v1.py.image \
      instrument/patched/sfa_v1.py
EQUIVALENT: patched/sfa_v1.py == baseline/sfa_v1.py.image + probes
```

## 6. 公共启动器（给 root 审计的 diff 摘要）

`scripts/launch_phase_service.sh` 的改动**只有 4 处**，全部 opt-in、默认零行为变化：

1. 文件头注释新增 `PREPARE_INPUT_EXTRA_DOCKER_ARGS` 说明（标注由
   prepare_input 子 scope 实验引入、默认关闭）；
2. 把原来紧跟 `DOCKER_ARGS+=(` 的"镜像 tag + `vllm serve ...` 命令"拆成独立数组
   `DOCKER_CMD_ARGS`（**逐字不变**，只是搬进另一个数组）；
3. `--extra-serve-args` 与 `--no-async-scheduling` 改为追加到 `DOCKER_CMD_ARGS`
   （它们本来就是容器命令的参数，不是 docker 选项——原先能工作只是偶然）；
4. 新增 `--serve-only`（起服务、绑核、写 manifest，然后退出且保留容器）与
   在收尾处调用 `run_manifest.py`。

**修过的 bug**：最初的钩子把 `PREPARE_INPUT_EXTRA_DOCKER_ARGS` 追加在镜像 tag
之后，docker 把它当成容器命令传给了 `vllm serve`，报
`vllm: error: unrecognized arguments: -v ...`（S1a 的 B 臂因此失败）。
拆分数组后位置正确：选项在 tag 之前。已用 `--dry-run` 核对参数顺序。

`--serve-only` 与 `run_manifest.py` 是给 `measurement_profiling` 的复用接口：
同一容器、同一 `run_manifest.json`，不需要二次启动服务。

## 7. 第二层插桩：三次迭代的经过（含两个必须记住的教训）

| 轮次 | run_id | 挂载了什么 | 结果 |
|---|---|---|---|
| C2b | `sub-s1-c2b-gdn` | + `attention_v1.py`、core `gdn_attn.py` | 定位到"3 次未插桩调用"，但 `pi: gdn.*` **命中 0 次** → 挂错了文件；`probe_notes.json` 未生成（`PI_SUBSCOPE_META` 当时还没进启动器） |
| C2c | `sub-s1-c2c-gdnb` | + **`vllm_ascend/ops/gdn_attn_builder.py`** | ✅ 首次拿到 GDN 六段分解；❌ 首次启动因 `AttributeError` 失败（见 7.1） |
| C2d | `sub-s1-c2d-gdnb` | + ctor / 三个 `_attach_*` / `_build_actual_seq_lengths` | ✅ 完整 12 段分解 + 4 个 kv_cache_group 结构 |

### 7.1 教训一：`pi_note` 的参数表达式同样在推理路径上

C2c 第一次启动直接失败：

```
AttributeError: 'AscendGDNAttentionMetadataBuilder' object has no attribute 'layer_names'
RuntimeError: NPUModelRunner init failed, error is ...
```

根因：`pi_note(key, {"n_layers": len(self.layer_names)})` 里**参数表达式在调用之前
求值**，所以"探针内部的 try/except"根本保护不到它。这个 builder 不存
`layer_names`（那是 model-runner 调用点才知道的信息），于是引擎起不来。

**修法**：新增 `lazy(fn)`（`pi_subscope._LazyNote`），把诊断值的构造推迟到
`pi_note` 的 try 块内部执行：

```python
pi_note("...", lazy(lambda: {"n_layers": len(self.layer_names)}))
```

并加了单测 `test_pi_subscope.test_notes` 与一个"会抛异常的 lazy 值不外泄"的
专项用例。**这是本项目"插桩必须不影响行为"最好的反面教材**：探针的每一条
语句——包括参数表达式——都必须在 probe 自己的异常边界内。

### 7.2 教训二：AST 等价性检查器自己也会误伤

C2c 的 `pi_scope` 包装 `return self._attach_...(…)` 时，检查器一度报警。核查后
确认：**`return` 写在 `with` 块内是合法的，剥掉 `with` 后就是原来的裸 `return`**，
所以只要不把 `return call()` 改写成 `x = call(); return x`，等价性就能保持。
同时修掉了检查器的一个真实缺陷：它一度把**基线的** `try/except` 也拆掉
（那会掩盖真实控制流）。现在的规则是"只有 body 完全由探针构成时才拆"，
基线 try 块原样保留。

## 8. 复现清单

```bash
# 0) 自检（不需要 NPU）
python3 instrument/verify_patch.py instrument/baseline/model_runner_v1.py.image \
    instrument/patched/model_runner_v1.py        # 共 5 个文件，见 instrument/apply.sh
python3 instrument/test_pi_subscope.py
python3 instrument/test_parse_subscope.py

# 1) 探针开销（a3-22，绑到非 chip3 的核）
taskset -c 200-203 python3 instrument/probe_overhead.py --n 200000

# 2) 第一层：A/B/C/D1/D2 阶梯
scripts/run_subscope_matrix.sh --scenario s1 --modes A,B,C,D1,D2

# 3) 第二层：attn builder 细分
scripts/launch_subscope_service.sh --run-id sub-s1-c2d-gdnb --mode on \
    --attn-probes on --model-key qwen35-08b --chip 3 --max-num-seqs 1 \
    --requests 1 --concurrency 1 --prompt-tokens 128 --max-tokens 64 \
    --cudagraph-mode FULL_DECODE_ONLY --pystack-interval-us 0

# 4) 解析 + 口径正确的阶梯表
python3 scripts/parse_subscope.py runs/<id>/pi-subscope/raw.csv \
    --outdir data/subscope/<id> --probe-floor-us 2.2 --attn-breakdown
python3 scripts/extract_ladder.py --project-root . --scenario s1 --stamp S1b

# 5) NPU 派发常数（在空闲容器内）
scripts/npu_dispatch_bench.py --n 3000
```

## 9. 交接给下游

| 下游 | 需要什么 | 在哪 |
|---|---|---|
| `measurement_profiling` | 容器名 / manifest / engine PID+TID / 配置 / perf 目标 | 见 `runs/sub-s1-c2d-gdnb/run_manifest.json`（`engine_core.pid/tid`、`handoff.*`） |
| 文档 owner | 子阶段图谱、图、口径修正 | `docs/05-hotspots.md` §5bis、`docs/09-historical-data-caveat.md` §5、`figures/05-subscope-breakdown.svg` |
| 优化方向 | 候选与预期收益 | §10 |

## 10. 优化候选（按预期收益排序，均待 perf 交叉确认）

| 候选 | 依据 | 预期收益 | 风险 |
|---|---|---|---|
| GDN 的 3 个 mamba group 合并成 1 次 build | 3 组同 spec、同 block_size、同 builder 类，每步 3×371 µs | 若合并成 1 次，理论上省 2/3 → **~740 µs/步（22% of prepare_input）** | 需要确认 3 组的 metadata 是否真的可共享（`non_spec_state_indices_tensor` 等按 gid 不同） |
| 去掉 `compute_num_computed_tokens` 的重复调用 | 它只返回缓存 tensor 却耗 43.2 µs/call × 3 | ~130 µs/步（3.9%） | 低（纯缓存读取） |
| FULL graph 下 `pad_graph_inputs` 只在 batch 形状变化时重算 | 95.3 µs/call × 3，B=1 时形状恒定 | ~286 µs/步（8.7%） | 中（graph replay 语义要核实） |
| `treat_single_token_prefills_with_state_as_decodes` 免分配 | 56.4 µs/call，纯 CPU 张量 | ~169 µs/步（5.1%） | 低 |
