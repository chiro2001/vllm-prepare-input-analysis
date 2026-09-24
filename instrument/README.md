# prepare_input 子阶段插桩（sub-scope instrumentation）

目标：把 vllm-ascend 的粗粒度 `prepare input` scope 拆成有性能意义的子 scope，
回答"scope 内部谁最贵"，并给出**插桩自身开销**的实测。

## 1. 为什么要子 scope

历史实测（`docs/03-share-in-inference.md`）里 `prepare input` 占单步 48–59%，
但 perf 采样是进程级的，热点函数分不清属于哪个子步骤。已有的 LiteProfiler
只有 `prepare input` 一个粗 scope，因此必须在 `NPUModelRunner` 内部再打点。

## 2. 基线口径（必须一致，否则数字不可比）

| 项 | 值 |
|---|---|
| 镜像 | `local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler` |
| 基线文件 sha256 | `bf65ca111750da9a411853cc59e2fe756756fa58304f48bd53d370e3811bab1f` |
| 基线内容 | vllm-ascend `f2f74a16c` + LiteProfiler v1/v2 patch（镜像内 `/vllm-workspace/vllm-ascend`） |
| vLLM | 0.26.0 `568afb3a13806beb53bb2e6bd518269357b237c0`（未改动） |

`baseline/model_runner_v1.py.image` 就是从镜像里 `cat` 出来的那一份，只读参考。

## 3. 文件

| 文件 | 作用 |
|---|---|
| `pi_subscope.py` | 探针运行时（`pi_scope` / `pi_prepare_input` / `pi_step_info`） |
| `patched/model_runner_v1.py` | 插桩版 runner（= 基线 + 49 处 `with pi_scope(...)` + 1 个 import + 1 次 `pi_step_info(...)`） |
| `baseline/model_runner_v1.py.image` | 镜像内基线副本（只读） |
| `patches/*.py.patch` | 每个文件的 `diff -u baseline patched`（5 个文件） |
| `verify_patch.py` | **AST 等价性证明**：剥掉探针后与基线逐节点相同 |
| `probe_overhead.py` | 探针自身开销实测（ns/次、µs/step） |
| `test_pi_subscope.py` | 探针机制单测（早返回 / 异常 / 关闭态零副作用） |
| `test_parse_subscope.py` | 解析脚本冒烟测试（11 个 canonical group 求和校验） |
| `apply.sh` / `revert.sh` | 校验 + 打印挂载参数 / 还原 |

## 4. 使用

```bash
# 1) 自检（不需要 NPU）
python3 instrument/verify_patch.py \
    instrument/baseline/model_runner_v1.py.image instrument/patched/model_runner_v1.py
python3 instrument/test_pi_subscope.py
python3 instrument/test_parse_subscope.py

# 2) 探针开销实测（在 a3-22 上，绑到 worker cpuset）
taskset -c 122 python3 instrument/probe_overhead.py --n 200000

# 3) 三模式 A/B/C（baseline / mounted-off / on）
scripts/launch_subscope_service.sh --run-id sub-B1-on --mode on \
    --model-key qwen35-08b --chip 3 --max-num-seqs 1 --concurrency 1 \
    --requests 1 --prompt-tokens 128 --max-tokens 64 \
    --cudagraph-mode FULL_DECODE_ONLY --pystack-interval-us 0

# 4) 解析
python3 scripts/parse_subscope.py runs/<run-id>/pi-subscope/raw.csv \
    --outdir data/subscope/<run-id> --probe-floor-us <ns/1000>
```

## 5. 设计约束

1. **关闭态零开销**：`PI_SUBSCOPE` 未设置为 `on` 时，`pi_prepare_input()` 直接
   返回调用方的 context manager（不包一层），`pi_scope()` 返回共享 `nullcontext`。
   `mounted-off` 模式因此只测量"挂载+import"的代价（实测 ≈ 0）。
2. **不写 Image**：用 `-v` 覆盖两个文件，不 `docker commit`、不改只读参考目录。
3. **不做每步 I/O**：内存累积，每 `PI_SUBSCOPE_DUMP_EVERY`（默认 25）步写一次，
   退出时 `atexit` 兜底。
4. **可解释的守恒式**：每个 scope 同时记 `self_us`（独占）与 `incl_us`（含子），
   因此 `Σ self_us(非 TOTAL) + unattributed = pi: TOTAL`，`meta.json` 里会校验残差。
5. **探针不改变控制流**：`verify_patch.py` 用 AST 等价性证明这一点（早返回、
   异常路径都覆盖在 `test_pi_subscope.py` 里）。

## 6. 子 scope 清单（49 个探针 → 11 个 canonical group）

canonical group 的定义在 `scripts/parse_subscope.py:GROUPS`，图例顺序在
`GROUP_ORDER`。按"进入 prepare input 的先后"排列：

| 探针名 | 说明 |
|---|---|
| `pi: sync_input_prep` | `synchronize_input_prep()` 的 `__enter__`（async scheduling 时等事件）+ snapshot 占位修补循环 |
| `pi: update_states` | `_update_states()`（含 deferred corrections 的构造） |
| `pi: tokens_list` | `[num_scheduled_tokens[i] for i in req_ids]` |
| `pi: phase_classify` | LiteProfiler 的 prefill/decode 判定（4 个 CPU torch op） |
| `pi: prepare_inputs` | `_prepare_inputs()` 聚合 |
| `pi: in.*` | `_prepare_inputs` 内部 24 个探针（block table / attn_state / positions / token_indices / H2D / spec meta ...） |
| `pi: cascade_attn_prefix_lens` | cascade attention 前缀长度（默认关闭） |
| `pi: batch_exec_and_padding` | cudagraph dispatcher + DP all_reduce |
| `pi: ubatch_slices` | microbatch 切分（ascend 恒 `(None, None)`） |
| `pi: mamba_preprocess` | 仅 `mamba_cache_mode == "align"` |
| `pi: dsa_positions` | 仅 `use_compress` |
| `pi: pad_query_start_loc` | FULL graph / SP 下的 query_start_loc padding |
| `pi: build_attention_metadata` | `_build_attention_metadata()` 聚合 |
| `pi: am.*` | 其内部 5 个探针（`max_seq_len` 的 `.item()`、cm_base 准备、group 循环、builder.build、逐层赋值） |
| `pi: sanitize_placeholder_ids` | 占位 input_ids 清理 |

## 7. 已知限制

* `pi: build_attention_metadata` 与 `pi: prepare_inputs` 是聚合探针（self_us≈0），
  它们真正的成本在各自的子探针里；汇总表里按 `incl_us` 看。
* 探针本身耗时（`probe_overhead.py` 实测，Kunpeng 920B 上另测）是**下界噪声**：
  `self_us_p50 < probe_floor_us` 的 scope 在 `meta.json.scopes_below_probe_floor`
  里单列，不能当作真实成本引用。
* 只插桩 `vllm_ascend/worker/model_runner_v1.py`。`attention_v1.py` 的
  `AscendAttentionMetadataBuilder.build` 由 `pi: am.builder_build` 在外层包住，
  不再往下拆（如需更细，见 `agents/subscope_instrumentation/REPORT.md` 的后续建议）。
