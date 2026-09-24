# prepare_input 无卡负载复刻 harness

目标：**不挂任何 NPU 卡**，在 a3-22（Kunpeng 920B）上复现 vLLM 0.26.0 +
vllm-ascend 0.26.0rc1 里 `prepare_input` 阶段的 CPU 侧负载，用于在没有卡的时候
迭代 CPU 侧优化。

固定身份：vLLM `568afb3a1` / vllm-ascend `f2f74a16c` / 镜像
`quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler`（python 3.12.13、
torch 2.10.0+cpu、triton-ascend）。

---

## 1. 快速开始

```bash
# 1) 同步到 a3-22（唯一允许写入的远端路径）
bash harness/scripts/sync.sh push

# 2) 跑一步 replay
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
  bash harness/scripts/pi-docker.sh "cd /work/harness && python -m pi_harness.runner.cli --help"'

# 3) 一个具体负载
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh \
  "cd /work/harness && PI_MODEL_TMP=/tmp/pi_models python -m pi_harness.runner.cli \
     --batch 32 --isl 2048 --osl 128 --steps 200 --has-gdn \
     --triton-launch-us 0 --out /work/data/harness --tag demo"'

# 4) 参数扫描（长表 CSV）
ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh \
  "cd /work/harness && PI_MODEL_TMP=/tmp/pi_models python -m pi_harness.runner.cli \
     --sweep isl --steps 60 --max-num-batched-tokens 16384 --out /work/data/harness --tag hw64 --has-gdn"'
```

`pi-docker.sh` 的关键约束（写在脚本里，别绕过）：
**不挂 `/dev/davinci*`**、`--network none`、`--cpuset-cpus 200-215`（绝不用真机实验切片
`120-159`）、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

---

## 2. 为什么可以无卡跑：三层 shim

实测结论：`import torch_npu` / `import vllm` / `import vllm_ascend` 在**没有 NPU 设备**
的容器里都能成功；真正会崩的是三类**设备操作**。`pi_harness/shim` 针对性地替换它们：

| 层 | 崩在哪 | shim 做什么 |
|---|---|---|
| 1. `torch.npu.*` 命名空间 | `torch.npu.current_stream()` / `synchronize()` / `zeros(device="npu")` → `aclInit 507008` | 换成 no-op / CPU 语义（`CpuShimStream`、`CpuShimEvent`、假显存查询） |
| 2. tensor / factory | `.npu()` / `.to("npu")` / `torch.zeros(device="npu")` / `pin_memory=True` | device npu→cpu、`pin_memory`→False；同时**按字节记账**被降级的拷贝（`degraded_copy_bytes`） |
| 3. triton-ascend | `NPUUtils.get_arch()` SystemError；`_compute_slot_mapping_kernel[...]()` 每步 launch 崩在 `get_current_stream` | 替换 `get_arch`/`get_aicore_num`；把 slot-mapping kernel 换成**向量化 numpy 等价实现**并记账 |

还有一个非设备坑：容器里 uid 1003 没有 passwd 条目 → `getpass.getuser()` 抛
`KeyError` → torch inductor cache 崩。`pi-docker.sh` 用
`-e USER=... -e LOGNAME=... -e TORCHINDUCTOR_CACHE_DIR=/tmp/...` 修掉。

---

## 3. 代码结构

```
harness/
  scripts/pi-docker.sh      # 无卡容器执行器（唯一的运行入口）
  scripts/sync.sh           # 本地 <-> a3-22 同步
  pi_harness/
    env.py                  # bootstrap：shim -> import vllm -> 打桩分布式 group -> import NPUModelRunner
    shim/__init__.py        # 三层 shim + report()
    runner/
      config.py             # RunnerConfig（全部 CLI 参数）
      modelcfg.py           # 把真模型 config.json 摊平成可离线 inspect 的最小 config
      build.py              # object.__new__(NPUModelRunner) + 手工装配 + AttrAudit
      synth.py              # 合成 SchedulerOutput（语义对齐真 scheduler）
      triton_cpu.py         # slot-mapping kernel 的 CPU 等价实现 + launch 记账
      timer.py              # 子步骤计时（类上 wrap，不改 vLLM 源码）
      replay.py             # 逐步执行 _update_states + _prepare_inputs
      cli.py                # `python -m pi_harness.runner.cli`
  INTERFACES.md             # 跨 agent 接口契约（谁写哪个目录）
```

---

## 4. 保真度边界（必须一起读 `docs/06-synthetic-load.md`）

**已复现**
* `_update_states` + `_prepare_inputs` 的**真实 Python 代码路径**（逐行执行，未抄写）
* 真实 `NPUInputBatch` / `MultiGroupBlockTable` / `CpuGpuBuffer` 数据结构
* 真实的尺寸参数链：`max_model_len` -> `token_ids_cpu` 宽度、`max_num_blocks_per_req`
  -> block table 宽度 -> 每步 `commit_block_table` 的搬运字节数
* 真实的 chunked-prefill / decode / spec-decode 分支选择（`_build_attn_state`）

**未复现（逐条登记，含影响方向）**
* H2D/D2H DMA 的真实耗时（退化为 CPU->CPU；已按字节记账，可用实测带宽补回）
* Triton kernel launch 的 Python 侧开销（`--triton-launch-us` 注入；默认 0 = 低估）
* device op（acl 调用）的内核态开销（默认 0 = 低估）
* P>1 的集合通信（`FakeGroupCoordinator` 会在被调用时**直接抛错**，不静默）
* 权重加载 / KV cache 分配 / 图捕获（不在这条路径上）

**审计手段**（防止"静默跳过"）
* `AttrAudit`：记录"没被装配却被访问"的属性；当前为空
* `meta.readonly_attrs_skipped`：记录只读 property 被跳过的赋值；当前为空
* `shim.report()["warnings"]` / `UNSHIMMED_DEVICE_OP`：未 shim 的设备操作会**抛错**
  而不是静默通过

---

## 5. 输出格式

`--out DIR` 会写三份文件（`--tag` 做前缀）：

| 文件 | 内容 |
|---|---|
| `per_step_*.csv` | 每步：`update_states_us` / `prepare_inputs_us` / `deferred_fixup_us` / 请求数 / 调度 token 数 / `triton_cpu_us` / `triton_launches` |
| `substep_*.csv` | 每步 x 每个子函数耗时（**单位秒，每步增量**；列名即被 wrap 的函数） |
| `summary_*.json` | p50/p90/p99/mean、子步骤汇总、host 信息、config、meta |

`--sweep <dim>` 额外写 `sweep_<dim>_*.csv`（长表，一行一个配置）。
