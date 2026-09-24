# harness 内部接口契约（replay_harness 维护，动手前必读）

所有代码在本地 `agents/replay_harness/harness/` 编写，`scripts/sync.sh push` 推到
a3-22 的 `~/projects/vllm/prepare-input-phase/harness/`。**不要在 a3-22 上直接改代码**。

## 0. 目录归属（禁止跨写）

| 路径 | owner | 说明 |
|---|---|---|
| `harness/scripts/pi-docker.sh`, `harness/scripts/sync.sh` | replay_harness | 无卡容器执行器 + 同步 |
| `harness/probes/**`, `harness/pi_harness/shim/**` | probe_devdeps | 设备依赖清单 + torch_npu shim |
| `harness/pi_harness/trace/**`, `harness/pi_harness/workload/**` | trace_schema | record 钩子 + trace schema + 合成 workload |
| `harness/pi_harness/runner/**` | replay_harness | P1 状态构造 + step 驱动 + sweep CLI |
| `harness/pi_harness/profiling/**`, `harness/scripts/prof_*.sh` | prof_measure | 无卡 PMU/topdown/flamegraph 采集 |
| `data/harness/<name>_*.json|csv` | 各自 | 用 owner 前缀，勿覆盖他人文件 |
| `agents/<name>/REPORT.md` | 各自 | 交接报告 |

## 1. 运行环境（唯一合法入口）

```bash
bash harness/scripts/pi-docker.sh 'python -c "import torch; print(torch.__version__)"'
```

约束：**不许 --device /dev/davinci\***，**不许用 CPU 120-159**，线程数 ≤16。

## 2. shim 接口（probe_devdeps 提供）

```python
# harness/pi_harness/shim/__init__.py
def install(device: str = "cpu") -> None:
    """在 import vllm / vllm_ascend 之前调用。
    注入 fake torch_npu 模块 + torch.npu 命名空间；把 device 统一映射到 CPU。
    必须幂等、必须在 sys.modules 里留下可被 `import torch_npu` 命中的模块。"""

def report() -> dict:
    """返回被 shim 命中的 API 计数（打桩记录），用于保真度文档。"""
```

## 3. workload 接口（trace_schema 提供）

```python
# harness/pi_harness/workload/schema.py

@dataclass
class StepRecord:
    step_idx: int
    phase: str                      # prefill | decode | mixed | spec-decode
    scheduler_output: dict          # vllm SchedulerOutput 的 JSON-safe 表示
    req_states: list[dict]          # 每请求: req_id/prompt_token_ids/num_computed_tokens
                                    #   /block_ids/num_output_tokens/spec_len
    model_output: dict | None       # 采样结果（喂 scheduler.update_from_output）
    t_prepare_input_us: float | None

def load_jsonl(path) -> list[StepRecord]: ...
def save_jsonl(recs, path) -> None: ...

# harness/pi_harness/workload/generate.py
def synth_trace(cfg: "SweepConfig") -> list[StepRecord]:
    """按参数（batch/isl/osl/block_size/prefix 命中率/spec_k/...）合成 trace。
    必须复用 record 出来的同一 schema，且带上 __synth__ 标记便于 A/B。"""
```

## 4. runner 接口（replay_harness 提供）

```python
# harness/pi_harness/runner/replay.py
class PrepareInputReplay:
    def __init__(self, cfg: dict): ...
    def build(self) -> None: ...              # 构造真实 NPUModelRunner 状态
    def run(self, steps) -> StepTiming: ...   # 逐步执行 _update_states + _prepare_inputs
```

## 5. 保真度红线

任何被 shim / CPU 近似 / 跳过的东西，必须在 `docs/06-synthetic-load.md` 的
「已知偏差」表里逐条登记（含影响方向：高估/低估 CPU 负载）。禁止为了让数字好看而伪造。
