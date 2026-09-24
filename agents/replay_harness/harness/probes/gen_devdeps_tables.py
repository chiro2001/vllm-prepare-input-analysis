#!/usr/bin/env python3
"""C/D + manifest：生成设备依赖清单与子步骤矩阵。

输入（都在 `data/harness/`）：
  * `devdeps_scope.json`            E 的 scope/调用链静态结果（scope_prepare_input.py）
  * `devdeps_dynamic_hits.json`     C 的动态命中（probe_dynamic_hits.py）
  * `devdeps_shim_smoke.json`       D 的冒烟（smoke_shim.py）
  * `devdeps_shim_patch_selftest.json`  参考补丁自测（shim_patch_proposal.py）
  * `devdeps_manifest_{container,host}.json`

输出：
  * `devdeps_apilist.{md,json}`     API × 静态/动态 × 类别 × shim 行为 × 风险
  * `devdeps_matrix.{md,json}`      prepare_input 子步骤 × 可执行性 × 偏差 × 风险
  * `devdeps_manifest.json`         合并 manifest

用法（容器内，路径都是 /work）::

    python harness/probes/gen_devdeps_tables.py            # 写盘
    python harness/probes/gen_devdeps_tables.py --print    # 顺便把 md 打到 stdout

本地（无容器）也可以用 `--source-root <快照根>` 复用静默路径。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("PI_REPO_ROOT", "/work"))
DATA = Path(os.environ.get("PI_DEVD_DEPS_OUT", REPO / "data" / "harness"))


# ---------------------------------------------------------------------------
# 静态扫描
# ---------------------------------------------------------------------------

def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def chain_ranges(scope: dict) -> dict[str, list[tuple[int, int]]]:
    """prepare_input scope 覆盖到的所有函数体（文件 → 行区间）。"""
    want: dict[str, set[int]] = {}
    for s in scope.get("steps", []):
        for c in s.get("calls", []):
            if c.get("def_file") and c.get("def_line"):
                want.setdefault(c["def_file"], set()).add(int(c["def_line"]))
    want.setdefault(scope["model_runner_v1"], set()).add(int(scope["scope_start_line"]))

    out: dict[str, list[tuple[int, int]]] = {}
    for f, line_starts in want.items():
        p = Path(f)
        if not p.exists():
            continue
        src = p.read_text(errors="replace")
        tree = ast.parse(src, filename=str(p))
        spans: list[tuple[int, int]] = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno in line_starts:
                spans.append((n.lineno, n.end_lineno or n.lineno))
        out[f] = sorted(set(spans))
    return out


def scan(pattern: str, ranges: dict[str, list[tuple[int, int]]]) -> tuple[int, list[str]]:
    rx = re.compile(pattern)
    total = 0
    sites: list[str] = []
    for f, spans in ranges.items():
        lines = Path(f).read_text(errors="replace").splitlines()
        for a, b in spans:
            for i in range(a - 1, min(b, len(lines))):
                n = len(rx.findall(lines[i]))
                if n:
                    total += n
                    sites.append(f"{Path(f).name}:{i + 1}")
    return total, sites


def scan_tree(pattern: str, roots: list[Path]) -> tuple[int, dict[str, int]]:
    rx = re.compile(pattern)
    total = 0
    per_file: dict[str, int] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            try:
                src = p.read_text(errors="replace")
            except Exception:
                continue
            n = len(rx.findall(src))
            if n:
                per_file[str(p)] = n
                total += n
    return total, per_file


def top_files(per_file: dict[str, int], k: int = 5) -> list[str]:
    return [f"{Path(f).name}×{n}" for f, n in
            sorted(per_file.items(), key=lambda kv: -kv[1])[:k]]


# ---------------------------------------------------------------------------
# 动态计数
# ---------------------------------------------------------------------------

def dynamic_index(data: Path | None = None) -> dict:
    data = data or DATA
    idx: dict[str, int] = {}
    srcs: dict[str, str] = {}

    dyn = _load(data / "devdeps_dynamic_hits.json")
    if dyn:
        for k, v in (dyn.get("totals") or {}).items():
            idx[k] = idx.get(k, 0) + v
            srcs[k] = "probe_dynamic_hits.py"

    patch = _load(data / "devdeps_shim_patch_selftest.json")
    if patch:
        for k, v in ((patch.get("patch_report") or {}).get("calls") or {}).items():
            idx[f"patch:{k}"] = idx.get(f"patch:{k}", 0) + v
            srcs[f"patch:{k}"] = "shim_patch_proposal.py selftest"
        for k, v in ((patch.get("shim_report") or {}).get("calls") or {}).items():
            idx.setdefault(f"shim:{k}", v)
            srcs.setdefault(f"shim:{k}", "shim.report()")

    shim = _load(data / "devdeps_shim_report.json")
    if shim:
        for k, v in (shim.get("calls") or {}).items():
            idx[f"shim:{k}"] = max(idx.get(f"shim:{k}", 0), v)
            srcs.setdefault(f"shim:{k}", "shim.report()")

    dyn_full = _load(data / "devdeps_dynamic_hits.json") or {}
    for k, v in (dyn_full.get("triton_cpu_stats") or {}).items():
        if isinstance(v, (int, float)):
            idx[f"triton_cpu.{k}"] = int(v)
            srcs[f"triton_cpu.{k}"] = "runner/triton_cpu.stats()"

    return {"values": idx, "sources": srcs}


# ---------------------------------------------------------------------------
# C. API 清单（策展表；静态计数由脚本填，动态计数从上面合并）
# ---------------------------------------------------------------------------

# (api, regex, 动态 key 列表, 类别, shim 行为, 风险, [可选: 出现位置覆盖文本])
API_ROWS: list[tuple] = [
    # ---- torch.npu 命名空间 ----
    ("torch.npu.current_stream", r"torch\.npu\.current_stream\b",
     ["torch.npu.current_stream", "shim:npu.stream"],
     "设备交互",
     "返回单例 CpuShimStream；`with torch.npu.stream(s)` 退化为 nullcontext（计数）",
     "真机只是取 stream 对象（亚 µs，无同步）；shim≈0 → 中性。若代码依赖 stream 之间的可见性/同步语义，会丢失顺序（无卡下无在飞 kernel，可接受）"),
    ("torch.npu.Event", r"torch\.npu\.Event\b",
     ["torch.npu.Event()", "shim:Event.record", "shim:Event.synchronize"],
     "设备交互",
     "CpuShimEvent：record/wait/synchronize/query 全 no-op（计数）",
     "`synchronize_input_prep` 每步 `prepare_inputs_event.synchronize()` 真机可能阻塞（async scheduling 下是主要隐藏同步点）；shim=0 → **低估**"),
    ("torch.npu.Stream", r"torch\.npu\.Stream\b",
     ["torch.npu.Stream()", "shim:Stream.synchronize", "shim:Stream.wait_stream"],
     "设备交互",
     "CpuShimStream（计数）",
     "真机构造 stream 有一次性 ctx 成本；shim≈0 → 轻度低估"),
    ("torch.npu.stream", r"torch\.npu\.stream\b",
     ["torch.npu.stream", "shim:npu.stream"],
     "设备交互",
     "contextlib.nullcontext（计数）",
     "真机 stream 切换 + 事件依赖 ~µs 级；shim=0 → 低估（prepare_input 主路径不在 stream 内，影响主要在后处理/采样）"),
    ("torch.npu.synchronize", r"torch\.npu\.synchronize\b",
     ["torch.npu.synchronize"],
     "设备交互",
     "no-op（计数）",
     "真机是整设备同步，通常是大幅 CPU 阻塞；shim=0 → **显著低估**，必须靠 PI_OP_COST_* 或真机数值补"),
    ("torch.npu.current_device", r"torch\.npu\.current_device\b",
     ["torch.npu.current_device"],
     "设备交互",
     "常量 0（计数）",
     "真机第一次会触发 aclInit（一次性百 ms 级）；稳态≈0 → 中性"),
    ("torch.npu.set_device", r"torch\.npu\.set_device\b", [],
     "设备交互",
     "no-op（计数）",
     "真机首次 set_device 触发 aclInit/ctx 建立（一次性），后续接近 0；shim 把一次性成本抹掉（对单步稳态影响可忽略）"),
    ("torch.npu.empty_cache", r"torch\.npu\.empty_cache\b",
     ["torch.npu.empty_cache"], "设备交互",
     "no-op（计数）",
     "真机若真的回收会阻塞（十几 ms 级）；shim=0 → 低估，但只在显存压力路径上"),
    ("torch.npu.mem_get_info", r"torch\.npu\.mem_get_info\b",
     ["torch.npu.mem_get_info"], "设备交互",
     "返回常量 (1<<40, 1<<40)（计数）",
     "真机是驱动查询（µs 级）；shim 常数返回，**不影响 CPU 负载形状**"),
    ("torch.npu.memory_stats/allocated/reserved/reset_peak", r"torch\.npu\.(memory_stats|memory_allocated|memory_reserved|reset_peak_memory_stats|max_memory_allocated)\b",
     [], "设备交互",
     "常量 / no-op（计数）",
     "低风险（统计口径，不进 prepare_input 主路径）"),
    ("torch.npu.device_count / is_available", r"torch\.npu\.(device_count|is_available)\b",
     [], "设备交互",
     "shim 返回 1 / True（**与真机无卡事实相反**，为的是让上层走正常分支）",
     "重要：真机 is_available=True、device_count>0；shim 必须伪造为可用，否则 vLLM 会走 fallback 分支，改变代码路径。必须在报告里显式登记"),
    ("torch.npu.get_device_name / get_device_properties", r"torch\.npu\.get_device_(name|properties)\b",
     [], "设备交互",
     "返回假名字字符串 / _DummyDeviceProps",
     "真机返回值会决定 num_sms、显存容量等；**shim 必须给出与目标卡一致的常量**，否则 cudagraph/batch padding 决策会变"),
    ("torch.npu.is_current_stream_capturing", r"torch\.npu\.is_current_stream_capturing\b",
     [], "设备交互",
     "False（计数）",
     "无卡 harness 不做 graph capture；真机 eager 步也返回 False → 中性"),
    ("torch.npu.ExternalEvent / NPUGraph / graph_*", r"torch\.npu\.(ExternalEvent|NPUGraph|graph_task_\w+|graph_pool_handle|graph)\b",
     [], "设备交互",
     "CpuShimEvent / 空类 / no-op",
     "只在 aclgraph 采集路径使用（本 harness 不开 graph）→ 低风险"),
    ("torch.npu.config.allow_internal_format", r"torch\.npu\.config\.allow_internal_format",
     [], "纯CPU",
     "`model_runner_v1.py:224` 是**只写**（`torch.npu.config.allow_internal_format = True`），写入能过；"
     "真实 `_npuConfig` 对象**可写不可读**（读会 AttributeError，实测 C16）",
     "**低**：源码里只有这一处写入、没有读取点（已全仓 grep 确认）；shim 若替换 config 对象仍须保证 `allow_internal_format` 可写、不要引入读取依赖"),
    ("torch.npu.*（模块级属性读取）", r"torch\.npu\.(?!(?:current_stream|Event|Stream|stream|synchronize|current_device|set_device|empty_cache|mem_get_info|memory_\w+|device_count|is_available|get_device_\w+|is_current_stream_capturing|ExternalEvent|NPUGraph|graph\w*|config|default_stream|set_stream|set_compile_mode))[A-Za-z_0-9]+",
     [], "设备交互", "shim 覆盖不到的属性读取会 AttributeError（见 report().warnings）",
     "**审计钩子**：这条用于发现新出现的、未被 shim 登记的 API"),

    # ---- torch.accelerator ----
    ("torch.accelerator.synchronize", r"torch\.accelerator\.synchronize\b",
     ["torch.accelerator.synchronize"], "设备交互",
     "no-op（计数）；参考补丁也接管",
     "`GPUModelRunner._sync_device()` 走它（execute_model 的 profiling 分支，在 scope 之前）；shim=0 → 低估"),
    ("torch.accelerator.* 其它", r"torch\.accelerator\.(?!synchronize)[A-Za-z_]+", [],
     "设备交互",
     "参考补丁把 current_device_index/set_device_index/device_count/current_accelerator/empty_cache/memory_* 全部接管",
     "未接管时 `torch.accelerator.current_device_index()` 直接 aclInit 507008（实测）"),

    # ---- torch_npu ----
    ("import torch_npu", r"^\s*import torch_npu|from torch_npu",
     [], "纯CPU",
     "**不伪造**，直接用真实模块（无卡下 import 成功，仅 warning）",
     "真实 import 有 ~2-4 s 一次性开销；不影响单步负载"),
    ("torch_npu.npu / torch_npu.utils", r"torch_npu\.(npu|utils)\b", [],
     "纯CPU",
     "保留真实模块；`torch_npu.utils.get_cann_version()` 实测可返回 '9.1.0'",
     "低（导入型依赖）"),
    ("torch_npu.profiler.dynamic_profile", r"torch_npu\.profiler\.(dynamic_profile|dp)\b", [],
     "设备交互",
     "真实模块可 import；`dp.start/stop` 需 devprof，未在无卡下调用",
     "vllm_ascend.worker.worker 顶层 import 它（会加载 profiler 库，一次性 ~百 ms）"),
    ("torch_npu.op_plugin.atb._atb_ops", r"torch_npu\.op_plugin", [],
     "设备交互",
     "真实模块可 import（实测 OK）",
     "注册 ATB 算子扩展；import 期无设备操作"),
    ("torch_npu._inductor", r"torch_npu\._inductor", [],
     "设备交互",
     "**不可 import**（无卡下 `triton ... get_arch` SystemError）",
     "只在 inductor 编译路径用；本 harness 必须不走 torch.compile"),
    ("torch_npu.multiprocessing.reductions", r"torch_npu\.multiprocessing", [],
     "纯CPU", "真实模块可 import", "低（只有权重传输/ipc 路径用）"),

    # ---- acl / CANN ----
    ("acl.*", r"\bacl\.[A-Za-z_0-9]+", [], "设备交互",
     "不拦截；实测 `import acl` OK，`acl.init()` 返回 500000（无设备）",
     "vllm_ascend 的 prepare_input 链**不直接调用 acl**（静态命中 0）；若后续引入，需要显式登记"),
    ("CANN 环境变量 / ASCEND_HOME_PATH", r"ASCEND_HOME_PATH|/usr/local/Ascend", [],
     "纯CPU", "保留真实 CANN 目录（镜像自带 cann-9.1.0）", "低"),

    # ---- Triton ----
    ("triton launch（_compute_slot_mapping_kernel）", r"_compute_slot_mapping_kernel\s*\[|_kernel\[",
     ["triton_cpu.launches", "triton_cpu.fallback_ms"], "设备交互",
     "**必需**：`pi_harness/runner/triton_cpu.py::install()` 把 `_compute_slot_mapping_kernel` 换成 numpy 等价实现（同时替换 `block_table` 模块内的名字绑定），"
     "并记账 `launches`/`launch_ns`/`fallback_ms`；`set_launch_cost_us()` 可注入真机 Python launch 开销。"
     "**仅靠 shim 的 triton proxy 不够**（它只截 `get_current_target`）：launch 仍走 ascend driver → `LazySetDevice 107001` 崩（实测对照见 `devdeps_dynamic_hits.errors`）",
     "真机 CPU 侧成本 = Python launch 开销（参数绑定/特化/cache 查找/launcher 调度，需真机标定，初值 10-50 µs/次），设备侧执行与之并行；"
     "本替换额外付出 numpy 向量化成本 → 偏差方向**不确定**，必须用 `launch_cost_us` 标定后再比",
     "ascend: `vllm_ascend/worker/block_table.py:179`（kernel 定义 `ops/triton/compute_slot_mapping.py:12`）；core 同名实现: `vllm/v1/worker/block_table.py:166`（定义同文件:347）"),
    ("triton.runtime.driver.active.get_current_stream（未拦截时的失败点）",
     r"get_current_stream|_npu_getCurrentRawStreamNoWait", [], "设备交互",
     "**未拦截**（shim 只截 `get_current_target`）",
     "实测栈：`triton/runtime/jit.py:572 run()` → `driver.active.get_current_stream(device)` → "
     "`triton/backends/ascend/driver.py:259` → `backend_register.py:283 _npu_getCurrentRawStreamNoWait` → "
     "`RuntimeError LazySetDevice 107001`。要通用覆盖（不止 slot mapping）必须在这一层做 driver proxy",
     "无直接 vllm 调用点（经 triton runtime 进入）；证据: `data/harness/devdeps_dynamic_hits.json` 的 errors 字段"),
    ("torch.Generator（**类型注解**依赖）",
     r"torch\.Generator", [], "纯CPU",
     "shim 必须用**子类**替换 `torch.Generator`，**不能**用函数替换：库在 import 期对 `torch.Generator | None` 求值，"
     "函数对象没有 `__or__` → `TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'`",
     "**高**：纯 import 期类型系统约束，与设备无关，但会让整条 import 链崩；属于 shim 自身引入的新风险",
     "vllm-ascend / transformers 侧（`torch.Generator` 注解）；证据：replay_harness 实测（P1 冒烟）"),
    ("init_ascend_config(vllm_config)",
     r"init_ascend_config|get_ascend_config", [], "纯CPU",
     "无卡 harness **手工补装**：`_prepare_inputs` 末尾 `lmhead_tp_enable()` 会读 ascend config；真机由 `NPUWorker.init_device()` 建立",
     "**中**：漏装直接 AttributeError；补装后要确认读到的字段（`scheduler_config.profiling_chunk_config.need_timing`、`enable_enpu`）与真机一致，"
     "否则 `execute_model` 顶部的 profiling 分支会走不同路径",
     "`vllm_ascend/ascend_config.py`；消费点 `vllm_ascend/worker/model_runner_v1.py`（`lmhead_tp_enable()` / `self.ascend_config`）"),
    ("KVCacheConfig（kv_cache_groups）",
     r"kv_cache_config|KVCacheConfig", [], "纯CPU",
     "无卡 harness **手工构造**最小 `KVCacheConfig`：`_may_reorder_batch` 读 `kv_cache_config.kv_cache_groups`；真机由 `initialize_kv_cache()` 产出",
     "**中**：`kv_cache_groups[].kv_cache_spec.block_size` 决定 `BlockTable.blocks_per_phys_block` 与 physical/logical 映射；填错会让 slot_mapping 数值错（**静默**）",
     "`vllm/v1/kv_cache_interface.py`；消费点 `vllm_ascend/worker/model_runner_v1.py::_may_reorder_batch`"),
    ("triton.runtime.driver.active.get_current_target", r"get_current_target",
     ["triton.get_current_target", "shim:triton.get_current_target"], "设备交互",
     "替换为 cpu target（`backend='cpu'`）",
     "这是 import 期唯一必须拦截的 triton API（flash_linear_attention 的 ops/utils.py:123）"),
    ("torch.zeros(..., device='npu') 类 device 重映射", r"device\s*=\s*(self\.device|torch\.device\(\"npu)|['\"]npu",
     ["patch:device_remapped"], "两者都有",
     "**当前 shim 未覆盖**（实测 10/17 冒烟失败）；`probes/shim_patch_proposal.py` 给出参考实现（factory + Tensor.to/npu + accelerator）",
     "不覆盖 → prepare_input 第一步 `CpuGpuBuffer(device=npu)` 就崩；覆盖后所有 npu tensor 变 CPU tensor（**有意近似**）"),
    ("pin_memory=True", r"pin_memory",
     ["patch:pin_memory_downgraded"], "两者都有",
     "容器无 pinned allocator（实测 `torch.zeros(4, pin_memory=True)` RuntimeError）→ 必须降级为 False 并登记",
     "真机 pin 页让 H2D 走 DMA（CPU 侧便宜）；shim 的 CPU→CPU memcpy 由 CPU 真搬数据 → 大 buffer 时**高估**，pageable 场景反而**低估**"),
    ("CpuGpuBuffer.copy_to_gpu / copy_to_cpu", r"copy_to_gpu|copy_to_cpu",
     [], "两者都有",
     "两块都变 CPU tensor，退化为 CPU→CPU memcpy",
     "见上条；这是 prepare_input 里**调用次数最多**的设备交互（链内静态 29 次调用点）"),
    ("Tensor.npu() / .to('npu')", r"\.npu\(\)|\.to\(\s*[\"']npu",
     ["patch:device_remapped", "shim:UNSHIMMED_DEVICE_OP"], "设备交互",
     "需要 device 重映射层；否则穿透到 privateuse1 → aclInit/guard",
     "prepare_input 主路径本身不用（静态 0 命中），但 runner 构造期大量使用"),
    (".cpu / .gpu / .np 缓冲三元组（CpuGpuBuffer）", r"\.copy_to_gpu|\.gpu\b|\.np\b", [], "两者都有",
     "shim 下三者的差异只剩「哪个 Python 对象」；`_np` 与 `cpu` 共享内存的语义必须保持",
     "若把 `.np` 与 `.cpu` 解耦（例如复制而非共享），会静默改变正确性（写 np 不生效）"),
]


# ---------------------------------------------------------------------------
# D. 子步骤矩阵（策展；证据来自 scope_*.json / A8 / smoke / 静态阅读）
# ---------------------------------------------------------------------------

MATRIX_ROWS: list[dict] = [
    dict(step="synchronize_input_prep（prepare_inputs_event.synchronize/record）",
         loc="vllm/v1/worker/gpu_model_runner.py:3810（ascend 未覆盖）",
         runnable="是（shim 下 no-op）",
         shim="torch.npu.Event → CpuShimEvent（synchronize/record 计数）",
         real_cpu="async scheduling 下每步阻塞等上一步 H2D 完成；是隐藏的真同步点",
         bias="低估", risk="高：这是 scope 的**入口**，真机上可能是 prepare_input 内最大单项；无卡下无法复现等待时间"),
    dict(step="_update_states（NPUModelRunner 覆写 + 父类）",
         loc="vllm_ascend/worker/model_runner_v1.py:816 → vllm/v1/worker/gpu_model_runner.py:1169",
         runnable="是（需分布式 group 桩：get_pp_group/get_dcp_group）",
         shim="Event/Stream/device 重映射；triton 不需要",
         real_cpu="纯 Python 记账为主（dict/list 维护）+ block table 的 CPU 侧 memcpy + `_zero_block_ids`",
         bias="中性",
         risk="中：`num_accepted_tokens_event.synchronize()`、`_kv_block_zeroer.zero_block_ids` 会碰设备"),
    dict(step="block_table.commit_block_table（CpuGpuBuffer.copy_to_gpu(num_reqs)）",
         loc="vllm_ascend/worker/block_table.py:288",
         runnable="是（device 重映射后）",
         shim="device 重映射 + copy_to_gpu 退化 CPU→CPU",
         real_cpu="pinned CPU→NPU 异步 H2D：CPU 侧是 O(num_reqs×max_blocks) 的 memcpy + launch",
         bias="小 batch 中性；大 batch 视 pinned 与否，shim 可能高估（CPU 真搬）或低估（pageable staging）",
         risk="中：真机是**异步**的，可以和后继 CPU 计算重叠（代码注释明确说这里是为了 overlap）；shim 变成同步 memcpy → 破坏 overlap 结构"),
    dict(step="compute_slot_mapping（Triton kernel launch）",
         loc="vllm_ascend/worker/block_table.py:150（launch 在 :179）；kernel 定义 ops/triton/compute_slot_mapping.py:12",
         runnable="**否**（当前 shim 下 `LazySetDevice 107001`）",
         shim="必须替换 kernel launch（CPU 实现 or 补 triton driver proxy 的 get_current_stream/launch）",
         real_cpu="Python 侧 launch 开销（参数打包、stream 查询、下发）+ 设备侧并行执行；每步 1 次 launch，grid=num_reqs+1",
         bias="不确定",
         risk="**高（本轮头号缺口）**：不替换则 prepare_input 根本跑不完；用 numpy 复刻的 CPU 成本与真机 launch 成本量级需标定（建议真机 perf 出该 launch 的 CPU 时间）"),
    dict(step="req_indices = np.repeat(arange_np, num_scheduled_tokens)",
         loc="vllm_ascend/worker/model_runner_v1.py:908",
         runnable="是", shim="无（纯 CPU）",
         real_cpu="纯 numpy，O(total_tokens) 分配", bias="中性",
         risk="低：唯一风险是 shim 让 array 尺寸/类型与真机不同（dtype 必须保持 int32/64 一致）"),
    dict(step="_get_cumsum_and_arange（cu_num_tokens / query_pos.np）",
         loc="vllm_ascend/worker/model_runner_v1.py:929（helper 在 vllm/v1/worker/gpu_model_runner.py）",
         runnable="是", shim="无（写的是 self.query_pos.np，CPU 镜像）",
         real_cpu="np.cumsum + np.arange + 一次 CpuGpuBuffer.copy_to_gpu", bias="中性",
         risk="低；copy_to_gpu 见上"),
    dict(step="positions 计算（np.add(num_computed_tokens_cpu[req_indices], query_pos, out=positions_np)）",
         loc="vllm_ascend/worker/model_runner_v1.py:932",
         runnable="是", shim="无（纯 numpy，fancy-index + 加法）",
         real_cpu="O(total_tokens) numpy GEMV-like gather；NPU 上还有一次 device 侧等价计算（后面 seq_lens/positions 的 torch 版本）",
         bias="中性", risk="低（CPU 侧最稳的一段）"),
    dict(step="token_indices + torch.index_select（input_ids 组装）",
         loc="vllm_ascend/worker/model_runner_v1.py:981-990",
         runnable="是", shim="无（input_ids.cpu 是 CPU buffer）",
         real_cpu="int64 乘加 + torch.index_select 写 CPU buffer（O(total_tokens) gather）",
         bias="中性", risk="低；注意 `input_ids.cpu` 与 `.gpu` 在 shim 下是两块 CPU buffer，语义一致"),
    dict(step="query_start_loc.np 填值 + copy_to_gpu",
         loc="vllm_ascend/worker/model_runner_v1.py:1037-1039",
         runnable="是", shim="device 重映射；copy_to_gpu→CPU memcpy",
         real_cpu="O(num_reqs) 写 + H2D（很小，pinned）", bias="中性",
         risk="低"),
    dict(step="optimistic_seq_lens_cpu（torch.add + fill_）",
         loc="vllm_ascend/worker/model_runner_v1.py:1056-1061",
         runnable="是", shim="无（CPU tensor 运算）",
         real_cpu="O(num_reqs) CPU tensor 加法 + 清零；真机后续若 `_needs_seq_lens_cpu_sync` 才有 D2H",
         bias="中性（无 async spec decode 时）/ 低估（有 async spec decode 时缺 D2H 同步）",
         risk="中：async spec decode + `_needs_seq_lens_cpu_sync` 路径上真机有 D2H + event 同步"),
    dict(step="discard_mask（numpy 比较 + 两次 copy_to_gpu）",
         loc="vllm_ascend/worker/model_runner_v1.py:1091-1099",
         runnable="是", shim="copy_to_gpu→CPU memcpy",
         real_cpu="bool→uint8 转换 + 两次 H2D（各 O(num_reqs)）", bias="中性", risk="低"),
    dict(step="num_accepted_tokens（Event.synchronize + np 赋值 + copy_to_gpu）",
         loc="vllm_ascend/worker/model_runner_v1.py:1103-1118",
         runnable="是（shim 下 event no-op）",
         shim="torch.npu.Event → CpuShimEvent；copy_to_gpu→CPU memcpy",
         real_cpu="混合模型才走 `num_accepted_tokens_event.synchronize()`（真机会阻塞）；非混合路径只做 O(num_reqs) 拷贝",
         bias="低估", risk="中：混合模型（Mamba/align）路径上 event 同步真机是同步点"),
    dict(step="num_computed_tokens（.copy_(non_blocking=True) / to(device)）",
         loc="vllm_ascend/worker/model_runner_v1.py:1133-1158",
         runnable="是", shim="device 重映射（`.to(self.device)`）+ non_blocking 参数保留但无异步",
         real_cpu="pinned H2D 异步拷贝 O(num_reqs)；async spec decode 时还有 D2H→GPU 的修正 kernel",
         bias="低估（异步被退化为同步，且设备侧 kernel 不存在）",
         risk="中：`update_num_computed_tokens_for_batch_change` 是设备侧 kernel（无卡下不执行）"),
    dict(step="req_indices / query_pos / num_scheduled_tokens 的 copy_to_gpu",
         loc="vllm_ascend/worker/model_runner_v1.py:1160-1166",
         runnable="是", shim="copy_to_gpu→CPU memcpy",
         real_cpu="3 次小 H2D（O(tokens)、O(num_reqs)）", bias="中性偏低估（丢失异步重叠）", risk="低"),
    dict(step="positions（device 侧表达式 num_computed_tokens[req_indices_gpu]+query_pos_gpu）",
         loc="vllm_ascend/worker/model_runner_v1.py:1240-1244",
         runnable="是", shim="无（CPU 上等价的 torch 索引+加法）",
         real_cpu="设备侧 gather+add kernel（CPU 只付 launch ~µs）+ 可能的 D2H 同步",
         bias="中性偏高估（CPU 版真的搬数据；真机 kernel 只付 launch）",
         risk="中：CPU 版 O(total_tokens) 内存访问，若 token 多会明显比真机 CPU 侧贵"),
    dict(step="seq_lens（切片赋值 + fill_）",
         loc="vllm_ascend/worker/model_runner_v1.py:1245-1248",
         runnable="是", shim="无", real_cpu="设备侧 kernel，CPU 侧仅 launch",
         bias="中性偏高估（CPU 版算力等价但访存模式不同）", risk="低"),
    dict(step="_prepare_input_ids（含 spec/async 分支）",
         loc="vllm_ascend/worker/model_runner_v1.py:1067 → vllm/v1/worker/gpu_model_runner.py:1761",
         runnable="是", shim="device 重映射 + copy 退化",
         real_cpu="CPU 侧 index/拷贝 + 若干 H2D/D2H；spec decode 时还写入 GPU buffer",
         bias="低估（异步拷贝被同步化）", risk="中：spec decode 分支的 `_draft_token_ids` 处理是设备相关"),
    dict(step="spec_decode_metadata（np.repeat/cumsum + pin_memory().to(device)）",
         loc="vllm_ascend/worker/model_runner_v1.py:1370-1400 `_calc_spec_decode_metadata`",
         runnable="是（pin 需降级）", shim="pin_memory 降级 + device 重映射",
         real_cpu="O(total_draft_tokens) numpy + 5 次 `.pin_memory().to(device, non_blocking=True)` H2D",
         bias="低估（丢失 pinned 异步语义）",
         risk="中：MTP/spec decode 场景偏差最大的一段；建议真机单独测这 5 次拷贝"),
    dict(step="lora（set_active_loras）",
         loc="vllm_ascend/worker/model_runner_v1.py:1411-1413",
         runnable="路径未启用（lora_config 为空）时应为 no-op；启用时需要 LoRA 权重在 CPU/设备两侧",
         shim="无额外（若启用需 device 重映射）",
         real_cpu="CPU 侧索引 + 设备侧 copy（LoRA 权重行）", bias="中性（未启用时 0 成本）",
         risk="低：本 harness 不启用 LoRA；启用时需重新审计"),
    dict(step="logits_indices（query_start_loc.gpu[1:]-1 / spec metadata）",
         loc="vllm_ascend/worker/model_runner_v1.py:1267 与 1391",
         runnable="是", shim="无（返回 view，不拷贝）",
         real_cpu="纯 view 运算（真机也只是切片+sub kernel）", bias="中性", risk="低"),
    dict(step="lmhead_tp pad（nn.functional.pad）",
         loc="vllm_ascend/worker/model_runner_v1.py:1417-1419",
         runnable="是（CPU 版 pad）", shim="无",
         real_cpu="设备侧 pad（CPU 只付 launch）；pad 到 max_num_reqs×uniform_query_len",
         bias="中性偏高估（CPU 版真分配+拷贝）", risk="低；仅 lmhead_tp_enable() 时触发"),
    dict(step="_build_attn_state（AscendAttentionState 判定）",
         loc="vllm_ascend/worker/model_runner_v1.py:1424",
         runnable="是", shim="无", real_cpu="纯 numpy all/eq 比较 O(num_reqs)",
         bias="中性", risk="低"),
    dict(step="_compute_prev_positions（async scheduling）",
         loc="vllm_ascend/worker/model_runner_v1.py:949",
         runnable="是", shim="no（纯 CPU）", real_cpu="O(num_reqs) 索引",
         bias="中性", risk="低"),
    dict(step="_determine_batch_execution_and_padding（plan/padding 决策）",
         loc="vllm_ascend/worker/model_runner_v1.py:2776（父类 vllm/v1/worker/gpu_model_runner.py:3877）",
         runnable="是（依赖 device_count/显存常量）",
         shim="必须伪造 is_available=True、device_count、get_device_properties（否则分支/断言不同）",
         real_cpu="纯 Python 决策 + cudagraph batche 查表 + DP all-gather（TP1/DP1 时无通信）",
         bias="中性（但**分支敏感**）",
         risk="中：若 shim 报 device_count=0，会走与真机不同的 padding/CUDAGraph 分支，导致 prepare_input 的输入形状改变"),
    dict(step="maybe_create_ubatch_slices / dynamic_eplb / compress / mamba 分支",
         loc="vllm_ascend/worker/model_runner_v1.py:1977/1985/2032/1994",
         runnable="默认配置下均不触发（enable_dbo=False / dynamic_eplb=False / use_compress=False / mamba_cache_mode≠align）",
         shim="—", real_cpu="—", bias="中性", risk="低；触发时需要按各自分支单独审计"),
    dict(step="_pad_query_start_loc_for_fia（FULL graph / SP 时补 query_start_loc）",
         loc="vllm_ascend/worker/model_runner_v1.py:834（调用点 :2056）",
         runnable="是", shim="copy_to_gpu→CPU memcpy",
         real_cpu="O(num_reqs_padded) 写 + 一次 H2D", bias="中性偏低估", risk="低；仅 graph/SP 时进入"),
    dict(step="_build_attention_metadata（Ascend 注意力元数据）",
         loc="vllm_ascend/worker/model_runner_v1.py:2875",
         runnable="是（需真实 attn backend 对象存在）",
         shim="device 重映射；内部若有 pin/copy 需退化",
         real_cpu="大量 Python 记账 + 若干设备张量准备（seq_lens/block_tables/attn mask）",
         bias="中性",
         risk="中：这是 scope 内**代码量最大**的一段（含 MLA/DSA/GDN 等多种 builder），无卡下只验证到构造层面"),
    dict(step="_sanitize_placeholder_input_ids_for_forward",
         loc="vllm_ascend/worker/model_runner_v1.py:2079",
         runnable="是", shim="device 重映射（masked_fill_ 在 CPU tensor 上）",
         real_cpu="设备侧 masked_fill kernel（CPU 只付 launch）",
         bias="中性偏高估", risk="低；仅 spec decode placeholder 时进入"),
    dict(step="_preprocess（input_ids/inputs_embeds/positions/model_kwargs）",
         loc="vllm/v1/worker/gpu_model_runner.py:3490",
         runnable="是", shim="device 重映射；spec decode 时 `input_ids.gpu[...].clamp_(min=0)` 变成 CPU clamp",
         real_cpu="纯文本模型 + 无 spec 时几乎全是 view/切片（CPU ≈ 0）；有 spec 时 1 次 clamp kernel；多模态时大量设备工作",
         bias="中性（文本路径）/ 低估（多模态、prompt embeds）",
         risk="低（文本主线）；多模态路径需要额外 shim"),
    dict(step="update_cos_sin(positions)（scope 的最后一个语句）",
         loc="vllm_ascend/ops/rotary_embedding.py:144",
         runnable="是（全局 cache 未初始化时直接 return）",
         shim="无（但 `_cos/_sin/_cos_sin_cache` 必须在真机上存在并是设备张量）",
         real_cpu="3 个设备 kernel（index_select/repeat/copy into global cache），CPU 只付 launch；真机上有 O(max_num_batched_tokens×rope_dim) 的设备侧带宽消耗",
         bias="中性偏低估（CPU 版在 shim 下要么 return 要么按 CPU 张量算；不影响 CPU 负载形状）",
         risk="中：这是设备侧热点（不在本 CPU 分析口径内），但决定「scope 边界」—— 归因时必须说明它算在 prepare_input 里"),
    dict(step="AscendKVBlockZeroer.zero_block_ids（_update_states → _zero_block_ids）",
         loc="vllm_ascend/worker/utils.py:169（调用 vllm/v1/worker/gpu_model_runner.py:1147）",
         runnable="部分（需要 `new_block_ids_to_zero` 非空 + device 重映射）",
         shim="device 重映射 + pin 降级 + triton kernel 替换（`_zero_kv_blocks_kernel`）",
         real_cpu="pinned buffer 写入 + H2D 异步 + 1 次 triton launch（grid 与 block 数成正比）",
         bias="不确定（同 compute_slot_mapping）",
         risk="中：只在有新块需要清零的 step 触发；harness 若不构造该场景会**漏掉**这部分成本"),
    dict(step="【装配】init_ascend_config(vllm_config)",
         loc="`vllm_ascend/ascend_config.py`（真机由 NPUWorker.init_device 建立）",
         runnable="是（harness 手工补装）",
         shim="无（纯 Python 配置对象）",
         real_cpu="真机是 worker 初始化期的一次性工作（≈0，不在单步热路径）",
         bias="中性",
         risk="**中**：`_prepare_inputs` 末尾 `lmhead_tp_enable()` 与 `execute_model` 顶部的 "
              "`profiling_chunk_config.need_timing` 分支都读它；补装值必须与真机一致，否则整条分支不同（会导致占比口径不可比）"),
    dict(step="【装配】最小 KVCacheConfig（kv_cache_groups）",
         loc="`vllm/v1/kv_cache_interface.py`（真机由 initialize_kv_cache 产出）",
         runnable="是（harness 手工构造）",
         shim="无",
         real_cpu="真机是引擎初始化期的一次性工作（≈0）",
         bias="中性（但**若字段填错会静默影响 slot_mapping 数值**）",
         risk="**中**：`block_size` / `blocks_per_phys_block` / `UniformTypeKVCacheSpecs` 会改变 "
              "`BlockTable` 的 physical↔logical 映射，进而改变 `compute_slot_mapping` 的结果与成本形状"),
    dict(step="【装配】分布式 group 桩（get_pp_group / get_dcp_group）",
         loc="`vllm/distributed/parallel_state.py:1377`（未初始化时 AssertionError）",
         runnable="是（harness 手工打桩 world_size=1）",
         shim="无（Python 对象桩）",
         real_cpu="真机 TP1/PP1 下这些 group 也已建立，访问是 O(1) 属性读",
         bias="中性",
         risk="低-中：`BlockTable.__init__` 读 `get_dcp_group().world_size`；`execute_model`/`_update_states` 读 "
              "`get_pp_group().is_last_rank`。桩值必须等价于 TP1/PP1（`is_last_rank=True`），否则会跳进不同的分支"),
    dict(step="【装配】triton_cpu.install（slot-mapping kernel 的 CPU 等价实现）",
         loc="`harness/pi_harness/runner/triton_cpu.py::install()`",
         runnable="是",
         shim="**必须**：替换 `vllm_ascend.ops.triton.compute_slot_mapping._compute_slot_mapping_kernel` "
              "**以及** `vllm_ascend.worker.block_table` 模块里的同名绑定（`from ... import` 已固化）",
         real_cpu="真机：Python launch 开销 + 设备侧并行执行；harness：numpy 向量化（**真搬数据**）",
         bias="不确定",
         risk="**高**：算错即静默污染 slot_mapping → 后续 attention 元数据全歪。必须与真机逐值对比；"
              "另需 `set_launch_cost_us()` 注入 launch 成本再比时间"),
]


def api_table_md(rows, chain, dyn, pkg_counts) -> str:
    out = ["| API | 出现位置 (file:line) | 静态命中 (链内/包内) | 运行时命中 | 类别 | shim 行为 | 风险 |",
           "|---|---|---|---|---|---|---|"]
    for row in rows:
        api, rx, dyn_keys, cat, shim, risk = row[:6]
        pos_override = row[6] if len(row) > 6 else None
        c_hits, c_sites = scan(rx, chain)
        p_hits, p_files = pkg_counts.get(api, (0, {}))
        if pos_override:
            pos = pos_override
        else:
            shown = c_sites[:6] if c_sites else [f"{n}" for n in top_files(p_files, 3)]
            pos = ", ".join(shown[:6]) or "—"
            if len(c_sites) > 6:
                pos += f" …(+{len(c_sites) - 6})"
        d_hits = sum(dyn["values"].get(k, 0) for k in dyn_keys) if dyn_keys else 0
        d_src = ""
        if dyn_keys:
            srcs = sorted({dyn["sources"].get(k, "") for k in dyn_keys if dyn["values"].get(k)})
            d_src = f" ({', '.join(s for s in srcs if s)})" if srcs else ""
        out.append(f"| `{api}` | {pos} | {c_hits} / {p_hits} | {d_hits}{d_src} | {cat} | "
                   f"{shim} | {risk} |")
    return "\n".join(out)


def matrix_md(rows) -> str:
    out = ["| 子步骤 | 代码位置 | 无卡可执行? | 需要的 shim | 真机 CPU 侧行为 | shim 偏差方向 | 风险 |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| {step} | `{loc}` | {runnable} | {shim} | {real_cpu} | {bias} | {risk} |".format(**r))
    return "\n".join(out)


def main() -> int:
    import shlex
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--ascend-src", default="/work/refs/vllm-ascend/vllm_ascend")
    ap.add_argument("--vllm-src", default="/work/refs/vllm/vllm/v1")
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    data = Path(args.data)
    scope = _load(data / "devdeps_scope.json")
    if not scope:
        print("missing devdeps_scope.json (run scope_prepare_input.py first)", file=sys.stderr)
        return 2
    chain = chain_ranges(scope)
    dyn = dynamic_index(data)

    roots = [Path(args.ascend_src), Path(args.vllm_src)]
    pkg_counts = {}
    for row in API_ROWS:
        api, rx = row[0], row[1]
        pkg_counts[api] = scan_tree(rx, roots)

    generated = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    header = (
        f"> 生成时间：{generated}　|　脚本：`harness/probes/gen_devdeps_tables.py`　|　"
        f"scope：`{Path(scope['model_runner_v1']).name}` L{scope['scope_start_line']}-{scope['scope_end_line']} "
        f"(`execute_model`)\n>\n"
        f"> 静态扫描范围：**链内** = prepare_input scope 的直连语句 + 其可达被调函数（AST 展开 ≤2 层，"
        f"覆盖 {len(chain)} 个文件）；"
        f"**包内** = `{args.ascend_src}` + `{args.vllm_src}` 全量。\n>\n"
        f"> 运行时命中：`harness/probes/probe_dynamic_hits.py`（计数代理）+ `probes/shim_patch_proposal.py` 自测；"
        f"**prepare_input 主路径的逐步计数需要 runner（replay_harness）产出后再回填**。\n"
    )
    apilist_md = "# prepare_input 设备依赖 API 清单（静态 × 动态交叉验证）\n\n" + header + "\n" + \
        api_table_md(API_ROWS, chain, dyn, pkg_counts) + "\n"
    # 动态计数明细（不隐藏任何被计数的 key）
    apilist_md += ("\n## 动态实验计数明细（原始 key，未做任何加权）\n\n"
                   "| key | 次数 | 来源 |\n|---|---|---|\n")
    for k, v in sorted(dyn["values"].items(), key=lambda kv: -kv[1]):
        apilist_md += f"| `{k}` | {v} | {dyn['sources'].get(k, '')} |\n"
    apilist_md += ("\n计数口径：`probe_dynamic_hits.py` 在同一次运行里给 `torch.npu.*` / "
                   "`torch.accelerator.*` / `torch_npu.*` 全部套了计数代理，"
                   "所以「运行时命中 = 0」只说明**本轮动态实验没走到**，不代表真机不调用；"
                   "prepare_input 主路径的逐步计数等 runner（replay_harness）跑起来后回填同一张表。\n")
    (data / "devdeps_apilist.md").write_text(apilist_md)

    matrix_md_txt = ("# prepare_input 子步骤 × 无卡可执行性 × shim × 风险\n\n" + header + "\n" +
                     matrix_md(MATRIX_ROWS) + "\n")
    (data / "devdeps_matrix.md").write_text(matrix_md_txt)

    matrix_json = {
        "generated_at": generated,
        "scope": {k: scope[k] for k in ("scope_label", "scope_start_line", "scope_end_line",
                                        "owner_function", "top_level_statements", "scope_steps",
                                        "model_runner_v1", "model_runner_v1_sha256")},
        "rows": MATRIX_ROWS,
        "summary": {
            "total": len(MATRIX_ROWS),
            "runnable_yes": sum(1 for r in MATRIX_ROWS if r["runnable"].startswith("是") or r["runnable"].startswith("部分")),
            "runnable_no": sum(1 for r in MATRIX_ROWS if r["runnable"].startswith("**否")),
            "bias_underestimate": sum(1 for r in MATRIX_ROWS if "低估" in r["bias"]),
        },
    }
    (data / "devdeps_matrix.json").write_text(json.dumps(matrix_json, indent=2, ensure_ascii=False) + "\n")

    apilist_json = {
        "generated_at": generated,
        "chain_files": {k: v for k, v in chain.items()},
        "rows": [],
        "dynamic": dyn,
    }
    for row in API_ROWS:
        api, rx, dyn_keys, cat, shim, risk = row[:6]
        c_hits, c_sites = scan(rx, chain)
        p_hits, p_files = pkg_counts.get(api, (0, {}))
        apilist_json["rows"].append({
            "api": api, "regex": rx, "chain_hits": c_hits, "chain_sites": c_sites,
            "pkg_hits": p_hits, "pkg_files": p_files, "category": cat,
            "shim": shim, "risk": risk,
            "pos_override": row[6] if len(row) > 6 else None,
            "runtime_hits": sum(dyn["values"].get(k, 0) for k in dyn_keys),
            "runtime_keys": dyn_keys,
        })
    (data / "devdeps_apilist.json").write_text(json.dumps(apilist_json, indent=2, ensure_ascii=False) + "\n")

    # ---------------- manifest 合并 ----------------
    host = _load(data / "devdeps_manifest_host.json") or {}
    cont = _load(data / "devdeps_manifest_container.json") or {}

    def sha(p: Path) -> str | None:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            return None

    commands = []
    for p in sorted((data / "devdeps_stacks").glob("*.txt")):
        head = p.read_text(errors="replace").splitlines()[:8]
        cmds = [l for l in head if l.startswith("#")]
        commands.append({"log": str(p.relative_to(data)), "meta": cmds})

    manifest = {
        "generated_at": generated,
        "owner": "probe_devdeps",
        "script_sha256": {
            "gen_devdeps_tables.py": sha(Path(__file__)),
            "scope_prepare_input.py": sha(Path(__file__).parent / "scope_prepare_input.py"),
            "import_gradient.py": sha(Path(__file__).parent / "import_gradient.py"),
            "smoke_shim.py": sha(Path(__file__).parent / "smoke_shim.py"),
            "probe_dynamic_hits.py": sha(Path(__file__).parent / "probe_dynamic_hits.py"),
            "shim_patch_proposal.py": sha(Path(__file__).parent / "shim_patch_proposal.py"),
            "collect_manifest.py": sha(Path(__file__).parent / "collect_manifest.py"),
        },
        "image": host,
        "container": cont,
        "artifacts": sorted(str(p.relative_to(data)) for p in data.glob("devdeps_*")),
        "logs": commands,
        "run_entrypoint": "ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh \"<cmd>\"'",
        "notes": [
            "无卡容器：不挂任何 /dev/davinci*；--network none；绑核 200-215（不使用 120-159）",
            "TORCH_DEVICE_BACKEND_AUTOLOAD=0；PYTHONPATH=/work/harness；cwd=/work",
            "容器内 uid 无 passwd 条目，必须给 USER/LOGNAME（pi-docker.sh 已加）",
        ],
    }
    (data / "devdeps_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n")

    print(f"wrote {data}/devdeps_apilist.md / devdeps_apilist.json / "
          f"devdeps_matrix.md / devdeps_matrix.json / devdeps_manifest.json")
    if args.do_print:
        print("\n" + apilist_md + "\n" + matrix_md_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
