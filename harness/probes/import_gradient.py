#!/usr/bin/env python3
"""A. 导入梯度实验（无卡容器内逐条执行，全部落盘）。

每一步都是一个**独立的 python 子进程**，stdout/stderr 原样保存到
``data/harness/devdeps_stacks/<step_id>.txt``，摘要写进
``data/harness/devdeps_imports.json``。

用法（唯一合法运行入口，见 harness/INTERFACES.md 第 1 节）::

    ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
        bash harness/scripts/pi-docker.sh "python harness/probes/import_gradient.py --list"'
    ssh a3-22 '... pi-docker.sh "python harness/probes/import_gradient.py"'            # 全跑
    ssh a3-22 '... pi-docker.sh "python harness/probes/import_gradient.py --only A4c"'

环境变量:
  PI_REPO_ROOT   仓库根（容器内默认 /work）
  PI_DEVD_DEPS_OUT 输出目录（默认 $PI_REPO_ROOT/data/harness）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO = Path(os.environ.get("PI_REPO_ROOT", "/work")).resolve()
OUT = Path(os.environ.get("PI_DEVD_DEPS_OUT", REPO / "data" / "harness")).resolve()
STACKS = OUT / "devdeps_stacks"


# --------------------------------------------------------------------------
# 每一步的 python 代码。/work 是容器里的仓库根，PYTHONPATH=/work/harness。
# --------------------------------------------------------------------------

A1_TORCH = r"""
import json, os, sys
import torch

def safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as e:            # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"

info = {
    "python": sys.version.split()[0],
    "torch.__version__": torch.__version__,
    "torch.version.cuda": torch.version.cuda,
    "torch.version.rocm": getattr(torch.version, "rocm", None),
    "torch.version.hip": getattr(torch.version, "hip", None),
    "torch.version.xpu": safe(lambda: torch.version.xpu),
    "env.TORCH_DEVICE_BACKEND_AUTOLOAD": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
    "env.TORCH_DEVICE": os.environ.get("TORCH_DEVICE"),
    "hasattr(torch,'npu')": hasattr(torch, "npu"),
    "hasattr(torch,'cuda')": hasattr(torch, "cuda"),
    "hasattr(torch,'xpu')": hasattr(torch, "xpu"),
    "privateuse1_name": safe(torch._C._get_privateuse1_backend_name),
    "torch_npu_in_sys_modules": [k for k in sys.modules if "npu" in k.lower()],
    "cuda_is_available": safe(torch.cuda.is_available),
    "cuda_device_count": safe(torch.cuda.device_count),
    "torch.device('npu:0')": safe(lambda: str(torch.device("npu:0"))),
    "torch.device('npu')": safe(lambda: str(torch.device("npu"))),
    "torch.device('privateuseone:0')": safe(lambda: str(torch.device("privateuseone:0"))),
    "torch.zeros(1, device='npu')": safe(lambda: str(torch.zeros(1, device="npu"))),
    "torch.zeros(1, device='privateuseone')": safe(lambda: str(torch.zeros(1, device="privateuseone"))),
    "torch.zeros(4, pin_memory=True).shape": safe(lambda: tuple(torch.zeros(4, pin_memory=True).shape)),
    "torch.Tensor.npu exists": hasattr(torch.Tensor, "npu"),
    "torch.cuda.Stream()": safe(lambda: str(torch.cuda.Stream())),
}
print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
"""

A2_IMPORT_TORCH_NPU = r"""
import json, sys, os
import torch
print("env.TORCH_DEVICE_BACKEND_AUTOLOAD =", os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"))
print("torch:", torch.__version__, "| hasattr(torch,'npu') before import torch_npu =", hasattr(torch, "npu"))
import torch_npu
print("torch_npu OK")
print("torch_npu.__file__ =", getattr(torch_npu, "__file__", None))
print("torch_npu.__version__ =", getattr(torch_npu, "__version__", None))
print("hasattr(torch,'npu') after import torch_npu =", hasattr(torch, "npu"))
print("torch.npu.is_available() =", getattr(torch, "npu", None) and torch.npu.is_available())
print("sys.modules torch_npu =", sys.modules.get("torch_npu"))
"""

A2C_TORCH_AUTOLOAD_FIRST = r"""
import json, sys, os
print("env.TORCH_DEVICE_BACKEND_AUTOLOAD =", os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"))
import torch
mods = sorted(k for k in sys.modules if "npu" in k.lower())
print("modules matching 'npu' right after `import torch`:", mods)
print("hasattr(torch,'npu') =", hasattr(torch, "npu"))
import importlib
print("importlib.util.find_spec('torch_npu') =", importlib.util.find_spec("torch_npu"))
"""

A3_IMPORT_VLLM = r"""
import vllm, sys
print("vllm.__version__ =", vllm.__version__)
print("vllm.__file__ =", vllm.__file__)
import vllm.v1.worker.gpu_model_runner as gmr
print("gpu_model_runner OK:", gmr.__file__)
"""

A3B_IMPORT_VLLM_C = r"""
import vllm._C as C
print("vllm._C OK:", C.__file__)
print("has ops:", len([x for x in dir(C) if not x.startswith('_')]))
"""

A4A_IMPORT_VLLM_ASCEND = r"""
import vllm_ascend
print("vllm_ascend OK:", vllm_ascend.__file__)
print("version:", getattr(vllm_ascend, "__version__", None))
from vllm_ascend import _build_info
print("_build_info:", {k: getattr(_build_info, k, None) for k in dir(_build_info) if not k.startswith("_")})
"""

A4B_IMPORT_VLLM_ASCEND_PLATFORM = r"""
import vllm_ascend.platform as p
print("vllm_ascend.platform OK:", p.__file__)
print("NPUPlatform:", p.NPUPlatform)
print("device_name:", p.NPUPlatform.device_name)
"""

A4C_IMPORT_VLLM_ASCEND_MODEL_RUNNER = r"""
import vllm_ascend.worker.model_runner_v1 as m
print("vllm_ascend.worker.model_runner_v1 OK:", m.__file__)
print("NPUModelRunner:", m.NPUModelRunner)
print("MRO:", [c.__name__ for c in m.NPUModelRunner.__mro__])
"""

# 正确的导入顺序（先 import vllm 触发 platform plugin → vllm_ascend.ops 完成初始化）
A4D_IMPORT_ORDER_FIX = r"""
import vllm                      # 触发 vllm.platform_plugins: ascend -> vllm_ascend:register
print("step1 import vllm OK")
import vllm_ascend.ops           # 先完成 ops 包（循环导入的另一端）
print("step2 import vllm_ascend.ops OK")
import vllm_ascend.worker.model_runner_v1 as m
print("step3 import vllm_ascend.worker.model_runner_v1 OK:", m.__file__)
print("NPUModelRunner:", m.NPUModelRunner)
print("MRO:", [c.__name__ for c in m.NPUModelRunner.__mro__])
import vllm_ascend.platform as p
print("NPUPlatform.device_type =", p.NPUPlatform.device_type)
"""

A4E_IMPORT_ORDER_ALTERNATIVE = r"""
import vllm_ascend.ops
print("ops first OK")
import vllm_ascend.worker.model_runner_v1 as m
print("model_runner_v1 OK (no explicit import vllm):", m.__file__)
"""

A4F_IMPORT_ORDER_REVERSED = r"""
import vllm_ascend.worker.model_runner_v1 as m     # 反例：直接 import = 循环导入
print("UNEXPECTED OK", m.__file__)
"""

# 每条探针都在**独立子进程**里跑：aclInit 失败有可能直接 abort/segfault，
# 独立进程才能把「崩在第几条」精确记录下来。
A8_DEVICE_RUNTIME = r'''
import json, os, subprocess, sys, textwrap

CHILD = textwrap.dedent(r"""
import json, os, sys, traceback
import torch, torch_npu
expr = os.environ["PI_PROBE_EXPR"]
g = {"torch": torch, "torch_npu": torch_npu}
frame_depth = 0
try:
    v = eval(expr, g)
    rec = {"ok": True, "value": repr(v)[:200]}
except BaseException as e:
    tb = traceback.format_exc().splitlines()
    frames = [l.strip() for l in tb if l.strip().startswith("File ")]
    rec = {"ok": False, "exc": f"{type(e).__name__}: {e}"[:400],
           "raise_site": frames[-1][:240] if frames else ""}
sys.stdout.write("__RESULT__" + json.dumps(rec) + "\n")
""")

PROBES = [
    ("torch.npu.is_available()",                      "torch.npu.is_available()"),
    ("torch.npu.device_count()",                      "torch.npu.device_count()"),
    ("torch.npu.current_device()",                    "torch.npu.current_device()"),
    ("torch.npu.get_device_name(0)",                  "torch.npu.get_device_name(0)"),
    ("torch.npu.get_device_properties(0)",            "torch.npu.get_device_properties(0)"),
    ("torch.npu.current_stream()",                    "torch.npu.current_stream()"),
    ("torch.npu.default_stream()",                    "torch.npu.default_stream()"),
    ("torch.npu.Stream()",                            "torch.npu.Stream()"),
    ("torch.npu.Event()",                             "torch.npu.Event()"),
    ("torch.npu.Event(enable_timing=True)",           "torch.npu.Event(enable_timing=True)"),
    ("torch.npu.ExternalEvent()",                     "torch.npu.ExternalEvent()"),
    ("torch.npu.synchronize()",                       "torch.npu.synchronize()"),
    ("torch.npu.mem_get_info()",                      "torch.npu.mem_get_info()"),
    ("torch.npu.empty_cache()",                       "torch.npu.empty_cache()"),
    ("torch.npu.memory_allocated()",                  "torch.npu.memory_allocated()"),
    ("torch.npu.set_device(0)",                       "torch.npu.set_device(0)"),
    ("torch.npu.is_current_stream_capturing()",       "torch.npu.is_current_stream_capturing()"),
    ("torch.npu.config.allow_internal_format=True",   "setattr(torch.npu.config,'allow_internal_format',True)"),
    ("torch.zeros(4, device='npu')",                  "torch.zeros(4, device='npu')"),
    ("torch.zeros(4, device='npu:0')",                "torch.zeros(4, device='npu:0')"),
    ("torch.zeros(4).npu()",                          "torch.zeros(4).npu()"),
    ("torch.zeros(4).to('npu')",                      "torch.zeros(4).to('npu')"),
    ("torch.zeros(4, pin_memory=True)",               "torch.zeros(4, pin_memory=True)"),
    ("hasattr(torch.Tensor,'npu')",                   "hasattr(torch.Tensor,'npu')"),
    ("torch_npu.npu.is_available()",                  "torch_npu.npu.is_available()"),
    ("torch_npu.npu.current_device()",                "torch_npu.npu.current_device()"),
    ("torch_npu.utils.get_cann_version()",            "__import__('torch_npu.utils',fromlist=['x']).get_cann_version()"),
    ("import acl",                                    "__import__('acl').__name__"),
    ("import torch_npu.profiler.dynamic_profile",     "__import__('torch_npu.profiler.dynamic_profile',fromlist=['x']).__name__"),
    ("import torch_npu.op_plugin.atb._atb_ops",       "__import__('torch_npu.op_plugin.atb._atb_ops',fromlist=['x']).__name__"),
]

results = []
for name, expr in PROBES:
    env = dict(os.environ, PI_PROBE_EXPR=expr)
    try:
        p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                           text=True, env=env, timeout=180)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -999, "", "(timeout>180s)"
    rec = {"api": name, "child_rc": rc}
    for line in out.splitlines():
        if line.startswith("__RESULT__"):
            rec.update(json.loads(line[len("__RESULT__"):]))
    if "ok" not in rec:
        rec["ok"] = False
        rec["exc"] = f"subprocess died rc={rc} (signal/crash)" if rc < 0 else "no result"
        rec["stderr_tail"] = "\n".join(err.strip().splitlines()[-6:])[:500]
    results.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)

print("\n=== SUMMARY (api / ok / detail) ===", flush=True)
for r in results:
    print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['api']:44s} "
          f"{(r.get('value') or r.get('exc') or '')[:120]}", flush=True)
'''

A9_TORCH_NPU_SURFACE = r"""
# torch_npu 顶层 API 面（哪些子模块可 import / 需要 CANN 才有）

import importlib, json, traceback
import torch_npu

print("torch_npu.__file__ =", torch_npu.__file__)
print("torch_npu.__version__ =", getattr(torch_npu, "__version__", None))
print("dir(torch_npu) =", sorted(n for n in dir(torch_npu) if not n.startswith("_"))[:80])

for mod in [
    "torch_npu.npu",
    "torch_npu.npu.utils",
    "torch_npu.utils",
    "torch_npu.profiler",
    "torch_npu.profiler.dynamic_profile",
    "torch_npu.op_plugin",
    "torch_npu.op_plugin.atb",
    "torch_npu.op_plugin.atb._atb_ops",
    "torch_npu.multiprocessing",
    "torch_npu.multiprocessing.reductions",
    "torch_npu._inductor",
    "torch_npu.distributed",
    "torch_npu.asd",
]:
    try:
        m = importlib.import_module(mod)
        print(f"OK   import {mod}  ->  {getattr(m, '__file__', '<no file>')}")
    except BaseException as e:
        loc = traceback.format_exc().splitlines()
        loc = [l.strip() for l in loc if l.strip().startswith("File ")]
        print(f"FAIL import {mod}  ->  {type(e).__name__}: {e}   @ {loc[-1] if loc else ''}")

try:
    import acl
    print("OK   import acl ->", acl.__file__)
    print("acl.init() ->", acl.init())
except BaseException as e:
    print("FAIL import acl ->", type(e).__name__, e)
"""

A5A_PLUGINS = r"""
import os, sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
import vllm, vllm.envs as envs, vllm.plugins as vplugins
from importlib.metadata import entry_points

print("VLLM_PLUGINS repr =", repr(envs.VLLM_PLUGINS))
for group in ("vllm.general_plugins", "vllm.platform_plugins", "vllm.io_processor_plugins"):
    eps = list(entry_points(group=group))
    print(f"group {group}: {[(e.name, e.value) for e in eps]}")
allowed = envs.VLLM_PLUGINS
eps = list(entry_points(group="vllm.general_plugins"))
print("general plugins that pass the allowlist:", [e.name for e in eps if allowed is None or e.name in allowed])
loaded = vplugins.load_plugins_by_group("vllm.general_plugins")
print("load_plugins_by_group('vllm.general_plugins') ->", sorted(loaded))
"""

A5B_TARGET_DEVICE = r"""
import os, sys
import vllm.envs as envs
print("VLLM_TARGET_DEVICE env =", repr(os.environ.get("VLLM_TARGET_DEVICE")))
print("envs.VLLM_TARGET_DEVICE =", repr(envs.VLLM_TARGET_DEVICE))
print("VLLM_USE_V1 env =", repr(os.environ.get("VLLM_USE_V1")))
print("envs.VLLM_USE_V1 =", repr(getattr(envs, "VLLM_USE_V1", "<missing>")))
try:
    from vllm.platforms import current_platform
    print("current_platform =", type(current_platform), current_platform)
    print("device_name =", getattr(current_platform, "device_name", None))
    print("device_type =", getattr(current_platform, "device_type", None))
except BaseException as e:
    import traceback; traceback.print_exc()
    print("current_platform resolution FAILED:", type(e).__name__, e)
    raise SystemExit(3)
"""

A6_ENV_NPU_SMI = r"""
import os, subprocess, sys

def sh(cmd):
    print("$ " + cmd)
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.stderr:
        sys.stdout.write("[stderr] " + p.stderr)
    print(f"[rc={p.returncode}]")
    return p.returncode

for c in ("npu-smi", "ascend-dmi", "msprof"):
    sh(f"command -v {c} || echo 'NOT FOUND: {c}'")
sh("npu-smi info")
sh("ls -l /dev/davinci* 2>&1 | head -5")
sh("ls -l /dev/devmm_svm /dev/hisi_hdc 2>&1 | head -5")
sh("ls /usr/local/Ascend 2>&1 | head -5")
sh("echo NPU_VISIBLE_DEVICES=$NPU_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES")
"""

A7_HOSTINFO = r"""
import json, os, platform, subprocess, sys

def sh(cmd):
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return p.stdout.strip(), p.returncode

import numpy
info = {
    "python": sys.version,
    "executable": sys.executable,
    "numpy.__version__": numpy.__version__,
    "uname_a": platform.uname()._asdict(),
    "os_release": dict(
        l.split("=", 1) for l in open("/etc/os-release").read().splitlines() if "=" in l
    ),
    "lscpu_summary": {},
    "cgroup_cpuset": None,
    "nproc": os.cpu_count(),
    "threads_env": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ATEN_CPU_SECTION", "OPENBLAS_NUM_THREADS")},
    "torch_threads": None,
    "env_PYTHONPATH": os.environ.get("PYTHONPATH"),
    "cwd": os.getcwd(),
}
out, _ = sh("lscpu")
keep = ("Architecture", "Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket",
        "Socket(s)", "NUMA node(s)", "CPU max MHz", "CPU min MHz", "L1d cache", "L2 cache",
        "L3 cache", "BogoMIPS", "Flags")
for line in out.splitlines():
    if ":" not in line:
        continue
    k, v = line.split(":", 1)
    if k.strip() in keep:
        info["lscpu_summary"][k.strip()] = v.strip()[:400]
try:
    cs, _ = sh("cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null")
    info["cgroup_cpuset"] = cs
except Exception:
    pass
try:
    import torch
    info["torch_threads"] = torch.get_num_threads()
    info["torch.__version__"] = torch.__version__
    info["torch_interop_threads"] = torch.get_num_interop_threads()
except Exception:
    pass
print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
"""


STEPS: list[dict] = [
    dict(
        id="A1_torch_basic",
        title="import torch（autoload=0）：版本 / backend / torch.npu 存在性 / torch.zeros(device='npu')",
        env={},
        code=A1_TORCH,
    ),
    dict(
        id="A2a_import_torch_npu_autoload0",
        title="import torch_npu，TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        code=A2_IMPORT_TORCH_NPU,
    ),
    dict(
        id="A2b_import_torch_npu_autoload1",
        title="import torch_npu，TORCH_DEVICE_BACKEND_AUTOLOAD=1",
        env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "1"},
        code=A2_IMPORT_TORCH_NPU,
    ),
    dict(
        id="A2c_torch_with_autoload1",
        title="只 import torch，TORCH_DEVICE_BACKEND_AUTOLOAD=1（看是否自动挂载）",
        env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "1"},
        code=A2C_TORCH_AUTOLOAD_FIRST,
    ),
    dict(
        id="A3a_import_vllm",
        title="import vllm + vllm.v1.worker.gpu_model_runner",
        env={},
        code=A3_IMPORT_VLLM,
    ),
    dict(
        id="A3b_import_vllm_C",
        title="import vllm._C",
        env={},
        code=A3B_IMPORT_VLLM_C,
    ),
    dict(
        id="A4a_import_vllm_ascend",
        title="import vllm_ascend",
        env={},
        code=A4A_IMPORT_VLLM_ASCEND,
    ),
    dict(
        id="A4b_import_vllm_ascend_platform",
        title="import vllm_ascend.platform",
        env={},
        code=A4B_IMPORT_VLLM_ASCEND_PLATFORM,
    ),
    dict(
        id="A4c_import_vllm_ascend_model_runner_v1",
        title="import vllm_ascend.worker.model_runner_v1（目标模块）",
        env={},
        code=A4C_IMPORT_VLLM_ASCEND_MODEL_RUNNER,
    ),
    dict(
        id="A4d_import_order_vllm_then_ops_then_worker",
        title="正确导入顺序验证：import vllm → vllm_ascend.ops → worker.model_runner_v1",
        env={},
        code=A4D_IMPORT_ORDER_FIX,
    ),
    dict(
        id="A4e_import_order_ops_first_only",
        title="只先 import vllm_ascend.ops，再 import worker.model_runner_v1（不显式 import vllm）",
        env={},
        code=A4E_IMPORT_ORDER_ALTERNATIVE,
    ),
    dict(
        id="A4f_import_order_reversed_counterexample",
        title="反例：直接 import worker.model_runner_v1（循环导入）",
        env={},
        code=A4F_IMPORT_ORDER_REVERSED,
    ),
    dict(
        id="A8_device_runtime_probes",
        title="运行时设备交互探针：逐条独立子进程调用 torch.npu.* / torch_npu.*（记录 aclInit 失败点）",
        env={},
        code=A8_DEVICE_RUNTIME,
    ),
    dict(
        id="A9_torch_npu_surface",
        title="torch_npu 顶层子模块可导入性（profiler / op_plugin / multiprocessing / acl）",
        env={},
        code=A9_TORCH_NPU_SURFACE,
    ),
    dict(
        id="A5a_plugins_empty",
        title="VLLM_PLUGINS=\"\"（空串）vs unset 对 general plugin 加载的影响",
        env={"VLLM_PLUGINS": ""},
        code=A5A_PLUGINS,
    ),
    dict(
        id="A5a2_plugins_unset",
        title="VLLM_PLUGINS unset（全部插件）",
        env={},
        unset=["VLLM_PLUGINS"],
        code=A5A_PLUGINS,
    ),
    dict(
        id="A5b_vllm_target_device_npu",
        title="VLLM_TARGET_DEVICE=npu 下的 current_platform 解析",
        env={"VLLM_TARGET_DEVICE": "npu"},
        code=A5B_TARGET_DEVICE,
    ),
    dict(
        id="A5b2_vllm_target_device_unset",
        title="VLLM_TARGET_DEVICE unset（默认 cuda）下的 current_platform 解析",
        env={},
        code=A5B_TARGET_DEVICE,
    ),
    dict(
        id="A5c_vllm_use_v1_0",
        title="VLLM_USE_V1=0",
        env={"VLLM_USE_V1": "0"},
        code=A5B_TARGET_DEVICE,
    ),
    dict(
        id="A6_env_npu_smi",
        title="无卡容器里 npu-smi / ascend-dmi / /dev/davinci* 的情况",
        env={},
        code=A6_ENV_NPU_SMI,
    ),
    dict(
        id="A7_hostinfo_manifest",
        title="numpy / uname / lscpu 摘要 / 线程配置",
        env={},
        code=A7_HOSTINFO,
    ),
]


def _extract_traceback(stderr: str, max_lines: int = 40) -> list[str]:
    lines = stderr.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("Traceback (most recent call last)")]
    if not starts:
        return [l for l in lines if l][:max_lines]
    block = lines[starts[0]:]
    # 截断到第一个非 traceback 段落（通常是下一段 chained 输出后的空行）
    out: list[str] = []
    for l in block:
        if out and l.strip() == "" and out[-1].strip() == "":
            break
        out.append(l)
        if len(out) >= max_lines:
            break
    return out


def _conclusion(rc: int, stderr: str, stdout: str) -> str:
    if rc == 0:
        tail = [l for l in stdout.splitlines() if l.strip()]
        return "成功 (rc=0) | " + (tail[-1][:160] if tail else "no stdout")
    lines = [l for l in stderr.splitlines() if l.strip() and not l.startswith(" ")]
    err = lines[-1] if lines else "(no stderr)"
    return f"失败 (rc={rc}) | {err[:200]}"


def run_step(step: dict, outdir: Path, extra_env: dict | None = None, suffix: str = "") -> dict:
    env = dict(os.environ)
    env.update(extra_env or {})
    env.update(step.get("env") or {})
    for k in step.get("unset") or []:
        env.pop(k, None)
    code = textwrap.dedent(step["code"]).strip("\n")
    cmd = [sys.executable, "-c", code]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    dur = time.time() - t0

    log = outdir / f"{step['id']}{suffix}.txt"
    with open(log, "w") as fh:
        fh.write(f"# step_id     : {step['id']}\n")
        fh.write(f"# title       : {step['title']}\n")
        fh.write(f"# cwd         : {REPO}\n")
        fh.write(f"# returncode  : {p.returncode}\n")
        fh.write(f"# duration_s  : {dur:.2f}\n")
        fh.write("# extra env   : " + json.dumps(extra_env or {}) + "\n")
        fh.write("# env overrides: " + json.dumps({**(step.get('env') or {}),
                                                     **{k: "<unset>" for k in (step.get('unset') or [])}}) + "\n")
        fh.write("# ---- COMMAND (python -c) ----\n")
        fh.write(code + "\n")
        fh.write("# ---- STDOUT ----\n")
        fh.write(p.stdout)
        fh.write("\n# ---- STDERR ----\n")
        fh.write(p.stderr)
        fh.write("\n# ---- END ----\n")

    return dict(
        id=step["id"],
        title=step["title"],
        returncode=p.returncode,
        duration_s=round(dur, 3),
        extra_env=dict(extra_env or {}),
        env_overrides={**(step.get("env") or {}), **{k: "<unset>" for k in (step.get("unset") or [])}},
        cmd=f"{sys.executable} -c '<{len(code)} chars>'  (完整代码见 {log.name})",
        stdout_head=p.stdout.splitlines()[:40],
        stderr_head=p.stderr.splitlines()[:40],
        traceback_head40=_extract_traceback(p.stderr),
        log=str(log.relative_to(REPO) if str(log).startswith(str(REPO)) else log),
        conclusion=_conclusion(p.returncode, p.stderr, p.stdout),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定 step id（前缀匹配）")
    ap.add_argument("--list", action="store_true", help="列出所有 step id 后退出")
    ap.add_argument("--out", default=str(OUT), help=f"输出目录（默认 {OUT}）")
    ap.add_argument("--set-env", action="append", default=[], metavar="K=V",
                    help="给所有子步骤额外注入的环境变量（可重复）。例如 --set-env USER=probe")
    ap.add_argument("--suffix", default="", help="输出文件名后缀，例如 _envfix")
    args = ap.parse_args()

    if args.list:
        for s in STEPS:
            print(f"{s['id']:45s} {s['title']}")
        return 0

    out = Path(args.out).resolve()
    stacks = out / "devdeps_stacks"
    stacks.mkdir(parents=True, exist_ok=True)

    sel = STEPS
    if args.only:
        sel = [s for s in STEPS if any(s["id"].startswith(x) for x in args.only)]
        if not sel:
            print("no step matched", args.only, file=sys.stderr)
            return 2

    extra_env = {}
    for kv in args.set_env:
        k, _, v = kv.partition("=")
        extra_env[k] = v

    results = []
    for s in sel:
        print(f"=== {s['id']} :: {s['title']}", flush=True)
        r = run_step(s, stacks, extra_env=extra_env, suffix=args.suffix)
        results.append(r)
        print(f"    rc={r['returncode']}  {r['conclusion']}", flush=True)

    me = Path(__file__).resolve()
    manifest = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        host=os.uname().nodename,
        script=str(me),
        script_sha256=hashlib.sha256(me.read_bytes()).hexdigest(),
        python=sys.version,
        executable=sys.executable,
        repo_root=str(REPO),
        extra_env=extra_env,
        steps=results,
    )
    jout = out / f"devdeps_imports{args.suffix}.json"
    jout.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {jout} and {len(results)} logs under {stacks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
