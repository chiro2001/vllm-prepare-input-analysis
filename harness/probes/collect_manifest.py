#!/usr/bin/env python3
"""收集 devdeps 实验 manifest（在**容器内**跑）。

产出 `data/harness/devdeps_manifest_container.json`：镜像里的 python/包版本、
环境变量、CPU/cgroup 可见性、绑核事实、以及本目录下探针脚本的 sha256。

宿主机侧信息（镜像 digest、docker run 参数）由 `harness/probes/collect_host_manifest.sh`
收集成 `devdeps_manifest_host.json`；`gen_devdeps_tables.py` 把两者合并进
`data/harness/devdeps_manifest.json`。

用法::

    python harness/probes/collect_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("PI_REPO_ROOT", "/work"))
OUT = Path(os.environ.get("PI_DEVD_DEPS_OUT", REPO / "data" / "harness"))
PROBES = Path(__file__).resolve().parent


def sh(cmd: str) -> dict:
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return {"cmd": cmd, "rc": p.returncode, "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip()[:2000]}


def main() -> int:
    info: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uname": platform.uname()._asdict(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "environ_subset": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "HOME", "USER", "LOGNAME", "TORCH_DEVICE_BACKEND_AUTOLOAD",
            "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_PLUGINS",
            "VLLM_TARGET_DEVICE", "VLLM_USE_V1", "OMP_NUM_THREADS", "PI_HARNESS_ROOT")},
        "cgroup_cpuset": sh("cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null "
                            "|| cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null")["stdout"],
        "nproc_visible": os.cpu_count(),
        "npu_smi": sh("command -v npu-smi || echo '<not found>'"),
        "davinci_dev": sh("ls /dev/davinci* 2>&1 || true"),
        "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
        "ascend_toolkit": sh("ls /usr/local/Ascend 2>/dev/null || true")["stdout"],
    }
    try:
        import numpy
        import torch
        info["numpy"] = numpy.__version__
        info["torch"] = torch.__version__
        info["torch.version.cuda"] = torch.version.cuda
        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except Exception as e:                       # noqa: BLE001
        info["torch"] = f"<{e}>"
    try:
        import torch_npu
        info["torch_npu"] = torch_npu.__version__
        info["torch_npu_file"] = torch_npu.__file__
    except Exception as e:                       # noqa: BLE001
        info["torch_npu"] = f"<{e}>"
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception as e:                       # noqa: BLE001
        info["triton"] = f"<{e}>"

    info["pip_list"] = sh(
        "python -m pip list --format=freeze 2>/dev/null | "
        "grep -Ei '^(vllm|vllm-ascend|torch|torch-npu|numpy|triton|triton-ascend|"
        "decorator|scipy|pyyaml|requests|setuptools)' | sort"
    )["stdout"].splitlines()
    info["vllm_files"] = sh("python -c \"import vllm, vllm_ascend; "
                            "print(vllm.__file__); print(vllm_ascend.__file__)\" 2>/dev/null")["stdout"].splitlines()
    info["lscpu"] = sh("lscpu | head -30")["stdout"]

    # vLLM / vllm-ascend 源码 commit（镜像里是源码目录）
    info["source_commits"] = {}
    for name, path in (("vllm", "/vllm-workspace/vllm"), ("vllm-ascend", "/vllm-workspace/vllm-ascend")):
        info["source_commits"][name] = sh(f"git -C {path} rev-parse HEAD 2>/dev/null || echo '<no git>'")["stdout"]

    def sha(p: Path) -> str | None:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            return None

    info["probe_script_sha256"] = {
        str(p.relative_to(REPO) if str(p).startswith(str(REPO)) else p): sha(p)
        for p in sorted(PROBES.glob("*.py")) + sorted(PROBES.glob("*.sh"))
    }
    info["harness_sha256"] = {
        str(p): sha(Path(p)) for p in (
            "/work/harness/scripts/pi-docker.sh",
            "/work/harness/INTERFACES.md",
            "/work/harness/pi_harness/shim/__init__.py",
        )
    }
    info["source_sha256"] = {
        p: sha(Path(p)) for p in (
            "/work/refs/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
            "/work/refs/vllm-ascend/vllm_ascend/worker/block_table.py",
            "/work/refs/vllm-ascend/vllm_ascend/ops/triton/compute_slot_mapping.py",
            "/work/refs/vllm/vllm/v1/utils.py",
            "/work/refs/vllm/vllm/v1/worker/gpu_model_runner.py",
            "/work/refs/vllm/vllm/v1/worker/gpu_input_batch.py",
        )
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "devdeps_manifest_container.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: info[k] for k in
                      ("torch", "torch_npu", "numpy", "triton", "cgroup_cpuset",
                       "torch_num_threads", "torch_num_interop_threads")},
                     indent=2, ensure_ascii=False))
    print(f"wrote {OUT}/devdeps_manifest_container.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
