"""prepare_input 无卡 CPU profiling 工具包（owner: prof_measure）。

口径版本：prof-v1（2026-09-24 建立）。

设计约束（与 plan/COORDINATION.md 第 6 节、harness/INTERFACES.md 对齐）：

* 容器里只跑负载：被测进程在 pi-docker.sh 同款无卡容器里跑
  （--network none、不挂 NPU、绑 200-239 或 360-399、线程数 ≤16）；
  PMU / perf 采集一律在宿主上以 root 身份做，通过 perf -p <host pid>
  或 libkperfx 的 target=process 挂到容器进程上。
* 一次采集只开一种 instrument：perf 采样与 PMU 计数不共存（会互相抢占
  通用计数器，导致 multiplex 置信度下降），因此同一份负载分多次 pass 采集，
  manifest 里逐 pass 记录。
* 每个占比都带分子分母；topdown / IPC 必带 time_running / time_enabled。
* 采不到的东西写清楚原因，禁止编造。
"""

from __future__ import annotations

import os
from pathlib import Path

CALIBER_VERSION = "prof-v1"

# 默认 libkperfx 安装位置（a3-22 用户目录，绝不写别人的目录）
DEFAULT_LIBKPERFX = Path(os.environ.get("PI_LIBKPERFX", Path.home() / "tools" / "libkperfx"))
# flamegraph-rs 可执行文件（cargo install flamegraph）
DEFAULT_FLAMEGRAPH_RS = Path(
    os.environ.get("PI_FLAMEGRAPH_RS", Path.home() / "tools" / "cargo" / "bin" / "flamegraph")
)
# py-spy（Python 帧火焰图；pip install --user py-spy）
DEFAULT_PYSPY = Path(os.environ.get("PI_PYSPY", Path.home() / ".local" / "bin" / "py-spy"))


def libkperfx_python_dir(root: Path | None = None) -> Path:
    """返回 libkperfx 的 python 绑定目录（内含 kperfx.py）。"""
    return Path(root or DEFAULT_LIBKPERFX) / "python"


def ensure_kperfx_importable(root: Path | None = None) -> str:
    """把 libkperfx 的 python 绑定加进 sys.path，返回 libkperfx 根目录。"""
    import sys

    base = Path(root or DEFAULT_LIBKPERFX)
    py_dir = libkperfx_python_dir(base)
    if not (py_dir / "kperfx.py").is_file():
        raise FileNotFoundError(
            f"找不到 {py_dir}/kperfx.py；请先在 a3-22 上 make libkperfx"
        )
    lib = base / "libkperfx.so"
    if lib.is_file():
        os.environ.setdefault("KPERFX_LIB", str(lib))
    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))
    return str(base)
