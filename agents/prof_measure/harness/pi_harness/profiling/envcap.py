#!/usr/bin/env python3
"""环境快照 + manifest 工具：核/频率/SMT/NUMA/噪声/镜像 digest/脚本哈希。

用法：
    python3 -m pi_harness.profiling.envcap --cpus 200-215 --json env.json
    python3 -m pi_harness.profiling.envcap --cpus 200-215 --image IMG --json env.json
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import CALIBER_VERSION


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def now_ts() -> str:
    """ISO8601 本地时间戳（时区由宿主机决定）。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run(cmd: Sequence[str], timeout: float = 20.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        return 127, "", str(exc)


def read_text(path: os.PathLike[str] | str, default: str | None = None) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return default


def parse_cpu_list(spec: str) -> list[int]:
    """解析 200-215 / 200,202-204 形式的核列表。"""
    out: list[int] = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def format_cpu_list(cpus: Iterable[int]) -> str:
    """把核列表压回 200-215 形式。"""
    cpus = sorted(set(int(c) for c in cpus))
    if not cpus:
        return ""
    ranges: list[tuple[int, int]] = []
    lo = prev = cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append((lo, prev))
        lo = prev = c
    ranges.append((lo, prev))
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def sha256_file(path: os.PathLike[str] | str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def write_json(path: os.PathLike[str] | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# 拓扑 / 频率 / NUMA
# --------------------------------------------------------------------------- #
def numa_map() -> dict[int, int]:
    """{cpu: numa_node}，来自 /sys/devices/system/node/node*/cpulist。"""
    out: dict[int, int] = {}
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        return out
    for node_dir in base.glob("node[0-9]*"):
        try:
            node = int(node_dir.name[4:])
        except ValueError:
            continue
        cpulist = read_text(node_dir / "cpulist", "") or ""
        for cpu in parse_cpu_list(cpulist):
            out[cpu] = node
    return out


def cpu_env(cpus: Sequence[int]) -> dict[str, Any]:
    """采集核相关的口径信息（频率 / SMT 兄弟 / NUMA）。"""
    numa = numa_map()
    per_cpu: dict[str, Any] = {}
    for cpu in cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        sib = read_text(base / "topology" / "thread_siblings_list", "") or ""
        cur = read_text(base / "cpufreq" / "scaling_cur_freq", "") or ""
        smax = read_text(base / "cpufreq" / "scaling_max_freq", "") or ""
        cmax = read_text(base / "cpufreq" / "cpuinfo_max_freq", "") or ""
        per_cpu[str(cpu)] = {
            "thread_siblings_list": sib,
            "numa_node": numa.get(cpu),
            "scaling_cur_freq_khz": int(cur) if cur.isdigit() else None,
            "scaling_max_freq_khz": int(smax) if smax.isdigit() else None,
            "cpuinfo_max_freq_khz": int(cmax) if cmax.isdigit() else None,
        }

    # SMT 竞争判定：cpuset 内是否出现同一个物理核的两个逻辑核
    phys: dict[str, list[int]] = {}
    for cpu in cpus:
        sib = per_cpu[str(cpu)]["thread_siblings_list"] or str(cpu)
        phys.setdefault(sib, []).append(cpu)
    smt_pairs = {k: v for k, v in phys.items() if len(v) > 1}

    freqs = [v["scaling_cur_freq_khz"] for v in per_cpu.values() if v["scaling_cur_freq_khz"]]
    nodes = sorted({v["numa_node"] for v in per_cpu.values() if v["numa_node"] is not None})
    governor = None
    if cpus:
        governor = read_text(
            Path(f"/sys/devices/system/cpu/cpu{cpus[0]}/cpufreq/scaling_governor"), "")
    return {
        "cpus": list(cpus),
        "cpuset": format_cpu_list(cpus),
        "n_cpus": len(cpus),
        "n_physical_cores": len(phys),
        "smt_engaged": bool(smt_pairs),
        "smt_sibling_groups": {k: v for k, v in sorted(smt_pairs.items())},
        "numa_nodes": nodes,
        "per_cpu": per_cpu,
        "scaling_cur_freq_khz_min": min(freqs) if freqs else None,
        "scaling_cur_freq_khz_max": max(freqs) if freqs else None,
        "scaling_cur_freq_khz_mean": (sum(freqs) / len(freqs)) if freqs else None,
        "governor": governor,
    }


def cpu_model() -> dict[str, Any]:
    info: dict[str, Any] = {}
    text = read_text("/proc/cpuinfo", "") or ""
    for line in text.splitlines():
        if ":" not in line:
            if not line.strip():
                break
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if k in ("CPU implementer", "CPU architecture", "CPU variant", "CPU part",
                 "CPU revision", "model name", "Hardware", "BogoMIPS"):
            info.setdefault(k, v)
    return info


def host_env() -> dict[str, Any]:
    loadavg = (read_text("/proc/loadavg", "") or "").split()
    rc, lscpu, _ = run(["lscpu"])
    summary: dict[str, str] = {}
    if rc == 0:
        for line in lscpu.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                summary[k.strip()] = v.strip()
    mem_kb = None
    for line in (read_text("/proc/meminfo", "") or "").splitlines():
        if line.startswith("MemTotal:"):
            mem_kb = int(line.split()[1])
            break
    nproc = None
    try:
        nproc = len(os.sched_getaffinity(0))
    except Exception:  # pragma: no cover
        pass
    return {
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "nproc_affinity": nproc,
        "cpu_model": cpu_model(),
        "lscpu": {k: summary.get(k) for k in
                  ("Architecture", "CPU(s)", "Thread(s) per core", "Core(s) per socket",
                   "Socket(s)", "NUMA node(s)", "CPU max MHz", "CPU min MHz",
                   "L1d cache", "L2 cache", "L3 cache", "Model name") if k in summary},
        "mem_total_kb": mem_kb,
        "uptime": read_text("/proc/uptime", ""),
        "loadavg": loadavg[:3],
        "perf_event_paranoid": read_text("/proc/sys/kernel/perf_event_paranoid", ""),
        "perf_event_max_sample_rate": read_text(
            "/proc/sys/kernel/perf_event_max_sample_rate", ""),
        "ts": now_ts(),
    }


def image_env(image: str) -> dict[str, Any]:
    """docker 镜像信息（digest 必须写进 manifest）。"""
    out: dict[str, Any] = {"image": image}
    rc, stdout, stderr = run(["docker", "image", "inspect", image,
                              "--format", "{{json .}}"], timeout=30)
    if rc == 0 and stdout.strip():
        try:
            data = json.loads(stdout.strip())
            out["image_id"] = data.get("Id")
            out["repo_digests"] = data.get("RepoDigests")
            out["created"] = data.get("Created")
        except json.JSONDecodeError:
            out["raw"] = stdout.strip()[:2000]
    else:
        out["error"] = f"docker image inspect failed: {stderr.strip()[:200]}"
    return out


def top_procs(n: int = 12) -> list[dict[str, Any]]:
    rc, stdout, _ = run(["ps", "-eo", "pid,user,pcpu,pmem,comm", "--sort=-pcpu"])
    rows: list[dict[str, Any]] = []
    if rc == 0:
        for line in stdout.splitlines()[1 : n + 1]:
            parts = line.split(None, 4)
            if len(parts) == 5:
                try:
                    rows.append({"pid": int(parts[0]), "user": parts[1],
                                 "pcpu": float(parts[2]), "pmem": float(parts[3]),
                                 "comm": parts[4]})
                except ValueError:
                    continue
    return rows


def build_manifest(*,
                   tag: str,
                   cmd: Sequence[str] | str,
                   cpus: Sequence[int],
                   image: str | None = None,
                   scripts: Sequence[str] = (),
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """构造一次采集的完整 manifest。"""
    man: dict[str, Any] = {
        "caliber_version": CALIBER_VERSION,
        "tag": tag,
        "ts": now_ts(),
        "cmd": list(cmd) if not isinstance(cmd, str) else cmd,
        "host": host_env(),
        "cpu": cpu_env(cpus),
        "top_procs": top_procs(),
    }
    if image:
        man["image"] = image_env(image)
    if scripts:
        man["script_sha256"] = {str(s): sha256_file(s) for s in scripts}
    if extra:
        man["extra"] = extra
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description="环境快照（核/频率/SMT/NUMA/镜像）")
    ap.add_argument("--cpus", default="200-215", help="核列表，如 200-215")
    ap.add_argument("--image", default=None, help="docker 镜像（可选，取 digest）")
    ap.add_argument("--json", default=None, help="输出 JSON 路径；缺省打印 stdout")
    args = ap.parse_args()

    cpus = parse_cpu_list(args.cpus)
    env: dict[str, Any] = {
        "caliber_version": CALIBER_VERSION,
        "host": host_env(),
        "cpu": cpu_env(cpus),
        "top_procs": top_procs(),
    }
    if args.image:
        env["image"] = image_env(args.image)
    if args.json:
        write_json(args.json, env)
    else:
        print(json.dumps(env, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
