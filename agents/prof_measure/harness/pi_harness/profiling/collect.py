#!/usr/bin/env python3
"""无卡 profiling 采集总控：容器跑负载 + 宿主 root 采集（perf / libkperfx）。

一次 run 会做这些事（每步产物留在 out 目录，manifest 记录口径）：

1. 环境快照 + 噪声门（采前 / 采后各一次）；
2. pmu pass：libkperfx 挂到容器进程，取 cycles/instructions/... + 置信度；
3. topdown pass：920B preset（l1 = 1 组；full = 9 组，每组一次负载）；
4. perf pass：perf record -g --call-graph dwarf -> 火焰图 / 热点函数 /
   热点指令 / 源码行（可选 py-spy 出 Python 帧图）；
5. 汇总 summary.json / summary.csv / manifest.json / cmd.sh。

关键工程点：

* STOP 门控：容器以 bash -c 'kill -STOP $$; exec <cmd>' 启动，宿主先把 perf /
  kperfx 挂好（各自写 ready 标记），再 SIGCONT，避免短负载丢头；
* 一次只跑一种 instrument：perf 采样与 PMU 计数不同时开，避免抢通用计数器；
* 容器只跑负载：不挂 NPU、--network none、绑 200-239/360-399、线程 <=16。

用法见 python3 -m pi_harness.profiling.collect --help。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import traceback
import time
from pathlib import Path
from typing import Any, Sequence

from . import (CALIBER_VERSION, DEFAULT_FLAMEGRAPH_RS, DEFAULT_LIBKPERFX,
               DEFAULT_PYSPY)
from .envcap import build_manifest, now_ts, parse_cpu_list, run as sh_run, write_json
from .hostnoise import gate as noise_gate

DEFAULT_IMAGE = "quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler"
DEFAULT_CPUS = "200-215"

#: pi_env.sh 缺失时的兜底（正常情况下以 pi_env.sh 为唯一真源）
FALLBACK_DOCKER_ENVS = [
    "-e", "HOME=/tmp",
    "-e", "USER=REMOTE_USER",
    "-e", "LOGNAME=REMOTE_USER",
    "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/ti_cache",
    "-e", "TRITON_CACHE_DIR=/tmp/triton_cache",
    "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
    "-e", "PI_HARNESS_ROOT=/work/harness",
    "-e", "PYTHONPATH=/work/harness",
    "-e", "PYTHONDONTWRITEBYTECODE=1",
]


def repo_root() -> Path:
    """harness/pi_harness/profiling/collect.py -> 仓库根目录。"""
    return Path(__file__).resolve().parents[3]


def sudo_prefix() -> list[str]:
    return ["sudo", "-n"] if os.geteuid() != 0 else []


def cfg_user_name() -> str:
    """容器内 USER/LOGNAME 的兜底值（getpass.getuser 需要它）。"""
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        return f"uid{os.getuid()}"


def pi_env_values(harness_dir: Path) -> dict[str, str]:
    """从 harness/scripts/pi_env.sh 取「无卡容器环境变量」唯一真源。

    返回 {"PI_DOCKER_ENVS": ..., "PI_MODELS_DIR": ..., "PI_CPUS": ...}；
    取不到时用内置兜底，保证脚本仍可用（但会在 cmd.sh 里体现差异）。
    """
    script = harness_dir / "scripts" / "pi_env.sh"
    out: dict[str, str] = {}
    if script.is_file():
        cmd = (f'source "{script}" && printf "%s\\n" '
               '"${PI_DOCKER_ENVS[@]}" "PI_MODELS_DIR=${PI_MODELS_DIR}" "PI_CPUS=${PI_CPUS}"')
        rc, stdout, _ = sh_run(["bash", "-c", cmd], timeout=30)
        if rc == 0:
            lines = [ln for ln in stdout.splitlines() if ln]
            envs: list[str] = []
            for ln in lines:
                if ln.startswith("PI_MODELS_DIR="):
                    out["PI_MODELS_DIR"] = ln.split("=", 1)[1]
                elif ln.startswith("PI_CPUS="):
                    out["PI_CPUS"] = ln.split("=", 1)[1]
                else:
                    envs.append(ln)
            if envs:
                out["PI_DOCKER_ENVS"] = envs
    return out


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Pass:
    """一次负载 pass（自己的容器 + 一组仪器）。"""

    def __init__(self, name: str, out_dir: Path) -> None:
        self.name = name
        self.dir = out_dir / "passes" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.info: dict[str, Any] = {"name": name, "ts_start": now_ts(),
                                     "instruments": {}}
        self.procs: list[tuple[str, subprocess.Popen, Path | None]] = []

    def start_container(self, cfg: "Config", gate: bool = True) -> tuple[str, int]:
        cidfile = self.dir / "container.cid"
        if cidfile.exists():
            cidfile.unlink()
        name = f"pi-prof-{cfg.tag}-{self.name}".replace("/", "-")[:60]
        inner = shlex.join(cfg.cmd)
        if gate:
            inner = f"kill -STOP $$; exec {inner}"
        cmd = [
            "docker", "run", "-d", "--cidfile", str(cidfile), "--name", name,
            "--network", "none", "--cpuset-cpus", cfg.cpus,
            "--user", f"{os.getuid()}:{os.getgid()}",
            *cfg.docker_envs,          # 唯一真源：harness/scripts/pi_env.sh
            "-v", f"{cfg.host_root}:/work",
            *(["-v", f"{cfg.models_dir}:/models:ro"] if cfg.models_dir else []),
            "-w", "/work",
            *cfg.docker_args,
            cfg.image, "bash", "-c", inner,
        ]
        rc, stdout, stderr = sh_run(cmd, timeout=120)
        self.info["docker_run"] = {"cmd": cmd, "rc": rc,
                                   "stdout": stdout.strip(), "stderr": stderr.strip()[:300]}
        if rc != 0:
            raise RuntimeError(f"docker run 失败: {stderr.strip()[:300]}")
        cid = stdout.strip().splitlines()[-1]
        rc, pid_s, _ = sh_run(["docker", "inspect", "-f", "{{.State.Pid}}", cid], timeout=30)
        if rc != 0 or not pid_s.strip().isdigit():
            raise RuntimeError("无法取到容器 init 的宿主 PID")
        pid = int(pid_s.strip())
        self.info.update({"container": cid, "container_name": name,
                          "host_pid": pid, "gate": gate})
        return cid, pid

    def start_instrument(self, name: str, cmd: Sequence[str],
                         ready_file: Path | None = None,
                         log: Path | None = None) -> None:
        log = log or (self.dir / f"{name}.log")
        fh = open(log, "wb")
        # 独立进程组：停机时整组发信号（sudo 与真正的 perf/kperfx 都在组里）
        proc = subprocess.Popen(list(cmd), stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.procs.append((name, proc, ready_file))
        self.info["instruments"][name] = {"cmd": list(cmd), "pid": proc.pid,
                                          "log": str(log),
                                          "ready_file": str(ready_file) if ready_file else None}

    def wait_ready(self, timeout: float = 30.0, settle_s: float = 0.5) -> dict[str, bool]:
        out: dict[str, bool] = {}
        deadline = time.time() + timeout
        for name, proc, ready in self.procs:
            if ready is None:
                time.sleep(settle_s)
                out[name] = proc.poll() is None
                continue
            ok = False
            while time.time() < deadline:
                if ready.exists():
                    ok = True
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
            out[name] = ok
        time.sleep(settle_s)
        self.info["ready"] = out
        return out

    def cont(self, pid: int) -> None:
        sh_run([*sudo_prefix(), "kill", "-CONT", str(pid)], timeout=10)

    def wait_target(self, cid: str, timeout: float) -> dict[str, Any]:
        t0 = time.time()
        rc, stdout, stderr = sh_run(["docker", "wait", cid], timeout=timeout)
        res = {"exit_code": int(stdout.strip()) if stdout.strip().isdigit() else None,
               "rc": rc, "wall_s": round(time.time() - t0, 3),
               "stderr": stderr.strip()[:200]}
        if rc != 0:
            res["stopped_reason"] = "timeout_or_error"
            sh_run(["docker", "kill", cid], timeout=30)
        else:
            res["stopped_reason"] = "exit"
        self.info["target"] = res
        return res

    def stop_instruments(self, grace: float = 20.0) -> None:
        for name, proc, _ in self.procs:
            if proc.poll() is not None:
                continue
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                continue
            if name.startswith("perf"):
                # perf 收到 SIGINT 才会 flush perf.data；sudo 与 perf 同组
                sh_run([*sudo_prefix(), "kill", "-INT", f"-{pgid}"], timeout=10)
            else:
                try:
                    os.killpg(pgid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            self.info["instruments"][name]["exit_code"] = proc.returncode

    def wait_instrument(self, name: str, timeout: float = 120.0) -> bool:
        """等某个仪器自己跑完（不主动发信号）。"""
        for iname, proc, _ in self.procs:
            if iname != name:
                continue
            try:
                proc.wait(timeout=max(1.0, timeout))
            except subprocess.TimeoutExpired:
                return False
            self.info["instruments"][name]["exit_code"] = proc.returncode
            return True
        return False

    def cleanup(self, cfg: "Config") -> None:
        cid = self.info.get("container")
        if cid:
            sh_run(["docker", "kill", cid], timeout=30)
            sh_run(["docker", "rm", "-f", cid], timeout=60)


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tag = args.tag
        self.cpus = args.cpus
        self.cpu_list = parse_cpu_list(args.cpus)
        self.image = args.image
        self.host_root = Path(args.host_root) if args.host_root else repo_root()
        self.harness_dir = self.host_root / "harness"
        pi_env = pi_env_values(self.harness_dir)
        self.docker_envs: list[str] = pi_env.get("PI_DOCKER_ENVS", FALLBACK_DOCKER_ENVS)
        self.pi_env_source = bool(pi_env)
        self.models_dir = (None if args.no_models_mount
                           else pi_env.get("PI_MODELS_DIR") or os.environ.get("PI_MODELS_DIR"))
        self.cmd: list[str] = list(args.cmd)
        self.docker_args: list[str] = args.docker_arg or []
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.out = Path(args.out) if args.out else (
            self.host_root / "data" / "harness" / "prof_runs" / f"{self.tag}_{ts}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.timeout = args.timeout
        self.perf_freq = args.perf_freq
        self.callgraph = args.callgraph
        self.pyspy = args.pyspy
        self.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        self.topdown_level = args.topdown_level
        self.topdown_mode = args.topdown_mode
        self.pmu_events = args.pmu_events
        self.warmup = args.warmup
        self.gate = not args.no_gate
        self.attach = args.attach
        self.duration = args.duration
        self.window = args.window
        self.annotate_top = args.annotate_top
        self.noise_seconds = args.noise_seconds

    def pmu_module_cmd(self, *rest: str) -> list[str]:
        # sudo 会清环境：把工具路径显式传进去（root 的 $HOME 不是我们的 $HOME）
        envs = [
            f"PYTHONPATH={self.harness_dir}",
            f"PI_LIBKPERFX={os.environ.get('PI_LIBKPERFX', DEFAULT_LIBKPERFX)}",
            f"PI_FLAMEGRAPH_RS={os.environ.get('PI_FLAMEGRAPH_RS', DEFAULT_FLAMEGRAPH_RS)}",
            f"PI_PYSPY={os.environ.get('PI_PYSPY', DEFAULT_PYSPY)}",
        ]
        return [*sudo_prefix(), "env", *envs, "python3", "-m", *rest]


def do_pmu_pass(cfg: Config, name: str, extra: Sequence[str], out_json: Path) -> Pass:
    """一个 PMU 计数 pass（kperfx，宿主 root）。"""
    p = Pass(name, cfg.out)
    if cfg.attach:
        cid, pid = None, cfg.attach
    else:
        cid, pid = p.start_container(cfg, cfg.gate)
        if not cfg.gate and cfg.warmup > 0:
            print(f"[collect] {name}: 先裸跑 {cfg.warmup:.0f}s 再上仪器", flush=True)
            time.sleep(cfg.warmup)
    ready = p.dir / "pmu.ready"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = cfg.pmu_module_cmd("pi_harness.profiling.pmu", "count", "--pid", str(pid))
    # 窗口模式：负载由 --steady-seconds 保证"够长"，我们只精确量一段 window；
    # 否则测到进程退出为止（短负载必需，但会把解释器收尾也算进来）。
    window = cfg.window or (cfg.duration if cfg.attach else None)
    if window:
        cmd += ["--duration", str(window)]
    else:
        cmd += ["--wait-exit", "--timeout", str(cfg.timeout)]
    cmd += ["--ready-file", str(ready), "--out", str(out_json), *extra]
    p.start_instrument("pmu", cmd, ready_file=ready)
    ready_map = p.wait_ready()
    if not ready_map.get("pmu"):
        rc = p.procs[0][1].poll()
        log = (p.dir / "pmu.log")
        tail = log.read_text(errors="replace")[-300:] if log.is_file() else ""
        p.info["pmu_failed"] = {"ready": False, "exit_code": rc, "log_tail": tail}
        print(f"[collect] 警告：{name} 的 PMU 会话没起来（exit={rc}），"
              f"细节见 {log}", flush=True)
        if cid:   # 进程已经死了就没必要再等
            p.wait_target(cid, 5)
            p.cleanup(cfg)
        return p
    if cid:
        p.cont(pid)
        if window:
            p.info["target"] = {"stopped_reason": "window_elapsed_kill",
                                "window_s": window}
            p.wait_instrument("pmu", timeout=window + 120)
            p.cleanup(cfg)          # 量完就收工，不必等负载自己跑完
        else:
            p.wait_target(cid, cfg.timeout)
    else:
        deadline = time.time() + (cfg.duration or 10) + 60
        while p.procs[0][1].poll() is None and time.time() < deadline:
            time.sleep(0.2)
    p.stop_instruments()
    p.cleanup(cfg)
    return p


def run_topdown(cfg: Config, pass_prefix: str = "topdown") -> dict[str, Any]:
    """920B topdown：l1 只跑第 1 组；full 跑 9 组（每组一次负载）。"""
    n_groups = 1 if cfg.topdown_level == "l1" else 9
    split_dir = cfg.out / "pmu" / "topdown_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    passes: list[dict[str, Any]] = []
    if cfg.topdown_mode == "mux":
        out_json = cfg.out / "pmu" / "topdown_mux.json"
        p = do_pmu_pass(cfg, f"{pass_prefix}_mux",
                        ["--preset", "920b_topdown_full_1", "--multiplex"], out_json)
        passes.append(p.info)
        return {"mode": "mux", "passes": passes, "split": None,
                "note": "mux 模式：多组同开，按 time_running/time_enabled 缩放，置信度可能 <1"}
    for g in range(1, n_groups + 1):
        out_json = split_dir / f"split_{g}.json"
        p = do_pmu_pass(cfg, f"{pass_prefix}_g{g}",
                        ["--preset", f"920b_topdown_full_{g}"], out_json)
        passes.append(p.info)
    # 多租户共享机上单个 pass 可能被抢核：对明显偏慢的组重采（最多 2 轮），
    # 让 9 组覆盖同一段"干净"时间，避免 topdown 分量被一次污染带偏。
    retries: list[str] = []
    for attempt in range(2):
        walls = [(p["name"], (p.get("target") or {}).get("wall_s")) for p in passes]
        walls = [(n, w) for n, w in walls if w]
        if len(walls) < 3:
            break
        lo = min(w for _, w in walls)
        hi = max(w for _, w in walls)
        if lo <= 0 or hi / lo <= 1.3:
            break
        worst = max(walls, key=lambda x: x[1])[0]
        try:
            g = int(worst.rsplit("_g", 1)[1])
        except ValueError:
            break
        tag = f"{pass_prefix}_g{g}r{attempt + 1}"
        print(f"[collect] {worst} 墙钟 {hi:.1f}s 偏慢（min {lo:.1f}s），重采 group {g}", flush=True)
        p = do_pmu_pass(cfg, tag, ["--preset", f"920b_topdown_full_{g}"],
                        split_dir / f"split_{g}.json")
        p.info["retry_of"] = worst
        passes.append(p.info)
        retries.append(tag)
    return {"mode": "split", "passes": passes, "split": str(split_dir),
            "retries": retries}


def postprocess(cfg: Config, pmu_json: Path | None, perf_pass: Pass | None,
                topdown_info: dict[str, Any] | None) -> dict[str, Any]:
    """汇总 PMU / topdown / 火焰图 / 热点 / 注释。"""
    summary: dict[str, Any] = {"artifacts": {}}
    if pmu_json and pmu_json.is_file():
        data = json.loads(pmu_json.read_text())
        summary["pmu"] = {
            "counters": data.get("counters"),
            "ipc": data.get("derived", {}).get("ipc"),
            "time_enabled": data.get("time_enabled"),
            "time_running": data.get("time_running"),
            "confidence": data.get("confidence"),
            "wall_s": data.get("wall_s"),
        }
        summary["artifacts"]["pmu"] = str(pmu_json)

    if topdown_info:
        summary["topdown_meta"] = {k: v for k, v in topdown_info.items() if k != "passes"}
        if topdown_info["mode"] == "split":
            td_out = cfg.out / "pmu" / "topdown.json"
            mod = cfg.pmu_module_cmd("pi_harness.profiling.topdown",
                                     "--split-dir", str(topdown_info["split"]),
                                     "--out", str(td_out))
            rc, out, err = sh_run(mod, timeout=300)
            if td_out.is_file():
                td = json.loads(td_out.read_text())
                summary["topdown"] = {k: td.get(k) for k in
                                      ("level1", "level2", "level3_mem", "ooo_stall",
                                       "ipc", "confidence", "notes")}
                summary["artifacts"]["topdown"] = str(td_out)
            else:
                summary.setdefault("caveats", []).append(
                    f"topdown 汇总失败: {err.strip()[:200]}")

    if perf_pass:
        postprocess_perf(cfg, perf_pass.dir, perf_pass.info.get("container"),
                         summary, cleanup=lambda: perf_pass.cleanup(cfg))
    return summary


def postprocess_perf(cfg: Config, pdir: Path, cid: str | None,
                     summary: dict[str, Any], cleanup=None) -> None:
    """perf 侧后处理：symfs 符号化 → 火焰图 → 热点 → 指令级注释。

    每一步都单独兜错（见 :func:`step`）：某一步失败不影响其它产物，
    失败原因与 traceback 记进 summary，避免整轮采集白跑。
    """
    if pdir is not None:
        data = pdir / "perf.data"
        symfs: Path | None = None
        if data.is_file():
            symfs = pdir / "symfs"
            syminfo = step(summary, "symfs", build_symfs, cfg, cid, data, symfs)
            if syminfo:
                summary["symfs"] = {k: v for k, v in syminfo.items()
                                    if k not in ("failed", "copy_errors")}
                if syminfo.get("failed"):
                    summary.setdefault("caveats", []).append(
                        f"有 {len(syminfo['failed'])} 个 DSO 抽取失败（符号可能不全）")
        sym_args = ["--symfs", str(symfs)] if symfs else []
        if data.is_file():
            hs = pdir / "hotspots.json"
            cmd = cfg.pmu_module_cmd("pi_harness.profiling.annotate", "hotspots",
                                     "-i", str(data), "-o", str(hs), "--top", "40",
                                     *sym_args)
            rc, out, err = sh_run(cmd, timeout=1800)
            if hs.is_file() and raw_address_ratio(json.loads(hs.read_text())) > 0.5:
                # 偶发：symfs 已就绪却仍解析出裸地址。重建 symfs（含镜像缓存）后重解析，
                # 再不行就如实写 caveat —— 此时 topdown/IPC 仍有效，但"热点函数"结论不可用。
                print("[collect] 警告：热点榜裸地址过多，重建 symfs 后重解析", flush=True)
                syminfo = step(summary, "symfs(rebuild)", build_symfs, cfg, cid, data, symfs)
                if syminfo:
                    summary["symfs"] = {k: v for k, v in syminfo.items()
                                        if k not in ("failed", "copy_errors")}
                sh_run(cmd, timeout=1800)
                if hs.is_file():
                    ratio = raw_address_ratio(json.loads(hs.read_text()))
                    if ratio > 0.5:
                        summary.setdefault("caveats", []).append(
                            f"热点榜仍有 {ratio:.0%} 是裸地址：符号解析未生效，"
                            f"函数级热点结论不可用（topdown/IPC 不受影响）")
            if hs.is_file():
                hotspots = json.loads(hs.read_text())
                summary["hotspots_top"] = hotspots["top"][:20]
                summary["artifacts"]["hotspots"] = str(hs)
                adir = pdir / "annotate"
                merged = adir / "annotate_top.json"
                cmd = cfg.pmu_module_cmd("pi_harness.profiling.annotate", "top",
                                         "-i", str(data), "--hotspots", str(hs),
                                         "--out-dir", str(adir), "--json", str(merged),
                                         "--top", str(cfg.annotate_top), *sym_args)
                sh_run(cmd, timeout=3600)
                if merged.is_file():
                    ann = json.loads(merged.read_text())
                    summary["annotate"] = ann
                    summary["artifacts"]["annotate_json"] = str(merged)
                else:
                    summary.setdefault("caveats", []).append("perf annotate 未产出结果")
            svg = pdir / "flamegraph.svg"
            folded = pdir / "flamegraph.folded"
            tool = "flamegraph-rs" if DEFAULT_FLAMEGRAPH_RS.is_file() else "builtin"
            cmd = cfg.pmu_module_cmd("pi_harness.profiling.flamegraph", "from-perfdata",
                                     str(data), "-o", str(svg), "--collapsed", str(folded),
                                     "--tool", tool, "--title", f"on-CPU {cfg.tag}",
                                     *sym_args)
            rc, out, err = sh_run(cmd, timeout=1800)
            if svg.is_file():
                summary["artifacts"]["flamegraph_svg"] = str(svg)
                summary["flamegraph_tool"] = tool
            else:
                summary.setdefault("caveats", []).append(
                    f"火焰图生成失败: {err.strip()[:200]}")
        if cfg.pyspy and (pdir / "pyspy.svg").is_file():
            summary["artifacts"]["pyspy_svg"] = str(pdir / "pyspy.svg")
        if cleanup:
            cleanup()


def write_cmd_sh(cfg: Config, extra: dict[str, Any]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# 本次采集的复现命令（由 pi_harness.profiling.collect 生成）",
        "set -euo pipefail",
        "",
        f"# tag={cfg.tag} cpus={cfg.cpus} image={cfg.image}",
        f"# caliber={CALIBER_VERSION} ts={now_ts()}",
        "REPO=${REPO:-%s}" % cfg.host_root,
        "",
        "# --- 采样前噪声门 ---",
        f"cd \"$REPO/harness\" && python3 -m pi_harness.profiling.hostnoise --cpus {cfg.cpus} --seconds {cfg.noise_seconds}",
        "",
        "# --- 负载（容器内，等价 prof_harness.sh 的入口）---",
        '# 环境变量唯一真源（USER/LOGNAME/TORCHINDUCTOR_CACHE_DIR/... 缺一不可）',
        'source "$REPO/harness/scripts/pi_env.sh"',
        "pi_env_selfcheck || exit 1",
        f"CIL='{shlex.join(cfg.cmd)}'",
        f"docker run --rm --network none --cpuset-cpus {cfg.cpus} \\",
        "  --user \"$(id -u):$(id -g)\" \"${PI_DOCKER_ENVS[@]}\" \\",
        "  -v \"$REPO:/work\" -v \"${PI_MODELS_DIR}:/models:ro\" -w /work \\",
        f"  {cfg.image} \\",
        "  bash -c \"$CIL\"",
    ]
    for k, v in (extra or {}).items():
        lines += ["", f"# --- {k} ---", *[f"# {ln}" for ln in str(v).splitlines()]]
    path = cfg.out / "cmd.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def step(summary: dict[str, Any], name: str, fn, *a, **kw):
    """跑一个后处理步骤；失败只记 caveat + traceback，不中断其它步骤。"""
    try:
        return fn(*a, **kw)
    except Exception as exc:
        summary.setdefault("caveats", []).append(f"{name} 失败: {exc!r}")
        summary.setdefault("errors", {})[name] = traceback.format_exc()[-1500:]
        print(f"[collect] {name} 失败: {exc!r}", flush=True)
        return None


RAW_ADDR_RE = re.compile(r"^\[[.k]\]\s*0x[0-9a-fA-F]+$")


def raw_address_ratio(hotspots: dict[str, Any], top: int = 20) -> float:
    """热点榜里「裸地址」（没解析出符号）的比例。"""
    rows = (hotspots.get("top") or [])[:top]
    if not rows:
        return 0.0
    n = sum(1 for r in rows if RAW_ADDR_RE.match(str(r.get("symbol", "")).strip()))
    return n / len(rows)


def _check_pass_consistency(pass_infos: list[dict[str, Any]],
                            ratio_limit: float = 1.3) -> list[str]:
    """检查各 pass 的墙钟时间是否一致。

    同一份固定工作量的负载，每个 pass 的墙钟时间应当接近；某个 pass 明显更慢
    通常意味着被别的租户抢核（a3-22 是多租户共享机）。这时 topdown 分量仍是
    组内比值，但绝对 IPC 会失真，需要在 caveat 里点出来。
    """
    walls: list[tuple[str, float]] = []
    for info in pass_infos:
        name = info.get("name")
        tgt = info.get("target") or {}
        if name and tgt.get("wall_s"):
            walls.append((name, float(tgt["wall_s"])))
    if len(walls) < 2:
        return []
    lo = min(w for _, w in walls)
    hi = max(w for _, w in walls)
    if lo > 0 and hi / lo > ratio_limit:
        worst = ", ".join(f"{n}={w:.1f}s" for n, w in walls if w > lo * ratio_limit)
        return [f"pass 墙钟时间不一致（min={lo:.1f}s max={hi:.1f}s，超 {ratio_limit}x: {worst}）"
                f"：疑似被其他租户抢核，绝对 IPC/耗时需谨慎"]
    return []


def write_summary_csv(path: Path, tag: str, summary: dict[str, Any]) -> None:
    rows: list[tuple[str, Any, str, str]] = []
    ts = now_ts()
    pmu = summary.get("pmu") or {}
    if pmu:
        for key in ("ipc", "time_enabled", "time_running", "confidence", "wall_s"):
            rows.append((key, pmu.get(key), "pmu/count.json",
                         "CPU busy cycles 口径，不是 wall time"))
        for name, val in (pmu.get("counters") or {}).items():
            rows.append((f"counter.{name}", val, "pmu/count.json", "libkperfx 区间增量"))
    td = summary.get("topdown") or {}
    level2 = td.get("level2") or {}
    for lvl, obj in (("level1", td.get("level1")),
                     ("level2.frontend", level2.get("frontend")),
                     ("level2.backend", level2.get("backend")),
                     ("level3_mem", td.get("level3_mem")),
                     ("ooo_stall", td.get("ooo_stall"))):
        for name, val in (obj or {}).items():
            rows.append((f"topdown.{lvl}.{name}", val, "pmu/topdown.json",
                         "百分点（占 dispatch slots = 6*cycles）"))
    for r in summary.get("hotspots_top") or []:
        rows.append((f"hotspot.{r['symbol']}", r["overhead_pct"],
                     "perf/hotspots.json", r.get("dso", "")))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "ts", "metric", "value", "source", "note"])
        for metric, value, source, note in rows:
            w.writerow([tag, ts, metric, value, source, note])


def post_only(cfg: Config) -> int:
    """对已有 run 目录重做后处理（不重跑负载）：perf 符号/热点/annotate/火焰图 + 汇总。

    用途：采集当时符号解析失败、或 postprocess 中途异常时，不用再花十几分钟重采。
    要求 run 目录里有 `pmu/count.json`、`pmu/topdown_split/`、`passes/perf/perf.data`
    与 `passes/perf/symfs`（symfs 缺失且容器已删除时会如实写 caveat）。
    """
    run = cfg.out
    if not run.is_dir():
        prof_err = f"run 目录不存在: {run}"
        print(f"[collect] {prof_err}", file=sys.stderr)
        return 2
    summary: dict[str, Any] = {"artifacts": {}}
    pmu_json = run / "pmu" / "count.json"
    if pmu_json.is_file():
        data = json.loads(pmu_json.read_text())
        summary["pmu"] = {
            "counters": data.get("counters"),
            "ipc": data.get("derived", {}).get("ipc"),
            "time_enabled": data.get("time_enabled"),
            "time_running": data.get("time_running"),
            "confidence": data.get("confidence"),
            "wall_s": data.get("wall_s"),
        }
        summary["artifacts"]["pmu"] = str(pmu_json)
    split_dir = run / "pmu" / "topdown_split"
    if split_dir.is_dir():
        td_out = run / "pmu" / "topdown.json"
        mod = cfg.pmu_module_cmd("pi_harness.profiling.topdown",
                                 "--split-dir", str(split_dir), "--out", str(td_out))
        sh_run(mod, timeout=600)
        if td_out.is_file():
            td = json.loads(td_out.read_text())
            summary["topdown"] = {k: td.get(k) for k in
                                  ("level1", "level2", "level3_mem", "ooo_stall",
                                   "ipc", "confidence", "notes")}
            summary["artifacts"]["topdown"] = str(td_out)
    pdir = run / "passes" / "perf"
    postprocess_perf(cfg, pdir, None, summary)

    summary.update({
        "caliber_version": CALIBER_VERSION,
        "tag": cfg.tag,
        "ts": now_ts(),
        "out": str(run),
        "cpus": cfg.cpus,
        "image": cfg.image,
        "cmd": cfg.cmd,
        "mode": "post_only",
    })
    write_json(run / "summary.json", summary)
    write_summary_csv(run / "summary.csv", cfg.tag, summary)
    print(json.dumps({"tag": cfg.tag, "out": str(run),
                      "ipc": (summary.get("topdown") or {}).get("ipc"),
                      "topdown_l1": (summary.get("topdown") or {}).get("level1"),
                      "hot_top3": [r["symbol"] for r in
                                   (summary.get("hotspots_top") or [])[:3]],
                      "annotated": (summary.get("annotate") or {}).get("annotated"),
                      "caveats": summary.get("caveats", [])}, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="无卡 profiling 采集总控（容器负载 + 宿主 root PMU/perf）")
    ap.add_argument("--tag", required=True,
                    help="产物 tag（--post-only 时可省略）")
    ap.add_argument("--cpus", default=DEFAULT_CPUS,
                    help="绑核（禁止 120-159；建议 200-239 / 360-399）")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--host-root", default=None, help="仓库根（默认从脚本位置推断）")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--stages", default="pmu,perf", help="pmu,topdown,perf 的任意组合")
    ap.add_argument("--pmu-events",
                    default="cpu_cycles,inst_retired,inst_spec,br_pred,br_mis_pred,l1d_cache_refill")
    ap.add_argument("--topdown-level", default="l1", choices=["l1", "full"])
    ap.add_argument("--topdown-mode", default="split", choices=["split", "mux"])
    ap.add_argument("--perf-freq", type=int, default=999)
    ap.add_argument("--callgraph", default="dwarf")
    ap.add_argument("--pyspy", action="store_true", help="额外跑 py-spy（Python 帧图）")
    ap.add_argument("--duration", type=float, default=None,
                    help="attach 模式 / py-spy 采样窗口秒数")
    ap.add_argument("--window", type=float, default=None,
                    help="门控模式下每个 pass 精确量多少秒（量完就杀容器，省时间）")
    ap.add_argument("--annotate-top", type=int, default=3,
                    help="perf annotate 覆盖热点榜前 N 个函数（默认 3）")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="先跑 N 秒再上仪器（默认 0 = STOP 门控）")
    ap.add_argument("--no-gate", action="store_true", help="不用 STOP 门控")
    ap.add_argument("--attach", type=int, default=None, help="挂到已有宿主 PID")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--noise-seconds", type=float, default=5.0)
    ap.add_argument("--docker-arg", action="append", default=None,
                    help="附加 docker 参数（可重复）")
    ap.add_argument("--no-models-mount", action="store_true",
                    help="不挂 $PI_MODELS_DIR 到 /models（默认与 pi-docker.sh 一致）")
    ap.add_argument("cmd", nargs="*", help="容器内命令（-- 之后）")
    ap.add_argument("--post-only", metavar="RUN_DIR", default=None,
                    help="只对已有 run 目录重做后处理（不重跑负载）")
    args = ap.parse_args()

    if args.post_only:
        args.out = args.post_only
        if not args.tag:
            args.tag = Path(args.post_only).name
        cfg = Config(args)
        return post_only(cfg)
    if not args.attach and not args.cmd:
        ap.error("要么给 --attach PID，要么在 -- 后面给容器内命令")
    cfg = Config(args)
    t_start = time.time()

    print(f"[collect] out={cfg.out} cpus={cfg.cpus} stages={cfg.stages}", flush=True)
    noise_pre = noise_gate(cfg.cpu_list, cfg.noise_seconds)
    write_json(cfg.out / "noise_pre.json", noise_pre)
    if not noise_pre["ok"]:
        print(f"[collect] 警告：采样前噪声超阈 {noise_pre['hot_cpus']} "
              f"load1/cpu={noise_pre['load1_per_cpu']}", flush=True)

    manifest = build_manifest(tag=cfg.tag, cmd=cfg.cmd or f"attach:{cfg.attach}",
                              cpus=cfg.cpu_list, image=cfg.image,
                              scripts=[str(Path(__file__).resolve())],
                              extra={"stages": cfg.stages, "cpus_arg": cfg.cpus,
                                     "docker_args": cfg.docker_args,
                                     "callgraph": cfg.callgraph,
                                     "perf_freq": cfg.perf_freq,
                                     "topdown_level": cfg.topdown_level,
                                     "topdown_mode": cfg.topdown_mode,
                                     "gate": cfg.gate, "out": str(cfg.out)})
    write_json(cfg.out / "manifest.json", manifest)

    perf_pass: Pass | None = None
    pmu_json: Path | None = None
    topdown_info: dict[str, Any] | None = None
    pass_infos: list[dict[str, Any]] = []

    if "pmu" in cfg.stages:
        pmu_json = cfg.out / "pmu" / "count.json"
        p = do_pmu_pass(cfg, "pmu", ["--events", cfg.pmu_events], pmu_json)
        pass_infos.append(p.info)
        print(f"[collect] pmu pass 完成: {pmu_json}", flush=True)
    if "topdown" in cfg.stages:
        topdown_info = run_topdown(cfg)
        pass_infos.extend(topdown_info.get("passes") or [])
        print(f"[collect] topdown pass 完成: {topdown_info.get('mode')}", flush=True)
    if "perf" in cfg.stages:
        perf_pass = do_perf_pass(cfg, "perf")
        pass_infos.append(perf_pass.info)
        print(f"[collect] perf pass 完成: {perf_pass.dir}", flush=True)

    pass_caveats = _check_pass_consistency(pass_infos)
    try:
        summary = postprocess(cfg, pmu_json, perf_pass, topdown_info)
    except Exception as exc:  # 后处理失败也要留下可诊断的 summary
        summary = {"artifacts": {}}
        summary.setdefault("caveats", []).append(f"后处理异常: {exc!r}")
        if perf_pass is not None:
            perf_pass.cleanup(cfg)
    if pass_caveats:
        summary.setdefault("caveats", []).extend(pass_caveats)
    summary.update({
        "caliber_version": CALIBER_VERSION,
        "tag": cfg.tag,
        "ts": now_ts(),
        "out": str(cfg.out),
        "cpus": cfg.cpus,
        "image": cfg.image,
        "cmd": cfg.cmd or f"attach:{cfg.attach}",
        "stages": cfg.stages,
        "passes": pass_infos,
        "wall_s_total": round(time.time() - t_start, 3),
    })
    noise_post = noise_gate(cfg.cpu_list, cfg.noise_seconds)
    write_json(cfg.out / "noise_post.json", noise_post)
    if not noise_post["ok"]:
        summary.setdefault("caveats", []).append(
            f"采样后噪声超阈 {noise_post['hot_cpus']} load1/cpu={noise_post['load1_per_cpu']}")
    write_json(cfg.out / "summary.json", summary)
    write_summary_csv(cfg.out / "summary.csv", cfg.tag, summary)
    write_cmd_sh(cfg, {"manifest": str(cfg.out / "manifest.json")})
    sh_run([*sudo_prefix(), "chown", "-R",
            f"{os.getuid()}:{os.getgid()}", str(cfg.out)], timeout=120)

    print(json.dumps({"tag": cfg.tag, "out": str(cfg.out),
                      "ipc": (summary.get("pmu") or {}).get("ipc"),
                      "topdown_l1": (summary.get("topdown") or {}).get("level1"),
                      "hot_top3": [r["symbol"] for r in (summary.get("hotspots_top") or [])[:3]],
                      "caveats": summary.get("caveats", [])},
                     ensure_ascii=False))
    return 0

def do_perf_pass(cfg: Config, name: str) -> Pass:
    """一个 perf 采样 pass（宿主 root，on-CPU cycles 采样 + 调用栈）。"""
    p = Pass(name, cfg.out)
    perf_dir = p.dir
    if cfg.attach:
        cid, pid = None, cfg.attach
    else:
        cid, pid = p.start_container(cfg, cfg.gate)
        if not cfg.gate and cfg.warmup > 0:
            print(f"[collect] {name}: 先裸跑 {cfg.warmup:.0f}s 再上仪器", flush=True)
            time.sleep(cfg.warmup)
    data = perf_dir / "perf.data"
    cg = [] if cfg.callgraph in ("none", "") else ["--call-graph", cfg.callgraph]
    cmd = [*sudo_prefix(), "perf", "record", "-o", str(data),
           "-F", str(cfg.perf_freq), "-g", *cg, "-p", str(pid)]
    p.start_instrument("perf", cmd, log=perf_dir / "perf.log")
    deadline = time.time() + 20
    while time.time() < deadline and not data.exists():
        time.sleep(0.05)
    time.sleep(0.6)
    if cfg.pyspy:
        svg = perf_dir / "pyspy.svg"
        pyspy = os.environ.get("PI_PYSPY", str(DEFAULT_PYSPY))
        py_dur = int(min(cfg.duration or 15, (cfg.window or 15) - 2) or 1)
        py_cmd = [*sudo_prefix(), pyspy, "record", "--pid", str(pid),
                  "--duration", str(max(1, py_dur)), "--rate", "199",
                  "--format", "flamegraph", "-o", str(svg), "--native"]
        proc = subprocess.Popen(py_cmd, stdout=open(perf_dir / "pyspy.log", "wb"),
                                stderr=subprocess.STDOUT, start_new_session=True)
        p.procs.append(("pyspy", proc, None))
        p.info["instruments"]["pyspy"] = {"cmd": py_cmd, "pid": proc.pid, "svg": str(svg)}
    if cid:
        p.cont(pid)
        if cfg.window:
            time.sleep(cfg.window)
            p.info["target"] = {"stopped_reason": "window_elapsed_kill",
                                "window_s": cfg.window}
        else:
            p.wait_target(cid, cfg.timeout)
    else:
        end = time.time() + (cfg.window or cfg.duration or 30)
        while pid_alive(pid) and time.time() < end:
            time.sleep(0.2)
    p.stop_instruments()
    # 容器先留着：后面要从容器里抽 DSO 做符号解析（symfs），postprocess 结束再删
    return p


#: 无论 mmap 解析是否命中，都保证从容器里抽出来的关键 DSO。
#: 少了它们热点会退化成裸地址（实测：libpython → 11.43% 的 _PyEval_EvalFrameDefault）。
CURATED_DSOS = (
    "/usr/local/python3.12.13/bin/python3.12",
    "/usr/local/python3.12.13/lib/libpython3.12.so.1.0",
    "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so",
    "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_python.so",
    "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch.so",
    "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libc10.so",
    "/usr/local/python3.12.13/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-aarch64-linux-gnu.so",
    "/usr/local/python3.12.13/lib/python3.12/site-packages/numpy/_multiarray_umath.cpython-312-aarch64-linux-gnu.so",
    "/usr/lib64/libc.so.6",
    "/usr/lib64/libm.so.6",
    "/usr/lib64/libstdc++.so.6",
    "/usr/lib64/libgcc_s.so.1",
    "/usr/lib64/libcrypto.so.3",
    "/usr/lib64/libz.so.1",
    "/usr/lib/ld-linux-aarch64.so.1",
)


def list_perf_dsos(cfg: Config, perf_data: Path) -> list[str]:
    """perf.data 里出现过的用户态 DSO 绝对路径。

    注意：`perf script -F dso` 在 6.6 上输出空行（不给出路径），必须从默认
    调用栈输出末尾的 `(路径)` 字段里取。
    """
    from .flamegraph import list_dso_paths

    try:
        return list_dso_paths(perf_data, sudo=True)
    except Exception as exc:  # pragma: no cover
        print(f"[collect] 列 DSO 失败: {exc}", flush=True)
        return []


def build_symfs(cfg: Config, cid: str | None, perf_data: Path,
                symfs_dir: Path) -> dict[str, Any]:
    """把容器里的 DSO 抽到 symfs 目录，让 perf 能解析 Python/torch 符号。

    宿主上没有 /usr/local/python3.12.13/lib/libpython3.12.so.1.0 这类文件，
    不抽的话热点会退化成裸地址。docker cp 支持对已退出（未删除）的容器操作。

    注意：即便宿主上存在同名文件（如 /usr/lib64/libc.so.6），也**必须**用容器里的
    那一份 —— 版本不同会解析出错误符号（实测 memcpy/hash 相位会退化成裸地址）。
    /work 下的是宿主挂载进来的文件，跳过。
    """
    info: dict[str, Any] = {"symfs": str(symfs_dir), "copied": [], "existing": [],
                            "failed": [], "cached": []}
    if not cid:
        return info
    dsos = list_perf_dsos(cfg, perf_data)
    info["n_dsos_from_perfdata"] = len(dsos)
    info["n_curated"] = len(CURATED_DSOS)
    seen = set(dsos)
    for d in CURATED_DSOS:
        if d not in seen:
            dsos.append(d)
            seen.add(d)
    info["n_dsos"] = len(dsos)
    symfs_dir.mkdir(parents=True, exist_ok=True)
    rc, img_id, _ = sh_run(["docker", "inspect", "-f", "{{.Image}}", cid], timeout=60)
    cache_dir = Path.home() / "tools" / "symfs_cache" / (
        (img_id.strip().replace("sha256:", "")[:16] or "unknown") if rc == 0 else "unknown")
    for d in dsos:
        if (d.startswith("/work/") or d == str(cfg.host_root)
                or d.startswith("/lib/modules/")):   # 内核模块在宿主上，容器里没有
            info["existing"].append(d)
            continue
        dest = symfs_dir / d.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / d.lstrip("/")
        if cached.is_file():
            try:
                shutil.copy2(cached, dest)
            except OSError as exc:
                info.setdefault("copy_errors", []).append(f"{d}: {exc}")
                continue
            info["cached"].append(d)
            continue
        rc, out, err = sh_run(["docker", "cp", f"{cid}:{d}", str(dest)], timeout=180)
        if rc == 0 and dest.exists():
            info["copied"].append(d)
            cached.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(dest, cached)
            except OSError as exc:
                info.setdefault("copy_errors", []).append(f"cache {d}: {exc}")
        else:
            info["failed"].append({"dso": d, "err": err.strip()[:120]})
    return info


def _path_present(path: str) -> bool:
    """宿主上有没有这个文件；EACCES 视为存在（例如 /lib/modules 下的内核模块）。"""
    try:
        return Path(path).exists()
    except OSError:
        return True


if __name__ == "__main__":
    raise SystemExit(main())
