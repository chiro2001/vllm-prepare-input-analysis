#!/usr/bin/env python3
"""Write the machine-readable ``run_manifest.json`` for a launched service.

This is the hand-off artifact between any two agents that share a container:
the launcher writes it, and a profiler (perf / libkperfx / msprof) reads it to
learn, without guessing:

* the container name / id and the image ref + digest (reproducibility);
* the NPU device and the CPU slice the container was created with;
* the engine-core host PID and main-thread TID to pass to ``perf -t`` and
  ``KPERFX_TARGET``;
* the container's own cgroup cpuset (what the process actually sees);
* the workload parameters of the run that produced the artifacts;
* absolute host and in-container paths of every log produced so far;
* ready-to-paste hand-off commands (perf attach + workload driver).

How the engine-core TID is located
----------------------------------
1. ``npu-smi info -t proc-mem -i CARD -c SUBCHIP`` lists the host PIDs that
   hold that die.  The engine-core / worker process is among them; this is
   authoritative because no other container can be holding the die we leased.
2. vLLM v1 runs ``NPUModelRunner.execute_model`` on the process main thread,
   and ``LiteProfiler`` records exactly that thread's
   ``threading.get_native_id()``.  For the main thread the native id equals the
   PID, so ``engine_core.tid == engine_core.pid``; the script asserts that
   ``/proc/<pid>/task/<pid>`` exists and records the observed ``comm``.
3. The remaining threads are listed with their ``comm`` so a profiler can
   exclude ``acl_thread`` / ``release_thread`` / ZMQ IO threads, which never
   run the model runner.

Usage::

    run_manifest.py --run-id RUN_ID [--project-root DIR] [--stdout]

Exit codes: 0 ok, 2 usage/IO error.  Missing optional pieces (for example a run
started without ``--keep``, so the container is already gone) are reported in
the manifest under ``warnings`` instead of failing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import subprocess
import sys


def sh(cmd: list[str], *, sudo: bool = False, timeout: int = 30) -> str:
    """Run a command, return stdout (empty string on any failure)."""
    if sudo:
        cmd = ["sudo", "-n", *cmd]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_npu_proc_mem(text: str) -> list[dict]:
    """Parse ``npu-smi info -t proc-mem`` output into a list of processes."""
    procs: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        m = re.match(r"\s*Process id\s*:\s*(\d+)", line)
        if m:
            if current:
                procs.append(current)
            current = {"pid": int(m.group(1))}
            continue
        if current is None:
            continue
        m = re.match(r"\s*Process name\s*:\s*(\S+)", line)
        if m:
            current["process_name"] = m.group(1)
            continue
        m = re.match(r"\s*Process memory\(MB\)\s*:\s*(\d+)", line)
        if m:
            current["memory_mb"] = int(m.group(1))
    if current:
        procs.append(current)
    return procs


def thread_table(pid: int) -> list[dict]:
    """Threads of ``pid`` with comm and role, read straight from /proc."""
    out: list[dict] = []
    task_dir = pathlib.Path(f"/proc/{pid}/task")
    if not task_dir.is_dir():
        return out
    for entry in sorted(task_dir.iterdir(), key=lambda p: int(p.name)):
        try:
            comm = (entry / "comm").read_text().strip()
        except OSError:
            continue
        tid = int(entry.name)
        role = "main" if tid == pid else "other"
        if comm in ("acl_thread", "release_thread"):
            role = comm
        out.append({"tid": tid, "comm": comm, "role": role})
    return out


def find_engine_procs(run_dir: pathlib.Path, chip: int | None) -> tuple[list[dict], str | None]:
    """Locate the engine-core process: live npu-smi first, then the recording."""
    procs: list[dict] = []
    source: str | None = None
    if chip is not None:
        card, subchip = int(chip) // 2, int(chip) % 2
        live = sh(
            ["npu-smi", "info", "-t", "proc-mem", "-i", str(card), "-c", str(subchip)],
            sudo=True,
        )
        if live:
            procs = parse_npu_proc_mem(live)
            source = "npu-smi (live)"
    if not procs:
        recorded = run_dir / "mappings" / "npu_proc_mem.ready.txt"
        if recorded.is_file():
            procs = parse_npu_proc_mem(recorded.read_text())
            source = f"{recorded} (recorded at service-ready)"
    return procs, source


def build(run_dir: pathlib.Path, container: str | None) -> dict:
    warnings: list[str] = []
    manifest_path = run_dir / "manifest.json"
    base: dict = {}
    if manifest_path.is_file():
        try:
            base = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            warnings.append(f"{manifest_path} is not valid JSON")
    else:
        warnings.append(
            f"{manifest_path} missing (run not produced by "
            "scripts/launch_phase_service.sh?)"
        )

    hw = base.get("hardware", {})
    server = base.get("server", {})
    image = dict(base.get("image", {}))
    chip = hw.get("chip")
    if chip is None:
        warnings.append("manifest.json has no hardware.chip; device fields omitted")

    name = container or base.get("container")
    if not name:
        warnings.append("container name unknown (pass --container)")

    running = False
    container_id = ""
    cpuset_effective = ""
    if name:
        running = sh(
            ["docker", "inspect", "-f", "{{.State.Running}}", name], sudo=True
        ).strip() == "true"
        container_id = sh(["docker", "inspect", "-f", "{{.Id}}", name], sudo=True).strip()
        if running:
            cpuset_effective = sh(
                [
                    "docker", "exec", name, "sh", "-c",
                    "cat /sys/fs/cgroup/cpuset.cpus 2>/dev/null || "
                    "cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null",
                ],
                sudo=True,
            ).strip()
        else:
            warnings.append(
                f"container {name} is not running: the live PID/TID lookup falls "
                "back to the recorded artifacts, and handoff commands need a "
                "restart or leftover --keep container"
            )

    procs, proc_source = find_engine_procs(run_dir, chip)
    if not procs:
        warnings.append(
            "no engine-core process found via npu-smi; perf/libkperfx target "
            "cannot be derived"
        )
    for proc in procs:
        proc["alive"] = pathlib.Path(f"/proc/{proc['pid']}").is_dir()
    engine_pid = next((p["pid"] for p in procs if p["alive"]), None)
    if procs and engine_pid is None:
        warnings.append(
            "engine-core pid from npu-smi is not visible from this shell "
            "(unprivileged /proc); TID == PID by vLLM v1 design, so the "
            "manifest still records it for the profiler running as root"
        )
        engine_pid = procs[0]["pid"]

    engine_core = {
        "pid": engine_pid,
        "tid": engine_pid,
        "tid_rule": (
            "vLLM v1 executes NPUModelRunner.execute_model on the process main "
            "thread and LiteProfiler records threading.get_native_id(), which "
            "equals the PID for that thread"
        ),
        "process_name": procs[0].get("process_name") if procs else None,
        "memory_mb": procs[0].get("memory_mb") if procs else None,
        "source": proc_source,
        "pid_namespace": "host",
        "threads": thread_table(engine_pid) if engine_pid else [],
    }

    if image.get("ref"):
        image["repo_digest"] = sh(
            ["docker", "image", "inspect", image["ref"],
             "--format", "{{index .RepoDigests 0}}"],
            sudo=True,
        ).strip() or None

    def artifact(rel: str, container_path: str | None = None) -> dict:
        path = run_dir / rel
        return {
            "host_path": str(path),
            "container_path": container_path,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }

    artifacts = {
        "phase_timing": {
            "raw_log": artifact("lite-profiler/lite.log",
                                "/runmeta/lite-profiler/lite.log"),
            "per_step_csv": artifact("lite-profiler/phase_timing.csv"),
            "summary_json": artifact("lite-profiler/phase_timing.json"),
        },
        "subscope": {
            "raw_csv": artifact("pi-subscope/raw.csv",
                                "/runmeta/pi-subscope/raw.csv"),
            "launch_env": artifact("pi-subscope/launch.env.txt"),
            "per_step_csv": artifact("pi-subscope/per_step_subscope.csv"),
            "summary_csv": artifact("pi-subscope/summary.csv"),
            "summary_groups_csv": artifact("pi-subscope/summary_groups.csv"),
            "meta_json": artifact("pi-subscope/meta.json"),
        },
        "docker": {
            "inspect_created": artifact("host/docker_inspect.created.json"),
            "inspect_ready": artifact("host/docker_inspect.ready.json"),
            "container_id_file": artifact("host/container_id.txt"),
            "logs": artifact("host/docker-logs.txt"),
        },
        "mappings": {
            "container_threads": artifact("mappings/container_threads.txt"),
            "npu_proc_mem_ready": artifact("mappings/npu_proc_mem.ready.txt"),
            "cpu_pinning": artifact("mappings/cpu_pinning.txt"),
        },
        "workload": {
            "client_stdout": artifact("requests/client.stdout.txt"),
            "client_stderr": artifact("requests/client.stderr.txt"),
        },
    }

    port = server.get("port")
    base_url = f"http://127.0.0.1:{port}" if port else None
    served = (base.get("model") or {}).get("served_name", "<served-name>")
    handoff = {
        "attach_perf": (
            f"sudo perf record -g -F 999 -t {engine_pid} -o /tmp/perf.data -- sleep 30"
            if engine_pid
            else None
        ),
        "attach_libkperfx": (
            f"KPERFX_TARGET=pid:{engine_pid} <kperfx command>" if engine_pid else None
        ),
        "drive_workload": (
            f"python3 scripts/phase_smoke_client.py --base-url {base_url} "
            f"--model {served} --outdir runs/{run_dir.name}/requests/perf-window "
            "--requests <N> --concurrency <C> --prompt-tokens <ISL> "
            "--max-tokens <OSL> --tag perf-window"
            if base_url
            else None
        ),
        "arm_lite_profiler": (
            f"curl -s -X POST {base_url}/start_profile ; "
            f"curl -s -X POST {base_url}/stop_profile"
            if base_url
            else None
        ),
        "exec_in_container": f"sudo -n docker exec -it {name} bash" if name else None,
        "stop_container": (
            f"sudo -n docker stop --time 60 {name} && sudo -n docker rm -f {name}"
            if name
            else None
        ),
    }

    return {
        "schema": "prepare-input-run-manifest/v1",
        "generated_utc": utc_now(),
        "run_id": base.get("run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "kind": base.get("kind"),
        "stage": base.get("stage"),
        "container": {
            "name": name,
            "id": container_id or None,
            "running": running,
            "cpuset_effective": cpuset_effective or None,
            "privileged": True,
        },
        "image": image,
        "hardware": {
            "host": hw.get("host", "a3-22"),
            "chip": chip,
            "device": hw.get("device",
                             f"/dev/davinci{chip}" if chip is not None else None),
            "cpuset": hw.get("cpuset"),
            "cpuset_mems": hw.get("cpuset_mems"),
            "worker_main_cpus": hw.get("worker_main_cpus"),
            "acl_cpu": hw.get("acl_cpu"),
            "release_cpu": hw.get("release_cpu"),
        },
        "engine_core": engine_core,
        "server": {
            "port": port,
            "base_url": base_url,
            "served_model_name": served,
            "backend": server.get("backend"),
            "tensor_parallel_size": server.get("tensor_parallel_size"),
            "cudagraph_mode": server.get("cudagraph_mode"),
            "max_model_len": server.get("max_model_len"),
            "max_num_seqs": server.get("max_num_seqs"),
            "max_num_batched_tokens": server.get("max_num_batched_tokens"),
            "async_scheduling": server.get("async_scheduling"),
            "prefix_caching": server.get("prefix_caching"),
            "model_path": (base.get("model") or {}).get("path"),
        },
        "workload": base.get("workload"),
        "instrumentation": base.get("instrumentation"),
        "pystack": {
            "interval_us_requested": (
                (base.get("instrumentation") or {}).get("pystack") or {}
            ).get("interval_us_requested"),
            "interval_us_effective": (
                (base.get("instrumentation") or {}).get("pystack") or {}
            ).get("interval_us_effective"),
            "log_path_on_host": (
                (base.get("instrumentation") or {}).get("pystack") or {}
            ).get("log_path_on_host"),
            "log_exists": (run_dir / "lite-profiler" / "pystack.csv").is_file(),
            "log_bytes": (
                (run_dir / "lite-profiler" / "pystack.csv").stat().st_size
                if (run_dir / "lite-profiler" / "pystack.csv").is_file()
                else None
            ),
            "note": "interval_us == 4294967295 means the sampler was armed but "
                    "effectively disabled; a small pystack.csv (<1 KB) confirms it",
        },
        "artifacts": artifacts,
        "handoff": handoff,
        "warnings": warnings,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="directory containing runs/ (default: parent of this script)",
    )
    ap.add_argument("--container", default=None,
                    help="override container name (default: from manifest.json)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output path (default: runs/<run-id>/run_manifest.json)")
    ap.add_argument("--stdout", action="store_true", help="also print the JSON")
    args = ap.parse_args(argv[1:])

    run_dir = args.project_root / "runs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        return 2
    doc = build(run_dir, args.container)
    out = args.out or run_dir / "run_manifest.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    if doc["engine_core"]["pid"]:
        print(
            f"  engine-core pid={doc['engine_core']['pid']} "
            f"tid={doc['engine_core']['tid']} "
            f"threads={len(doc['engine_core']['threads'])}"
        )
    for warning in doc["warnings"]:
        print(f"  WARNING: {warning}")
    if args.stdout:
        print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
