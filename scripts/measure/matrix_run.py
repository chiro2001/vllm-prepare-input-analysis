#!/usr/bin/env python3
"""Orchestrate the chip3 measurement matrix: one service per config, many points.

The expensive part of a real-machine matrix is service startup (~155 s for
Qwen3.5-0.8B: weight load + cudagraph capture), so the matrix is organised as

    service config  ->  launch once (``--keep``)  ->  N workload points  ->  stop

while holding the chip3 lease for the whole batch.  Every point is executed by
``point_run.py``; this script only sequences services/points, keeps the lease
record, and aggregates ``data/measure/<run_id>/points.json`` as it goes (so a
crash still leaves a usable prefix).

Usage (on a3-22):
    python3 scripts/measure/matrix_run.py --plan scripts/measure/plans/matrix-main.json \
        --run-id m1 [--only-tags a,b] [--only-services s1] [--skip-lock] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

PROJECT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"
MEASURE = SCRIPTS / "measure"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def run(cmd: list[str], timeout: float | None = None, check: bool = False):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{cmd} -> rc={proc.returncode}\n{proc.stdout[-2000:]}\n"
                           f"{proc.stderr[-2000:]}")
    return proc


def sudo_docker(*args: str) -> subprocess.CompletedProcess:
    return run(["sudo", "-n", "docker", *args])


class Lease:
    """chip3 lease (mkdir-based) held for the whole matrix batch."""

    def __init__(self, owner: str, purpose: str, enabled: bool = True):
        self.owner = owner
        self.purpose = purpose
        self.enabled = enabled
        self.held = False

    def acquire(self, wait_s: int) -> bool:
        if not self.enabled:
            log("lock disabled (--skip-lock): assuming exclusive access")
            self.held = True
            return True
        proc = run([str(SCRIPTS / "chip3_lock.sh"), "acquire", "--owner", self.owner,
                    "--purpose", self.purpose, "--wait", str(wait_s)])
        if proc.returncode == 0:
            self.held = True
            log(f"chip3 lease acquired by {self.owner}")
            return True
        log(f"chip3 lease NOT acquired (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-300:]}")
        return False

    def release(self) -> None:
        if self.held and self.enabled:
            run([str(SCRIPTS / "chip3_lock.sh"), "release", "--owner", self.owner])
            log("chip3 lease released")
        self.held = False


def stop_container(name: str) -> None:
    inspect = sudo_docker("container", "inspect", name)
    if inspect.returncode != 0:
        return
    log(f"stopping container {name}")
    sudo_docker("stop", "--time", "30", name)
    sudo_docker("rm", "-f", name)


def repin_worker(container: str, chip: int, main: str,
                 acl: str | None = None, rel: str | None = None) -> None:
    """Re-pin the running worker to a *quieter* CPU subset than the default.

    ``hostnoise_gate.sh`` shows the 120-159 slice is shared with foreign
    tenants (their affinity masks include it), so when the gate recommends an
    idle subset we honour it for the measured window.  This changes only the
    worker's affinity, never host state, and the choice is recorded in the
    point records through the service spec.
    """
    cmd = [str(SCRIPTS / "pin_worker_cpus.sh"), "--chip", str(chip),
           "--container", container, "--main", main]
    if acl:
        cmd += ["--acl", acl]
    if rel:
        cmd += ["--rel", rel]
    proc = run(cmd)
    log(f"re-pin {container}: rc={proc.returncode} main={main}")


def launch_service(svc: dict, run_id: str, chip: int, extra: list[str]) -> dict:
    """Bring one service up with --keep; returns {run_dir, port, container}."""
    service_run_id = f"{run_id}-{svc['key']}"
    launcher = SCRIPTS / ("measure/launch_service_prof.sh"
                          if svc.get("launcher") == "prof"
                          else "launch_phase_service.sh")
    cmd = [str(launcher),
           "--run-id", service_run_id,
           "--model-key", svc.get("model_key", "qwen35-08b"),
           "--chip", str(chip),
           "--keep",
           "--requests", str(svc.get("warmup_requests", 1)),
           "--concurrency", "1",
           "--prompt-tokens", str(svc.get("warmup_prompt_tokens", 128)),
           "--max-tokens", str(svc.get("warmup_max_tokens", 8)),
           "--warmup-requests", str(svc.get("bringup_warmup_requests", 1)),
           "--pystack-interval-us", str(svc.get("pystack_interval_us", 0)),
           "--cudagraph-mode", svc.get("cudagraph_mode", "FULL_DECODE_ONLY"),
           "--max-model-len", str(svc.get("max_model_len", 2048)),
           "--max-num-seqs", str(svc.get("max_num_seqs", 32)),
           "--max-num-batched-tokens", str(svc.get("max_num_batched_tokens", 2048)),
           "--async-scheduling", svc.get("async_scheduling", "on"),
           ]
    if svc.get("launcher") == "prof":
        cmd += ["--pid-host", svc.get("pid_host", "off"),
                "--pyperf", svc.get("pyperf", "off"),
                "--container-prefix", svc.get("container_prefix", "pi-prof")]
    if svc.get("cpu_pinning"):
        cmd += ["--cpu-pinning", svc["cpu_pinning"]]
    if svc.get("phase_timing"):
        cmd += ["--phase-timing", svc["phase_timing"]]
    if svc.get("extra_serve_args"):
        cmd += ["--extra-serve-args", svc["extra_serve_args"]]
    cmd += extra
    log(f"launching service {svc['key']} (run_id={service_run_id})")
    proc = run(cmd, timeout=3600)
    tail = (proc.stdout or "")[-4000:] + (proc.stderr or "")[-4000:]
    if proc.returncode != 0:
        raise RuntimeError(f"service {svc['key']} launch failed rc={proc.returncode}\n{tail}")
    m = re.search(r"PHASE_RESULT=DONE run_id=(\S+) run_dir=(\S+) port=(\d+) chip=(\d+)",
                  proc.stdout or "")
    if not m:
        raise RuntimeError(f"cannot parse launcher output:\n{tail}")
    # The container name is chosen by the launcher (it differs between the
    # bring-up and profiling forks), so take it from the log line rather than
    # reconstructing it here.
    cname = re.search(r"creating (\S+) on chip", proc.stdout or "")
    return {"run_id": m.group(1), "run_dir": m.group(2), "port": int(m.group(3)),
            "container": (cname.group(1) if cname
                          else f"pi-phase-chip{chip}-{service_run_id}"),
            "launch_stdout_tail": (proc.stdout or "")[-2000:]}


def noise_snapshot(container: str, outdir_root: pathlib.Path,
                   tag: str, cpus: str, chip: int) -> dict:
    """Pre/post measurement host-noise snapshot (recorded per service, per phase)."""
    out = outdir_root / "noise"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.json"
    proc = run([str(SCRIPTS / "hostnoise_gate.sh"), "--cpus", cpus,
                "--json", str(path), "--sample-s", "2.0",
                "--chip3", f"npu{chip // 2}/chip{chip % 2}"], timeout=180)
    summary: dict = {"file": str(path), "rc": proc.returncode}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cpu = data.get("cpu") or {}
        summary.update({
            "cpu_mean_busy_pct": cpu.get("mean_busy_pct"),
            "cpu_max_busy_pct": cpu.get("max_busy_pct"),
            "hot_cpus": cpu.get("hot_cpus"),
            "verdict": data.get("verdict") or (data.get("cpu") or {}).get("verdict"),
        })
    except (OSError, json.JSONDecodeError):
        pass
    log(f"noise[{tag}] -> {json.dumps(summary)}")
    return summary


def run_point(point: dict, service: dict, run_id: str, chip: int,
              outdir_root: pathlib.Path, extra: list[str]) -> dict:
    tag = point["tag"]
    outdir = outdir_root / "points" / tag
    cmd = [sys.executable, str(MEASURE / "point_run.py"),
           "--base-url", f"http://127.0.0.1:{service['port']}",
           "--model", point.get("model", service.get("served_name", "qwen35-08b")),
           "--container", service["container"],
           "--service-run-dir", service["run_dir"],
           "--chip", str(chip),
           "--run-id", run_id,
           "--tag", tag,
           "--outdir", str(outdir),
           "--group", point.get("group", ""),
           "--note", point.get("note", ""),
           "--requests", str(point.get("requests", 1)),
           "--concurrency", str(point.get("concurrency", 1)),
           "--prompt-tokens", str(point.get("prompt_tokens", 128)),
           "--max-tokens", str(point.get("max_tokens", 64)),
           "--seed", str(point.get("seed", 1024)),
           "--rounds", str(point.get("rounds", 1)),
           "--phase", point.get("phase", "on"),
           "--client-cpus", point.get("client_cpus", svc_client_cpus(chip))]
    for flag, key in (("--perf-s", "perf_s"), ("--perf-delay", "perf_delay"),
                      ("--pmu-window", "pmu_window"), ("--pmu-delay", "pmu_delay")):
        if point.get(key):
            cmd += [flag, str(point[key])]
    if point.get("perf_freq"):
        cmd += ["--perf-freq", str(point["perf_freq"])]
    if point.get("perf_callgraph"):
        cmd += ["--perf-callgraph", point["perf_callgraph"]]
    if point.get("pmu_groups"):
        cmd += ["--pmu-groups", point["pmu_groups"]]
    cmd += extra
    log(f"point {tag} (group {point.get('group','?')}) -> {outdir}")
    proc = run(cmd, timeout=point.get("timeout_s", 7200))
    record: dict = {"tag": tag, "rc": proc.returncode,
                    "stdout_tail": (proc.stdout or "")[-3000:],
                    "stderr_tail": (proc.stderr or "")[-3000:]}
    point_json = outdir / "point.json"
    if point_json.is_file():
        try:
            record["point"] = json.loads(point_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["parse_error"] = str(exc)
    else:
        record["missing"] = str(point_json)
    summary = parse_point_stdout(proc.stdout or "")
    if summary:
        record["summary"] = summary
        log(f"  -> {json.dumps(summary)}")
    return record


def parse_point_stdout(text: str) -> dict:
    m = re.search(r"\{\s*\"tag\"[\s\S]*?\n\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def svc_client_cpus(chip: int) -> str:
    first = 40 * chip
    return f"{first},{first + 1}"


def script_hashes() -> dict:
    """sha256 of every script that can influence a run (COORDINATION.md §6)."""
    import hashlib

    out: dict = {}
    candidates = [
        SCRIPTS / "launch_phase_service.sh",
        SCRIPTS / "container_launch.sh",
        SCRIPTS / "pin_worker_cpus.sh",
        SCRIPTS / "hostnoise_gate.sh",
        SCRIPTS / "chip3_lock.sh",
        MEASURE / "launch_service_prof.sh",
        MEASURE / "point_run.py",
        MEASURE / "ecmap.py",
        MEASURE / "analyze_lite.py",
        MEASURE / "perf_capture.sh",
        MEASURE / "pmu_tid_sweep.py",
    ]
    for path in candidates:
        if not path.is_file():
            out[path.name] = None
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        out[path.name] = digest.hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--chip", type=int, default=3)
    ap.add_argument("--data-root", default=None,
                    help="default: data/measure/<run-id>")
    ap.add_argument("--lock-wait", type=int, default=0,
                    help="seconds to wait for the chip3 lease (0 = fail fast)")
    ap.add_argument("--skip-lock", action="store_true")
    ap.add_argument("--only-services", default="")
    ap.add_argument("--only-tags", default="")
    ap.add_argument("--skip-tags", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    data_root = pathlib.Path(args.data_root) if args.data_root else (
        PROJECT / "data" / "measure" / args.run_id)
    data_root.mkdir(parents=True, exist_ok=True)
    points_path = data_root / "points.json"
    records: list[dict] = []
    if points_path.is_file():
        try:
            records = json.loads(points_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            records = []

    only_services = {s for s in args.only_services.split(",") if s}
    only_tags = {s for s in args.only_tags.split(",") if s}
    skip_tags = {s for s in args.skip_tags.split(",") if s}

    services = {svc["key"]: svc for svc in plan["services"]}
    points = plan["points"]
    if only_tags:
        points = [p for p in points if p["tag"] in only_tags]
    if skip_tags:
        points = [p for p in points if p["tag"] not in skip_tags]
    ordered_services = [s for s in plan["services"]
                        if any(p["service"] == s["key"] for p in points)]
    if only_services:
        ordered_services = [s for s in ordered_services if s["key"] in only_services]
        points = [p for p in points if p["service"] in only_services]

    log(f"plan={args.plan} run_id={args.run_id} services={[s['key'] for s in ordered_services]} "
        f"points={len(points)}")
    if args.dry_run:
        for svc in ordered_services:
            n = len([p for p in points if p["service"] == svc["key"]])
            log(f"  service {svc['key']}: {n} point(s) "
                f"{[p['tag'] for p in points if p['service'] == svc['key']]}")
        return 0

    lease = Lease("measurement_profiling",
                  f"matrix run {args.run_id} ({len(points)} points)",
                  enabled=not args.skip_lock)
    if not lease.acquire(args.lock_wait):
        return 75

    banner = {
        "run_id": args.run_id, "plan": str(args.plan), "started_utc": utc(),
        "chip": args.chip, "services": {}, "points_planned": len(points),
        "scripts_sha256": script_hashes(),
    }
    current_service: dict | None = None
    try:
        for svc in ordered_services:
            svc_points = [p for p in points if p["service"] == svc["key"]]
            if not svc_points:
                continue
            service = launch_service(svc, args.run_id, args.chip, [])
            banner["services"][svc["key"]] = {
                "run_id": service["run_id"], "run_dir": service["run_dir"],
                "port": service["port"], "container": service["container"],
                "spec": svc,
            }
            (data_root / "banner.json").write_text(
                json.dumps(banner, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            current_service = service
            noise = {"before": noise_snapshot(
                service["container"], data_root, f"{svc['key']}-before",
                f"{40 * args.chip}-{40 * args.chip + 39}", args.chip)}
            if svc.get("repin_main"):
                repin_worker(service["container"], args.chip, svc["repin_main"],
                             svc.get("repin_acl"), svc.get("repin_rel"))
            for point in svc_points:
                record = run_point(point, service, args.run_id, args.chip,
                                   data_root, [])
                records.append(record)
                points_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
            noise["after"] = noise_snapshot(
                service["container"], data_root, f"{svc['key']}-after",
                f"{40 * args.chip}-{40 * args.chip + 39}", args.chip)
            banner["services"][svc["key"]]["noise"] = noise
            (data_root / "banner.json").write_text(
                json.dumps(banner, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stop_container(service["container"])
            current_service = None
    finally:
        if current_service is not None:
            stop_container(current_service["container"])
        banner["finished_utc"] = utc()
        (data_root / "banner.json").write_text(
            json.dumps(banner, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lease.release()

    ok = sum(1 for r in records if r.get("rc") == 0)
    log(f"done: {ok}/{len(records)} points rc=0 -> {points_path}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
