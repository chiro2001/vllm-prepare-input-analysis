#!/usr/bin/env python3
"""Resolve the engine-core (EC) host PID/TID of a live pi-phase service on a3-22.

Why this exists
---------------
Everything on the CPU side of this project is measured *per engine-core thread*:

* ``perf record -t <TID>`` must name a **host** TID -- the container has its own
  PID namespace, so the TID that appears in the LiteProfiler log (e.g. 131) is
  *not* the host TID;
* ``libkperfx`` with ``KPERFX_TARGET=tid`` likewise takes a host TID;
* ``/proc/<pid>/task/<tid>/{stat,schedstat}`` (used for on-CPU time) is only
  addressable by host TID.

The authoritative mapping is:

    npu-smi info -t proc-mem -i <card> -c <subchip>   ->  host PID that holds the die

For the ``uni`` backend (what this project uses) the EngineCore *is* that
process and runs its engine loop on its **main thread**, so
``host_tid == host_pid``.  We still verify that a thread with that TID exists
and that the LiteProfiler log (when present) wrote ``prepare input`` rows for
the matching container-namespace TID; ``--verify-run-dir`` does that check.

Usage (on a3-22):
    python3 scripts/measure/ecmap.py --chip 3 [--container NAME] [--json OUT]
        [--verify-run-dir RUN_DIR] [--threads]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys


def sh(cmd: list[str], timeout: float = 20.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        return 127, "", str(exc)


def sudo(cmd: list[str], timeout: float = 20.0) -> tuple[int, str, str]:
    if os.geteuid() == 0:
        return sh(cmd, timeout)
    return sh(["sudo", "-n"] + cmd, timeout)


def read_text(path: str, timeout: float = 10.0) -> str | None:
    """Read a /proc file, falling back to ``sudo -n cat`` on EACCES."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except PermissionError:
        rc, out, _ = sudo(["cat", path], timeout)
        return out if rc == 0 else None


def npu_host_pid(chip: int) -> int | None:
    card, subchip = chip // 2, chip % 2
    rc, out, err = sudo(["npu-smi", "info", "-t", "proc-mem", "-i", str(card),
                         "-c", str(subchip)])
    if rc != 0:
        return None
    m = re.search(r"Process id:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def container_pid(name: str) -> int | None:
    rc, out, _ = sudo(["docker", "inspect", "-f", "{{.State.Pid}}", name])
    if rc != 0:
        return None
    try:
        pid = int(out.strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def ns_pid_of_thread(host_pid: int, host_tid: int) -> int | None:
    """Container-namespace TID of a host thread (NSpid innermost entry)."""
    text = read_text(f"/proc/{host_pid}/task/{host_tid}/status")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("NSpid:"):
            parts = line.split()[1:]
            return int(parts[-1]) if parts else None
    return None


def list_threads(host_pid: int) -> list[dict]:
    out: list[dict] = []
    taskdir = pathlib.Path(f"/proc/{host_pid}/task")
    entries = sorted(taskdir.glob("*"), key=lambda p: int(p.name)) if taskdir.is_dir() else []
    if not entries:
        rc, text, _ = sudo(["sh", "-c", f"ls /proc/{host_pid}/task"])
        if rc == 0:
            for tid in text.split():
                comm = read_text(f"/proc/{host_pid}/task/{tid}/comm")
                out.append({"host_tid": int(tid), "comm": (comm or "").strip(),
                            "ns_tid": ns_pid_of_thread(host_pid, int(tid))})
        return out
    for entry in entries:
        try:
            tid = int(entry.name)
        except ValueError:
            continue
        comm = read_text(f"{entry}/comm") or ""
        out.append({"host_tid": tid, "comm": comm.strip(),
                    "ns_tid": ns_pid_of_thread(host_pid, tid)})
    return out


def cpu_affinity(host_tid: int) -> str | None:
    rc, out, _ = sh(["taskset", "-pc", str(host_tid)])
    if rc == 0 and ":" in out:
        return out.split(":", 1)[1].strip()
    rc, out, _ = sudo(["taskset", "-pc", str(host_tid)])
    return out.split(":", 1)[1].strip() if rc == 0 and ":" in out else None


def lite_log_evidence(run_dir: pathlib.Path, ns_tid: int | None) -> dict:
    """Cross-check: which TIDs wrote ``prepare input`` rows, and with which pid."""
    log = run_dir / "lite-profiler" / "lite.log"
    evidence: dict = {"lite_log": str(log), "exists": log.is_file()}
    if not log.is_file():
        return evidence
    per_tid: dict[str, int] = {}
    pid_of_tid: dict[str, str] = {}
    try:
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("prepare input|"):
                    continue
                parts = line.rstrip("\n").split("|")
                if len(parts) != 5:
                    continue
                per_tid[parts[3]] = per_tid.get(parts[3], 0) + 1
                pid_of_tid[parts[3]] = parts[4]
    except OSError as exc:
        evidence["error"] = str(exc)
        return evidence
    evidence["prepare_input_rows_per_ns_tid"] = per_tid
    evidence["ns_pid_of_ns_tid"] = pid_of_tid
    if ns_tid is not None:
        evidence["matches_engine_tid"] = str(ns_tid) in per_tid
    return evidence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", type=int, default=3)
    ap.add_argument("--container", default=None,
                    help="container name (default: label pick pi.chip=<chip>)")
    ap.add_argument("--json", default=None, help="write the mapping to this file")
    ap.add_argument("--threads", action="store_true", help="list all threads")
    ap.add_argument("--verify-run-dir", default=None,
                    help="service run dir; cross-checks the LiteProfiler log")
    args = ap.parse_args(argv)

    doc: dict = {"chip": args.chip, "host": os.uname().nodename, "errors": []}

    if args.container:
        doc["container"] = args.container
    else:
        rc, out, _ = sudo(["docker", "ps", "--filter",
                           f"label=pi.chip={args.chip}", "--format", "{{.Names}}"])
        names = [n for n in out.split() if n]
        doc["containers_with_label"] = names
        doc["container"] = names[0] if len(names) == 1 else None
        if len(names) != 1:
            doc["errors"].append(f"expected exactly 1 container labelled pi.chip={args.chip}, "
                                 f"found {names}")

    pid = npu_host_pid(args.chip)
    doc["npu_host_pid"] = pid
    if pid is None:
        doc["errors"].append("npu-smi reports no process on this die")
    elif not pathlib.Path(f"/proc/{pid}").is_dir():
        doc["errors"].append(f"/proc/{pid} does not exist")

    if doc.get("container"):
        cpid = container_pid(doc["container"])
        doc["container_main_host_pid"] = cpid
        if pid is not None and cpid is not None and pid != cpid:
            doc["note_process_model"] = (
                "npu pid != container main pid: the engine core runs in a child "
                "process (mp backend?) or the container has several processes"
            )
    if pid is not None and pathlib.Path(f"/proc/{pid}").is_dir():
        doc["ec_host_pid"] = pid
        doc["ec_host_tid"] = pid  # engine loop runs on the process main thread (uni backend)
        doc["ec_comm"] = (read_text(f"/proc/{pid}/task/{pid}/comm") or "").strip() or None
        doc["ec_cpu_affinity"] = cpu_affinity(pid)
        doc["ec_ns_tid"] = ns_pid_of_thread(pid, pid)
        if args.threads:
            doc["threads"] = list_threads(pid)
    if args.verify_run_dir:
        doc["lite_log_evidence"] = lite_log_evidence(
            pathlib.Path(args.verify_run_dir), doc.get("ec_ns_tid"))

    text = json.dumps(doc, indent=2, sort_keys=True)
    print(text)
    if args.json:
        pathlib.Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0 if doc.get("ec_host_tid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
