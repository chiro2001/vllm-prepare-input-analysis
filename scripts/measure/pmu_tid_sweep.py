#!/usr/bin/env python3
"""Serial libkperfx topdown sweep against one live host TID (Kunpeng 920B).

Why this exists
---------------
``perf record`` answers "which function is hot"; it cannot answer "how much of
the pipeline was stalled on L2/DRAM".  On the 920B that second question needs
the ``920b_topdown_full_1..9`` preset family, which has to be run **one group
at a time** (the hardware fits 8 counters per group, the full tree needs 66).
libkperfx is the in-house collector for that; this script drives it against a
**thread that is already running** (the vLLM engine-core thread), which the
``kperfx`` CLI cannot do (``kperfx run`` always measures a *child* command and
``kperfx split`` re-runs that child once per group).

What it produces (``--outdir DIR``)
-----------------------------------
``pmu_group<g>.json``      verbatim libkperfx JSON, written by the library's own
                           C writer (``kperfx_ctx_write_output``), i.e. the same
                           bytes ``kperfx --format json --output stdout`` emits
``pmu_group<g>.meta.json`` group status, confidence, time_enabled/time_running,
                           useronly_forced, counters, timestamps, stderr tail
``topdown.json``           ``kperfx.topdown_920b(merged)``: four level-1 buckets,
                           the level-2/3 sub-items, per-group confidence
``libkperfx.json``         every counter (raw + rate) plus derived metrics (IPC,
                           CPI, GHz) and the time_enabled/time_running each
                           counter inherits from its group
``manifest.json``          command, groups, window, script sha256, min/mean
                           confidence, ``all_confident``,
                           ``low_confidence_groups``, exit code
``pmu_sweep.log``          full text log

Data-quality rules baked in
---------------------------
* one preset group at a time, strictly serial (libkperfx "whole groups are
  never mixed");
* every group records ``time_enabled``/``time_running``; ``confidence`` is
  ``time_running/time_enabled`` of the group leader -- that is what libkperfx
  exposes, and the kernel does not report per-counter running time inside a
  multiplexed group, so each counter inherits its group's pair;
* a group below ``--min-confidence`` (default 0.99) is retried ``--retries``
  times (default 1) and, if it stays low, is listed in
  ``manifest.json:low_confidence_groups``;
* counters that stay at 0 are listed in ``manifest.json:zero_counting_events``
  (detects dead PMU events);
* the sweep refuses to start while another ``perf``/``kperfx`` session measures
  the same target, or while another instrumenting process can run on the target
  thread's current CPU.  Mixing collectors silently degrades confidence --
  0.35 was observed on this box while a second collector was active.  Use
  ``--force`` to override.

Safety rails (project rules)
----------------------------
* never touches ``/dev/davinci*``; this is a CPU-only collector and therefore
  does not take the chip3 lock (it only *observes* a TID);
* ``--selftest`` refuses to place its workload on CPUs 120-159 (the reserved
  chip3 slice);
* PMU opens need root: the PMU part is re-executed through ``sudo -n``
  (``--no-sudo`` / already-root supported); any libkperfx ``useronly_forced``
  fallback is recorded in the manifest instead of being silently averaged in.

Usage (a3-22)
-------------
    python3 scripts/measure/pmu_tid_sweep.py --tid <HOST_TID> --window 5 \
            --groups 1-9 --tag a-c64 --outdir data/measure/<run>/pmu
    python3 scripts/measure/pmu_tid_sweep.py --list-groups
    python3 scripts/measure/pmu_tid_sweep.py --selftest --selftest-cpu 200

Exit codes: 0 = every requested group captured confidently; 1 = partial (some
groups usable); 2 = nothing usable / refused to start.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import platform
import re
import signal
import socket
import subprocess
import sys
import time

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_PRESET_PREFIX = "920b_topdown_full_"
DEFAULT_MIN_CONFIDENCE = 0.99
DEFAULT_LIBKPERFX_PYTHON = REPO_ROOT / "tools" / "libkperfx" / "python"
FALLBACK_LIBKPERFX_PYTHON = pathlib.Path(
    "/home/REMOTE_USER/projects/vllm/prepare-input-phase/tools/libkperfx/python"
)
RESERVED_CPU_LO, RESERVED_CPU_HI = 120, 159  # chip3 slice: selftest stays out

# What each preset group contributes.  Kept next to the code so ``--list-groups``
# can answer "which group carries L1/L2/L3/LLC/DRAM?" without the README.
GROUP_ROLES = {
    1: "L1 topdown buckets + IPC backbone (cycles, inst_retired, inst_spec, "
       "fetch_bubble, exec_stall, total_resource_stall)",
    2: "memory-stall hierarchy = L1/L2/LLC/DRAM bound split "
       "(mem_stall_anyload, memstall_l1miss/l2miss/l3miss)",
    3: "bad-spec attribution (branch mispredicts, o3/nuke/if flushes)",
    4: "instruction side: L2I cache + iTLB + pcbuf/ptag stalls",
    5: "ptag/mpq stall split + ports 0-1",
    6: "ports 2-6 + load cancels (dtlb/misalign/lq-full)",
    7: "L2 bottleneck reasons (buf/snp/arb) + DRAM_LOCAL + load cancels",
    8: "DRAM_REMOTE / DRAM_REMOTE_CACHE (NUMA!) + store stalls",
    9: "software events (context switches, migrations, page faults, cpu clock)",
}

CACHE_LEVEL_MAP = {
    "L1D": "no dedicated cache counter in the presets; L1 stalls live in group 2 "
           "(mem_stall_anyload / memstall_l1miss).  Direct access/refill counters "
           "l1d_cache,l1d_cache_refill exist but sit in no preset group",
    "L1I": "group 4 (l2i_* is the instruction-cache hierarchy; l1i_cache_refill is "
           "available as a custom event only)",
    "L2":  "group 2 (memstall_l1miss = L1-miss traffic, memstall_l2miss = L2 miss) "
           "+ group 7 (l2_bound_buf/snp/arb) + group 4 (l2i_*)",
    "LLC/L3": "group 2 only, and only as the derived topdown slot "
              "(memstall_l2miss - memstall_l3miss); there is no L3 access/fill "
              "counter and no L3C uncore PMU on this kernel",
    "DRAM": "group 2 (memstall_l3miss = LLC miss), group 7 (dram_local), group 8 "
            "(dram_remote, dram_remote_cache); no DDRC uncore PMU",
}

# Which preset groups a topdown output field needs.  Used to tell a consumer
# whether ``topdown.json`` value is trustworthy for the groups it asked for.
TOPDOWN_REQUIREMENTS = {
    "l1_topdown_pct": [1],
    "frontend_latency_bound": [1],
    "frontend_bandwidth_bound": [1],
    "resource_bound": [1, 2],
    "core_bound": [1, 2],
    "mem_bound": [1, 2],
    "mem_l1_bound": [1, 2],
    "mem_l2_bound": [1, 2],
    "mem_l3_dram_bound": [1, 2],
    "mem_mem_bound": [1, 2],
    "mem_store_bound": [1, 2],
    "ooo_stall_split": [2, 4, 5],
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_groups(spec: str) -> list[int]:
    """'1-9' / '1,3,7' / '1-3,7,8' -> [1, 3, 7, 8] (sorted, unique)."""
    out: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            if not (lo.isdigit() and hi.isdigit()):
                raise SystemExit(f"bad --groups element: {part!r}")
            out.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise SystemExit(f"bad --groups element: {part!r}")
    if not out:
        raise SystemExit("--groups is empty")
    if min(out) < 1 or max(out) > 9:
        raise SystemExit("--groups must be within 1-9 (920b_topdown_full_1..9)")
    return sorted(out)


def resolve_kperfx_python(explicit: str | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    env = os.environ.get("KPERFX_PYTHON_DIR")
    if env:
        candidates.append(pathlib.Path(env))
    candidates += [DEFAULT_LIBKPERFX_PYTHON, FALLBACK_LIBKPERFX_PYTHON]
    for cand in candidates:
        if (cand / "kperfx.py").is_file():
            return cand.resolve()
    raise SystemExit(
        "cannot find the libkperfx python binding (looked for kperfx.py in: "
        + ", ".join(str(c) for c in candidates)
        + "); pass --kperfx-python or set KPERFX_PYTHON_DIR"
    )


def import_kperfx(kperfx_python: pathlib.Path):
    if str(kperfx_python) not in sys.path:
        sys.path.insert(0, str(kperfx_python))
    import kperfx  # noqa: E402  (deliberately late: sys.path set just above)

    return kperfx


def emit_raw_json_with_c_writer(kperfx, session, result) -> bool:
    """Print libkperfx's own JSON for ``result`` on stdout.

    The ctypes binding carries the whole C context, so the canonical writer
    ``kperfx_ctx_write_output(ctx, res)`` -- the one behind
    ``kperfx --format json --output stdout`` -- can be called directly.  That
    keeps the artifacts in the library's exact schema instead of a lookalike.
    """
    lib = getattr(kperfx, "_lib", None)
    if lib is None:
        return False
    fn = getattr(lib, "kperfx_ctx_write_output", None)
    if fn is None:
        return False
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(kperfx.Result)]
    try:
        rc = int(fn(session.ctx, ctypes.byref(result)))
    except Exception:  # pragma: no cover - defensive
        return False
    return rc == 0


def emit_raw_json_fallback(kperfx, result) -> str:
    """Schema-compatible re-implementation of ``kperfx_result_write(JSON)``.

    Only used when the C writer cannot be reached; field order and formatting
    mirror ``src/kperfx_ctx.c``.
    """
    lines = ["{", '  "count": %d,' % result.count, '  "entries": [']
    for i in range(result.count):
        e = result.entries[i]
        lines.append(
            '%s\n    {"name":"%s","id":"%s","config":"0x%x","delta":%d,"rate":%.6f}'
            % ("," if i else "", e.name.decode(), e.id.decode(), e.config,
               e.delta, e.rate)
        )
    lines.append("\n  ],")
    lines.append('  "time_enabled": %d,' % result.time_enabled)
    lines.append('  "time_running": %d,' % result.time_running)
    lines.append('  "confidence": %.6f,' % result.confidence)
    lines.append('  "sampled": %s,' % ("true" if result.sampled else "false"))
    lines.append('  "sample_count": %d,' % result.sample_count)
    lines.append('  "lost_samples": %d,' % result.lost_samples)
    lines.append('  "gated": %s,' % ("true" if result.gated else "false"))
    lines.append('  "iter_gated": %s,'
                 % ("true" if result.iter_gated else "false"))
    lines.append('  "iter_start_at": %d,' % result.iter_start_at)
    lines.append('  "iter_end_at": %d,' % result.iter_end_at)
    lines.append('  "iter_start_seen": %d,' % result.iter_start_seen)
    lines.append('  "iter_end_seen": %d,' % result.iter_end_seen)
    lines.append('  "roi_start_seen": %d,' % result.roi_start_seen)
    lines.append('  "roi_end_seen": %d,' % result.roi_end_seen)
    lines.append('  "useronly_forced": %s,'
                 % ("true" if result.useronly_forced else "false"))
    lines.append('  "instances": %d,' % result.instances)
    try:
        pmu = kperfx.pmu_name()
    except Exception:  # pragma: no cover - defensive
        pmu = "unknown"
    lines.append('  "pmu": "%s"' % pmu)
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# /proc introspection: target thread + competing instrument sessions
# --------------------------------------------------------------------------- #
def read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def proc_stat_fields(pid: int) -> list[str] | None:
    text = read_text(pathlib.Path(f"/proc/{pid}/stat"))
    if text is None:
        return None
    # comm sits in parentheses and may contain spaces/parentheses itself, so
    # split on the *last* ')' and keep everything after it as numeric fields:
    #   "<pid> (<comm>) <state> <ppid> ..."
    # returned as [comm, state, ppid, ...]  (processor/CPU stays at index 37)
    head, _, rest = text.rpartition(")")
    comm = head.split("(", 1)[1].strip() if "(" in head else head.strip()
    return [comm] + rest.split()


def proc_cmdline(pid: int) -> str:
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def proc_cpu(pid: int) -> int:
    fields = proc_stat_fields(pid)
    if not fields or len(fields) < 38:
        return -1
    try:
        return int(fields[37])  # /proc/<pid>/stat field 39, 0-based 37
    except ValueError:
        return -1


def allowed_cpus(pid: int) -> set[int]:
    status = read_text(pathlib.Path(f"/proc/{pid}/status")) or ""
    for line in status.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            cpus: set[int] = set()
            for part in line.split(":", 1)[1].strip().split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo, _, hi = part.partition("-")
                    try:
                        cpus.update(range(int(lo), int(hi) + 1))
                    except ValueError:
                        pass
                else:
                    try:
                        cpus.add(int(part))
                    except ValueError:
                        pass
            return cpus
    return set()


def describe_tid(tid: int) -> dict:
    fields = proc_stat_fields(tid)
    if fields is None:
        raise RuntimeError(f"target tid {tid} does not exist (/proc/{tid} is gone)")
    return {
        "tid": tid,
        "comm": fields[0],
        "state": fields[1] if len(fields) > 1 else "?",
        "cpu": proc_cpu(tid),
        "cpus_allowed": sorted(allowed_cpus(tid)),
    }


INSTRUMENT_HINT = re.compile(
    r"(^|/|\s)(perf|kperfx|pmu_tid_sweep\.py)(\s|$)|"
    r"perf_(stat|record|top|report|capture)|pmu_tid_sweep"
)

WRAPPER_BINARIES = {
    "sudo", "taskset", "env", "nice", "ionice", "time", "nohup", "setsid",
    "stdbuf", "chrt", "timeout", "bash", "sh", "zsh", "dash",
}

# Subcommands that actually open PMU counters.  `perf report|annotate|script|
# diff|evlist` and `kperfx events|presets|merge|info|env` only read files, and
# this project runs `perf report` right after every perf capture -- treating
# those as collectors would refuse perfectly good sweeps.
PERF_PMU_SUBCOMMANDS = {"stat", "record", "top", "bench", "stat-record",
                        "record-stat"}
KPERFX_PMU_SUBCOMMANDS = {"run", "split", "probe", "probe-mux", "verify",
                          "count", "sweep"}


def classify_collector(cmdline: str) -> str | None:
    """Return 'perf'/'kperfx'/'kperfx-sweep' when this process *is* a PMU
    collector, else None.

    Shell wrappers are skipped so that a ``bash -c '... perf stat ...'`` job is
    judged by the real ``perf`` child process (which is visible separately), and
    a shell that merely *mentions* our script name is never mistaken for a
    collector.
    """
    tokens = cmdline.split()
    index = 0
    while index < len(tokens):
        base = os.path.basename(tokens[index])
        if base in WRAPPER_BINARIES:
            index += 1
            while index < len(tokens) and (
                tokens[index].startswith("-")
                or re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", tokens[index])
                or re.fullmatch(r"[\d,\-\.]+", tokens[index])
            ):
                index += 1
            continue
        break
    if index >= len(tokens):
        return None
    argv0 = os.path.basename(tokens[index])
    rest = tokens[index + 1:]
    if argv0 == "perf" or argv0.startswith("perf.") or argv0.startswith("perf_"):
        # perf's subcommand is its first non-option token
        sub = next((t for t in rest if not t.startswith("-")), None)
        return "perf" if sub in PERF_PMU_SUBCOMMANDS else None
    if argv0.startswith("kperfx"):
        # kperfx allows global options (--preset X) before the subcommand
        return "kperfx" if any(t in KPERFX_PMU_SUBCOMMANDS for t in rest) else None
    if argv0.startswith("python") and re.search(r"pmu_tid_sweep\.py", cmdline):
        return "kperfx-sweep" if "--inner" in tokens else None
    return None


def ancestor_pids(pid: int, limit: int = 12) -> set[int]:
    """The process plus its parents (so our own toolchain is never a conflict)."""
    chain: set[int] = set()
    current = pid
    for _ in range(limit):
        if current <= 0 or current in chain:
            break
        chain.add(current)
        fields = proc_stat_fields(current)
        if not fields or len(fields) < 3:
            break
        try:
            current = int(fields[2])  # ppid
        except ValueError:
            break
    return chain


def instrument_scope(cmdline: str) -> tuple[list[int], list[int]]:
    """(tids, cpus) a perf/kperfx command line explicitly measures.

    ``-t/--tid/-p/--pid N`` -> tids, ``-C N`` / ``-C N-M`` / ``--cpu=N`` ->
    cpus.  An empty result means the collector has no scope of its own (i.e.
    it is system-wide).
    """
    tids: list[int] = []
    cpus: list[int] = []
    tokens = cmdline.split()
    for index, token in enumerate(tokens):
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token in ("-t", "--tid", "-p", "--pid") and nxt.isdigit():
            tids.append(int(nxt))
            continue
        if token in ("-C", "--cpu") and nxt:
            cpus.extend(int(c) for c in re.findall(r"\d+", nxt))
            continue
        match = re.match(r"^(?:-t|-p|--tid|--pid)[=:]?(\d+)$", token)
        if match:
            tids.append(int(match.group(1)))
            continue
        match = re.match(r"^(?:-C|--cpu)[=:]?([\d,\-]+)$", token)
        if match:
            cpus.extend(int(c) for c in re.findall(r"\d+", match.group(1)))
    return tids, cpus


def find_pmu_conflicts(tid: int, target_cpu: int, self_pids: set[int]) -> list[dict]:
    """Other PMU consumers that could steal counters from this target."""
    conflicts: list[dict] = []
    own = set(self_pids) | ancestor_pids(os.getpid())
    my_group = os.getpgrp()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in own:
            continue
        cmdline = proc_cmdline(pid)
        kind = classify_collector(cmdline)
        if kind is None:
            continue
        comm = (proc_stat_fields(pid) or [""])[0]
        try:
            if os.getpgid(pid) == my_group:
                continue  # our own job (e.g. our inner captures)
        except OSError:
            pass
        cpus = allowed_cpus(pid)
        cpu_now = proc_cpu(pid)
        same_target = re.search(rf"(?<!\d){tid}(?!\d)", cmdline) is not None
        measured_tids, measured_cpus = instrument_scope(cmdline)
        measured_on_target_cpu = [
            other for other in measured_tids
            if other != tid and target_cpu >= 0 and proc_cpu(other) == target_cpu
        ]
        # A scoped collector that names its own -t/-p target cannot disturb us
        # unless that very target runs on our CPU; an explicit -C <our cpu> or a
        # collector without any scope does cover every CPU it may run on.
        cpu_scoped_instrument = (
            target_cpu >= 0 and target_cpu in measured_cpus
        ) or (
            not measured_tids and not measured_cpus and
            (target_cpu >= 0 and (cpu_now == target_cpu or target_cpu in cpus))
        )
        if same_target or measured_on_target_cpu or cpu_scoped_instrument:
            conflicts.append(
                {
                    "pid": pid,
                    "kind": kind,
                    "comm": comm,
                    "cpu_now": cpu_now,
                    "same_target": bool(same_target),
                    "measured_tids": measured_tids[:8],
                    "measured_cpus": measured_cpus[:12],
                    "measured_tids_on_target_cpu": measured_on_target_cpu[:8],
                    "cpu_scoped_instrument": bool(cpu_scoped_instrument),
                    "cmdline": cmdline[:240],
                }
            )
    return conflicts


def thread_runtime_ns(tid: int) -> int | None:
    """Per-thread runtime from /proc/<tid>/schedstat (first field, ns)."""
    text = read_text(pathlib.Path(f"/proc/{tid}/schedstat"))
    if not text:
        return None
    parts = text.split()
    if len(parts) < 1:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def find_busy_tid(pid: int, sample_s: float = 0.4) -> tuple[int, dict]:
    """Busy thread of ``pid`` -> (tid, evidence).

    ``workload_gen`` runs its load in a pthread while the main thread only
    joins; measuring the *pid* would count a sleeping task (perf prints
    "<not counted>", kperfx returns zeros).  ``/proc/<tid>/stat`` utime is
    useless for this: the thread-group leader reports the *process* total, so
    the leader looks as busy as the worker.  ``/proc/<tid>/schedstat`` runtime
    is per-thread and disambiguates cleanly.
    """
    try:
        tasks = sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return pid, {"reason": "no /proc/<pid>/task"}
    if len(tasks) == 1:
        return tasks[0], {"reason": "single thread", "tasks": tasks}

    runtime_before = {t: thread_runtime_ns(t) for t in tasks}
    time.sleep(sample_s)
    runtime_after = {t: thread_runtime_ns(t) for t in tasks}
    if all(runtime_before[t] is not None and runtime_after[t] is not None
           for t in tasks):
        delta = {t: runtime_after[t] - runtime_before[t] for t in tasks}
        best = max(delta, key=lambda t: delta[t])
        return best, {"source": "schedstat_runtime_ns", "delta": delta}

    # fallback: utime delta, tie-broken towards a currently running thread
    def utime(tid: int) -> int:
        fields = proc_stat_fields(tid)
        if not fields or len(fields) < 13:
            return 0
        try:
            return int(fields[12])  # utime = /proc/<tid>/stat field 14
        except ValueError:
            return 0

    before = {t: utime(t) for t in tasks}
    time.sleep(sample_s)
    delta = {t: utime(t) - before[t] for t in tasks}
    running = [t for t in tasks if (proc_stat_fields(t) or ["", "?"])[1] == "R"]
    order = running or [t for t in tasks if t != pid] or tasks
    best = max(order, key=lambda t: delta[t])
    return best, {"source": "utime_delta", "delta": delta,
                  "running": running, "leader": pid}


# --------------------------------------------------------------------------- #
# inner mode: exactly one preset group, one count window
# --------------------------------------------------------------------------- #
def inner_main(args) -> int:
    kperfx_python = resolve_kperfx_python(args.kperfx_python)
    kperfx = import_kperfx(kperfx_python)
    preset = f"{args.preset_prefix}{args.group}"
    family = args.preset_prefix.rstrip("_")
    try:
        n_groups = kperfx.preset_groups(family)
        if n_groups and args.group > n_groups:
            print(f"inner: group {args.group} > {n_groups} in {family}",
                  file=sys.stderr)
            return 3
        events = kperfx.preset_events(family, args.group - 1)
    except Exception as exc:
        print(f"inner: unknown preset {preset!r}: {exc!r}", file=sys.stderr)
        return 3

    session = None
    t0 = time.time()
    started = utc_now()
    try:
        session = kperfx.Session(
            preset=preset,
            target="tid",
            pid=args.tid,
            mode="count",
            format="json",
            output="stdout",
        )
        session.start()
    except Exception as exc:  # permission, unknown event, group too large, ...
        print(f"inner: kperfx start failed for {preset}: {exc!r}", file=sys.stderr)
        return 3

    time.sleep(args.window)
    try:
        result = session.stop()
    except Exception as exc:
        print(f"inner: kperfx stop failed for {preset}: {exc!r}", file=sys.stderr)
        return 3
    t1 = time.time()
    finished = utc_now()

    wrote = emit_raw_json_with_c_writer(kperfx, session, result)
    if not wrote:
        sys.stdout.write(emit_raw_json_fallback(kperfx, result))
    sys.stdout.flush()

    meta = {
        "group": args.group,
        "preset": preset,
        "events": events,
        "tid": args.tid,
        "window_requested_s": args.window,
        "wall_s": round(t1 - t0, 6),
        "started_at_utc": started,
        "finished_at_utc": finished,
        "time_enabled_ns": int(result.time_enabled),
        "time_running_ns": int(result.time_running),
        "confidence": float(result.confidence),
        "useronly_forced": bool(result.useronly_forced),
        "instances": int(result.instances),
        "counters": int(result.count),
        "raw_json_writer": "kperfx_ctx_write_output" if wrote else "python-fallback",
    }
    print("KPERFX_META " + json.dumps(meta), file=sys.stderr)
    try:
        session.close()
    except Exception:  # pragma: no cover - defensive
        pass
    return 0


# --------------------------------------------------------------------------- #
# outer mode helpers
# --------------------------------------------------------------------------- #
def run_with_timeout(cmd: list[str], timeout: float):
    """Run cmd in its own process group; return (rc, stdout, stderr, timed_out)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            out, err = proc.communicate()
    return proc.returncode, out, err, timed_out


def scan_counter_lines(source: str) -> list[dict]:
    """Parse ``perf stat`` text output into [{event, value, running_pct}]."""
    rows: list[dict] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pct = re.search(r"\((\d+\.\d+)%\)", stripped)
        m = re.match(
            r"^([\d,]+|<not counted>|<not supported>)\s+([A-Za-z0-9_\-:.]+)",
            stripped,
        )
        if not m:
            continue
        raw = m.group(1)
        rows.append(
            {
                "event": m.group(2),
                "value": None if raw.startswith("<") else int(raw.replace(",", "")),
                "running_pct": float(pct.group(1)) if pct else None,
                "line": stripped,
            }
        )
    return rows


class Sweep:
    """Serial multi-group capture against one TID."""

    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.outdir = pathlib.Path(args.outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.kperfx_python = resolve_kperfx_python(args.kperfx_python)
        self.groups = parse_groups(args.groups)
        self.records: dict[int, dict] = {}
        self.python = sys.executable or "python3"
        self.use_sudo = bool(args.sudo) and os.geteuid() != 0
        self.script_sha = sha256_file(SCRIPT_PATH)
        self.started = utc_now()
        self.t_start = time.time()

    # -- logging ---------------------------------------------------------- #
    def say(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    # -- environment checks ----------------------------------------------- #
    def preflight(self) -> dict:
        info: dict = {"refused": None}
        try:
            kperfx = import_kperfx(self.kperfx_python)
            info["kperfx_version"] = kperfx.version()
            info["pmu"] = kperfx.pmu_name()
            info["backend"] = kperfx.backend_name()
            info["preset_groups"] = kperfx.preset_groups(
                self.args.preset_prefix.rstrip("_")
            )
        except Exception as exc:
            info["refused"] = (
                f"cannot use libkperfx from {self.kperfx_python}: {exc!r}"
            )
            return info
        if self.use_sudo:
            rc = subprocess.run(["sudo", "-n", "true"],
                                capture_output=True).returncode
            if rc != 0:
                info["refused"] = (
                    "sudo -n is not usable and we are not root: PMU counting needs "
                    "root (perf_event_paranoid=2).  Run as root or allow "
                    "passwordless sudo."
                )
                return info
        info["sudo"] = self.use_sudo
        tid = self.args.tid
        info["target"] = describe_tid(tid)
        if self.args.pid and self.args.pid > 0:
            try:
                threads = [int(t) for t in os.listdir(f"/proc/{self.args.pid}/task")]
            except OSError:
                threads = []
            info["pid"] = self.args.pid
            info["tid_in_pid"] = (tid in threads) if threads else None
            if threads and tid not in threads:
                self.say(f"warning: tid {tid} is not a thread of pid {self.args.pid}")
        conflicts = find_pmu_conflicts(
            tid, info["target"]["cpu"], {os.getppid(), os.getpid()}
        )
        info["conflicts"] = conflicts
        if conflicts and not self.args.force:
            info["refused"] = (
                f"another PMU collector looks active on the same target/CPU "
                f"({len(conflicts)} process(es)); mixing collectors corrupts "
                "confidence.  Use --force to override."
            )
        return info

    # -- one group --------------------------------------------------------- #
    def inner_command(self, group: int) -> list[str]:
        cmd = [
            self.python,
            str(SCRIPT_PATH),
            "--inner",
            "--tid", str(self.args.tid),
            "--group", str(group),
            "--window", f"{self.args.window:.6f}",
            "--preset-prefix", self.args.preset_prefix,
            "--kperfx-python", str(self.kperfx_python),
            "--quiet",
        ]
        return (["sudo", "-n"] + cmd) if self.use_sudo else cmd

    def capture_group(self, group: int) -> dict:
        cmd = self.inner_command(group)
        timeout = self.args.window + 45.0
        rc, out, err, timed_out = run_with_timeout(cmd, timeout)
        record = {
            "group": group,
            "preset": f"{self.args.preset_prefix}{group}",
            "cmd": cmd,
            "rc": rc,
            "timed_out": timed_out,
            "stdout": out,
            "stderr": err,
            "raw_json": None,
            "error": None,
            "inner_meta": None,
        }
        for line in err.splitlines():
            if line.startswith("KPERFX_META "):
                try:
                    record["inner_meta"] = json.loads(line[len("KPERFX_META "):])
                except json.JSONDecodeError:
                    pass
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict) and "entries" in parsed:
                record["raw_json"] = parsed
        except json.JSONDecodeError:
            pass
        if record["raw_json"] is None:
            record["error"] = ("timeout" if timed_out
                               else f"no kperfx JSON on stdout (rc={rc})")
        elif rc != 0:
            record["error"] = f"inner rc={rc}"
        elif not record["raw_json"].get("entries"):
            record["error"] = "empty counter list"
        return record

    # -- orchestration ------------------------------------------------------ #
    def run(self) -> int:
        args = self.args
        self.say(f"pmu_tid_sweep start tid={args.tid} groups={self.groups} "
                 f"window={args.window}s prefix={args.preset_prefix} "
                 f"-> {self.outdir}")
        try:
            preflight = self.preflight()
        except SystemExit as exc:            # argparse-style refusals
            preflight = {"refused": str(exc)}
        except Exception as exc:             # vanished tid, unreadable /proc, ...
            preflight = {"refused": f"preflight failed: {exc!r}"}
        if preflight.get("refused"):
            self.say("refused: " + str(preflight["refused"]))
            self.write_manifest(preflight, exit_code=2)
            return 2
        if preflight.get("conflicts"):
            for c in preflight["conflicts"]:
                self.say(f"conflict (forced through): pid={c['pid']} "
                         f"comm={c['comm']} cmd={c['cmdline'][:120]}")
        if preflight.get("tid_in_pid") is False:
            self.say("continuing anyway (tid is not listed under --pid)")
        for key in ("kperfx_version", "pmu", "backend", "preset_groups", "sudo"):
            if key in preflight:
                self.say(f"{key}={preflight[key]}")
        self.say(f"target={json.dumps(preflight.get('target'))} "
                 f"script_sha256={self.script_sha}")

        for group in self.groups:
            attempt = 0
            while True:
                attempt += 1
                t0 = time.time()
                rec = self.capture_group(group)
                rec["attempt"] = attempt
                rec["outer_wall_s"] = round(time.time() - t0, 4)
                conf = (float(rec["raw_json"].get("confidence", 0.0))
                        if rec["raw_json"] else None)
                low = conf is None or conf < args.min_confidence
                retry = attempt <= args.retries and (rec["error"] is not None or low)
                if retry:
                    why = rec["error"] or f"confidence {conf:.4f} < {args.min_confidence}"
                    self.say(f"group {group} attempt {attempt}: {why} -> retry")
                    continue
                if rec["error"] is None and low:
                    self.say(f"group {group}: LOW CONFIDENCE {conf:.4f} "
                             "(kept and listed in manifest)")
                self.records[group] = rec
                self.persist_group(rec)
                break

        ok_groups = [g for g, r in self.records.items() if r["error"] is None]
        bad_groups = [g for g in self.groups if g not in ok_groups]
        for g in ok_groups:
            r = self.records[g]
            js = r["raw_json"]
            self.say(f"group {g}: OK conf={js['confidence']:.4f} "
                     f"t_en={js['time_enabled']}ns t_run={js['time_running']}ns "
                     f"useronly={js['useronly_forced']} "
                     f"counters={len(js['entries'])} "
                     f"wall={r['outer_wall_s']}s attempt={r['attempt']}")
        for g in bad_groups:
            self.say(f"group {g}: FAILED ({self.records[g]['error']})")

        if ok_groups:
            self.write_aggregates(ok_groups, bad_groups)
        exit_code = 2 if not ok_groups else (1 if bad_groups else 0)
        self.write_manifest(preflight, exit_code=exit_code,
                            ok=ok_groups, bad=bad_groups)
        self.say(f"done: {len(ok_groups)}/{len(self.groups)} groups ok "
                 f"(exit {exit_code})")
        return exit_code

    # -- artifacts --------------------------------------------------------- #
    def persist_group(self, rec: dict) -> None:
        group = rec["group"]
        raw = rec["raw_json"]
        json_path = self.outdir / f"pmu_group{group}.json"
        if raw is not None:
            json_path.write_text(rec["stdout"], encoding="utf-8")
        else:
            json_path.write_text(
                json.dumps(
                    {
                        "error": rec["error"],
                        "rc": rec["rc"],
                        "timed_out": rec["timed_out"],
                        "cmd": rec["cmd"],
                        "stdout_bytes": len(rec["stdout"]),
                        "stderr_tail": rec["stderr"][-4000:],
                        "note": "capture failed - no kperfx JSON was produced",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        counters = [
            {
                "name": e.get("name"),
                "id": e.get("id"),
                "config": e.get("config"),
                "delta": e.get("delta"),
                "rate": e.get("rate"),
            }
            for e in (raw or {}).get("entries", [])
        ]
        meta = {
            "schema": "pi-pmu-group-meta-v1",
            "tag": self.args.tag,
            "group": group,
            "preset": rec["preset"],
            "events": (rec["inner_meta"] or {}).get("events"),
            "tid": self.args.tid,
            "pid": self.args.pid,
            "status": "ok" if rec["error"] is None else "failed",
            "attempts": rec["attempt"],
            "rc": rec["rc"],
            "timed_out": rec["timed_out"],
            "error": rec["error"],
            "outer_wall_s": rec["outer_wall_s"],
            "inner_wall_s": (rec["inner_meta"] or {}).get("wall_s"),
            "started_at_utc": (rec["inner_meta"] or {}).get("started_at_utc"),
            "finished_at_utc": (rec["inner_meta"] or {}).get("finished_at_utc"),
            "confidence": (raw or {}).get("confidence"),
            "time_enabled": (raw or {}).get("time_enabled"),
            "time_running": (raw or {}).get("time_running"),
            "useronly_forced": (raw or {}).get("useronly_forced"),
            "counters": counters,
            "counters_map": {c["name"]: c["delta"] for c in counters if c["name"]},
            "cmd": rec["cmd"],
            "stderr_tail": rec["stderr"][-4000:],
            "raw_json_writer": (rec["inner_meta"] or {}).get("raw_json_writer"),
        }
        (self.outdir / f"pmu_group{group}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def merged_counters(self, ok_groups: list[int]) -> tuple[dict, dict]:
        """Merge the entries of every successful group by event config."""
        merged: dict[int, int] = {}
        owner: dict[int, int] = {}
        for g in ok_groups:
            for entry in self.records[g]["raw_json"].get("entries", []):
                cfg = entry.get("config")
                cfg = int(cfg, 16) if isinstance(cfg, str) else int(cfg)
                merged[cfg] = merged.get(cfg, 0) + int(entry.get("delta", 0))
                owner.setdefault(cfg, g)
        return merged, owner

    def write_aggregates(self, ok_groups: list[int], bad_groups: list[int]) -> None:
        kperfx = import_kperfx(self.kperfx_python)
        merged, owner = self.merged_counters(ok_groups)
        topdown = kperfx.topdown_920b(merged) if merged else {}
        conf_by_group = {
            str(g): float(self.records[g]["raw_json"]["confidence"]) for g in ok_groups
        }
        four_buckets = {
            k: topdown.get(k)
            for k in ("frontend_bound", "bad_spec", "retiring", "backend_bound")
        }
        subitems = {k: v for k, v in topdown.items() if k not in four_buckets}
        coverage = {
            field: {"needs_groups": need,
                    "complete": set(need) <= set(ok_groups)}
            for field, need in TOPDOWN_REQUIREMENTS.items()
        }
        (self.outdir / "topdown.json").write_text(
            json.dumps(
                {
                    "schema": "pi-pmu-topdown-v1",
                    "tag": self.args.tag,
                    "preset_prefix": self.args.preset_prefix,
                    "issue_width": 6,
                    "groups_requested": self.groups,
                    "groups_used": ok_groups,
                    "groups_missing": bad_groups,
                    "complete_for_request": not bad_groups,
                    "complete_topdown_tree": ok_groups == list(range(1, 10)),
                    "l1_topdown_pct": four_buckets,
                    "subitems_pct": subitems,
                    "coverage": coverage,
                    "coverage_incomplete": sorted(
                        field for field, cov in coverage.items()
                        if not cov["complete"]
                    ),
                    "per_group_confidence": conf_by_group,
                    "merged_counters_by_config": {hex(k): v
                                                  for k, v in merged.items()},
                    "counter_group_owner": {hex(k): v for k, v in owner.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        per_counter: dict[str, dict] = {}
        entries: list[dict] = []
        for g in ok_groups:
            js = self.records[g]["raw_json"]
            for entry in js.get("entries", []):
                name = entry.get("name")
                per_counter[name] = {
                    "delta": entry.get("delta"),
                    "config": entry.get("config"),
                    "rate_per_s": entry.get("rate"),
                    "group": g,
                    "preset": f"{self.args.preset_prefix}{g}",
                    # libkperfx reports the *group leader's* time_enabled /
                    # time_running; the kernel does not expose per-counter
                    # running time inside a multiplexed group, so each counter
                    # inherits its group's pair.
                    "time_enabled_ns": js.get("time_enabled"),
                    "time_running_ns": js.get("time_running"),
                    "confidence": js.get("confidence"),
                    "useronly_forced": js.get("useronly_forced"),
                }
                entries.append(dict(entry, group=g))
        cycles = per_counter.get("CPU_CYCLES", {}).get("delta") or 0
        retired = per_counter.get("INST_RETIRED", {}).get("delta") or 0
        spec = per_counter.get("INST_SPEC", {}).get("delta") or 0
        t_run = per_counter.get("CPU_CYCLES", {}).get("time_running_ns") or 0
        t_en = per_counter.get("CPU_CYCLES", {}).get("time_enabled_ns") or 0
        derived = {
            "cpu_cycles": cycles,
            "inst_retired": retired,
            "inst_spec": spec,
            "ipc": (retired / cycles) if cycles else None,
            "ipc_against_inst_spec": (spec / cycles) if cycles else None,
            "cpi": (cycles / retired) if retired else None,
            "inst_spec_over_inst_retired": (spec / retired) if retired else None,
            "cpu_ghz_over_group1_time_running": (cycles / t_run) if t_run else None,
            "group1_time_enabled_ns": t_en,
            "group1_time_running_ns": t_run,
            "group1_confidence": (t_run / t_en) if t_en else None,
        }
        (self.outdir / "libkperfx.json").write_text(
            json.dumps(
                {
                    "schema": "pi-pmu-counters-v1",
                    "tag": self.args.tag,
                    "counters": {n: rec["delta"]
                                 for n, rec in per_counter.items()},
                    "per_counter": per_counter,
                    "entries": entries,
                    "derived": derived,
                    "topdown_pct": topdown,
                    "meta": {
                        "tid": self.args.tid,
                        "pid": self.args.pid,
                        "groups_used": ok_groups,
                        "groups_missing": bad_groups,
                        "window_s": self.args.window,
                        "preset_prefix": self.args.preset_prefix,
                        "per_group_confidence": conf_by_group,
                        "min_confidence": self.args.min_confidence,
                        "time_running_semantics": (
                            "time_enabled/time_running are the group leader's "
                            "values reported by libkperfx; every counter of a "
                            "group inherits that pair (the kernel does not expose "
                            "per-counter running time for a multiplexed group)"
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_manifest(self, preflight: dict, exit_code: int,
                       ok: list[int] | None = None,
                       bad: list[int] | None = None) -> None:
        ok = ok if ok is not None else []
        bad = bad if bad is not None else self.groups
        confs = {
            str(g): float(self.records[g]["raw_json"]["confidence"])
            for g in ok if (self.records.get(g) or {}).get("raw_json")
        }
        low = sorted(int(g) for g, c in confs.items()
                     if c < self.args.min_confidence)
        zero_events: list[str] = []
        for g in ok:
            for entry in self.records[g]["raw_json"].get("entries", []):
                if int(entry.get("delta", 0)) == 0:
                    zero_events.append(entry.get("name"))
        inner_template = None
        if self.groups:
            head = self.inner_command(self.groups[0])
            inner_template = head[:5] + ["..."]
        manifest = {
            "schema": "pi-pmu-sweep-manifest-v1",
            "tag": self.args.tag,
            "command": {
                "outer": [self.python, str(SCRIPT_PATH)] + sys.argv[1:],
                "inner_template": inner_template,
                "sudo": self.use_sudo,
            },
            "tid": self.args.tid,
            "pid": self.args.pid,
            "groups": self.groups,
            "window_s": self.args.window,
            "preset_prefix": self.args.preset_prefix,
            "retries": self.args.retries,
            "confidence_threshold": self.args.min_confidence,
            "script": {"path": str(SCRIPT_PATH), "sha256": self.script_sha},
            "script_sha256": self.script_sha,
            "python": self.python,
            "python_version": platform.python_version(),
            "kperfx_python_dir": str(self.kperfx_python),
            "kperfx": {k: v for k, v in preflight.items()
                       if k in ("kperfx_version", "pmu", "backend",
                                "preset_groups")},
            "host": {
                "hostname": socket.gethostname(),
                "nproc": os.cpu_count(),
                "kernel": platform.release(),
                "perf_event_paranoid": (read_text(
                    pathlib.Path("/proc/sys/kernel/perf_event_paranoid")) or ""
                ).strip(),
            },
            "target": preflight.get("target"),
            "conflicts": preflight.get("conflicts", []),
            "refused": preflight.get("refused"),
            "started_at_utc": self.started,
            "finished_at_utc": utc_now(),
            "wall_s": round(time.time() - self.t_start, 3),
            "groups_ok": ok,
            "groups_failed": bad,
            "per_group": {
                str(g): {
                    "status": ("ok" if self.records[g]["error"] is None
                               else "failed"),
                    "attempts": self.records[g]["attempt"],
                    "confidence": (self.records[g]["raw_json"] or {}).get(
                        "confidence"),
                    "time_enabled_ns": (self.records[g]["raw_json"] or {}).get(
                        "time_enabled"),
                    "time_running_ns": (self.records[g]["raw_json"] or {}).get(
                        "time_running"),
                    "useronly_forced": (self.records[g]["raw_json"] or {}).get(
                        "useronly_forced"),
                    "wall_s": self.records[g]["outer_wall_s"],
                    "error": self.records[g]["error"],
                }
                for g in self.groups if g in self.records
            },
            "min_confidence": min(confs.values()) if confs else None,
            "mean_confidence": (sum(confs.values()) / len(confs)) if confs else None,
            "all_confident": bool(confs) and all(
                c >= self.args.min_confidence for c in confs.values()
            ),
            "low_confidence_groups": low,
            "zero_counting_events": sorted(set(zero_events)),
            "exit_code": exit_code,
            "notes": [
                "one preset group at a time (8 counters fit, the topdown tree "
                "needs 66)",
                "pmu_group*.json is written by libkperfx's own C writer",
                "confidence = time_running/time_enabled of the group leader",
                "topdown/IPC are only meaningful together with the confidence "
                "recorded here",
            ],
        }
        (self.outdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# --list-groups
# --------------------------------------------------------------------------- #
def list_groups_main(args) -> int:
    kperfx_python = resolve_kperfx_python(args.kperfx_python)
    kperfx = import_kperfx(kperfx_python)
    prefix = args.preset_prefix
    family = prefix.rstrip("_")
    n = kperfx.preset_groups(family)
    print(f"libkperfx {kperfx.version()}  pmu={kperfx.pmu_name()}  "
          f"backend={kperfx.backend_name()}")
    print(f"preset family: {family}  groups={n}  pattern: {prefix}<1..{n}>")
    print()
    for g in range(1, n + 1):
        events = kperfx.preset_events(family, g - 1)
        print(f"group {g}: {len(events)} events")
        for ev in events:
            print(f"    {ev}")
        role = GROUP_ROLES.get(g)
        if role:
            print(f"    role: {role}")
        print()
    print("cache-level coverage on 920B:")
    for level, text in CACHE_LEVEL_MAP.items():
        print(f"  {level:8s} {text}")
    print()
    print("formula:")
    print(kperfx.preset_formula(family))
    return 0


# --------------------------------------------------------------------------- #
# --selftest : PMU counts vs perf counts on a synthetic single-thread load
# --------------------------------------------------------------------------- #
WORKLOAD_SRC = (
    "import sys, time\n"
    "x = 12345\n"
    "end = time.monotonic() + float(sys.argv[1])\n"
    "while time.monotonic() < end:\n"
    "    for _ in range(200000):\n"
    "        x = (x * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF\n"
)


def start_workload(cpu: int, seconds: float,
                   kind_name: str = "alu") -> tuple[subprocess.Popen, int, str]:
    """Start a single-threaded CPU load pinned to ``cpu`` -> (proc, pid, kind)."""
    gen = REPO_ROOT / "tools" / "libkperfx" / "build" / "workload_gen"
    if gen.is_file():
        cmd = ["taskset", "-c", str(cpu), str(gen),
               "--kind", kind_name, "--threads", "1",
               "--duration-ms", str(int(seconds * 1000))]
        kind = f"libkperfx workload_gen --kind {kind_name}"
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT, start_new_session=True)
        return proc, proc.pid, kind
    cmd = ["taskset", "-c", str(cpu), sys.executable, "-c", WORKLOAD_SRC,
           f"{seconds:.3f}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, start_new_session=True)
    return proc, proc.pid, "python alu loop"


def perf_reference(tid: int, window: float, log_say) -> dict:
    """perf counts for one steady window, with fallbacks.

    The 920B PMU is shared: when another session holds counters, perf reports
    ``<not counted>`` instead of failing, so retry and, if needed, fall back to
    one event per perf process.
    """
    single = [("cycles",), ("instructions",)]
    attempts: list[dict] = []
    plan: list[tuple[float, list[str]]] = [
        (window, ["cycles", "instructions"]),
        (window, ["cycles", "instructions"]),
        (window, ["cycles"]),
        (window, ["instructions"]),
    ]
    values: dict[str, int] = {}
    pct: dict[str, float | None] = {}
    for index, (win, events) in enumerate(plan, start=1):
        cmd = ["sudo", "-n", "perf", "stat", "-t", str(tid),
               "-e", ",".join(events), "--", "sleep", f"{win:.3f}"]
        rc, out, err, timed_out = run_with_timeout(cmd, win + 60)
        rows = [r for r in scan_counter_lines(out + err) if r["value"] is not None]
        got = {r["event"].split(":")[0]: r for r in rows}
        log_say(f"perf attempt {index} events={events} rc={rc} "
                f"counts={[(k, v['value']) for k, v in got.items()]}")
        attempts.append({"cmd": cmd, "rc": rc, "events": events,
                         "counts": {k: v["value"] for k, v in got.items()},
                         "running_pct": {k: v["running_pct"]
                                         for k, v in got.items()},
                         "raw": (out + err)[-2000:], "timed_out": timed_out})
        for name, row in got.items():
            values.setdefault(name, row["value"])
            pct.setdefault(name, row["running_pct"])
        if all(item in values for item in single[0] + single[1]):
            break
    return {
        "cycles": values.get("cycles"),
        "instructions": values.get("instructions"),
        "running_pct": pct,
        "attempts": attempts,
        "window_s": window,
    }


def selftest_main(args) -> int:
    cpu = args.selftest_cpu
    if RESERVED_CPU_LO <= cpu <= RESERVED_CPU_HI:
        print(f"refusing: CPU {cpu} is inside the reserved chip3 slice "
              f"{RESERVED_CPU_LO}-{RESERVED_CPU_HI}")
        return 2
    stamp = args.tag if args.tag != "pmu" else utc_stamp()
    base = (pathlib.Path(args.selftest_dir).resolve() if args.selftest_dir
            else REPO_ROOT / "data" / "profiles" / f"pmu-selftest-{stamp}")
    base.mkdir(parents=True, exist_ok=True)
    log = open(base / "selftest.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    groups = parse_groups(args.groups)
    kperfx_python = resolve_kperfx_python(args.kperfx_python)
    ref_window = args.selftest_window
    sweep_seconds = len(groups) * args.window
    full_seconds = 9 * args.selftest_full_window if args.selftest_full else 0.0
    load_seconds = ref_window + sweep_seconds + full_seconds + 30.0
    say(f"selftest cpu={cpu} groups={groups} sweep_window={args.window}s "
        f"perf_window={ref_window}s full9_window={args.selftest_full_window}s "
        f"workload={load_seconds:.0f}s dir={base}")

    # 1. preset-group evidence (also what PMU_NOTES.md quotes)
    rc, out, err, _ = run_with_timeout(
        [sys.executable, str(SCRIPT_PATH), "--list-groups",
         "--preset-prefix", args.preset_prefix,
         "--kperfx-python", str(kperfx_python)], 120)
    lg = base / "list-groups.txt"
    lg.write_text(out + (f"\n[stderr]\n{err}" if err.strip() else ""),
                  encoding="utf-8")
    say(f"--list-groups rc={rc} -> {lg}")
    if rc != 0:
        say("cannot list preset groups; aborting selftest")
        log.close()
        return 2

    # 2. workload: measure the *busy thread*, not the (sleeping) process leader
    proc, pid, kind = start_workload(cpu, load_seconds, args.selftest_kind)
    time.sleep(1.0)
    tid, evidence = find_busy_tid(pid)
    say(f"workload={kind} pid={pid} busy_tid={tid} cpu={cpu} "
        f"selection={json.dumps(evidence)}")
    if evidence.get("source") == "schedstat_runtime_ns":
        if evidence["delta"].get(tid, 0) <= 0:
            say("warning: the chosen thread used no CPU time in the sample window")
    elif evidence.get("delta", {}).get(tid, 0) <= 0:
        say("warning: the chosen thread burned no user time in the sample window")
    try:
        target = describe_tid(tid)
    except RuntimeError as exc:
        say(f"selftest aborted: {exc}")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        log.close()
        return 2
    say(f"target={json.dumps(target)}")
    if target["cpu"] != cpu:
        say(f"warning: workload reports CPU {target['cpu']} instead of {cpu}")

    results: dict = {
        "schema": "pi-pmu-selftest-v1",
        "stamp": stamp, "cpu": cpu, "workload": kind, "pid": pid, "tid": tid,
        "groups": groups, "sweep_window_s": args.window,
        "perf_window_s": ref_window, "target": target,
        "tid_selection": evidence,
    }
    try:
        # 3. perf reference: same instrument class (counting, no sampling)
        perf = perf_reference(tid, ref_window, say)
        (base / "perf_reference.json").write_text(
            json.dumps(perf, indent=2), encoding="utf-8")
        (base / "perf_reference.txt").write_text(
            "\n\n".join(f"# {' '.join(a['cmd'])}\n{a['raw']}"
                        for a in perf["attempts"]), encoding="utf-8")
        results["perf"] = perf
        if perf["cycles"] is None or perf["instructions"] is None:
            say("perf reference unavailable after 4 attempts "
                "(PMU held by another session?) - comparison will be reported "
                "as not measured")

        # 4. the sweep under test (same window length -> counts compare directly)
        sweep_dir = base / f"sweep-g{''.join(str(g) for g in groups)}-w{args.window:g}"
        rc_sw, out_sw, err_sw, _ = run_with_timeout(
            [sys.executable, str(SCRIPT_PATH),
             "--tid", str(tid), "--window", f"{args.window:.3f}",
             "--groups", args.groups, "--tag", f"selftest-{stamp}",
             "--outdir", str(sweep_dir), "--kperfx-python", str(kperfx_python)],
            len(groups) * (args.window + 45) + 120)
        (base / "sweep_stdout.txt").write_text(out_sw + err_sw, encoding="utf-8")
        say(f"sweep rc={rc_sw} -> {sweep_dir}")
        lib = json.loads((sweep_dir / "libkperfx.json").read_text())
        sw_cycles = lib["derived"]["cpu_cycles"]
        sw_inst = lib["derived"]["inst_retired"]
        results["sweep"] = {
            "rc": rc_sw, "dir": str(sweep_dir),
            "cpu_cycles": sw_cycles, "inst_retired": sw_inst,
            "ipc": lib["derived"]["ipc"], "cpi": lib["derived"]["cpi"],
            "cpu_ghz": lib["derived"]["cpu_ghz_over_group1_time_running"],
            "per_group_confidence": lib["meta"]["per_group_confidence"],
            "manifest": str(sweep_dir / "manifest.json"),
        }
        comparison = []
        for name, ref, got in (
            ("cycles", results["perf"]["cycles"], sw_cycles),
            ("instructions", results["perf"]["instructions"], sw_inst),
        ):
            rel = abs(got - ref) / ref if ref else None
            comparison.append({
                "metric": name, "perf": ref, "kperfx": got,
                "rel_err": rel,
                "measured": rel is not None,
                "pass_5pct": bool(rel is not None and rel <= 0.05),
            })
            say(f"compare {name}: perf={ref} kperfx={got} "
                f"rel_err={'n/a' if rel is None else f'{rel:.4%}'} "
                f"pass={comparison[-1]['pass_5pct']}")
        results["comparison"] = comparison

        # 5. full 9-group short run (all-green confidence check)
        if args.selftest_full:
            full_dir = base / f"full9-w{args.selftest_full_window:g}"
            rc_f, out_f, err_f, _ = run_with_timeout(
                [sys.executable, str(SCRIPT_PATH),
                 "--tid", str(tid),
                 "--window", f"{args.selftest_full_window:g}",
                 "--groups", "1-9", "--tag", f"selftest-full-{stamp}",
                 "--outdir", str(full_dir), "--kperfx-python", str(kperfx_python)],
                9 * (args.selftest_full_window + 45) + 120)
            (base / "full9_stdout.txt").write_text(out_f + err_f, encoding="utf-8")
            man = json.loads((full_dir / "manifest.json").read_text())
            say(f"full9 rc={rc_f} all_confident={man['all_confident']} "
                f"min_conf={man['min_confidence']} wall={man['wall_s']}s")
            results["full9"] = {
                "rc": rc_f, "dir": str(full_dir),
                "all_confident": man["all_confident"],
                "min_confidence": man["min_confidence"],
                "per_group_confidence": {
                    g: v["confidence"] for g, v in man["per_group"].items()
                },
                "low_confidence_groups": man["low_confidence_groups"],
                "zero_counting_events": man["zero_counting_events"],
                "wall_s": man["wall_s"],
            }
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        time.sleep(0.2)

    (base / "selftest.json").write_text(json.dumps(results, indent=2),
                                        encoding="utf-8")
    ok = all(row["pass_5pct"] for row in results.get("comparison", []))
    if not all(row.get("measured") for row in results.get("comparison", [])):
        say("verdict caveat: at least one metric had no perf reference")
    if args.selftest_full and "full9" in results:
        ok = ok and bool(results["full9"]["all_confident"])
    (base / "selftest.json").write_text(json.dumps(dict(results, verdict_ok=ok),
                                                   indent=2), encoding="utf-8")
    say(f"selftest artifacts in {base}")
    say(f"selftest verdict: {'PASS' if ok else 'FAIL'}")
    log.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# argument parsing / dispatch
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="libkperfx 920b_topdown_full sweep against one live host TID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--tid", type=int, help="host thread id to measure")
    ap.add_argument("--pid", type=int, default=0,
                    help="optional host pid owning --tid (recorded + validated)")
    ap.add_argument("--window", type=float, default=5.0,
                    help="seconds per preset group")
    ap.add_argument("--groups", default="1-9",
                    help="preset groups, e.g. 1-9 / 1,3,7 / 1-3,7,8")
    ap.add_argument("--tag", default="pmu", help="label stored in the artifacts")
    ap.add_argument("--outdir", help="output directory")
    ap.add_argument("--preset-prefix", default=DEFAULT_PRESET_PREFIX,
                    help="preset family prefix; the group number is appended")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                    help="confidence below this is flagged (and retried)")
    ap.add_argument("--retries", type=int, default=1,
                    help="extra attempts for a failed / low-confidence group")
    ap.add_argument("--force", action="store_true",
                    help="start even if another PMU collector looks active")
    ap.add_argument("--kperfx-python", default=None,
                    help="directory holding kperfx.py (default: repo tools/)")
    ap.add_argument("--no-sudo", dest="sudo", action="store_false",
                    help="do not wrap the PMU capture in sudo -n")
    ap.add_argument("--sudo", dest="sudo", action="store_true",
                    help="wrap the PMU capture in sudo -n (default unless root)")
    ap.add_argument("--list-groups", action="store_true",
                    help="print every preset group (events + role + formula)")
    ap.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--group", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--selftest", action="store_true",
                    help="run the PMU-vs-perf reference selftest")
    ap.add_argument("--selftest-cpu", type=int, default=200,
                    help="CPU for the selftest workload (120-159 is forbidden)")
    ap.add_argument("--selftest-kind", default="alu",
                    help="workload_gen kind for the load (alu, l2, l3, dram, "
                         "branch, mixed, ...)")
    ap.add_argument("--selftest-window", type=float, default=10.0,
                    help="seconds used for BOTH the perf reference and the sweep")
    ap.add_argument("--selftest-dir", default=None,
                    help="artifact dir (default data/profiles/pmu-selftest-<stamp>)")
    ap.add_argument("--selftest-full", dest="selftest_full",
                    action="store_true", default=True,
                    help="also run the 9-group all-green check")
    ap.add_argument("--no-selftest-full", dest="selftest_full",
                    action="store_false",
                    help="skip the extra 9-group run")
    ap.add_argument("--selftest-full-window", type=float, default=3.0,
                    help="window for the extra 9-group run")
    ap.set_defaults(sudo=True)
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.list_groups:
        return list_groups_main(args)
    if args.inner:
        if not args.tid:
            raise SystemExit("--inner needs --tid")
        return inner_main(args)
    if args.selftest:
        return selftest_main(args)
    if not args.tid:
        raise SystemExit("--tid is required (or use --list-groups / --selftest)")
    if not args.outdir:
        raise SystemExit("--outdir is required")
    if args.window <= 0:
        raise SystemExit("--window must be > 0 seconds")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")
    log_path = pathlib.Path(args.outdir).resolve() / "pmu_sweep.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        return Sweep(args, log).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
