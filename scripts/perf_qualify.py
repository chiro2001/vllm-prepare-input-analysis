#!/usr/bin/env python3
"""perf_qualify.py - qualify the perf *collection contract* on a3-22.

Runs a controlled, NPU-free synthetic load on the reserved CPU slice and
checks every capability the prepare_input analysis will later depend on:

  * `perf stat` plain counting (user/kernel split),
  * `perf stat` forced multiplexing -> time_enabled/time_running ratio,
  * `perf record -g --call-graph fp`   (C victim, symbolised),
  * `perf record -g --call-graph dwarf` (CPython victim, realistic call tree),
  * `perf report --stdio` / `perf report --header-only` contract,
  * `perf annotate --stdio` source/instruction attribution,
  * `perf script` textual sample contract,
  * `perf record -p PID` thread/process targeting (how the engine core thread
    will be sampled later),
  * libkperfx multiplex confidence as an independent running-ratio probe.

Everything is written under --out (default data/toolchain/raw) and summarised
into data/toolchain/perf-qualification-a3-22.json.

Usage (as root, on a3-22):
    sudo -n python3 scripts/perf_qualify.py --cpus 122-157
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

# NOTE: this script is normally re-executed via `sudo -n`, which resets $HOME to
# /root.  Deriving the project root from $HOME would then write outside the
# allowed trees, so it is derived from this file's location instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get(
    "PREPARE_INPUT_ROOT", os.path.dirname(_HERE)
)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "toolchain", "raw")
DEFAULT_DATADIR = os.path.join(PROJECT_ROOT, "data", "profiles", "toolchain-qualification")
DEFAULT_JSON = os.path.join(
    PROJECT_ROOT, "data", "toolchain", "perf-qualification-a3-22.json"
)
SYNTH_DIR = os.path.join(PROJECT_ROOT, "scripts", "synth")
LIBKPERFX = os.path.join(PROJECT_ROOT, "tools", "libkperfx", "kperfx")

STAT_RE = re.compile(r"^\s*(?P<value>[\d,]+(?:\.\d+)?)\s+(?P<counter>\S+)\s*(?:#.*?)?\s*(?:\(\s*(?P<pct>[\d.]+)%\s*\))?\s*$")
REPORT_RE = re.compile(r"^\s*(?P<pct>\d+\.\d+)%\s+(?P<comm>\S+)\s+(?P<dso>\S+)\s+(?:\[.\]\s*)?(?P<sym>.+?)\s*$")


def sh(cmd, **kw):
    """Run a command, never raise; return (rc, stdout, stderr, wall_s)."""
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, p.stdout, p.stderr, time.monotonic() - t0


def parse_stat(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = STAT_RE.match(line)
        if not m:
            continue
        cnt = m.group("counter")
        try:
            val = float(m.group("value").replace(",", ""))
        except ValueError:
            continue
        out[cnt] = {
            "value": val,
            "raw": m.group("value"),
            "enabled_pct": float(m.group("pct")) if m.group("pct") else None,
        }
    return out


def parse_report(text: str, limit: int = 25) -> list:
    rows = []
    for line in text.splitlines():
        m = REPORT_RE.match(line)
        if m and "kernel.kallsyms" not in line:
            rows.append(
                {
                    "overhead_pct": float(m.group("pct")),
                    "comm": m.group("comm"),
                    "dso": m.group("dso"),
                    "symbol": m.group("sym").strip(),
                }
            )
        if len(rows) >= limit:
            break
    return rows


def run(cmd, log_name, outdir, timeout=600):
    rc, so, se, wall = sh(cmd, timeout=timeout)
    body = f"$ {' '.join(cmd)}\n# rc={rc} wall_s={wall:.3f}\n\n{so}\n----- stderr -----\n{se}\n"
    path = os.path.join(outdir, log_name)
    with open(path, "w") as fh:
        fh.write(body)
    return {"cmd": cmd, "rc": rc, "wall_s": round(wall, 3),
            "stdout": so, "stderr": se, "log": path}


def cpu_occupancy(cpus: list, sample_s: float = 2.0) -> dict:
    """Sampled per-CPU busy fraction from /proc/stat deltas."""
    want = list(range(cpus[0], cpus[-1] + 1)) if len(cpus) > 2 and cpus[1] == cpus[0] + 1 else cpus

    def snap():
        d = {}
        with open("/proc/stat") as fh:
            for line in fh:
                if not line.startswith("cpu") or line.startswith("cpu "):
                    continue
                f = line.split()
                idx = int(f[0][3:])
                vals = [int(x) for x in f[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                d[idx] = (sum(vals), idle)
        return d

    a = snap()
    time.sleep(sample_s)
    b = snap()
    busy = {}
    for idx in want:
        if idx not in a or idx not in b:
            continue
        dt = b[idx][0] - a[idx][0]
        di = b[idx][1] - a[idx][1]
        if dt > 0:
            busy[idx] = round(100.0 * (dt - di) / dt, 2)
    vals = list(busy.values())
    return {
        "cpus": f"{want[0]}-{want[-1]}" if len(want) > 1 else str(want[0]),
        "sample_s": sample_s,
        "per_cpu_busy_pct": busy,
        "max_busy_pct": max(vals) if vals else None,
        "mean_busy_pct": round(sum(vals) / len(vals), 2) if vals else None,
    }


def parse_cpu_spec(spec: str) -> list:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def supported_perf_events() -> set:
    """`perf list` -> set of bare PMU event names (handles Kunpeng naming)."""
    rc, so, se, _ = sh(["perf", "list", "--no-desc", "--no-group-desc"])
    if rc != 0:
        rc, so, se, _ = sh(["perf", "list"])
    names = set()
    for line in (so or se).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or " " in line.split("[")[0].strip():
            pass
        m = re.match(r"^([A-Za-z0-9_:./-]+)", line)
        if m:
            names.add(m.group(1))
    return names


def build_multiplex_set(wanted: list, supported: set, need: int = 9) -> tuple:
    """Return (events, dropped).  Only events perf itself advertises are used,
    so `perf stat` cannot fail with 'invalid or unsupported event'."""
    keep = [e for e in wanted if e in supported or ":" in e]
    dropped = [e for e in wanted if e not in supported and ":" not in e]
    return keep[:need] if len(keep) >= need else keep, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", default="122-157")
    ap.add_argument("--pin-cpu", type=int, default=None,
                    help="single CPU for the pinned single-thread runs")
    ap.add_argument("--duration-ms", type=int, default=2500)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--datadir", default=DEFAULT_DATADIR,
                    help="where perf.data files go; they stay on the target and are "
                         "never part of the deliverable (dwarf captures are ~100 MB)")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("perf_qualify: re-executing under sudo -n", file=sys.stderr)
        argv = [os.path.abspath(__file__), *sys.argv[1:]]
        os.execvp("sudo", ["sudo", "-n", sys.executable, *argv])  # type: ignore[arg-type]

    cpus = parse_cpu_spec(args.cpus)
    pin = args.pin_cpu if args.pin_cpu is not None else cpus[0]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.datadir, exist_ok=True)
    raw = {}
    result = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {},
        "cpus": {"slice": args.cpus, "pin_cpu": pin, "list": cpus},
        "artifacts": {},
        "checks": {},
    }

    for key, cmd in {
        "uname": ["uname", "-a"],
        "hostname": ["hostname"],
        "perf_version": ["perf", "--version"],
        "paranoid": ["cat", "/proc/sys/kernel/perf_event_paranoid"],
        "nproc": ["nproc"],
        "cpufreq_governor": ["bash", "-lc",
                             "cat /sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor 2>/dev/null || echo n/a" % pin],
        "msprof_residue": ["bash", "-lc",
                           "ps -eo pid,comm,args --no-headers | grep -Ei 'msprof|devkit|mindstudio' | grep -v grep || echo NONE"],
        "perf_residue": ["bash", "-lc",
                         "ps -eo pid,comm,args --no-headers | grep -E '(^|/)perf (record|stat|top|trace)' | grep -v grep || echo NONE"],
    }.items():
        rc, so, se, _ = sh(cmd)
        result["host"][key] = (so or se).strip().splitlines()[:4]

    # ---- build the synthetic victim -------------------------------------
    binpath = os.path.join(args.datadir, "probe_load")
    if not args.skip_build:
        raw["build"] = run(
            ["gcc", "-O2", "-g", "-fno-omit-frame-pointer", "-std=gnu11",
             "-o", binpath, os.path.join(SYNTH_DIR, "probe_load.c"), "-lpthread", "-lm"],
            "01-build-probe-load.log", args.out)
        result["artifacts"]["probe_load_bin"] = binpath
    result["checks"]["build_probe_load"] = {
        "ok": os.path.exists(binpath),
        "binary": binpath,
        "sha256": sh(["sha256sum", binpath])[1].split()[0] if os.path.exists(binpath) else None,
    }
    if not os.path.exists(binpath):
        result["verdicts"] = {"build_probe_load": "FAIL",
                              "reason": "compiler failed; see 01-build-probe-load.log"}
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        print("ABORT: victim binary missing:", binpath, file=sys.stderr)
        print("json:", args.json)
        return 1

    def probe(*extra, dur=None):
        return ["taskset", "-c", str(pin), binpath,
                "--duration-ms", str(dur or args.duration_ms), *extra]

    # ---- 1. perf stat: plain, user/kernel split --------------------------
    raw["stat_basic"] = run(
        ["perf", "stat", "-e",
         "task-clock,cycles,instructions,branches,branch-misses,cycles:u,cycles:k",
         "--", *probe()],
        "02-perf-stat-basic.log", args.out)
    basic = parse_stat(raw["stat_basic"]["stderr"] + raw["stat_basic"]["stdout"])
    result["checks"]["perf_stat_basic"] = {
        "ok": raw["stat_basic"]["rc"] == 0 and "cycles" in basic and basic["cycles"]["value"] > 0,
        "counters": basic,
        "ipc_from_stat": (
            round(basic["instructions"]["value"] / basic["cycles"]["value"], 4)
            if basic.get("cycles", {}).get("value") and basic.get("instructions", {}).get("value")
            else None
        ),
    }

    # ---- 2. perf stat: forced multiplexing ------------------------------
    supported = supported_perf_events()
    wanted = ["cycles", "instructions", "branches", "branch-misses",
              "L1-dcache-loads", "L1-dcache-load-misses", "L1-dcache-stores",
              "L1-icache-loads", "L1-icache-load-misses",
              "dTLB-loads", "dTLB-load-misses", "iTLB-loads", "iTLB-load-misses",
              "cache-references", "cache-misses", "bus-cycles", "ref-cycles",
              "stalled-cycles-frontend", "stalled-cycles-backend",
              "cpu-clock", "task-clock", "context-switches", "page-faults"]
    many, dropped = build_multiplex_set(wanted, supported, need=9)
    result["checks"]["perf_stat_multiplex_event_availability"] = {
        "requested": wanted, "used": many, "dropped_not_advertised": dropped,
        "n_supported_seen": len(supported),
    }
    raw["stat_mux"] = run(
        ["perf", "stat", "-e", ",".join(many), "--", *probe(dur=1500)],
        "03-perf-stat-multiplex.log", args.out)
    mux = parse_stat(raw["stat_mux"]["stderr"] + raw["stat_mux"]["stdout"])
    enabled = {k: v["enabled_pct"] for k, v in mux.items() if v["enabled_pct"] is not None}
    result["checks"]["perf_stat_multiplex"] = {
        "ok": len(mux) > 0,
        "n_requested": len(many),
        "n_reported": len(mux),
        "enabled_pct": enabled,
        "min_enabled_pct": min(enabled.values()) if enabled else None,
        "note": "enabled_pct 就是 time_running/time_enabled，perf 用 (xx.xx%) 标注",
    }

    # ---- 3. perf record on the C victim --------------------------------
    data_c = os.path.join(args.datadir, "perf-c.data")
    raw["record_c"] = run(
        ["perf", "record", "-o", data_c, "-F", "999", "-g", "--call-graph", "fp",
         "--", *probe(dur=3000)],
        "04-perf-record-c.log", args.out)
    raw["report_c"] = run(["perf", "report", "--stdio", "-i", data_c, "--no-children",
                           "--percent-limit", "0.2", "-g", "none"],
                          "05-perf-report-c.log", args.out)
    raw["header_c"] = run(["perf", "report", "--header-only", "-i", data_c],
                          "06-perf-header-c.log", args.out)
    raw["script_c"] = run(["bash", "-lc", f"perf script -i {data_c} | head -40"],
                          "07-perf-script-c.log", args.out)
    raw["annotate_c"] = run(
        ["bash", "-lc",
         f"perf annotate --stdio -i {data_c} -s probe_hot_branch 2>&1 | head -60"],
        "08-perf-annotate-c.log", args.out)
    raw["tools"] = run(
        ["bash", "-lc", "for t in objdump addr2line nm readelf gdb; do "
                       "printf '%-10s %s\\n' \"$t\" \"$(command -v $t || echo MISSING)\"; done"],
        "08b-annotation-toolchain.log", args.out)
    top_c = parse_report(raw["report_c"]["stdout"])
    ann = raw["annotate_c"]["stdout"]
    result["checks"]["perf_record_c"] = {
        "ok": raw["record_c"]["rc"] == 0 and os.path.exists(data_c),
        "data_bytes": os.path.getsize(data_c) if os.path.exists(data_c) else 0,
        "report_top": top_c,
        "report_nonempty": bool(top_c),
        "annotate_ok": "probe_hot_branch" in ann and len(ann.splitlines()) > 5,
        "annotate_source_lines": sum(
            1 for ln in ann.splitlines() if re.match(r"^\s*[\d.]+ :", ln)
        ),
        "annotate_instruction_lines": sum(
            1 for ln in ann.splitlines() if re.match(r"^\s*[\d.]+ :\s+[0-9a-f]+:", ln)
        ),
        "annotation_toolchain": raw["tools"]["stdout"].strip().splitlines(),
        "script_lines": len(raw["script_c"]["stdout"].splitlines()),
        "header_has_cmdline": "cmdline" in raw["header_c"]["stdout"],
    }
    result["artifacts"]["perf_c_data"] = data_c

    # ---- 4. perf record on the CPython victim --------------------------
    py = shutil.which("python3") or "/usr/bin/python3"
    data_py = os.path.join(args.datadir, "perf-py.data")
    raw["record_py"] = run(
        ["perf", "record", "-o", data_py, "-F", "999", "-g",
         "--call-graph", "dwarf,32768",
         "--", "taskset", "-c", str(pin), py,
         os.path.join(SYNTH_DIR, "probe_load.py"),
         "--duration-ms", "3000", "--reqs", "64", "--tokens", "512"],
        "09-perf-record-py.log", args.out)
    raw["report_py"] = run(["perf", "report", "--stdio", "-i", data_py, "--no-children",
                            "--percent-limit", "0.2", "-g", "none"],
                           "10-perf-report-py.log", args.out)
    raw["script_py"] = run(["bash", "-lc", f"perf script -i {data_py} | head -60"],
                           "11-perf-script-py.log", args.out)
    top_py = parse_report(raw["report_py"]["stdout"])
    py_syms = [r["symbol"] for r in top_py]
    result["checks"]["perf_record_python"] = {
        "ok": raw["record_py"]["rc"] == 0 and bool(top_py),
        "data_bytes": os.path.getsize(data_py) if os.path.exists(data_py) else 0,
        "report_top": top_py,
        "python_frames": [s for s in py_syms if "python" in s.lower() or s.startswith(("Py", "_Py", "builtin_", "list_", "dict_"))][:10],
        "script_lines": len(raw["script_py"]["stdout"].splitlines()),
    }
    result["artifacts"]["perf_py_data"] = data_py

    # ---- 5. perf record -p PID (how the engine thread is sampled) -------
    p = subprocess.Popen(["taskset", "-c", str(pin), binpath, "--duration-ms", "3000"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.4)
    data_pid = os.path.join(args.datadir, "perf-pid.data")
    raw["record_pid"] = run(
        ["perf", "record", "-o", data_pid, "-F", "499", "-g", "--call-graph", "fp",
         "-p", str(p.pid), "--", "sleep", "1.5"],
        "12-perf-record-pid.log", args.out)
    p.wait(timeout=30)
    raw["report_pid"] = run(["perf", "report", "--stdio", "-i", data_pid, "--no-children"],
                            "13-perf-report-pid.log", args.out)
    top_pid = parse_report(raw["report_pid"]["stdout"])
    result["checks"]["perf_record_pid"] = {
        "ok": raw["record_pid"]["rc"] == 0 and bool(top_pid),
        "target_pid": p.pid,
        "report_top": top_pid[:10],
    }

    # ---- 6. libkperfx multiplex confidence (independent running ratio) --
    if os.path.exists(LIBKPERFX):
        ev = ("CPU_CYCLES,INST_RETIRED,INST_SPEC,BR_MIS_PRED,BR_PRED,"
              "STALL_FRONTEND,STALL_BACKEND,L1D_TLB,L1I_TLB,L1D_CACHE,"
              "L1D_CACHE_REFILL,L1I_CACHE_REFILL,LD_RETIRED,ST_RETIRED,"
              "MEM_ACCESS,TOTAL_RESOURCE_STALL")
        raw["kperfx_mux"] = run(
            ["bash", "-lc",
             f"KPERFX_EVENTS={ev} KPERFX_MULTIPLEX=1 KPERFX_FORMAT=json "
             f"{LIBKPERFX} run -- taskset -c {pin} {binpath} --duration-ms 1500"],
            "14-kperfx-multiplex.log", args.out)
        txt = raw["kperfx_mux"]["stdout"] + raw["kperfx_mux"]["stderr"]
        def jnum(key):
            m = re.search(r'"%s"\s*:\s*([\d.]+)' % key, txt)
            return float(m.group(1)) if m else None

        te, tr, conf = jnum("time_enabled"), jnum("time_running"), jnum("confidence")
        deltas = dict(re.findall(r'"name":"(\w+)".*?"delta":(\d+)', txt))
        result["checks"]["libkperfx_mux"] = {
            "ok": raw["kperfx_mux"]["rc"] == 0,
            "requested_events": len(ev.split(",")),
            "hw_group_limit": 8,
            "instances": jnum("instances"),
            "time_enabled": te,
            "time_running": tr,
            "confidence": conf,
            "running_over_enabled": round(tr / te, 4) if te and tr else None,
            "useronly_forced": '"useronly_forced": true' in txt,
            "zero_delta_events": [k for k, v in deltas.items() if v == "0"],
            "note": "confidence == time_running/time_enabled == 8/16 for 16 events on an "
                    "8-counter PMU; this is the independent cross-check of perf's (xx.xx%) "
                    "enabled column.",
        }

    # ---- 7. slice occupancy (host noise on the reserved CPUs) -----------
    result["checks"]["cpu_slice_idle_before"] = cpu_occupancy(cpus, sample_s=2.0)
    hot = subprocess.Popen(["taskset", "-c", f"{cpus[0]}-{cpus[-1]}", binpath,
                            "--duration-ms", "4000", "--threads", "8"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    occ = cpu_occupancy(cpus, sample_s=2.0)
    hot.wait(timeout=60)
    result["checks"]["cpu_slice_during_synthetic_load"] = occ

    # ---- verdicts -------------------------------------------------------
    c = result["checks"]
    result["verdicts"] = {
        "perf_stat": "PASS" if c["perf_stat_basic"]["ok"] else "FAIL",
        "perf_multiplex_running_ratio": "PASS" if c["perf_stat_multiplex"]["min_enabled_pct"] is not None else "FAIL",
        "perf_record_fp_c": "PASS" if c["perf_record_c"]["ok"] and c["perf_record_c"]["report_nonempty"] else "FAIL",
        "perf_annotate": "PASS" if c["perf_record_c"]["annotate_ok"] else "FAIL",
        "perf_script": "PASS" if c["perf_record_c"]["script_lines"] > 0 else "FAIL",
        "perf_record_dwarf_python": "PASS" if c["perf_record_python"]["ok"] else "FAIL",
        "perf_record_pid": "PASS" if c["perf_record_pid"]["ok"] else "FAIL",
        "libkperfx_multiplex": "PASS" if c.get("libkperfx_mux", {}).get("ok") else "SKIP/FAIL",
    }
    result["raw_logs"] = {k: v["log"] for k, v in raw.items()}

    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(json.dumps(result["verdicts"], indent=2))
    print("json:", args.json)
    return 0 if all(v == "PASS" for v in result["verdicts"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
