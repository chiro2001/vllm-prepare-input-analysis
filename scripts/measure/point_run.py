#!/usr/bin/env python3
"""Run ONE measured point against a live pi-phase service on chip3 (a3-22).

What a "point" is
-----------------
One combination of (workload shape x service config) executed against an
already-running vLLM service.  The service itself is brought up once per
configuration by ``scripts/launch_phase_service.sh --keep`` (2.5 min of
startup), so a matrix run brings up a handful of services and then fires many
points at each of them.

Per point this script records, into ``<outdir>``:

* ``point.json``      - the machine-readable point record (schema pi-point-v1);
* ``requests/``       - per-request client JSON (TTFT / ITL / output tokens);
* ``lite.log``        - the LiteProfiler slice produced inside the window;
* ``phases/``         - analyzer output (``analyze_lite.py``) when available;
* ``perf/``           - optional perf capture (``perf_capture.sh``);
* ``pmu/``            - optional libkperfx sweep (``pmu_tid_sweep.py``).

Instrument exclusivity: ``--perf-s`` and ``--pmu-window`` are mutually
exclusive (perf sampling and PMU counting must never share the run), which the
script enforces.

Usage (on a3-22):
    python3 scripts/measure/point_run.py \
        --container pi-phase-chip3-<run> --service-run-dir runs/<run> \
        --base-url http://127.0.0.1:18100 --model qwen35-08b \
        --run-id <matrix-run-id> --tag a-c64-isl128 \
        --requests 64 --concurrency 64 --prompt-tokens 128 --max-tokens 512 \
        --outdir data/measure/<matrix-run-id>/points/a-c64-isl128
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ecmap  # noqa: E402  (same directory, deliberately a plain module)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_epoch_us() -> int:
    return int(time.time() * 1_000_000)


class Sh:
    """Thin subprocess wrapper that never raises on non-zero exit."""

    @staticmethod
    def run(cmd: list[str], timeout: float | None = None,
            env: dict | None = None, cwd: str | None = None):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=env, cwd=cwd)


class CpuSampler(threading.Thread):
    """Sample on-CPU time / context switches of one host TID at low frequency.

    ``/proc/<pid>/task/<tid>/schedstat`` gives ``sum_exec_runtime`` (ns on CPU)
    and ``sum_wait_time`` (ns off CPU, i.e. runnable-but-not-running + sleep);
    ``/proc/<tid>/stat`` gives utime/stime in USER_HZ ticks.  Both are cheap to
    read, so a 0.25 s cadence costs nothing measurable and gives us the
    "main thread is pinned at ~x CPU" evidence the historical report had to
    reconstruct from ``perf stat task-clock``.
    """

    HZ = os.sysconf("SC_CLK_TCK")

    def __init__(self, pid: int, tid: int, interval: float = 0.25):
        super().__init__(daemon=True, name="cpu-sampler")
        self.pid = pid
        self.tid = tid
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def _read_one(self) -> dict | None:
        base = f"/proc/{self.pid}/task/{self.tid}"
        sched = ecmap.read_text(base + "/schedstat")
        stat = ecmap.read_text(base + "/stat")
        status = ecmap.read_text(base + "/status")
        if sched is None or stat is None:
            return None
        sched_fields = sched.split()
        try:
            exec_ns = int(sched_fields[0])
            wait_ns = int(sched_fields[1])
        except (IndexError, ValueError):
            return None
        utime = stime = None
        try:
            rest = stat.rsplit(")", 1)[1].split()
            utime, stime = int(rest[11]), int(rest[12])
        except (IndexError, ValueError):
            pass
        vol = nonvol = None
        if status:
            for line in status.splitlines():
                if line.startswith("voluntary_ctxt_switches:"):
                    vol = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches:"):
                    nonvol = int(line.split()[1])
        return {
            "t_mono": time.monotonic(),
            "t_utc": now_utc(),
            "exec_ns": exec_ns,
            "wait_ns": wait_ns,
            "utime_ticks": utime,
            "stime_ticks": stime,
            "vol_ctxt": vol,
            "nonvol_ctxt": nonvol,
        }

    def run(self) -> None:  # noqa: D102
        while not self._stop.is_set():
            sample = self._read_one()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=5)
        if not self.samples:
            return {"n_samples": 0}
        first, last = self.samples[0], self.samples[-1]
        wall = last["t_mono"] - first["t_mono"]
        ticks = (lambda a, b: None if a is None or b is None else (b - a) / self.HZ * 1e3)
        return {
            "n_samples": len(self.samples),
            "wall_s": wall,
            "exec_ns_delta": last["exec_ns"] - first["exec_ns"],
            "wait_ns_delta": last["wait_ns"] - first["wait_ns"],
            "utime_delta_ms": ticks(first["utime_ticks"], last["utime_ticks"]),
            "stime_delta_ms": ticks(first["stime_ticks"], last["stime_ticks"]),
            "vol_ctxt_delta": (None if first["vol_ctxt"] is None or last["vol_ctxt"] is None
                               else last["vol_ctxt"] - first["vol_ctxt"]),
            "nonvol_ctxt_delta": (None if first["nonvol_ctxt"] is None
                                  or last["nonvol_ctxt"] is None
                                  else last["nonvol_ctxt"] - first["nonvol_ctxt"]),
            "on_cpu_ratio": (last["exec_ns"] - first["exec_ns"]) / (wall * 1e9)
                            if wall > 0 else None,
            "samples": self.samples,
        }


def http_post(url: str, timeout: float = 300.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


METRIC_KEEP = (
    # token / request accounting (exact scheduled work, better than prompt guesses)
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:request_success_total", "vllm:request_prompt_tokens_sum",
    "vllm:request_generation_tokens_sum",
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc", "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    # latency histograms (sum + count => mean)
    "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum", "vllm:e2e_request_latency_seconds_count",
    "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_count",
    # speculative decoding (MTP arms)
    "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    # engine step counters, when this build exports them
    "vllm:engine_step_total", "vllm:num_scheduled_tokens_total",
)


def scrape_metrics(base_url: str) -> dict[str, float]:
    """Read /metrics and keep the families we understand (best effort)."""
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=20) as resp:
            text = resp.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return out
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name or not value:
            continue
        family = name.split("{", 1)[0]
        if family not in METRIC_KEEP:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


def metrics_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    delta = {}
    for key, value in after.items():
        if key in before:
            delta[key] = value - before[key]
        else:
            delta[key] = value
    return {"before": before, "after": after, "delta": delta}


def file_size(path: pathlib.Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def service_metadata(run_dir: pathlib.Path | None) -> dict:
    if not run_dir:
        return {}
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    keep = ("run_id", "container", "image", "model", "server", "hardware", "timing")
    return {k: data[k] for k in keep if k in data}


def container_serve_spec(container: str) -> dict:
    """Read the container's real command line and pull out the serve flags.

    The launcher manifest records most server settings, but the flags passed
    through ``--extra-serve-args`` (MTP/speculative config, extended cudagraph
    capture sizes, ...) are only visible on the container itself.  Recording
    them here is what lets the analysis tell an MTP-on point from an MTP-off
    one from the data rather than from the tag name.
    """
    proc = Sh.run(["sudo", "-n", "docker", "inspect", "-f", "{{json .Config.Cmd}}",
                   container])
    out: dict = {"cmd": None, "speculative_config": None, "serve_flags": {}}
    if proc.returncode != 0:
        out["error"] = proc.stderr.strip()[-300:]
        return out
    try:
        cmd = json.loads(proc.stdout.strip() or "null")
    except json.JSONDecodeError as exc:
        out["error"] = f"cannot parse docker inspect output: {exc}"
        return out
    if not isinstance(cmd, list):
        out["error"] = "Config.Cmd is not a list"
        return out
    out["cmd"] = cmd
    flags: dict = {}
    for index, token in enumerate(cmd):
        if not str(token).startswith("--"):
            continue
        value = cmd[index + 1] if index + 1 < len(cmd) else None
        if value is not None and str(value).startswith("--"):
            value = None
        flags[str(token)] = value
    out["serve_flags"] = flags
    spec_raw = flags.get("--speculative-config")
    if spec_raw:
        try:
            spec = json.loads(spec_raw)
        except json.JSONDecodeError:
            spec = {"raw": spec_raw}
        out["speculative_config"] = spec
    comp_raw = flags.get("--compilation-config")
    if comp_raw:
        try:
            out["compilation_config"] = json.loads(comp_raw)
        except json.JSONDecodeError:
            out["compilation_config"] = {"raw": comp_raw}
    return out


class BackgroundCapture(threading.Thread):
    """Runs perf or libkperfx in the background, after an optional delay."""

    def __init__(self, name: str, cmd: list[str], outdir: pathlib.Path, delay: float):
        super().__init__(daemon=True, name=f"capture-{name}")
        self.name = name
        self.cmd = cmd
        self.outdir = outdir
        self.delay = delay
        self.rc: int | None = None
        self.stdout = ""
        self.stderr = ""
        self.started_mono: float | None = None
        self.finished_mono: float | None = None

    def run(self) -> None:  # noqa: D102
        if self.delay > 0:
            time.sleep(self.delay)
        self.started_mono = time.monotonic()
        try:
            proc = subprocess.run(self.cmd, capture_output=True, text=True)
            self.rc = proc.returncode
            self.stdout = proc.stdout
            self.stderr = proc.stderr
        except Exception as exc:  # noqa: BLE001
            self.rc = -1
            self.stderr = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished_mono = time.monotonic()

    def report(self) -> dict:
        return {
            "cmd": self.cmd,
            "rc": self.rc,
            "delay_s": self.delay,
            "window_s": (None if self.started_mono is None or self.finished_mono is None
                         else self.finished_mono - self.started_mono),
            "started_utc": (None if self.started_mono is None else now_utc()),
            "stdout_tail": self.stdout[-4000:],
            "stderr_tail": self.stderr[-4000:],
        }


def maybe_analyze(lite_slice: pathlib.Path, outdir: pathlib.Path,
                  analyzer: pathlib.Path | None) -> dict:
    """Run the lite.log analyzer if it is present next to this script.

    Two denominators are produced on purpose (see ANALYZER_NOTES.md §5):

    * ``phases/summary.json``   - numerator over the ``Step:Model`` dispatch
      scope (the historical, comparable-to-0.26-lite-trace口径);
    * ``phases_loop/summary.json`` - numerator over the whole engine loop
      (``Step:Schedule`` anchored, ``--window-mode next``), which is the better
      proxy for "what fraction of the step period did prepare input eat".

    The historical report's ``prepare_input_share_of_step_model = 100.4%`` is an
    artifact of a broken anchor window; both numbers above are reported so the
    difference is visible instead of silent.
    """
    if analyzer is None or not analyzer.is_file():
        return {"available": False, "reason": "analyze_lite.py not found"}
    phases = outdir / "phases"
    phases.mkdir(parents=True, exist_ok=True)
    proc = Sh.run([sys.executable, str(analyzer), str(lite_slice),
                   "--csv", str(phases / "steps.csv"),
                   "--json", str(phases / "summary.json"),
                   "--txt", str(phases / "summary.txt")], timeout=600)
    result = {"available": proc.returncode == 0, "rc": proc.returncode,
              "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    summary = phases / "summary.json"
    if summary.is_file():
        try:
            result["summary"] = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    loop = outdir / "phases_loop"
    loop.mkdir(parents=True, exist_ok=True)
    proc2 = Sh.run([sys.executable, str(analyzer), str(lite_slice),
                    "--csv", str(loop / "steps.csv"),
                    "--json", str(loop / "summary.json"),
                    "--txt", str(loop / "summary.txt"),
                    "--step-scope", "Step:Schedule", "--window-mode", "next"],
                   timeout=600)
    result["loop_denominator"] = {"rc": proc2.returncode}
    summary2 = loop / "summary.json"
    if summary2.is_file():
        try:
            result["loop_summary"] = json.loads(summary2.read_text(encoding="utf-8"))
            result["loop_denominator"]["ok"] = True
        except (OSError, json.JSONDecodeError):
            result["loop_denominator"]["ok"] = False
    return result


def maybe_post_process_profiles(outdir: pathlib.Path, lite_slice: pathlib.Path,
                                scope: str = "prepare input") -> dict:
    """Turn the raw perf capture into scope-attributed evidence.

    Runs immediately after the capture **while the target container is still
    alive**, because CPython's perf-trampoline map (``/tmp/perf-<tid>.map``) is
    what makes Python frames resolvable at all: once the container is gone,
    ``perf script`` can only print raw addresses (PERF_SYMBOLS.md §6).

    Two products per perf point:

    * ``perf/scope-attr/``   - samples split by LiteProfiler scope (historic
      hole G1) with a flame graph rooted at ``prepare input``;
    * ``perf/breakdown/``    - inclusive/children overhead + caller/callee trees
      for the ``builder.build()`` call path (the headline hotspot question).
    """
    result: dict = {}
    perf_dir = outdir / "perf"
    perf_data = perf_dir / "perf.data"
    if not perf_data.is_file():
        return {"available": False, "reason": "no perf.data"}

    attr_dir = perf_dir / "scope-attr"
    attr = Sh.run([sys.executable, str(HERE / "scope_attribution.py"),
                   "--perf-data", str(perf_data), "--lite-log", str(lite_slice),
                   "--scope", scope, "--outdir", str(attr_dir)], timeout=3600)
    result["scope_attribution"] = {"rc": attr.returncode,
                                   "stdout_tail": attr.stdout[-1500:],
                                   "stderr_tail": attr.stderr[-1500:]}
    summary = attr_dir / "summary.json"
    if summary.is_file():
        try:
            doc = json.loads(summary.read_text(encoding="utf-8"))
            result["scope_attribution"].update({
                "samples_in_scope": doc.get("samples_in_scope"),
                "samples_outside_scope": doc.get("samples_outside_scope"),
                "scope_time_fraction": doc.get("implied_scope_time_fraction"),
                "sample_share_in_scope": doc.get("sample_share_in_scope"),
                "leaf_top_in_scope": (doc.get("leaf_top_in_scope") or [])[:15],
                "renders": {k: v.get("ok") for k, v in
                            (doc.get("renders") or {}).items()},
            })
        except (OSError, json.JSONDecodeError):
            pass

    bd_dir = perf_dir / "breakdown"
    bd = Sh.run([sys.executable, str(HERE / "build_breakdown.py"),
                 "--perf-data", str(perf_data), "--outdir", str(bd_dir)],
                timeout=3600)
    result["build_breakdown"] = {"rc": bd.returncode, "stderr_tail": bd.stderr[-800:]}
    bd_summary = bd_dir / "summary.json"
    if bd_summary.is_file():
        try:
            doc = json.loads(bd_summary.read_text(encoding="utf-8"))
            result["build_breakdown"]["found"] = [
                key for key, val in (doc.get("symbols") or {}).items()
                if val.get("found_in_flat")]
            result["build_breakdown"]["not_found"] = [
                key for key, val in (doc.get("symbols") or {}).items()
                if not val.get("found_in_flat")]
        except (OSError, json.JSONDecodeError):
            pass
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--service-run-dir", default=None)
    ap.add_argument("--chip", type=int, default=3)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--group", default="",
                    help="matrix group tag (A=concurrency, B=ISL, C=chunk, D=mixed, "
                         "E=MTP, F=blocksize/churn, R=reproduction)")
    ap.add_argument("--note", default="", help="free-form note stored in the point record")
    # workload
    ap.add_argument("--requests", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat the whole workload N times inside the window")
    ap.add_argument("--client-timeout", type=float, default=1800.0)
    ap.add_argument("--client-cpus", default="120,121")
    # instrumentation
    ap.add_argument("--phase", choices=("on", "off"), default="on")
    ap.add_argument("--perf-s", type=float, default=0.0,
                    help="perf record window in seconds (0 = no perf)")
    ap.add_argument("--perf-delay", type=float, default=3.0)
    ap.add_argument("--perf-freq", type=int, default=999)
    ap.add_argument("--perf-callgraph", default="fp", choices=("fp", "dwarf", "lbr"))
    ap.add_argument("--pmu-window", type=float, default=0.0,
                    help="libkperfx seconds per preset group (0 = no PMU)")
    ap.add_argument("--pmu-delay", type=float, default=3.0)
    ap.add_argument("--pmu-groups", default="1-9")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args(argv)

    if args.perf_s > 0 and args.pmu_window > 0:
        ap.error("--perf-s and --pmu-window are mutually exclusive "
                 "(perf sampling and PMU counting must not share a run)")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    service_run_dir = pathlib.Path(args.service_run_dir) if args.service_run_dir else None

    started = time.time()
    point: dict = {
        "schema": "pi-point-v1",
        "run_id": args.run_id,
        "tag": args.tag,
        "group": args.group,
        "note": args.note,
        "created_utc": now_utc(),
        "host": os.uname().nodename,
        "service": {"container": args.container, "base_url": args.base_url,
                    "chip": args.chip, **service_metadata(service_run_dir)},
        "workload": {
            "requests": args.requests, "concurrency": args.concurrency,
            "prompt_tokens_target": args.prompt_tokens, "max_tokens": args.max_tokens,
            "seed": args.seed, "rounds": args.rounds, "temperature": 0.0,
            "streaming": True,
        },
        "instrumentation": {
            "phase": args.phase,
            "perf": {"seconds": args.perf_s, "delay_s": args.perf_delay,
                     "freq": args.perf_freq, "call_graph": args.perf_callgraph},
            "pmu": {"window_s": args.pmu_window, "delay_s": args.pmu_delay,
                    "groups": args.pmu_groups},
        },
        "errors": [],
    }

    # ------------------------------------------------------------- EC thread
    rc, out, err = Sh.run([sys.executable, str(HERE / "ecmap.py"),
                           "--chip", str(args.chip), "--container", args.container,
                           "--json", str(outdir / "ecmap.json")])
    doc: dict = {}
    try:
        doc = json.loads((outdir / "ecmap.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        point["errors"].append(f"ecmap failed: rc={rc} {err[-400:]}")
    pid, tid = doc.get("ec_host_pid"), doc.get("ec_host_tid")
    if tid is None:
        point["errors"].append("cannot resolve the engine-core host TID; aborting")
        (outdir / "point.json").write_text(json.dumps(point, indent=2, sort_keys=True) + "\n")
        return 2
    point["ec"] = {k: doc.get(k) for k in
                   ("container", "ec_host_pid", "ec_host_tid", "ec_comm",
                    "ec_cpu_affinity", "ec_ns_tid")}

    serve = container_serve_spec(args.container)
    if serve.get("error"):
        point["errors"].append(f"cannot read container serve spec: {serve['error']}")
    point["service"]["serve_cmd"] = serve.get("cmd")
    point["service"]["serve_flags"] = serve.get("serve_flags")
    server = point["service"].setdefault("server", {})
    if isinstance(server, dict):
        spec = serve.get("speculative_config")
        server["speculative_config"] = spec
        if isinstance(spec, dict):
            server["num_speculative_tokens"] = spec.get("num_speculative_tokens")
            server["mtp"] = "on" if spec else "off"
        elif spec is None:
            server["num_speculative_tokens"] = None
            server["mtp"] = "off"
        if serve.get("compilation_config"):
            server["compilation_config"] = serve["compilation_config"]

    sampler = CpuSampler(pid, tid)
    sampler.start()

    # -------------------------------------------------------------- captures
    captures: list[BackgroundCapture] = []
    perf_dir = pmu_dir = None
    if args.perf_s > 0:
        perf_dir = outdir / "perf"
        perf_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["bash", str(HERE / "perf_capture.sh"),
               "--tid", str(tid), "--seconds", f"{args.perf_s:.3f}",
               "--name", args.tag, "--outdir", str(perf_dir),
               "--freq", str(args.perf_freq), "--call-graph", args.perf_callgraph]
        captures.append(BackgroundCapture("perf", cmd, perf_dir, args.perf_delay))
    if args.pmu_window > 0:
        pmu_dir = outdir / "pmu"
        pmu_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / "pmu_tid_sweep.py"),
               "--tid", str(tid), "--window", f"{args.pmu_window:.3f}",
               "--groups", args.pmu_groups, "--tag", args.tag,
               "--outdir", str(pmu_dir)]
        captures.append(BackgroundCapture("pmu", cmd, pmu_dir, args.pmu_delay))
    for cap in captures:
        cap.start()

    # ------------------------------------------------------- profile window
    lite_log = (service_run_dir / "lite-profiler" / "lite.log") if service_run_dir else None
    offset_before = file_size(lite_log) if (lite_log and args.phase == "on") else 0
    point["window"] = {"lite_log": str(lite_log) if lite_log else None,
                       "offset_before_bytes": offset_before}

    if args.phase == "on":
        status, body = http_post(args.base_url.rstrip("/") + "/start_profile")
        point["window"]["start_profile"] = {"http_status": status, "body": body[:500]}
        if status != 200:
            point["errors"].append(f"/start_profile -> HTTP {status}: {body[:200]}")
        time.sleep(1.5)

    # ------------------------------------------------------------- workload
    metrics_before = scrape_metrics(args.base_url)
    client = HERE.parent / "phase_smoke_client.py"
    rounds: list[dict] = []
    wall0 = time.monotonic()
    for index in range(args.rounds):
        rdir = outdir / "requests" / f"r{index:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        cmd = ["taskset", "-c", args.client_cpus, sys.executable, str(client),
               "--base-url", args.base_url, "--model", args.model, "--outdir", str(rdir),
               "--requests", str(args.requests), "--concurrency", str(args.concurrency),
               "--prompt-tokens", str(args.prompt_tokens),
               "--max-tokens", str(args.max_tokens),
               "--seed", str(args.seed + 1000 * index),
               "--tag", f"{args.tag}-r{index}", "--timeout", str(args.client_timeout)]
        t0 = time.monotonic()
        proc = Sh.run(cmd, timeout=args.client_timeout + 120)
        entry = {"round": index, "rc": proc.returncode,
                 "wall_s": time.monotonic() - t0,
                 "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-2500:]}
        summary_path = rdir / "client_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                entry["aggregate"] = summary.get("aggregate", {})
                entry["n_ok"] = summary.get("n_ok")
                entry["n_error"] = summary.get("n_error")
            except (OSError, json.JSONDecodeError) as exc:
                entry["parse_error"] = str(exc)
        rounds.append(entry)
        if proc.returncode != 0:
            point["errors"].append(f"round {index}: client rc={proc.returncode}")
    workload_wall = time.monotonic() - wall0
    point["rounds"] = rounds
    point["workload_wall_s"] = workload_wall
    metrics_after = scrape_metrics(args.base_url)
    point["server_metrics"] = metrics_delta(metrics_before, metrics_after)

    for cap in captures:
        cap.join(timeout=max(10.0, (cap.delay + 60.0)))
    point["captures"] = {cap.name: cap.report() for cap in captures}

    if args.phase == "on":
        status, body = http_post(args.base_url.rstrip("/") + "/stop_profile")
        point["window"]["stop_profile"] = {"http_status": status, "body": body[:500]}
        time.sleep(1.0)

    point["cpu"] = sampler.stop()

    # --------------------------------------------------------- lite.log slice
    if lite_log and args.phase == "on" and lite_log.is_file():
        size_after = file_size(lite_log)
        slice_path = outdir / "lite.log"
        try:
            with open(lite_log, "rb") as src:
                src.seek(offset_before)
                data = src.read()
            # if the logger truncated the file, fall back to the whole file
            truncated = size_after < offset_before
            if truncated:
                data = lite_log.read_bytes()
            slice_path.write_bytes(data)
            point["window"].update({
                "offset_after_bytes": size_after,
                "truncated": truncated,
                "slice_bytes": len(data),
                "slice_rows": data.count(b"\n"),
            })
            if not args.no_analyze:
                point["phases"] = maybe_analyze(
                    slice_path, outdir, HERE / "analyze_lite.py")
        except OSError as exc:
            point["errors"].append(f"lite.log slice failed: {exc}")
    else:
        if args.phase == "on":
            point["errors"].append(f"lite.log missing: {lite_log}")

    # ------------------------------------------------- profile post-process
    # Must happen now: the python-frame maps live in the target container's
    # /tmp and disappear with it.
    if (outdir / "perf" / "perf.data").is_file() and lite_log:
        slice_for_attr = outdir / "lite.log"
        if slice_for_attr.is_file():
            point["profile_postprocess"] = maybe_post_process_profiles(
                outdir, slice_for_attr)

    # --------------------------------------------------------------- finish
    point["wall_s"] = time.time() - started
    point["finished_utc"] = now_utc()
    (outdir / "point.json").write_text(
        json.dumps(point, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    agg = [r.get("aggregate") or {} for r in rounds]
    ttft = [a.get("ttft_s_mean") for a in agg if a.get("ttft_s_mean") is not None]
    tps = [a.get("output_tps_aggregate") for a in agg if a.get("output_tps_aggregate")]
    print(json.dumps({
        "tag": args.tag,
        "rounds": len(rounds),
        "workload_wall_s": round(workload_wall, 3),
        "ec_tid": tid,
        "on_cpu_ratio": point["cpu"].get("on_cpu_ratio"),
        "ttft_s_mean": (sum(ttft) / len(ttft)) if ttft else None,
        "tps_aggregate_mean": (sum(tps) / len(tps)) if tps else None,
        "lite_rows": point.get("window", {}).get("slice_rows"),
        "errors": point["errors"],
    }, indent=2, sort_keys=True))
    return 0 if not point["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
