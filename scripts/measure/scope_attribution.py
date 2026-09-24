#!/usr/bin/env python3
"""Attribute perf samples to LiteProfiler scopes using wall-clock time.

Why this exists (historical hole G1)
------------------------------------
``perf record -t <tid>`` samples the thread, not the program's scopes, so a
flat report can only say "this thread is front-end bound" -- it cannot say
"the samples inside ``prepare input`` are X".  The historical report had to flag
this as a data hole and the project plan lists it as **G1**.

Both sides carry absolute time, so the hole is closable without changing the
instrumented program:

* LiteProfiler rows are ``<scope>|<dur us>|<start epoch us>|<tid>|<pid>``
  (``time.time_ns() // 1000`` at scope entry);
* ``perf script --ns -F time`` stamps every sample with nanoseconds since boot,
  and ``/proc/stat``'s ``btime`` converts that to epoch time.

So: put every sample into the scope window that contains it, and emit a folded
stack file per scope.  The result is a flame graph whose root *is*
``prepare input`` -- the evidence the parent task asked for.

Two caveats, reported explicitly rather than hidden:

1. A perf sample belongs to the sample's *leaf* moment; scopes are interval
   based, so nesting is resolved innermost-first and samples outside every
   window go to ``<outside>``.
2. perf's clock is ``CLOCK_MONOTONIC`` derived and the LiteProfiler clock is
   ``time.time()``; the conversion is exact to the extent ``btime`` is, i.e.
   sub-millisecond on this host (verified by the calibration block this script
   prints: the count of samples landing in *no* scope should be small, and the
   implied idle fraction should match the analyzer's idle phase).

Usage (on a3-22):
    python3 scripts/measure/scope_attribution.py \
        --perf-data data/.../perf/perf.data \
        --lite-log  data/.../lite.log \
        --scope "prepare input" --outdir data/.../scope-attr \
        [--tid N] [--tools-bin tools/bin] [--top 40]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

# perf script prints comm, then "<pid>/<tid>" -- or just "<tid>" when the two
# are equal (the common case for the engine-core main thread).
SAMPLE_RE = re.compile(
    r"^(?P<comm>\S+)\s+(?:(?P<pid>\d+)/)?(?P<tid>\d+)\s+"
    r"(?P<time>\d+\.\d+):\s*(?P<rest>.*)$")
EVENT_RE = re.compile(r"^\s*(?P<count>\d+)\s+(?P<event>[A-Za-z0-9_:./-]+):\s*$")
FRAME_RE = re.compile(r"^\s+(?P<addr>[0-9a-fA-F]+)\s+(?P<sym>.*?)\s*\((?P<dso>[^)]*)\)\s*$")
FRAME_IP_RE = re.compile(r"^\s+(?P<addr>[0-9a-fA-F]+)\s*$")


def sh(cmd: list[str], timeout: float = 3600, sudo: bool = True):
    if sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def boot_epoch() -> float:
    with open("/proc/stat", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("btime "):
                return float(line.split()[1])
    raise RuntimeError("no btime in /proc/stat")


def load_lite_scopes(path: pathlib.Path, tid: int | None) -> list[dict]:
    rows: list[dict] = []
    tid_counts: Counter[int] = Counter()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 5:
                continue
            name, dur, start, row_tid, _pid = parts
            try:
                entry = {"name": name, "dur_us": float(dur),
                         "start_us": int(start), "tid": int(row_tid)}
            except ValueError:
                continue
            if entry["name"].startswith("phase: "):
                continue
            rows.append(entry)
            tid_counts[entry["tid"]] += 1
    if tid is None and tid_counts:
        tid = tid_counts.most_common(1)[0][0]
    rows = [r for r in rows if r["tid"] == tid]
    # end time is start+dur, but overlapping/nested scopes are resolved by
    # "smallest containing window wins", so sort by start then duration.
    rows.sort(key=lambda r: (r["start_us"], -r["dur_us"]))
    return rows


def intervals_for(rows: list[dict], name: str) -> list[tuple[int, int]]:
    return [(r["start_us"], r["start_us"] + int(r["dur_us"]))
            for r in rows if r["name"] == name]


class ScopeSweep:
    """Linear sweep that answers 'innermost scope at time t' for sorted samples.

    Earlier revisions walked back from each sample through every scope row,
    which is O(samples x rows) and was slow enough to matter on a 10 s capture
    (hundreds of thousands of row visits).  A sweep over interval open/close
    events is exact for nested intervals -- scopes in this system nest, they do
    not partially overlap -- and costs O((samples + rows) log rows) once.

    Semantics: an interval owns [start, start+dur); at an identical instant the
    innermost (most recently opened) scope wins, which is what a nested
    ``pi: builder_build`` inside ``prepare input`` should do.
    """

    def __init__(self, rows: list[dict]):
        self.rows = sorted(rows, key=lambda r: (r["start_us"], -int(r["dur_us"])))
        events: list[tuple[int, int, int]] = []  # (time, kind, row index)
        for idx, row in enumerate(self.rows):
            events.append((row["start_us"], 0, idx))
            events.append((row["start_us"] + int(row["dur_us"]), 1, idx))
        events.sort(key=lambda e: (e[0], e[1]))
        self.events = events
        self.cursor = 0
        self.active: list[int] = []  # row indices, oldest first
        self._active_set: set[int] = set()
        self._last_t = -(1 << 62)

    def _advance(self, t: int) -> None:
        events = self.events
        while self.cursor < len(events) and events[self.cursor][0] <= t:
            _time, kind, idx = events[self.cursor]
            if kind == 0:
                self.active.append(idx)
                self._active_set.add(idx)
            else:
                if idx in self._active_set:
                    self._active_set.discard(idx)
                    self.active = [i for i in self.active if i != idx]
            self.cursor += 1

    def at(self, t: int) -> str | None:
        if t < self._last_t:
            raise ValueError(
                f"ScopeSweep.at() is monotonic: got t={t} after t={self._last_t}. "
                "Sort the samples by timestamp before attributing them.")
        self._last_t = t
        self._advance(t)
        if not self.active:
            return None
        # innermost = most recently opened
        return self.rows[self.active[-1]]["name"]

    def innermost_by_duration(self, t: int) -> str | None:
        """Alternative rule: smallest containing interval. Used for the audit."""
        if t < self._last_t:
            raise ValueError("ScopeSweep.innermost_by_duration() is monotonic")
        self._last_t = t
        self._advance(t)
        best, best_dur = None, None
        for idx in self.active:
            row = self.rows[idx]
            dur = int(row["dur_us"])
            if best_dur is None or dur < best_dur:
                best, best_dur = row["name"], dur
        return best


def parse_perf_script(path: pathlib.Path, btime: float) -> list[dict]:
    """Perf script text -> [{epoch_us, tid, frames:[...]}] (stack root first)."""
    samples: list[dict] = []
    current: dict | None = None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = SAMPLE_RE.match(line)
            if m:
                if current and current["frames"]:
                    samples.append(current)
                current = {
                    "epoch_us": int((btime + float(m.group("time"))) * 1_000_000),
                    "tid": int(m.group("tid")),
                    "comm": m.group("comm"),
                    "event": (m.group("rest") or "").strip(),
                    "frames": [],
                }
                continue
            if current is None:
                continue
            if EVENT_RE.match(line):
                continue  # the "   1 cycles:" header line
            fm = FRAME_RE.match(line)
            if fm:
                sym = fm.group("sym").strip() or "[unknown]"
                dso = fm.group("dso").strip()
                current["frames"].append(f"{sym} ({dso})" if dso else sym)
                continue
            if FRAME_IP_RE.match(line):
                current["frames"].append("[unknown]")
    if current and current["frames"]:
        samples.append(current)
    for s in samples:
        # perf prints leaf first; folded stacks want root first
        s["frames"] = list(reversed(s["frames"]))
    return samples


def render_flamegraph(folded: pathlib.Path, out_svg: pathlib.Path, title: str,
                      tools_bin: pathlib.Path) -> dict:
    binary = tools_bin / "inferno-flamegraph"
    if not binary.is_file():
        return {"ok": False, "reason": f"{binary} not found"}

    def attempt(extra: list[str]) -> tuple[int, str]:
        with open(folded, "r", encoding="utf-8") as src, \
                open(out_svg, "w", encoding="utf-8") as dst:
            proc = subprocess.run([str(binary), "--title", title, *extra],
                                  stdin=src, stdout=dst, stderr=subprocess.PIPE,
                                  text=True, timeout=600)
        return proc.returncode, (proc.stderr or "")

    try:
        rc, err = attempt([])
        if rc != 0 and out_svg.is_file() and out_svg.stat().st_size == 0:
            # Some inferno builds reject flags this one does not know; retry
            # with the plain invocation before giving up.
            rc, err = attempt(["--pretty-xml"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "reason": str(exc)}
    return {"ok": rc == 0 and out_svg.stat().st_size > 0,
            "rc": rc, "stderr": err[-400:],
            "bytes": out_svg.stat().st_size if out_svg.is_file() else 0}


def union_seconds(intervals: list[tuple[int, int]]) -> float:
    """Total wall time covered by a union of [start, end) intervals.

    Summing the interval lengths double counts whenever two scopes overlap --
    on this service several threads record a ``prepare input`` scope, so the
    naive sum (48.5 s) exceeded the 20 s capture.  Sampling-based attribution is
    immune to that, but the time-fraction cross-check must use the union.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total / 1e6


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf-data", required=True)
    ap.add_argument("--lite-log", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scope", default="prepare input")
    ap.add_argument("--tid", type=int, default=None,
                    help="thread id whose lite.log rows define the windows "
                         "(default: the busiest tid)")
    ap.add_argument("--tools-bin", default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--script-keep", action="store_true",
                    help="also keep the full perf script text (gzipped)")
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--perf-script-file", default=None,
                    help="reuse an existing perf script dump instead of running perf")
    args = ap.parse_args(argv)

    perf_data = pathlib.Path(args.perf_data)
    lite_log = pathlib.Path(args.lite_log)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tools_bin = pathlib.Path(args.tools_bin) if args.tools_bin else (
        pathlib.Path.home() / "projects" / "vllm" / "prepare-input-phase" / "tools" / "bin")
    btime = boot_epoch()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    rows = load_lite_scopes(lite_log, args.tid)
    if not rows:
        print("ERROR: no scope rows for the requested tid", file=sys.stderr)
        return 2
    tid = rows[0]["tid"]
    sweep = ScopeSweep(rows)
    scope_windows = intervals_for(rows, args.scope)

    # ---------------------------------------------------------- perf script
    script_path = pathlib.Path(args.perf_script_file) if args.perf_script_file else (
        outdir / ("perf.script.gz" if args.script_keep else "perf.script.tmp"))
    if args.perf_script_file is None:
        # Default field set on purpose: it is the only one that keeps callchains
        # symbolised.  Passing -F (even with +ip) downgrades them to raw IPs.
        cmd = ["perf", "script", "--ns", "-i", str(perf_data), "--header"]
        proc = sh(cmd, sudo=not args.no_sudo)
        if proc.returncode != 0:
            print(f"ERROR: perf script rc={proc.returncode}: {proc.stderr[-500:]}",
                  file=sys.stderr)
            return 2
        if args.script_keep:
            with gzip.open(script_path, "wt", encoding="utf-8") as fh:
                fh.write(proc.stdout)
        else:
            script_path.write_text(proc.stdout, encoding="utf-8")
    samples = parse_perf_script(script_path, btime)
    if not args.script_keep and args.perf_script_file is None:
        try:
            script_path.unlink()
        except OSError:
            pass

    sample_tids = Counter(s["tid"] for s in samples)
    samples_of_tid = [s for s in samples if s["tid"] == tid]

    # ------------------------------------------------------- attribution
    in_scope_folded: Counter[str] = Counter()
    scope_counter: Counter[str] = Counter()
    outside_folded: Counter[str] = Counter()
    leaf_in_scope: Counter[str] = Counter()
    leaf_outside: Counter[str] = Counter()
    n_in = n_out = n_other_tid = 0
    samples.sort(key=lambda s: s["epoch_us"])
    for s in samples:
        if s["tid"] != tid:
            n_other_tid += 1
            continue
        name = sweep.at(s["epoch_us"])
        stack = ";".join(s["frames"]) or "[unknown]"
        leaf = s["frames"][-1] if s["frames"] else "[unknown]"
        if name is not None:
            scope_counter[name] += 1
        if args.scope in (name,):
            # scope root frame keeps the flame graph self-describing
            in_scope_folded[f"{args.scope};{stack}"] += 1
            leaf_in_scope[leaf] += 1
            n_in += 1
        else:
            root = name if name else "<outside>"
            outside_folded[f"{root};{stack}"] += 1
            leaf_outside[leaf] += 1
            n_out += 1

    # The lite.log slice usually spans more wall time than the perf capture (the
    # profiling window opens before the capture and closes after it), so the
    # wall-clock cross-check must be clipped to the capture window.  Without the
    # clip the union covers the whole log -- ~96 s here -- and dwarfs the 20 s
    # capture, which is how an earlier revision printed a nonsensical 2.4x.
    capture_lo = capture_hi = 0
    if samples_of_tid:
        capture_lo = min(s["epoch_us"] for s in samples_of_tid)
        capture_hi = max(s["epoch_us"] for s in samples_of_tid)
    capture_us = capture_hi - capture_lo
    clipped = [(max(a, capture_lo), min(b, capture_hi)) for a, b in scope_windows]
    clipped = [(a, b) for a, b in clipped if b > a]
    window_us = sum(b - a for a, b in clipped)
    window_union_us = union_seconds(clipped) * 1e6
    log_union_us = union_seconds(scope_windows) * 1e6

    in_folded_path = outdir / "folded-in-scope.txt"
    out_folded_path = outdir / "folded-outside-scope.txt"
    in_folded_path.write_text(
        "\n".join(f"{k} {v}" for k, v in in_scope_folded.most_common()) + "\n",
        encoding="utf-8")
    out_folded_path.write_text(
        "\n".join(f"{k} {v}" for k, v in outside_folded.most_common()) + "\n",
        encoding="utf-8")

    renders = {
        "in_scope": render_flamegraph(in_folded_path, outdir / "flame-in-scope.svg",
                                      f"samples inside '{args.scope}' (tid {tid})",
                                      tools_bin),
        "outside_scope": render_flamegraph(out_folded_path,
                                           outdir / "flame-outside-scope.svg",
                                           f"samples outside '{args.scope}' (tid {tid})",
                                           tools_bin),
    }

    summary = {
        "schema": "pi-scope-attribution-v1",
        "perf_data": str(perf_data),
        "lite_log": str(lite_log),
        "target_scope": args.scope,
        "target_tid": tid,
        "boot_epoch": btime,
        "started_utc": started,
        "samples_total": len(samples),
        "samples_of_target_tid": len(samples_of_tid),
        "samples_other_tids": n_other_tid,
        "samples_in_scope": n_in,
        "samples_outside_scope": n_out,
        "scope_windows": len(scope_windows),
        "scope_windows_in_capture": len(clipped),
        "scope_window_total_us": window_us,
        "scope_window_union_us": window_union_us,
        "scope_window_union_whole_log_us": log_union_us,
        "scope_window_overlap_ratio": (window_us / window_union_us)
                                     if window_union_us else None,
        "capture_span_us": capture_us,
        # Use the union: several threads record a `prepare input` scope on this
        # service, so the naive sum can exceed the capture span.
        "implied_scope_time_fraction": (window_union_us / capture_us)
                                       if capture_us else None,
        "sample_share_in_scope": (n_in / len(samples_of_tid)) if samples_of_tid else None,
        "per_scope_samples": dict(scope_counter.most_common()),
        "leaf_top_in_scope": leaf_in_scope.most_common(args.top),
        "leaf_top_outside_scope": leaf_outside.most_common(args.top),
        "renders": renders,
        "method_note": (
            "perf sample ns-since-boot + /proc/stat btime -> epoch us; a sample is "
            "attributed to the innermost lite.log scope whose [start, start+dur) "
            "contains it. Scope time fraction and sample share should agree; a large "
            "gap means the two clocks disagree (report it, do not hide it)."),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "outdir": str(outdir),
        "target_tid": tid,
        "samples": {"total": len(samples), "of_tid": len(samples_of_tid),
                    "in_scope": n_in, "outside": n_out, "other_tids": n_other_tid},
        "scope_time_fraction": summary["implied_scope_time_fraction"],
        "sample_share_in_scope": summary["sample_share_in_scope"],
        "top_in_scope": leaf_in_scope.most_common(10),
        "renders": {k: v.get("ok") for k, v in renders.items()},
    }, indent=2))
    return 0 if n_in else 1


if __name__ == "__main__":
    raise SystemExit(main())
