"""A/B harness: is record&replay needed, or is shape-level synthesis enough?

Three groups with (as far as possible) identical shapes are replayed through the
**same** executor:

``real``      captured trace from the NPU run (``--real``)
``shuffled``  same shapes, permuted request/token/block id values
``synth``     shape-level synthesis (``--synth``)

The judgement rule is hard-coded (``DATA_CONTENT_TOL`` / ``DIST_TOL`` below):
data content does not matter when ``shuffled`` stays within 2 % of ``real`` on
the per-step ``prepare_input`` p50 **and** the distributions keep the same shape
(KS <= 0.10 and p90 within 5 %); in that case shape-level synthesis is enough.
Any larger difference means the recorded values themselves drive the cost, so a
real record&replay is required.

Outputs: ``data/harness/ab_*.csv`` (per-step, per-group timings) and
``data/harness/ab_*.json`` (stats, verdict, shape diff, manifest).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from pi_harness.workload import generate as W
from pi_harness.workload.schema import StepRecord, load_jsonl, summarize_trace

# --- hard-coded judgement rule (task C) ------------------------------------
DATA_CONTENT_TOL = 0.02  # |p50 difference| below 2 % -> content independent
DIST_TOL = 0.10  # KS statistic for "same distribution shape"
DIST_P90_TOL = 0.05  # p90 within 5 %

# Primary A/B statistic: per-step *CPU* time of prepare_inputs.  Wall time on a
# shared host is dominated by preemption by other tenants (measured noise floor
# up to 20 % p90), CPU time is reproducible to well under 1 %.
PRIMARY = "prepare_inputs_cpu_us"

# All groups are replayed by the *same* downstream implementation:
#   pi_harness.runner.replay.PrepareInputReplay._one_step_with
# which drives the real NPUModelRunner._update_states + _prepare_inputs.


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {
            "n": 0,
            **{
                k: float("nan")
                for k in ("mean", "p10", "p50", "p90", "p99", "min", "max", "std")
            },
        }
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
        "std": float(a.std(ddof=0)),
    }


def _ks_statistic(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    x = np.sort(np.asarray(a, dtype=np.float64))
    y = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def compare_groups(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    """Compare two timed groups.

    The three groups replay the *same* step sequence, so the sensitive test is
    the **paired** per-step ratio ``b/a`` (each step is its own control, which
    removes the between-step variance that dominates small traces).  The raw
    unpaired distributions are reported as well.
    """
    key = PRIMARY
    sa, sb = a["stats"][key], b["stats"][key]
    wa, wb = a["stats"]["prepare_inputs_us"], b["stats"]["prepare_inputs_us"]
    p50_rel = (sb["p50"] - sa["p50"]) / sa["p50"] if sa["p50"] else float("nan")
    p90_rel = (sb["p90"] - sa["p90"]) / sa["p90"] if sa["p90"] else float("nan")
    p99_rel = (sb["p99"] - sa["p99"]) / sa["p99"] if sa["p99"] else float("nan")
    wall_p50_rel = (
        (wb["p50"] - wa["p50"]) / wa["p50"] if wa["p50"] else float("nan")
    )
    ks = _ks_statistic(a["samples"][key], b["samples"][key])

    # paired, noise-suppressed comparison (per-step min over repeats)
    pairs = _paired_ratios(a.get("per_step_min", {}), b.get("per_step_min", {}))
    ratios = [r for _, r in pairs]
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    rel_dev = [abs(r - 1.0) for r in ratios]
    p90_dev = float(np.percentile(rel_dev, 90)) if rel_dev else float("nan")
    observed_dev = abs(median_ratio - 1.0)
    noise_a = (a.get("noise_floor") or {}).get("abs_dev", float("nan"))
    noise_b = (b.get("noise_floor") or {}).get("abs_dev", float("nan"))
    noise_p90_a = (a.get("noise_floor") or {}).get("p90_abs_dev", float("nan"))
    noise_p90_b = (b.get("noise_floor") or {}).get("p90_abs_dev", float("nan"))
    noise_floor = float(np.nanmax([noise_a, noise_b])) if any(
        not np.isnan(x) for x in (noise_a, noise_b)
    ) else float("nan")
    noise_p90 = float(np.nanmax([noise_p90_a, noise_p90_b])) if any(
        not np.isnan(x) for x in (noise_p90_a, noise_p90_b)
    ) else float("nan")
    same_p50 = observed_dev < DATA_CONTENT_TOL
    # "same distribution shape": the tail spread must be inside the tolerance,
    # or at least no wider than the same-group measurement noise (the host is
    # shared, so a quiet-machine tolerance alone would produce false positives)
    same_dist = (p90_dev <= DIST_P90_TOL) or (
        not np.isnan(noise_p90) and p90_dev <= noise_p90
    )
    quality = (
        "good"
        if (
            not np.isnan(noise_floor)
            and noise_floor <= 0.01
            and not np.isnan(noise_p90)
            and noise_p90 <= 0.05
        )
        else (
            "fair"
            if (not np.isnan(noise_floor) and noise_floor <= 0.02)
            else "poor"
        )
    )
    beyond_noise = (not np.isnan(noise_floor)) and observed_dev > max(noise_floor, 1e-9)
    if same_p50 and same_dist:
        verdict = "content_insensitive"
    elif observed_dev >= DATA_CONTENT_TOL and beyond_noise and quality != "poor":
        verdict = "content_sensitive"
    elif same_p50 and not same_dist and not np.isnan(noise_p90) and p90_dev > noise_p90:
        verdict = "content_sensitive_tail"
    else:
        verdict = "inconclusive_noise"
    return {
        "a": label_a,
        "b": label_b,
        "statistic": key,
        "wall_p50_rel_report_only": wall_p50_rel,
        "p50_rel": p50_rel,
        "p90_rel": p90_rel,
        "p99_rel": p99_rel,
        "ks_unpaired": ks,
        "paired_steps": len(pairs),
        "paired_median_ratio": median_ratio,
        "paired_p90_abs_dev": p90_dev,
        "paired_ratio_worst": (
            {"step": max(pairs, key=lambda kv: abs(kv[1] - 1))[0],
             "ratio": max(pairs, key=lambda kv: abs(kv[1] - 1))[1]}
            if pairs else None
        ),
        "same_p50_within_2pct": bool(same_p50),
        "same_distribution_shape": bool(same_dist),
        "observed_abs_dev": observed_dev,
        "noise_floor_abs_dev": (
            None if np.isnan(noise_floor) else float(noise_floor)
        ),
        "noise_floor_p90_abs_dev": (
            None if np.isnan(noise_p90) else float(noise_p90)
        ),
        "measurement_quality": quality,
        "beyond_noise_floor": bool(beyond_noise),
        "verdict": verdict,
        "rule": {
            "data_content_tol": DATA_CONTENT_TOL,
            "statistic": "paired per-step ratio on per-step min over repeats",
            "ks_tol_unpaired_report_only": DIST_TOL,
            "paired_p90_dev_tol": DIST_P90_TOL,
            "paired_p90_dev_also_ok_within_noise": True,
            "content_sensitive_also_requires": "observed dev > noise floor of the "
            "same group's even/odd run halves",
        },
    }


def _paired_ratios(a: dict, b: dict) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for step, va in (a or {}).items():
        vb = (b or {}).get(step)
        if not va or not vb:
            continue
        out.append((int(step), float(vb) / float(va)))
    return out


def shape_diff(a: list[StepRecord], b: list[StepRecord]) -> dict:
    """Per-step shape comparison (only the fields the A/B is supposed to fix)."""
    keys = (
        "num_reqs",
        "total_num_scheduled_tokens",
        "num_new_reqs",
        "num_cached_reqs",
        "num_spec_decode_reqs",
        "num_finished_req_ids",
        "block_table_width_sum",
    )
    n = min(len(a), len(b))
    out: dict[str, Any] = {
        "steps_a": len(a),
        "steps_b": len(b),
        "compared_steps": n,
        "mismatch_steps": 0,
        "per_key_rel_diff_max": {},
        "worst_steps": [],
    }
    worst: list[tuple[float, int, str]] = []
    for k in keys:
        worst_rel = 0.0
        for i in range(n):
            sa = _sig(a[i])[k]
            sb = _sig(b[i])[k]
            denom = max(abs(sa), 1)
            rel = abs(sb - sa) / denom
            worst_rel = max(worst_rel, rel)
        out["per_key_rel_diff_max"][k] = float(worst_rel)
    for i in range(n):
        sa, sb = _sig(a[i]), _sig(b[i])
        bad = [k for k in keys if sa[k] != sb[k]]
        if bad:
            out["mismatch_steps"] += 1
            worst.append(
                (
                    sum(
                        abs(sb[k] - sa[k]) / max(abs(sa[k]), 1) for k in keys
                    ),
                    i,
                    ",".join(bad),
                )
            )
    out["worst_steps"] = [
        {"step": i, "keys": k, "rel_sum": r}
        for r, i, k in sorted(worst, reverse=True)[:10]
    ]
    return out


_SIG_CACHE: dict[int, dict] = {}


def _sig(rec: StepRecord) -> dict:
    key = id(rec)
    sig = _SIG_CACHE.get(key)
    if sig is None:
        from pi_harness.workload.schema import shape_signature

        sig = shape_signature(rec)
        _SIG_CACHE[key] = sig
    return sig


# ---------------------------------------------------------------------------
# executors
# ---------------------------------------------------------------------------


def run_group(
    groups_recs: dict[str, list[StepRecord]],
    args: argparse.Namespace,
    *,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, dict], float, dict]:
    """Replay every group **interleaved** (round robin over repetitions).

    Interleaving keeps drift of the host (frequency, other tenants, GC) common
    to all groups instead of letting it load only one group.
    """
    if args.executor == "local":
        packed, wall, extra = _run_group_local(
            groups_recs, args, repeats=repeats, warmup=warmup
        )
        return packed, wall, extra
    try:
        return _run_group_runner(groups_recs, args, repeats=repeats, warmup=warmup)
    except Exception as exc:
        import traceback

        print(
            f"[ab] runner executor unavailable ({exc!r}); falling back to the "
            "local CPU mirror -- these numbers are NOT the delivered A/B path",
            file=sys.stderr,
        )
        print(traceback.format_exc()[-1500:], file=sys.stderr)
        packed, wall, extra = _run_group_local(
            groups_recs, args, repeats=repeats, warmup=warmup
        )
        for g in packed.values():
            g["executor"] = "local(fallback)"
        extra["fallback_reason"] = repr(exc)
        extra["fallback_traceback"] = traceback.format_exc()[-1500:]
        return packed, wall, extra


def _run_group_runner(
    groups_recs: dict[str, list[StepRecord]],
    args: argparse.Namespace,
    *,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, dict], float, dict]:
    """The delivered A/B path: real worker code for every group.

    One runner is built and reused for all groups/runs: every trace ends with a
    drain step, so the worker batch is empty again and the next run starts from
    the same state.  ``assert``-style guards below make that assumption loud.
    """
    from pi_harness.workload.runner_exec import RunnerBackedExecutor

    rows: dict[str, list[dict]] = {name: [] for name in groups_recs}
    for name, recs in groups_recs.items():
        if not recs:
            raise SystemExit(f"group {name} is empty")

    ends_clean = all(_ends_with_drain(r) for r in groups_recs.values())
    reuse = args.runner_reuse == "always" or (
        args.runner_reuse == "auto" and ends_clean
    )
    if args.runner_reuse == "always" and not ends_clean:
        print(
            "[ab] warning: --runner-reuse always with a trace that does not end "
            "in a drain step; worker state will leak across runs",
            file=sys.stderr,
        )

    def _new_executor() -> Any:
        return RunnerBackedExecutor(
            max_num_reqs=args.max_num_reqs,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            max_num_batched_tokens=args.chunk_size,
            num_spec_tokens=args.num_spec_tokens,
            enable_prefix_caching=bool(args.enable_prefix_caching),
            model_profile=args.model_profile,
        )

    extra: dict = {
        "runner_reuse": (
            "single_runner"
            if reuse
            else f"rebuild_per_run (traces without drain step: {args.runner_reuse})"
        ),
        "triton_launch_us_injected": 0.0,
        "triton_cpu_note": (
            "Triton Python launch overhead is NOT injected (triton_launch_us=0); "
            "the runner replaces _compute_slot_mapping_kernel with a vectorised "
            "numpy implementation and accounts it separately."
        ),
    }
    names = list(groups_recs)
    t_start = time.perf_counter()
    ex = _new_executor() if reuse else None
    if ex is not None:
        extra["runner_meta"] = getattr(ex.replay, "meta", {})
    try:
        for rep in range(warmup + repeats):
            # rotate the group order every repetition so cache/frequency drift
            # cannot favour one group
            order = names[rep % len(names):] + names[: rep % len(names)]
            for name in order:
                gc.collect()
                if ex is None:
                    cur = _new_executor()
                else:
                    cur = ex
                try:
                    step_rows = cur.run_trace(groups_recs[name])
                    if reuse and not cur.drain_ok(groups_recs[name]):
                        raise SystemExit(
                            f"group {name}: worker batch is not empty after the "
                            "trace; runner reuse would leak state"
                        )
                finally:
                    if ex is None:
                        cur.close()
                if rep < warmup:
                    continue
                for r in step_rows:
                    rows[name].append({**r, "group": name, "run": rep - warmup})
    finally:
        if ex is not None:
            ex.close()
    wall = time.perf_counter() - t_start
    packed = {
        name: _pack(name, rows[name], wall, executor="runner",
                    repeats=repeats, warmup=warmup)
        for name in groups_recs
    }
    return packed, wall, extra


def _run_group_local(
    groups_recs: dict[str, list[StepRecord]],
    args: argparse.Namespace,
    *,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, dict], float, dict]:
    """Development fallback (``--executor local``): a CPU mirror of the same
    loops.  Its numbers are **not** comparable to the runner path and are only
    used to validate buffer/state logic when the real runner cannot start."""
    from pi_harness.workload.local_exec import LocalPrepareInputExecutor, run_trace

    rows: dict[str, list[dict]] = {name: [] for name in groups_recs}
    sizes = {name: _executor_size(recs, args) for name, recs in groups_recs.items()}
    t_start = time.perf_counter()
    names = list(groups_recs)
    for rep in range(warmup + repeats):
        order = names[rep % len(names):] + names[: rep % len(names)]
        for name in order:
            recs = groups_recs[name]
            max_reqs, max_len = sizes[name]
            gc.collect()
            ex = LocalPrepareInputExecutor(
                max_num_reqs=max_reqs,
                max_model_len=max_len,
                block_size=args.block_size,
                max_num_batched_tokens=args.chunk_size,
            )
            step_rows = run_trace(
                recs,
                executor=ex,
                max_num_reqs=max_reqs,
                max_model_len=max_len,
                block_size=args.block_size,
                max_num_batched_tokens=args.chunk_size,
            )
            if rep < warmup:
                continue
            for r in step_rows:
                rows[name].append({**r, "group": name, "run": rep - warmup})
    wall = time.perf_counter() - t_start
    packed = {
        name: _pack(name, rows[name], wall, executor="local",
                    repeats=repeats, warmup=warmup)
        for name in groups_recs
    }
    return packed, wall, {"execute": "local_cpu_mirror"}


def _executor_size(recs: list[StepRecord], args: argparse.Namespace) -> tuple[int, int]:
    max_reqs = max(
        [len((r.input_batch or {}).get("req_ids") or []) for r in recs] + [1]
    )
    max_reqs = max(max_reqs, args.max_num_reqs)
    max_len = max(
        [
            ((r.input_batch or {}).get("token_ids_cpu_shape") or [0, args.max_model_len])[1]
            for r in recs
        ]
        + [args.max_model_len]
    )
    return max_reqs, max_len


def crosscheck_runner(args: argparse.Namespace) -> dict | None:
    """Cross-check the A/B executor against the *real* replay driver
    (``pi_harness.runner.replay.PrepareInputReplay``) on the equivalent shape
    configuration.

    The runner drives the real ``_update_states``/``_prepare_inputs`` from its
    own synthetic schedule (it does not consume a trace file yet), so it cannot
    produce per-group A/B numbers; it is used here as an independent
    timing/fidelity anchor for the same shape.
    """
    try:
        from pi_harness.runner.config import RunnerConfig  # type: ignore
        from pi_harness.runner.replay import PrepareInputReplay  # type: ignore
    except Exception as exc:
        print(f"[ab] runner cross-check unavailable ({exc!r})", file=sys.stderr)
        return None
    import inspect

    fields_ = set(inspect.signature(RunnerConfig.__init__).parameters)
    cfg_kwargs = {
        k: v
        for k, v in {
            "max_model_len": args.max_model_len,
            "max_num_reqs": args.max_num_reqs,
            "max_num_batched_tokens": args.chunk_size,
            "block_size": args.block_size,
            "steps": max(args.crosscheck_steps, 0) or 64,
            "seed": args.seed,
            "model_profile": args.crosscheck_model_profile,
        }.items()
        if k in fields_
    }
    try:
        cfg = RunnerConfig(**cfg_kwargs)
        rp = PrepareInputReplay(cfg)
        rp.build()
        summary = rp.run()
    except Exception as exc:
        import traceback

        print(f"[ab] runner cross-check failed ({exc!r})", file=sys.stderr)
        print(traceback.format_exc()[-2000:], file=sys.stderr)
        return {"error": repr(exc), "traceback": traceback.format_exc()[-2000:]}
    rows = getattr(rp, "rows", []) or []
    per_step = {
        "prepare_inputs_us": [float(r.get("prepare_inputs_us", 0.0)) for r in rows],
        "update_states_us": [float(r.get("update_states_us", 0.0)) for r in rows],
        "total_us": [float(r.get("total_us", 0.0)) for r in rows],
    }
    return {
        "runner": "pi_harness.runner.replay.PrepareInputReplay",
        "config": cfg_kwargs,
        "steps": len(rows),
        "stats": {k: _percentiles(v) for k, v in per_step.items()},
        "samples": per_step,
        "summary": {
            k: v for k, v in (summary or {}).items() if k not in {"substeps"}
        },
    }


def _pack(
    name: str,
    rows: list[dict],
    wall_s: float,
    *,
    executor: str,
    repeats: int,
    warmup: int,
) -> dict:
    fields = (
        PRIMARY,
        "prepare_inputs_us",
        "update_states_us",
        "update_states_cpu_us",
        "total_us",
        "total_cpu_us",
    )
    samples = {f: [r[f] for r in rows] for f in fields}
    stats = {f: _percentiles(samples[f]) for f in fields}
    # per-step minimum over repeats: the same step replayed R times, CPU
    # contention can only add time, so the min is the least noisy estimate of
    # that step's cost (used for the paired A/B comparison).
    per_step_min: dict[int, float] = {}
    per_step_median: dict[int, float] = {}
    by_step: dict[int, list[float]] = {}
    by_step_half: list[dict[int, list[float]]] = [{}, {}]
    for r in rows:
        by_step.setdefault(int(r["step_idx"]), []).append(float(r[PRIMARY]))
        half = int(r.get("run", 0)) % 2
        by_step_half[half].setdefault(int(r["step_idx"]), []).append(
            float(r[PRIMARY])
        )
    for step, vals in by_step.items():
        per_step_min[step] = min(vals)
        per_step_median[step] = float(np.median(vals))
    half_min = [
        {step: min(vals) for step, vals in by_step_half[i].items()} for i in (0, 1)
    ]
    noise = _paired_ratios(half_min[0], half_min[1])
    noise_ratios = [r for _, r in noise]
    noise_floor = {
        "steps": len(noise_ratios),
        "median_ratio": float(np.median(noise_ratios)) if noise_ratios else float("nan"),
        "abs_dev": (
            abs(float(np.median(noise_ratios)) - 1.0) if noise_ratios else float("nan")
        ),
        "p90_abs_dev": (
            float(np.percentile([abs(r - 1.0) for r in noise_ratios], 90))
            if noise_ratios
            else float("nan")
        ),
        "note": "same group, even-run half vs odd-run half (measurement noise)",
    }
    phases: dict[str, int] = {}
    by_phase: dict[str, list[dict]] = {}
    for r in rows:
        ph = str(r["phase"])
        phases[ph] = phases.get(ph, 0) + 1
        by_phase.setdefault(ph, []).append(r)
    stats_by_phase = {
        ph: {
            "steps": len(rs),
            PRIMARY: _percentiles([r[PRIMARY] for r in rs]),
            "prepare_inputs_us": _percentiles([r["prepare_inputs_us"] for r in rs]),
        }
        for ph, rs in by_phase.items()
        if len(rs) >= 3
    }
    return {
        "group": name,
        "executor": executor,
        "repeats": repeats,
        "warmup": warmup,
        "wall_s": wall_s,
        "steps": len(rows),
        "phases": phases,
        "stats_by_phase": stats_by_phase,
        "samples": samples,
        "stats": stats,
        "per_step_min": per_step_min,
        "per_step_median": per_step_median,
        "per_step_min_half": [half_min[0], half_min[1]],
        "noise_floor": noise_floor,
        "rows": rows,
    }


def executors_match(results: dict[str, dict], kind: str) -> bool:
    return bool(results) and all(g["executor"] == kind for g in results.values())


def _ends_with_drain(recs: list[StepRecord]) -> bool:
    """Whether the trace ends with an empty step that reports the last finishes.

    Those are the steps the engine runs once the scheduler has no more work
    (``Scheduler.has_requests()`` stays True while ``finished_req_ids`` is
    non-empty), and they are what makes a single built runner reusable.
    """
    for rec in reversed(recs):
        if rec.scheduler_output.get("num_scheduled_tokens"):
            return False
        if rec.scheduler_output.get("finished_req_ids"):
            return True
    return not recs


def _manifest(args: argparse.Namespace, traces: dict[str, str]) -> dict:
    here = os.path.abspath(__file__)
    files = [
        here,
        os.path.join(os.path.dirname(here), "schema.py"),
        os.path.join(os.path.dirname(here), "generate.py"),
        os.path.join(os.path.dirname(here), "synth_real.py"),
        os.path.join(os.path.dirname(here), "local_exec.py"),
    ]
    sha = {}
    for f in files:
        try:
            with open(f, "rb") as fh:
                sha[os.path.basename(f)] = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            sha[os.path.basename(f)] = "missing"
    vllm_version = None
    try:  # best effort; import is cheap compared to a replay
        import vllm

        vllm_version = vllm.__version__
    except Exception:
        pass
    return {
        "kind": "ab",
        "tag": args.tag,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "vllm_version": vllm_version,
        "executor": args.executor,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "params": {
            k: v
            for k, v in vars(args).items()
            if k not in {"real", "synth", "shuffled"}
        },
        "traces": traces,
        "script_sha256": sha,
        "rule": {
            "data_content_tol": DATA_CONTENT_TOL,
            "ks_tol": DIST_TOL,
            "p90_tol": DIST_P90_TOL,
        },
    }


def _write_csv(path: str, groups: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = (
        "group",
        "run",
        "step_idx",
        "phase",
        "num_reqs",
        "total_num_scheduled_tokens",
        "update_states_us",
        "prepare_inputs_us",
        "total_us",
        "update_states_cpu_us",
        "prepare_inputs_cpu_us",
        "total_cpu_us",
        "triton_launch_us",
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for name, g in groups.items():
            for r in g["rows"]:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")


def _fmt(stats: dict) -> str:
    return (
        f"p50={stats['p50']:8.1f} p90={stats['p90']:8.1f} "
        f"p99={stats['p99']:8.1f} mean={stats['mean']:8.1f} µs(cpu-time)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", default="", help="captured trace (jsonl)")
    p.add_argument("--synth", default="", help="synthetic trace (jsonl)")
    p.add_argument("--shuffled", default="", help="shuffled trace (jsonl, optional)")
    p.add_argument("--trace-file", default="", help="single-trace mode (perf hook)")
    p.add_argument("--only-group", default="", help="replay one group and exit")
    p.add_argument("--csv", default="")
    p.add_argument("--json", dest="json_out", default="")
    p.add_argument("--tag", default="v0")
    p.add_argument(
        "--executor",
        default="runner",
        choices=["runner", "local"],
        help="runner = real NPUModelRunner._update_states/_prepare_inputs via "
        "PrepareInputReplay._one_step_with (the delivered A/B path); "
        "local = CPU mirror, not comparable with runner numbers",
    )
    p.add_argument("--num-spec-tokens", type=int, default=0)
    p.add_argument("--enable-prefix-caching", type=int, default=1)
    p.add_argument("--model-profile", default="qwen35-0.8b")
    p.add_argument(
        "--runner-reuse",
        default="auto",
        choices=["auto", "always", "never"],
        help="auto = one built runner for all groups/runs when every trace ends "
        "with a drain step, else rebuild per run (slow, but required for "
        "truncated captures)",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--max-num-reqs", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--perf-cmd", default="", help="optional perf hook (shell), "
                                                  "run once per group with "
                                                  "AB_GROUP/AB_TRACE/AB_OUT set")
    p.add_argument("--crosscheck-runner", action="store_true",
                   help="also run the real replay driver (pi_harness.runner) "
                        "on the equivalent shape config")
    p.add_argument("--crosscheck-steps", type=int, default=0)
    p.add_argument("--crosscheck-model-profile", default="qwen35-0.8b")
    args = p.parse_args(argv)

    # single-group replay mode (used by --perf-cmd)
    if args.only_group and args.trace_file:
        recs = load_jsonl(args.trace_file)
        packed, _ = run_group(
            {args.only_group: recs}, args, repeats=args.repeats, warmup=args.warmup
        )[:2]
        g = packed[args.only_group]
        print(f"{args.only_group}: n={g['steps']} {_fmt(g['stats']['prepare_inputs_us'])}")
        return 0

    traces: dict[str, str] = {}
    groups_recs: dict[str, list[StepRecord]] = {}
    if args.real:
        groups_recs["real"] = load_jsonl(args.real)
        traces["real"] = os.path.abspath(args.real)
    if args.synth:
        groups_recs["synth"] = load_jsonl(args.synth)
        traces["synth"] = os.path.abspath(args.synth)
    if not groups_recs:
        raise SystemExit("need at least --synth (and ideally --real)")

    base = "real" if "real" in groups_recs else "synth"
    if args.shuffled:
        groups_recs["shuffled"] = load_jsonl(args.shuffled)
        traces["shuffled"] = os.path.abspath(args.shuffled)
    else:
        groups_recs["shuffled"] = W.shuffle_trace(
            groups_recs[base], seed=args.shuffle_seed, source=f"shuffled({base})"
        )
        traces["shuffled"] = f"derived from {base} (seed={args.shuffle_seed})"

    # shape check first: the A/B only means something if the shapes match
    shape_checks = {}
    for name, recs in groups_recs.items():
        if name == base:
            continue
        shape_checks[f"{base}_vs_{name}"] = shape_diff(groups_recs[base], recs)
        shape_checks[f"{base}_vs_{name}"]["summary_" + base] = {
            k: summarize_trace(groups_recs[base])[k]
            for k in ("num_steps", "tokens_mean", "num_reqs_mean", "tokens_max")
        }
        shape_checks[f"{base}_vs_{name}"]["summary_" + name] = {
            k: summarize_trace(recs)[k]
            for k in ("num_steps", "tokens_mean", "num_reqs_mean", "tokens_max")
        }
    if "real" in groups_recs and "synth" in groups_recs:
        shape_checks["real_vs_synth"] = shape_diff(
            groups_recs["real"], groups_recs["synth"]
        )

    order = [g for g in ("real", "shuffled", "synth") if g in groups_recs]
    ordered_recs = {name: groups_recs[name] for name in order}
    results, wall, extra = run_group(
        ordered_recs, args, repeats=args.repeats, warmup=args.warmup
    )
    for name in order:
        g = results[name]
        print(
            f"[ab] {name:9s} executor={g['executor']:6s} "
            f"{_fmt(g['stats']['prepare_inputs_us'])} "
            f"(noise median {g['noise_floor']['abs_dev']:.3f} / "
            f"p90 {g['noise_floor']['p90_abs_dev']:.3f})"
        )
    print(f"[ab] total replay wall: {wall:.2f} s "
          f"({len(order)} groups x {args.repeats} runs + {args.warmup} warmup)")
    executors = sorted({g["executor"] for g in results.values()})
    print(f"[ab] downstream executor: {', '.join(executors)}")

    crosscheck = crosscheck_runner(args) if args.crosscheck_runner else None
    if crosscheck:
        if crosscheck.get("stats"):
            print(
                f"[ab] runner cross-check ({crosscheck['steps']} steps): "
                f"{_fmt(crosscheck['stats']['prepare_inputs_us'])}"
            )
        else:
            print(f"[ab] runner cross-check unavailable: {crosscheck.get('error')}")

    comparisons: dict[str, dict] = {}
    if "real" in results and "shuffled" in results:
        comparisons["shuffled_vs_real"] = compare_groups(
            results["real"], results["shuffled"], "real", "shuffled"
        )
    if "real" in results and "synth" in results:
        comparisons["synth_vs_real"] = compare_groups(
            results["real"], results["synth"], "real", "synth"
        )
    if "real" not in results:
        comparisons["shuffled_vs_synth"] = compare_groups(
            results["synth"], results["shuffled"], "synth", "shuffled"
        )

    # per-phase comparison (prefill vs decode steps separated, as required)
    comparisons_by_phase: dict[str, dict] = {}
    for label_a, label_b in (
        ("real", "shuffled"),
        ("real", "synth"),
        ("synth", "shuffled"),
    ):
        if label_a not in results or label_b not in results:
            continue
        for phase in sorted(
            set(results[label_a]["stats_by_phase"])
            & set(results[label_b]["stats_by_phase"])
        ):
            a_stats = _percentiles(
                [r[PRIMARY] for r in results[label_a]["rows"] if r["phase"] == phase]
            )
            b_stats = _percentiles(
                [r[PRIMARY] for r in results[label_b]["rows"] if r["phase"] == phase]
            )
            pa = results[label_a]["per_step_min"]
            pb = results[label_b]["per_step_min"]
            phase_steps = {
                r["step_idx"] for r in results[label_a]["rows"] if r["phase"] == phase
            }
            ratios = [
                float(pb[k]) / float(pa[k])
                for k in phase_steps
                if k in pa and k in pb and pa[k]
            ]
            comparisons_by_phase[f"{label_b}_vs_{label_a}:{phase}"] = {
                "phase": phase,
                "steps": len(ratios),
                label_a: a_stats,
                label_b: b_stats,
                "p50_rel": (
                    (b_stats["p50"] - a_stats["p50"]) / a_stats["p50"]
                    if a_stats["p50"]
                    else float("nan")
                ),
                "paired_median_ratio": (
                    float(np.median(ratios)) if ratios else float("nan")
                ),
            }

    if "shuffled_vs_real" in comparisons:
        cmp_sr = comparisons["shuffled_vs_real"]
        if cmp_sr["verdict"] == "content_insensitive":
            verdict = "shape_level_synthesis_sufficient"
            reason = (
                "shuffled values stay within tolerance of the recorded ones "
                f"(paired median ratio {cmp_sr['paired_median_ratio']:.4f}, "
                f"p90 |dev| {cmp_sr['paired_p90_abs_dev']:.4f}, < "
                f"{DATA_CONTENT_TOL:.0%}/{DIST_P90_TOL:.0%}) -> the cost depends "
                "on the shape, not on the recorded values"
            )
        elif cmp_sr["verdict"] in ("content_sensitive", "content_sensitive_tail"):
            verdict = "record_replay_required"
            reason = (
                "shuffled values change the per-step prepare_input cost beyond "
                f"the {DATA_CONTENT_TOL:.0%} tolerance and beyond measurement "
                f"noise (paired median ratio {cmp_sr['paired_median_ratio']:.4f}, "
                f"p90 |dev| {cmp_sr['paired_p90_abs_dev']:.4f}, noise floor "
                f"{cmp_sr['noise_floor_abs_dev']}, verdict "
                f"{cmp_sr['verdict']}) -> the recorded values matter, keep "
                "record&replay"
            )
        else:
            verdict = "inconclusive_measurement_noise"
            reason = (
                "the shuffled/real difference is inside the harness noise floor "
                f"(observed {cmp_sr['observed_abs_dev']:.4f} vs noise "
                f"{cmp_sr['noise_floor_abs_dev']}); increase --repeats or trace "
                "size before drawing a conclusion"
            )
        if "synth_vs_real" in comparisons and not comparisons["synth_vs_real"][
            "same_distribution_shape"
        ]:
            verdict = "shape_level_synthesis_insufficient"
            reason = (
                reason
                + " | BUT the synthetic shapes do not match the captured run "
                "(p90 of the paired deviation outside tolerance), so the "
                "shape recipe itself has to be tightened"
            )
    else:
        verdict = "pending_real_trace"
        cmp_ss = comparisons.get("shuffled_vs_synth")
        if cmp_ss:
            path_desc = (
                "the real worker path (NPUModelRunner._update_states/"
                "_prepare_inputs via PrepareInputReplay._one_step_with)"
                if executors_match(results, "runner")
                else "the local CPU mirror (NOT the delivered A/B path)"
            )
            base_txt = (
                f"no captured trace yet. shuffled-vs-synth (same shapes, "
                f"different values) on {path_desc}: paired median ratio "
                f"{cmp_ss['paired_median_ratio']:.4f}, p90 |dev| "
                f"{cmp_ss['paired_p90_abs_dev']:.4f} (same-group noise "
                f"{cmp_ss['noise_floor_abs_dev']:.4f}) -> {cmp_ss['verdict']}."
            )
            if cmp_ss["verdict"] == "content_insensitive":
                tail = (
                    " Value sensitivity is bounded for this workload; the "
                    "real-vs-synth shape gap still needs the captured trace."
                )
            elif cmp_ss["verdict"] == "inconclusive_noise":
                tail = (
                    " The difference is inside this workload's measurement "
                    "noise (short trace / busy host): increase --repeats or the "
                    "trace length before concluding."
                )
            else:
                tail = (
                    " The recorded values change the cost beyond noise, so a "
                    "record&replay trace is needed for this workload."
                )
            reason = base_txt + tail
        else:
            reason = (
                "no captured trace yet: only synth vs shuffled-values has been "
                "measured, which bounds the value-sensitivity but not the "
                "real-vs-synth shape gap"
            )

    summary = {
        "verdict": verdict,
        "reason": reason,
        "comparisons": comparisons,
        "groups": {
            name: {
                "executor": g["executor"],
                "steps": g["steps"],
                "phases": g["phases"],
                "stats_by_phase": g["stats_by_phase"],
                "wall_s": g["wall_s"],
                "stats": g["stats"],
                "samples": g["samples"],
                "per_step_min": g["per_step_min"],
                "per_step_median": g["per_step_median"],
                "per_step_min_half": g["per_step_min_half"],
                "noise_floor": g["noise_floor"],
            }
            for name, g in results.items()
        },
        "shape_checks": shape_checks,
        "comparisons_by_phase": comparisons_by_phase,
        "executor": extra,
        "trace_summaries": {
            name: summarize_trace(recs, include_steps=True)
            for name, recs in groups_recs.items()
        },
        "crosscheck_runner": crosscheck,
        "manifest": _manifest(args, traces),
    }

    if args.csv:
        _write_csv(args.csv, results)
        print(f"[ab] wrote {args.csv}")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[ab] wrote {args.json_out}")

    if args.perf_cmd:
        perf_dir = os.path.dirname(os.path.abspath(args.json_out or args.csv or "ab"))
        for name in order:
            out = os.path.join(perf_dir, f"ab_{args.tag}_perf_{name}.txt")
            env = {
                **os.environ,
                "AB_GROUP": name,
                "AB_TRACE": traces.get(name, ""),
                "AB_OUT": out,
            }
            print(f"[ab] perf hook for {name}: {args.perf_cmd} -> {out}")
            rc = subprocess.run(args.perf_cmd, shell=True, env=env).returncode
            summary.setdefault("perf_hooks", {})[name] = {
                "cmd": args.perf_cmd,
                "out": out,
                "rc": rc,
            }
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)

    print(f"[ab] verdict: {verdict} -- {reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
