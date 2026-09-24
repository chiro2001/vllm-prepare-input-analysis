#!/usr/bin/env python3
"""Build the synthetic POINTS.json used to self-test plot_measure.py.

The values are *not* measurements: the magnitudes are shaped after
``runs/smoke-qwen3508b-tp1-async-on-*/lite-profiler/summary`` (a decode step of
~5.5 ms with ~3.9 ms of ``prepare input``) and the slopes are chosen so that the
figures exercise monotonic curves, a *crossover between prepare input and the
device side*, and two MTP states.  Groups covered: A (concurrency), B (ISL),
C (chunked prefill), E (MTP on/off), plus two deliberately broken records (one
without an analyzer summary, one with a foreign schema) that must be reported
as warnings.

Usage::

    make_fake_points.py [OUT.json]      # default: points.json next to this file
"""

from __future__ import annotations

import json
import pathlib
import sys


def aggregate(ttft_s, itl_ms, tps):
    return {"ttft_s_mean": ttft_s, "mean_itl_ms": itl_ms,
            "output_tps_aggregate": tps}


def block(n_steps, prepare, forward, step, post=140.0, sample=150.0,
          draft=2.7, sched=330.0):
    def stats(value, spread=0.03):
        return {
            "n": n_steps,
            "p50_us": round(value, 3),
            "p90_us": round(value * (1 + spread), 3),
            "p99_us": round(value * (1 + 3 * spread), 3),
            "mean_us": round(value * (1 + spread / 2), 3),
            "min_us": round(value * 0.9, 3),
            "max_us": round(value * 1.15, 3),
            "sum_us": round(value * n_steps, 3),
        }

    return {
        "n_steps": n_steps,
        "step_dur": {
            "p50_us": round(step, 3),
            "p90_us": round(step * 1.04, 3),
            "p99_us": round(step * 1.2, 3),
            "mean_us": round(step * 1.02, 3),
        },
        "prepare input": stats(prepare),
        "forward": stats(forward, 0.06),
        "post process": stats(post, 0.3),
        "sample_token": stats(sample, 0.3),
        "draft_token": stats(draft, 0.1),
        "coverage": {
            "sum_measured_mean_us": round(prepare + forward + post + sample +
                                          draft + sched, 3),
            "ratio_of_step_dur": round(
                (prepare + forward + post + sample + draft + sched) / step, 3
            ),
        },
    }


def point(run_id, tag, group, concurrency, isl, osl, chunk, prepare, forward,
          step, ttft_s, itl_ms, tps, on_cpu, mtp_tokens=0, note=""):
    decode = block(64, prepare, forward, step)
    return {
        "schema": "pi-point-v1",
        "run_id": run_id,
        "tag": tag,
        "group": group,
        "note": note,
        "workload": {
            "requests": concurrency,
            "concurrency": concurrency,
            "prompt_tokens_target": isl,
            "max_tokens": osl,
            "seed": 1024,
            "rounds": 2,
            "temperature": 0.0,
            "streaming": True,
        },
        "service": {
            "container": f"pi-phase-chip3-{run_id}",
            "server": {
                "max_model_len": 32768,
                "max_num_seqs": 256,
                "max_num_batched_tokens": chunk,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "async_scheduling": "on",
                "num_speculative_tokens": mtp_tokens,
                "tensor_parallel_size": 1,
                "prefix_caching": False,
                "dtype": "bfloat16",
            },
            "model": {"served_name": "qwen35-08b"},
        },
        "ec": {"ec_host_pid": 424242, "ec_host_tid": 424242, "ec_comm": "EngineCore"},
        "cpu": {"on_cpu_ratio": on_cpu, "exec_ns_delta": int(on_cpu * 30e9)},
        "rounds": [{"round": 0, "rc": 0, "wall_s": 30.0,
                    "aggregate": aggregate(ttft_s, itl_ms, tps)},
                   {"round": 1, "rc": 0, "wall_s": 30.6,
                    "aggregate": aggregate(ttft_s * 1.01, itl_ms * 1.02,
                                           tps * 0.99)}],
        "phases": {
            "available": True,
            "rc": 0,
            "summary": {
                "schema": "pi-lite-summary-v1",
                "lite_log": f"data/measure/{run_id}/points/{tag}/lite.log",
                "rows": 900,
                "malformed": 0,
                "selected_tid": 131,
                "tids": {"131": 900},
                "orphans": 6,
                "n_steps": 66,
                "steps_by_phase": {"prefill": 2, "prefill+decode": 0,
                                   "decode": 62, "idle": 0},
                "by_phase": {"decode": decode, "all": decode},
                "share": {"decode": {
                    "prepare_over_step_p50": round(prepare / step, 4),
                    "prepare_over_step_mean": round(prepare / step, 4)}},
                "scope_inventory": [],
                "warnings": [],
            },
        },
        "errors": [],
    }


def main(argv):
    out = pathlib.Path(argv[1]) if len(argv) > 1 else (
        pathlib.Path(__file__).resolve().parent / "points.json"
    )
    run_id = "selftest-plot-20260924T0000Z"
    points = []

    # group A: concurrency sweep, ISL=128, OSL=1024.  The slopes are chosen so
    # that the device side dominates at low concurrency and prepare input
    # overtakes it inside the sweep, which is what the crossover figure must
    # annotate (a ~34 us/request CPU slope against a ~8 us/request device one).
    for concurrency in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        prepare = 900.0 + 34.0 * concurrency
        forward = 1400.0 + 8.0 * concurrency
        step = prepare + forward + 620.0
        points.append(point(
            run_id, f"a-b{concurrency}", "A", concurrency, 128, 1024, 2048,
            prepare, forward, step,
            ttft_s=0.35 + 0.004 * concurrency,
            itl_ms=18.0 + 0.05 * concurrency,
            tps=45.0 + 2.0 * concurrency,
            on_cpu=0.30 + 0.0018 * concurrency,
        ))

    # group B: ISL sweep, OSL=1, single request
    for isl in (128, 512, 2048, 8192):
        prepare = 2600.0 + 0.0021 * isl
        forward = 900.0 + 0.061 * isl
        step = prepare + forward + 400.0
        points.append(point(
            run_id, f"b-isl{isl}", "B", 1, isl, 1, 8192,
            prepare, forward, step,
            ttft_s=0.30 + 0.00035 * isl,
            itl_ms=20.0, tps=1.0, on_cpu=0.31 + 0.000004 * isl,
        ))

    # group C: chunked prefill, ISL=32768, bs=1
    for chunk in (512, 2048, 8192):
        prepare = 2650.0 + 0.012 * min(chunk, 512)
        forward = 800.0 + 0.24 * chunk
        step = prepare + forward + 380.0
        points.append(point(
            run_id, f"c-chunk{chunk}", "C", 1, 32768, 1, chunk,
            prepare, forward, step,
            ttft_s=1.1 + 0.00028 * chunk,
            itl_ms=20.0, tps=0.9, on_cpu=0.33,
        ))

    # group E: MTP off/on at the mixed workload shape (B=64): the draft path
    # adds ~30 us of pin+H2D to prepare input and ~1.6x to the device step.
    for state, tokens, extra in (("off", 0, 0.0), ("on", 1, 30.0)):
        concurrency = 64
        prepare = 900.0 + 34.0 * concurrency + extra
        forward = (1400.0 + 8.0 * concurrency) * (1.6 if tokens else 1.0)
        step = prepare + forward + 640.0
        points.append(point(
            run_id, f"e-mtp-{state}", "E", concurrency, 2048, 256, 2048,
            prepare, forward, step,
            ttft_s=2.4, itl_ms=19.5 if tokens else 22.0,
            tps=3000.0 * (1.35 if tokens else 1.0),
            on_cpu=0.42, mtp_tokens=tokens,
            note=f"mtp-{state}",
        ))

    # one point whose analyzer never ran: kept as a row with empty metrics
    broken = point(run_id, "r-no-analyzer", "R", 8, 128, 256, 2048,
                   3000.0, 1800.0, 5400.0, 0.6, 19.0, 400.0, 0.35)
    broken["phases"] = {"available": False, "reason": "analyze_lite.py not found"}
    points.append(broken)

    # one record with a foreign schema: must be skipped entirely
    points.append({"schema": "not-a-point", "tag": "x-foreign"})

    out.write_text(json.dumps(points, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(points)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
