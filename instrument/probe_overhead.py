#!/usr/bin/env python3
"""Measure the cost of the pi_subscope probes on the target CPU.

Run this on the *target* machine (a3-22, Kunpeng 920B) inside the same cpuset
the worker uses, with the same Python interpreter as the container:

    taskset -c 122 python3 instrument/probe_overhead.py --n 200000

It prints one JSON object with the per-probe cost, so the report can state the
instrumentation overhead as a measured number instead of a guess.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


def bench(fn, n: int, rounds: int = 7) -> dict:
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        fn(n)
        samples.append((time.perf_counter_ns() - t0) / n)
    samples.sort()
    return {
        "ns_per_op_median": round(statistics.median(samples), 1),
        "ns_per_op_min": round(samples[0], 1),
        "ns_per_op_p90": round(samples[int(0.9 * (len(samples) - 1))], 1),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200_000, help="iterations per round")
    ap.add_argument(
        "--scopes-per-step",
        type=int,
        default=22,
        help="number of probes actually executed per decode step "
        "(measured with scripts/parse_subscope.py --count-scopes)",
    )
    args = ap.parse_args(argv[1:])

    # Must be set before the module import: PI_ENABLED is resolved once.
    os.environ.setdefault("PI_SUBSCOPE", "on")
    os.environ.setdefault("PI_SUBSCOPE_LOG", "/dev/null")
    os.environ.setdefault("PI_SUBSCOPE_DUMP_EVERY", "1000000000")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import pi_subscope as ps  # noqa: E402

    if not ps.PI_ENABLED:
        raise SystemExit("PI_SUBSCOPE was not on at import time")

    n = args.n
    res: dict = {
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "affinity": sorted(os.sched_getaffinity(0))[:4],
        "n_per_round": n,
        "loadavg": list(os.getloadavg()),
    }

    res["empty_loop"] = bench(lambda k: empty_loop(k), n)
    res["nullcontext_enter_exit"] = bench(lambda k: null_loop(k), n)
    res["single_scope"] = bench(lambda k: single(k), n)
    res["nested_pair"] = bench(lambda k: nested(k), n)
    res["with_realistic_step"] = bench(lambda k: realistic(k, args.scopes_per_step), max(1, n // 20))

    loop = res["nullcontext_enter_exit"]["ns_per_op_median"]
    res["marginal_ns_per_scope"] = round(
        res["single_scope"]["ns_per_op_median"] - loop, 1
    )
    res["marginal_ns_per_nested_pair"] = round(
        res["nested_pair"]["ns_per_op_median"] - loop, 1
    )
    res["scopes_per_step_assumed"] = args.scopes_per_step
    per_step_us = (
        res["marginal_ns_per_scope"] * args.scopes_per_step / 1000.0
    )
    res["estimated_probe_us_per_step"] = round(per_step_us, 2)
    res["whole_step_ns"] = res["with_realistic_step"]["ns_per_op_median"]
    res["measured_step_overhead_us"] = round(
        res["with_realistic_step"]["ns_per_op_median"] / 1000.0, 2
    )
    print(json.dumps(res, indent=2))
    return 0


def empty_loop(k: int) -> None:
    """The loop skeleton only: what any of the variants below costs for free."""
    for _ in range(k):
        pass


def null_loop(k: int) -> None:
    """Loop + a context manager enter/exit with no timing: the fair baseline."""
    import contextlib

    nc = contextlib.nullcontext
    with contextlib.nullcontext():
        for _ in range(k):
            with nc():
                pass


def single(k: int) -> None:
    import pi_subscope as ps

    f = ps.pi_scope
    for _ in range(k):
        with f("pi: X"):
            pass


def nested(k: int) -> None:
    import pi_subscope as ps

    f = ps.pi_scope
    for _ in range(k):
        with f("pi: OUTER"):
            with f("pi: INNER"):
                pass


def realistic(k: int, scopes_per_step: int) -> None:
    """One whole step: _Step wrapper + scopes_per_step probes + emit."""
    import contextlib

    import pi_subscope as ps

    f = ps.pi_scope
    for _ in range(k):
        with ps.pi_prepare_input(contextlib.nullcontext()):
            for _ in range(scopes_per_step):
                with f("pi: synthetic"):
                    pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
