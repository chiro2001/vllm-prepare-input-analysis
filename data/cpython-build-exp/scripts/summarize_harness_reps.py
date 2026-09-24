#!/usr/bin/env python3
"""summarize_harness_reps.py <exp-dir> -- aggregate repeated harness runs."""

import glob
import json
import os
import statistics
import sys


def main():
    exp = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(exp, "data", "harness-arm*", "*summary*.json"))):
        tag = os.path.basename(f).split("_summary")[0]
        if not tag.startswith("rep"):
            continue
        d = json.load(open(f))
        p = d.get("prepare_inputs_us", {})
        rows.append((tag, p.get("p50"), p.get("mean"), p.get("p90"), p.get("p99"),
                     d.get("update_states_us", {}).get("p50")))
    by = {}
    for tag, p50, mean, p90, p99, upd in rows:
        arm = tag.split("_", 1)[1]
        by.setdefault(arm, []).append((p50, mean, p90, p99, upd))
    print("raw runs:")
    for tag, p50, mean, p90, p99, upd in rows:
        print("  %-16s p50=%7.1f mean=%7.1f p90=%7.1f p99=%7.1f update50=%5.1f"
              % (tag, p50, mean, p90, p99, upd))
    print()
    print("%-10s %3s %9s %9s %10s %9s %9s" %
          ("arm", "n", "p50_med", "p50_min", "mean_med", "p90_med", "upd_med"))
    for arm in sorted(by):
        v = by[arm]
        print("%-10s %3d %9.1f %9.1f %10.1f %9.1f %9.1f" % (
            arm, len(v), statistics.median(x[0] for x in v), min(x[0] for x in v),
            statistics.median(x[1] for x in v), statistics.median(x[2] for x in v),
            statistics.median(x[4] for x in v)))
    s = by.get("stock")
    if s:
        b = statistics.median(x[0] for x in s)
        for arm in sorted(by):
            if arm == "stock":
                continue
            m = statistics.median(x[0] for x in by[arm])
            print("delta %s vs stock = %+.2f%% (p50 median)" % (arm, 100 * (m - b) / b))


if __name__ == "__main__":
    main()
