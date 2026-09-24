#!/usr/bin/env python3
"""Measure the per-call cost of small NPU ops (dispatch + Python overhead).

Motivation: on B=1 decode the Ascend GDN metadata builder spends ~43 us per
call inside `compute_num_computed_tokens`, which only *returns an already
cached tensor*.  If a single small NPU op costs ~10-40 us of wall time before
any device work is even relevant, then a metadata builder that issues ~10 such
ops per call is fully explained by dispatch cost rather than by computation.

This benchmark therefore measures, for a *small* tensor, exactly the shape of
work the builder does: `torch.empty`, `fill_`, slicing subtraction, `gather`,
`index_select`, `copy_`, `expand_as`, `.item()`.  Each is measured with the same
loop structure so the Python loop cost is subtracted out.

It never touches the NPU except to allocate two tiny tensors, and takes a few
seconds.  Run it inside the container (so torch_npu is importable) but *not*
while a measurement run owns the chip:

    sudo -n docker run --rm --device /dev/davinci3 --privileged \\
        --entrypoint /bin/bash local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler \\
        -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \\
            python3 /work/scripts/npu_dispatch_bench.py --n 3000'

Output is JSON on stdout plus a human table on stderr.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def bench(fn, n: int, rounds: int = 5) -> dict:
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000, help="iterations per op")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--cpu-only", action="store_true",
                    help="run only the CPU control group")
    args = ap.parse_args(argv[1:])

    import torch

    res: dict = {"torch": torch.__version__, "n_per_op": args.n, "groups": {}}

    def measure(device: str) -> dict:
        n = args.n
        out: dict = {}
        a8 = torch.zeros(8, dtype=torch.int32, device=device)
        a2048 = torch.zeros(2048, dtype=torch.int32, device=device)
        idx = torch.arange(8, dtype=torch.int64, device=device)
        sd = {"device": device, "dtype": "int32", "shape": [8]}
        out["empty"] = bench(lambda k: [torch.empty(1, dtype=torch.int32, device=device) for _ in range(k)], n) | sd
        out["fill_"] = bench(lambda k: [a8.fill_(0) for _ in range(k)], n) | sd
        # Three distinct buffers: the in-place form
        # `sub(a[1:], a[:-1], out=a[1:])` is rejected by torch's overlap check
        # on both cpu and npu, and even aliasing the *second* operand is
        # rejected, so each operand gets its own tensor.
        src = torch.zeros(8, dtype=torch.int32, device=device)
        dst = torch.zeros(8, dtype=torch.int32, device=device)
        out["sub_slice_out"] = bench(
            lambda k: [torch.sub(a8[1:], src[:7], out=dst[:7]) for _ in range(k)], n
        ) | sd
        out["index_select"] = bench(
            lambda k: [torch.index_select(a2048, 0, idx) for _ in range(k)], n
        ) | {"device": device, "dtype": "int32", "shape": [2048]}
        out["gather"] = bench(
            lambda k: [torch.gather(a8, 0, idx) for _ in range(k)], n
        ) | sd
        out["copy_"] = bench(lambda k: [a8.copy_(a8) for _ in range(k)], n) | sd
        out["expand_as"] = bench(lambda k: [a8[0].expand_as(a8) for _ in range(k)], n) | sd
        out["to_int64"] = bench(
            lambda k: [a8.to(torch.int64) for _ in range(k)], n
        ) | sd
        out["arange"] = bench(
            lambda k: [torch.arange(2, dtype=torch.int32, device=device) for _ in range(k)], n
        ) | sd
        if device != "cpu":
            out["item_sync"] = bench(lambda k: [a8[0].item() for _ in range(k)], max(1, n // 10)) | sd
        return out

    res["groups"]["cpu"] = measure("cpu")

    if not args.cpu_only:
        try:
            import torch_npu  # noqa: F401

            res["groups"]["npu"] = measure("npu")
            # The headline constant: what one tiny op costs before any real work.
            npu = res["groups"]["npu"]
            cpu = res["groups"]["cpu"]
            res["dispatch_overhead_us"] = {
                op: round((npu[op]["ns_per_op_median"] - cpu[op]["ns_per_op_median"]) / 1000, 2)
                for op in npu
            }
            res["headline_us_per_small_npu_op"] = res["dispatch_overhead_us"]["fill_"]
        except Exception as exc:  # pragma: no cover - environment dependent
            res["npu_error"] = f"{type(exc).__name__}: {exc}"

    text = json.dumps(res, indent=2)
    if args.json_out:
        open(args.json_out, "w").write(text + "\n")
    print(text)

    if "npu" in res["groups"]:
        print("\n=== summary (median ns/op) ===", file=sys.stderr)
        print(f"{'op':<18}{'cpu':>12}{'npu':>12}{'npu-cpu (us)':>16}", file=sys.stderr)
        for op in res["groups"]["cpu"]:
            c = res["groups"]["cpu"][op]["ns_per_op_median"]
            n_ = res["groups"]["npu"].get(op, {}).get("ns_per_op_median")
            if n_ is None:
                continue
            print(f"{op:<18}{c:>12.0f}{n_:>12.0f}{(n_ - c) / 1000:>16.2f}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
