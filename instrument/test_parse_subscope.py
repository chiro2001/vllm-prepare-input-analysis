#!/usr/bin/env python3
"""Smoke test for ``scripts/parse_subscope.py`` on a synthetic raw log.

The synthetic log mimics a decode step: the 11 canonical groups carry hand
picked shares, the remainder is emitted as the unattributed gap, and the
parser must (a) reproduce the shares, (b) pass its own invariant check and
(c) write all four output files.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "parse_subscope", ROOT / "scripts" / "parse_subscope.py"
)
parse_subscope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_subscope)

SCOPE_US = {
    "pi: sync_input_prep": 3.0,
    "pi: update_states": 22.0,
    "pi: tokens_list": 1.0,
    "pi: phase_classify": 4.0,
    "pi: prepare_inputs": 0.0,  # aggregate row, self time is ~0
    "pi: in.block_table_commit": 18.0,
    "pi: in.req_indices": 2.0,
    "pi: in.attn_state": 6.0,
    "pi: in.positions": 9.0,
    "pi: in.token_indices_select": 14.0,
    "pi: in.query_start_loc": 21.0,
    "pi: in.optimistic_seq_lens": 7.0,
    "pi: in.prepare_input_ids": 33.0,
    "pi: in.discard_mask": 12.0,
    "pi: in.num_accepted_tokens": 9.0,
    "pi: in.num_computed_tokens": 16.0,
    "pi: in.tokens_to_gpu": 44.0,
    "pi: in.positions_assembly": 15.0,
    "pi: in.seq_lens": 11.0,
    "pi: in.slot_mapping": 13.0,
    "pi: in.logits_indices": 5.0,
    "pi: batch_exec_and_padding": 8.0,
    "pi: build_attention_metadata": 0.0,
    "pi: am.max_seq_len": 6.0,
    "pi: am.cm_base_pre": 9.0,
    "pi: am.group_loop": 26.0,
    "pi: am.builder_build": 19.0,
    "pi: am.layer_assign": 4.0,
}


def write_raw(path: pathlib.Path, n_steps: int = 40, start_step: int = 1) -> float:
    wall = 400.0
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(parse_subscope.RAW_FIELDS))
        for i in range(n_steps):
            step = start_step + i
            jitter = 1.0 + 0.05 * ((i % 5) - 2)
            for scope, us in SCOPE_US.items():
                writer.writerow(
                    [
                        step, "decode", scope, f"{us * jitter:.3f}", f"{us:.3f}",
                        f"{us:.3f}", 1, f"{wall:.3f}", 8, 8, 4242, 4242,
                    ]
                )
            writer.writerow(
                [step, "decode", "pi: TOTAL", f"{wall:.3f}", f"{wall:.3f}",
                 f"{wall:.3f}", 1, f"{wall:.3f}", 8, 8, 4242, 4242]
            )
    return wall


def main() -> int:
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        raw = tmpdir / "raw.csv"
        wall = write_raw(raw)
        out = tmpdir / "out"
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "parse_subscope.py"),
             str(raw), "--outdir", str(out)],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print(rc.stdout, rc.stderr)
            return 1
        meta = json.loads((out / "meta.json").read_text())
        for name in ("per_step_subscope.csv", "summary.csv", "summary_groups.csv"):
            if not (out / name).is_file():
                fails.append(f"{name} not written")
        if abs(meta["prepare_input_wall_us"]["p50"] - wall) > 1e-6:
            fails.append(f"wall p50 {meta['prepare_input_wall_us']['p50']} != {wall}")
        if meta["invariant_max_abs_residual_us"] > 1e-6:
            fails.append("invariant residual non-zero")

        groups = {
            r["group"]: float(r["self_us_p50"])
            for r in csv.DictReader(open(out / "summary_groups.csv"))
        }
        expected = {
            "synchronize+update_states": 25.0,
            "block_table commit": 18.0,
            "attn_state": 6.0,
            "positions+token_indices": 25.0,
            "query_start_loc": 21.0,
            "input_ids/H2D": 128.0,
            "bookkeeping other": 37.0,   # incl. pi: in.slot_mapping (13.0)
            "spec decode metadata": 0.0,
            "batch padding decision": 8.0,
            "scheduler glue": 5.0,
            "attn metadata": 64.0,
        }
        for group, want in expected.items():
            got = groups.get(group)
            if got is None:
                fails.append(f"group {group!r} missing")
            elif abs(got - want) > 1e-6:
                fails.append(f"group {group!r}: {got} != {want}")
        wall_gap = groups[parse_subscope.UNATTR]
        accounted = sum(v for k, v in groups.items() if k != parse_subscope.UNATTR)
        if abs(accounted + wall_gap - wall) > 1e-6:
            fails.append("group shares do not add up to the wall time")
        if meta["unattributed_pct_p50"] <= 0:
            fails.append("unattributed gap should be positive in this fixture")

    for f in fails:
        print(f"FAIL - {f}")
    if not fails:
        print("PASS parse_subscope smoke test "
              f"(accounted {accounted:.0f}us + unattributed {wall_gap:.0f}us "
              f"= {wall:.0f}us)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
