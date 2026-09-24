#!/usr/bin/env python3
"""Unit tests for the pi_subscope probe mechanics.

Run without any NPU:  ``python3 instrument/test_pi_subscope.py``

Covers the four things that can go wrong in a probe library:
  1. exclusive/inclusive accounting must add up to the measured wall time;
  2. an early ``return`` inside the wrapped ``prepare input`` scope must still
     emit the step (vLLM returns early for empty batches);
  3. an exception must propagate, and must not leave the stack dirty;
  4. when disabled, the probes must be pure pass-throughs.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent


def _fresh(enabled: bool, log: pathlib.Path, dump_every: int = 1):
    """Import pi_subscope in a clean interpreter-ish state."""
    for mod in ("pi_subscope",):
        sys.modules.pop(mod, None)
    os.environ["PI_SUBSCOPE"] = "on" if enabled else "off"
    os.environ["PI_SUBSCOPE_LOG"] = str(log)
    os.environ["PI_SUBSCOPE_DUMP_EVERY"] = str(dump_every)
    sys.path.insert(0, str(HERE))
    import pi_subscope  # noqa: PLC0415

    return pi_subscope


def _burn(us: float) -> None:
    end = time.perf_counter_ns() + us * 1000
    while time.perf_counter_ns() < end:
        pass


def _read(log: pathlib.Path) -> list[dict]:
    with open(log, newline="") as fh:
        return list(csv.DictReader(fh))


def test_accounting() -> list[str]:
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "raw.csv"
        ps = _fresh(True, log)
        for _ in range(3):
            with ps.pi_prepare_input(contextlib.nullcontext()):
                ps.pi_step_info("decode", 8, 8)
                with ps.pi_scope("pi: outer"):
                    with ps.pi_scope("pi: inner"):
                        _burn(300)
                    _burn(300)
        ps.pi_flush()
        rows = _read(log)
        steps = {r["step"] for r in rows}
        if steps != {"1", "2", "3"}:
            fails.append(f"expected 3 steps, got {steps}")
        for step in steps:
            rs = [r for r in rows if r["step"] == step]
            wall = float(next(r for r in rs if r["scope"] == "pi: TOTAL")["self_us"])
            attributed = sum(
                float(r["self_us"]) for r in rs if r["scope"] != "pi: TOTAL"
            )
            if not 0 <= attributed <= wall:
                fails.append(f"step {step}: attributed {attributed} > wall {wall}")
            if wall < 600:
                fails.append(f"step {step}: wall {wall}us below the burnt 600us")
            inner = next(r for r in rs if r["scope"] == "pi: inner")
            if float(inner["incl_us"]) < 300:
                fails.append(f"step {step}: inner incl {inner['incl_us']}us < 300us")
            outer = next(r for r in rs if r["scope"] == "pi: outer")
            if float(outer["incl_us"]) < float(inner["incl_us"]):
                fails.append(f"step {step}: outer incl < inner incl")
    return fails


def test_early_return() -> list[str]:
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "raw.csv"
        ps = _fresh(True, log)

        def one_step() -> str:
            with ps.pi_prepare_input(contextlib.nullcontext()):
                with ps.pi_scope("pi: sync_input_prep"):
                    _burn(100)
                return "EMPTY_MODEL_RUNNER_OUTPUT"

        for _ in range(2):
            if one_step() != "EMPTY_MODEL_RUNNER_OUTPUT":
                fails.append("early return value lost")
        ps.pi_flush()
        rows = _read(log)
        if len({r["step"] for r in rows}) != 2:
            fails.append("early-return steps were not recorded")
    return fails


def test_exception() -> list[str]:
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "raw.csv"
        ps = _fresh(True, log)
        try:
            with ps.pi_prepare_input(contextlib.nullcontext()):
                with ps.pi_scope("pi: maybe_boom"):
                    raise RuntimeError("boom")
        except RuntimeError:
            pass
        else:
            fails.append("exception was swallowed")
        # A following step must still be accounted correctly.
        with ps.pi_prepare_input(contextlib.nullcontext()):
            with ps.pi_scope("pi: after"):
                _burn(50)
        ps.pi_flush()
        rows = _read(log)
        if not any(r["scope"] == "pi: after" for r in rows):
            fails.append("step after an exception was not recorded")
    return fails


def test_disabled() -> list[str]:
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "raw.csv"
        ps = _fresh(False, log)
        inner = contextlib.nullcontext()
        if ps.pi_prepare_input(inner) is not inner:
            fails.append("disabled mode must return the caller's context manager")
        if ps.pi_scope("pi: X") is not ps.pi_scope("pi: X"):
            fails.append("disabled mode must return one shared nullcontext")
        with ps.pi_prepare_input(contextlib.nullcontext()):
            with ps.pi_scope("pi: X"):
                pass
        ps.pi_flush()
        if log.exists():
            fails.append("disabled mode wrote a log file")
    return fails


def test_notes() -> list[str]:
    """pi_note must survive to disk, and be readable *before* exit."""
    import json

    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "raw.csv"
        meta = pathlib.Path(tmp) / "notes.json"
        for mod in ("pi_subscope",):
            sys.modules.pop(mod, None)
        os.environ["PI_SUBSCOPE"] = "on"
        os.environ["PI_SUBSCOPE_LOG"] = str(log)
        os.environ["PI_SUBSCOPE_META"] = str(meta)
        os.environ["PI_SUBSCOPE_DUMP_EVERY"] = "1"
        sys.path.insert(0, str(HERE))
        import pi_subscope as ps  # noqa: PLC0415

        ps.pi_note("k1", {"a": 1})
        ps.pi_note("k1", {"a": 999})  # first value wins
        with ps.pi_prepare_input(contextlib.nullcontext()):
            with ps.pi_scope("pi: x"):
                pass
        if not meta.is_file():
            fails.append("pi_note did not reach disk before exit")
            return fails
        doc = json.loads(meta.read_text())
        if doc["notes"].get("k1") != {"a": 1}:
            fails.append(f"first-value-wins broken: {doc['notes'].get('k1')}")
        if doc.get("n_notes") != 1:
            fails.append(f"unexpected note count: {doc.get('n_notes')}")
    return fails


def main() -> int:
    tests = {
        "accounting": test_accounting,
        "early_return": test_early_return,
        "exception_safety": test_exception,
        "disabled_is_passthrough": test_disabled,
        "notes_persist": test_notes,
    }
    rc = 0
    for name, fn in tests.items():
        fails = fn()
        if fails:
            rc = 1
            print(f"FAIL {name}")
            for f in fails:
                print(f"     - {f}")
        else:
            print(f"PASS {name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
