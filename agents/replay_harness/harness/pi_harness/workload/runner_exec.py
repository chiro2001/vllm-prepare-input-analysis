"""Replay a trace through the **real** worker code path.

Every group of the A/B (``real`` / ``shuffled`` / ``synth``) is replayed with
``pi_harness.runner.replay.PrepareInputReplay._one_step_with``: the trace record
is decoded into a real ``vllm.v1.core.sched.output.SchedulerOutput`` and handed
to the worker's own ``NPUModelRunner._update_states`` + ``_prepare_inputs``
(which internally use the real ``NPUInputBatch`` / block tables / numpy index
arithmetic).  Nothing about the measured code is re-implemented here.

Timing comes from ``pi_harness.runner.timer``: ``wrap_prepare_input_path(timer)``
patches the worker classes once, and for every step the accumulated inclusive
time is read (and reset) so we get a per-step ``_prepare_inputs`` figure from
the real function.  ``triton_launch_us`` is reported as 0 by request — the
Triton Python launch overhead is *not* injected (see the report's fidelity
table); the runner still accounts its own numpy fallback separately.

``drain`` steps (``total_num_scheduled_tokens == 0``) only run
``_update_states``; the real worker returns right after it as well.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from pi_harness.workload.schema import StepRecord

TRITON_LAUNCH_US = 0.0  # not injected (recorded explicitly for the A/B)


def disable_pin_memory() -> bool:
    """Make ``Tensor.pin_memory()`` a no-op in the no-card container.

    ``torch_npu`` hooks ``pin_memory`` and the hook calls ``aclInit``, which
    fails without a device (observed: ``aclInit, error code is 507008`` while
    building a pinned buffer in ``_calc_spec_decode_metadata``).  The shim in
    ``pi_harness/shim`` maps ``device`` arguments but not this allocation hook.

    Pinning only matters for real host↔device transfers, which this harness does
    not model at all, so dropping it changes no measured CPU work; the deviation
    is registered in the report.  Set ``PI_PIN_MEMORY=1`` to keep the real
    behaviour (useful on a real card).
    """
    if os.environ.get("PI_PIN_MEMORY", "0").strip().lower() in {"1", "true", "yes"}:
        return False
    try:
        import torch
    except Exception:  # pragma: no cover
        return False
    fn = torch.Tensor.pin_memory
    if getattr(fn, "_pi_pin_noop", False):
        return True

    def pin_memory(self, *args, **kwargs):  # pragma: no cover - trivial
        return self

    pin_memory._pi_pin_noop = True  # type: ignore[attr-defined]
    torch.Tensor.pin_memory = pin_memory
    return True


class RunnerBackedExecutor:
    """Wraps ``PrepareInputReplay`` for trace-driven, per-step timed replay."""

    def __init__(
        self,
        *,
        max_num_reqs: int = 256,
        max_model_len: int = 8192,
        block_size: int = 128,
        max_num_batched_tokens: int = 2048,
        num_spec_tokens: int = 0,
        enable_prefix_caching: bool = True,
        model_profile: str = "qwen35-0.8b",
    ) -> None:
        disable_pin_memory()
        from pi_harness.runner.config import RunnerConfig
        from pi_harness.runner.replay import PrepareInputReplay
        from pi_harness.runner.timer import SubStepTimer, wrap_prepare_input_path

        self.cfg = RunnerConfig(
            model_profile=model_profile,
            max_model_len=max_model_len,
            max_num_reqs=max_num_reqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
            num_spec_tokens=num_spec_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.replay = PrepareInputReplay(self.cfg)
        self.replay.build()
        # `PrepareInputReplay.build()` already installed the sub-step timer on
        # the worker classes (`wrap_prepare_input_path`); reuse *that* instance,
        # because a second call would silently no-op (`_pi_timer_wrapped` guard)
        # and leave us reading an empty timer.
        if self.replay.timer is None:  # pragma: no cover - timing disabled
            from pi_harness.runner.timer import SubStepTimer, wrap_prepare_input_path

            self.replay.timer = SubStepTimer()
            wrap_prepare_input_path(self.replay.timer)
        self.timer = self.replay.timer
        self._cls_name = type(self.replay.runner).__name__
        self._last_cpu = {"prepare_inputs_cpu_us": 0.0, "update_states_cpu_us": 0.0}
        self._last_wall = {"prepare_inputs_us": 0.0, "update_states_us": 0.0}
        self._install_cpu_probes()

    def _install_cpu_probes(self) -> None:
        """Measure **thread CPU time** of the two real entry points.

        The host is shared, so wall-clock time per step is polluted by
        preemption (measured: p90 noise ~16 % on this machine).  Thread CPU time
        is unaffected by that, which is why the A/B verdict uses it as the
        primary statistic.  The probes are bound on the *instance*, so no vLLM /
        harness class is modified and other runners are untouched.
        """
        runner = self.replay.runner
        orig_prepare = runner._prepare_inputs
        orig_update = runner._update_states
        cpu = self._last_cpu
        wall = self._last_wall

        if not getattr(orig_prepare, "_pi_cpu_probe", False):
            def prepare_probe(*args, **kwargs):
                t0 = time.thread_time_ns()
                w0 = time.perf_counter_ns()
                try:
                    return orig_prepare(*args, **kwargs)
                finally:
                    cpu["prepare_inputs_cpu_us"] = (time.thread_time_ns() - t0) / 1000.0
                    wall["prepare_inputs_us"] = (time.perf_counter_ns() - w0) / 1000.0

            prepare_probe._pi_cpu_probe = True  # type: ignore[attr-defined]
            runner._prepare_inputs = prepare_probe

        if not getattr(orig_update, "_pi_cpu_probe", False):
            def update_probe(*args, **kwargs):
                t0 = time.thread_time_ns()
                w0 = time.perf_counter_ns()
                try:
                    return orig_update(*args, **kwargs)
                finally:
                    cpu["update_states_cpu_us"] = (time.thread_time_ns() - t0) / 1000.0
                    wall["update_states_us"] = (time.perf_counter_ns() - w0) / 1000.0

            update_probe._pi_cpu_probe = True  # type: ignore[attr-defined]
            runner._update_states = update_probe

    # ------------------------------------------------------------------ helpers
    def _label(self, name: str) -> list[str]:
        """Timer labels for a method (the runner may be the audited subclass)."""
        return [
            f"{self._cls_name}.{name}",
            f"NPUModelRunner.{name}",
            f"GPUModelRunner.{name}",
        ]

    def _timer_us(self, name: str) -> float | None:
        for label in self._label(name):
            if label in self.timer.inclusive:
                return self.timer.inclusive[label] * 1e6
        return None

    def drain_ok(self, recs: list[StepRecord]) -> bool:
        """Whether the last recorded step leaves the worker batch empty (so the
        runner can be reused for the next group/run)."""
        for rec in reversed(recs):
            if rec.scheduler_output.get("num_scheduled_tokens"):
                return False
            if rec.scheduler_output.get("finished_req_ids"):
                return True
        return not recs

    # -------------------------------------------------------------------- replay
    def run_trace(self, recs: list[StepRecord]) -> list[dict]:
        from pi_harness.trace.replay import decode_scheduler_output

        rows: list[dict] = []
        for rec in recs:
            so = decode_scheduler_output(rec.scheduler_output)
            is_drain = int(so.total_num_scheduled_tokens) == 0
            self._last_cpu["prepare_inputs_cpu_us"] = 0.0
            self._last_cpu["update_states_cpu_us"] = 0.0
            self._last_wall["prepare_inputs_us"] = 0.0
            self._last_wall["update_states_us"] = 0.0
            self.timer.reset()
            t0 = time.perf_counter_ns()
            c0 = time.process_time_ns()
            if is_drain:
                # exactly what the worker does for an empty step: update the
                # cached states, then return without preparing inputs
                self.replay.runner._update_states(so)
                t1 = time.perf_counter_ns()
                c1 = time.process_time_ns()
                rows.append(
                    {
                        "step_idx": rec.step_idx,
                        "phase": rec.phase,
                        "num_reqs": 0,
                        "total_num_scheduled_tokens": 0,
                        "prepare_inputs_us": 0.0,
                        "prepare_inputs_cpu_us": 0.0,
                        "update_states_us": (
                            self._last_wall["update_states_us"]
                            or (t1 - t0) / 1000.0
                        ),
                        "update_states_cpu_us": self._last_cpu["update_states_cpu_us"],
                        "total_us": (t1 - t0) / 1000.0,
                        "total_cpu_us": (c1 - c0) / 1000.0,
                        "triton_launch_us": TRITON_LAUNCH_US,
                        "drain": True,
                    }
                )
                continue
            self.replay._one_step_with(rec.step_idx, so)
            t1 = time.perf_counter_ns()
            c1 = time.process_time_ns()
            prep = self._timer_us("_prepare_inputs")
            upd = self._timer_us("_update_states")
            if prep is None:
                # the runner's SubStepTimer only owns the wrappers of the first
                # PrepareInputReplay built in this process; fall back to the
                # probe that wraps the very same method
                prep = self._last_wall["prepare_inputs_us"]
            if upd is None:
                upd = self._last_wall["update_states_us"]
            rows.append(
                {
                    "step_idx": rec.step_idx,
                    "phase": rec.phase,
                    "num_reqs": int(len(so.num_scheduled_tokens)),
                    "total_num_scheduled_tokens": int(so.total_num_scheduled_tokens),
                    "prepare_inputs_us": (
                        float(prep) if prep is not None else (t1 - t0) / 1000.0
                    ),
                    "prepare_inputs_cpu_us": self._last_cpu["prepare_inputs_cpu_us"],
                    "update_states_us": float(upd) if upd is not None else 0.0,
                    "update_states_cpu_us": self._last_cpu["update_states_cpu_us"],
                    "total_us": (t1 - t0) / 1000.0,
                    "total_cpu_us": (c1 - c0) / 1000.0,
                    "triton_launch_us": TRITON_LAUNCH_US,
                    "drain": False,
                }
            )
        return rows

    def close(self) -> None:
        # leave the wrappers in place: they belong to the runner package, and
        # unwrapping would also unhook the runner's own instrumentation
        return None


def runner_available() -> bool:
    try:
        import pi_harness.runner.replay  # noqa: F401

        return True
    except Exception:
        return False
