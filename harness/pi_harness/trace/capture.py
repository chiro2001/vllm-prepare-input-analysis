"""Record side: environment-gated capture hook for the real (NPU) run.

Usage (no vLLM source change required)::

    PI_CAPTURE=/work/data/harness/trace_real.jsonl \\
    PYTHONPATH=/work/harness:/work/harness/pi_harness/trace \\
    python -m vllm.entrypoints.openai.api_server ...      # or any launcher

The hook is *only* installed when ``PI_CAPTURE`` is set:

* ``install()`` returns ``None`` (and does nothing) without the env var, so the
  module can live in a production tree;
* importing this module pulls in **stdlib only** (no vllm / no torch), so it
  cannot break a service that does not use it;
* ``sitecustomize.py`` next to this file calls :func:`install` at interpreter
  start-up when the directory is on ``PYTHONPATH`` — that is what makes the
  hook reach the *worker* process without touching vllm's launcher.

What is captured per step (JSONL, one line per step, schema v1):

    SchedulerOutput (field-level, JSON-safe)  <- argument of execute_model
    input_batch snapshot                      <- at _prepare_inputs() entry
    model_output (sampled_token_ids)          <- return of execute_model/sample_tokens
    timings (µs)                              <- for _update_states/_prepare_inputs

Cost control:

* records are accumulated in a list and written when ``PI_CAPTURE_FLUSH_EVERY``
  steps have accumulated (default 64) and at process exit (``atexit``);
* ``PI_CAPTURE_MAX_STEPS`` stops capturing (the wrappers become pass-through
  calls) after N steps, default 0 = unlimited;
* ``PI_CAPTURE_TIMING=0`` / ``PI_CAPTURE_INPUT_BATCH=0`` disable the two extra
  monkeypatches (execute_model is still wrapped).
"""

from __future__ import annotations

import functools
import importlib
import json
import os
import threading
import time
from typing import Any

from pi_harness.workload.schema import (
    SCHEMA_VERSION,
    StepRecord,
    encode_input_batch,
    encode_model_output,
    encode_req_state,
    encode_scheduler_output,
    infer_phase,
)

DEFAULT_TARGET = "vllm_ascend.worker.model_runner_v1:NPUModelRunner"

__all__ = ["install", "maybe_install", "Capture", "CaptureStats"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class CaptureStats:
    """Cheap counters (returned by :meth:`Capture.stats`)."""

    __slots__ = (
        "target",
        "path",
        "steps_recorded",
        "steps_flushed",
        "flushes",
        "errors",
        "enabled",
        "max_steps",
        "flush_every",
    )

    def __init__(self, target: str, path: str, max_steps: int, flush_every: int):
        self.target = target
        self.path = path
        self.steps_recorded = 0
        self.steps_flushed = 0
        self.flushes = 0
        self.errors = 0
        self.enabled = True
        self.max_steps = max_steps
        self.flush_every = flush_every

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class _Pending:
    __slots__ = ("record", "t0_ns", "scheduler_output")

    def __init__(self, record: StepRecord, t0_ns: int, scheduler_output: Any):
        self.record = record
        self.t0_ns = t0_ns
        self.scheduler_output = scheduler_output


class Capture:
    """Monkeypatch installer/holder for one model-runner class."""

    def __init__(
        self,
        path: str,
        *,
        target: str = DEFAULT_TARGET,
        max_steps: int = 0,
        flush_every: int = 64,
        record_timing: bool = True,
        record_input_batch: bool = True,
        note: str | None = None,
    ) -> None:
        self.path = path
        self.target = target
        self.stats = CaptureStats(target, path, max_steps, flush_every)
        self.record_timing = record_timing
        self.record_input_batch = record_input_batch
        self.note = note or os.environ.get("PI_CAPTURE_NOTE", "")
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        self._step_idx = 0
        self._seen_reqs: set[str] = set()
        self._installed_cls: type | None = None
        self._orig: dict[str, Any] = {}
        self._closed = False

    # -- installation --------------------------------------------------------
    def install(self, cls: type | None = None) -> "Capture":
        cls = cls or _resolve_target(self.target)
        self._installed_cls = cls
        self._orig["execute_model"] = cls.execute_model
        cls.execute_model = self._wrap_execute_model(cls.execute_model)
        if hasattr(cls, "sample_tokens"):
            self._orig["sample_tokens"] = cls.sample_tokens
            cls.sample_tokens = self._wrap_sample_tokens(cls.sample_tokens)
        if self.record_timing and hasattr(cls, "_update_states"):
            self._orig["_update_states"] = cls._update_states
            cls._update_states = self._wrap_update_states(cls._update_states)
        if hasattr(cls, "_prepare_inputs"):
            self._orig["_prepare_inputs"] = cls._prepare_inputs
            cls._prepare_inputs = self._wrap_prepare_inputs(cls._prepare_inputs)
        return self

    # -- step lifecycle ------------------------------------------------------
    def _begin(self, model_runner: Any, scheduler_output: Any) -> _Pending | None:
        if not self.stats.enabled:
            return None
        try:
            so_dict = encode_scheduler_output(scheduler_output)
        except Exception:
            self.stats.errors += 1
            return None
        rec = StepRecord(
            step_idx=self._step_idx,
            phase="unknown",
            scheduler_output=so_dict,
            req_states=[],
            model_output=None,
            t_prepare_input_us=None,
            input_batch=None,
            timings={},
            meta={
                "source": "capture",
                "synth": False,
                "schema_version": SCHEMA_VERSION,
                "note": self.note,
            },
        )
        pending = _Pending(rec, time.perf_counter_ns(), scheduler_output)
        self._pending = pending
        return pending

    def _finalize(self, model_runner: Any, output: Any) -> None:
        pending = self._pending
        if pending is None:
            return
        rec = pending.record
        try:
            if rec.input_batch is None and self.record_input_batch:
                ib = getattr(model_runner, "input_batch", None)
                if ib is not None:
                    rec.input_batch = encode_input_batch(ib)
            rec.timings["execute_model_us"] = round(
                (time.perf_counter_ns() - pending.t0_ns) / 1000.0, 3
            )
            if output is not None:
                enc = encode_model_output(output)
                if enc is not None:
                    rec.model_output = enc
            self._capture_req_states(model_runner, rec)
            rec.phase = infer_phase(rec.scheduler_output, rec.req_states)
        except Exception:
            self.stats.errors += 1
        self._push(rec)
        self._pending = None

    def _capture_req_states(self, model_runner: Any, rec: StepRecord) -> None:
        if rec.req_states:
            return
        for rid in rec.scheduler_output.get("finished_req_ids") or []:
            self._seen_reqs.discard(rid)
        requests = getattr(model_runner, "requests", None)
        if not isinstance(requests, dict):
            return
        ib = getattr(model_runner, "input_batch", None)
        rid_order = list(getattr(ib, "req_ids", []) or [])
        sched_ids = list((rec.scheduler_output.get("num_scheduled_tokens") or {}))
        seen: set[str] = set()
        out: list[dict] = []
        for rid in list(rid_order) + [r for r in sched_ids if r not in rid_order]:
            if rid in seen:
                continue
            st = requests.get(rid)
            if st is None:
                continue
            seen.add(rid)
            enc = encode_req_state(rid, st)
            if rid in self._seen_reqs:
                # prompt ids are only carried when the request first shows up;
                # afterwards the reader keeps them in its own cache
                enc["prompt_token_ids"] = None
            else:
                self._seen_reqs.add(rid)
            out.append(enc)
        rec.req_states = out

    def _push(self, rec: StepRecord) -> None:
        self._buf.append(rec.to_dict())
        self.stats.steps_recorded += 1
        self._step_idx += 1
        if self.stats.max_steps and self._step_idx >= self.stats.max_steps:
            self.stats.enabled = False
            self.flush()
            return
        if self.stats.flush_every and len(self._buf) >= self.stats.flush_every:
            self.flush()

    # -- output --------------------------------------------------------------
    def flush(self) -> int:
        with self._lock:
            if not self._buf:
                return 0
            batch, self._buf = self._buf, []
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for d in batch:
                fh.write(json.dumps(d, separators=(",", ":"), ensure_ascii=False))
                fh.write("\n")
        self.stats.steps_flushed += len(batch)
        self.stats.flushes += 1
        return len(batch)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()
        self.uninstall()

    def uninstall(self) -> None:
        cls = self._installed_cls
        if cls is None:
            return
        for name, orig in self._orig.items():
            try:
                setattr(cls, name, orig)
            except Exception:  # pragma: no cover
                pass
        self._installed_cls = None

    def __enter__(self) -> "Capture":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- wrappers ------------------------------------------------------------
    def _wrap_execute_model(self, orig: Any) -> Any:
        @functools.wraps(orig)
        def wrapper(self_mr: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.stats.enabled:
                return orig(self_mr, scheduler_output, *args, **kwargs)
            pending = self._begin(self_mr, scheduler_output)
            try:
                out = orig(self_mr, scheduler_output, *args, **kwargs)
            except BaseException:
                self._pending = None
                raise
            if pending is not None and out is not None:
                # execute_model returned the sampled output itself (no split
                # sampling); otherwise sample_tokens() finalises it.
                self._finalize(self_mr, out)
            return out

        return wrapper

    def _wrap_sample_tokens(self, orig: Any) -> Any:
        @functools.wraps(orig)
        def wrapper(self_mr: Any, *args: Any, **kwargs: Any) -> Any:
            out = orig(self_mr, *args, **kwargs)
            try:
                if self._pending is not None:
                    self._finalize(self_mr, out)
            except Exception:
                self.stats.errors += 1
            return out

        return wrapper

    def _wrap_update_states(self, orig: Any) -> Any:
        @functools.wraps(orig)
        def wrapper(self_mr: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.stats.enabled:
                return orig(self_mr, scheduler_output, *args, **kwargs)
            t0 = time.perf_counter_ns()
            try:
                return orig(self_mr, scheduler_output, *args, **kwargs)
            finally:
                pending = self._pending
                if pending is not None:
                    pending.record.timings["update_states_us"] = round(
                        (time.perf_counter_ns() - t0) / 1000.0, 3
                    )

        return wrapper

    def _wrap_prepare_inputs(self, orig: Any) -> Any:
        @functools.wraps(orig)
        def wrapper(self_mr: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.stats.enabled:
                return orig(self_mr, *args, **kwargs)
            t0 = time.perf_counter_ns()
            pending = self._pending
            if pending is not None and self.record_input_batch:
                try:
                    ib = getattr(self_mr, "input_batch", None)
                    if ib is not None:
                        pending.record.input_batch = encode_input_batch(
                            ib, max_rows=len(getattr(ib, "req_ids", []) or []) or None
                        )
                except Exception:
                    self.stats.errors += 1
            try:
                return orig(self_mr, *args, **kwargs)
            finally:
                if pending is not None:
                    us = round((time.perf_counter_ns() - t0) / 1000.0, 3)
                    pending.record.timings["prepare_inputs_us"] = us
                    us_upd = float(pending.record.timings.get("update_states_us") or 0.0)
                    pending.record.t_prepare_input_us = round(us_upd + us, 3)

        return wrapper


def _resolve_target(spec: str) -> type:
    if ":" not in spec:
        raise ValueError(f"PI_CAPTURE_TARGET must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in cls_name.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"{spec} is not a class")
    return obj


_ACTIVE: Capture | None = None


def install(
    path: str | None = None,
    *,
    cls: type | None = None,
    target: str | None = None,
    max_steps: int | None = None,
    flush_every: int | None = None,
) -> Capture | None:
    """Install the capture hook.  Returns ``None`` when disabled by env.

    ``path`` defaults to ``$PI_CAPTURE``; if that is unset nothing happens, so
    callers can invoke this unconditionally in production code paths.
    """
    global _ACTIVE
    path = path or os.environ.get("PI_CAPTURE")
    if not path:
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    target = target or os.environ.get("PI_CAPTURE_TARGET", DEFAULT_TARGET)
    cap = Capture(
        path,
        target=target,
        max_steps=_env_int("PI_CAPTURE_MAX_STEPS", 0) if max_steps is None else max_steps,
        flush_every=(
            _env_int("PI_CAPTURE_FLUSH_EVERY", 64) if flush_every is None else flush_every
        ),
        record_timing=_env_bool("PI_CAPTURE_TIMING", True),
        record_input_batch=_env_bool("PI_CAPTURE_INPUT_BATCH", True),
    )
    cap.install(cls)
    import atexit

    atexit.register(cap.close)
    _ACTIVE = cap
    return cap


def maybe_install() -> Capture | None:
    """``install()``, swallowing import errors (used by sitecustomize)."""
    try:
        return install()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[pi_harness.trace.capture] install failed: {exc!r}", flush=True)
        return None


def active() -> Capture | None:
    return _ACTIVE
