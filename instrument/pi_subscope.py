"""Sub-scope timing probes for the vLLM-Ascend ``prepare input`` phase.

This module is *instrumentation only*: it must never change inference
behaviour and must be free of cost when disabled.

Design notes (see ``instrument/README.md`` and
``agents/subscope_instrumentation/REPORT.md`` for the measured numbers):

* Opt-in via ``PI_SUBSCOPE=on``.  When it is not ``on``:
    - :func:`pi_prepare_input` returns the caller's context manager unchanged
      (not even wrapped), so the instrumented build executes *exactly* the
      same bytecode as the baseline build at the ``prepare input`` call site;
    - :func:`pi_scope` returns a shared ``nullcontext``.
* Thread-local state.  vLLM's worker main thread is the only thread that runs
  the model runner, but ``LiteProfiler`` established the same convention, so
  records carry the native thread id.
* No per-step I/O.  Records are accumulated in memory and appended to
  ``PI_SUBSCOPE_LOG`` every ``PI_SUBSCOPE_DUMP_EVERY`` steps (one open/write
  per flush) and once at interpreter exit.
* Exclusivity invariant.  For every scope we record
  ``self_us`` (exclusive) *and* ``incl_us`` (self + children).  Because every
  probe is properly nested, ``sum(self_us over all scopes except pi: TOTAL)``
  must be ``<= pi: TOTAL.self_us``; the gap is time spent in the phase that
  no probe covers (``pi: (unattributed)`` in the parser output).

Row schema written to ``PI_SUBSCOPE_LOG``::

    step,phase,scope,self_us,incl_us,max_us,n_calls,wall_us,num_reqs,total_tokens,tid,pid

``wall_us`` is the wall time of the whole ``prepare input`` scope for that
step, repeated on every row of the step so that a consumer can join on it.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time
from typing import Any

__all__ = [
    "PI_ENABLED",
    "flush_notes",
    "lazy",
    "pi_flush",
    "pi_note",
    "pi_prepare_input",
    "pi_scope",
    "pi_step_info",
]

_TRUE = ("1", "on", "true", "yes")

PI_ENABLED: bool = os.getenv("PI_SUBSCOPE", "off").strip().lower() in _TRUE
LOG_PATH: str = os.getenv("PI_SUBSCOPE_LOG", "") or ""
META_PATH: str = os.getenv("PI_SUBSCOPE_META", "") or ""
DUMP_EVERY: int = max(1, int(os.getenv("PI_SUBSCOPE_DUMP_EVERY", "25") or 25))

_perf = time.perf_counter_ns
_nullcontext = contextlib.nullcontext
_NULL = _nullcontext()

HEADER = (
    "step,phase,scope,self_us,incl_us,max_us,n_calls,wall_us,"
    "num_reqs,total_tokens,tid,pid\n"
)


class _TLS(threading.local):
    """Per-thread probe state.  ``__init__`` runs once per thread."""

    def __init__(self) -> None:
        self.stack: list[list] = []
        self.self_ns: dict[str, int] = {}
        self.incl_ns: dict[str, int] = {}
        self.max_ns: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self.step: int = 0
        self.wall_ns: int = 0
        self.phase: str = "unknown"
        self.num_reqs: int = -1
        self.total_tokens: int = -1


_state = _TLS()

_lock = threading.Lock()
_pending: list[str] = []
_steps_since_flush = 0
# Run-level notes (structure, not timings): attn group identity, probe hashes...
# written once to META_PATH at exit so a parser can explain *what* was measured.
_notes: dict[str, Any] = {}
_notes_lock = threading.Lock()
_notes_dirty = False


class _LazyNote:
    """Deferred value for :func:`pi_note_lazy`.

    Wrapping the value in a callable keeps the *argument expression* inside the
    probe: if building the diagnostic dict raises (e.g. a builder that does not
    have the attribute the probe assumes), the exception is raised during the
    call, inside ``pi_note_lazy``'s ``try`` block, instead of escaping into the
    inference path.  A real incident on 2026-09-24 made this necessary:
    ``len(self.layer_names)`` raised ``AttributeError`` inside
    ``AscendGDNAttentionMetadataBuilder.build`` and the engine failed to start.
    """

    __slots__ = ("_fn",)

    def __init__(self, fn) -> None:
        self._fn = fn

    def __call__(self) -> Any:
        return self._fn()


def lazy(fn):
    """Mark a callable as a deferred note value (see :class:`_LazyNote`)."""
    return _LazyNote(fn)


def pi_note(key: str, value: Any) -> None:
    """Record a run-level fact (not a timing).  No-op when disabled.

    Keys are stable strings; the first value wins, so a per-step call site can
    note something once without a guard.  Failures are swallowed: notes are
    diagnostics, never part of the inference path.
    """
    if not PI_ENABLED:
        return
    try:
        global _notes_dirty
        if isinstance(value, _LazyNote):
            value = value()
        with _notes_lock:
            if str(key) not in _notes:
                _notes[str(key)] = value
                _notes_dirty = True
    except Exception:
        pass


def _write_notes() -> None:
    """Write the note file if there is anything new.

    Called on every flush (not only at exit) because a long-running service is
    usually inspected *while* it is up: the notes must be on disk before the
    profiler stage reads them.  The write is skipped unless a new key was
    recorded, so the steady-state cost is one boolean check per flush.
    """
    global _notes_dirty
    if not META_PATH or not _notes or not _notes_dirty:
        return
    try:
        import json

        with _notes_lock:
            payload = {
                "pid": os.getpid(),
                "log_path": LOG_PATH,
                "n_notes": len(_notes),
                "notes": dict(_notes),
            }
        tmp = META_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, META_PATH)
        _notes_dirty = False
    except Exception:
        pass


def _acc(st: _TLS, name: str, self_ns: int, dt: int) -> None:
    sn = st.self_ns
    sn[name] = sn.get(name, 0) + self_ns
    inc = st.incl_ns
    inc[name] = inc.get(name, 0) + dt
    mx = st.max_ns
    if dt > mx.get(name, -1):
        mx[name] = dt
    cl = st.calls
    cl[name] = cl.get(name, 0) + 1


class _Scope:
    """Timing probe; one instance per ``with pi_scope(...)`` site."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> "_Scope":
        st = _state
        st.stack.append([self._name, _perf(), 0])
        return self

    def __exit__(self, *exc: object) -> bool:
        # Broad exception handling on purpose: instrumentation must never
        # propagate into the inference path.
        try:
            st = _state
            name, t0, child = st.stack.pop()
            dt = _perf() - t0
            self_ns = dt - child
            if self_ns < 0:
                self_ns = 0
            _acc(st, name, self_ns, dt)
            if st.stack:
                st.stack[-1][2] += dt
        except Exception:
            pass
        return False


def pi_scope(name: str):
    """Return the probe context manager for ``name`` (nullcontext when off)."""
    if not PI_ENABLED:
        return _NULL
    return _Scope(name)


class _Step:
    """Wraps the LiteProfiler ``prepare input`` scope for one engine step."""

    __slots__ = ("_inner", "_t0")

    def __init__(self, inner) -> None:
        self._inner = inner
        self._t0 = 0

    def __enter__(self):
        st = _state
        if st.stack:
            # Left-over band from an exception-interrupted step; drop it so a
            # broken probe can never corrupt the next step's accounting.
            st.stack.clear()
        st.self_ns.clear()
        st.incl_ns.clear()
        st.max_ns.clear()
        st.calls.clear()
        st.step += 1
        st.phase = "unknown"
        st.num_reqs = -1
        st.total_tokens = -1
        st.wall_ns = 0
        self._t0 = _perf()
        st.stack.append(["pi: TOTAL", _perf(), 0])
        return self._inner.__enter__()

    def __exit__(self, *exc: object) -> bool:
        res = self._inner.__exit__(*exc)
        try:
            st = _state
            st.stack.pop()
            st.wall_ns = _perf() - self._t0
        except Exception:
            pass
        _emit_step()
        return bool(res)


def pi_prepare_input(inner):
    """Wrap the ``prepare input`` record_function context manager.

    When the probes are disabled this returns ``inner`` itself, so there is no
    additional Python frame on the hot path.
    """
    if not PI_ENABLED:
        return inner
    return _Step(inner)


def pi_step_info(phase: str, num_reqs: int, total_tokens: int) -> None:
    """Attach the step classification used by the parsers."""
    if not PI_ENABLED:
        return
    try:
        st = _state
        st.phase = phase
        st.num_reqs = num_reqs
        st.total_tokens = total_tokens
    except Exception:
        pass


def _emit_step() -> None:
    global _steps_since_flush
    if not PI_ENABLED:
        return
    try:
        st = _state
        if not st.self_ns:
            return
        wall = st.wall_ns
        tid = threading.get_native_id()
        pid = os.getpid()
        wall_us = wall / 1000.0
        rows = []
        for name, self_ns in st.self_ns.items():
            rows.append(
                f"{st.step},{st.phase},{name},{self_ns / 1000.0:.3f},"
                f"{st.incl_ns[name] / 1000.0:.3f},{st.max_ns[name] / 1000.0:.3f},"
                f"{st.calls[name]},{wall_us:.3f},{st.num_reqs},{st.total_tokens},"
                f"{tid},{pid}"
            )
        rows.append(
            f"{st.step},{st.phase},pi: TOTAL,{wall_us:.3f},{wall_us:.3f},"
            f"{wall_us:.3f},1,{wall_us:.3f},{st.num_reqs},{st.total_tokens},"
            f"{tid},{pid}"
        )
        _steps_since_flush += 1
        with _lock:
            _pending.extend(rows)
            if _steps_since_flush >= DUMP_EVERY:
                _flush_locked()
        # Keep probe_notes.json fresh while the service is still running.
        _write_notes()
    except Exception:
        pass


def _flush_locked() -> None:
    global _steps_since_flush
    if not _pending or not LOG_PATH:
        _pending.clear()
        _steps_since_flush = 0
        return
    try:
        need_header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
        with open(LOG_PATH, "a") as fh:
            if need_header:
                fh.write(HEADER)
            fh.write("\n".join(_pending))
            fh.write("\n")
    except OSError:
        pass
    _pending.clear()
    _steps_since_flush = 0


def flush_notes() -> None:
    """Force a notes write (used by tests)."""
    _write_notes()


def pi_flush() -> None:
    """Force a flush (used by ``atexit`` and by the smoke test)."""
    with _lock:
        _flush_locked()


def _reset_thread_state() -> None:
    st = _state
    st.stack.clear()
    st.self_ns.clear()
    st.incl_ns.clear()
    st.max_ns.clear()
    st.calls.clear()


def _flush_thread_state() -> None:
    """Close any band left open by an exception, flush timings and notes."""
    _reset_thread_state()
    pi_flush()
    _write_notes()


if PI_ENABLED:
    atexit.register(_flush_thread_state)
