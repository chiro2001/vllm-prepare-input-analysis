"""Record/replay schema shared by the capture hook, the synthetic workload and
the A/B runner.

This module is intentionally *stdlib-only at import time*.  It never imports
``torch`` / ``vllm`` at module scope so that the capture hook can be imported
inside a normal serving process (and inside local unit tests without a card).
Objects that only exist at runtime (``SchedulerOutput``, ``ModelRunnerOutput``)
are reconstructed lazily inside :func:`decode_scheduler_output` /
:func:`decode_model_output`, which are the only functions that import ``vllm``.

Schema (v1)
-----------
One JSONL line == one :class:`StepRecord` (see ``INTERFACES.md`` §3).  Extra
fields (``input_batch``, ``timings``, ``meta``) are additive and optional; a
reader that only knows the v1 fields must ignore them.

Two conventions keep traces small and are part of the contract:

* ``req_states[i]["prompt_token_ids"]`` is set only on the step where the
  request enters the worker batch; later steps carry ``None`` (readers keep a
  per-request cache).  The full prompt is still present in
  ``scheduler_output.scheduled_new_reqs[].prompt_token_ids``.
* ``input_batch.token_ids_cpu_shape`` is a *shape*, not the token buffer
  content; the values are reconstructible from the prompts + sampled ids.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# StepRecord
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One scheduling step, exactly what ``_update_states``/``_prepare_inputs``
    consume (``scheduler_output``) plus the runtime state they read
    (``req_states`` / ``input_batch``) and what they produced (``model_output``).

    ``__synth__`` from INTERFACES.md lives in ``meta["source"]``
    (``"synth"`` / ``"capture"`` / ``"shuffled"``).
    """

    step_idx: int
    phase: str  # prefill | decode | mixed | spec-decode
    scheduler_output: dict  # JSON-safe vllm SchedulerOutput
    req_states: list[dict]  # per request runtime state
    model_output: dict | None  # sampling result (feeds scheduler.update_from_output)
    t_prepare_input_us: float | None = None
    # --- additive extensions -------------------------------------------------
    input_batch: dict | None = None  # InputBatch snapshot (req_ids/…/block_table)
    timings: dict | None = None  # sub-step timings (µs), when measured
    meta: dict | None = None  # source markers / generator params

    # -- helpers -------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StepRecord":
        known = {f.name for f in fields(cls)}
        extra = {k: v for k, v in d.items() if k not in known}
        rec = cls(**{k: v for k, v in d.items() if k in known})
        if extra:
            rec.meta = {**(rec.meta or {}), "_extra": extra}
        return rec

    @property
    def num_reqs(self) -> int:
        return len(self.scheduler_output.get("num_scheduled_tokens") or {})

    @property
    def total_num_scheduled_tokens(self) -> int:
        so = self.scheduler_output
        return int(
            so.get("total_num_scheduled_tokens")
            or sum((so.get("num_scheduled_tokens") or {}).values())
        )

    def is_synth(self) -> bool:
        return bool((self.meta or {}).get("synth", False))


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------


def save_jsonl(recs: Iterable[StepRecord], path: str | os.PathLike) -> int:
    """Write records as one compact JSON object per line.  Returns line count."""
    path = os.fspath(path)
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in recs:
            d = rec.to_dict() if isinstance(rec, StepRecord) else dict(rec)
            d.setdefault("schema_version", SCHEMA_VERSION)
            fh.write(json.dumps(d, separators=(",", ":"), ensure_ascii=False))
            fh.write("\n")
            n += 1
    os.replace(tmp, path)
    return n


def load_jsonl(path: str | os.PathLike) -> list[StepRecord]:
    """Load a trace file produced by ``save_jsonl`` or by the capture hook."""
    out: list[StepRecord] = []
    with open(os.fspath(path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(StepRecord.from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# JSON-safe encoding of runtime objects (duck typed: no vllm import needed)
# ---------------------------------------------------------------------------

_MISSING = object()


def _g(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _json_safe(value: Any, depth: int = 0) -> tuple[Any, bool]:
    """Best-effort JSON-safe conversion.  Returns ``(value, ok)``.

    ``ok=False`` marks a value that could not be represented exactly; the
    caller records the lossy marker instead of silently dropping it.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value, True
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return repr(value), False
        return value, True
    if depth > 6:
        return {"__unsupported__": type(value).__name__, "__reason__": "depth"}, False
    if isinstance(value, (list, tuple, set, frozenset)):
        items = []
        ok = True
        for v in value:
            jv, vok = _json_safe(v, depth + 1)
            items.append(jv)
            ok = ok and vok
        return items, ok
    if isinstance(value, dict):
        out: dict = {}
        ok = True
        for k, v in value.items():
            jv, vok = _json_safe(v, depth + 1)
            out[str(k)] = jv
            ok = ok and vok
        return out, ok
    # numpy scalars / arrays and torch tensors: shape-only summary.
    shape = _g(value, "shape", _MISSING)
    if shape is not _MISSING and not isinstance(shape, (int, float, str)):
        try:
            shape_list = [int(s) for s in shape]
        except TypeError:
            shape_list = []
        try:
            tolist = value.tolist()  # type: ignore[attr-defined]
            jv, ok = _json_safe(tolist, depth + 1)
            return {"__ndarray__": True, "shape": shape_list, "data": jv}, ok
        except Exception:
            return {
                "__unsupported__": type(value).__name__,
                "shape": shape_list,
                "dtype": str(_g(value, "dtype", None)),
            }, False
    try:
        tolist = value.tolist()  # numpy scalar
        return _json_safe(tolist, depth + 1)
    except Exception:
        pass
    if isinstance(value, (bytes, bytearray)):
        return value.hex(), True
    return {"__unsupported__": type(value).__name__}, False


def _encode_block_ids(block_ids: Any) -> Any:
    """``tuple[list[int], ...]`` (per KV cache group) -> list[list[int]]."""
    if block_ids is None:
        return None
    return [[int(b) for b in group] for group in block_ids]


def encode_sampling_params(sp: Any) -> dict | None:
    """Encode ``SamplingParams`` losslessly for the fields that are primitives.

    Everything that is JSON-safe from ``__dict__`` is kept; the rest is listed
    in ``__unsupported_fields__`` so the fidelity table can name the gap.
    """
    if sp is None:
        return None
    if isinstance(sp, dict):
        return sp
    out: dict = {"__type__": type(sp).__name__}
    try:
        out["__repr__"] = repr(sp)[:1024]
    except Exception:
        pass
    prims: dict = {}
    unsupported: list[str] = []
    for k, v in vars(sp).items() if hasattr(sp, "__dict__") else []:
        jv, ok = _json_safe(v)
        if ok:
            prims[k] = jv
        else:
            unsupported.append(k)
    if not prims:
        # fall back to the documented public fields
        for k in (
            "n",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "ignore_eos",
            "stop",
            "stop_token_ids",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "include_stop_str_in_output",
        ):
            v = _g(sp, k, _MISSING)
            if v is not _MISSING:
                jv, ok = _json_safe(v)
                if ok:
                    prims[k] = jv
                else:
                    unsupported.append(k)
    out["primitives"] = prims
    if unsupported:
        out["__unsupported_fields__"] = sorted(set(unsupported))
    return out


def decode_sampling_params(d: dict | None) -> Any:
    """Rebuild a real ``SamplingParams`` when possible (lazy vllm import)."""
    if d is None:
        return None
    if not isinstance(d, dict) or "primitives" not in d:
        return d
    prims = dict(d.get("primitives") or {})
    try:
        import inspect

        from vllm.sampling_params import SamplingParams

        ctor = set(inspect.signature(SamplingParams.__init__).parameters)
        kwargs = {k: v for k, v in prims.items() if k in ctor and k != "self"}
        obj = SamplingParams(**kwargs)
        for k, v in prims.items():
            if k not in ctor:
                try:
                    setattr(obj, k, v)
                except Exception:
                    pass
        return obj
    except Exception:
        return _StubSamplingParams(prims)


class _StubSamplingParams:
    """Fallback container when ``SamplingParams(**prims)`` is not constructible."""

    def __init__(self, prims: dict):
        self.__dict__.update(prims)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StubSamplingParams({sorted(self.__dict__)})"


def encode_mm_features(mm_features: Any) -> list[dict]:
    """Structural (lossy for tensors) encoding of ``MultiModalFeatureSpec``."""
    out = []
    for feat in mm_features or []:
        item: dict = {"__type__": type(feat).__name__}
        for k, v in vars(feat).items() if hasattr(feat, "__dict__") else []:
            jv, ok = _json_safe(v)
            item[k] = jv if ok else {"__unsupported__": type(v).__name__}
        if len(item) == 1:
            for f in getattr(type(feat), "__dataclass_fields__", {}) or {}:
                item[f] = {"__unsupported__": "dataclass-field"}
        out.append(item)
    return out


def new_request_data_to_dict(rd: Any) -> dict:
    """``NewRequestData`` -> JSON-safe dict (see ``vllm/v1/core/sched/output.py``)."""
    if isinstance(rd, dict):
        return rd
    prompt_embeds = _g(rd, "prompt_embeds", None)
    pe_shape = None
    if prompt_embeds is not None:
        try:
            pe_shape = [int(s) for s in prompt_embeds.shape]
        except Exception:
            pe_shape = None
    return {
        "req_id": _g(rd, "req_id"),
        "prompt_token_ids": _g(rd, "prompt_token_ids"),
        "mm_features": encode_mm_features(_g(rd, "mm_features", [])),
        "sampling_params": encode_sampling_params(_g(rd, "sampling_params")),
        "pooling_params": encode_sampling_params(_g(rd, "pooling_params")),
        "block_ids": _encode_block_ids(_g(rd, "block_ids", [])),
        "num_computed_tokens": int(_g(rd, "num_computed_tokens", 0)),
        "lora_request": _g(rd, "lora_request"),  # None for our workloads
        "prompt_embeds_shape": pe_shape,
        "prompt_is_token_ids": _g(rd, "prompt_is_token_ids"),
        "prefill_token_ids": _g(rd, "prefill_token_ids"),
    }


def cached_request_data_to_dict(rd: Any) -> dict:
    """``CachedRequestData`` -> JSON-safe dict."""
    if isinstance(rd, dict):
        return rd
    resumed = _g(rd, "resumed_req_ids", set()) or set()
    return {
        "req_ids": list(_g(rd, "req_ids", []) or []),
        "resumed_req_ids": sorted(resumed),
        "new_token_ids": [list(x) for x in (_g(rd, "new_token_ids", []) or [])],
        "all_token_ids": {
            str(k): list(v) for k, v in (_g(rd, "all_token_ids", {}) or {}).items()
        },
        "new_block_ids": [
            _encode_block_ids(x) for x in (_g(rd, "new_block_ids", []) or [])
        ],
        "num_computed_tokens": [int(x) for x in (_g(rd, "num_computed_tokens", []) or [])],
        "num_output_tokens": [int(x) for x in (_g(rd, "num_output_tokens", []) or [])],
    }


def _encode_kv_block_copies(copies: Any) -> list[dict] | None:
    if copies is None:
        return None
    out = []
    for c in copies:
        if isinstance(c, dict):
            out.append(dict(c))
        else:
            out.append(
                {
                    "src_block_id": int(_g(c, "src_block_id", -1)),
                    "dst_block_id": int(_g(c, "dst_block_id", -1)),
                }
            )
    return out


def _encode_connector_meta(meta: Any) -> dict | None:
    if meta is None:
        return None
    # Connector metadata is arbitrary; record the marker only (workers rebuild
    # it from the connector itself in a real replay).
    return {"__unsupported__": type(meta).__name__}


def encode_scheduler_output(so: Any) -> dict:
    """``vllm.v1.core.sched.output.SchedulerOutput`` -> JSON-safe dict."""
    if isinstance(so, dict):
        return so
    enc_stats = _g(so, "scheduled_encoder_input_stats", None)
    return {
        "scheduled_new_reqs": [
            new_request_data_to_dict(r) for r in (_g(so, "scheduled_new_reqs", []) or [])
        ],
        "scheduled_cached_reqs": cached_request_data_to_dict(
            _g(so, "scheduled_cached_reqs", None)
        ),
        "num_scheduled_tokens": {
            str(k): int(v)
            for k, v in (_g(so, "num_scheduled_tokens", {}) or {}).items()
        },
        "total_num_scheduled_tokens": int(_g(so, "total_num_scheduled_tokens", 0)),
        "scheduled_spec_decode_tokens": {
            str(k): [int(t) for t in v]
            for k, v in (_g(so, "scheduled_spec_decode_tokens", {}) or {}).items()
        },
        "scheduled_encoder_inputs": {
            str(k): [int(t) for t in v]
            for k, v in (_g(so, "scheduled_encoder_inputs", {}) or {}).items()
        },
        "num_common_prefix_blocks": [
            int(x) for x in (_g(so, "num_common_prefix_blocks", []) or [])
        ],
        "finished_req_ids": sorted(_g(so, "finished_req_ids", set()) or set()),
        "free_encoder_mm_hashes": list(_g(so, "free_encoder_mm_hashes", []) or []),
        "scheduled_encoder_input_stats": (
            None
            if enc_stats is None
            else {
                "num_inputs": int(_g(enc_stats, "num_inputs", 0)),
                "output_tokens": int(_g(enc_stats, "output_tokens", 0)),
            }
        ),
        "preempted_req_ids": (
            None
            if _g(so, "preempted_req_ids", None) is None
            else sorted(_g(so, "preempted_req_ids"))
        ),
        "has_structured_output_requests": bool(
            _g(so, "has_structured_output_requests", False)
        ),
        "pending_structured_output_tokens": bool(
            _g(so, "pending_structured_output_tokens", False)
        ),
        "num_invalid_spec_tokens": _g(so, "num_invalid_spec_tokens", None),
        "kv_connector_metadata": _encode_connector_meta(
            _g(so, "kv_connector_metadata", None)
        ),
        "ec_connector_metadata": _encode_connector_meta(
            _g(so, "ec_connector_metadata", None)
        ),
        "new_block_ids_to_zero": (
            None
            if _g(so, "new_block_ids_to_zero", None) is None
            else [int(x) for x in _g(so, "new_block_ids_to_zero")]
        ),
        "kv_cache_block_copies": _encode_kv_block_copies(
            _g(so, "kv_cache_block_copies", None)
        ),
        "num_spec_tokens_to_schedule": int(_g(so, "num_spec_tokens_to_schedule", 0)),
    }


# --- runtime state ---------------------------------------------------------


def encode_req_state(req_id: str, st: Any) -> dict:
    """Encode one ``CachedRequestState`` (worker-side ``self.requests[req_id]``)."""
    if isinstance(st, dict):
        return st
    output_token_ids = _g(st, "output_token_ids", []) or []
    spec_token_ids = _g(st, "spec_token_ids", None)
    if spec_token_ids is None:
        spec_token_ids = _g(st, "spec_token_ids", []) or []
    prompt_token_ids = _g(st, "prompt_token_ids", None)
    return {
        "req_id": req_id,
        "prompt_token_ids": prompt_token_ids,
        "num_prompt_tokens": int(
            _g(st, "num_prompt_tokens", len(prompt_token_ids or []))
        ),
        "num_computed_tokens": int(_g(st, "num_computed_tokens", 0)),
        "block_ids": _encode_block_ids(_g(st, "block_ids", [])),
        "num_output_tokens": len(output_token_ids),
        "spec_len": len(spec_token_ids),
        "num_tokens": int(_g(st, "num_tokens", 0) or 0),
    }


def encode_model_output(mo: Any) -> dict | None:
    """Encode ``ModelRunnerOutput`` (only the sampling fields that matter)."""
    if mo is None:
        return None
    if isinstance(mo, dict):
        return mo
    sampled = _g(mo, "sampled_token_ids", None)
    if sampled is None:
        return None
    return {
        "req_ids": list(_g(mo, "req_ids", []) or []),
        "req_id_to_index": {
            str(k): int(v) for k, v in (_g(mo, "req_id_to_index", {}) or {}).items()
        },
        "sampled_token_ids": [[int(t) for t in row] for row in sampled],
    }


def encode_input_batch(ib: Any, max_rows: int | None = None) -> dict:
    """Snapshot the ``InputBatch`` fields ``_update_states``/``_prepare_inputs``
    actually read: request order, per-req counters, block tables and the token
    id buffer shape."""
    if ib is None:
        return {}
    if isinstance(ib, dict):
        return ib
    req_ids = list(_g(ib, "req_ids", []) or [])
    n = len(req_ids) if max_rows is None else min(len(req_ids), max_rows)

    def _rows(cpu_arr: Any, rows: int) -> list[int]:
        if cpu_arr is None:
            return []
        try:
            return [int(x) for x in cpu_arr[:rows]]
        except TypeError:  # torch tensor
            return [int(x) for x in cpu_arr[:rows].tolist()]

    num_computed = _g(ib, "num_computed_tokens_cpu", None)
    num_prompt = _g(ib, "num_prompt_tokens_cpu", None)
    if num_prompt is None:
        num_prompt = _g(ib, "num_prompt_tokens", None)
    num_tokens_no_spec = _g(ib, "num_tokens_no_spec", None)

    token_ids_cpu = _g(ib, "token_ids_cpu", None)
    tshape = list(_g(ib, "shape", []) or [])
    if not tshape and token_ids_cpu is not None:
        tshape = [_g(token_ids_cpu, "shape", [])]
        try:
            tshape = [int(s) for s in tshape[0]]
        except Exception:
            tshape = []

    block_table: list[list[list[int]]] = []
    num_blocks_per_row: list[list[int]] = []
    mbt = _g(ib, "block_table", None)
    tables = _g(mbt, "block_tables", None) if mbt is not None else None
    if tables is None and mbt is not None and isinstance(mbt, list):
        tables = mbt
    for tbl in tables or []:
        bnp = _g(tbl, "block_table", None)
        nbpr = _g(tbl, "num_blocks_per_row", None)
        rows: list[list[int]] = []
        widths: list[int] = []
        for i in range(n):
            w = int(nbpr[i]) if nbpr is not None else 0
            widths.append(w)
            if bnp is None:
                rows.append([])
                continue
            arr = _g(bnp, "np", bnp)  # NumpyTensor wrapper -> np array
            try:
                rows.append([int(x) for x in arr[i, :w]])
            except Exception:
                rows.append([])
        block_table.append(rows)
        num_blocks_per_row.append(widths)

    spec_token_ids = _g(ib, "spec_token_ids", None)
    snap = {
        "req_ids": req_ids[:n],
        "num_computed_tokens_cpu": _rows(num_computed, n),
        "num_prompt_tokens_cpu": _rows(num_prompt, n),
        "num_tokens_no_spec": _rows(num_tokens_no_spec, n),
        "block_table": block_table,
        "num_blocks_per_row": num_blocks_per_row,
        "token_ids_cpu_shape": tshape,
        "token_ids_cpu_dtype": str(_g(token_ids_cpu, "dtype", None)),
        "spec_token_ids": [
            [int(t) for t in row] for row in (spec_token_ids or [])[:n]
        ],
        "num_reqs": n,
    }
    return snap


# ---------------------------------------------------------------------------
# phase classification (shared by capture + synth so the two are comparable)
# ---------------------------------------------------------------------------


def infer_phase(scheduler_output: dict, req_states: list[dict] | None = None) -> str:
    """Classify a step exactly the same way for captured and synthetic traces."""
    so = scheduler_output or {}
    nst = so.get("num_scheduled_tokens") or {}
    if not nst:
        # a drain step only carries `finished_req_ids`; the real worker runs
        # `_update_states` and then returns before `_prepare_inputs`
        return "drain" if so.get("finished_req_ids") else "empty"
    if so.get("scheduled_spec_decode_tokens"):
        return "spec-decode"
    computed = {}
    for st in req_states or []:
        computed[str(st.get("req_id"))] = int(st.get("num_computed_tokens", 0))
    multi = []
    single = []
    for rid, ntok in nst.items():
        if ntok > 1:
            multi.append(rid)
        else:
            single.append(rid)
    if multi and single:
        return "mixed"
    if multi:
        # chunked prefill vs a single big prefill: both are "prefill"
        return "prefill"
    # all requests scheduled exactly one token
    if computed and all(computed.get(rid, 0) == 0 for rid in nst):
        return "prefill"
    return "decode"


# ---------------------------------------------------------------------------
# shape signature (used by the A/B harness to prove "same shape")
# ---------------------------------------------------------------------------


def shape_signature(rec: StepRecord | dict) -> dict:
    """A content-free summary of one step: everything the A/B fixes constant."""
    if isinstance(rec, StepRecord):
        rec = rec.to_dict()
    so = rec.get("scheduler_output") or {}
    nst = so.get("num_scheduled_tokens") or {}
    req_states = rec.get("req_states") or []
    ib = rec.get("input_batch") or {}
    bt = ib.get("block_table") or []
    row_widths: list[int] = []
    for group in bt:
        for row in group:
            row_widths.append(len(row))
    per_req_tokens = sorted(int(v) for v in nst.values())
    counts: dict[int, int] = {}
    for v in per_req_tokens:
        counts[v] = counts.get(v, 0) + 1
    return {
        "step_idx": rec.get("step_idx"),
        "phase": rec.get("phase"),
        "num_reqs": len(nst),
        "total_num_scheduled_tokens": int(
            so.get("total_num_scheduled_tokens") or sum(nst.values())
        ),
        "num_new_reqs": len(so.get("scheduled_new_reqs") or []),
        "num_cached_reqs": len(
            (so.get("scheduled_cached_reqs") or {}).get("req_ids") or []
        ),
        "num_spec_decode_reqs": len(so.get("scheduled_spec_decode_tokens") or {}),
        "num_finished_req_ids": len(so.get("finished_req_ids") or []),
        "num_common_prefix_blocks": so.get("num_common_prefix_blocks") or [],
        "per_req_token_hist": {str(k): counts[k] for k in sorted(counts)},
        "block_table_groups": len(bt),
        "block_table_rows": len(row_widths),
        "block_table_width_sum": int(sum(row_widths)),
        "block_table_width_max": int(max(row_widths) if row_widths else 0),
        "req_states": len(req_states),
        "num_computed_tokens_hist": _hist(
            [int(r.get("num_computed_tokens", 0)) for r in req_states]
        ),
        "num_prompt_tokens_hist": _hist(
            [int(r.get("num_prompt_tokens", 0)) for r in req_states]
        ),
    }


def _hist(values: list[int], max_bins: int = 8) -> dict[str, int]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {str(lo): len(values)}
    step = max(1, (hi - lo) // max_bins)
    out: dict[str, int] = {}
    for v in values:
        b = lo + ((v - lo) // step) * step
        out[str(b)] = out.get(str(b), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def summarize_trace(recs: list[StepRecord], include_steps: bool = False) -> dict:
    """Aggregate shape summary of a whole trace (for the manifest/report)."""
    sigs = [shape_signature(r) for r in recs]
    phases: dict[str, int] = {}
    for s in sigs:
        phases[str(s["phase"])] = phases.get(str(s["phase"]), 0) + 1
    totals = [s["total_num_scheduled_tokens"] for s in sigs]
    reqs = [s["num_reqs"] for s in sigs]
    out = {
        "num_steps": len(sigs),
        "phases": phases,
        "total_tokens": int(sum(totals)),
        "tokens_mean": (sum(totals) / len(totals)) if totals else 0.0,
        "tokens_max": max(totals) if totals else 0,
        "num_reqs_mean": (sum(reqs) / len(reqs)) if reqs else 0.0,
        "num_reqs_max": max(reqs) if reqs else 0,
    }
    if include_steps:
        out["steps"] = sigs
    return out


__all__ = [
    "SCHEMA_VERSION",
    "StepRecord",
    "load_jsonl",
    "save_jsonl",
    "encode_scheduler_output",
    "encode_input_batch",
    "encode_model_output",
    "encode_req_state",
    "encode_sampling_params",
    "decode_sampling_params",
    "new_request_data_to_dict",
    "cached_request_data_to_dict",
    "infer_phase",
    "shape_signature",
    "summarize_trace",
]
