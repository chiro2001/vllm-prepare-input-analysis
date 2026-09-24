"""Replay side of the record/replay data path.

``load_jsonl()`` (re-exported from :mod:`pi_harness.workload.schema`) gives
back :class:`StepRecord` objects; :func:`to_scheduler_output` turns one record
back into a *real* ``vllm.v1.core.sched.output.SchedulerOutput`` whose field
types line up with the dataclass the worker consumes (``set``/``tuple``/``None``
restored), so it can be fed straight into ``NPUModelRunner._update_states``.

Importing this module does **not** import vllm/torch; the vllm imports happen
inside the decode functions (lazy) so the module is usable in a plain Python
process for schema checks.
"""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Iterator

from pi_harness.workload.schema import (  # noqa: F401  (re-export)
    StepRecord,
    decode_sampling_params,
    load_jsonl,
    save_jsonl,
)

__all__ = [
    "load_jsonl",
    "save_jsonl",
    "StepRecord",
    "decode_scheduler_output",
    "decode_model_output",
    "to_scheduler_output",
    "to_model_output",
    "iter_steps",
    "describe_decode_notes",
]


def _filter_kwargs(cls: Any, kwargs: dict) -> tuple[dict, list[str]]:
    """Drop keyword arguments the installed vllm version does not accept."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return kwargs, []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    return accepted, dropped


def _decode_mm_features(items: list[dict] | None) -> tuple[list, list[str]]:
    if not items:
        return [], []
    notes: list[str] = []
    out: list = []
    try:
        from vllm.multimodal.inputs import MultiModalFeatureSpec

        for item in items:
            kwargs, dropped = _filter_kwargs(MultiModalFeatureSpec, dict(item))
            kwargs.pop("__type__", None)
            try:
                out.append(MultiModalFeatureSpec(**kwargs))
                if dropped:
                    notes.append(f"mm_feature dropped={dropped}")
            except Exception as exc:  # pragma: no cover - multimodal path
                notes.append(f"mm_feature not reconstructed: {exc!r}")
                out.append(item)
    except Exception as exc:  # pragma: no cover
        notes.append(f"MultiModalFeatureSpec unavailable: {exc!r}")
        out = list(items)
    return out, notes


def decode_scheduler_output(d: dict) -> Any:
    """JSON dict -> real ``SchedulerOutput``.

    Raises ``RuntimeError`` with the accumulated notes if the structure is not
    reconstructible at all.
    """
    obj, notes = decode_scheduler_output_with_notes(d)
    if obj is None:
        raise RuntimeError("cannot decode SchedulerOutput: " + "; ".join(notes))
    return obj


def decode_scheduler_output_with_notes(d: dict) -> tuple[Any, list[str]]:
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        ScheduledEncoderInputStats,
        SchedulerOutput,
    )
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    notes: list[str] = []

    new_reqs = []
    for rd in d.get("scheduled_new_reqs") or []:
        mm_features, mm_notes = _decode_mm_features(rd.get("mm_features"))
        notes.extend(mm_notes)
        kwargs = {
            "req_id": rd["req_id"],
            "prompt_token_ids": rd.get("prompt_token_ids"),
            "mm_features": mm_features,
            "sampling_params": decode_sampling_params(rd.get("sampling_params")),
            "pooling_params": decode_sampling_params(rd.get("pooling_params")),
            "block_ids": tuple(
                list(g) for g in (rd.get("block_ids") or [])
            ),
            "num_computed_tokens": int(rd.get("num_computed_tokens", 0)),
            "lora_request": rd.get("lora_request"),
            "prompt_embeds": None,
            "prompt_is_token_ids": rd.get("prompt_is_token_ids"),
            "prefill_token_ids": rd.get("prefill_token_ids"),
        }
        if rd.get("prompt_embeds_shape") is not None:
            notes.append(
                f"req {rd['req_id']}: prompt_embeds present in capture "
                f"(shape={rd['prompt_embeds_shape']}) but not replayable"
            )
        kwargs, dropped = _filter_kwargs(NewRequestData, kwargs)
        if dropped:
            notes.append(f"NewRequestData dropped fields: {dropped}")
        new_reqs.append(NewRequestData(**kwargs))

    crd = d.get("scheduled_cached_reqs") or {}
    cached = CachedRequestData(
        req_ids=list(crd.get("req_ids") or []),
        resumed_req_ids=set(crd.get("resumed_req_ids") or []),
        new_token_ids=[list(x) for x in (crd.get("new_token_ids") or [])],
        all_token_ids={
            str(k): list(v) for k, v in (crd.get("all_token_ids") or {}).items()
        },
        new_block_ids=[
            None if x is None else tuple(list(g) for g in x)
            for x in (crd.get("new_block_ids") or [])
        ],
        num_computed_tokens=[int(x) for x in (crd.get("num_computed_tokens") or [])],
        num_output_tokens=[int(x) for x in (crd.get("num_output_tokens") or [])],
    )

    copies = d.get("kv_cache_block_copies")
    if copies is not None:
        copies = [
            KVCacheBlockCopy(
                src_block_id=int(c["src_block_id"]),
                dst_block_id=int(c["dst_block_id"]),
            )
            for c in copies
        ]

    stats = d.get("scheduled_encoder_input_stats")
    enc_stats = (
        None
        if stats is None
        else ScheduledEncoderInputStats(
            num_inputs=int(stats.get("num_inputs", 0)),
            output_tokens=int(stats.get("output_tokens", 0)),
        )
    )
    for key in ("kv_connector_metadata", "ec_connector_metadata"):
        if d.get(key):
            notes.append(f"{key} present in capture but dropped on replay")

    kwargs = {
        "scheduled_new_reqs": new_reqs,
        "scheduled_cached_reqs": cached,
        "num_scheduled_tokens": {
            str(k): int(v) for k, v in (d.get("num_scheduled_tokens") or {}).items()
        },
        "total_num_scheduled_tokens": int(d.get("total_num_scheduled_tokens", 0)),
        "scheduled_spec_decode_tokens": {
            str(k): [int(t) for t in v]
            for k, v in (d.get("scheduled_spec_decode_tokens") or {}).items()
        },
        "scheduled_encoder_inputs": {
            str(k): [int(t) for t in v]
            for k, v in (d.get("scheduled_encoder_inputs") or {}).items()
        },
        "num_common_prefix_blocks": [
            int(x) for x in (d.get("num_common_prefix_blocks") or [])
        ],
        "finished_req_ids": set(d.get("finished_req_ids") or []),
        "free_encoder_mm_hashes": list(d.get("free_encoder_mm_hashes") or []),
        "scheduled_encoder_input_stats": enc_stats,
        "preempted_req_ids": (
            None
            if d.get("preempted_req_ids") is None
            else set(d.get("preempted_req_ids") or [])
        ),
        "has_structured_output_requests": bool(
            d.get("has_structured_output_requests", False)
        ),
        "pending_structured_output_tokens": bool(
            d.get("pending_structured_output_tokens", False)
        ),
        "num_invalid_spec_tokens": d.get("num_invalid_spec_tokens"),
        "kv_connector_metadata": None,
        "ec_connector_metadata": None,
        "new_block_ids_to_zero": (
            None
            if d.get("new_block_ids_to_zero") is None
            else [int(x) for x in d["new_block_ids_to_zero"]]
        ),
        "kv_cache_block_copies": copies,
        "num_spec_tokens_to_schedule": int(d.get("num_spec_tokens_to_schedule", 0)),
    }
    kwargs, dropped = _filter_kwargs(SchedulerOutput, kwargs)
    if dropped:
        notes.append(f"SchedulerOutput dropped fields: {dropped}")
    return SchedulerOutput(**kwargs), notes


def decode_model_output(d: dict | None) -> Any:
    """JSON dict -> ``ModelRunnerOutput`` (or ``None``)."""
    if d is None:
        return None
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=list(d.get("req_ids") or []),
        req_id_to_index={
            str(k): int(v) for k, v in (d.get("req_id_to_index") or {}).items()
        },
        sampled_token_ids=[
            [int(t) for t in row] for row in (d.get("sampled_token_ids") or [])
        ],
    )


def to_scheduler_output(rec: StepRecord | dict) -> Any:
    d = rec.scheduler_output if isinstance(rec, StepRecord) else rec["scheduler_output"]
    return decode_scheduler_output(d)


def to_model_output(rec: StepRecord | dict) -> Any:
    d = rec.model_output if isinstance(rec, StepRecord) else rec.get("model_output")
    return decode_model_output(d)


def iter_steps(path: str | os.PathLike) -> Iterator[tuple[StepRecord, Any, Any]]:
    """Yield ``(record, SchedulerOutput, ModelRunnerOutput|None)`` per step."""
    for rec in load_jsonl(path):
        yield rec, to_scheduler_output(rec), to_model_output(rec)


def describe_decode_notes(rec: StepRecord | dict) -> list[str]:
    d = rec.scheduler_output if isinstance(rec, StepRecord) else rec["scheduler_output"]
    _, notes = decode_scheduler_output_with_notes(d)
    return notes


def _main(argv: list[str]) -> int:  # pragma: no cover - smoke CLI
    import argparse
    import collections

    ap = argparse.ArgumentParser(description="validate/decode a trace file")
    ap.add_argument("trace")
    ap.add_argument("--show", type=int, default=3, help="print first N steps")
    args = ap.parse_args(argv)

    recs = load_jsonl(args.trace)
    print(f"loaded {len(recs)} steps from {args.trace}")
    notes_all: list[str] = []
    for i, rec in enumerate(recs):
        so, notes = decode_scheduler_output_with_notes(rec.scheduler_output)
        notes_all.extend(f"step {i}: {n}" for n in notes)
        if i < args.show:
            print(
                f"  step={rec.step_idx} phase={rec.phase} "
                f"new={len(so.scheduled_new_reqs)} "
                f"cached={len(so.scheduled_cached_reqs.req_ids)} "
                f"tok={so.total_num_scheduled_tokens} "
                f"finished={sorted(so.finished_req_ids)} "
                f"spec={len(so.scheduled_spec_decode_tokens)} "
                f"common_prefix={so.num_common_prefix_blocks}"
            )
    counter = collections.Counter(notes_all)
    for note, n in counter.most_common(20):
        print(f"  note x{n}: {note}")
    print("decode OK" if not counter else "decode OK (with notes above)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))
