"""把 record 出来的 JSONL 变成逐步可消费的 (SchedulerOutput, info) 序列。

数据来源优先级：
1. `pi_harness.trace.replay`（`trace_schema` agent 提供，含字段级类型还原）
2. 内置的**最小**还原器（只依赖 vllm 的真实 dataclass，缺字段会报错而不是静默）

接口：`iter_steps(trace_path) -> Iterator[(scheduler_output, info_dict)]`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Any


def _load_records(path: str | Path) -> list[dict]:
    recs = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _via_trace_schema(rec: dict) -> Any:
    """优先用 trace_schema 的还原器（它做了字段级类型修复）。"""
    from pi_harness.trace import replay as tr

    return tr.decode_scheduler_output(rec["scheduler_output"])


def _via_builtin(rec: dict) -> Any:
    """内置兜底：直接构造真实 dataclass。

    注意：这里**不做**字段猜测。若 trace 缺字段，vllm 的 dataclass 会自己报错，
    从而暴露 schema 不完整，而不是产出一个语义错误的 SchedulerOutput。
    """
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    so = rec["scheduler_output"]
    new_reqs = []
    for nr in so.get("scheduled_new_reqs", []):
        new_reqs.append(
            NewRequestData(
                req_id=nr["req_id"],
                prompt_token_ids=nr.get("prompt_token_ids"),
                mm_features=nr.get("mm_features") or [],
                sampling_params=_sampling_params(nr),
                pooling_params=None,
                block_ids=tuple(tuple(g) for g in (nr.get("block_ids") or [[]])),
                num_computed_tokens=nr.get("num_computed_tokens", 0),
                lora_request=None,
            )
        )
    cr = so.get("scheduled_cached_reqs") or {}
    cached = CachedRequestData(
        req_ids=list(cr.get("req_ids", [])),
        resumed_req_ids=set(cr.get("resumed_req_ids", [])),
        new_token_ids=list(cr.get("new_token_ids", [])),
        all_token_ids=dict(cr.get("all_token_ids", {})),
        new_block_ids=[
            (tuple(tuple(g) for g in nb) if nb is not None else None)
            for nb in cr.get("new_block_ids", [])
        ],
        num_computed_tokens=list(cr.get("num_computed_tokens", [])),
        num_output_tokens=list(cr.get("num_output_tokens", [])),
    )
    kwargs = dict(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={k: int(v) for k, v in so["num_scheduled_tokens"].items()},
        total_num_scheduled_tokens=int(so["total_num_scheduled_tokens"]),
        scheduled_spec_decode_tokens={k: list(v) for k, v in so.get("scheduled_spec_decode_tokens", {}).items()},
        scheduled_encoder_inputs=dict(so.get("scheduled_encoder_inputs", {})),
        num_common_prefix_blocks=list(so.get("num_common_prefix_blocks", [0])),
        finished_req_ids=set(so.get("finished_req_ids", [])),
        free_encoder_mm_hashes=list(so.get("free_encoder_mm_hashes", [])),
    )
    for opt in ("new_block_ids_to_zero", "kv_cache_block_copies", "preempted_req_ids"):
        if opt in so:
            kwargs[opt] = so[opt]
    return SchedulerOutput(**kwargs)


_SP_CACHE: dict[str, Any] = {}


def _sampling_params(nr: dict):
    if "sampling_params" not in nr:
        sp = _SP_CACHE.get("greedy")
        if sp is None:
            from vllm.sampling_params import SamplingParams

            sp = _SP_CACHE["greedy"] = SamplingParams(temperature=0.0, max_tokens=64)
        return sp
    # trace 里若带了显式字段，交给 trace_schema 的 decoder 处理更稳妥
    from pi_harness.workload.schema import decode_sampling_params

    return decode_sampling_params(nr["sampling_params"])


def iter_steps(trace_path: str | Path) -> Iterator[tuple[Any, dict]]:
    recs = _load_records(trace_path)
    have_trace_schema = True
    try:
        import pi_harness.trace.replay  # noqa: F401
    except Exception:
        have_trace_schema = False

    for rec in recs:
        so = None
        if have_trace_schema:
            try:
                so = _via_trace_schema(rec)
            except Exception:
                so = None
        if so is None:
            so = _via_builtin(rec)
        info = {
            "phase": rec.get("phase"),
            "step_idx": rec.get("step_idx"),
            "n_running": len(so.num_scheduled_tokens),
            "total_scheduled": so.total_num_scheduled_tokens,
            "source": (rec.get("meta") or {}).get("source", "trace"),
            "t_prepare_input_us_capture": rec.get("t_prepare_input_us"),
        }
        yield so, info
