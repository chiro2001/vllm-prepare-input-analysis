"""Synthetic side: shape-level workload that reproduces the *real* scheduling
shapes without a card.

Preferred path (``engine="real"``, default): :mod:`pi_harness.workload.synth_real`
drives the **real** ``vllm.v1.core.sched.scheduler.Scheduler`` +
``KVCacheManager`` on CPU with a stand-in ``VllmConfig`` and a deterministic
fake "model" (sampled token ids).  Everything that shapes the load — admission
order, chunked-prefill splitting, prefix-cache hits, block allocation,
preemption, MTP draft scheduling — is then produced by the same code that runs
on the device.

Fallback (``engine="pure"``): :mod:`pi_harness.workload.synth_pure` emulates the
same admission/chunking rules in plain Python for environments where vllm
cannot be imported.  ``meta.engine`` records which path produced a trace.

All traces use the record schema (``pi_harness.workload.schema.StepRecord``), so
``synth`` and ``real`` go through the *same* downstream replay code path.

CLI::

    python -m pi_harness.workload.generate --batch 8 --isl 2048 --osl 128 \\
        --block-size 128 --chunk-size 2048 --arrival poisson --steps 200 \\
        --out data/harness/trace_synth_b8.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from typing import Any

from pi_harness.workload.schema import SCHEMA_VERSION, StepRecord, save_jsonl


@dataclass
class SynthConfig:
    """Shape knobs (INTERFACES.md §3 / task A parameter list)."""

    batch: int = 8  # concurrent requests
    isl: int = 1024  # input sequence length
    osl: int = 128  # output sequence length
    block_size: int = 128
    max_model_len: int = 8192
    spec_k: int = 0  # MTP draft tokens
    prefix_hit_ratio: float = 0.0
    chunk_size: int = 2048  # == max_num_batched_tokens (chunked prefill)
    arrival: str = "simultaneous"  # simultaneous | poisson | staircase
    steps: int = 0  # 0 = run until all requests finish
    seed: int = 0
    # --- extras -------------------------------------------------------------
    engine: str = "real"  # real | pure
    shuffle_values: bool = False  # emit the shuffled-values variant directly
    max_num_seqs: int = 0  # 0 -> batch
    num_blocks: int = 0  # 0 -> auto (fits the whole batch)
    vocab_size: int = 131072
    token_id_base: int = 7
    prefix_len: int = 0  # 0 -> isl//2, block aligned
    arrival_rate: float = 0.0  # poisson rate (reqs/step), 0 = auto
    arrival_group: int = 0  # staircase group size, 0 = auto
    arrival_period: int = 0  # staircase period in steps, 0 = auto
    accept_ratio: float = 0.6  # MTP: fraction of drafts accepted (shape only)
    request_id_prefix: str = "req"

    @classmethod
    def from_mapping(cls, cfg: Any) -> "SynthConfig":
        """Accept a dataclass / namespace / dict (e.g. the runner's SweepConfig)."""
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, dict):
            data = dict(cfg)
        elif hasattr(cfg, "__dict__"):
            data = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
        else:  # pragma: no cover - defensive
            raise TypeError(f"cannot build SynthConfig from {type(cfg)!r}")
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        out = cls(**filtered)
        if extra:
            out._extra_params = extra  # type: ignore[attr-defined]
        return out

    def resolved(self) -> "SynthConfig":
        """Fill in the ``0 -> auto`` knobs and normalise."""
        cfg = copy.deepcopy(self)
        cfg.max_num_seqs = cfg.max_num_seqs or max(cfg.batch, 1)
        if cfg.prefix_len <= 0:
            cfg.prefix_len = max(
                cfg.block_size, (cfg.isl // 2) // cfg.block_size * cfg.block_size
            )
        if cfg.num_blocks <= 0:
            cfg.num_blocks = (
                int(
                    math.ceil(
                        cfg.max_model_len / cfg.block_size * cfg.max_num_seqs * 1.3
                    )
                )
                + 16
            )
        if cfg.chunk_size <= 0:
            cfg.chunk_size = cfg.isl
        cfg.arrival = str(cfg.arrival).lower()
        cfg.engine = str(cfg.engine).lower()
        if cfg.arrival not in {"simultaneous", "poisson", "staircase"}:
            raise ValueError(f"unknown arrival mode {cfg.arrival!r}")
        return cfg

    def steps_estimate(self) -> int:
        """Upper bound of the number of engine steps of this workload."""
        chunks = math.ceil(self.isl / max(self.chunk_size, 1))
        return int(self.osl + chunks * max(self.batch, 1) + 8 * self.batch + 16)

    def to_dict(self) -> dict:
        d = asdict(self)
        extra = getattr(self, "_extra_params", None)
        if extra:
            d["_extra_params"] = extra
        return d


def synth_trace(cfg: Any) -> list[StepRecord]:
    """Build a synthetic trace carrying the same schema as a captured one."""
    from pi_harness.workload import synth_pure, synth_real

    c = SynthConfig.from_mapping(cfg).resolved()
    engine = c.engine
    if engine == "real":
        try:
            recs = synth_real.run(c)
        except Exception as exc:
            print(
                f"[generate] real-scheduler synthesis failed ({exc!r}); "
                "falling back to pure synthesis",
                flush=True,
            )
            recs = synth_pure.run(c)
            engine = "pure(fallback)"
    else:
        recs = synth_pure.run(c)

    for rec in recs:
        rec.meta = {
            **(rec.meta or {}),
            "source": "synth",
            "synth": True,
            "engine": engine,
            "params": c.to_dict(),
            "schema_version": SCHEMA_VERSION,
        }
    if c.shuffle_values:
        recs = shuffle_trace(recs, seed=c.seed, source="synth-shuffled")
        for rec in recs:
            rec.meta["synth"] = True
            rec.meta["engine"] = engine
    return recs


def arrival_plan(c: SynthConfig, rng: random.Random) -> list[int]:
    """Arrival step for each request, per the ``arrival`` mode."""
    n = c.batch
    horizon = c.steps if c.steps > 0 else c.steps_estimate()
    if c.arrival == "simultaneous":
        return [0] * n
    if c.arrival == "poisson":
        rate = c.arrival_rate or n / max(1.0, 0.25 * horizon)
        plan: list[int] = []
        t = 0.0
        for _ in range(n):
            t += rng.expovariate(max(rate, 1e-6))
            plan.append(int(t))
        return plan
    # staircase: fixed-size groups every ``period`` steps
    group = c.arrival_group or max(1, n // 4)
    period = c.arrival_period or max(1, horizon // 8)
    return [min(n, (i // group) * period) for i in range(n)]


def build_prompts(c: SynthConfig, plan: list[int]) -> tuple[list[list[int]], list[bool]]:
    """Deterministic token ids: a shared block-aligned prefix for the fraction
    ``prefix_hit_ratio`` of requests (a genuine prefix-cache hit), unique tails
    otherwise.  ``token_id_base``/``vocab_size`` make the values controllable."""
    import numpy as np

    rng = random.Random(c.seed + 1)
    vocab = max(int(c.vocab_size), 64)
    prefix_len = min(c.prefix_len, max(c.isl - 1, 1))
    shared = (
        (np.arange(prefix_len, dtype=np.int64) * 131 + c.token_id_base) % (vocab - 1)
        + 1
    ).tolist()

    prompts: list[list[int]] = []
    hits: list[bool] = []
    for i in range(c.batch):
        hit = c.prefix_hit_ratio > 0.0 and rng.random() < c.prefix_hit_ratio
        tail_len = c.isl - prefix_len if hit else c.isl
        tail = (
            (
                np.arange(tail_len, dtype=np.int64) * 977
                + c.token_id_base
                + (i + 1) * 4099
            )
            % (vocab - 1)
            + 1
        ).tolist()
        prompts.append((shared[:prefix_len] if hit else []) + tail)
        hits.append(hit)
    return prompts, hits


# ---------------------------------------------------------------------------
# value shuffling (A/B intervention: same shape, different content)
# ---------------------------------------------------------------------------


def _permutation(size: int, seed: int) -> list[int]:
    perm = list(range(size))
    random.Random(seed).shuffle(perm)
    return perm


class _ValueShuffler:
    """Consistent bijections over request ids / token ids / block ids.

    Applied to *every* occurrence of a value in a trace so keys still join
    (``num_scheduled_tokens`` keys vs ``req_states`` ids, block ids in the
    scheduler output vs the block table, …).  Shapes are untouched.
    """

    def __init__(self, recs: list[StepRecord], seed: int) -> None:
        req_ids: set[str] = set()
        token_ids: set[int] = set()
        block_ids: set[int] = set()
        for rec in recs:
            so = rec.scheduler_output
            req_ids.update(so.get("num_scheduled_tokens") or {})
            req_ids.update((so.get("scheduled_cached_reqs") or {}).get("req_ids") or [])
            req_ids.update(r["req_id"] for r in so.get("scheduled_new_reqs") or [])
            req_ids.update(so.get("finished_req_ids") or [])
            req_ids.update(r.get("req_id") for r in rec.req_states)
            req_ids.update((rec.input_batch or {}).get("req_ids") or [])
            for rd in so.get("scheduled_new_reqs") or []:
                token_ids.update(rd.get("prompt_token_ids") or [])
            for st in rec.req_states:
                token_ids.update(st.get("prompt_token_ids") or [])
            for ids in _iter_block_ids(so):
                block_ids.update(ids)
            for ids in _iter_block_ids_from_states(rec):
                block_ids.update(ids)
        self.req_map = {
            rid: f"s{idx:05d}"
            for idx, rid in enumerate(sorted(r for r in req_ids if r is not None))
        }
        tok_perm = _permutation(max(token_ids) + 1 if token_ids else 1, seed + 17)
        self.tok_map = tok_perm
        self.blk_map = _permutation(max(block_ids) + 1 if block_ids else 1, seed + 29)

    def req(self, rid: str | None) -> str | None:
        return None if rid is None else self.req_map.get(rid, rid)

    def tok(self, t: int) -> int:
        if t is None or t < 0 or t >= len(self.tok_map):
            return t
        return self.tok_map[t]

    def blk(self, b: int) -> int:
        if b is None or b < 0 or b >= len(self.blk_map):
            return b
        return self.blk_map[b]


def _iter_block_ids(so: dict):
    for rd in so.get("scheduled_new_reqs") or []:
        for group in rd.get("block_ids") or []:
            yield group
    for group in (so.get("scheduled_cached_reqs") or {}).get("new_block_ids") or []:
        for blocks in group or []:
            yield blocks
    for b in so.get("new_block_ids_to_zero") or []:
        yield [b]
    for c in so.get("kv_cache_block_copies") or []:
        yield [c["src_block_id"], c["dst_block_id"]]


def _iter_block_ids_from_states(rec: StepRecord):
    for st in rec.req_states or []:
        for group in st.get("block_ids") or []:
            yield group
    for group in (rec.input_batch or {}).get("block_table") or []:
        for row in group:
            yield row


def _shuffle_record(rec: StepRecord, sh: _ValueShuffler) -> StepRecord:
    d = copy.deepcopy(rec.to_dict())
    so = d["scheduler_output"]
    so["num_scheduled_tokens"] = {
        sh.req(k): v for k, v in (so.get("num_scheduled_tokens") or {}).items()
    }
    so["scheduled_spec_decode_tokens"] = {
        sh.req(k): [sh.tok(t) for t in v]
        for k, v in (so.get("scheduled_spec_decode_tokens") or {}).items()
    }
    so["scheduled_encoder_inputs"] = {
        sh.req(k): list(v) for k, v in (so.get("scheduled_encoder_inputs") or {}).items()
    }
    so["finished_req_ids"] = [sh.req(r) for r in so.get("finished_req_ids") or []]
    if so.get("preempted_req_ids") is not None:
        so["preempted_req_ids"] = [sh.req(r) for r in so["preempted_req_ids"]]
    if so.get("num_invalid_spec_tokens"):
        so["num_invalid_spec_tokens"] = {
            sh.req(k): v for k, v in so["num_invalid_spec_tokens"].items()
        }
    for rd in so.get("scheduled_new_reqs") or []:
        rd["req_id"] = sh.req(rd["req_id"])
        if rd.get("prompt_token_ids") is not None:
            rd["prompt_token_ids"] = [sh.tok(t) for t in rd["prompt_token_ids"]]
        if rd.get("prefill_token_ids") is not None:
            rd["prefill_token_ids"] = [sh.tok(t) for t in rd["prefill_token_ids"]]
        rd["block_ids"] = [
            [sh.blk(b) for b in group] for group in (rd.get("block_ids") or [])
        ]
    crd = so.get("scheduled_cached_reqs") or {}
    crd["req_ids"] = [sh.req(r) for r in crd.get("req_ids") or []]
    crd["resumed_req_ids"] = [sh.req(r) for r in crd.get("resumed_req_ids") or []]
    crd["new_token_ids"] = [
        [sh.tok(t) for t in row] for row in crd.get("new_token_ids") or []
    ]
    crd["all_token_ids"] = {
        sh.req(k): [sh.tok(t) for t in v]
        for k, v in (crd.get("all_token_ids") or {}).items()
    }
    crd["new_block_ids"] = [
        None
        if group is None
        else [[sh.blk(b) for b in blocks] for blocks in group]
        for group in crd.get("new_block_ids") or []
    ]
    if so.get("new_block_ids_to_zero") is not None:
        so["new_block_ids_to_zero"] = [
            sh.blk(b) for b in so["new_block_ids_to_zero"]
        ]
    if so.get("kv_cache_block_copies") is not None:
        so["kv_cache_block_copies"] = [
            {
                "src_block_id": sh.blk(c["src_block_id"]),
                "dst_block_id": sh.blk(c["dst_block_id"]),
            }
            for c in so["kv_cache_block_copies"]
        ]

    for st in d["req_states"]:
        st["req_id"] = sh.req(st["req_id"])
        if st.get("prompt_token_ids") is not None:
            st["prompt_token_ids"] = [sh.tok(t) for t in st["prompt_token_ids"]]
        st["block_ids"] = [
            [sh.blk(b) for b in group] for group in (st.get("block_ids") or [])
        ]

    ib = d.get("input_batch")
    if ib:
        ib["req_ids"] = [sh.req(r) for r in ib.get("req_ids") or []]
        ib["block_table"] = [
            [[sh.blk(b) for b in row] for row in group]
            for group in ib.get("block_table") or []
        ]
        ib["spec_token_ids"] = [
            [sh.tok(t) for t in row] for row in ib.get("spec_token_ids") or []
        ]

    mo = d.get("model_output")
    if mo:
        mo["req_ids"] = [sh.req(r) for r in mo.get("req_ids") or []]
        mo["req_id_to_index"] = {
            sh.req(k): v for k, v in (mo.get("req_id_to_index") or {}).items()
        }
        mo["sampled_token_ids"] = [
            [sh.tok(t) for t in row] for row in mo.get("sampled_token_ids") or []
        ]

    out = StepRecord.from_dict(d)
    out.meta = {
        **(out.meta or {}),
        "source": "shuffled",
        "synth": bool((rec.meta or {}).get("synth", False)),
    }
    return out


def shuffle_trace(
    recs: list[StepRecord],
    *,
    seed: int = 0,
    source: str = "shuffled",
) -> list[StepRecord]:
    """Same shapes, permuted values (request ids, token ids, block ids)."""
    sh = _ValueShuffler(recs, seed)
    out = [_shuffle_record(rec, sh) for rec in recs]
    for rec, src in zip(out, recs):
        rec.meta = {
            **(rec.meta or {}),
            "source": source,
            "shuffled_from": (src.meta or {}).get("source", "unknown"),
            "shuffle_seed": seed,
            "synth": bool((src.meta or {}).get("synth", False)),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="synthesise a shape-level trace")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--isl", type=int, default=1024)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--spec-k", type=int, default=0)
    p.add_argument("--prefix-hit-ratio", type=float, default=0.0)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--arrival", default="simultaneous")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", default="real", choices=["real", "pure"])
    p.add_argument("--shuffle-values", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=0)
    p.add_argument("--num-blocks", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=131072)
    p.add_argument("--token-id-base", type=int, default=7)
    p.add_argument("--prefix-len", type=int, default=0)
    p.add_argument("--arrival-rate", type=float, default=0.0)
    p.add_argument("--arrival-group", type=int, default=0)
    p.add_argument("--arrival-period", type=int, default=0)
    p.add_argument("--accept-ratio", type=float, default=0.6)
    p.add_argument("--request-id-prefix", default="req")
    p.add_argument("--out", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--limit-steps", type=int, default=0, help="truncate output")
    return p


def main(argv: list[str] | None = None) -> int:
    import hashlib
    import time

    args = _build_argparser().parse_args(argv)
    cfg = SynthConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k in {f.name for f in fields(SynthConfig)}
        }
    )
    recs = synth_trace(cfg)
    if args.limit_steps:
        recs = recs[: args.limit_steps]
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        n = save_jsonl(recs, args.out)
        print(f"wrote {n} steps -> {args.out}")
    if args.manifest:
        from pi_harness.workload.schema import summarize_trace

        here = os.path.dirname(os.path.abspath(__file__))
        sha = {}
        for fname in (
            "generate.py",
            "schema.py",
            "synth_real.py",
            "synth_pure.py",
            "worker_state.py",
        ):
            try:
                with open(os.path.join(here, fname), "rb") as fh:
                    sha[fname] = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                sha[fname] = "missing"
        manifest = {
            "kind": "trace",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "params": cfg.resolved().to_dict(),
            "trace": os.path.abspath(args.out) if args.out else None,
            "num_steps": len(recs),
            "summary": summarize_trace(recs),
            "sha256": sha,
            "image": os.environ.get("PI_IMAGE", ""),
            "host": os.environ.get("HOSTNAME", ""),
        }
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"wrote manifest -> {args.manifest}")
    if not args.out:
        from pi_harness.workload.schema import summarize_trace

        print(json.dumps(summarize_trace(recs), indent=2)[:4000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
