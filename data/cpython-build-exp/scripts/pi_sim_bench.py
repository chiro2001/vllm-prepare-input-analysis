#!/usr/bin/env python3
"""pi_sim_bench.py -- CPU-side "prepare_input-like" microbenchmark.

Purpose
-------
Measure how much of the *real* vLLM 0.26.0 `prepare_input` CPU cost is
explained by CPython's opcode dispatch mechanism.  The real load (see
docs/01, docs/05 of this repo) is:

  * `AscendGDNAttentionMetadataBuilder.build()` called 3x per engine step
    (~303 us each on Kunpeng 920B),
  * a 25-field `CommonAttentionMetadata` dataclass that is rebuilt /
    `.replace()`d several times per step,
  * per-request Python loops over small batches (1..64 reqs),
  * dict lookups (`req_id_to_index`), attribute chains
    (`self.vllm_config.scheduler_config.max_num_seqs`),
  * many 1-3 line methods and small tensor ops on shape (1,)..(64,),
  * object churn (per-request state objects created/dropped every step),
  * no heavy numerics at all.

The benchmark below reproduces exactly those characteristics with numpy
(no torch, no vLLM, no device) so that any of the three CPython builds can
run byte-identical code.  Every phase is deterministic and fixed-iteration:
the same script run under arm A/B/C performs the *same number of
bytecodes*, only the interpreter's dispatch strategy differs.

Usage
-----
    python pi_sim_bench.py --steps 300 --rounds 5 --reqs 64 \
        --json out.json --label armA

Output: JSON with per-round wall time of the measured region, per-phase
times, and a deterministic work digest (used to prove all arms executed
identical work).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, fields, replace

import numpy as np

# --------------------------------------------------------------------------
# 1. shape of the real load: a 25-field dataclass + short methods
# --------------------------------------------------------------------------


@dataclass
class Cfg:
    block_size: int
    max_num_seqs: int
    dtype: str


@dataclass
class SchedulerCfg:
    max_num_seqs: int
    max_model_len: int
    chunked_prefill_enabled: bool


@dataclass
class VllmConfig:
    scheduler_config: SchedulerCfg
    cache_config: Cfg
    model_config: Cfg


@dataclass
class AttnMeta:
    """Stands in for vllm.v1.attention.backends.utils.CommonAttentionMetadata
    (25 fields, plain dataclass, rebuilt + `.replace()`d every step)."""

    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: np.ndarray
    max_seq_len: int
    seq_lens: np.ndarray
    block_table_tensor: np.ndarray
    slot_mapping: np.ndarray
    causal: bool
    num_computed_tokens_cpu: np.ndarray
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefills: int
    num_decodes: int
    spec_sequence_masks: np.ndarray
    spec_sequence_indices: np.ndarray
    non_spec_sequence_indices: np.ndarray
    spec_actual_seq_lengths: np.ndarray
    non_spec_actual_seq_lengths: np.ndarray
    graph_batch_size: int
    reorder_batch_threshold: int
    state_indices_tensor: np.ndarray
    dcp_world_size: int
    cp_world_size: int
    use_spec_decode: bool

    # --- short methods (1-3 lines), exactly the shape that dominates the
    #     self-time of the real builder -------------------------------
    def compute_num_computed_tokens(self, i: int) -> int:
        return int(self.seq_lens[i]) - int(self.num_computed_tokens_cpu[i])

    def num_valid_tokens(self) -> int:
        return int(self.num_actual_tokens) - int(self.num_prefill_tokens)

    def is_uniform_batch(self) -> bool:
        return self.num_reqs == self.num_decodes

    def max_block_table_len(self) -> int:
        return int(self.block_table_tensor.shape[1])

    def with_num_computed(self, arr) -> "AttnMeta":
        return replace(self, num_computed_tokens_cpu=arr)


class ReqState:
    """Per-request state object; attribute-heavy, no __slots__ (like
    vllm.v1.core.sched.request_queue / CachedRequestState)."""

    def __init__(self, req_id: str, prompt_len: int, block_size: int):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.num_computed_tokens = 0
        self.block_ids = [0] * (prompt_len // block_size + 1)
        self.num_blocks = len(self.block_ids)
        self.status = "waiting"
        self.output_token_ids: list[int] = []
        self.spec_token_ids: list[int] = []
        self.kv_allocated = False
        self.max_tokens = 0

    def advance(self, n: int) -> None:
        self.num_computed_tokens += n
        self.output_token_ids.append(n)

    def last_block(self) -> int:
        return self.block_ids[-1]


class Builder:
    """Mimics AscendGDNAttentionMetadataBuilder: attribute chains into
    config objects, dict lookups, small arrays, 3 builds per step."""

    def __init__(self, vllm_config: VllmConfig, max_num_reqs: int):
        self.vllm_config = vllm_config
        self.max_num_reqs = max_num_reqs
        self.req_id_to_index: dict[str, int] = {}
        self.seq_lens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed = np.zeros(max_num_reqs, dtype=np.int32)
        self.block_table = np.zeros((max_num_reqs, 4), dtype=np.int32)
        self.slot_mapping = np.zeros(max_num_reqs, dtype=np.int32)
        self.mask = np.zeros(max_num_reqs, dtype=bool)
        # NOTE: must be initialised (not np.empty) -- the work digest below is
        # our proof that every arm executed *identical* work, and uninitialised
        # memory makes the digest vary run to run.
        self.buf = np.arange(max_num_reqs + 1, dtype=np.int32)
        self.calls = 0

    def record(self, state: ReqState, idx: int) -> None:
        self.req_id_to_index[state.req_id] = idx
        self.seq_lens[idx] = state.prompt_len
        self.num_computed[idx] = state.num_computed_tokens

    def index_of(self, req_id: str) -> int:
        return self.req_id_to_index[req_id]

    def maybe_index_of(self, req_id: str) -> int:
        return self.req_id_to_index.get(req_id, -1)

    def build(self, num_reqs: int, num_actual_tokens: int) -> AttnMeta:
        self.calls += 1
        block_size = self.vllm_config.cache_config.block_size
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        max_model_len = self.vllm_config.model_config.max_model_len = (
            self.vllm_config.scheduler_config.max_model_len
        )
        seq_lens = self.seq_lens[:num_reqs].copy()
        qsl = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(seq_lens, out=qsl[1:num_reqs + 1])
        slot = (np.arange(num_actual_tokens, dtype=np.int32) + block_size)
        mask = seq_lens > 0
        idxs = np.nonzero(mask)[0]
        return AttnMeta(
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=int(qsl[-1]) if num_reqs else 0,
            query_start_loc=qsl,
            max_seq_len=int(seq_lens.max()) if num_reqs else 0,
            seq_lens=seq_lens,
            block_table_tensor=self.block_table[:num_reqs],
            slot_mapping=slot,
            causal=True,
            num_computed_tokens_cpu=self.num_computed[:num_reqs].copy(),
            num_prefill_tokens=num_actual_tokens,
            num_decode_tokens=0,
            num_prefills=num_reqs,
            num_decodes=0,
            spec_sequence_masks=mask,
            spec_sequence_indices=idxs,
            non_spec_sequence_indices=idxs,
            spec_actual_seq_lengths=qsl,
            non_spec_actual_seq_lengths=qsl,
            graph_batch_size=max_num_seqs,
            reorder_batch_threshold=1,
            state_indices_tensor=self.block_table[:num_reqs],
            dcp_world_size=1,
            cp_world_size=1,
            use_spec_decode=False,
        )

    def patched_for_graph(self, meta: AttnMeta, graph_batch_size: int) -> AttnMeta:
        """the graph-capture path re-`replace()`s the metadata each step"""
        return replace(
            meta,
            graph_batch_size=graph_batch_size,
            num_decode_tokens=meta.num_reqs,
            use_spec_decode=True,
        )


# --------------------------------------------------------------------------
# 2. phases (each = one 'step' of the engine loop)
# --------------------------------------------------------------------------


def phase_churn(builder: Builder, reqs: list[ReqState], n: int) -> int:
    acc = 0
    new = [None] * n
    for i in range(n):
        st = ReqState(f"req-{i:04d}", 16 + (i % 7) * 8, 64)
        st.advance(1)
        new[i] = st
        acc += st.last_block() + st.num_blocks + len(st.output_token_ids)
    reqs[:n] = new
    return acc


def phase_attr(builder: Builder, reqs: list[ReqState], n: int) -> int:
    """attribute chains + dataclass field reads, no dict/array work"""
    acc = 0
    block_size = builder.vllm_config.cache_config.block_size
    sched = builder.vllm_config.scheduler_config
    for i in range(n):
        st = reqs[i]
        acc += st.num_computed_tokens + st.kv_allocated + st.max_tokens
        st.status = "running" if st.num_computed_tokens < sched.max_model_len else "done"
        acc += block_size + sched.max_num_seqs
    return acc


def phase_dict(builder: Builder, reqs: list[ReqState], n: int) -> int:
    acc = 0
    for i in range(n):
        rid = reqs[i].req_id
        builder.req_id_to_index[rid] = i
        acc += builder.index_of(rid)
        acc += builder.maybe_index_of(rid + "-missing")
    return acc


def phase_calls(builder: Builder, reqs: list[ReqState], n: int) -> int:
    acc = 0
    for i in range(n):
        builder.record(reqs[i], i)
    meta = builder.build(n, n)
    for _ in range(3):  # builder.build() runs 3x per engine step
        acc += meta.compute_num_computed_tokens(0)
        acc += meta.num_valid_tokens()
        acc += meta.is_uniform_batch()
        acc += meta.max_block_table_len()
        meta = meta.with_num_computed(builder.num_computed[:n].copy())
        meta = builder.patched_for_graph(meta, 64)
    return acc


def phase_numpy(builder: Builder, reqs: list[ReqState], n: int) -> int:
    """small-tensor arithmetic, shape (1,)..(64,), like cu_seqlens /
    slot_mapping / block tables in the real builder"""
    acc = 0
    seq_lens = builder.seq_lens[:n]
    qsl = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(seq_lens, out=qsl[1:n + 1])
    diff = np.diff(qsl).astype(np.int64)
    acc += int(diff.sum())
    acc += int(np.searchsorted(qsl, 32))
    slot = np.arange(n, dtype=np.int64) + 1
    acc += int((slot[: n // 2] + slot[n // 2:]).sum())
    mask = seq_lens > 8
    idx = np.nonzero(mask)[0]
    acc += int(idx.size)
    patched = builder.slot_mapping[:n].copy()
    patched[mask[:patched.size]] = -1
    acc += int(patched[0]) + len(patched.tolist())
    acc += int(builder.buf[:n + 1].astype(np.int32).max()) + 1
    return acc


PHASES = (
    ("churn", phase_churn),
    ("attr", phase_attr),
    ("dict", phase_dict),
    ("calls", phase_calls),
    ("numpy", phase_numpy),
)


def one_step(builder: Builder, reqs: list[ReqState], n: int) -> int:
    """interleaved 'engine step': the real loop mixes all of the above."""
    acc = 0
    for _, fn in PHASES:
        acc += fn(builder, reqs, n)
    return acc


# --------------------------------------------------------------------------
# 3. driver
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300, help="steps per round")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reqs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20, help="warmup steps")
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--mode", choices=("interleaved", "isolated"), default="interleaved",
                    help="interleaved = realistic mixed step; isolated = per-phase loops")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cfg = VllmConfig(
        scheduler_config=SchedulerCfg(max_num_seqs=args.reqs, max_model_len=4096,
                                      chunked_prefill_enabled=True),
        cache_config=Cfg(block_size=128, max_num_seqs=args.reqs, dtype="bfloat16"),
        model_config=Cfg(block_size=64, max_num_seqs=args.reqs, dtype="bfloat16"),
    )
    builder = Builder(cfg, max_num_reqs=args.reqs)
    reqs = [ReqState(f"req-{i:04d}", 16 + (i % 7) * 8, 128) for i in range(args.reqs)]

    out: dict = {
        "label": args.label,
        "mode": args.mode,
        "steps": args.steps,
        "rounds_requested": args.rounds,
        "reqs": args.reqs,
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "measurements": [],
        "phase_measurements": {},
    }

    # warmup (also populates the adaptive-specialization/IC state like the
    # real service does before the first measured step)
    if args.mode == "interleaved":
        for _ in range(args.warmup):
            one_step(builder, reqs, args.reqs)
    else:
        for _, fn in PHASES:
            for _ in range(max(1, args.warmup // 4)):
                fn(builder, reqs, args.reqs)

    digest = 0
    gc.collect()
    for rnd in range(args.rounds):
        if args.mode == "interleaved":
            gc.collect()
            t0 = time.perf_counter()
            acc = 0
            for _ in range(args.steps):
                acc += one_step(builder, reqs, args.reqs)
            dt = time.perf_counter() - t0
            digest += acc
            out["measurements"].append({
                "round": rnd,
                "seconds": dt,
                "steps": args.steps,
                "ns_per_step": dt / args.steps * 1e9,
            })
        else:
            for name, fn in PHASES:
                gc.collect()
                t0 = time.perf_counter()
                acc = 0
                for _ in range(args.steps):
                    acc += fn(builder, reqs, args.reqs)
                dt = time.perf_counter() - t0
                digest += acc
                rec = out["phase_measurements"].setdefault(name, [])
                rec.append({"round": rnd, "seconds": dt, "steps": args.steps,
                            "ns_per_step": dt / args.steps * 1e9})

    out["work_digest"] = digest
    if args.mode == "interleaved":
        secs = [m["seconds"] for m in out["measurements"]]
        out["median_seconds"] = statistics.median(secs)
        out["median_ns_per_step"] = statistics.median(
            m["ns_per_step"] for m in out["measurements"])
        out["min_ns_per_step"] = min(m["ns_per_step"] for m in out["measurements"])
        out["best_seconds"] = min(secs)
    else:
        for name, recs in out["phase_measurements"].items():
            out.setdefault("phase_median_ns_per_step", {})[name] = statistics.median(
                r["ns_per_step"] for r in recs)

    text = json.dumps(out, indent=1, sort_keys=True)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
