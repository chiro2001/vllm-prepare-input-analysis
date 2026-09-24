#!/usr/bin/env python3
"""prepare_input CPU 微基准（无卡，纯 CPU）。

目的：为 `docs/07-bottleneck-analysis.md` 的成本模型标定"每元素/每次调用"的
一阶系数，并把 `_prepare_inputs` 的 CPU 算子序列复刻成一个可控的合成负载。

设计原则
--------
1. 只测 **CPU 侧真实出现过的原语**（numpy / python / ATen CPU / .tolist()），
   不做与 vLLM 无关的通用 benchmark。
2. 每个测点报 `median` 与 `p10/p90`，并扣除空循环开销。
3. 规模轴与 `plan/EXECUTION.md` §2 的 (B, T, K, G) 对齐，方便和真机数据对齐。
4. H2D（`copy_to_gpu` / `pin_memory().to()`）**不在本脚本范围内**：CPU-only torch
   没有 NPU allocator，测出的数字不可信；这些项由 replay harness 在 chip3 上标定。
   本脚本只探测 `Tensor.pin_memory()` 在无加速器时的行为。

用法
----
    taskset -c 200-203 python microbench_prepare_input.py --out-dir data/model

输出（写入 --out-dir）：
    microbench-numpy.csv      numpy 原语：per-call / per-element
    microbench-python.csv     Python 解释器原语（dict/list/attr/tolist）
    microbench-torch.csv      ATen CPU 原语（index_select / add / copy_ / 分配）
    microbench-composite.csv  复刻版 `_prepare_inputs` CPU 序列（按 B/T/spec）
    microbench-meta.json      运行环境元数据（CPU 型号、核、affinity、库版本）
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import typing as t

import numpy as np

try:  # torch 可选：缺了也能跑 numpy/python 两组
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# 计时内核
# --------------------------------------------------------------------------


def _empty_loop_ns(repeats: int) -> float:
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        pass
    return (time.perf_counter_ns() - t0) / repeats


def bench(fn: t.Callable[[], object], repeats: int = 200, rounds: int = 9) -> dict:
    """返回 fn() 单次调用的耗时统计（ns），已扣除空循环开销。"""
    fn()  # warmup
    empty = _empty_loop_ns(repeats)
    samples: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(repeats):
            fn()
        samples.append((time.perf_counter_ns() - t0) / repeats - empty)
    samples.sort()
    med = statistics.median(samples)
    p10 = samples[max(0, int(0.1 * (len(samples) - 1)))]
    p90 = samples[min(len(samples) - 1, int(0.9 * (len(samples) - 1)))]
    return {
        "median_ns": med,
        "p10_ns": p10,
        "p90_ns": p90,
        "min_ns": samples[0],
        "median_us": med / 1e3,
        "p10_us": p10 / 1e3,
        "p90_us": p90 / 1e3,
        "repeats": repeats,
        "rounds": rounds,
    }


Row = dict[str, t.Any]


# --------------------------------------------------------------------------
# 1. numpy 原语
# --------------------------------------------------------------------------

DECODE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
TOKEN_T = [1, 64, 512, 2048, 8192, 32768]
BLOCK_K = [16, 64, 128, 256]  # max_model_len/128: 2048/128=16 ... 32768/128=256


def bench_numpy() -> list[Row]:
    rows: list[Row] = []

    def add_row(name: str, group: str, size: int, unit: str, fn, repeats: int, rounds: int = 9):
        st = bench(fn, repeats=repeats, rounds=rounds)
        r = {"name": name, "group": group, "size": size, "unit": unit, **st}
        r["ns_per_element"] = st["median_ns"] / max(size, 1)
        rows.append(r)

    for B in DECODE_B:
        arange = np.arange(B, dtype=np.int32)
        counts = np.ones(B, dtype=np.int32)
        rows_buf = np.empty(B, dtype=np.int32)
        py_list = list(range(B))
        mask = np.zeros(max(B, 1), dtype=bool)

        add_row("np.repeat(arange_B, ones_B)", "decode", B, "req",
                lambda a=arange, c=counts: np.repeat(a, c), 300)
        add_row("np.cumsum(counts_B)", "decode", B, "req",
                lambda c=counts: np.cumsum(c), 300)
        add_row("np.array(list_B, int32)", "decode", B, "req",
                lambda l=py_list: np.array(l, dtype=np.int32), 300)
        add_row("np.all(counts_B == 1)", "decode", B, "req",
                lambda c=counts: bool(np.all(c == 1)), 300)
        add_row("np.nonzero(mask_B)", "decode", B, "req",
                lambda m=mask: np.nonzero(m)[0], 300)
        add_row("np.add(buf, buf, out=buf)", "decode", B, "req",
                lambda b=rows_buf: np.add(b, b, out=b), 300)

        def _cumsum_and_arange(a=arange, c=counts, T=B):
            cu = np.cumsum(c)
            offs = np.repeat(cu - c, c)
            out = np.empty(T, dtype=np.int32)
            np.subtract(a[:T], offs, out=out)
            return out

        add_row("_get_cumsum_and_arange(B=T)", "decode", B, "req", _cumsum_and_arange, 300)

    for T in TOKEN_T:
        arange = np.arange(T, dtype=np.int32)
        src = np.arange(T, dtype=np.int32)
        buf = np.empty(T, dtype=np.int32)
        idx = np.arange(T, dtype=np.int64)
        flat = np.arange(T, dtype=np.int32)

        add_row("np.subtract(arange_T, offs_T, out)", "token", T, "token",
                lambda a=arange, b=buf: np.subtract(a, 0, out=b), 200, 7)
        add_row("np.take(flat_T, idx_T) -> alloc", "token", T, "token",
                lambda f=flat, i=idx: np.take(f, i), 200, 7)
        add_row("np.repeat(arange_T, ones_T) alloc", "token", T, "token",
                lambda a=arange: np.repeat(a, 1), 200, 7)
        add_row("np.gather(positions: buf[req_indices])", "token", T, "token",
                lambda f=flat, i=idx: f[i], 200, 7)

    return rows


# --------------------------------------------------------------------------
# 2. Python 解释器原语（对应逐请求循环 / list comp / dict / tolist）
# --------------------------------------------------------------------------


class _ReqState:
    """模拟 CachedRequestState 的属性访问面（num_tokens 是 property）。"""

    __slots__ = ("num_prompt_tokens", "output_token_ids")

    def __init__(self, prompt_len: int, out_len: int):
        self.num_prompt_tokens = prompt_len
        self.output_token_ids = list(range(out_len))

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)


def bench_python() -> list[Row]:
    rows: list[Row] = []

    def add_row(name: str, group: str, size: int, unit: str, fn, repeats: int, rounds: int = 9):
        st = bench(fn, repeats=repeats, rounds=rounds)
        r = {"name": name, "group": group, "size": size, "unit": unit, **st}
        r["ns_per_element"] = st["median_ns"] / max(size, 1)
        rows.append(r)

    for B in DECODE_B + [512]:
        req_ids = [f"req-{i}" for i in range(B)]
        prev_map = {rid: i for i, rid in enumerate(req_ids)}
        req_state = {rid: _ReqState(128, 64) for rid in req_ids}
        arr = np.ones(B, dtype=np.int32)
        reps = 300 if B <= 256 else 100

        add_row("_compute_prev_positions: for+dict.get", "churn", B, "req",
                lambda r=req_ids, m=prev_map: [m.get(x, -1) for x in r], reps)

        def attr_loop():
            out = []
            for rid in req_ids:
                out.append(req_state[rid].num_tokens)
            return out

        add_row("[requests[r].num_tokens for r in req_ids]", "churn", B, "req", attr_loop, reps)
        add_row("np.ndarray.tolist() int32[B]", "tolist", B, "req",
                lambda a=arr: a.tolist(), reps)
        add_row("arr[1:].tolist() (query_start_loc[1:])", "tolist", B, "req",
                lambda a=arr: a[1:].tolist(), reps)
        add_row("dict build {rid: i}", "churn", B, "req",
                lambda r=req_ids: {rid: i for i, rid in enumerate(r)}, reps)

        def set_ops(r=req_ids):
            finished = set(r[::16])
            return bool(finished)

        add_row("set(finished_req_ids)", "churn", B, "req", set_ops, reps)

    if torch is not None:
        for B in DECODE_B + [512]:
            tt = torch.ones(B, dtype=torch.int32)
            rows_buf = torch.empty(B, dtype=torch.int32)
            reps = 300 if B <= 256 else 100
            add_row("torch.Tensor.tolist() int32[B]", "tolist", B, "req",
                    lambda x=tt: x.tolist(), reps)
            add_row("torch slice+sub (query_lens)", "tolist", B, "req",
                    lambda x=tt: x[1:] - x[:-1], reps)
            add_row("torch fill_(0) tail", "tolist", B, "req",
                    lambda x=rows_buf: x.fill_(0), reps)

    return rows


# --------------------------------------------------------------------------
# 3. ATen CPU 原语
# --------------------------------------------------------------------------


def bench_torch() -> list[Row]:
    rows: list[Row] = []
    if torch is None:
        return rows
    torch.set_num_threads(1)

    def add_row(name: str, group: str, size: int, unit: str, fn, repeats: int, rounds: int = 9,
                note: str | None = None):
        st = bench(fn, repeats=repeats, rounds=rounds)
        r = {"name": name, "group": group, "size": size, "unit": unit, **st}
        r["ns_per_element"] = st["median_ns"] / max(size, 1)
        if note:
            r["note"] = note
        rows.append(r)

    one = torch.zeros(1, dtype=torch.int32)
    o2 = torch.zeros(1, dtype=torch.int32)
    add_row("ATen per-call: add(1-elem, out=)", "fixed", 1, "call",
            lambda a=one, b=o2: torch.add(a, b, out=a), 2000)
    add_row("ATen per-call: empty(4,) alloc", "fixed", 1, "call",
            lambda: torch.empty(8, dtype=torch.int32), 2000)
    add_row("ATen per-call: zeros(3,8) alloc", "fixed", 1, "call",
            lambda: torch.zeros(3, 8, dtype=torch.int32), 2000)

    for B in DECODE_B:
        flat = torch.arange(B * 8, dtype=torch.int32)
        idx = torch.arange(B, dtype=torch.int64)
        out = torch.empty(B, dtype=torch.int32)
        add_row("torch.index_select(flat, 0, idx, out)", "decode", B, "req",
                lambda f=flat, i=idx, o=out: torch.index_select(f, 0, i, out=o), 300)
        add_row("torch.add(a, b, out=)", "decode", B, "req",
                lambda o=out: torch.add(o, o, out=o), 300)
        add_row("torch tensor from numpy view", "decode", B, "req",
                lambda a=np.arange(B, dtype=np.int32): torch.from_numpy(a), 300)

    for T in TOKEN_T:
        flat = torch.arange(T * 2, dtype=torch.int32)
        idx = torch.arange(T, dtype=torch.int64) % (2 * T)
        out = torch.empty(T, dtype=torch.int32)
        add_row("torch.index_select(flat, 0, idx, out)", "token", T, "token",
                lambda f=flat, i=idx, o=out: torch.index_select(f, 0, i, out=o), 200, 7)
        add_row("torch.index_select alloc (no out=)", "token", T, "token",
                lambda f=flat, i=idx: torch.index_select(f, 0, i), 200, 7)

    try:
        src = torch.arange(8, dtype=torch.int32)
        p = src.pin_memory()
        st = bench(lambda: src.pin_memory(), repeats=50, rounds=5)
        rows.append({
            "name": "Tensor.pin_memory() (probe)", "group": "h2d_probe", "size": 8, "unit": "call",
            **st, "ns_per_element": st["median_ns"] / 8,
            "note": f"pin_memory ok; is_pinned={p.is_pinned()}; 无 NPU 时不可作为 H2D 证据",
        })
    except Exception as exc:  # pragma: no cover
        rows.append({
            "name": "Tensor.pin_memory() (probe)", "group": "h2d_probe", "size": 8, "unit": "call",
            "median_ns": float("nan"), "p10_ns": float("nan"), "p90_ns": float("nan"),
            "min_ns": float("nan"), "repeats": 0, "rounds": 0, "ns_per_element": float("nan"),
            "median_us": float("nan"), "p10_us": float("nan"), "p90_us": float("nan"),
            "note": f"pin_memory 不可用: {type(exc).__name__}: {exc}",
        })

    return rows


# --------------------------------------------------------------------------
# 4. 复刻版 `_prepare_inputs`（CPU-only 子集的合成负载）
# --------------------------------------------------------------------------


class PrepareInputReplica:
    """把 `_prepare_inputs` 的 CPU 侧算子序列按代码顺序复刻。

    覆盖：S2/S4/S5/S7/S9/S11(CPU 部分)/S13/S16/S19(CPU 部分)/S23(CPU 部分)
          + `_update_states` 的 O(B) 稳态标量更新
          + `_build_attention_metadata` 的 `.tolist()` / `pin_memory` 调用（不含真 H2D）

    不覆盖（必须在真机/replay 上标定）：真正的 H2D DMA、device kernel launch、
    `_update_states` 的 condense/swap 突变路径、MM/LoRA/DCP/GDN 分支。
    """

    def __init__(self, max_num_reqs: int, max_num_tokens: int, max_model_len: int,
                 block_size: int = 128, num_layers: int = 28):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.num_layers = num_layers
        self.K = (max_model_len + block_size - 1) // block_size
        # 与 core/ascend 一致：arange_np / positions / query_pos 都是 **int64**
        # (core gpu_model_runner.py:841 `np.arange(..., dtype=np.int64)`，
        #  ascend model_runner_v1.py:453–456 `_positions_cpu_buf` int64)
        self.arange_np = np.arange(max_num_tokens + 1, dtype=np.int64)
        self._positions_np_buf = np.empty(max_num_tokens, dtype=np.int64)
        self.query_pos_np = np.empty(max_num_tokens, dtype=np.int64)
        self.num_computed_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        self.token_ids_flat = np.arange(max_num_reqs * max_model_len, dtype=np.int32)
        self.query_start_loc_np = np.zeros(max_num_reqs + 2, dtype=np.int32)
        self.optimistic_seq_lens = np.zeros(max_num_reqs, dtype=np.int32)
        self.block_table_np = np.zeros((max_num_reqs, self.K), dtype=np.int32)
        self.req_state: dict[str, _ReqState] = {}
        if torch is not None:
            torch.set_num_threads(1)
            self.token_ids_flat_t = torch.from_numpy(self.token_ids_flat)
            self.input_ids_cpu = torch.empty(max_num_tokens, dtype=torch.int32)
        self.h2d_calls = 0

    def _touch_h2d(self, n: int) -> None:
        """占位：真实实现是 copy_to_gpu / pin_memory().to()；这里只计调用次数。"""
        self.h2d_calls += 1

    def step(self, B: int, T: int, spec: int = 0, count_h2d: bool = True) -> str:
        assert T <= len(self.arange_np) - 1, (T, len(self.arange_np))
        # --- 上游：engine 侧 tokens list → np.array  (ascend 1918–1927) ---
        if B > len(self.req_state):
            for i in range(len(self.req_state), B):
                self.req_state[f"req-{i}"] = _ReqState(128, 0)
        req_ids = list(self.req_state.keys())[:B]
        per_req = self._split_tokens(B, T)
        num_scheduled_tokens = np.array(per_req, dtype=np.int32)

        # --- S1 commit_block_table: H2D，O(B·K) ---
        if count_h2d:
            self._touch_h2d(B * self.K)

        # --- S2 req_indices ---
        req_indices = np.repeat(self.arange_np[:B], num_scheduled_tokens)

        # --- S3 num_valid_tokens / _build_attn_state（数据依赖分支） ---
        if spec:
            num_valid = np.array([n + spec for n in num_scheduled_tokens], dtype=np.int32)
        else:
            num_valid = num_scheduled_tokens
        nct = self.num_computed_cpu[:B]
        nst = num_scheduled_tokens
        if bool(np.all(nst == 1)):
            attn_state = "DecodeOnly"
        elif bool(np.all(nct == 0)):
            attn_state = "PrefillNoCache"
        elif bool(np.all(num_valid == 1)):
            attn_state = "ChunkedPrefill"
        else:
            attn_state = "SpecDecoding" if spec else "PrefillCacheHit"

        # --- S4 _get_cumsum_and_arange ---
        cu_num_tokens = np.cumsum(nst)
        cumsums_offsets = np.repeat(cu_num_tokens - nst, nst)
        query_pos = np.subtract(self.arange_np[:T], cumsums_offsets, out=self.query_pos_np[:T])

        # --- S5 positions ---
        positions_np = np.add(nct[req_indices], query_pos, out=self._positions_np_buf[:T])

        # --- S7 _compute_prev_positions ---
        prev_map = self.req_state
        _prev_positions = [prev_map.get(r, -1) for r in req_ids]
        if count_h2d:
            self._touch_h2d(B)

        # --- S9 token_indices + index_select ---
        # 真实实现里 positions < max_model_len；这里按 max_model_len 取模以避免缓冲越界
        token_indices = (positions_np % self.max_model_len) + req_indices * self.max_model_len
        if torch is not None:
            ti = torch.from_numpy(token_indices)   # 已是 int64，无需 cast
            torch.index_select(self.token_ids_flat_t, 0, ti, out=self.input_ids_cpu[:T])
        else:
            np.take(self.token_ids_flat, token_indices)

        # --- S11 query_start_loc 填充（+ H2D 全量）---
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:B + 1] = cu_num_tokens
        if count_h2d:
            self._touch_h2d(self.max_num_reqs + 2)

        # --- S13 optimistic_seq_lens ---
        np.add(nct, nst, out=self.optimistic_seq_lens[:B])
        if B < self.max_num_reqs:
            self.optimistic_seq_lens[B:].fill(0)

        # --- S16 discard 统计 ---
        num_tokens = [self.req_state[r].num_tokens for r in req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        discard_mask = self.optimistic_seq_lens[:B] < num_tokens_np
        discard_idx = np.nonzero(discard_mask)[0]
        if count_h2d:
            self._touch_h2d(len(discard_idx))
            self._touch_h2d(B)

        # --- S18/S19/S20/S21 上屏（H2D 与 device op 计数）---
        if count_h2d:
            self._touch_h2d(B)          # num_computed_tokens
            self._touch_h2d(T)          # req_indices
            self._touch_h2d(T)          # query_pos
            self._touch_h2d(B)          # num_scheduled_tokens
            self._touch_h2d(T)          # positions（device 计算，入队算 1 次）
            self._touch_h2d(B)          # seq_lens
            self._touch_h2d(0)          # slot_mapping: kernel launch
            self._touch_h2d(0)          # logits_indices: device op

        # --- S24 spec metadata ---
        if spec:
            for _ in range(5):
                if count_h2d:
                    self._touch_h2d(B * spec)

        # --- _update_states 稳态：O(B) 标量数组搬运 ---
        for _ in range(8):
            self.num_computed_cpu[:B] = self.num_computed_cpu[:B]

        # --- _build_attention_metadata（ascend attention_v1.build）---
        qsl = self.query_start_loc_np[:B + 1]
        if count_h2d:
            self._touch_h2d(B + 1)      # query_start_loc_cpu.pin_memory().to()
        actual_seq_lengths_q = qsl[1:].tolist()
        seq_lens_list = self.optimistic_seq_lens[:B].tolist()
        query_lens_t = np.subtract(self.query_start_loc_np[1:B + 1], self.query_start_loc_np[:B])
        _ = actual_seq_lengths_q, seq_lens_list, query_lens_t

        # --- 逐 layer 赋值 metadata（O(L)）---
        meta = {"attn_state": attn_state}
        for _ in range(self.num_layers):
            meta["attn_state"] = attn_state

        # 状态推进：模拟 decode / prefill 的 num_computed 变化
        self.num_computed_cpu[:B] += nst
        return attn_state

    @staticmethod
    def _split_tokens(B: int, T: int) -> list[int]:
        if B == 0:
            return []
        base, rem = divmod(T, B)
        out = [base] * B
        for i in range(rem):
            out[i] += 1
        return out


def bench_composite() -> list[Row]:
    rows: list[Row] = []
    grid: list[tuple[str, int, int, int]] = []
    for B in [1, 8, 32, 64, 128, 256]:
        for m in [1, 4, 16, 64, 256]:
            if B * m <= 65536:
                grid.append(("decode" if m == 1 else "chunked", B, m, 0))
    for B in [1, 8, 32, 64, 128]:
        grid.append(("spec_mtp2", B, 1, 2))

    for label, B, m, spec in grid:
        T = B * m
        replica = PrepareInputReplica(max_num_reqs=256, max_num_tokens=65536,
                                      max_model_len=8192, num_layers=28)
        replica.step(B, T, spec=spec)  # warmup + 计数
        h2d_calls = replica.h2d_calls

        def run(r=replica, B=B, T=T, spec=spec):
            r.step(B, T, spec=spec)

        st = bench(run, repeats=60, rounds=9)
        r = {
            "name": f"replica_{label}", "group": label, "size": B * m,
            "B": B, "tokens_per_req": m, "T": T, "spec": spec,
            "h2d_calls": h2d_calls, **st,
        }
        r["ns_per_element"] = st["median_ns"] / max(B, 1)
        rows.append(r)
    return rows


# --------------------------------------------------------------------------
# 5. main
# --------------------------------------------------------------------------


def metaline(tag: str, row: Row) -> str:
    return (
        f"[{tag}] {row['name']:<45} size={row['size']:<6} "
        f"median={row['median_us']:8.3f}us p10={row['p10_us']:8.3f} p90={row['p90_us']:8.3f} "
        f"ns/el={row['ns_per_element']:8.2f}"
    )


def write_csv(path: str, rows: list[Row]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v).replace(",", ";"))
            f.write(",".join(vals) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                low = line.lower()
                if low.startswith("cpu model") or low.startswith("hardware"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/model")
    ap.add_argument("--only", default="all",
                    choices=["all", "numpy", "python", "torch", "composite"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results: dict[str, list[Row]] = {}
    if args.only in ("all", "numpy"):
        results["numpy"] = bench_numpy()
        write_csv(os.path.join(args.out_dir, "microbench-numpy.csv"), results["numpy"])
    if args.only in ("all", "python"):
        results["python"] = bench_python()
        write_csv(os.path.join(args.out_dir, "microbench-python.csv"), results["python"])
    if args.only in ("all", "torch"):
        results["torch"] = bench_torch()
        write_csv(os.path.join(args.out_dir, "microbench-torch.csv"), results["torch"])
    if args.only in ("all", "composite"):
        results["composite"] = bench_composite()
        write_csv(os.path.join(args.out_dir, "microbench-composite.csv"), results["composite"])

    for tag, rows in results.items():
        print(f"\n===== {tag} =====")
        for r in rows:
            size = r.get("size") if "size" in r else f"B={r.get('B')},T={r.get('T')}"
            print(metaline(tag, {**r, "size": size}))

    try:
        affinity = sorted(os.sched_getaffinity(0))
    except Exception:
        affinity = []
    meta = {
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", None),
        "torch_num_threads": (torch.get_num_threads() if torch is not None else None),
        "affinity": [affinity[0], affinity[-1], len(affinity)] if affinity else None,
        "env": {k: v for k, v in os.environ.items()
                if k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "cpu_model": _cpu_model(),
        "note": "H2D/DMA 与 device kernel launch 不在本脚本范围（需 chip3）。",
    }
    with open(os.path.join(args.out_dir, "microbench-meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("\nmeta:", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
