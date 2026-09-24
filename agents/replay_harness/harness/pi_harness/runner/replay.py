"""P1 replay driver：逐 step 执行真实的 `_update_states` + `_prepare_inputs`。"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .build import AttrAudit, build_runner
from .config import RunnerConfig
from .synth import SynthScheduler
from .timer import SubStepTimer, percentile, wrap_prepare_input_path
from . import triton_cpu


def _make_input_batch(runner, cfg: RunnerConfig):
    """重造一个干净的 NPUInputBatch（长 run 里负载跑完后重置用）。"""
    InputBatchCls = type(runner.input_batch)
    # 注意：runner.input_batch 可能是 build.py 里的原始类（未做审计子类包装）
    return InputBatchCls(
        max_num_reqs=cfg.max_num_reqs,
        max_model_len=runner.model_config.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        device=runner.device,
        pin_memory=cfg.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[cfg.block_size],
        kernel_block_sizes=[[cfg.block_size]],
        is_spec_decode=bool(runner.num_spec_tokens),
        logitsprocs=None,
        logitsprocs_need_output_token_ids=False,
        is_pooling_model=False,
        num_speculative_tokens=runner.num_spec_tokens,
        cp_kv_cache_interleave_size=1,
    )


class PrepareInputReplay:
    def __init__(self, cfg: RunnerConfig, timing: bool = True):
        self.cfg = cfg
        self.timing = timing
        self.timer = SubStepTimer() if timing else None
        self.runner = None
        self.meta: dict = {}
        self.rows: list[dict] = []
        self.substep_rows: list[dict] = []
        self._warmed = False
        self._trace_iter = None
        self.sched = None
        self._unit_note = "timings are per-step DELTAS in seconds"
        self.last_wall_s: float | None = None

    # ------------------------------------------------------------------ lifecycle
    def build(self) -> None:
        AttrAudit.reset()
        if self.cfg.trace_path:
            # trace 模式：并发数与 token 预算由 **trace 本身**决定，preset 的值只是
            # 初始猜测。必须先把 buffer 尺寸抬到 trace 的实际最大值，否则
            # `validate()` 会报 "batch > max_num_reqs"，或者更糟——
            # `CpuGpuBuffer` 越界写。这是"数据驱动尺寸"的正确做法。
            self._autosize_from_trace()
        errs = self.cfg.validate()
        if errs:
            raise ValueError(
                "harness 配置不合法（真机引擎也会拒绝这些配置）：\n  - "
                + "\n  - ".join(errs)
            )
        self.runner, self.meta = build_runner(self.cfg)
        if self.timer is not None:
            wrap_prepare_input_path(self.timer)
        self.sched = SynthScheduler(self.cfg, self.cfg.block_size, self.runner.model_config.max_model_len)
        self._trace_iter = None
        if self.cfg.trace_path:
            from .trace_source import iter_steps

            self._trace_iter = iter_steps(self.cfg.trace_path)
            self.meta["trace_path"] = self.cfg.trace_path
            self.meta["trace_autosize"] = getattr(self, "_trace_autosize", None)
        self.meta["attr_audit"] = {k: dict(v) for k, v in AttrAudit.store().items()}

    def _autosize_from_trace(self) -> None:
        """扫一遍 trace，把 max_num_reqs / max_num_batched_tokens 抬到实际峰值。"""
        from .trace_source import iter_steps

        max_reqs = self.cfg.max_num_reqs
        max_tokens = self.cfg.max_num_batched_tokens
        max_isl = self.cfg.isl
        import itertools

        for so, _info in itertools.islice(iter_steps(self.cfg.trace_path), 20000):
            max_reqs = max(max_reqs, len(so.num_scheduled_tokens))
            max_tokens = max(max_tokens, int(so.total_num_scheduled_tokens))
            for nrd in getattr(so, "scheduled_new_reqs", []):
                pt = getattr(nrd, "prompt_token_ids", None)
                if pt:
                    max_isl = max(max_isl, len(pt))
        self.cfg.max_num_reqs = max_reqs
        self.cfg.max_num_batched_tokens = max_tokens
        self.cfg.isl = max_isl
        # trace 里的 prompt 可能比 preset 的 max_model_len 长（例如 prefill 长上下文 trace
        # 配 realmachine 的 2048）。真机跑这种 trace 时的引擎一定开了足够大的
        # max_model_len，所以这里同步抬高——否则 block table 尺寸不够。
        need_mml = max_isl + max(self.cfg.osl, 1)
        if self.cfg.max_model_len is None or self.cfg.max_model_len < need_mml:
            self.cfg.max_model_len = need_mml
        self._trace_autosize = {
            "max_num_reqs": max_reqs,
            "max_num_batched_tokens": max_tokens,
            "max_isl": max_isl,
            "max_model_len": self.cfg.max_model_len,
        }

    def _one_step(self, step_idx: int) -> dict | None:
        r = self.runner
        if self._trace_iter is not None:
            nxt = next(self._trace_iter, None)
            if nxt is None:
                return None
            sch_out, info = nxt
        else:
            sch_out, info = self.sched.step(step_idx, r.num_spec_tokens)
        if sch_out.total_num_scheduled_tokens == 0 and not sch_out.finished_req_ids:
            return None
        if sch_out.total_num_scheduled_tokens == 0:
            return None

        with torch.inference_mode():
            tri_before = triton_cpu.stats()
            t0 = time.perf_counter()
            deferred = r._update_states(sch_out)
            t1 = time.perf_counter()

            req_ids = r.input_batch.req_ids
            missing = [rid for rid in req_ids if rid not in sch_out.num_scheduled_tokens]
            if missing:
                raise RuntimeError(
                    "harness 不变式被打破：input_batch 里的请求不在 "
                    f"scheduled num_scheduled_tokens 中：{missing[:8]}。"
                    "这违反 vLLM scheduler 契约（新请求只有拿到 token 预算才会进入 "
                    "scheduled_new_reqs），请修负载生成器而不是绕过这里。"
                )
            num_scheduled_tokens_np = np.array(
                [sch_out.num_scheduled_tokens[i] for i in req_ids], dtype=np.int32
            )
            logits_indices, spec_decode_metadata, total_num_scheduled_tokens = (
                r._prepare_inputs(sch_out, num_scheduled_tokens_np)
            )
            t2 = time.perf_counter()
            if deferred is not None:
                deferred()
            t3 = time.perf_counter()
            tri_after = triton_cpu.stats()

        return {
            "step": step_idx,
            "phase": info,
            "update_states_us": (t1 - t0) * 1e6,
            "prepare_inputs_us": (t2 - t1) * 1e6,
            "deferred_fixup_us": (t3 - t2) * 1e6,
            "total_us": (t3 - t0) * 1e6,
            "num_reqs": len(req_ids),
            "total_scheduled_tokens": int(sch_out.total_num_scheduled_tokens),
            "logits_indices_len": int(logits_indices.shape[0]),
            "has_spec_decode": spec_decode_metadata is not None,
            "triton_cpu_us": (tri_after["fallback_ms"] - tri_before["fallback_ms"]) * 1e3,
            "triton_launches": tri_after["launches"] - tri_before["launches"],
        }

    def _one_step_with(self, step_idx: int, sch_out):
        """用给定的 SchedulerOutput 跑一步（调试/固定 trace 用）。"""
        r = self.runner
        with torch.inference_mode():
            deferred = r._update_states(sch_out)
            req_ids = r.input_batch.req_ids
            missing = [rid for rid in req_ids if rid not in sch_out.num_scheduled_tokens]
            if missing:
                raise RuntimeError(
                    "harness 不变式被打破：input_batch 里的请求不在 "
                    f"num_scheduled_tokens 中：{missing[:8]}"
                )
            num_scheduled_tokens_np = np.array(
                [sch_out.num_scheduled_tokens[i] for i in req_ids], dtype=np.int32
            )
            logits_indices, spec_decode_metadata, total = r._prepare_inputs(
                sch_out, num_scheduled_tokens_np
            )
            if deferred is not None:
                deferred()
        return {
            "step": step_idx,
            "logits_indices_len": int(logits_indices.shape[0]),
            "has_spec_decode": spec_decode_metadata is not None,
            "total": total,
        }

    def run(self, steps: int | None = None, warmup: int = 3,
            steady_seconds: float | None = None) -> dict:
        if self.runner is None:
            self.build()
        steps = steps or self.cfg.steps
        for i in range(warmup):
            self._one_step(-1 - i)
        if self.timer is not None:
            self.timer.reset()
        self.rows = []
        self.substep_rows = []
        # 注意：`timer.inclusive` 是**自 reset 起的累计值**。CSV 里必须写每步增量，
        # 否则对列求和会得到累计和的再求和（一个无意义的巨大数）。
        _prev_cum: dict[str, float] = {}

        def _snapshot_delta(step_idx: int) -> dict:
            cur = dict(self.timer.inclusive)
            delta = {}
            for k, v in cur.items():
                d = v - _prev_cum.get(k, 0.0)
                if d < 0:  # 防御：极端情况下时钟回拨
                    d = 0.0
                delta[k] = d
            _prev_cum.clear()
            _prev_cum.update(cur)
            return {"step": step_idx, **delta}

        t_start = time.perf_counter()
        if steady_seconds:
            # PMU 采集用：负载太短（几百 µs/步）时，perf/kperfx 根本挂不上。
            # 这里按**墙钟**跑够 duration，步数自然增长；用 step 取模让负载在
            # 长 run 里保持"稳态形状"（请求反复 new->prefill->decode->finish）。
            i = 0
            deadline = t_start + float(steady_seconds)
            while time.perf_counter() < deadline:
                row = self._one_step(i)
                if row is None:
                    # 负载跑完了就重建（重新灌入请求），保持 CPU 一直有活干
                    self.sched = SynthScheduler(
                        self.cfg, self.cfg.block_size, self.runner.model_config.max_model_len
                    )
                    self.runner.requests.clear()
                    self.runner.input_batch = _make_input_batch(self.runner, self.cfg)
                    i = 0
                    continue
                self.rows.append(row)
                if self.timer is not None:
                    self.substep_rows.append(_snapshot_delta(i))
                i += 1
        else:
            for i in range(steps):
                row = self._one_step(i)
                if row is None:
                    break
                self.rows.append(row)
                if self.timer is not None:
                    self.substep_rows.append(_snapshot_delta(i))
        wall = time.perf_counter() - t_start
        self.last_wall_s = wall
        return self.summary(wall)

    # ------------------------------------------------------------------ reporting
    def summary(self, wall_s: float | None = None) -> dict:
        pi = [r["prepare_inputs_us"] for r in self.rows]
        us = [r["update_states_us"] for r in self.rows]
        tot = [r["total_us"] for r in self.rows]
        out = {
            "config": self.cfg.to_dict(),
            "meta": self.meta,
            "steps_run": len(self.rows),
            "wall_s": wall_s,
            "prepare_inputs_us": {
                "p50": percentile(pi, 50),
                "p90": percentile(pi, 90),
                "p99": percentile(pi, 99),
                "mean": float(np.mean(pi)) if pi else float("nan"),
                "sum": float(np.sum(pi)),
            },
            "update_states_us": {
                "p50": percentile(us, 50),
                "p90": percentile(us, 90),
                "mean": float(np.mean(us)) if us else float("nan"),
            },
            "scope_total_us": {
                "p50": percentile(tot, 50),
                "p90": percentile(tot, 90),
                "mean": float(np.mean(tot)) if tot else float("nan"),
            },
            "substeps": (
                self.timer.as_rows(float(np.sum(tot)) if tot else 0.0)
                if self.timer is not None
                else []
            ),
            "host": {
                "node": platform.node(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        return out

    def dump(self, outdir: str | Path) -> tuple[Path, Path, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        per_step = outdir / f"per_step_{ts}.csv"
        summary = outdir / f"summary_{ts}.json"
        substeps = outdir / f"substep_{ts}.csv"
        with per_step.open("w") as f:
            if self.rows:
                f.write(",".join(self.rows[0].keys()) + "\n")
                for row in self.rows:
                    f.write(
                        ",".join(
                            json.dumps(v) if isinstance(v, dict) else str(v)
                            for v in row.values()
                        )
                        + "\n"
                    )
        summary.write_text(
            json.dumps(self.summary(self.last_wall_s), indent=2, ensure_ascii=False)
        )
        if self.substep_rows:
            keys = sorted({k for row in self.substep_rows for k in row})
            with substeps.open("w") as f:
                f.write(",".join(keys) + "\n")
                for row in self.substep_rows:
                    f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        return per_step, summary, substeps
