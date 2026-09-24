"""合成 workload（P1 自测用；正式 record&replay 数据层在 `pi_harness/workload/`）。

刻意复刻 vLLM 0.26 scheduler 的**输出语义**（而不是调用真 scheduler），
因为真 scheduler 需要的 KV cache manager / engine 装配在无卡环境里代价过高。
合成器的忠实度由 `docs/06-synthetic-load.md` 里与真机 trace 的 A/B 对照来背书。

产生的东西：
  * `vllm.v1.core.sched.output.SchedulerOutput`（真实 dataclass）
  * 每步的 `num_scheduled_tokens` / `num_computed_tokens` / block 分配 / prefix 命中
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import RunnerConfig


@dataclass
class _Req:
    req_id: str
    prompt_len: int
    block_ids: list[int]
    num_computed: int
    num_output_tokens: int = 0
    admitted: bool = False
    finished: bool = False


class SynthScheduler:
    """最小但语义正确的 chunked-prefill + decode + prefix-cache 调度器。"""

    def __init__(self, cfg: RunnerConfig, block_size: int, max_model_len: int):
        self.cfg = cfg
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.rng = random.Random(cfg.seed)
        self.reqs: dict[str, _Req] = {}
        self.order: list[str] = []
        self._next_block = 1  # block 0 保留
        self._free_blocks: list[int] = []
        # 共享前缀（prefix cache 命中）用的 block 列表
        self._shared_prefix_blocks: list[int] = []
        self._shared_prefix_len = 0
        self._pending: list[str] = []
        # 真 vLLM 语义：`finished_req_ids` 是"在上一步与当前步之间完成"的请求，
        # 且**不会**出现在同一 step 的 num_scheduled_tokens 里。
        # 因此合成器把完成事件延后一步上报。
        self._finished_pending: set[str] = set()
        self._init_requests()

    # ------------------------------------------------------------------ helpers
    def _alloc(self, n: int) -> list[int]:
        out = []
        for _ in range(n):
            if self._free_blocks:
                out.append(self._free_blocks.pop())
            else:
                out.append(self._next_block)
                self._next_block += 1
        return out

    def _init_requests(self) -> None:
        cfg = self.cfg
        n = cfg.batch
        # 确定共享前缀长度：prefix_hit_ratio 比例的请求有一段公共前缀
        n_shared = int(round(cfg.batch * cfg.prefix_hit_ratio))
        if n_shared > 0:
            self._shared_prefix_len = max(self.block_size, (cfg.isl // 4) // self.block_size * self.block_size)
            self._shared_prefix_blocks = self._alloc(self._shared_prefix_len // self.block_size)
        for i in range(n):
            req_id = f"req-{i:05d}"
            shared = i < n_shared and self._shared_prefix_len > 0
            r = _Req(
                req_id=req_id,
                prompt_len=cfg.isl,
                block_ids=list(self._shared_prefix_blocks) if shared else [],
                num_computed=self._shared_prefix_len if shared else 0,
            )
            self.reqs[req_id] = r
        self._pending = [r.req_id for r in self.reqs.values()]

    def _arrivals(self, step: int) -> list[str]:
        if self.cfg.arrival == "simultaneous" or step == 0:
            return list(self._pending)
        # stair: 每一步放 1/8；poisson: 均值 1/8
        k = max(1, len(self._pending) // 8)
        if self.cfg.arrival == "poisson":
            k = max(1, int(self.rng.expovariate(1.0 / max(k, 1))))
        k = min(k, len(self._pending))
        return self._pending[:k]

    # ------------------------------------------------------------------ main api
    def step(self, step_idx: int, spec_k: int) -> tuple[object, dict]:
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            SchedulerOutput,
        )

        cfg = self.cfg
        finished_reported = set(self._finished_pending)
        self._finished_pending.clear()
        new_reqs: list[object] = []
        cached_ids: list[str] = []
        num_scheduled: dict[str, int] = {}
        spec_tokens: dict[str, list[int]] = {}
        new_block_ids: list[tuple[list[int], ...] | None] = []
        num_computed_list: list[int] = []
        num_output_list: list[int] = []

        # 1) 本轮新到达的请求先进入 running（受 max_num_seqs 限制）
        arrivals = self._arrivals(step_idx)
        n_running = sum(
            1
            for r in self.reqs.values()
            if r.admitted and not r.finished and r.req_id not in finished_reported
        )
        admitted_now = []
        for rid in arrivals:
            if n_running + len(admitted_now) >= cfg.max_num_reqs:
                break
            admitted_now.append(rid)
        new_req_ids: set[str] = set()
        for rid in admitted_now:
            self._pending.remove(rid)
            r = self.reqs[rid]
            r.admitted = True
            self.order.append(rid)
            new_req_ids.add(rid)
            new_reqs.append(
                NewRequestData(
                    req_id=r.req_id,
                    prompt_token_ids=_make_token_ids(r.prompt_len, cfg.seed, cfg.shuffle_values, rid),
                    mm_features=[],
                    sampling_params=_make_sampling_params(),
                    pooling_params=None,
                    block_ids=(r.block_ids,),          # tuple[list[int], ...] 每个 KV cache group 一项
                    num_computed_tokens=r.num_computed,  # prefix cache 命中的 token 数
                    lora_request=None,
                )
            )

        # 2) 预算分配：先 decode 后 prefill（对齐 vLLM 的 running-first 语义）
        budget = cfg.max_num_batched_tokens
        running = [
            self.reqs[rid]
            for rid in self.order
            if self.reqs[rid].admitted
            and not self.reqs[rid].finished
            and rid not in finished_reported
        ]
        decodes = [r for r in running if r.num_computed >= r.prompt_len]
        prefills = [r for r in running if r.num_computed < r.prompt_len]

        for r in decodes:
            n = 1 + spec_k
            if n > budget:
                break
            budget -= n
            num_scheduled[r.req_id] = n
            if spec_k:
                spec_tokens[r.req_id] = _make_token_ids(spec_k, cfg.seed, cfg.shuffle_values, r.req_id)

        chunk = cfg.extra.get("chunk_size", cfg.max_num_batched_tokens)
        for r in prefills:
            if budget <= 0:
                break
            remaining = r.prompt_len - r.num_computed
            n = min(remaining, chunk, budget)
            budget -= n
            num_scheduled[r.req_id] = n

        # 回滚"已进入 scheduled_new_reqs 但没拿到 token 预算"的请求。
        # 真 vLLM 语义：新请求**只有真的被调度到 token** 才会出现在
        # `scheduled_new_reqs` 里；否则它应当继续留在 waiting 队列。
        # 少了这一步，worker 的 `input_batch` 里会出现"不在 num_scheduled_tokens 里"
        # 的请求，`[num_scheduled_tokens[i] for i in req_ids]` 直接 KeyError。
        unscheduled_new = [r for r in new_reqs if r.req_id not in num_scheduled]
        if unscheduled_new:
            _drop = {r.req_id for r in unscheduled_new}
            new_reqs = [r for r in new_reqs if r.req_id not in _drop]
            for rid in _drop:
                r = self.reqs[rid]
                r.admitted = False
                if rid in self.order:
                    self.order.remove(rid)
            # 放回等待队列头部，保持原顺序
            self._pending = [rid for rid in self._pending] + sorted(
                _drop, key=lambda x: int(x.split("-")[1])
            )

        # 3) 为 prefill 扩展 block（对齐 block_size 的分配粒度）
        #    注意：真 vLLM 语义 —— 本步**新加入**的请求只出现在
        #    `scheduled_new_reqs`（携带完整 block_ids），绝不出现在
        #    `scheduled_cached_reqs`（那是"worker 已缓存该请求状态"的增量 diff）。
        for r in running:
            n = num_scheduled.get(r.req_id, 0)
            if n == 0:
                continue
            if r.req_id in new_req_ids:
                continue
            need = (r.num_computed + n + self.block_size - 1) // self.block_size
            have = len(r.block_ids)
            if need > have:
                extra = self._alloc(need - have)
                r.block_ids.extend(extra)
                cached_ids.append(r.req_id)
                num_computed_list.append(r.num_computed)
                num_output_list.append(r.num_output_tokens)
                new_block_ids.append((extra,))
            else:
                cached_ids.append(r.req_id)
                num_computed_list.append(r.num_computed)
                num_output_list.append(r.num_output_tokens)
                new_block_ids.append(None)

        # 4) 推进状态（模拟本步执行完成后的 scheduler 视角）
        for r in running:
            n = num_scheduled.get(r.req_id, 0)
            if n == 0:
                continue
            r.num_computed += n
            # output_token_ids 长度 = 已算 token 数 - prompt 长度（prefill 阶段为 0）
            r.num_output_tokens = max(0, r.num_computed - r.prompt_len)
            if r.num_computed >= r.prompt_len + cfg.osl:
                r.finished = True
                self._finished_pending.add(r.req_id)

        out = SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=CachedRequestData(
                req_ids=cached_ids,
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=new_block_ids,
                num_computed_tokens=num_computed_list,
                num_output_tokens=num_output_list,
            ),
            num_scheduled_tokens=num_scheduled,
            total_num_scheduled_tokens=sum(num_scheduled.values()),
            scheduled_spec_decode_tokens=spec_tokens,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=finished_reported,
            free_encoder_mm_hashes=[],
        )
        info = {
            "n_running": len(running),
            "n_prefill": len(prefills),
            "n_decode": len(decodes),
            "total_scheduled": out.total_num_scheduled_tokens,
        }
        return out, info

    def done(self) -> bool:
        return all(r.finished for r in self.reqs.values())


# --------------------------------------------------------------------------- utils
_SAMPLING_PARAMS_CACHE = {}


def _make_sampling_params():
    """真实 `SamplingParams`，参数固定（GREEDY, 无 logprobs/penalties）。"""
    if "greedy" not in _SAMPLING_PARAMS_CACHE:
        from vllm.sampling_params import SamplingParams

        _SAMPLING_PARAMS_CACHE["greedy"] = SamplingParams(temperature=0.0, max_tokens=64)
    return _SAMPLING_PARAMS_CACHE["greedy"]


def _make_token_ids(n: int, seed: int, shuffle: bool, req_id: str) -> list[int]:
    if shuffle:
        # A/B：同形状、不同取值。用一个只依赖 req_id/seed 的确定性 PRNG。
        rng = random.Random(hash((seed, req_id, "shuffle")) & 0xFFFFFFFF)
        return [rng.randrange(0, 1000) for _ in range(n)]
    rng = random.Random(seed)
    return [rng.randrange(0, 1000) for _ in range(n)]
