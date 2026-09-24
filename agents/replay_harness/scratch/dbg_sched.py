"""诊断：打印 synth scheduler 每步产出的 SchedulerOutput 摘要。"""
import sys
sys.path.insert(0, "/work/harness")

from pi_harness.runner.config import RunnerConfig
from pi_harness.runner.synth import SynthScheduler

cfg = RunnerConfig(batch=4, isl=512, osl=8, max_num_reqs=8,
                   max_num_batched_tokens=1024, block_size=128, steps=6)
cfg.extra["chunk_size"] = 256
s = SynthScheduler(cfg, cfg.block_size, 32768)
for step in [-1, -2, 0, 1, 2, 3]:
    out, info = s.step(step, 0)
    new = [r.req_id for r in out.scheduled_new_reqs]
    print(f"step={step:3d} new={new} cached={out.scheduled_cached_reqs.req_ids} "
          f"nsched={out.num_scheduled_tokens} finished={out.finished_req_ids} "
          f"ncomp={out.scheduled_cached_reqs.num_computed_tokens}")
