"""冒烟：跑通 P1 replay（真实 _update_states + _prepare_inputs）。"""
import json
import sys
import traceback

sys.path.insert(0, "/work/harness")

from pi_harness.runner.config import RunnerConfig
from pi_harness.runner.replay import PrepareInputReplay

cfg = RunnerConfig(
    batch=8,
    isl=512,
    osl=16,
    max_num_reqs=16,
    max_num_batched_tokens=4096,
    block_size=128,
    steps=30,
)
cfg.extra["chunk_size"] = 512

rl = PrepareInputReplay(cfg)
try:
    rl.build()
    print("[replay] meta:", json.dumps(rl.meta, indent=2)[:800])
    # 逐 step 手动跑，便于诊断
    for step in range(-3, 6):
        out, info = rl.sched.step(step, 0)
        print(f"[dbg] step={step} new={[x.req_id for x in out.scheduled_new_reqs]} "
              f"cached={out.scheduled_cached_reqs.req_ids} nsched={out.num_scheduled_tokens} "
              f"requests={sorted(rl.runner.requests)} "
              f"input_batch={rl.runner.input_batch.req_ids} "
              f"ncomp={out.scheduled_cached_reqs.num_computed_tokens}", flush=True)
        try:
            row = rl._one_step_with(step, out)
            print(f"[dbg]   -> ok {row}", flush=True)
        except Exception:
            traceback.print_exc()
            break
    s = rl.summary()
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("[replay] steps_run:", s["steps_run"])
print("[replay] prepare_inputs_us:", json.dumps(s["prepare_inputs_us"], indent=2))
print("[replay] update_states_us:", json.dumps(s["update_states_us"], indent=2))
print("[replay] substeps:")
for row in s["substeps"][:20]:
    print(f"   {row['substep']:55s} calls={row['calls']:6d} total_us={row['total_us']:10.1f}")
print("[replay] first 3 rows:", json.dumps(rl.rows[:3], indent=2))
print("[replay] attr_audit:", json.dumps(rl.meta.get("attr_audit"), indent=2))
