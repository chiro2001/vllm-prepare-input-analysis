"""微基准：triton_cpu 兜底实现的绝对成本（用于从 prepare_input 时间里扣掉它）。"""
import sys, time, statistics
sys.path.insert(0, "/work/harness")
import numpy as np, torch
from pi_harness.runner.triton_cpu import _slot_mapping_cpu

def bench(num_reqs, total_tokens, max_num_tokens=16384, block_size=128,
          max_blocks=2048, reps=200):
    bt = torch.arange(num_reqs * max_blocks, dtype=torch.int32).reshape(num_reqs, max_blocks)
    qsl = torch.tensor([i * (total_tokens // max(num_reqs, 1)) for i in range(num_reqs + 1)],
                       dtype=torch.int32)
    pos = torch.arange(total_tokens, dtype=torch.int64) % 4096
    sm = torch.zeros(max_num_tokens, dtype=torch.int32)
    for _ in range(20):
        _slot_mapping_cpu(total_tokens, max_num_tokens, qsl, pos, bt, block_size, sm, -1)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        _slot_mapping_cpu(total_tokens, max_num_tokens, qsl, pos, bt, block_size, sm, -1)
        ts.append((time.perf_counter_ns() - t0) / 1000.0)
    return dict(num_reqs=num_reqs, total_tokens=total_tokens,
                max_num_tokens=max_num_tokens,
                p50_us=statistics.median(ts), p90_us=sorted(ts)[int(0.9*len(ts))],
                mean_us=statistics.fmean(ts))

print("case, p50_us, p90_us")
rows=[]
for num_reqs, tot in [(1,1),(8,8),(32,32),(64,64),(8,2048),(32,8192),(64,16384),(1,8192)]:
    r = bench(num_reqs, tot)
    rows.append(r)
    print(f"reqs={num_reqs:3d} tokens={tot:6d} max_tokens={r['max_num_tokens']:6d} "
          f"p50={r['p50_us']:8.2f}us p90={r['p90_us']:8.2f}us")
# 纯输出写回成本
sm = torch.zeros(16384, dtype=torch.int32); arr = np.zeros(16384, dtype=np.int32)
ts=[]
for _ in range(500):
    t0=time.perf_counter_ns(); sm[:].copy_(torch.from_numpy(arr)); ts.append((time.perf_counter_ns()-t0)/1000)
print(f"[copy 16384 int32] p50={statistics.median(ts):.2f}us")
