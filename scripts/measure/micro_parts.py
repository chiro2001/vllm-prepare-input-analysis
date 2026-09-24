import sys, time, statistics
sys.path.insert(0, "/work/harness")
import numpy as np, torch
import pi_harness.env as env
env.bootstrap()
import pi_harness.runner.triton_cpu as tc
tc.install()

def b(fn, reps=300, warm=30):
    for _ in range(warm): fn()
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter_ns(); fn(); ts.append((time.perf_counter_ns()-t0)/1000)
    return statistics.median(ts)

bt = torch.arange(64*2048, dtype=torch.int32).reshape(64,2048)
pos = torch.arange(16384, dtype=torch.int64)
qsl = torch.arange(9, dtype=torch.int32)*1
sm  = torch.zeros(16384, dtype=torch.int32)
out = np.full(16384, -1, dtype=np.int32)

print("empty lambda        ", b(lambda: None))
print("torch.Tensor.numpy  ", b(lambda: bt.numpy()))
print("pos.numpy           ", b(lambda: pos.numpy()))
print("np.full(16384,i32)  ", b(lambda: np.full(16384,-1,dtype=np.int32)))
print("torch.from_numpy    ", b(lambda: torch.from_numpy(out)))
print("sm[:8].copy_        ", b(lambda: sm[:8].copy_(torch.from_numpy(out[:8]))))
print("sm[:16384].copy_    ", b(lambda: sm[:16384].copy_(torch.from_numpy(out))))
print("np.repeat 0         ", b(lambda: np.repeat(np.arange(0), np.zeros(0,dtype=np.int64))))
print("np.cumsum(8)        ", b(lambda: np.cumsum(np.arange(8))))
print("bt[arr,arr]         ", b(lambda: bt.numpy()[[0,1,2],[0,1,2]]))
n=8
lens=np.ones(n,dtype=np.int64)
def full():
    _=pos.numpy()[:n]; _=bt.numpy(); _=np.cumsum(lens)
    cum=np.cumsum(lens); _=np.repeat(np.arange(n),lens)
    _=np.arange(n)-np.repeat(cum-lens,lens)
print("subset ops          ", b(full))
