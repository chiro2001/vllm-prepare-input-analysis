"""探测：能不能用真实 vllm config 类构造 harness 需要的配置对象。"""
import sys, inspect
sys.path.insert(0, "/work/harness")
import pi_harness.env as env
env.bootstrap()


def show(name, cls):
    try:
        sig = inspect.signature(cls)
    except Exception as e:
        sig = f"<sig err {e}>"
    print(f"--- {name}: {sig}")


import vllm.config as C

for n in ["SchedulerConfig", "CacheConfig", "ParallelConfig", "CompilationConfig",
          "LoRAConfig", "ObservabilityConfig", "LoadConfig", "VllmConfig",
          "SpeculativeConfig", "ModelConfig", "DeviceConfig", "KVTransferConfig"]:
    if hasattr(C, n):
        show(n, getattr(C, n))

print("=== try construct ===")
try:
    sc = C.SchedulerConfig()
    print("SchedulerConfig() ok:", sc.max_num_seqs, sc.max_num_batched_tokens, sc.enable_chunked_prefill)
except Exception as e:
    print("SchedulerConfig() FAIL", type(e).__name__, e)

try:
    cc = C.CacheConfig()
    print("CacheConfig() ok:", cc.block_size, cc.cache_dtype)
except Exception as e:
    print("CacheConfig() FAIL", type(e).__name__, e)

try:
    pc = C.ParallelConfig()
    print("ParallelConfig() ok:", pc.world_size, pc.tensor_parallel_size)
except Exception as e:
    print("ParallelConfig() FAIL", type(e).__name__, e)

try:
    mc = C.ModelConfig()
    print("ModelConfig() ok:", mc)
except Exception as e:
    print("ModelConfig() FAIL", type(e).__name__, str(e)[:200])

try:
    vc = C.VllmConfig()
    print("VllmConfig() ok:", vc)
except Exception as e:
    print("VllmConfig() FAIL", type(e).__name__, str(e)[:300])

R = env.import_runner_class()
print("mro:", [c.__name__ for c in R.__mro__])
