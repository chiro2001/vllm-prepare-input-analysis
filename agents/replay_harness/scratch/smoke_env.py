"""冒烟：无卡下能否拿到真实 NPUModelRunner 类 + 构造 NPUInputBatch。"""
import sys, time, traceback

sys.path.insert(0, "/work/harness")

import pi_harness.env as env

t0 = time.perf_counter()
env.bootstrap()
print(f"[smoke] bootstrap ok in {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
NPUModelRunner = env.import_runner_class()
print(f"[smoke] imported NPUModelRunner={NPUModelRunner} in {time.perf_counter()-t0:.2f}s")

import torch

NPUInputBatch = env.import_input_batch_class()

t0 = time.perf_counter()
ib = NPUInputBatch(
    max_num_reqs=8,
    max_model_len=4096,
    max_num_batched_tokens=2048,
    device=torch.device("cpu"),
    pin_memory=False,
    vocab_size=1024,
    block_sizes=[128],
    kernel_block_sizes=[[128]],
    is_spec_decode=False,
    logitsprocs=None,
    logitsprocs_need_output_token_ids=False,
    is_pooling_model=False,
    num_speculative_tokens=0,
    cp_kv_cache_interleave_size=1,
)
print(f"[smoke] NPUInputBatch built in {time.perf_counter()-t0:.2f}s; "
      f"block_table groups={len(ib.block_table.block_tables)} "
      f"bt={ib.block_table[0].block_table.cpu.shape}")

from pi_harness.shim import report
rep = report()
print("[smoke] shim calls:", rep["calls"])
print("[smoke] shim warnings:", len(rep["warnings"]))
