#!/usr/bin/env bash
# 第二轮：torch_npu 无卡行为的边界（torch_npu 可以被 import！所以 shim 只需拦设备创建/操作）
set -u
run() {
  local name="$1"; shift
  echo "=== ${name} ==="
  "$@" 2>&1 | grep -v "owner does not match\|warnings.warn\|Permission mismatch\|LOG_WARNING\|can not use command" | tail -14
  echo "--- exit=${PIPESTATUS[0]} ---"
}

run "D1 import vllm_ascend.worker.model_runner_v1" python -c 'import vllm_ascend.worker.model_runner_v1 as m;print("ok", m.NPUModelRunner)'
run "D2 torch_npu is_available/device_count" python -c '
import torch, torch_npu
print("is_available", torch.npu.is_available())
try: print("device_count", torch.npu.device_count())
except Exception as e: print("device_count ERR", type(e).__name__, e)'
run "D3 torch.zeros(device=npu)" python -c '
import torch, torch_npu
print(torch.zeros(3, dtype=torch.int32, device="npu"))'
run "D4 torch.npu.current_stream()" python -c '
import torch, torch_npu
print(torch.npu.current_stream())'
run "D5 torch.npu.synchronize()" python -c '
import torch, torch_npu
torch.npu.synchronize(); print("ok")'
run "D6 torch_npu.NPUEvent" python -c '
import torch, torch_npu
print(torch.npu.Event)'
run "D7 empty(device=npu) + copy_" python -c '
import torch, torch_npu
c = torch.arange(8, dtype=torch.int32)
g = torch.zeros(8, dtype=torch.int32, device="npu")
g.copy_(c); print("ok")'
run "D8 torch.device('npu:0')" python -c '
import torch, torch_npu
d = torch.device("npu:0"); print(d, d.type, d.index)'
