#!/usr/bin/env bash
# 由 replay_harness 临时使用：快速导入梯度探测（probe_devdeps 的正式版在 harness/probes/）
set -u
run() {
  local name="$1"; shift
  echo "=== ${name} ==="
  # shellcheck disable=SC2068
  "$@" 2>&1 | tail -12
  echo "--- exit=${PIPESTATUS[0]} ---"
}

run "C1 torch(autoload=0)" python -c 'import torch;print(torch.__version__, hasattr(torch,"npu"))'
run "C2 import vllm (plugins default)" python -c 'import vllm;print("ok",vllm.__version__)'
run "C3 import vllm (VLLM_PLUGINS='')" env VLLM_PLUGINS= python -c 'import vllm;print("ok",vllm.__version__)'
run "C4 current_platform (VLLM_PLUGINS='')" env VLLM_PLUGINS= python -c 'from vllm.platforms import current_platform;print(current_platform)'
run "C5 import torch_npu" python -c 'import torch_npu;print("ok")'
run "C6 import torch_npu (autoload=1)" env TORCH_DEVICE_BACKEND_AUTOLOAD=1 python -c 'import torch_npu;print("ok")'
run "C7 import vllm_ascend" python -c 'import vllm_ascend;print("ok")'
run "C8 vllm.v1.utils CpuGpuBuffer(device=cpu)" python -c '
import torch
from vllm.v1.utils import CpuGpuBuffer
b = CpuGpuBuffer(1024, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
b.np[:5]=1; b.copy_to_gpu(5); print("ok", b.np[:5], b.gpu[:5])'
run "C9 get_dcp_group without init" python -c '
import vllm
from vllm.distributed import get_dcp_group
print(get_dcp_group())'
run "C10 device=privateuse1 available?" python -c '
import torch
print([d for d in ["cpu","cuda","npu","privateuse1"] if True])
print("npu backend:", torch._C._get_privateuse1_backend_name())'
