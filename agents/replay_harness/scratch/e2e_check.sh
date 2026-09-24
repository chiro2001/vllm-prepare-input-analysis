set -u
echo "##### 1. 核心 replay（realmachine 口径）"
python -m pi_harness.runner._replay_main --preset realmachine --batch 1 --isl 128 --osl 64 --steps 300 --no-timing 2>/dev/null | tail -12
echo
echo "##### 2. 非法配置必须被拒绝"
python -m pi_harness.runner._replay_main --preset realmachine --isl 2048 --osl 64 --steps 10 2>&1 | grep -E "error|  - " | head -3
echo
echo "##### 3. trace 驱动（P2 通路）"
python -m pi_harness.runner._replay_main --trace /work/data/harness/trace_synth_prefill.jsonl --slot-mapping-mode noop --no-timing 2>/dev/null | tail -10
echo
echo "##### 4. CLI --help 与 sweep 帮助"
python -m pi_harness.runner.cli --help >/dev/null && echo "cli help OK"
python -m pi_harness.runner.cli --sweep isl --sweep-values 128 --steps 5 --preset realmachine --out /tmp/x 2>/dev/null | tail -3
echo
echo "##### 5. shim 审计（无未 shim 的设备操作）"
python - <<'PY'
import sys; sys.path.insert(0,"/work/harness")
import pi_harness.env as env
env.bootstrap()
from pi_harness.shim import report
r = report()
print("installed:", r["installed"], "| pin_memory_available:", r["pin_memory_available"])
print("warnings:", r["warnings"])
print("UNSHIMMED_DEVICE_OP count:", r["calls"].get("UNSHIMMED_DEVICE_OP", 0))
PY
