#!/usr/bin/env bash
# 宿主机侧 manifest（镜像 digest / docker run 参数 / 宿主 CPU 事实）。
#
#   ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
#       bash harness/probes/collect_host_manifest.sh'
#
# 产出 data/harness/devdeps_manifest_host.json（只读命令，不启容器）。
set -euo pipefail

ROOT="${PI_HOST_ROOT:-$HOME/projects/vllm/prepare-input-phase}"
IMG="${PI_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}"
OUT="$ROOT/data/harness/devdeps_manifest_host.json"

python3 - "$ROOT" "$IMG" "$OUT" <<'PY'
import json, subprocess, sys, time, os, hashlib

root, img, out = sys.argv[1], sys.argv[2], sys.argv[3]

def sh(cmd):
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return p.stdout.strip() or p.stderr.strip()

inspect = sh(f"docker inspect --format '{{{{json .RepoDigests}}}}|{{{{.Id}}}}|{{{{.Created}}}}' {img}")
parts = inspect.split("|")
info = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": sh("hostname"),
    "image_ref": img,
    "image_repodigests": json.loads(parts[0]) if parts and parts[0].startswith("[") else parts[0],
    "image_id": parts[1] if len(parts) > 1 else None,
    "image_created": parts[2] if len(parts) > 2 else None,
    "docker_run_template": (
        "docker run --rm --network none --user $(id -u):$(id -g) "
        "--cpuset-cpus 200-215 -e HOME=/tmp -e USER=REMOTE_USER -e LOGNAME=REMOTE_USER "
        "-e TORCHINDUCTOR_CACHE_DIR=/tmp/ti_cache -e TRITON_CACHE_DIR=/tmp/triton_cache "
        "-e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PI_HARNESS_ROOT=/work/harness "
        "-e PYTHONPATH=/work/harness -e PYTHONDONTWRITEBYTECODE=1 "
        f"-v {root}:/work -w /work {img} bash -c <cmd>"
    ),
    "no_npu_devices_mounted": "yes (pi-docker.sh 从不传 --device /dev/davinci*)",
    "cpuset_used_by_probes": sh("echo $PI_CPUS") or "200-215 (默认)",
    "host_cpu_model": sh("lscpu | grep -m1 'Model name' || true"),
    "host_nproc": sh("nproc"),
    "host_numa": sh("lscpu | grep -i 'NUMA node(s)' || true"),
    "pi_docker_sha256": hashlib.sha256(
        open(os.path.join(root, "harness/scripts/pi-docker.sh"), "rb").read()).hexdigest(),
}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(info, fh, indent=2, ensure_ascii=False)
print(json.dumps({k: info[k] for k in ("image_id", "image_created", "hostname")},
                 indent=2, ensure_ascii=False))
print("wrote", out)
PY
