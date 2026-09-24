#!/usr/bin/env bash
# run_harness_arm.sh <stock|A|B|C> [extra cli args...]
#
# Runs the project's no-card `prepare_input` harness (real vLLM 0.26.0 +
# vllm-ascend 0.26.0rc1 code path) against one of our rebuilt CPythons.
#
# How the CPython swap is done without touching any image:
#   the image's `bin/python3.12` has DT_RPATH=/usr/local/python3.12.13/lib,
#   and DT_RPATH beats LD_LIBRARY_PATH, so the *only* reliable override is a
#   read-only file bind-mount over libpython3.12.so.1.0 itself.  Everything
#   else in the image (stdlib, site-packages, torch, CANN) stays untouched.
#
# `stock` runs the unmodified image (control: proves the swap itself is inert
# or, if not, bounds by how much).
#
# Never mounts /dev/davinci*, uses --network none and CPUs 200-215 only.
set -uo pipefail

arm=${1:?usage: run_harness_arm.sh <stock|A|B|C> [cli args...]}
shift || true

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
root=${PI_HOST_ROOT:-/home/REMOTE_USER/projects/vllm/prepare-input-phase}
img=${PI_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}
cpus=${PI_CPUS:-200-215}
steps=${STEPS:-200}
# NOTE: --out must be a *container* path (the host tree is mounted at /work).
outdir_host="$exp/data/harness-arm$arm"
out="/work/exp-cpython/data/harness-arm$arm"
tag=${TAG:-realmachine_b1_isl128_arm$arm}
mkdir -p "$outdir_host"

mounts=(-v "$root:/work" -v "$HOME/models:/models:ro")
if [ "$arm" != stock ]; then
	src="$exp/build-arm$arm/Python-3.12.13"
	lib="$src/libpython3.12.so.1.0"
	[ -f "$lib" ] || { echo "missing $lib" >&2; exit 1; }
	mounts+=(-v "$lib:/usr/local/python3.12.13/lib/libpython3.12.so.1.0:ro")
fi

cmd="cd /work/harness && python -m pi_harness.runner.cli --preset realmachine \
--batch 1 --isl 128 --steps $steps --out $out --tag $tag $*"

# --------------------------------------------------------------------------
# Optional PMU collection *of the container process itself* (PERF=1).
# The image ships no perf, but the host's perf only misses 6 shared libraries
# that are unrelated to the Python runtime (tracing/PMU backends), so bind-
# mounting exactly those 6 cannot shadow libc/libm or change what we measure.
# --------------------------------------------------------------------------
perfdir="$exp/data/perf"
if [ "${PERF:-0}" = 1 ]; then
	mkdir -p "$perfdir"
	for so in libopencsd_c_api.so.1 libopencsd.so.1 libbabeltrace-ctf.so.1 \
		libpfm.so.4 libtraceevent.so.1 libbabeltrace.so.1; do
		mounts+=(-v "/usr/lib64/$so:/usr/lib64/$so:ro")
	done
	mounts+=(-v /usr/bin/perf:/usr/bin/perf:ro)
	perfout="/work/exp-cpython/data/perf/harness-arm$arm.txt"
	cmd="perf stat -e cycles,instructions,branches,branch-misses -o $perfout -- bash -c '$cmd'"
fi

echo "# run_harness_arm arm=$arm $(date -Is) cpus=$cpus steps=$steps"
echo "# cmd: $cmd"
sudo -n docker run --rm \
	--network none \
	--user "$(id -u):$(id -g)" \
	--security-opt seccomp=unconfined \
	--cpuset-cpus "$cpus" \
	-e HOME=/tmp \
	-e USER=REMOTE_USER \
	-e LOGNAME=REMOTE_USER \
	-e TORCHINDUCTOR_CACHE_DIR=/tmp/ti_cache \
	-e TRITON_CACHE_DIR=/tmp/triton_cache \
	-e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
	-e PI_HARNESS_ROOT=/work/harness \
	-e PYTHONPATH=/work/harness \
	-e PYTHONDONTWRITEBYTECODE=1 \
	-e PI_MODEL_TMP=/tmp/pi_models \
	"${mounts[@]}" \
	-w /work \
	"$img" \
	bash -c "$cmd"
echo "# rc=$? $(date -Is)"
