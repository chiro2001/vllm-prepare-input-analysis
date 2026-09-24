#!/usr/bin/env bash
# A/B runner: does the prepare_input cost depend on the recorded *values*
# (record&replay needed) or only on the *shape* (shape-level synthesis enough)?
#
# Runs **inside** the no-card container (the only legal entry point is
# harness/scripts/pi-docker.sh, which mounts the repo at /work):
#
#   ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && \
#     bash harness/scripts/pi-docker.sh "bash harness/scripts/ab_run.sh --tag v1 --batch 8 --isl 2048"'
#
# When the real capture exists (produced by another agent with PI_CAPTURE set):
#
#   bash harness/scripts/ab_run.sh --tag v1real --real data/harness/trace_real.jsonl
#
# Everything printed/written stays inside the repo (/work == a3-22 checkout):
#   data/harness/trace_synth_<tag>.jsonl, data/harness/trace_synth_<tag>_manifest.json
#   data/harness/ab_<tag>.csv, data/harness/ab_<tag>.json
set -euo pipefail

# torch's getpass.getuser() needs a resolvable user inside the container
export USER="${USER:-pi}"
export LOGNAME="${LOGNAME:-$USER}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
HARNESS_ROOT="${PI_HARNESS_ROOT:-/work/harness}"
export PYTHONPATH="${HARNESS_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

TAG="v1"
SYNTH_ARGS=()
REAL=""
SHUFFLED=""
EXECUTOR="runner"
REPEATS=3
WARMUP=1
SHUFFLE_SEED=0
MAX_NUM_REQS=256
MAX_MODEL_LEN=8192
BLOCK_SIZE=128
CHUNK_SIZE=2048
RUN_SELFTEST=0
CROSSCHECK=0
CROSSCHECK_STEPS=0
SPEC_K=0
MODEL_PROFILE="qwen35-0.8b"
AB_RUNNER_REUSE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --real) REAL="$2"; shift 2;;
    --shuffled) SHUFFLED="$2"; shift 2;;
    --executor) EXECUTOR="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --shuffle-seed) SHUFFLE_SEED="$2"; shift 2;;
    --max-num-reqs) MAX_NUM_REQS="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --block-size) BLOCK_SIZE="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --selftest) RUN_SELFTEST=1; shift;;
    --crosscheck-runner) CROSSCHECK=1; shift;;
    --crosscheck-steps) CROSSCHECK_STEPS="$2"; shift 2;;
    --spec-k) SPEC_K="$2"; shift 2;;
    --model-profile) MODEL_PROFILE="$2"; shift 2;;
    --runner-reuse) AB_RUNNER_REUSE="$2"; shift 2;;
    *) SYNTH_ARGS+=("$1"); shift;;
  esac
done

cd /work
mkdir -p data/harness

SYNTH_TRACE="data/harness/trace_synth_${TAG}.jsonl"
SYNTH_MANIFEST="data/harness/trace_synth_${TAG}_manifest.json"
AB_CSV="data/harness/ab_${TAG}.csv"
AB_JSON="data/harness/ab_${TAG}.json"

if [[ "$RUN_SELFTEST" == "1" ]]; then
  echo "== self test (card free) =="
  python -m pi_harness.workload.selftest_local
fi

echo "== synthesise trace (tag=$TAG) =="
python -m pi_harness.workload.generate \
  --out "$SYNTH_TRACE" --manifest "$SYNTH_MANIFEST" \
  --block-size "$BLOCK_SIZE" --chunk-size "$CHUNK_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" --seed "$SHUFFLE_SEED" \
  --spec-k "$SPEC_K" \
  "${SYNTH_ARGS[@]}"

AB_ARGS=(
  --synth "$SYNTH_TRACE"
  --csv "$AB_CSV"
  --json "$AB_JSON"
  --tag "$TAG"
  --executor "$EXECUTOR"
  --repeats "$REPEATS"
  --warmup "$WARMUP"
  --shuffle-seed "$SHUFFLE_SEED"
  --max-num-reqs "$MAX_NUM_REQS"
  --max-model-len "$MAX_MODEL_LEN"
  --block-size "$BLOCK_SIZE"
  --chunk-size "$CHUNK_SIZE"
  --num-spec-tokens "$SPEC_K"
  --model-profile "$MODEL_PROFILE"
  --runner-reuse "$AB_RUNNER_REUSE"
)
if [[ -n "$REAL" ]]; then
  AB_ARGS+=(--real "$REAL")
fi
if [[ -n "$SHUFFLED" ]]; then
  AB_ARGS+=(--shuffled "$SHUFFLED")
fi
if [[ "$CROSSCHECK" == "1" ]]; then
  AB_ARGS+=(--crosscheck-runner --crosscheck-steps "$CROSSCHECK_STEPS")
fi

echo "== A/B run =="
python -m pi_harness.workload.ab "${AB_ARGS[@]}"

echo
echo "artifacts:"
ls -l "$SYNTH_TRACE" "$SYNTH_MANIFEST" "$AB_CSV" "$AB_JSON" 2>/dev/null || true
