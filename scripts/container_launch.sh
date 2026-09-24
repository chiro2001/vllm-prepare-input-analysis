#!/bin/bash
# Payload executed *inside* the container.
#
# The LiteProfiler image was produced with `docker commit` from a container
# started with `--entrypoint bash`, so the image keeps
# `Entrypoint=[bash] Cmd=[-lc sleep infinity]` and no longer sources the CANN
# environment the way the baseline image does.  Passing `vllm serve ...`
# straight to `docker run` therefore makes bash try to execute the vllm
# console script as a shell script ("line 2: import: command not found").
#
# This script restores the baseline behaviour: source the CANN env, then exec
# the real command.  It is copied into the run directory (mounted at /runmeta)
# so the exact payload stays with the evidence.
#
# NOTE: no `set -e/-u` here.  `/usr/local/Ascend/nnal/atb/set_env.sh` reads
# `$ZSH_VERSION` unconditionally, so a nounset parent shell aborts during
# sourcing (observed: "line 43: ZSH_VERSION: unbound variable", exit 1).

for env_script in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
do
  if [[ -f $env_script ]]; then
    # shellcheck disable=SC1090
    source "$env_script" || echo "WARN: source $env_script returned non-zero" >&2
  else
    echo "WARN: $env_script missing" >&2
  fi
done

if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  # shellcheck disable=SC1090
  source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}" \
    || echo "WARN: source atb set_env.sh returned non-zero" >&2
fi

exec vllm "$@"
