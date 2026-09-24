#!/usr/bin/env bash
# build_arm.sh <A|B|C>
#
# Runs INSIDE the vllm-ascend container (the same image that ships
# /usr/local/python3.12.13) with the experiment tree mounted at $WORK.
# It configures + builds one CPython 3.12.13 arm in-tree (no `make install`).
#
#   arm A : --without-computed-gotos                                  (mirror-status-quo)
#   arm B : --with-computed-gotos                                     (dispatch only)
#   arm C : --enable-optimizations --with-lto --with-computed-gotos   (upper bound)
#
# Every arm also gets `--enable-shared` (uniformly), because the image python
# was built with `CONFIG_ARGS = '--enable-shared'` and nothing else.
#
# Artifacts written under $WORK/logs and $WORK/data.
set -uo pipefail

arm=${1:?usage: build_arm.sh <A|B|C>}
work=${WORK:-/work}
jobslot=${JOBS:-16}

case "$arm" in
	A) cfg="--without-computed-gotos" ;;
	B) cfg="--with-computed-gotos" ;;
	C) cfg="--enable-optimizations --with-lto --with-computed-gotos" ;;
	*) echo "unknown arm '$arm'" >&2; exit 2 ;;
esac

srcdir="$work/build-arm$arm/Python-3.12.13"
logdir="$work/logs"
datadir="$work/data"
mkdir -p "$logdir" "$datadir"

meta="$logdir/arm$arm.meta.txt"
{
	echo "# arm=$arm"
	echo "configure_extra='$cfg'"
	echo "configure_full='./configure --prefix=$work/prefix-arm$arm --enable-shared $cfg'"
	echo "make_jobs=$jobslot"
	echo "date_start=$(date -Is)"
	echo "hostname=$(hostname)"
	echo "os=$(grep -h PRETTY_NAME /etc/os-release)"
	echo "kernel=$(uname -sr)"
	echo "gcc=$(gcc --version | head -1)"
	echo "ld=$(ld --version | head -1)"
	echo "make=$(make --version | head -1)"
	echo "container_cpuset=$(cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || echo NA)"
	echo "container_affinity=$(taskset -pc $$ 2>&1 | head -1)"
	echo "image=$(cat /work/.image_ref 2>/dev/null || echo NA)"
} >"$meta"

cd "$srcdir" || exit 1

t0=$(date +%s)
echo "[arm$arm] configure: --enable-shared $cfg"
./configure --prefix="$work/prefix-arm$arm" --enable-shared $cfg >"$logdir/arm$arm.configure.log" 2>&1
rc=$?
t1=$(date +%s)
echo "configure_rc=$rc" >>"$meta"
echo "configure_seconds=$((t1 - t0))" >>"$meta"
if [ "$rc" -ne 0 ]; then
	echo "[arm$arm] CONFIGURE FAILED (rc=$rc)"
	tail -40 "$logdir/arm$arm.configure.log"
	exit 1
fi

# record what configure decided
grep -n "USE_COMPUTED_GOTOS\|HAVE_COMPUTED_GOTOS" pyconfig.h >>"$meta"
./python -VV >"$logdir/arm$arm.version.txt" 2>&1

echo "[arm$arm] make -j$jobslot"
make -j"$jobslot" >"$logdir/arm$arm.build.log" 2>&1
rc=$?
t2=$(date +%s)
echo "make_rc=$rc" >>"$meta"
echo "make_seconds=$((t2 - t1))" >>"$meta"
if [ "$rc" -ne 0 ]; then
	echo "[arm$arm] MAKE FAILED (rc=$rc)"
	tail -60 "$logdir/arm$arm.build.log"
	exit 1
fi

# ------------------------------------------------------------------ verify
# sysconfig snapshot of *our* build
LD_LIBRARY_PATH="$srcdir" ./python - "$datadir/arm$arm.sysconfig.json" <<'PY'
import json, sys, sysconfig
out = sys.argv[1]
keep = [
    "USE_COMPUTED_GOTOS", "HAVE_COMPUTED_GOTOS", "CONFIG_ARGS", "CC", "CFLAGS",
    "CFLAGS_NODIST", "OPT", "LDFLAGS", "LDFLAGS_NODIST", "PY_CFLAGS",
    "PY_CORE_CFLAGS", "Py_DEBUG", "Py_ENABLE_SHARED", "SOABI", "EXT_SUFFIX",
    "PGO_PROF_USE_FLAG", "PROFILE_TASK", "VERSION", "prefix", "exec_prefix",
    "LIBDIR", "LDLIBRARY", "INSTSONAME", "MULTIARCH", "HOST_GNU_TYPE",
]
snap = {
    "python_version": sys.version,
    "executable": sys.executable,
    "config_vars": {k: sysconfig.get_config_var(k) for k in keep},
    "makefile_keys": {},
}
mk = sysconfig.get_makefile_filename()
snap["makefile"] = mk
import re
try:
    txt = open(mk).read()
    for key in ("PROFILE_TASK", "PY_CFLAGS_NODIST", "LDFLAGS_NODIST", "CONFIGURE_LDFLAGS_NODIST"):
        m = re.search(rf"^{key}\s*=\s*(.*)$", txt, re.M)
        snap["makefile_keys"][key] = m.group(1).strip() if m else None
except OSError as exc:
    snap["makefile_keys"]["error"] = str(exc)
with open(out, "w") as fh:
    json.dump(snap, fh, indent=1, sort_keys=True)
print("USE_COMPUTED_GOTOS =", sysconfig.get_config_var("USE_COMPUTED_GOTOS"))
print("HAVE_COMPUTED_GOTOS =", sysconfig.get_config_var("HAVE_COMPUTED_GOTOS"))
PY

{
	echo "--- pyconfig.h ---"
	grep -n "COMPUTED_GOTOS" pyconfig.h
	echo "--- dispatch site (ceval.c) ---"
	grep -n "USE_COMPUTED_GOTOS" Python/ceval.c | head -8
	echo "--- binary ---"
	ls -la ./python libpython3.12.so.1.0 2>/dev/null
	echo "--- sha256 ---"
	sha256sum ./python libpython3.12.so.1.0 2>/dev/null
	echo "date_end=$(date -Is)"
} >>"$logdir/arm$arm.version.txt"

echo "[arm$arm] OK  configure=$((t1 - t0))s make=$((t2 - t1))s"
