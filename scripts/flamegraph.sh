#!/usr/bin/env bash
# flamegraph.sh - flamegraph-rs driver for the prepare_input analysis.
#
# Why this wrapper exists
# -----------------------
# flamegraph-rs (`cargo install flamegraph`, crate name `flamegraph`) is a Rust
# binary.  The dev machine (x86_64) and the target a3-22 (aarch64) are different
# architectures, so one binary cannot serve both roles.  The capture side must
# run on a3-22 (that is where the engine core thread and the PMU are) and needs
# root for `perf record`.
#
# Two topologies:
#
#   * offload (default, always works) - a3-22 only produces `perf script` text,
#     gzip'd to a few tens of KB.  We pull that and render locally with
#     `inferno-collapse-perf` + `inferno-flamegraph`.  Identical renderer for
#     the real-machine and the synthetic runs, which is what the fidelity
#     comparison needs.
#   * target-side - if an aarch64 `flamegraph`/`inferno` exists on a3-22 (see
#     docs/04-profiling-methodology.md) `record --on-target` renders there and
#     only the SVG travels back.
#
# Raw perf.data (~hundreds of MB) deliberately stays on a3-22 under
# data/profiles/<name>/ and is never uploaded.
#
# Usage
#   scripts/flamegraph.sh record --name NAME (--tid TID | --pid PID | --cpu N |
#                                --cmd 'CMD') [--seconds S] [--freq 999]
#                                [--call-graph fp|dwarf] [--title T]
#                                [--on-target] [--reverse] [--remote a3-22]
#   scripts/flamegraph.sh render --name NAME [--reverse] [--title T]
#   scripts/flamegraph.sh folded --name NAME [--reverse]      # folded only
#   scripts/flamegraph.sh diff --a NAME --b NAME [--title T]
#   scripts/flamegraph.sh selftest
set -uo pipefail

LOCAL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE=${REMOTE:-a3-22}
# The remote project root must be an *absolute* path: ssh commands `cd` into it
# (so a relative path gets re-based and the log redirects land nowhere), and
# scp never runs the path through a shell (so `$HOME/...` stays literal).
# Two earlier bugs came from exactly those two mistakes, hence the probe:
#   * relative $rdir after `cd` -> "perf-record.log: No such file or directory"
#   * `~` inside ${VAR:-~/...} -> bash expands it to the *local* $HOME, so scp
#     looked for /home/chiro/... on a3-22
REMOTE_HOME=${REMOTE_HOME:-}
if [[ -z $REMOTE_HOME ]]; then
	REMOTE_HOME=$(ssh -o ConnectTimeout=15 "$REMOTE" 'echo $HOME') \
		|| { printf '[flamegraph] ERROR: cannot read $HOME on %s\n' "$REMOTE" >&2; exit 1; }
fi
REMOTE_ROOT=${REMOTE_ROOT:-$REMOTE_HOME/projects/vllm/prepare-input-phase}
CARGO_BIN=${CARGO_BIN:-$HOME/.cargo/bin}
PROFILEDIR=data/profiles
FIGDIR=figures

log() { printf '[flamegraph] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
ssh_r() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

cmd_record() {
	local name="" tid="" pid="" cpu="" cmdline="" seconds=20 freq=999
	local cg="fp" title="" reverse=0 on_target=0
	while (($#)); do
		case "$1" in
		--name) name=$2; shift 2 ;;
		--tid) tid=$2; shift 2 ;;
		--pid) pid=$2; shift 2 ;;
		--cpu) cpu=$2; shift 2 ;;
		--cmd) cmdline=$2; shift 2 ;;
		--seconds) seconds=$2; shift 2 ;;
		--freq) freq=$2; shift 2 ;;
		--call-graph) cg=$2; shift 2 ;;
		--title) title=$2; shift 2 ;;
		--on-target) on_target=1; shift ;;
		--reverse) reverse=1; shift ;;
		--remote) REMOTE=$2; shift 2 ;;
		*) die "record: unknown argument $1" ;;
		esac
	done
	[[ -n $name ]] || die "record needs --name"

	local ntarget=0 v
	for v in "$tid" "$pid" "$cpu" "$cmdline"; do [[ -n $v ]] && ntarget=$((ntarget + 1)); done
	((ntarget == 1)) || die "record needs exactly one of --tid / --pid / --cpu / --cmd"

	local rdir="$REMOTE_ROOT/$PROFILEDIR/$name"
	ssh_r "mkdir -p $rdir" || die "cannot create $rdir on $REMOTE"

	local target part
	if [[ -n $tid ]]; then target="-t $tid"
	elif [[ -n $pid ]]; then target="-p $pid"
	elif [[ -n $cpu ]]; then target="-C $cpu"
	else target=""; fi

	local rec="sudo -n perf record -o $rdir/perf.data -F $freq -g --call-graph $cg $target"
	if [[ -n $cmdline ]]; then
		# `printf %q` keeps the user command intact through ssh + bash
		part=$(printf '%q' "timeout $seconds bash -lc $(printf '%q' "$cmdline")")
		rec="$rec -- bash -lc $part"
	else
		rec="$rec -- sleep $seconds"
	fi

	log "record on $REMOTE: name=$name freq=$freq cg=$cg target=${tid:-${pid:-${cpu:-cmd}}} dur=${seconds}s"
	ssh_r "$rec > $rdir/perf-record.log 2>&1; echo rc=\$?"
	ssh_r "sudo -n chown \$(id -u):\$(id -g) $rdir/perf.data 2>/dev/null; \
	       perf report --header-only -i $rdir/perf.data > $rdir/perf-header.txt 2>&1; \
	       perf report --stdio -i $rdir/perf.data --no-children --percent-limit 0.1 -g none \
	         > $rdir/perf-report.txt 2>&1; \
	       perf script -i $rdir/perf.data > $rdir/perf.script 2>/dev/null; \
	       gzip -f -9 $rdir/perf.script; \
	       ls -la $rdir" | head -20

	mkdir -p "$LOCAL_ROOT/$PROFILEDIR/$name" "$LOCAL_ROOT/$FIGDIR"
	local tdir="$REMOTE_ROOT/$PROFILEDIR/$name"
	scp -q "$REMOTE:$tdir/perf.script.gz" "$REMOTE:$tdir/perf-report.txt" \
		"$REMOTE:$tdir/perf-header.txt" "$LOCAL_ROOT/$PROFILEDIR/$name/" \
		|| die "scp of perf script failed"
	log "pulled perf.script.gz ($(du -h "$LOCAL_ROOT/$PROFILEDIR/$name/perf.script.gz" | cut -f1))"

	if ((on_target)); then
		log "rendering on target"
		ssh_r "set -e; export PATH=\$PATH:$REMOTE_ROOT/tools/bin:\$HOME/.cargo/bin; \
		       FGBIN=\$(command -v flamegraph || echo \$HOME/.cargo/bin/flamegraph); \
		       IC=\$(command -v inferno-collapse-perf || echo \$HOME/.cargo/bin/inferno-collapse-perf); \
		       IF=\$(command -v inferno-flamegraph || echo \$HOME/.cargo/bin/inferno-flamegraph); \
		       test -x \$FGBIN && test -x \$IC && test -x \$IF; \
		       gzip -dc $rdir/perf.script.gz | \$IC > $rdir/folded.txt; \
		       \$IF $([[ $reverse == 1 ]] && echo --reverse) --title '$name' \
		         < $rdir/folded.txt > $rdir/$name.svg; echo ON_TARGET_OK" \
			|| die "on-target render failed"
		scp -q "$REMOTE:$tdir/$name.svg" "$LOCAL_ROOT/$FIGDIR/$name.svg" && log "svg: $FIGDIR/$name.svg"
		return 0
	fi

	cmd_render --name "$name" --title "${title:-$name}" $([[ $reverse == 1 ]] && echo --reverse)
}

cmd_render() {
	local name="" title="" reverse=0
	while (($#)); do
		case "$1" in
		--name) name=$2; shift 2 ;;
		--title) title=$2; shift 2 ;;
		--reverse) reverse=1; shift ;;
		*) die "render: unknown argument $1" ;;
		esac
	done
	[[ -n $name ]] || die "render needs --name"
	local d="$LOCAL_ROOT/$PROFILEDIR/$name" kat=""
	[[ -s $d/perf.script.gz || -s $d/perf.script ]] || die "no perf.script(.gz) in $d"
	mkdir -p "$d" "$LOCAL_ROOT/$FIGDIR"
	[[ -x $CARGO_BIN/inferno-collapse-perf ]] || die "inferno-collapse-perf missing (cargo install inferno)"
	[[ -x $CARGO_BIN/inferno-flamegraph ]] || die "inferno-flamegraph missing (cargo install inferno)"
	((reverse)) && kat="--reverse"

	if [[ -s $d/perf.script.gz ]]; then
		gzip -dc "$d/perf.script.gz" | "$CARGO_BIN/inferno-collapse-perf" >"$d/folded.txt"
	else
		"$CARGO_BIN/inferno-collapse-perf" <"$d/perf.script" >"$d/folded.txt"
	fi
	local nstacks nsamples
	nstacks=$(grep -c . "$d/folded.txt" || true)
	nsamples=$(awk '{s+=$NF} END {print s+0}' "$d/folded.txt")
	log "folded: stacks=$nstacks samples=$nsamples -> $PROFILEDIR/$name/folded.txt"

	# shellcheck disable=SC2086
	"$CARGO_BIN/inferno-flamegraph" $kat --title "${title:-$name}" --colors hot \
		--width 1800 --hash --fontsize 12 <"$d/folded.txt" \
		>"$LOCAL_ROOT/$FIGDIR/$name.svg" || die "inferno-flamegraph failed"
	log "svg: $FIGDIR/$name.svg ($(du -h "$LOCAL_ROOT/$FIGDIR/$name.svg" | cut -f1))"
}

cmd_folded() {
	local name="" reverse=0
	while (($#)); do
		case "$1" in
		--name) name=$2; shift 2 ;;
		--reverse) reverse=1; shift ;;
		*) die "folded: unknown argument $1" ;;
		esac
	done
	cmd_render --name "$name" --folded-only 2>/dev/null
	local d="$LOCAL_ROOT/$PROFILEDIR/$name"
	[[ -s $d/folded.txt ]] || {
		if [[ -s $d/perf.script.gz ]]; then
			gzip -dc "$d/perf.script.gz" >"$d/perf.script"
		fi
		"$CARGO_BIN/inferno-collapse-perf" <"$d/perf.script" >"$d/folded.txt"
	}
	log "folded: $d/folded.txt ($(grep -c . "$d/folded.txt") stacks)"
}

cmd_diff() {
	local a="" b="" title=""
	while (($#)); do
		case "$1" in
		--a) a=$2; shift 2 ;;
		--b) b=$2; shift 2 ;;
		--title) title=$2; shift 2 ;;
		*) die "diff: unknown argument $1" ;;
		esac
	done
	[[ -n $a && -n $b ]] || die "diff needs --a NAME --b NAME"
	local fa="$LOCAL_ROOT/$PROFILEDIR/$a/folded.txt"
	local fb="$LOCAL_ROOT/$PROFILEDIR/$b/folded.txt"
	[[ -s $fa && -s $fb ]] || die "both folded.txt must exist (run 'folded' first)"
	local out="$LOCAL_ROOT/$FIGDIR/diff-$a-vs-$b.svg"
	[[ -x $CARGO_BIN/inferno-diff-folded ]] || die "inferno-diff-folded missing"
	"$CARGO_BIN/inferno-diff-folded" "$fa" "$fb" --title "${title:-$a vs $b}" \
		--colors rust --width 1800 --fontsize 12 >"$out" || die "inferno-diff-folded failed"
	log "diff svg: $FIGDIR/$(basename "$out")"
}

cmd_selftest() {
	local name="selftest-$(date -u +%H%M%S)"
	# forward --on-target / --reverse etc. to the record step
	local extra=("$@")
	ssh_r "cd $REMOTE_ROOT && mkdir -p $PROFILEDIR/$name && \
	       gcc -O2 -g -fno-omit-frame-pointer -o $REMOTE_ROOT/$PROFILEDIR/$name/probe_load \
	           $REMOTE_ROOT/scripts/synth/probe_load.c -lpthread -lm && echo BUILT" >/dev/null \
		|| die "cannot build synthetic victim on $REMOTE"
	cmd_record --name "$name" --seconds 4 \
		--cmd "(cd $REMOTE_ROOT/$PROFILEDIR/$name && ./probe_load --duration-ms 4000 --threads 2)" \
		--title "flamegraph.sh selftest" "${extra[@]+"${extra[@]}"}"
	printf '%s\n' "$name"
}

main() {
	local cmd=${1:-}
	[[ -n $cmd ]] || die "usage: flamegraph.sh {record|render|folded|diff|selftest} ..."
	shift
	case "$cmd" in
	record) cmd_record "$@" ;;
	render) cmd_render "$@" ;;
	folded) cmd_folded "$@" ;;
	diff) cmd_diff "$@" ;;
	selftest) cmd_selftest "$@" ;;
	*) die "unknown subcommand: $cmd" ;;
	esac
}

main "$@"
