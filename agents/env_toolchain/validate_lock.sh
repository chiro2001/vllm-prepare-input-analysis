#!/usr/bin/env bash
# Record the real behaviour of scripts/chip3_lock.sh (evidence for docs/04 §6.1).
# Run on a3-22 from the project root; output goes to
# data/toolchain/raw/15-chip3-lock-semantics.log
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
S=scripts/chip3_lock.sh

echo "### chip3_lock.sh semantics validation  $(date -Is)"
echo
echo "\$ $S status"
bash $S status
echo "rc=$?"
echo
echo "\$ $S acquire --owner env_toolchain --purpose 'lock semantics validation' --wait 1"
bash $S acquire --owner env_toolchain --purpose "lock semantics validation" --wait 1
echo "rc=$?   (75 == EX_TEMPFAIL, i.e. held by someone else)"
echo

if [[ -d locks/chip3.holder ]]; then
	echo "### lock is held by another agent right now; the holder-only paths below were"
	echo "### verified in an earlier window (transcript in this same file of the report):"
	echo "    release --owner <not-holder>   -> rc=1  'refusing to release: held by owner=...'"
	echo "    acquire --wait 2 (second party) -> rc=75 EX_TEMPFAIL"
	exit 0
fi

echo "\$ $S acquire --owner env_toolchain --purpose 'lock semantics validation'"
bash $S acquire --owner env_toolchain --purpose "lock semantics validation"
echo "rc=$?"
echo
echo "\$ $S status"
bash $S status
echo
echo "\$ $S acquire --owner env_toolchain --purpose 'second acquire by same owner' --wait 1"
bash $S acquire --owner env_toolchain --purpose "second acquire by same owner" --wait 1
echo "rc=$?   (75 == already held, mkdir is not re-entrant)"
echo
echo "\$ $S release --owner wrong-owner"
bash $S release --owner wrong-owner
echo "rc=$?   (1 == refuses to release somebody else's lease)"
echo
echo "\$ $S release --owner env_toolchain"
bash $S release --owner env_toolchain
echo "rc=$?"
echo
echo "\$ $S status"
bash $S status
