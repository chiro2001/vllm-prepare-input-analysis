#!/usr/bin/env bash
# usage: nl.sh <file> <start> <end>
f="$1"; s="$2"; e="$3"
awk -v s="$s" -v e="$e" 'NR>=s && NR<=e {printf "%6d| %s\n", NR, $0}' "$f"
