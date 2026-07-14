#!/bin/bash
# warm_send_reactant.sh — send a Julia snippet to the persistent warm FabricPC+Reactant session
# (tools/warm_session_reactant.jl) and print its output.
# Usage:  tools/warm_send_reactant.sh snippet.jl   |   echo 'CODE' | tools/warm_send_reactant.sh
DIR="$(cd "$(dirname "$0")/.." && pwd)/.warm_reactant"
[ -f "$DIR/ready" ] || { echo "warm reactant session not running — start: julia --project=benchmark/jit tools/warm_session_reactant.jl  # allow-cold-start: warm session startup"; exit 1; }
SEQ=$(( $(cat "$DIR/seq" 2>/dev/null || echo 0) + 1 ))
cat "${1:-/dev/stdin}" > "$DIR/in.jl"
echo "$SEQ" > "$DIR/seq"
for _ in $(seq 1 900); do [ "$(cat "$DIR/done" 2>/dev/null)" = "$SEQ" ] && break; sleep 0.2; done
cat "$DIR/out.txt"
