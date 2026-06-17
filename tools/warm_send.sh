#!/bin/bash
# warm_send.sh — send a Julia snippet to the persistent warm FabricPC session and print its output.
# Usage:  tools/warm_send.sh snippet.jl     |     echo 'CODE' | tools/warm_send.sh
DIR="$(cd "$(dirname "$0")/.." && pwd)/.warm"
[ -f "$DIR/ready" ] || { echo "warm session not running — start: julia --project=. tools/warm_session.jl  # allow-cold-start: warm session startup"; exit 1; }
SEQ=$(( $(cat "$DIR/seq" 2>/dev/null || echo 0) + 1 ))
cat "${1:-/dev/stdin}" > "$DIR/in.jl"
echo "$SEQ" > "$DIR/seq"
for _ in $(seq 1 600); do [ "$(cat "$DIR/done" 2>/dev/null)" = "$SEQ" ] && break; sleep 0.2; done
cat "$DIR/out.txt"
