#!/bin/bash
# warm_send_zygote.sh — send a Julia snippet to the persistent warm FabricPC+Zygote+NPZ session
# (tools/warm_session_zygote.jl) and print its output. Usage: tools/warm_send_zygote.sh snippet.jl
# (or: echo 'CODE' | tools/warm_send_zygote.sh)
DIR="$(cd "$(dirname "$0")/.." && pwd)/.warm_zygote"
[ -f "$DIR/ready" ] || { echo "warm zygote session not running — start: julia --project=.warm/zygote_env tools/warm_session_zygote.jl  # allow-cold-start: warm session startup"; exit 1; }
SEQ=$(( $(cat "$DIR/seq" 2>/dev/null || echo 0) + 1 ))
cat "${1:-/dev/stdin}" > "$DIR/in.jl"
echo "$SEQ" > "$DIR/seq"
for _ in $(seq 1 600); do [ "$(cat "$DIR/done" 2>/dev/null)" = "$SEQ" ] && break; sleep 0.2; done
cat "$DIR/out.txt"
