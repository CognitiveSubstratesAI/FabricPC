#!/bin/bash
# warm_send_pcvb.sh — send a Julia snippet to the persistent warm FabricPC+Lux+Zygote+
# HypothesisTests session (tools/warm_session_pcvb.jl) and print its output.
# Usage: tools/warm_send_pcvb.sh snippet.jl (or: echo 'CODE' | tools/warm_send_pcvb.sh)
DIR="$(cd "$(dirname "$0")/.." && pwd)/.warm_pcvb"
[ -f "$DIR/ready" ] || { echo "warm pcvb session not running — start: julia --project=benchmark/pc_vs_backprop tools/warm_session_pcvb.jl  # allow-cold-start: warm session startup"; exit 1; }
SEQ=$(( $(cat "$DIR/seq" 2>/dev/null || echo 0) + 1 ))
cat "${1:-/dev/stdin}" > "$DIR/in.jl"
echo "$SEQ" > "$DIR/seq"
for _ in $(seq 1 1200); do [ "$(cat "$DIR/done" 2>/dev/null)" = "$SEQ" ] && break; sleep 0.2; done
cat "$DIR/out.txt"
