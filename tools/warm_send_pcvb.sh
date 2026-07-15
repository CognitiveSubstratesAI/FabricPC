#!/bin/bash
# warm_send_pcvb.sh — send a Julia snippet to the persistent warm FabricPC+Lux+Zygote+
# HypothesisTests session (tools/warm_session_pcvb.jl) and print its output.
# Usage: tools/warm_send_pcvb.sh snippet.jl (or: echo 'CODE' | tools/warm_send_pcvb.sh)
DIR="$(cd "$(dirname "$0")/.." && pwd)/.warm_pcvb"
[ -f "$DIR/ready" ] || { echo "warm pcvb session not running — start: julia --project=benchmark/pc_vs_backprop tools/warm_session_pcvb.jl  # allow-cold-start: warm session startup"; exit 1; }
# Wait/report logic lives in tools/warm_lib.sh — this used to be a copy-pasted loop that gave up
# after a fixed ceiling and cat'd out.txt ANYWAY, printing the previous run's output as this one's.
# Usage: warm_send_pcvb.sh [snippet.jl] [timeout_s]   (0 / omitted = wait as long as the job takes)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/tools/warm_lib.sh"
warm_send "$ROOT/.warm_pcvb" "tools/warm_session_pcvb.jl" "${1:-/dev/stdin}" "${2:-0}"
