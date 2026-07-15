#!/bin/bash
# warm_send.sh — send a Julia snippet to the persistent warm FabricPC session and print its output.
# Usage:  tools/warm_send.sh snippet.jl     |     echo 'CODE' | tools/warm_send.sh
DIR="$(cd "$(dirname "$0")/.." && pwd)/.warm"
[ -f "$DIR/ready" ] || { echo "warm session not running — start: julia --project=. tools/warm_session.jl  # allow-cold-start: warm session startup"; exit 1; }
# Wait/report logic lives in tools/warm_lib.sh — this used to be a copy-pasted loop that gave up
# after a fixed ceiling and cat'd out.txt ANYWAY, printing the previous run's output as this one's.
# Usage: warm_send.sh [snippet.jl] [timeout_s]   (0 / omitted = wait as long as the job takes)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/tools/warm_lib.sh"
warm_send "$ROOT/.warm" "tools/warm_session.jl" "${1:-/dev/stdin}" "${2:-0}"
