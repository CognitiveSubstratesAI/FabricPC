#!/bin/bash
# warm_send_zygote.sh — send a Julia snippet to the persistent warm FabricPC+Zygote+NPZ session
# (tools/warm_session_zygote.jl) and print its output. Usage: tools/warm_send_zygote.sh snippet.jl
# (or: echo 'CODE' | tools/warm_send_zygote.sh)
# Usage: tools/warm_send_zygote.sh [snippet.jl] [timeout_s]   (0 / omitted = wait as long as the
# job takes; the conformance suite runs ~19 min warm). Wait/report logic lives in tools/warm_lib.sh.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/tools/warm_lib.sh"
warm_send "$ROOT/.warm_zygote" "tools/warm_session_zygote.jl" "${1:-/dev/stdin}" "${2:-0}"
