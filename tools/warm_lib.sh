# tools/warm_lib.sh — the ONE send/wait implementation behind every warm_send_*.sh.
# Source it, then call: warm_send <state_dir> <session_script> [snippet] [timeout_s]
#
# WHY THIS FILE EXISTS: this logic was copy-pasted into four sender scripts, each with its own
# arbitrary ceiling (120s / 240s / 180s), and every copy FAILED OPEN — it gave up silently and
# cat'd out.txt anyway, so any job outliving the ceiling printed the PREVIOUS snippet's output as
# if it were this run's result. A stale PASS is worse than no answer. The conformance suite runs
# ~19 min warm, so "outlives the ceiling" is the normal case, not the exotic one.

# warm_send <state_dir> <session_script> [snippet_file|-] [timeout_s]
#   Waits as long as the job takes by default (timeout 0). Exits 0 ONLY when the output printed is
#   genuinely this snippet's. Non-zero and loud otherwise: 1 = no session, 2 = session died,
#   3 = timed out (job still running — out.txt is NOT yet a result).
warm_send() {
    local dir="$1" session="$2" snippet="${3:-/dev/stdin}" timeout="${4:-0}"
    if [ ! -f "$dir/ready" ]; then
        echo "ERROR: no warm session at $dir — start it with:" >&2
        echo "         julia --project=<env> $session   # allow-cold-start: warm session startup" >&2
        return 1
    fi
    local seq
    seq=$(( $(cat "$dir/seq" 2>/dev/null || echo 0) + 1 ))
    cat "$snippet" > "$dir/in.jl"
    : > "$dir/out.txt"          # no previous run's output can masquerade as this one's
    echo "$seq" > "$dir/seq"

    local t=0
    while [ "$(cat "$dir/done" 2>/dev/null)" != "$seq" ]; do
        # Liveness by PID (written by warm_serve), never by command-line matching: a `pgrep -f`
        # pattern also matches any shell whose argv happens to mention the session script — which
        # both hides a dead session and, in `pkill` form, kills the caller.
        if ! kill -0 "$(cat "$dir/pid" 2>/dev/null || echo 0)" 2> /dev/null; then
            echo "ERROR: the warm session died while running seq=$seq (OOM?). See $dir/session.log." >&2
            echo "       Partial output follows — it is INCOMPLETE, not a result:" >&2
            cat "$dir/out.txt"
            return 2
        fi
        sleep 1
        t=$((t + 1))
        if [ "$timeout" -gt 0 ] && [ "$t" -ge "$timeout" ]; then
            echo "ERROR: gave up after ${timeout}s — the session is STILL RUNNING seq=$seq." >&2
            echo "       Do NOT read out.txt as a result yet. Re-check with:" >&2
            echo "         [ \"\$(cat $dir/done)\" = $seq ] && cat $dir/out.txt" >&2
            return 3
        fi
    done
    cat "$dir/out.txt"
}
