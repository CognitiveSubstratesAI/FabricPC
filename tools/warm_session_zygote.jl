# tools/warm_session_zygote.jl — PERSISTENT warm FabricPC+Zygote+NPZ REPL, for conformance
# work (Tier A-D) that needs Zygote/NPZ loaded, which the bare `--project=.` warm session
# (tools/warm_session.jl) does not have as regular deps. Same protocol as warm_session.jl,
# but rooted at `.warm/zygote_env` (Manifest dev-paths FabricPC to the real repo root) and a
# SEPARATE state dir (`.warm_zygote/`) so it can run alongside the bare session without
# colliding on `.warm/seq`/`.warm/in.jl`/`.warm/done`.
#
# Start (background, once):  julia --project=.warm/zygote_env tools/warm_session_zygote.jl   # allow-cold-start: warm session startup
# Drive it:                  tools/warm_send_zygote.sh <snippet.jl>   (or: echo 'CODE' | tools/warm_send_zygote.sh)
try
    using Revise
catch
    @warn "Revise unavailable — src edits will NOT hot-reload"
end
using FabricPC, Random, Zygote, NPZ, Test


# The serve loop (and its revise-or-fail-loud policy) is shared — see tools/warm_serve.jl.
include(joinpath(@__DIR__, "warm_serve.jl"))
warm_serve(".warm_zygote", "WARM ZYGOTE SESSION READY (FabricPC+Zygote+NPZ+Test loaded)")
