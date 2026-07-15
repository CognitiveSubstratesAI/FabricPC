# tools/warm_session_pcvb.jl — PERSISTENT warm FabricPC+Lux+Zygote+HypothesisTests REPL,
# for the C-04 PC-vs-backprop comparison harness (docs/AUDIT_REGISTER.md), which needs
# Lux.jl/Optimisers.jl/HypothesisTests.jl loaded — deps the bare `--project=.` warm session
# (tools/warm_session.jl) does not have (and shouldn't: they don't belong in the package's
# own Project.toml, only this benchmark's own scoped environment,
# `benchmark/pc_vs_backprop/Project.toml`). Same protocol as warm_session.jl/
# warm_session_zygote.jl, rooted at `benchmark/pc_vs_backprop` (Manifest dev-paths FabricPC
# to the real repo root) and a SEPARATE state dir (`.warm_pcvb/`) so it can run alongside the
# other warm sessions without colliding on `.warm/seq`/`.warm_zygote/seq`.
#
# Start (background, once):  julia --project=benchmark/pc_vs_backprop tools/warm_session_pcvb.jl   # allow-cold-start: warm session startup
# Drive it:                  tools/warm_send_pcvb.sh <snippet.jl>   (or: echo 'CODE' | tools/warm_send_pcvb.sh)
try
    using Revise
catch
    @warn "Revise unavailable — src edits will NOT hot-reload"
end
using FabricPC, Random, Lux, Zygote, Optimisers, HypothesisTests, Statistics, Printf


# The serve loop (and its revise-or-fail-loud policy) is shared — see tools/warm_serve.jl.
include(joinpath(@__DIR__, "warm_serve.jl"))
warm_serve(".warm_pcvb", "WARM PCVB SESSION READY")
