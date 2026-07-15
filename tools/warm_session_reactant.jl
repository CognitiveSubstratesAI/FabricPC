# tools/warm_session_reactant.jl — PERSISTENT warm FabricPC+Reactant REPL, for the J-04
# compiled-PC-inference-vs-jax.jit benchmark (docs/AUDIT_REGISTER.md section 5). Same protocol
# as tools/warm_session.jl / tools/warm_session_zygote.jl, but rooted at `benchmark/jit`
# (Manifest dev-paths FabricPC to the real repo root, has Reactant+Enzyme+Revise) and a
# SEPARATE state dir (`.warm_reactant/`) so it can run alongside the other warm sessions
# without colliding on `.warm*/seq`/`in.jl`/`done`.
#
# Start (background, once):
#   julia --project=benchmark/jit tools/warm_session_reactant.jl   # allow-cold-start: warm session startup, avoids repeated Reactant cold-compiles per this session's iteration
# Drive it:  tools/warm_send_reactant.sh <snippet.jl>   (or: echo 'CODE' | tools/warm_send_reactant.sh)
try
    using Revise
catch
    @warn "Revise unavailable — src edits will NOT hot-reload"
end
using FabricPC, Random, Printf
using FabricPC: run_inference, initialize_graph_state, compile_inference
using Reactant   # triggers FabricPCReactantExt


# The serve loop (and its revise-or-fail-loud policy) is shared — see tools/warm_serve.jl.
include(joinpath(@__DIR__, "warm_serve.jl"))
warm_serve(".warm_reactant", "WARM REACTANT SESSION READY (FabricPC+Reactant loaded)")
