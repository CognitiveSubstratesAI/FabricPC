# tools/warm_session.jl — PERSISTENT warm FabricPC REPL.
# Loads FabricPC + Revise ONCE, then evaluates snippets on demand: it watches `.warm/seq`, runs the
# code in `.warm/in.jl`, writes captured stdout/stderr to `.warm/out.txt`, and echoes the sequence
# number into `.warm/done`. Round-trips are warm (<1s); the heavy load happens only at startup.
# Revise hot-reloads src/ edits between snippets, so iterating on source needs NO reload.
#
# Start (background, once):  julia --project=. tools/warm_session.jl   # allow-cold-start: warm session startup
# Drive it:                  tools/warm_send.sh <snippet.jl>           (or:  echo 'CODE' | tools/warm_send.sh)
# Revise MUST load before FabricPC so it tracks FabricPC's src/ files for hot-reload.
try
    using Revise
catch
    @warn "Revise unavailable — src edits will NOT hot-reload"
end
using FabricPC, Random
try
    ;
    using Zygote;
catch
    ;
    @warn "Zygote unavailable";
end


# The serve loop (and its revise-or-fail-loud policy) is shared — see tools/warm_serve.jl.
include(joinpath(@__DIR__, "warm_serve.jl"))
warm_serve(".warm", "WARM SESSION READY (FabricPC loaded)")
