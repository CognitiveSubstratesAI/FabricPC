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

const DIR = abspath(joinpath(@__DIR__, "..", ".warm_pcvb"))
mkpath(DIR)
const INF = joinpath(DIR, "in.jl")
const OUTF = joinpath(DIR, "out.txt")
const SEQF = joinpath(DIR, "seq")
const DONE = joinpath(DIR, "done")
function _serve()
    rm(INF; force=true)
    rm(DONE; force=true)
    write(joinpath(DIR, "ready"), "1")
    println("WARM PC-VS-BACKPROP SESSION READY (FabricPC+Lux+Zygote+Optimisers+HypothesisTests loaded)")
    seen = ""
    while true
        if isfile(SEQF) && isfile(INF)
            n = strip(read(SEQF, String))
            if n != seen && !isempty(n)
                seen = n
                code = read(INF, String)
                try
                    isdefined(Main, :Revise) && Base.invokelatest(Main.Revise.revise)
                catch
                end
                open(OUTF, "w") do io
                    redirect_stdout(io) do
                        redirect_stderr(io) do
                            try
                                Base.include_string(Main, code)
                            catch e
                                showerror(io, e, catch_backtrace())
                                println(io)
                            end
                        end
                    end
                end
                write(DONE, n)
            end
        end
        sleep(0.15)
    end
end

_serve()
