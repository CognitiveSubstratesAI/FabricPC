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

const DIR = abspath(joinpath(@__DIR__, "..", ".warm_reactant"))
mkpath(DIR)
const INF = joinpath(DIR, "in.jl")
const OUTF = joinpath(DIR, "out.txt")
const SEQF = joinpath(DIR, "seq")
const DONE = joinpath(DIR, "done")
function _serve()
    rm(INF; force=true)
    rm(DONE; force=true)
    write(joinpath(DIR, "ready"), "1")
    println("WARM REACTANT SESSION READY (FabricPC+Reactant loaded)")
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
