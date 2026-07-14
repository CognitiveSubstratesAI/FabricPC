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

const DIR = abspath(joinpath(@__DIR__, "..", ".warm_zygote"))
mkpath(DIR)
const INF = joinpath(DIR, "in.jl")
const OUTF = joinpath(DIR, "out.txt")
const SEQF = joinpath(DIR, "seq")
const DONE = joinpath(DIR, "done")
function _serve()
    rm(INF; force=true)
    rm(DONE; force=true)
    write(joinpath(DIR, "ready"), "1")
    println("WARM ZYGOTE SESSION READY (FabricPC+Zygote+NPZ+Test loaded)")
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
