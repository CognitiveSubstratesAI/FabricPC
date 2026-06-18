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
try; using Zygote; catch; @warn "Zygote unavailable"; end

const DIR  = abspath(joinpath(@__DIR__, "..", ".warm"))
mkpath(DIR)
const INF  = joinpath(DIR, "in.jl")
const OUTF = joinpath(DIR, "out.txt")
const SEQF = joinpath(DIR, "seq")
const DONE = joinpath(DIR, "done")
function _serve()
    rm(INF; force = true); rm(DONE; force = true)
    write(joinpath(DIR, "ready"), "1")
    println("WARM SESSION READY (FabricPC loaded)")
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
