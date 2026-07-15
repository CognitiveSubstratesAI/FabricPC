# tools/warm_serve.jl — the ONE serve loop behind every warm session (bare / zygote / pcvb /
# reactant). Each session script loads its own packages, then calls `warm_serve(state_dir, banner)`.
#
# Protocol (unchanged): watch `<dir>/seq`; when it changes, run the code in `<dir>/in.jl`, capture
# stdout+stderr to `<dir>/out.txt`, then echo the sequence number into `<dir>/done`. The client
# (tools/warm_lib.sh) waits for `done == seq` before reading `out.txt`.
#
# WHY THIS FILE EXISTS: this loop was copy-pasted into four session scripts, so the same
# fail-open revise bug existed in four places and was fixed in one of them. One implementation,
# one place to fix. Same for the client side (tools/warm_lib.sh).

"""
    warm_serve(state_dir, banner)

Serve snippets forever out of `<repo>/<state_dir>`, revising between each.

Revising is OUR job here, and it must FAIL LOUD:

  * We call `Revise.revise()` explicitly because Revise's automatic revision is driven by the REPL
    backend before each prompt evaluation — this loop never reaches a prompt, so nothing would
    revise on its own.
  * `throw=true`, never a bare `catch; end`. A swallowed revision failure drains the revision queue
    and leaves the session running OLD code, so `Revise.revision_queue` reads 0 — indistinguishable
    from "nothing changed". The session then reports results for source you already fixed. That
    cost 8 false test failures on 2026-07-15 and got misdiagnosed as "warm sessions lie"; Revise was
    working correctly the whole time, and the `catch` block was eating the evidence. A loud stop
    beats a stale PASS.
"""
function warm_serve(state_dir::AbstractString, banner::AbstractString)
    root = abspath(joinpath(@__DIR__, ".."))
    dir = abspath(joinpath(root, state_dir))
    # The snippet is written here and `include`d, so it must sit at the REPO ROOT: that is what
    # makes a snippet's own `include("test/foo.jl")` resolve from the root. Gitignored.
    snippet_path = joinpath(root, ".warm_snippet.jl")
    mkpath(dir)
    inf = joinpath(dir, "in.jl")
    outf = joinpath(dir, "out.txt")
    seqf = joinpath(dir, "seq")
    donef = joinpath(dir, "done")

    rm(inf; force=true)
    rm(donef; force=true)
    # Publish the PID so nothing ever has to pattern-match a command line to find or kill this
    # session. `pkill -f 'julia.*warm_session_zygote.jl'` reliably matches the SHELL running that
    # very command (its argv contains the pattern's own text) and kills it — twice today, once
    # even with the [b]racket trick, because the same command line also named the script plainly
    # in a `nohup` line. `kill $(cat <dir>/pid)` cannot misfire that way.
    write(joinpath(dir, "pid"), string(getpid()))
    write(joinpath(dir, "ready"), "1")
    println(banner)
    flush(stdout)

    seen = ""
    while true
        if isfile(seqf) && isfile(inf)
            n = strip(read(seqf, String))
            if n != seen && !isempty(n)
                seen = n
                code = read(inf, String)
                if isdefined(Main, :Revise)
                    try
                        Base.invokelatest(Main.Revise.revise; throw=true)
                    catch e
                        open(outf, "w") do io
                            println(io, "!!! REVISE FAILED — session is STALE. Snippet NOT run.")
                            println(io, "!!! Fix the source error, or restart the session.")
                            showerror(io, e, catch_backtrace())
                            println(io)
                        end
                        write(donef, n)
                        continue
                    end
                end
                open(outf, "w") do io
                    redirect_stdout(io) do
                        redirect_stderr(io) do
                            try
                                # Run the snippet as a FILE at the repo root, not via
                                # `include_string`. `include`'s relative-path resolution follows the
                                # task-local source path, which under `include_string` stays at the
                                # running session script in tools/ — so a snippet saying
                                # `include("test/foo.jl")` went looking for tools/test/foo.jl.
                                # `Base.include` sets the source path to this file, so repo-relative
                                # paths resolve from the repo root, which is what a snippet author
                                # expects to write. (include_string's `filename` argument does NOT
                                # fix this — verified by execution, not assumed.)
                                write(snippet_path, code)
                                Base.include(Main, snippet_path)
                            catch e
                                showerror(io, e, catch_backtrace())
                                println(io)
                            end
                        end
                    end
                end
                write(donef, n)
            end
        end
        sleep(0.15)
    end
end
