# Regression test for the F-04 dual-AD-backend load guard's PRECOMPILE-VS-LOAD-TIME behavior
# (docs/AUDIT_REGISTER.md F-04, docs/decisions.md #19/#23; bug + fix found/made 2026-07-14).
#
# BUG (fixed in ext/FabricPCZygoteExt.jl / ext/FabricPCEnzymeExt.jl, same commit as this file):
# both extensions used to call `_register_ad_backend!(...)` (src/nodes/autodiff.jl, mutates the
# global `FabricPC._AD_BACKEND` Ref) as BARE TOP-LEVEL module code. Per Julia's package-extension
# semantics, a top-level statement that mutates a DIFFERENT, already-loaded module's global state
# only takes effect in the ephemeral worker process that PRECOMPILES the extension — it is never
# replayed when the extension is later loaded from its cached `.ji`, which is what happens in
# every normal (non-first-ever) session. The fix moves the call into `__init__()`, which Julia
# runs on EVERY load, cached or not.
#
# WHY THIS NEEDS FRESH SUBPROCESSES, NOT a `@testset` in the normal warm `runtests.jl` run: the
# bug is specifically about the gap between "first-ever compile" and "load from an existing
# cache". `runtests.jl` itself (and any already-running warm/Revise session) already has
# FabricPCZygoteExt loaded — by the time any @testset in that process runs, the damage (or lack
# thereof) already happened at `using Zygote` time, and Revise does not re-run `__init__`. Only a
# genuinely fresh `julia` process, pointed at an environment where the extensions are ALREADY
# precompiled from a prior, separate process, reproduces the exact scenario that was broken.
#
# SLOW / OPT-IN: spawns up to 5 fresh `julia` processes (instantiate + 2 clean single-backend
# precompiles + 2 assertions), each with real cold-start/precompile cost the first time this
# runs against an un-instantiated depot (subsequent runs are fast: package sources are shared
# with the rest of this repo's other Zygote/Enzyme environments via the global depot, so only
# these two extensions' own compile is ever really "cold"). Gated the same way as runtests.jl's
# `FABRICPC_TIER_D_TRANSFORMER` block: set `FABRICPC_AD_BACKEND_SUBPROCESS_TESTS=1` to include it.
#
# IMPORTANT, VERIFIED, RESIDUAL GAP (2026-07-14) — read before "fixing" this test to assert more:
# the `__init__()` fix makes `_register_ad_backend!` genuinely RUN on every load (confirmed below:
# the single-backend case now correctly reflects `_AD_BACKEND[]`, which is the concrete bug this
# session found and fixed). It does NOT, however, make the dual-backend guard behave the way
# F-04's own description implies ("raises immediately... if the other backend tries to load
# afterward"), for two independent reasons specific to Julia's package-extension machinery:
#   1. An error thrown from an EXTENSION's `__init__` is caught by Julia's own extension-loading
#      callback (`Base.run_extension_callbacks`) and reported via `@error` to stderr — it is NOT
#      re-thrown to the `using X` call site. `try using Enzyme catch ... end` around the second
#      `using` does NOT catch anything; the calling code sees a normal, non-throwing `using`.
#   2. A module's top-level method definitions (here, the extension's `_ad_param_grads`/
#      `_ad_latent_grads` methods) are installed as part of RESTORING the module, which happens
#      BEFORE `__init__` runs (see the stacktrace this test's own subprocess captures). So the
#      second-loaded extension's methods overwrite the first's in the shared dispatch table
#      REGARDLESS of whether its `__init__` subsequently errors — the exact silent-override F-04
#      exists to prevent still happens; the guard only adds a loud stderr log next to it, it does
#      not prevent it.
# Net effect confirmed by direct probe: after `using FabricPC; using Zygote; using Enzyme` (both
# pre-cached), `FabricPC._AD_BACKEND[]` stays `:zygote` (the guard's `error()` call fires before
# the `_AD_BACKEND[] = name` assignment it guards) while `methods(FabricPC._ad_param_grads)`
# simultaneously shows the ENZYME method installed — i.e. the Ref and the actual dispatch
# disagree about which backend is active. This test locks in that TRUE current behavior (not the
# originally-hoped-for "throws and prevents the override" behavior) so it can't silently regress
# further, and so a future genuine fix (e.g. routing dispatch through a Ref-held function pointer
# that the guard installs only after the conflict check passes, instead of competing top-level
# method definitions) has a precise, executable description of the gap it needs to close.

using Test

const _ENV_DIR = joinpath(@__DIR__, "dual_ad_backend_env")
const _JULIA_EXE = joinpath(Sys.BINDIR, Base.julia_exename())

# Run `code` in a genuinely fresh `julia` subprocess against `_ENV_DIR`, with no startup file
# (determinism) and no inherited flags from whatever process is running this test. Returns
# (exitcode, combined_stdout_and_stderr) — combined because the F-04 guard's failure mode (see
# header) reports via `@error` (stderr / Logging.jl), not a returned/thrown value, so a test that
# only looked at stdout would miss it entirely.
function _run_julia_capture(code::AbstractString)
    io = IOBuffer()
    cmd = `$_JULIA_EXE --project=$_ENV_DIR --startup-file=no -e $code`
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    return proc.exitcode, String(take!(io))
end

@testset "F-04 dual-AD-backend guard: precompile-vs-load-time regression (subprocess)" begin
    # --- setup: get FabricPC/Zygote/Enzyme resolvable, then precompile EACH extension from a
    # CLEAN single-backend load (no conflict), so the crux test below genuinely loads both from
    # an existing cache rather than accidentally exercising a fresh-compile code path instead.
    ec, out = _run_julia_capture("using Pkg; Pkg.instantiate()")
    @test ec == 0
    ec == 0 || println("[instantiate FAILED]\n", out)

    ec, out = _run_julia_capture("using FabricPC; using Zygote; print(FabricPC._AD_BACKEND[])")
    @test ec == 0
    @test strip(out) == "zygote" || occursin("zygote", out)

    ec, out = _run_julia_capture("using FabricPC; using Enzyme; print(FabricPC._AD_BACKEND[])")
    @test ec == 0
    @test strip(out) == "enzyme" || occursin("enzyme", out)

    # --- the actual bug: POSITIVE case. A fresh process loading ONLY Zygote (both extensions
    # already precompiled by setup above, so this genuinely loads FabricPCZygoteExt from its
    # cached .ji, not a fresh compile) must see `_AD_BACKEND[]` become `:zygote` IMMEDIATELY —
    # before the fix this stayed `:none` forever in exactly this scenario (confirmed by direct
    # probe prior to the fix landing), with zero indication anything was wrong.
    @testset "positive case: single backend registers correctly from a pre-cached load" begin
        code = """
        using FabricPC
        backend0 = FabricPC._AD_BACKEND[]
        using Zygote
        backend1 = FabricPC._AD_BACKEND[]
        println("BEFORE=", backend0, " AFTER=", backend1)
        """
        ec, out = _run_julia_capture(code)
        @test ec == 0
        @test occursin("BEFORE=none AFTER=zygote", out)
    end

    # --- the crux case: TRUE current behavior of the dual-backend guard (see header for why
    # this is what it asserts, not "using Enzyme throws").
    @testset "crux case: dual load — guard fires (logged) but does not prevent dispatch override" begin
        code = """
        using FabricPC
        using Zygote
        zyg_backend = FabricPC._AD_BACKEND[]
        threw = false
        try
            using Enzyme
        catch
            threw = true
        end
        final_backend = FabricPC._AD_BACKEND[]
        ms = methods(FabricPC._ad_param_grads)
        mods = sort(string.(getproperty.(collect(ms), :module)))
        println("ZYG_BACKEND=", zyg_backend)
        println("THREW=", threw)
        println("FINAL_BACKEND=", final_backend)
        println("METHOD_COUNT=", length(ms))
        println("METHOD_MODULES=", join(mods, ","))
        """
        ec, out = _run_julia_capture(code)
        @test ec == 0

        # 1. The guard genuinely fires: its exact error message reaches stderr (via Julia's
        #    extension-loading `@error`, captured here since stdout+stderr are combined).
        @test occursin(
            "both Zygote and Enzyme AD backends are loaded in this session", out
        )
        @test occursin("loaded zygote first, then enzyme", out)

        # 2. ...but NOT as a catchable exception at the `using Enzyme` call site.
        @test occursin("THREW=false", out)

        # 3. The Ref itself is left at the FIRST-registered backend (the guard's `error()` call
        #    fires before the `_AD_BACKEND[] = name` line it guards, so that assignment never
        #    executes) ...
        @test occursin("ZYG_BACKEND=zygote", out)
        @test occursin("FINAL_BACKEND=zygote", out)

        # 4. ...while dispatch itself was ALREADY silently overwritten to Enzyme's method by the
        #    time __init__ ran (method registration precedes __init__ during module load) — the
        #    Ref and the real seam dispatch DISAGREE. This is the residual gap: F-04's guard logs
        #    loudly but does not stop the hazard it was written to stop.
        @test occursin("METHOD_COUNT=2", out)
        @test occursin("FabricPCEnzymeExt", out)
    end
end
