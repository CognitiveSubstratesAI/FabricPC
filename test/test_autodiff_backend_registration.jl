# Regression test for the F-04 dual-AD-backend guard (docs/AUDIT_REGISTER.md F-04,
# docs/decisions.md #15/#19/#23; bug found + two rounds of fix, 2026-07-14).
#
# ROUND 1 BUG (fixed in ext/FabricPCZygoteExt.jl / ext/FabricPCEnzymeExt.jl): both extensions
# called `_register_ad_backend!(...)` (src/nodes/autodiff.jl, mutates the global
# `FabricPC._AD_BACKEND` Ref) as BARE TOP-LEVEL module code. Per Julia's package-extension
# semantics, a top-level statement that mutates a DIFFERENT, already-loaded module's global
# state only takes effect in the ephemeral worker process that PRECOMPILES the extension — it
# is never replayed when the extension is later loaded from its cached `.ji`, which is what
# happens in every normal (non-first-ever) session. Fixed by moving the call into `__init__()`.
#
# ROUND 2 BUG (found verifying round 1's fix, fixed same day): the Symbol-Ref design from round
# 1 made the guard genuinely RUN on every load, but did not make it PREVENT anything — both
# extensions still defined the IDENTICAL `_ad_param_grads(::AbstractNode, ::NodeParams, inputs,
# z_latent)` signature, so the second-loaded extension's method definitions overwrote the
# first's in the shared dispatch table regardless of whether the guard's `error()` fired
# (module method definitions are installed while RESTORING the module, which precedes
# `__init__`), and `__init__` errors are swallowed by Julia's own extension-loading machinery
# (`Base.run_extension_callbacks`), never reaching the `using X` call site as a catchable
# exception. So the Ref could say "zygote" while actual dispatch silently ran Enzyme's method.
#
# ROUND 2 FIX (this test locks it in): BACKEND-TYPE DISPATCH. `ADBackend`/`NoBackend`/
# `ZygoteBackend`/`EnzymeBackend` (src/nodes/autodiff.jl) are marker types; each extension's
# real gradient method is keyed on a DISTINCT first-argument type
# (`_ad_param_grads(::ZygoteBackend, ...)` vs `_ad_param_grads(::EnzymeBackend, ...)`) — a
# structurally different signature, so there is no overwrite possible; both methods coexist
# harmlessly. `_AD_BACKEND[]` (an `ADBackend` instance) selects which one the unwrap-and-
# dispatch entry point (`_ad_param_grads(node, params, inputs, z_latent) ->
# _ad_param_grads(_AD_BACKEND[], node, params, inputs, z_latent)`) actually routes to.
# `_register_ad_backend!`'s conflict check still can't propagate its `error()` as a catchable
# exception (same Julia `__init__`-swallowing behavior as round 1) — but now that doesn't
# matter: the `_AD_BACKEND[] = backend` assignment sits AFTER the `error()` call in the same
# function body, so on conflict the error throws (swallowed, logged) BEFORE the Ref is
# touched — the Ref, and therefore real dispatch, stays on the FIRST-loaded backend. This is
# the actual fix: protected by control flow/state, not by exception propagation reaching the
# caller.
#
# WHY THIS NEEDS FRESH SUBPROCESSES, NOT a `@testset` in the normal warm `runtests.jl` run: the
# round-1 bug is specifically about the gap between "first-ever compile" and "load from an
# existing cache" — `runtests.jl` itself (and any already-running warm/Revise session) already
# has FabricPCZygoteExt loaded, so by the time any @testset in that process runs, the damage
# (or lack thereof) already happened at `using Zygote` time, and Revise does not re-run
# `__init__`. Only a genuinely fresh `julia` process, pointed at an environment where the
# extensions are ALREADY precompiled from a prior, separate process, reproduces the scenario.
#
# SLOW / OPT-IN: spawns up to 5 fresh `julia` processes. Gated the same way as runtests.jl's
# `FABRICPC_TIER_D_TRANSFORMER` block: set `FABRICPC_AD_BACKEND_SUBPROCESS_TESTS=1` to include.

using Test

const _ENV_DIR = joinpath(@__DIR__, "dual_ad_backend_env")
const _JULIA_EXE = joinpath(Sys.BINDIR, Base.julia_exename())

# Run `code` in a genuinely fresh `julia` subprocess against `_ENV_DIR`, with no startup file
# (determinism) and no inherited flags. Returns (exitcode, combined_stdout_and_stderr) —
# combined because the F-04 guard's failure mode reports via `@error` (stderr / Logging.jl),
# not a returned/thrown value, so a test that only looked at stdout would miss it entirely.
function _run_julia_capture(code::AbstractString)
    io = IOBuffer()
    cmd = `$_JULIA_EXE --project=$_ENV_DIR --startup-file=no -e $code`
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    return proc.exitcode, String(take!(io))
end

@testset "F-04 dual-AD-backend guard: precompile-vs-load-time + dispatch-safety regression" begin
    # --- setup: get FabricPC/Zygote/Enzyme resolvable, then precompile EACH extension from a
    # CLEAN single-backend load (no conflict), so the crux test below genuinely loads both from
    # an existing cache rather than accidentally exercising a fresh-compile code path instead.
    ec, out = _run_julia_capture("using Pkg; Pkg.instantiate()")
    @test ec == 0
    ec == 0 || println("[instantiate FAILED]\n", out)

    ec, out = _run_julia_capture(
        "using FabricPC; using Zygote; print(nameof(typeof(FabricPC._AD_BACKEND[])))"
    )
    @test ec == 0
    @test occursin("ZygoteBackend", out)

    ec, out = _run_julia_capture(
        "using FabricPC; using Enzyme; print(nameof(typeof(FabricPC._AD_BACKEND[])))"
    )
    @test ec == 0
    @test occursin("EnzymeBackend", out)

    # --- ROUND 1 bug: POSITIVE case. A fresh process loading ONLY Zygote (both extensions
    # already precompiled by setup above, so this genuinely loads FabricPCZygoteExt from its
    # cached .ji, not a fresh compile) must see `_AD_BACKEND[]` become `ZygoteBackend`
    # IMMEDIATELY — before the round-1 fix this stayed `NoBackend` forever in exactly this
    # scenario, with zero indication anything was wrong.
    @testset "positive case: single backend registers correctly from a pre-cached load" begin
        code = """
        using FabricPC
        backend0 = nameof(typeof(FabricPC._AD_BACKEND[]))
        using Zygote
        backend1 = nameof(typeof(FabricPC._AD_BACKEND[]))
        println("BEFORE=", backend0, " AFTER=", backend1)
        """
        ec, out = _run_julia_capture(code)
        @test ec == 0
        @test occursin("BEFORE=NoBackend AFTER=ZygoteBackend", out)
    end

    # --- ROUND 2 crux case: dual load. The guard's error still can't propagate as a catchable
    # exception (unchanged Julia limitation) — but now dispatch genuinely stays correct anyway,
    # because the Ref assignment (not just a log message) is what's gated by the conflict check.
    @testset "crux case: dual load — dispatch genuinely stays on the first-loaded backend" begin
        code = """
        using FabricPC
        using Zygote
        zyg_backend = nameof(typeof(FabricPC._AD_BACKEND[]))
        threw = false
        try
            using Enzyme
        catch
            threw = true
        end
        final_backend = nameof(typeof(FabricPC._AD_BACKEND[]))
        ms = methods(FabricPC._ad_param_grads)
        mods = sort(unique(string.(getproperty.(collect(ms), :module))))

        # THE ACTUAL SAFETY CLAIM, not just Ref bookkeeping: call the real seam entry point
        # (unwrap-and-dispatch) on a trivial Linear-shaped NodeParams/inputs/z_latent and check
        # the RESULT reflects Zygote's method having run, not Enzyme's. Zygote's `_ad_param_grads`
        # returns `NodeParams` built via `SoA(_fill_zero(...))`; Enzyme's returns `NodeParams`
        # built via `SoA(dwnt)` directly from `Enzyme.autodiff`'s Duplicated shadow -- both
        # produce a NodeParams of the same shape or this graph wouldn't run at all either way,
        # so shape alone can't distinguish them. Instead: dispatch on the TYPE actually selected
        # is the real, direct claim -- confirm the method Julia actually calls for this generic
        # signature is the ZygoteBackend one, via `which`. NOTE: `which(f, types)` requires
        # `types` to be a tuple that is itself a SUBTYPE of some method's signature -- `Any` in
        # arg positions 2/3 is BROADER than the extension methods' `::AbstractNode`/`::NodeParams`
        # constraints (Any is not <: AbstractNode), so it does not match and `which` throws
        # MethodError. Use concrete types that genuinely satisfy those constraints instead
        # (`FabricPC.Linear <: AbstractNode`, `FabricPC.NodeParams` itself) -- verified directly:
        # this resolves to FabricPCZygoteExt when `_AD_BACKEND[]` is ZygoteBackend, and (sanity
        # checked separately) to FabricPCEnzymeExt when forced to EnzymeBackend, confirming `which`
        # is actually discriminating on backend type and not just returning a fallback.
        m = which(
            FabricPC._ad_param_grads,
            (typeof(FabricPC._AD_BACKEND[]), FabricPC.Linear, FabricPC.NodeParams, Any, Any),
        )
        println("ZYG_BACKEND=", zyg_backend)
        println("THREW=", threw)
        println("FINAL_BACKEND=", final_backend)
        println("METHOD_COUNT=", length(ms))
        println("METHOD_MODULES=", join(mods, ","))
        println("DISPATCHED_MODULE=", m.module)
        """
        ec, out = _run_julia_capture(code)
        @test ec == 0

        # 1. The guard genuinely fires: its exact error message reaches stderr (via Julia's
        #    extension-loading `@error`, captured here since stdout+stderr are combined).
        @test occursin(
            "both Zygote and Enzyme AD backends are loaded in this session", out
        )
        @test occursin("loaded Zygote first", out)

        # 2. ...but NOT as a catchable exception at the `using Enzyme` call site (unchanged
        #    Julia limitation -- this is not something round 2 was meant to fix).
        @test occursin("THREW=false", out)

        # 3. The Ref is left at the FIRST-registered backend (the guard's `error()` call fires
        #    before the `_AD_BACKEND[] = backend` line it guards, so that assignment never
        #    executes) — same bookkeeping claim round 1's test already made.
        @test occursin("ZYG_BACKEND=ZygoteBackend", out)
        @test occursin("FINAL_BACKEND=ZygoteBackend", out)

        # 4. THE ROUND-2 FIX ITSELF: both backends' methods coexist (no overwrite -- 4 methods:
        #    the unwrap dispatcher, the NoBackend fallback, and one typed method per extension),
        #    and — the actual safety claim — the method Julia dispatches to for the CURRENTLY
        #    registered backend type is genuinely FabricPCZygoteExt's, not FabricPCEnzymeExt's.
        #    Before round 2, this would have shown FabricPCEnzymeExt (METHOD_COUNT=2, dispatch
        #    silently overwritten) even though the Ref itself said "zygote".
        @test occursin("METHOD_COUNT=4", out)
        @test occursin("FabricPCEnzymeExt", out)  # Enzyme's method DOES exist (coexists, harmlessly)
        @test occursin("FabricPCZygoteExt", out)  # ...and so does Zygote's
        @test occursin("DISPATCHED_MODULE=FabricPCZygoteExt", out)  # and it's the one that RUNS
    end
end
