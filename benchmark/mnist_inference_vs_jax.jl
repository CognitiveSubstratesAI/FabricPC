#!/usr/bin/env julia
# Compiled PC INFERENCE (relaxation loop only, no weight-gradient training) vs upstream's own
# jax.jit -- Julia side. Companion to benchmark/mnist_inference_vs_jax.py (run THAT first --
# it generates the shared weights/batch via jax.random + KaimingInitializer, RNG-trap style,
# dumps them as raw Float32 .bin files + a shapes.txt sidecar, and produces jax_result.json +
# the converged jax.jit z_latents for the numerics cross-check below).
#
# SCOPE (docs/AUDIT_REGISTER.md section 5 J-01/J-02/J-04, docs/decisions.md section 22): this
# benchmarks Layer 2 (Reactant/XLA compiled `compile_inference`, the E-step latent relaxation
# loop only) vs Layer 0 (eager Dict `run_inference`) IN-LANGUAGE, and cross-checks the Layer 2
# result against upstream's jax.jit(run_inference) result dumped by the .py companion. This is
# NOT the J-01/J-02 weight-gradient-training compile (still blocked) -- inference only, params
# fixed throughout.
#
# Run (needs Reactant -- an env with FabricPC dev-linked + Reactant, e.g. benchmark/jit):
#   julia --project=benchmark/jit benchmark/mnist_inference_vs_jax.jl [data_dir]
# (or drive it through a warm Reactant session -- see tools/warm_session_reactant.jl -- to
# avoid repeated multi-minute Reactant cold-compiles while iterating).

using FabricPC
using FabricPC: run_inference, initialize_graph_state, compile_inference
using Random, Printf
using Reactant   # triggers FabricPCReactantExt

const BDIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "mnist_inference_vs_jax_data")

function load_shapes(dir)
    d = Dict{String, Vector{Int}}()
    for line in eachline(joinpath(dir, "shapes.txt"))
        parts = split(line)
        d[parts[1]] = parse.(Int, parts[2:end])
    end
    return d
end

# Data was dumped by numpy's `ndarray.tofile()`, which ALWAYS writes C-order (row-major)
# bytes regardless of the source array's memory layout (see mnist_inference_vs_jax.py's
# dump() docstring -- an earlier version of this pair assumed `asfortranarray(...).tofile()`
# produced column-major bytes for a cheap `reshape`; it did not, silently scrambling every
# non-1xN array on read). Reconstruct correctly: reshape into the REVERSED shape (row-major
# bytes read naturally as column-major with axes reversed), then permute the axes back.
function load_arr(dir, name, shapes)
    shp = Tuple(shapes[name])
    bytes = read(joinpath(dir, "$name.bin"))
    data = reinterpret(Float32, bytes)
    nd = length(shp)
    return permutedims(reshape(collect(data), reverse(shp)), nd:-1:1)
end

function bench(f, n)
    for _ in 1:5
        f()
    end
    best = Inf
    for _ in 1:n
        best = min(best, @elapsed f())
    end
    return best
end

function main()
    shapes = load_shapes(BDIR)
    Wxh = load_arr(BDIR, "Wxh", shapes)
    bh = load_arr(BDIR, "bh", shapes)
    Why = load_arr(BDIR, "Why", shapes)
    by = load_arr(BDIR, "by", shapes)
    Xb = load_arr(BDIR, "Xb", shapes)
    Yb = load_arr(BDIR, "Yb", shapes)
    jax_jit_x = load_arr(BDIR, "jax_jit_x_z_latent", shapes)
    jax_jit_h = load_arr(BDIR, "jax_jit_h_z_latent", shapes)
    jax_jit_y = load_arr(BDIR, "jax_jit_y_z_latent", shapes)

    B, DX, DH, DY = 256, 784, 128, 10
    edge_xh = "x->h:in"
    edge_hy = "h->y:in"

    # Same NODE-CLASS CHOICE as scripts/generate_tier_d_mnist_fixtures.py / test_tier_d_mnist.jl:
    # Julia's fused `Linear` for all three nodes (== upstream's plain Linear for "x",
    # LinearExplicitGrad for "h"/"y", per Tier B/C/D).
    x = Linear((DX,), "x"; use_bias=false)
    h = Linear((DH,), "h"; activation=TanhActivation())
    y = Linear((DY,), "y"; energy=GaussianEnergy())

    inference = InferenceSGD(eta_infer=0.1, infer_steps=20, latent_decay=0.0)
    structure = graph([x, h, y], [Edge(x, h), Edge(h, y)], TaskMap(; x=x, y=y), inference)

    # RNG TRAP: GraphParams/clamps built DIRECTLY from the .py-generated arrays, never via
    # Julia's own initialize_params/RNG.
    params = GraphParams(
        Dict(
            "x" => NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            ),
            "h" => NodeParams(Dict(edge_xh => Wxh), Dict("b" => bh)),
            "y" => NodeParams(Dict(edge_hy => Why), Dict("b" => by))
        )
    )
    clamps = Dict{String, Any}("x" => Xb, "y" => Yb)
    rng = MersenneTwister(0)
    init = initialize_graph_state(structure, B, rng; clamps=clamps, params=params)

    # Layer 0 (eager Dict) -- correctness oracle + in-language baseline.
    eager = run_inference(params, init, clamps, structure)

    # Layer 2 (Reactant/XLA compiled) -- compile time measured separately from execution.
    println("compiling inference with Reactant/XLA ...")
    t_compile = @elapsed (ci = compile_inference(structure, params, clamps; batch=B))
    zl = ci(init)  # first post-compile execution (fast path: compile-time params reused)

    names = ci.plan.names
    idx = Dict(n => i for (i, n) in enumerate(names))
    max_eager_vs_jit = maximum(
        maximum(abs.(zl[i] .- eager.nodes[names[i]].z_latent)) for i in eachindex(zl)
    )

    # Cross-LANGUAGE numerics: Julia's Reactant-compiled converged z_latent vs upstream's
    # jax.jit converged z_latent, SAME weights/batch. A benchmark on a broken/diverged
    # computation is worthless -- this must be tight (float32 precision, ~1e-5-1e-6) before
    # the timing numbers below mean anything.
    d_x = maximum(abs.(zl[idx["x"]] .- jax_jit_x))
    d_h = maximum(abs.(zl[idx["h"]] .- jax_jit_h))
    d_y = maximum(abs.(zl[idx["y"]] .- jax_jit_y))

    t_eager = bench(() -> run_inference(params, init, clamps, structure), 10)

    # THREE JIT timings, because "julia_jit_steady_ms" was measuring something the JAX side never
    # pays. J-04's headline (7.87x) came from `ci(params, init)` — the SLOW path, which
    # `to_rarray`s every weight on EVERY call (~400 KB host->device for this MLP). JAX's timed
    # loop passes `params`/`init_state` that are ALREADY device-resident `jax.Array`s, so passing
    # them to a jitted fn copies nothing. We were charging Julia for a copy JAX doesn't make —
    # and one our own API already avoids (32b64a6 marshals params once at compile time; the
    # benchmark just never switched to it). Report all three rather than pick one, so the
    # marshalling cost is a MEASURED number instead of an attribution:
    #   remarshal — old headline: params re-marshalled per call (what 7.87x was measured on)
    #   fast      — params device-resident (compile-time), per-batch state marshalled per call.
    #               This is what a real E-step does (weights constant, batch arrives on host).
    #   resident  — params AND state device-resident: JAX's exact setup, isolating compute.
    # Residual asymmetry, stated not hidden: all three still copy the OUTPUT device->host via
    # `Array(...)`, which JAX's `block_until_ready` does not. That read is also what makes the
    # timing honest (decisions.md §11: timing the thunk without it shows a bogus ~1000x), so it
    # stays — it means these numbers are, if anything, conservative for Julia.
    t_jit_remarshal = bench(() -> ci(params, init), 30)
    t_jit = bench(() -> ci(init), 30)
    zl_host = ntuple(i -> init.nodes[ci.plan.names[i]].z_latent, length(ci.plan.names))
    zl_r = Reactant.to_rarray(zl_host)
    t_jit_resident = bench(() -> (o = ci.thunk(ci.params_r, zl_r); [Array(x) for x in o]), 30)
    # sync=true: Reactant's DOCUMENTED completion barrier (CompileOptions.jl: "blocking till the
    # computation is complete. This is recommended when benchmarking"). This is the true analog of
    # JAX's block_until_ready — it waits WITHOUT the device->host copy. We previously hand-rolled
    # the barrier as `Array(...)` and then documented the resulting copy as an unavoidable
    # asymmetry vs JAX; it was not unavoidable, it was us not reading the docs (audit 2026-07-16).
    # GATED on upstream #2647 (OPEN, "Buffer synchronisation is sometimes removed"): the emitted
    # wait can be DCE'd, which would silently restore the async artifact with no Array() left to
    # expose it. So we ASSERT the barrier is real below rather than trusting the flag.
    ci_sync = compile_inference(structure, params, clamps; batch=B, sync=true)
    t_sync = bench(() -> ci_sync.thunk(ci_sync.params_r, zl_r), 30)
    t_async_bare = bench(() -> ci.thunk(ci.params_r, zl_r), 30)
    if t_sync < 4 * t_async_bare
        @warn "#2647: sync=true's wait looks DCE'd — sync ($(t_sync*1e3) ms) is not meaningfully                slower than the bare async dispatch ($(t_async_bare*1e3) ms). Do NOT trust julia_jit_sync_ms."
    end


    println("=== RESULT ===")
    @printf("compile_time_s              = %.4f\n", t_compile)
    @printf("julia_eager_steady_ms       = %.4f\n", t_eager * 1e3)
    @printf("julia_jit_steady_ms         = %.4f   (params device-resident; HEADLINE)\n", t_jit * 1e3)
    @printf(
        "julia_jit_remarshal_ms      = %.4f   (params re-marshalled per call — the OLD J-04 path)\n",
        t_jit_remarshal * 1e3
    )
    @printf(
        "julia_jit_resident_ms       = %.4f   (params AND state device-resident — JAX's setup)\n",
        t_jit_resident * 1e3
    )
    @printf(
        "julia_jit_sync_ms           = %.4f   (sync=true barrier, NO host copy — the true JAX analog)\n",
        t_sync * 1e3
    )
    @printf(
        "  -> device->host copy cost          = %.4f ms  (resident - sync; the 'asymmetry' we used to concede)\n",
        (t_jit_resident - t_sync) * 1e3
    )
    @printf("  -> #2647 barrier check: sync/async_bare = %.0fx (want >>1; ~1 means the wait was DCE'd)\n",
            t_sync / t_async_bare)
    @printf(
        "  -> per-call param marshalling cost = %.4f ms  (remarshal - headline)\n",
        (t_jit_remarshal - t_jit) * 1e3
    )
    @printf(
        "  -> per-call state marshalling cost = %.4f ms  (headline - resident)\n",
        (t_jit - t_jit_resident) * 1e3
    )
    @printf("julia_jit_vs_julia_eager    = %.2fx\n", t_eager / t_jit)
    @printf("max|julia_jit - julia_eager| (in-language sanity) = %.3g\n", max_eager_vs_jit)
    @printf(
        "max|julia_jit - jax_jit| x/h/y (cross-language numerics) = %.3g / %.3g / %.3g\n",
        d_x, d_h, d_y
    )
end

main()
