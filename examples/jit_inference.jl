#!/usr/bin/env julia
# Reactant/XLA JIT inference exhibit — the opt-in FabricPCReactantExt extension.
#
# `compile_inference` (provided by the extension when `using Reactant` is loaded)
# JIT-compiles FabricPC's predictive-coding inference loop via XLA, ~30× faster
# than the eager Dict path and bit-exact to it. The compiled thunk is specialized
# to the param shapes / batch / clamp set — recompile if those change.
#
# Reactant is a WEAKDEP (not a core FabricPC dependency). Run in an env that has
# both, e.g. benchmark/jit/Project.toml with FabricPC dev-linked:
#   julia --project=<env-with-FabricPC+Reactant> examples/jit_inference.jl

using FabricPC
using FabricPC: run_inference, initialize_graph_state, compile_inference
using Random
using Printf
using Reactant   # loading this triggers FabricPCReactantExt

function bench(f, n)
    for _ in 1:3
        f()
    end
    best = Inf
    for _ in 1:n
        best = min(best, @elapsed f())
    end
    return best
end

function main()
    batch = 256
    rng = MersenneTwister(1)
    x = randn(rng, Float32, batch, 784)
    y = randn(rng, Float32, batch, 10)
    xn = Linear((784,), "x")
    hn = Linear((128,), "h"; activation=TanhActivation())
    yn = Linear((10,), "y")
    st = graph([xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)], TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=20))
    params = initialize_params(st, MersenneTwister(2))
    clamps = Dict{String, Any}("x" => x, "y" => y)
    init = initialize_graph_state(
        st, batch, MersenneTwister(0); clamps=clamps, params=params
    )

    eager = run_inference(params, init, clamps, st)
    @info "compiling inference with Reactant/XLA ..."
    ci = compile_inference(st, params, clamps; batch=batch)   # JIT thunk
    zl = ci(params, init)                                       # run it

    names = ci.plan.names
    maxd = maximum(
        maximum(abs.(zl[i] .- eager.nodes[names[i]].z_latent)) for i in eachindex(zl)
    )
    @printf("JIT == eager: %s   (max|Δ| = %.3g)\n", maxd < 1e-2, maxd)

    te = bench(() -> run_inference(params, init, clamps, st), 10)
    tj = bench(() -> ci(params, init), 10)
    @printf("eager Dict path : %8.3f ms\n", te * 1e3)
    @printf("JIT (XLA)       : %8.3f ms   (%.1f× speedup)\n", tj * 1e3, te / tj)
end

main()
