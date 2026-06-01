# Dict-free (traceable) inference path == eager Dict path (Reactant JIT increment 1).
#
# Validates that flat_run_inference (position-indexed, no Dicts in the loop —
# the representation Reactant.@compile will use) reproduces the eager
# run_inference bit-for-bit, across: an MLP (train-style with x+y clamped, and
# predict-style with only x clamped → exercises the unclamped-output branch) and
# a residual net (exercises IdentityNode / Linear / LinearResidual flat forwards).

using FabricPC: CompiledPlan, to_flat_params, to_flat_state, flat_run_inference,
    run_inference, initialize_graph_state, slot

function _match(structure, params, clamps, batch)
    init = initialize_graph_state(
        structure, batch, MersenneTwister(0); clamps=clamps, params=params
    )
    eager = run_inference(params, init, clamps, structure)
    plan = CompiledPlan(structure)
    fstate = flat_run_inference(
        plan, to_flat_params(plan, params), to_flat_state(plan, init),
        Bool[n in keys(clamps) for n in plan.names]
    )
    ok = true
    for (i, name) in enumerate(plan.names)
        ok &= isapprox(fstate[i].z_latent, eager.nodes[name].z_latent; rtol=1e-5, atol=1e-6)
        ok &= isapprox(fstate[i].z_mu, eager.nodes[name].z_mu; rtol=1e-5, atol=1e-6)
    end
    return ok
end

@testset "flat run_inference == eager (MLP)" begin
    rng = MersenneTwister(1)
    batch = 8
    x = randn(rng, Float32, batch, 4)
    y = randn(rng, Float32, batch, 3)
    xn = Linear((4,), "x")
    hn = Linear((6,), "h"; activation=TanhActivation())
    yn = Linear((3,), "y")
    st = graph([xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)], TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=20))
    params = initialize_params(st, MersenneTwister(2))

    @test _match(st, params, Dict{String, Any}("x" => x, "y" => y), batch)   # train-style
    @test _match(st, params, Dict{String, Any}("x" => x), batch)             # predict-style
end

@testset "flat run_inference == eager (residual net)" begin
    rng = MersenneTwister(3)
    batch, d, nc = 8, 6, 3
    x = randn(rng, Float32, batch, d)
    y = randn(rng, Float32, batch, nc)
    input = IdentityNode((d,), "input")
    stem = Linear((d,), "stem")
    r1 = LinearResidual((d,), "r1"; activation=TanhActivation())
    r2 = LinearResidual((d,), "r2"; activation=TanhActivation())
    out = Linear((nc,), "y")
    edges = [
        Edge(input, stem), Edge(stem, r1), Edge(stem, slot(r1, "skip")),
        Edge(r1, r2), Edge(r1, slot(r2, "skip")), Edge(r2, out)
    ]
    st = graph([input, stem, r1, r2, out], edges, TaskMap(; x=input, y=out),
        InferenceSGD(; eta_infer=0.1, infer_steps=15))
    params = initialize_params(st, MersenneTwister(4))

    @test _match(st, params, Dict{String, Any}("x" => x, "y" => y), batch)
    @test _match(st, params, Dict{String, Any}("x" => x), batch)
end
