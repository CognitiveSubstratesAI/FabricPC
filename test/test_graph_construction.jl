# graph() / graph_construction.jl vs upstream graph_construction.py (R-05b,
# docs/AUDIT_REGISTER.md section 8).
using Test
using FabricPC
using Random

@testset "graph_construction.jl: duplicate-edge handling matches upstream dict semantics" begin
    # A literal duplicate edge (same source, target, slot submitted twice) must be counted
    # ONCE, matching upstream's dict-keyed `edge_infos` (graph_construction.py:137-145) — not
    # twice, which is what happened before the `haskey` guard was added: `edge_keys_ordered`
    # was appended unconditionally, inflating `NodeInfo.in_edges`/`in_degree` for any node with
    # a duplicated in-edge (dormant in every shipped model, since none constructs a literal
    # duplicate edge, but a real silent-corruption landmine in the JIT flat path — see the
    # comment at the fix site in graph_construction.jl).
    xn = Linear((4,), "x")
    yn = Linear((3,), "y")

    st_dup = graph(
        [xn, yn], [Edge(xn, yn), Edge(xn, yn)],   # the SAME edge submitted twice
        TaskMap(; x=xn, y=yn), InferenceSGD(; eta_infer=0.1, infer_steps=5)
    )
    st_single = graph(
        [xn, yn], [Edge(xn, yn)],                 # the SAME edge submitted once
        TaskMap(; x=xn, y=yn), InferenceSGD(; eta_infer=0.1, infer_steps=5)
    )

    info_dup = st_dup.infos["y"]
    info_single = st_single.infos["y"]

    @test info_dup.in_degree == 1
    @test length(info_dup.in_edges) == 1
    @test info_dup.in_edges == info_single.in_edges
    @test info_dup.in_degree == info_single.in_degree

    # The graph must still train normally (forward/gradient path unaffected by the fix —
    # this is a topology-bookkeeping fix, not a numerics change).
    params = initialize_params(st_dup, MersenneTwister(1))
    x = randn(MersenneTwister(2), Float32, 4, 4)
    y = randn(MersenneTwister(3), Float32, 4, 3)
    _, energy, _ = train_step(
        params, Dict("x" => x, "y" => y), st_dup, 0.05, MersenneTwister(4)
    )
    @test isfinite(energy)
end
