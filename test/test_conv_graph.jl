# C-01 layer 3: a conv/pool graph END-TO-END. See decisions.md §32 (the three-layer gate).
#
# WHY THIS FILE EXISTS, stated so it is not mistaken for a duplicate of the conformance tier:
# `conformance/test_tier_conv.jl` gates the NODE API against upstream JAX but builds
# NodeParams/NodeState/NodeInfo BY HAND and never calls `graph()`. It is therefore structurally
# blind to everything between a node and a trained network: slot wiring, the real
# `initialize_params` dispatch (which threads `weight_init` and the ":in" edge-key convention),
# `FeedforwardStateInit` on rank-4 latents, and the optimizers. This file covers exactly that gap
# — and it is not hypothetical: both claims below were UNTESTED arguments until this ran.
#
#   * B2 (df1d470) widened 11 optimizer sites from Dict{String,Matrix{Float32}} to
#     Dict{String,Array{Float32}} on the ARGUMENT that a conv graph "forwards fine and dies at the
#     first weight update". No conv graph had ever taken a step. Now one does — SGD, AdamW, and
#     AdamW's second-step MOMENT STATE, all on a rank-4 kernel.
#   * C-08 (df1d470) claimed conv fan_in = C_in*prod(kernel). That was algebra. Here it is a
#     number, produced by the REAL initialize_params path.
#
# Deliberately oracle-free: fidelity vs upstream is the tier's job. This asserts the wired path
# runs, shapes chain, and the thing LEARNS.
using Test
using FabricPC
using Random
using Statistics

@testset "conv graph end-to-end (C-01 layer 3: the wired path)" begin
    rng = MersenneTwister(20260715)
    B, H, W_, CIN = 4, 8, 8, 2
    # A small CNN: x -> conv(ReLU,SAME) -> maxpool -> global-avgpool -> out
    x = Linear((H, W_, CIN), "x"; use_bias=false)
    conv = ConvNode((8, 8, 4), "conv", (3, 3); padding="SAME", activation=ReLUActivation())
    pool = MaxPool((4, 4, 4), "pool", (2, 2))            # stride defaults to the window
    gap = AvgPool((4,), "gap"; global_pool=true)         # (B,4,4,4) -> (B,4)
    out = Linear((3,), "out"; energy=GaussianEnergy())
    st = graph([x, conv, pool, gap, out],
               [Edge(x, conv), Edge(conv, pool), Edge(pool, gap), Edge(gap, out)],
               TaskMap(x=x, y=out),
               InferenceSGD(eta_infer=0.05, infer_steps=5, latent_decay=0.0))
    @test length(st.node_names) == 5
    @test st.node_order == ["x", "conv", "pool", "gap", "out"]

    @testset "real initialize_params (not hand-built) + C-08 fan-in AS A MEASUREMENT" begin
        params = initialize_params(st, rng)
        @test size(params.nodes["conv"].weights["x->conv:in"]) == (3, 3, 2, 4)  # (kernel…,C_in,C_out)
        @test size(params.nodes["conv"].biases["b"]) == (1, 1, 1, 4)            # rank R+2, broadcasts
        @test isempty(params.nodes["pool"].weights)                             # pooling is param-free
        @test isempty(params.nodes["gap"].weights)
        # C-08: Kaiming is ConvNode's DEFAULT weight_init. On a (3,3,2,4) kernel the ND fan-in is
        # 3*3*2 = 18 ⇒ std = sqrt(2/18) ≈ 0.3333. The pre-C-08 rank-2-only fan_in = shape[1] = 3
        # would give sqrt(2/3) ≈ 0.8165 — a 2.45x too-wide init, i.e. a ResNet training mystery
        # rather than a test failure. This is the check that makes the algebra a number.
        s = std(vec(params.nodes["conv"].weights["x->conv:in"]))
        @test isapprox(s, sqrt(2 / 18); rtol=0.15)      # loose: it is a 72-sample std
        @test !isapprox(s, sqrt(2 / 3); rtol=0.15)      # non-vacuous: the WRONG fan is excluded
    end

    params = initialize_params(st, rng)
    clamps = Dict{String,Any}("x" => randn(rng, Float32, B, H, W_, CIN),
                              "out" => randn(rng, Float32, B, 3))

    @testset "rank-4 latents chain through state init + inference" begin
        state = initialize_graph_state(st, B, rng; clamps=clamps, params=params)
        @test size(state.nodes["conv"].z_latent) == (B, 8, 8, 4)
        @test size(state.nodes["pool"].z_latent) == (B, 4, 4, 4)
        @test size(state.nodes["gap"].z_latent) == (B, 4)          # spatial collapsed
        s2 = run_inference(params, state, clamps, st)
        for n in st.node_names
            @test all(isfinite, s2.nodes[n].z_latent)
        end
        # ReLU actually applied, non-vacuously: it clipped real negatives to exact zeros.
        # (Was `any(pre_activation .< 0)`; upstream b6f64ad removed that field from NodeState.)
        @test all(s2.nodes["conv"].z_mu .>= 0) && any(s2.nodes["conv"].z_mu .== 0)
    end

    @testset "B2: a rank-4 kernel survives the optimizers (was an ARGUMENT until this ran)" begin
        batch = Dict{String,Any}("x" => clamps["x"], "y" => clamps["out"])  # TaskMap-keyed
        gp, _, _ = get_graph_param_gradient(params, batch, st, rng)
        gk = gp.nodes["conv"].weights["x->conv:in"]
        @test size(gk) == (3, 3, 2, 4)
        @test all(isfinite, gk)
        @test any(!iszero, gk)                                     # non-vacuous: it really got a grad
        # plain SGD on rank-4
        @test size(sgd_update(params.nodes["conv"], gp.nodes["conv"], 0.01).weights["x->conv:in"]) == (3, 3, 2, 4)
        # AdamW: construction preallocates rank-4 moment state; step twice to exercise it.
        opt = AdamW(params; lr=1e-3)
        p2 = step!(opt, params, gp)
        @test all(isfinite, p2.nodes["conv"].weights["x->conv:in"])
        @test p2.nodes["conv"].weights["x->conv:in"] != params.nodes["conv"].weights["x->conv:in"]
        p3 = step!(opt, p2, gp)                                    # 2nd step: moments are now rank-4 state
        @test all(isfinite, p3.nodes["conv"].weights["x->conv:in"])
    end

    @testset "it LEARNS (energy decreases over train_steps)" begin
        batch = Dict{String,Any}("x" => clamps["x"], "y" => clamps["out"])
        p, e1, _ = train_step(params, batch, st, 0.01, rng)
        p2, e2, _ = train_step(p, batch, st, 0.01, rng)
        @test all(isfinite, p2.nodes["conv"].weights["x->conv:in"])
        @test sum(e2) < sum(e1)        # executing is not learning; this is the difference
    end
end
