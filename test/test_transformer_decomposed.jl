# Decomposed (fully-PC) transformer stages: MhaResidual / LnMlp1 / Mlp2Residual.
# Forward correctness (reusing the Enzyme-validated _tb_* helpers) + the riskier new
# surface: a MULTI-SLOT node (Mlp2Residual: "in" + "residual" skip) must get local PC
# gradients for BOTH input edges from the autodiff seam. End-to-end pipeline training
# is the heavy example (examples/decomposed_transformer_pc.jl), not CI.

using FabricPC
using FabricPC: NodeParams, NodeState, NodeInfo, SlotInfo, compute_mu
using Enzyme
using Random, Test

const TD = FabricPC

@testset "Decomposed (fully-PC) transformer stages" begin
    @testset "MhaResidual: forward + residual + causal no-leak" begin
        rng = MersenneTwister(1)
        B, S, E, H = 3, 5, 8, 2
        node = MhaResidualNode((S, E), "mha"; num_heads=H, causal=true)
        params = TD.initialize_params(
            node, rng, (S, E), Dict("x->mha:in" => (S, E)), node.weight_init
        )
        x = randn(rng, Float32, B, S, E)
        z = compute_mu(node, params, Dict{String, Any}("x->mha:in" => x))
        @test size(z) == (B, S, E)
        @test !(z ≈ x)                                   # residual added attention
        x2 = copy(x)
        x2[:, S, :] .+= reshape(Float32.(1:E), 1, E)     # non-uniform ⇒ survives LayerNorm
        z2 = compute_mu(node, params, Dict{String, Any}("x->mha:in" => x2))
        @test z[:, 1, :] ≈ z2[:, 1, :] rtol = 1e-5       # causal: no future leak
    end

    @testset "LnMlp1: forward shape (S, ff)" begin
        rng = MersenneTwister(2)
        B, S, E, ff = 3, 4, 8, 16
        node = LnMlp1Node((S, ff), "mlp1")
        params = TD.initialize_params(
            node, rng, (S, ff), Dict("a->mlp1:in" => (S, E)), node.weight_init
        )
        x = randn(rng, Float32, B, S, E)
        z = compute_mu(node, params, Dict{String, Any}("a->mlp1:in" => x))
        @test size(z) == (B, S, ff)
    end

    @testset "Mlp2Residual: multi-slot forward + grads for BOTH edges" begin
        rng = MersenneTwister(3)
        B, S, E, ff = 3, 4, 8, 16
        node = Mlp2ResidualNode((S, E), "mlp2")
        ishapes = Dict("m1->mlp2:in" => (S, ff), "mha->mlp2:residual" => (S, E))
        params = TD.initialize_params(node, rng, (S, E), ishapes, node.weight_init)
        mlp1 = randn(rng, Float32, B, S, ff)
        res = randn(rng, Float32, B, S, E)
        inputs = Dict{String, Any}(
            "m1->mlp2:in" => mlp1, "mha->mlp2:residual" => res
        )
        z = compute_mu(node, params, inputs)
        @test size(z) == (B, S, E)
        expected =
            res .+
            TD._tb_dense(
                mlp1, params.weights["W_ff2"], reshape(params.biases["b_ff2"], 1, 1, :)
            )
        @test z ≈ expected rtol = 1e-5

        # The autodiff seam must return local PC input gradients for BOTH slots
        # (incl. the skip "residual"), pushing them to each source.
        zlat = randn(rng, Float32, B, S, E)
        st = NodeState(zlat, zeros(Float32, B, S, E), zeros(Float32, B, S, E),
            zeros(Float32, B), zeros(Float32, B, S, E), zeros(Float32, B, S, E))
        info = NodeInfo("mlp2", (E,), "Mlp2ResidualNode", Dict{String, SlotInfo}(),
            2, 1, collect(keys(inputs)), String[], nothing)
        _, igrads, sg = TD.forward_and_latent_grads(node, params, inputs, st, info, false)
        @test haskey(igrads, "m1->mlp2:in") && haskey(igrads, "mha->mlp2:residual")
        @test all(isfinite, igrads["m1->mlp2:in"])
        @test all(isfinite, igrads["mha->mlp2:residual"])
        @test all(isfinite, sg)
        # residual is added straight through ⇒ ∂z_mu/∂res = I, so the residual-edge
        # input grad equals the self-latent grad's negative (Gaussian: ∂E/∂res = z_mu−z).
        @test igrads["mha->mlp2:residual"] ≈ (z .- zlat) rtol = 1e-3
    end
end
