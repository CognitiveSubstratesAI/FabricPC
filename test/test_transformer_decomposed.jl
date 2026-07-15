# Decomposed (fully-PC) transformer stages: MhaResidual / LnMlp1 / Mlp2Residual.
# Forward correctness (reusing the Enzyme-validated _tb_* helpers) + the riskier new
# surface: a MULTI-SLOT node (Mlp2Residual: "in" + "residual" skip) must get local PC
# gradients for BOTH input edges from the autodiff seam. End-to-end pipeline training
# is the heavy example (examples/decomposed_transformer_pc.jl), not CI.

using FabricPC
using FabricPC: NodeParams, NodeState, NodeInfo, SlotInfo, compute_mu
using Zygote
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

        # F-02: the residual bypass must be decoupled from the "in" edge. This is a
        # pure forward composition property of compute_mu — no gradients involved.
        # Under muPC, scale_inputs pre-scales the "in" edge before compute_mu ever
        # sees it; a decoupled "skip" edge must carry the UNSCALED bypass value.
        skip = randn(rng, Float32, B, S, E)   # deliberately DIFFERENT from x
        z_skip = compute_mu(
            node, params, Dict{String, Any}("x->mha:in" => x, "x->mha:skip" => skip)
        )
        attn_only = z_skip .- skip
        @test z_skip ≈ skip .+ attn_only rtol = 1.0f-6            # residual == skip, not x
        @test !(z_skip ≈ z)                                       # differs from the no-skip case
        # No "skip" edge wired ⇒ falls back to "in" (today's byte-identical behavior,
        # matching the pre-fix code path whenever scaling is off).
        z_noskip = compute_mu(node, params, Dict{String, Any}("x->mha:in" => x))
        @test z_noskip ≈ z rtol = 1.0f-6
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
            zeros(Float32, B), zeros(Float32, B, S, E))
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

    @testset "EmbeddingNode: lookup + scatter weight-grad + discrete latent-grad" begin
        rng = MersenneTwister(1)
        B, S, E, V = 2, 3, 4, 7
        en = EmbeddingNode((S, E), "emb"; vocab_size=V)
        ep = TD.initialize_params(
            en, rng, (S, E), Dict("tok->emb:in" => (S,)), en.weight_init
        )
        idx = Float32[((b + s) % V) + 1 for b in 1:B, s in 1:S]
        inputs = Dict{String, Any}("tok->emb:in" => idx)
        zlat = randn(rng, Float32, B, S, E)
        st = NodeState(zlat, zeros(Float32, B, S, E), zeros(Float32, B, S, E),
            zeros(Float32, B), zeros(Float32, B, S, E))
        ns = TD.forward(en, ep, inputs, st)
        emb = ep.weights["embeddings"]
        @test all(ns.z_mu[b, s, :] ≈ emb[Int(idx[b, s]), :] for b in 1:B, s in 1:S)
        # weight grad = scatter-add of ∂E/∂z_mu into rows by token id (vs manual)
        _, gp = TD.forward_and_weight_grads(en, ep, inputs, st)
        gm = ns.z_mu .- zlat                                   # grad_mu (Gaussian)
        manual = zero(emb)
        for b in 1:B, s in 1:S
            manual[Int(idx[b, s]), :] .+= gm[b, s, :]
        end
        @test gp.weights["embeddings"] ≈ manual rtol = 1e-5
        # discrete input ⇒ zero input grad; self grad = grad_latent (z − z_mu)
        info = NodeInfo("emb", (E,), "EmbeddingNode", Dict{String, SlotInfo}(),
            1, 1, ["tok->emb:in"], String[], nothing)
        _, ig, sg = TD.forward_and_latent_grads(en, ep, inputs, st, info, false)
        @test all(iszero, ig["tok->emb:in"])
        @test sg ≈ (zlat .- ns.z_mu) rtol = 1e-5
    end

    @testset "VocabProjectionNode: embed → vocab probabilities (Softmax+CrossEntropy default)" begin
        rng = MersenneTwister(2)
        B, S, E, V = 2, 4, 8, 5
        vp = VocabProjectionNode((S, V), "vocab")
        vpp = TD.initialize_params(
            vp, rng, (S, V), Dict("h->vocab:in" => (S, E)), vp.weight_init
        )
        x = randn(rng, Float32, B, S, E)
        z = compute_mu(vp, vpp, Dict{String, Any}("h->vocab:in" => x))
        @test size(z) == (B, S, V)
        # default is now Softmax over the vocab axis (per upstream transformer_v2): a distribution.
        @test all(>=(0.0f0), z) && all(isapprox.(sum(z; dims=3), 1.0f0; atol=1.0f-5))
        logits = TD._tb_dense(
            x, vpp.weights["W_out"], reshape(vpp.biases["b_out"], 1, 1, :)
        )
        sm = exp.(logits .- maximum(logits; dims=3));
        sm = sm ./ sum(sm; dims=3)
        @test z ≈ sm                                           # = softmax(raw dense projection)
    end
end
