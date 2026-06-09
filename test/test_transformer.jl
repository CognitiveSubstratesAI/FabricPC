# PC-transformer (TransformerBlock node). A transformer block trained by LOCAL
# predictive coding — gradients via the Phase-D Enzyme seam, NO backprop.
#
#  (1) FORWARD MATH — `compute_mu` (vectorized column-major reshapes + native MHA)
#      equals an independent explicit-loop oracle to Float32 tol. Locks the
#      reshape/head-split/RoPE math in-repo (the column-major hazard).
#  (2) END-TO-END — a graph (source → TransformerBlock-as-output) autoencodes
#      random sequences purely by PC: per-batch energy falls and the converged-
#      state reconstruction energy drops. The transformer's every weight is learned
#      by local PC (Enzyme differentiates one node's local energy; never the network).

using FabricPC
using FabricPC: NodeParams, compute_mu
using Enzyme
using Random, Test

const TF = FabricPC

# Independent explicit-loop oracle for the block forward (same conventions as
# compute_mu: contiguous head blocks, adjacent-pair RoPE, √seq variance comp).
function _block_loop(node::TransformerBlock, p::NodeParams, x)
    B, S, E = size(x)
    H = node.num_heads
    Dh = E ÷ H
    inv2 = 1.0f0 / sqrt(2.0f0)
    g(k) = p.weights[k]
    bb(k) = p.biases[k]
    ln(x, gk, bk) = begin
        out = similar(x)
        for bi in 1:B, s in 1:S
            v = Float32[x[bi, s, e] for e in 1:E]
            mu = sum(v) / E
            var = sum((v .- mu) .^ 2) / E
            for e in 1:E
                out[bi, s, e] =
                    g(gk)[1, e] * (v[e] - mu) / sqrt(var + 1.0f-5) + bb(bk)[1, e]
            end
        end
        out
    end
    dns(x, Wk, bk) = begin
        O = size(g(Wk), 2)
        out = zeros(Float32, B, S, O)
        for bi in 1:B, s in 1:S, o in 1:O
            out[bi, s, o] =
                bb(bk)[1, o] + sum(x[bi, s, i] * g(Wk)[i, o] for i in 1:size(x, 3))
        end
        out
    end
    cosA, sinA = FabricPC._tb_rope_tables(Dh, S)
    rope(xh) = begin
        out = similar(xh)
        half = Dh ÷ 2
        for bi in 1:B, s in 1:S, i in 1:half
            e1 = 2i - 1
            e2 = 2i
            x1 = xh[bi, s, e1]
            x2 = xh[bi, s, e2]
            out[bi, s, e1] = x1 * cosA[s, i] - x2 * sinA[s, i]
            out[bi, s, e2] = x1 * sinA[s, i] + x2 * cosA[s, i]
        end
        out
    end
    xn1 = ln(x, "ln1_gamma", "ln1_beta")
    Q = dns(xn1, "W_q", "b_q")
    K = dns(xn1, "W_k", "b_k")
    V = dns(xn1, "W_v", "b_v")
    attn_concat = zeros(Float32, B, S, E)
    for h in 1:H
        cols = ((h - 1) * Dh + 1):(h * Dh)
        Qh = Q[:, :, cols]
        Kh = K[:, :, cols]
        Vh = V[:, :, cols]
        if node.use_rope
            Qh = rope(Qh)
            Kh = rope(Kh)
        end
        scale = sqrt(Float32(Dh))
        for bi in 1:B, qi in 1:S
            sc = Float32[
                sum(Qh[bi, qi, d] * Kh[bi, ki, d] for d in 1:Dh) / scale for ki in 1:S
            ]
            m = maximum(sc)
            ex = exp.(sc .- m)
            at = ex ./ sum(ex)
            for d in 1:Dh
                attn_concat[bi, qi, cols[d]] = sum(at[ki] * Vh[bi, ki, d] for ki in 1:S)
            end
        end
    end
    attn = dns(attn_concat, "W_o", "b_o") .* sqrt(Float32(S))
    xres1 = inv2 .* (x .+ attn)
    xn2 = ln(xres1, "ln2_gamma", "ln2_beta")
    ff1 = dns(xn2, "W_ff1", "b_ff1")
    ffa = TF.forward(node.internal_activation, ff1)
    ff2 = dns(ffa, "W_ff2", "b_ff2")
    return inv2 .* (xres1 .+ ff2)
end

@testset "PC-transformer (TransformerBlock, Enzyme seam)" begin
    @testset "forward math == explicit-loop oracle" begin
        rng = MersenneTwister(11)
        B, S, E, H = 3, 4, 8, 2
        node = TransformerBlock((S, E), "t"; num_heads=H, use_rope=true)
        params = TF.initialize_params(
            node, rng, (S, E), Dict("x->t:in" => (S, E)), node.weight_init
        )
        x = randn(rng, Float32, B, S, E)
        inputs = Dict{String, Any}("x->t:in" => x)
        zv = compute_mu(node, params, inputs)
        zl = _block_loop(node, params, x)
        @test size(zv) == (B, S, E)
        @test zv ≈ zl rtol = 1e-4
        # RoPE off path also matches.
        node2 = TransformerBlock((S, E), "t"; num_heads=H, use_rope=false)
        @test compute_mu(node2, params, inputs) ≈ _block_loop(node2, params, x) rtol = 1e-4
    end

    @testset "Dict-free flat kernel == compute_mu (Reactant-path equivalence)" begin
        # The positional ntuple/Val kernel used for the Reactant+Enzyme JIT path must
        # be numerically identical to the eager Dict-based compute_mu (default config:
        # GELU internal, identity output). Guards the two forms from drifting apart.
        rng = MersenneTwister(13)
        B, S, E, H = 3, 4, 8, 2
        node = TransformerBlock((S, E), "t"; num_heads=H, use_rope=true)
        params = TF.initialize_params(
            node, rng, (S, E), Dict("x->t:in" => (S, E)), node.weight_init
        )
        x = randn(rng, Float32, B, S, E)
        zc = compute_mu(node, params, Dict{String, Any}("x->t:in" => x))
        zf = TF._tb_block_flat(
            x, TF.flat_block_args(node, params)..., Val(H), Val(B), Val(true)
        )
        @test zf ≈ zc rtol = 1e-5
    end

    @testset "autoencodes sequences by local PC (no backprop)" begin
        rng = MersenneTwister(2025)
        B, S, E, H = 4, 4, 8, 2
        X = randn(rng, Float32, B, S, E)

        # source → TransformerBlock-as-output, both clamped (input=target=X): the
        # block's WEIGHTS learn the reconstruction by local PC (Enzyme grad of the
        # block's local energy). Both endpoints clamped ⇒ no free latents ⇒
        # infer_steps=1 suffices. AdamW for scale-robust (NaN-free) updates.
        xn = Linear((S, E), "x")                       # source (in_degree 0, no params)
        tn = TransformerBlock((S, E), "t"; num_heads=H, use_rope=true)
        structure = graph(
            [xn, tn], [Edge(xn, tn)], TaskMap(; x=xn, y=tn),
            InferenceSGD(; eta_infer=0.1, infer_steps=1)
        )
        params = initialize_params(structure, MersenneTwister(3))
        loader = [Dict("x" => X, "y" => X)]            # autoencode (target = input)

        _, e0, _ = get_graph_param_gradient(
            params, loader[1], structure, MersenneTwister(5)
        )
        opt = AdamW(params; lr=0.01)
        params, iters, _ = train_pcn(
            params, structure, loader, opt; num_epochs=20,
            rng=MersenneTwister(5), verbose=false
        )
        _, e1, _ = get_graph_param_gradient(
            params, loader[1], structure, MersenneTwister(5)
        )

        @test isfinite(e1)                      # stable (no NaN blow-up)
        @test iters[end][1] < iters[1][1]      # per-batch energy fell over training
        @test e1 < e0                           # converged-state reconstruction improved
        @test e1 < 0.85f0 * e0                  # a real (non-trivial) drop
    end
end
