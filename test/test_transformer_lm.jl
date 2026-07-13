# Assembled transformer language-model graph (src/models/transformer_lm.jl).
# Verifies the end-to-end seq-model assembly that wires the decomposed transformer
# nodes into a working graph(): inference shapes, PC training reduces energy, and
# autoregressive greedy generation. Uses the Zygote autodiff seam (TransformerBlock
# + VocabProjectionNode implement only compute_mu; Enzyme crashes on the multi-head block).

using FabricPC
using Zygote   # AD seam for TransformerBlock/VocabProjection (Enzyme crashes on the multi-head block)
using Random
using Test
import FabricPC: run_inference, initialize_graph_state

@testset "transformer_lm assembled graph" begin
    S, V, E, H, B = 5, 7, 16, 4, 3

    structure = transformer_lm(;
        seq_len=S, vocab_size=V, embed_dim=E, num_heads=H,
        num_blocks=1, infer_steps=12, eta_infer=0.1
    )
    # topology sanity: input → embed → block → skip → output
    @test structure.node_order == ["input", "embed", "transformer_0", "skip_0", "output"]
    @test structure.task_map["x"] == "input"
    @test structure.task_map["y"] == "output"

    prng = MersenneTwister(11)
    params = initialize_params(structure, prng)

    # (a) inference on a (B, S) integer token batch → output z_mu of (B, S, V).
    drng = MersenneTwister(7)
    tokens = rand(drng, 1:V, B, S)
    clamps = Dict{String, Any}("input" => Float32.(tokens))
    st = initialize_graph_state(structure, B, drng; clamps=clamps, params=params)
    fs = run_inference(params, st, clamps, structure)
    zmu = fs.nodes["output"].z_mu
    @test size(zmu) == (B, S, V)
    @test all(isfinite, zmu)

    # F-05: raw-id one-hot expansion (train_autoregressive.py:127-139). A throwaway single step on
    # deep-copied params/rng so it doesn't perturb the 50-step run below.
    let
        output_node = structure.task_map["y"]
        y_onehot = Float32[
            tokens[b, s] == v ? 1.0f0 : 0.0f0 for b in 1:B, s in 1:S, v in 1:V
        ]
        batch_raw = Dict{String, Any}("x" => Float32.(tokens), "y" => Float32.(tokens))
        batch_1h = Dict{String, Any}("x" => Float32.(tokens), "y" => y_onehot)
        _, e_raw, ce_raw, fs_raw = train_step_autoregressive(
            deepcopy(params), 0.05, batch_raw, structure, MersenneTwister(13)
        )
        _, e_1h, ce_1h, fs_1h = train_step_autoregressive(
            deepcopy(params), 0.05, batch_1h, structure, MersenneTwister(13)
        )
        # the clamped output node's LATENT (z_latent — the held-fixed target; z_mu is the network's
        # own prediction, not the clamp) must be the EXPANDED one-hot array, not the raw ids — this
        # is exactly what a compute_loss-only fix could never guarantee (it only sees the CE metric,
        # computed after clamping/inference already happened on whatever was clamped).
        @test fs_raw.nodes[output_node].z_latent ≈ y_onehot
        @test ce_raw ≈ compute_loss(fs_raw, y_onehot, output_node)
        # pre-shaped one-hot batch ⇒ pass-through, not re-expanded ⇒ identical step.
        @test e_raw ≈ e_1h
        @test ce_raw ≈ ce_1h
        @test fs_raw.nodes[output_node].z_mu ≈ fs_1h.nodes[output_node].z_mu
        # neither raw (ndims==2) nor one-hot (ndims==3) ⇒ reject loudly, not silently misshape.
        bad = Dict{String, Any}(
            "x" => Float32.(tokens), "y" => reshape(y_onehot, B, S, V, 1)
        )
        @test_throws ArgumentError train_step_autoregressive(
            deepcopy(params), 0.05, bad, structure, MersenneTwister(13)
        )
    end

    # (b) train_step_autoregressive over ~50 steps reduces energy; CE stays finite.
    # F-05: "y" is raw token ids (B, S), matching the loader contract — train_step_autoregressive
    # one-hot-expands it internally (train_autoregressive.py:127-139) before clamping/scoring.
    batch = Dict{String, Any}("x" => Float32.(tokens), "y" => Float32.(tokens))
    trng = MersenneTwister(3)
    # Train with AdamW(1e-3) — upstream's canonical optimizer (optax.adamw(0.001, weight_decay=0.1);
    # ab_experiment.py:12, every train_*.py docstring). Plain SGD at lr=0.02 DIVERGES on a transformer
    # (weights -> 1e7 -> NaN by step ~4); Adam's per-parameter adaptive step is what makes attention
    # trainable. With this, E drops ~191 -> 8 over 50 steps.
    opt = AdamW(params; lr=1.0f-3, weight_decay=0.1f0)
    energies = Float32[]
    ces = Float32[]
    for _ in 1:50
        params, e, ce, _ = train_step_autoregressive(params, opt, batch, structure, trng)
        push!(energies, e)
        push!(ces, ce)
    end
    @test all(isfinite, energies)
    @test all(isfinite, ces)
    @test energies[end] < energies[1]                 # PC learning reduces energy

    # (c) greedy generation: deterministic across RNGs, correct shapes (1-D and 2-D).
    prompt1 = [1, 2, 3]
    g1a = generate_autoregressive(
        params, structure, prompt1, 4, MersenneTwister(1); top_k=1
    )
    g1b = generate_autoregressive(
        params, structure, prompt1, 4, MersenneTwister(999); top_k=1
    )
    @test length(g1a) == length(prompt1) + 4
    @test g1a[1:3] == prompt1                          # prompt preserved as a prefix
    @test g1a == g1b                                   # greedy ⇒ RNG-independent
    @test all(1 .<= g1a .<= V)

    prompt2 = [1 2 3; 4 5 6]
    g2 = generate_autoregressive(params, structure, prompt2, 4, MersenneTwister(1); top_k=1)
    @test size(g2) == (2, 3 + 4)
    @test g2[:, 1:3] == prompt2
end

@testset "evaluate_autoregressive (C-06, train_autoregressive.py:595-710)" begin
    S, V, E, H, B = 4, 6, 12, 2, 3
    structure = transformer_lm(;
        seq_len=S, vocab_size=V, embed_dim=E, num_heads=H,
        num_blocks=1, infer_steps=8, eta_infer=0.1
    )
    params = initialize_params(structure, MersenneTwister(21))

    drng = MersenneTwister(22)
    tokens1 = rand(drng, 1:V, B, S)
    tokens2 = rand(drng, 1:V, B, S)
    onehot(t) = Float32[t[b, s] == v ? 1.0f0 : 0.0f0 for b in 1:B, s in 1:S, v in 1:V]
    test_loader_raw = [
        Dict{String, Any}("x" => Float32.(tokens1), "y" => Float32.(tokens1)),
        Dict{String, Any}("x" => Float32.(tokens2), "y" => Float32.(tokens2))
    ]
    test_loader_1h = [
        Dict{String, Any}("x" => Float32.(tokens1), "y" => onehot(tokens1)),
        Dict{String, Any}("x" => Float32.(tokens2), "y" => onehot(tokens2))
    ]

    @testset "basic metrics: shape, range, perplexity == exp(loss)" begin
        metrics = evaluate_autoregressive(
            params, structure, test_loader_raw, MersenneTwister(1)
        )
        @test Set(keys(metrics)) == Set(["loss", "perplexity", "accuracy", "num_batches"])
        @test metrics["num_batches"] == length(test_loader_raw) == 2
        @test isfinite(metrics["loss"])
        @test metrics["perplexity"] ≈ exp(metrics["loss"])
        @test 0.0 <= metrics["accuracy"] <= 1.0
    end

    @testset "raw ids vs pre-one-hotted ⇒ identical metrics" begin
        m_raw = evaluate_autoregressive(
            params, structure, test_loader_raw, MersenneTwister(9)
        )
        m_1h = evaluate_autoregressive(
            params, structure, test_loader_1h, MersenneTwister(9)
        )
        @test m_raw["loss"] ≈ m_1h["loss"]
        @test m_raw["accuracy"] ≈ m_1h["accuracy"]
    end

    @testset "debug=true doesn't change metrics (only adds logging)" begin
        m_plain = evaluate_autoregressive(
            params, structure, test_loader_raw, MersenneTwister(4)
        )
        m_debug = evaluate_autoregressive(
            params, structure, test_loader_raw, MersenneTwister(4); debug=true
        )
        @test m_plain == m_debug
    end

    @testset "empty test_loader ⇒ zeros, no crash" begin
        metrics = evaluate_autoregressive(
            params, structure, Dict{String, Any}[], MersenneTwister(1)
        )
        @test metrics["num_batches"] == 0
        @test metrics["loss"] == 0.0
        @test metrics["accuracy"] == 0.0
    end
end

@testset "_count_correct: exact ground truth" begin
    predictions = zeros(Float32, 1, 2, 3)
    predictions[1, 1, :] = Float32[0.1, 0.7, 0.2]        # argmax = 2
    predictions[1, 2, :] = Float32[0.9, 0.05, 0.05]      # argmax = 1
    y = zeros(Float32, 1, 2, 3)
    y[1, 1, :] = Float32[0, 1, 0]                        # target = 2, matches
    y[1, 2, :] = Float32[0, 0, 1]                        # target = 3, mismatch (pred = 1)
    @test FabricPC._count_correct(predictions, y) == 1
end
