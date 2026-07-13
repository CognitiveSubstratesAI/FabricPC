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

    # (b) train_step_autoregressive over ~50 steps reduces energy; CE stays finite.
    onehot(t) = Float32[t[b, s] == v ? 1.0f0 : 0.0f0 for b in 1:B, s in 1:S, v in 1:V]
    batch = Dict{String, Any}("x" => Float32.(tokens), "y" => onehot(tokens))
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
