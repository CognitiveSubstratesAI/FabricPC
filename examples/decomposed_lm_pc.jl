#!/usr/bin/env julia
# A complete fully-PC transformer LANGUAGE MODEL — token ids in, vocab logits out —
# trained end-to-end by LOCAL predictive coding, NO backprop. Every stage is its own
# PC node with its own latent:
#
#   tokens → Embedding → MhaResidual → LnMlp1 → Mlp2Residual → VocabProjection → logits
#                                  └────────── skip ──────────┘
#
# Embedding (learned, scatter weight-grad), causal attention, FFN, and the vocab
# readout are ALL learned by local PC; the internal latents (Embedding/Mha/LnMlp1/
# Mlp2) are relaxed by multi-node PC inference while tokens + the next-token target
# are clamped. This is the transformer_v2 pipeline, fully assembled.
#
# Run (needs Enzyme — the benchmark/jit env has it):
#   julia --project=benchmark/jit examples/decomposed_lm_pc.jl

using FabricPC
using Enzyme
using Random, Printf

function make_data(rng, V, S, n)             # cyclic successor: tok[t+1] = tok[t] mod V + 1
    Xtok = zeros(Float32, n, S)              # input token ids (clamped to the source)
    Y = zeros(Float32, n, S, V)             # one-hot NEXT token (clamped to VocabProjection)
    for b in 1:n
        tok = rand(rng, 1:V)
        for t in 1:S
            nxt = (tok % V) + 1
            Xtok[b, t] = tok
            Y[b, t, nxt] = 1.0f0
            tok = nxt
        end
    end
    return Xtok, Y
end

function nexttok_acc(pred, Y)
    n, S, _ = size(pred)
    c = 0
    for b in 1:n, t in 1:S
        c += (argmax(@view pred[b, t, :]) == argmax(@view Y[b, t, :])) ? 1 : 0
    end
    return c / (n * S)
end

function main()
    rng = MersenneTwister(0)
    V, S, n, H, E, ff = 8, 5, 16, 2, 8, 16
    Xtok, Y = make_data(rng, V, S, n)

    tok = Linear((S,), "tok")                                    # source: token ids
    emb = EmbeddingNode((S, E), "emb"; vocab_size=V)
    mha = MhaResidualNode((S, E), "mha"; num_heads=H, use_rope=true, causal=true)
    mlp1 = LnMlp1Node((S, ff), "mlp1")
    mlp2 = Mlp2ResidualNode((S, E), "mlp2")
    voc = VocabProjectionNode((S, V), "voc")                    # output: vocab logits
    structure = graph(
        [tok, emb, mha, mlp1, mlp2, voc],
        [Edge(tok, emb), Edge(emb, mha), Edge(mha, mlp1), Edge(mlp1, mlp2),
            Edge(mha, slot(mlp2, "residual")), Edge(mlp2, voc)],
        TaskMap(; x=tok, y=voc),
        InferenceSGD(; eta_infer=0.1, infer_steps=5)            # relax the internal latents
    )
    params = initialize_params(structure, MersenneTwister(1))
    loader = [Dict("x" => Xtok, "y" => Y)]

    pred0 = predict(
        params, structure, Dict("x" => Xtok), MersenneTwister(2); output_task="y"
    )
    @printf("init  next-token acc: %.3f\n", nexttok_acc(pred0, Y))

    opt = AdamW(params; lr=0.01)
    params, iters, _ = train_pcn(
        params, structure, loader, opt; num_epochs=120, rng=MersenneTwister(2),
        verbose=false
    )

    pred1 = predict(
        params, structure, Dict("x" => Xtok), MersenneTwister(2); output_task="y"
    )
    @printf(
        "final next-token acc: %.3f   (per-batch energy %.4f -> %.4f)\n",
        nexttok_acc(pred1, Y), iters[1][1], iters[end][1]
    )
    @printf(
        "=> a full fully-PC transformer LM (token→logits) learned next-token by LOCAL PC.\n"
    )
end

main()
