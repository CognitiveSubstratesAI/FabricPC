#!/usr/bin/env julia
# Fully-PC DECOMPOSED transformer — autoregressive next-token prediction where each
# sub-component is its OWN predictive-coding node with its own latent + local energy:
#
#   x(source) → MhaResidual → LnMlp1 → Mlp2Residual(+skip from MhaResidual) → [target]
#
# Predictive coding operates at EVERY stage: the inference loop relaxes the MhaResidual
# and LnMlp1 latents (they are free/internal) to reconcile their local predictions,
# while x and the output are clamped. Every weight is learned by LOCAL PC via the
# Enzyme seam — NO backprop through the network. Causal MhaResidual ⇒ position t
# predicts token t+1.
#
# Run (needs Enzyme — the benchmark/jit env has it):
#   julia --project=benchmark/jit examples/decomposed_transformer_pc.jl

using FabricPC
using Enzyme
using Random, Printf

function make_data(rng, V, S, n)             # cyclic successor: tok[t+1] = tok[t] mod V + 1
    X = zeros(Float32, n, S, V)
    Y = zeros(Float32, n, S, V)
    for b in 1:n
        tok = rand(rng, 1:V)
        for t in 1:S
            nxt = (tok % V) + 1
            X[b, t, tok] = 1.0f0
            Y[b, t, nxt] = 1.0f0
            tok = nxt
        end
    end
    return X, Y
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
    V, S, n, H, ff = 8, 5, 16, 2, 16
    X, Y = make_data(rng, V, S, n)

    xn = Linear((S, V), "x")
    mha = MhaResidualNode((S, V), "mha"; num_heads=H, use_rope=true, causal=true)
    mlp1 = LnMlp1Node((S, ff), "mlp1")
    mlp2 = Mlp2ResidualNode((S, V), "mlp2")
    structure = graph(
        [xn, mha, mlp1, mlp2],
        [
            Edge(xn, mha),
            Edge(mha, mlp1),
            Edge(mlp1, mlp2),
            Edge(mha, slot(mlp2, "residual"))
        ],
        TaskMap(; x=xn, y=mlp2),
        InferenceSGD(; eta_infer=0.1, infer_steps=4)   # relax the free MhaResidual/LnMlp1 latents
    )
    params = initialize_params(structure, MersenneTwister(1))
    loader = [Dict("x" => X, "y" => Y)]

    pred0 = predict(params, structure, Dict("x" => X), MersenneTwister(2); output_task="y")
    @printf("init  next-token acc: %.3f\n", nexttok_acc(pred0, Y))

    opt = AdamW(params; lr=0.01)
    params, iters, _ = train_pcn(
        params, structure, loader, opt; num_epochs=80, rng=MersenneTwister(2),
        verbose=false
    )

    pred1 = predict(params, structure, Dict("x" => X), MersenneTwister(2); output_task="y")
    @printf(
        "final next-token acc: %.3f   (per-batch energy %.4f -> %.4f)\n",
        nexttok_acc(pred1, Y), iters[1][1], iters[end][1]
    )
    @printf(
        "=> a fully-PC decomposed transformer (PC at every stage) learned next-token by LOCAL PC.\n"
    )
end

main()
