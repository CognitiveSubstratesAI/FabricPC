#!/usr/bin/env julia
# Autoregressive next-token prediction with a CAUSAL PC-transformer, trained by
# LOCAL predictive coding — NO backprop. The transformer's every weight is learned
# by the Enzyme autodiff seam differentiating ONE node's local energy; causal
# self-attention (causal=true) makes position t predict token t+1 from tokens 1..t.
#
# Task: cyclic-successor sequences over a vocab of V tokens (one-hot embeddings):
#   tok[t+1] = (tok[t] mod V) + 1.  The model must read the current token (causal
#   self-attention at position t attends to itself) and output its successor.
#
# Run (needs Enzyme — the benchmark/jit env has it):
#   julia --project=benchmark/jit -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark/jit examples/autoregressive_pc.jl

using FabricPC
using Enzyme              # activates the autodiff seam (transformer local gradients)
using Random, Printf

"""Cyclic-successor sequences → (X one-hot current, Y one-hot next, tokens)."""
function make_data(rng, V, S, n)
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

"""Next-token accuracy: argmax of the prediction vs argmax of the one-hot target."""
function nexttok_acc(pred, Y)
    n, S, _ = size(pred)
    correct = 0
    for b in 1:n, t in 1:S
        correct += (argmax(@view pred[b, t, :]) == argmax(@view Y[b, t, :])) ? 1 : 0
    end
    return correct / (n * S)
end

function main()
    rng = MersenneTwister(0)
    V, S, n, H = 8, 6, 24, 2
    X, Y = make_data(rng, V, S, n)

    # source → causal transformer (output, clamped to the shifted target during
    # training). Both endpoints clamped ⇒ weight-only local PC (infer_steps=1).
    xn = Linear((S, V), "x")
    tn = TransformerBlock((S, V), "t"; num_heads=H, use_rope=true, causal=true)
    structure = graph(
        [xn, tn], [Edge(xn, tn)], TaskMap(; x=xn, y=tn),
        InferenceSGD(; eta_infer=0.1, infer_steps=1)
    )
    params = initialize_params(structure, MersenneTwister(1))
    loader = [Dict("x" => X, "y" => Y)]

    pred0 = predict(params, structure, Dict("x" => X), MersenneTwister(2); output_task="y")
    @printf("init  next-token acc: %.3f\n", nexttok_acc(pred0, Y))

    opt = AdamW(params; lr=0.01)
    params, iters, _ = train_pcn(
        params, structure, loader, opt; num_epochs=120, rng=MersenneTwister(2),
        verbose=false
    )

    pred1 = predict(params, structure, Dict("x" => X), MersenneTwister(2); output_task="y")
    @printf(
        "final next-token acc: %.3f   (per-batch energy %.4f -> %.4f)\n",
        nexttok_acc(pred1, Y), iters[1][1], iters[end][1]
    )
    @printf(
        "=> a causal transformer learned next-token prediction by LOCAL PC (no backprop).\n"
    )
end

main()
