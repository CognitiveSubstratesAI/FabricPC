#!/usr/bin/env julia
# Associative memory by predictive coding — a StorkeyHopfield node learns to RECALL a
# set of stored ±1 patterns from noisy probes, trained by LOCAL PC (no backprop). The
# node's composite energy E_pc + s·E_hop pulls the latent toward both the upstream
# probe-prediction AND the attractor landscape encoded in W; learning shapes W so the
# clean patterns become attractors.
#
#   noisy probe (source) → StorkeyHopfield (clamped to clean pattern in training)
#
# Run (needs Enzyme — the benchmark/jit env has it):
#   julia --project=benchmark/jit examples/hopfield_assoc_memory.jl

using FabricPC
using Enzyme
using Random, Printf

"""P random ±1 patterns; each training sample = a pattern with `nflip` bits flipped."""
function make_data(rng, D, P, per_pat, nflip)
    pats = [rand(rng, (-1.0f0, 1.0f0), D) for _ in 1:P]
    n = P * per_pat
    Xn = zeros(Float32, n, D)      # noisy probe
    Yc = zeros(Float32, n, D)      # clean target
    row = 1
    for p in 1:P, _ in 1:per_pat
        v = copy(pats[p])
        for i in randperm(rng, D)[1:nflip]
            v[i] = -v[i]
        end
        Xn[row, :] = v
        Yc[row, :] = pats[p]
        row += 1
    end
    return Xn, Yc
end

"""Recall bit-accuracy: sign(prediction) vs the clean ±1 target."""
function recall_acc(pred, Yc)
    n, D = size(pred)
    correct = sum(sign.(pred) .== sign.(Yc))
    return correct / (n * D)
end

function main()
    rng = MersenneTwister(0)
    D, P, per_pat, nflip = 16, 4, 8, 3
    Xn, Yc = make_data(rng, D, P, per_pat, nflip)

    src = Linear((D,), "x")
    hop = StorkeyHopfield((D,), "h"; enforce_symmetry=true)   # learnable strength, tanh
    structure = graph(
        [src, hop], [Edge(src, hop)], TaskMap(; x=src, y=hop),
        InferenceSGD(; eta_infer=0.1, infer_steps=1)          # both clamped ⇒ weight-only
    )
    params = initialize_params(structure, MersenneTwister(1))
    loader = [Dict("x" => Xn, "y" => Yc)]

    p0 = predict(params, structure, Dict("x" => Xn), MersenneTwister(2); output_task="y")
    @printf("init  recall bit-acc: %.3f\n", recall_acc(p0, Yc))

    opt = AdamW(params; lr=0.02)
    params, iters, _ = train_pcn(
        params, structure, loader, opt; num_epochs=150, rng=MersenneTwister(2),
        verbose=false
    )

    p1 = predict(params, structure, Dict("x" => Xn), MersenneTwister(2); output_task="y")
    @printf(
        "final recall bit-acc: %.3f   (per-batch energy %.4f -> %.4f)\n",
        recall_acc(p1, Yc), iters[1][1], iters[end][1]
    )
    @printf("=> a Hopfield node learned associative recall (denoising) by LOCAL PC.\n")
end

main()
