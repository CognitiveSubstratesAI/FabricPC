#!/usr/bin/env julia
# FabricPC throughput benchmark (Phase E).
#
# Measures eager wall-clock for a full PC train_step (inference relaxation +
# local weight grads + SGD) on the MNIST-shaped net 784 → 128 → 10 at a few
# batch sizes. Min-of-N with GC disabled for stable numbers (same methodology as
# the NGCLearn benchmark). This is an EAGER baseline — no Reactant/JIT yet.
#
# Run:  julia --project=. benchmark/throughput.jl
# Commit numbers to benchmark/results.md.

using FabricPC
using Random
using Printf

function build(; hid=128)
    xn = Linear((784,), "x")
    hn = Linear((hid,), "h"; activation=TanhActivation())
    yn = Linear((10,), "y")
    return graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=30)
    )
end

function bench(f, n)
    for _ in 1:3                           # warm up (compile + cache)
        f()
    end
    GC.enable(false)
    best = Inf
    for _ in 1:n
        t = @elapsed f()
        best = min(best, t)
    end
    GC.enable(true)
    return best
end

function main()
    rng = MersenneTwister(0)
    st = build()
    params = initialize_params(st, rng)
    @printf("net 784→128→10, infer_steps=30, eager\n")
    @printf("%-8s %14s %16s\n", "batch", "train (ms/step)", "throughput (samp/s)")
    for bs in (16, 64, 256)
        x = randn(rng, Float32, bs, 784)
        y = zeros(Float32, bs, 10)
        for i in 1:bs
            y[i, rand(rng, 1:10)] = 1.0f0
        end
        bd = Dict("x" => x, "y" => y)
        train = bench(20) do
            train_step(params, bd, st, 0.002, rng)
        end
        @printf("%-8d %14.3f %16.1f\n", bs, train * 1e3, bs / train)
    end
end

main()
