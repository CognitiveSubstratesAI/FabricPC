#!/usr/bin/env julia
# Reactant/XLA JIT feasibility + payoff benchmark for FabricPC's PC inference.
#
# FabricPC's eager path uses Dict-keyed GraphState, which XLA cannot trace. This
# benchmark reformulates the SAME predictive-coding inference math (the hot loop:
# `infer_steps` × forward + local-gradient + latent update) over a fixed tuple of
# per-node arrays — the representation a full Reactant integration would use — and
# measures Reactant.@compile vs eager Julia on the MNIST-shaped MLP (784→128→10,
# Gaussian energy). It validates JIT == eager and reports the speedup, justifying
# the (multi-session) in-package GraphState refactor. See docs/decisions.md §11.
#
# Run (Reactant must be available):
#   julia --project=benchmark/jit benchmark/reactant_jit.jl
# (or any project that has Reactant in [deps]).

using Reactant
using Random
using Printf

# One PC inference relaxation: x clamped (input), y clamped (target, train-style),
# h relaxes. Gaussian energy, identity activations. Matches FabricPC's explicit
# local gradients: grad_h = e_h - e_y·W2ᵀ ; z_h -= eta·grad_h. Pure array ops,
# fixed control flow ⇒ XLA-traceable.
function pc_infer(W1, b1, W2, b2, x, zh, zy, eta, ::Val{STEPS}) where {STEPS}
    for _ in 1:STEPS
        mu_h = x * W1 .+ b1
        e_h = zh .- mu_h
        mu_y = zh * W2 .+ b2
        e_y = zy .- mu_y
        grad_h = e_h .- e_y * transpose(W2)
        zh = zh .- eta .* grad_h
    end
    return zh
end

function bench(f, n)
    for _ in 1:3
        f()
    end
    best = Inf
    for _ in 1:n
        t = @elapsed f()
        best = min(best, t)
    end
    return best
end

function main()
    rng = MersenneTwister(0)
    batch, din, dh, dout = 256, 784, 128, 10
    steps = 20
    x = randn(rng, Float32, batch, din)
    W1 = 0.05f0 .* randn(rng, Float32, din, dh)
    b1 = zeros(Float32, 1, dh)
    W2 = 0.05f0 .* randn(rng, Float32, dh, dout)
    b2 = zeros(Float32, 1, dout)
    zh = zeros(Float32, batch, dh)
    zy = randn(rng, Float32, batch, dout)
    eta = 0.1f0

    eager() = pc_infer(W1, b1, W2, b2, x, zh, zy, eta, Val(steps))
    ref = eager()

    args = Reactant.to_rarray.((W1, b1, W2, b2, x, zh, zy))
    etar = Reactant.to_rarray(eta)
    @info "compiling with Reactant $(pkgversion(Reactant)) ..."
    comp = @compile pc_infer(args..., etar, Val(steps))
    jit_out = Array(comp(args..., etar, Val(steps)))

    ok = isapprox(jit_out, ref; rtol=1e-2)
    @printf("JIT == eager: %s  (max|Δ|=%.3g)\n", ok, maximum(abs.(jit_out .- ref)))

    t_eager = bench(eager, 20)
    # Force materialization (Array) so the timing captures actual XLA compute, not
    # just async dispatch of the Reactant thunk.
    t_jit = bench(() -> Array(comp(args..., etar, Val(steps))), 20)
    @printf("net 784→128→10, batch %d, infer_steps %d\n", batch, steps)
    @printf("  eager : %8.3f ms\n", t_eager * 1e3)
    @printf("  JIT   : %8.3f ms   (%.1f× speedup)\n", t_jit * 1e3, t_eager / t_jit)
end

main()
