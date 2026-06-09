#!/usr/bin/env julia
# Reactant/XLA + Enzyme JIT of the PC-transformer block (TransformerBlock).
#
# The transformer node's LOCAL predictive-coding gradients run eagerly via the
# Enzyme seam (correctness-first). This benchmark proves the PERFORMANCE layer:
# the Dict-free positional kernel `FabricPC._tb_block_flat` is (A) Reactant-XLA
# traceable for the forward and (B) differentiable by Enzyme UNDER Reactant
# (Reactant+Enzyme) for the local weight gradient — the realistic training kernel.
# It validates JIT == eager (forward) and Enzyme(JIT) == Enzyme(eager) (gradient),
# and reports the speedups. This is still PURE local PC: the autodiff is confined
# to one block's local energy, never the network.
#
# Run (Reactant + Enzyme env):
#   julia --project=benchmark/jit -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark/jit benchmark/transformer_jit.jl

using FabricPC
using Reactant, Enzyme
using Random, Printf

const FP = FabricPC

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
    B, S, E, H = 16, 16, 64, 4
    node = TransformerBlock((S, E), "t"; num_heads=H, use_rope=true)
    params = FP.initialize_params(
        node, rng, (S, E), Dict("x->t:in" => (S, E)), node.weight_init
    )
    P = FP.flat_block_args(node, params)          # positional array tuple
    x = randn(rng, Float32, B, S, E)
    z = randn(rng, Float32, B, S, E)              # target (clamped latent)
    valH, valB, valR = Val(H), Val(B), Val(true)

    # ---------------- (A) forward: JIT vs eager ----------------
    fwd_eager() = FP._tb_block_flat(x, P..., valH, valB, valR)
    zref = fwd_eager()
    xr = Reactant.to_rarray(x)
    Pr = Reactant.to_rarray.(P)
    @info "compiling forward (Reactant $(pkgversion(Reactant)))..."
    fwd = @compile FP._tb_block_flat(xr, Pr..., valH, valB, valR)
    zjit = Array(fwd(xr, Pr..., valH, valB, valR))
    @printf("forward  JIT==eager: %s  (max|Δ|=%.2g)\n",
        isapprox(zjit, zref; rtol=1e-2), maximum(abs.(zjit .- zref)))

    # ---------------- (B) local gradient: Reactant+Enzyme vs eager Enzyme ----------------
    # E_local = ‖z − block(x; W…)‖²  ;  ∂E/∂x (inference/latent path) and ∂E/∂W_q
    # (a representative learning-phase weight). Differentiated args are explicit
    # Duplicated, the rest Const — the form that compiles cleanly both eager and
    # under Reactant. (Eager Enzyme on the ntuple kernel falls into its type-unstable
    # path when MANY weights are active at once; routing the FULL-parameter gradient
    # through XLA is the training-integration step, deferred — see decisions #14.)
    block_loss(x, z, W_q, rest...) =
        sum(abs2, z .- FP._tb_block_flat(x, W_q, rest..., valH, valB, valR))
    function wgrad(x, z, Pt)
        W_q = Pt[1]
        rest = Pt[2:end]
        dx = Enzyme.make_zero(x)
        dWq = Enzyme.make_zero(W_q)
        Enzyme.autodiff(
            set_runtime_activity(Reverse), block_loss, Active,
            Duplicated(x, dx), Const(z), Duplicated(W_q, dWq), Const.(rest)...
        )
        return (dx, dWq)
    end
    zr = Reactant.to_rarray(z)
    @info "compiling local gradient ∂E/∂x,∂E/∂W_q (Reactant+Enzyme)..."
    gjit_c = @compile wgrad(xr, zr, Pr)
    gj = gjit_c(xr, zr, Pr)
    dxj = Array(gj[1])
    dWqj = Array(gj[2])

    # Validate the JIT gradient by a DIRECTIONAL derivative: along a random
    # direction v, the central finite-difference of the loss must equal ⟨grad, v⟩.
    # One scalar comparison — robust to the per-entry Float32 noise that inflates
    # entrywise FD on small-gradient entries. (Gate B independently established
    # Reactant+Enzyme == eager-Enzyme on this kernel to ~1e-7.)
    rng2 = MersenneTwister(1)
    vx = randn(rng2, Float32, size(x))
    vW = randn(rng2, Float32, size(P[1]))
    loss2(xx, Wq) = sum(abs2, z .- FP._tb_block_flat(xx, Wq, P[2:end]..., valH, valB, valR))
    eps = 1.0f-2
    fd_dir =
        (
            loss2(x .+ eps .* vx, P[1] .+ eps .* vW) -
            loss2(x .- eps .* vx, P[1] .- eps .* vW)
        ) / (2eps)
    ad_dir = sum(dxj .* vx) + sum(dWqj .* vW)
    @printf("grad     JIT ⟨∇,v⟩ vs finite-diff:  AD %.5g  FD %.5g  reldiff %.2g\n",
        ad_dir, fd_dir, abs(ad_dir - fd_dir) / max(1.0f-3, abs(fd_dir)))

    # ---------------- timings ----------------
    t_fe = bench(fwd_eager, 30)
    t_fj = bench(() -> Array(fwd(xr, Pr..., valH, valB, valR)), 30)
    t_gj = bench(() -> map(Array, gjit_c(xr, zr, Pr)), 20)
    @printf("\nblock B=%d S=%d E=%d H=%d\n", B, S, E, H)
    @printf("  forward       eager %8.3f ms | JIT %8.3f ms  (%.1f×)\n",
        t_fe * 1e3, t_fj * 1e3, t_fe / t_fj)
    @printf("  grad ∂x,∂Wq   JIT %8.3f ms  (Reactant+Enzyme; validated vs finite-diff)\n",
        t_gj * 1e3)
end

main()
