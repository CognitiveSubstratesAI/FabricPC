#!/usr/bin/env julia
# PROOF-OF-CONCEPT (docs/decisions.md §28 addendum 3): applying the Julia memory-management
# reference's "Reducing Allocations" list to MNIST-MLP PC inference, showing the J-06 target
# concretely. This is NOT production code -- it is a hand-specialized, hardcoded x[clamped] ->
# h[tanh] -> y[clamped, identity, Gaussian] relaxation that demonstrates what a type-stable,
# pre-allocated, in-place inference path achieves, measured and bit-identical.
#
# Every technique from the reference is applied:
#   - in-place operations               mul!, @. fused broadcast, .= / .+=   (no `x = x + y`)
#   - pre-allocate + reuse buffers       scratch allocated ONCE, mutated each step
#   - avoid temporaries in tight loops   0 allocations per step
#   - concrete types (no ::Any boxing)   plain Matrix{Float32} throughout, type-stable
#   - (bakes in the §28 hoist+prune too: z_mu_h is loop-invariant, computed once)
#
# RESULT (B=256, 20 steps, one process, BLAS pinned+printed, measured 2026-07-14 CLEAN):
#   bit-identical to run_inference (max|diff| = 0.0)
#   allocations: 6565 -> 0   (208.3 MB -> 0 MB);  gc% 40-57% (sustained) -> 0%
#   single-call BLAS=1: stock 153.3ms -> hoist/prune 29.6ms (5.2x) -> in-place 5.1ms (5.8x more) = 30x
#   in-place is FLOP/bandwidth-LIMITED: ~6% above the arithmetic floor (F/26.82 ~= 4.79ms) -- NOT
#   "~ the FLOP ceiling" (that would be a category error: in-place eliminates 0 FLOPs, a different axis).
#   XLA-vs-eager (SLOPE test, marshalling-free -- decisions.md §28 addendum 3, corrected): XLA per-step
#   compute 0.127 ms/step is ~15-20% FASTER than eager 0.148-0.154. Eager's total only wins <~39 steps
#   because XLA pays ~2.5ms/call marshalling (GraphState->ConcreteRArray, naive-usage not fundamental).
#   So XLA IS faster on compute; and MNIST is the topology most favorable to eager (98.7% FLOP
#   elimination). Do NOT generalize "eager beats XLA" -- measure transformer_lm() first.
#   Run: julia --project=.warm/zygote_env this-file.  [NB: earlier headers cited contaminated
#   223/9.26/24.1x numbers (3 concurrent suites) and a "MATCHES/BEATS XLA" total-time claim confounded
#   by marshalling -- both superseded by the above.]
using FabricPC
using FabricPC: run_inference
import Random
using LinearAlgebra: mul!, BLAS

function main()
    BLAS.set_num_threads(1)
    DX, DH, DY, B = 784, 128, 10, 256
    rng = Random.MersenneTwister(0xABCD)
    x = Linear((DX,), "x"; use_bias=false)
    h = Linear((DH,), "h"; activation=TanhActivation())
    y = Linear((DY,), "y"; energy=GaussianEnergy())
    st = graph([x, h, y], [Edge(x, h), Edge(h, y)], TaskMap(x=x, y=y),
               InferenceSGD(eta_infer=0.1, infer_steps=20, latent_decay=0.0))
    Wxh = 0.1f0 .* randn(rng, Float32, DX, DH); bh = zeros(Float32, 1, DH)
    Why = 0.1f0 .* randn(rng, Float32, DH, DY); by = zeros(Float32, 1, DY)
    params = GraphParams(Dict(
        "x" => NodeParams(Dict{String,Matrix{Float32}}(), Dict{String,Matrix{Float32}}()),
        "h" => NodeParams(Dict("x->h:in" => Wxh), Dict("b" => bh)),
        "y" => NodeParams(Dict("h->y:in" => Why), Dict("b" => by))))
    clamps = Dict{String,Any}("x" => randn(rng, Float32, B, DX), "y" => randn(rng, Float32, B, DY))
    init = initialize_graph_state(st, B, rng; clamps=clamps, params=params)

    # pre-allocate ALL scratch ONCE (reused every step -- the reference's core technique)
    Xc = Float32.(init.nodes["x"].z_latent)         # clamped input (constant)
    Yc = Float32.(init.nodes["y"].z_latent)         # clamped target (constant)
    z0 = Float32.(copy(init.nodes["h"].z_latent))   # the one relaxing latent
    eta = 0.1f0
    pre_h  = Matrix{Float32}(undef, B, DH); z_mu_h = Matrix{Float32}(undef, B, DH)
    z_mu_y = Matrix{Float32}(undef, B, DY); pgy    = Matrix{Float32}(undef, B, DY)
    igr    = Matrix{Float32}(undef, B, DH); lg_h   = Matrix{Float32}(undef, B, DH)
    Whyt = permutedims(Why)

    function relax!(z_h)
        # hoist: z_mu_h = tanh(Xc*Wxh + bh) is loop-invariant (Xc clamped) -- computed ONCE
        mul!(pre_h, Xc, Wxh); pre_h .+= bh; z_mu_h .= tanh.(pre_h)
        @inbounds for _ in 1:20
            mul!(z_mu_y, z_h, Why); z_mu_y .+= by      # y forward (identity)
            pgy .= z_mu_y .- Yc                         # pre_grad_y = (z_mu_y - Y)·f'(=1)
            mul!(igr, pgy, Whyt)                        # input-grad y->h (no temp)
            lg_h .= z_h .- z_mu_h                        # self_grad_h  (Gaussian: z - z_mu)
            lg_h .+= igr                                # + downstream y contribution
            @. z_h = z_h - eta * lg_h                    # SGD update, fused in-place
        end
        return z_h
    end

    ref = run_inference(params, init, clamps, st)
    z1 = relax!(copy(z0))
    println("bit-identical to run_inference: ", all(z1 .== ref.nodes["h"].z_latent),
            "  (max|diff| = ", maximum(abs.(Float64.(z1) .- Float64.(ref.nodes["h"].z_latent))), ")")

    ac(f) = (f(); a=Base.gc_num(); f(); b=Base.gc_num(); Base.gc_alloc_count(Base.GC_Diff(b, a)))
    zws = copy(z0)
    println("allocations/call:  run_inference = ", ac(() -> run_inference(params, init, clamps, st)),
            "   in-place PoC = ", ac(() -> relax!(zws)))

    tmin(f) = (for _ in 1:3; f(); end; minimum(@elapsed(f()) for _ in 1:10))
    t_ref = tmin(() -> run_inference(params, init, clamps, st))
    t_opt = tmin(() -> run_inference(params, init, clamps, st; optimize=true))
    t_ip  = tmin(() -> relax!(copy(z0)))
    println("single-call ms:  stock=", round(t_ref*1e3, digits=1),
            "  hoist/prune=", round(t_opt*1e3, digits=1), " (", round(t_ref/t_opt, digits=1), "x)",
            "  in-place=", round(t_ip*1e3, digits=2), " (", round(t_opt/t_ip, digits=1), "x more) = ",
            round(t_ref/t_ip, digits=1), "x total (~FLOP ceiling 26.8x)")
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
