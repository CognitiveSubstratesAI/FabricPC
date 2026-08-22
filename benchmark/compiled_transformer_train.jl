#!/usr/bin/env julia
# Compiled full PC training step for a TransformerBlock graph (J-02b) -- E-step inference relaxation
# AND M-step local weight update in ONE XLA graph, with the transformer's local gradients supplied
# by Enzyme UNDER Reactant. Companion to benchmark/compiled_train_step.jl (the dense/SGD case).
#
# WHY this is the interesting case: benchmark/compiled_train_step.jl showed compiled buys nothing on
# compute for a dense MLP (decisions.md section 28 -- both memory-bandwidth-bound). The compiled
# lane's value proposition (auto-fusion of the transformer's LN->GEMM->softmax->GEMM chain) only
# materializes on the transformer.
#
# HOW it composes (the J-02b crux, verified): Enzyme.autodiff of the transformer's local energy
# COMPOSES inside `@trace track_numbers=false for` -- the compiled E-step calls _flat_input_grads(
# ::TransformerBlock) (partial-E/partial-x) every step, and the M-step calls _flat_weight_grads(
# ::TransformerBlock) (partial-E/partial-W for the whole flat_block_args tuple) once. Both are the
# same positional Enzyme kernel gate 2 (benchmark/transformer_jit.jl) proved correct to ~1e-7.
#
# VERIFICATION NOTE -- why this file smoke-tests rather than diffs against eager here: eager Enzyme
# ABORTS on the transformer forward ("Illegal calling convention fixup" -- batched_mul's BLAS
# convention, decisions.md; it works ONLY under Reactant), and Zygote (the stable eager oracle) is a
# DIFFERENT AD backend that cannot coexist in one process (F-04). So the numerical fidelity was
# established CROSS-SESSION: an eager Zygote train step (.warm/zygote_env) vs this compiled path gave
# max|dW|=5.96e-8, max|db|=3.35e-8, and the compiled W_q move (0.00955637) matched the eager move
# (0.00955640) to ~3e-8 -- a REAL, correct update, not DCE'd. To reproduce: run
# scratchpad tbtrain_ref.jl (Zygote) then tbtrain_c.jl (Reactant) from the J-02b session, or the
# two-session procedure in docs/AUDIT_REGISTER.md (J-02b). This file's self-contained gate is:
# (1) it compiles; (2) params are finite; (3) the update is real (moved > 0, so not DCE'd);
# (4) loop=true and loop=false agree (internal consistency, both Enzyme-under-Reactant).
#
# Run (Reactant + Enzyme env):
#   julia --project=benchmark/jit benchmark/compiled_transformer_train.jl

using FabricPC, Random, Printf
using FabricPC: IdentityNode, TransformerBlock, Edge, TaskMap, InferenceSGD, graph,
    initialize_params, initialize_graph_state, compile_train_step
using Reactant, Enzyme   # triggers FabricPCReactantExt + FabricPCEnzymeExt

max_w_delta(a, b, names) = maximum(
    maximum(abs.(a.nodes[n].weights[k] .- b.nodes[n].weights[k]))
    for n in names for k in keys(b.nodes[n].weights)
)

function main()
    rng = MersenneTwister(0)
    B, S, E, H = 2, 4, 16, 2
    lr = 0.01f0
    xn = IdentityNode((S, E), "x")
    hn = TransformerBlock((S, E), "h"; num_heads=H, use_rope=true)   # interior, unclamped
    yn = IdentityNode((S, E), "y")
    st = graph([xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)], TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.05, infer_steps=3))
    p = initialize_params(st, MersenneTwister(1))
    clamps = Dict{String, Any}(
        "x" => randn(rng, Float32, B, S, E), "y" => randn(rng, Float32, B, S, E)
    )
    init = initialize_graph_state(st, B, rng; clamps=clamps, params=p)

    println("== J-02b: compiled full PC train_step on a TransformerBlock graph ==")

    t0 = time()
    ct = compile_train_step(st, p, clamps; batch=B, lr=lr, loop=true)
    @printf("compile_train_step(loop=true) compiled in %.1fs\n", time() - t0)
    p1 = ct(p, init)

    h = p1.nodes["h"]
    allfinite =
        all(all(isfinite, v) for v in values(h.weights)) &&
        all(all(isfinite, v) for v in values(h.biases))
    moved = maximum(abs.(h.weights["W_q"] .- p.nodes["h"].weights["W_q"]))
    @printf(
        "params finite: %s   update magnitude max|dW_q| = %.4g  (GATE: > 0, real update -- no DCE)\n",
        allfinite, moved)

    # loop=false must agree with loop=true (both Enzyme-under-Reactant)
    ct2 = compile_train_step(st, p, clamps; batch=B, lr=lr, loop=false)
    d = max_w_delta(ct2(p, init), p1, ["h"])
    @printf(
        "loop=false vs loop=true: max|dW| = %.3e  (GATE: ~1e-7, internal consistency)\n", d
    )

    ok = allfinite && moved > 1e-4 && d < 1e-5
    println(ok ? "PASS" : "FAIL")
    return ok
end

main()
