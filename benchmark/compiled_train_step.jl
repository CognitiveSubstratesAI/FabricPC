#!/usr/bin/env julia
# Compiled full PC TRAINING step (E-step inference + local SGD M-step in ONE XLA graph) vs the
# eager Dict ground truth -- Julia side. This is the J-02 deliverable (docs/AUDIT_REGISTER.md
# section 5): unlike benchmark/mnist_inference_vs_jax.jl (inference only, params fixed), this
# compiles the WHOLE train_step -- run inference to convergence, then apply the closed-form local
# weight update -- and returns the UPDATED params.
#
# WHY it compiles with no Enzyme: for the dense lane (Linear/Identity/Skip/LinearResidual) BOTH the
# latent grads (inference) AND the weight grads (learning) are closed-form local rules, all matmuls
# -- so the entire step traces under `Reactant.@compile` with `track_numbers=false` (J-09), no
# autodiff. Enzyme-under-Reactant (J-10 gate 2) is only needed for the TransformerBlock weight
# backward, which is out of this dense scope.
#
# Run (needs Reactant -- an env with FabricPC dev-linked + Reactant, e.g. benchmark/jit):
#   julia --project=benchmark/jit benchmark/compiled_train_step.jl
# (or drive it through a warm Reactant session -- tools/warm_session_reactant.jl.)
#
# GATE: compiled-vs-eager max|Δ| must be ~1e-7 (float32 reassociation), NOT ~lr·|grad| (which would
# mean the M-step was dead-code-eliminated -- the DCE hazard of upstream #2647). A no-op update
# (params unchanged) is also asserted against.

using FabricPC, Random, Printf
using FabricPC: Linear, Edge, TaskMap, InferenceSGD, graph, initialize_params, initialize_graph_state,
                run_inference, compute_local_weight_gradients, sgd_update, TanhActivation
using Reactant   # triggers FabricPCReactantExt (defines compile_train_step)

# One eager Dict training step from a SHARED init state (the ground truth train_step body).
function eager_train_step(params, clamps, init, structure, lr)
    final = run_inference(params, init, clamps, structure)
    grads = compute_local_weight_gradients(params, final, structure)
    return sgd_update(params, grads, lr)
end

max_w_delta(a, b, names) = maximum(
    maximum(abs.(a.nodes[n].weights[k] .- b.nodes[n].weights[k]))
    for n in names for k in keys(b.nodes[n].weights)
)

function main()
    rng = MersenneTwister(0)
    xn = Linear((784,), "x")
    hn = Linear((128,), "h"; activation=TanhActivation())
    yn = Linear((10,), "y")
    st = graph([xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)], TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=20))
    p = initialize_params(st, MersenneTwister(1))
    B = 64
    lr = 0.01f0
    clamps = Dict{String, Any}("x" => randn(rng, Float32, B, 784), "y" => randn(rng, Float32, B, 10))
    init = initialize_graph_state(st, B, rng; clamps=clamps, params=p)

    println("== J-02: compiled full PC train_step vs eager ==")

    t0 = time()
    ct = compile_train_step(st, p, clamps; batch=B, lr=lr, loop=true)
    @printf("compile_train_step(loop=true) compiled in %.1fs\n", time() - t0)

    pC = ct(p, init)
    pE = eager_train_step(p, clamps, init, st, lr)

    dW = max_w_delta(pC, pE, ["h", "y"])
    moved = max_w_delta(pC, p, ["h", "y"])
    @printf("compiled vs eager:  max|Δw| = %.3e   (GATE: ~1e-7 reassociation, NOT ~lr*|grad|)\n", dW)
    @printf("update magnitude:   max|Δ vs initial| = %.4f   (GATE: > 0, a real update -- no DCE)\n", moved)

    # loop=false unrolled runner -- same correctness
    ct2 = compile_train_step(st, p, clamps; batch=B, lr=lr, loop=false)
    dWu = max_w_delta(ct2(p, init), pE, ["h", "y"])
    @printf("loop=false vs eager: max|Δw| = %.3e\n", dWu)

    # multi-step composition (feed output back) stays bounded
    pe = p; pc = p
    for _ in 1:3
        pe = eager_train_step(pe, clamps, init, st, lr)
        pc = ct(pc, init)
    end
    @printf("3-step compiled vs eager: max|Δw| = %.3e   (drift bounded)\n", max_w_delta(pc, pe, ["h", "y"]))

    ok = dW < 1e-6 && moved > 1e-3 && dWu < 1e-6
    println(ok ? "PASS" : "FAIL")
    return ok
end

main()
