# GATED DIAGNOSTIC (not run by default -- see test/runtests.jl): the full theory-vs-measurement
# closure via perturb-and-track power iteration -- docs/decisions.md #24's own methodology,
# extended to the momentum recursion's joint (z, v) state, with periodic renormalization
# (prevents numerical underflow, and is what revealed #24's original lambda_min=2.3 was
# under-converged -- a short, unrenormalized power iteration gives ~0.977-0.98/step, matching
# #24's number, but the ratio keeps drifting for hundreds more steps before truly plateauing).
# Several minutes. Requires test/test_inference_momentum.jl's helpers (build_graph, fresh_init,
# PARAMS, CLAMPS) to already be in scope -- run via test/runtests.jl's gate, not standalone.
#
# Run: FABRICPC_MOMENTUM_SPECTRUM_DIAGNOSTIC=1 julia --project=. test/runtests.jl
using Random

function perturb_z_diag!(state, structure, clamps, rng, eps)
    for name in structure.node_names
        haskey(clamps, name) && continue
        ns = state.nodes[name]
        state = FabricPC.update_node_in_state(
            state, name; z_latent = ns.z_latent .+ Float32(eps) .* randn(rng, Float32, size(ns.z_latent))
        )
    end
    return state
end

function joint_dist_diag(stateA, stateB, velA, velB, structure, clamps)
    d = 0.0
    for name in structure.node_names
        haskey(clamps, name) && continue
        d += sum(abs2, Float64.(stateA.nodes[name].z_latent) .- Float64.(stateB.nodes[name].z_latent))
        d += sum(abs2, Float64.(velA[name]) .- Float64.(velB[name]))
    end
    return sqrt(d)
end

"""Renormalized perturb-and-track power iteration on the joint (z, v) state (via the real
`FabricPC.momentum_inference_step` -- not a hand-rolled duplicate): estimates the LOCAL
linearized map's dominant eigenvalue modulus at whatever point `burn_in` steps lands on.
Rescales the perturbation back to `eps` every `renorm_every` steps (numerical hygiene over long
runs); returns per-step log-ratios (reconstructable across renormalizations)."""
function perturb_and_track_renorm(inf::InferenceSGDMomentum; burn_in=40, n_iters=400, renorm_every=10, eps=1.0f-4, seed=7)
    structure = build_graph(inf)
    state0 = fresh_init(structure)
    vel0 = Dict{String, Any}(name => zero(state0.nodes[name].z_latent) for name in structure.node_names)
    state, vel = state0, vel0
    for _ in 1:burn_in
        state, vel = FabricPC.momentum_inference_step(inf, PARAMS, state, vel, CLAMPS, structure)
    end

    rng = Random.MersenneTwister(seed)
    stateA, velA = state, vel
    stateB, velB = perturb_z_diag!(state, structure, CLAMPS, rng, eps), vel

    log_ratios = Float64[]
    d_prev = joint_dist_diag(stateA, stateB, velA, velB, structure, CLAMPS)
    for step in 1:n_iters
        stateA, velA = FabricPC.momentum_inference_step(inf, PARAMS, stateA, velA, CLAMPS, structure)
        stateB, velB = FabricPC.momentum_inference_step(inf, PARAMS, stateB, velB, CLAMPS, structure)
        d = joint_dist_diag(stateA, stateB, velA, velB, structure, CLAMPS)
        push!(log_ratios, log(d) - log(d_prev))
        d_prev = d
        if step % renorm_every == 0 && d > 0
            scale = Float32(eps / d)
            for name in structure.node_names
                haskey(CLAMPS, name) && continue
                zA, zB = stateA.nodes[name].z_latent, stateB.nodes[name].z_latent
                new_zB = zA .+ (zB .- zA) .* scale
                stateB = FabricPC.update_node_in_state(stateB, name; z_latent=new_zB)
                velB[name] = velA[name] .+ (velB[name] .- velA[name]) .* scale
            end
            d_prev = joint_dist_diag(stateA, stateB, velA, velB, structure, CLAMPS)
        end
    end
    return log_ratios
end

@testset "InferenceSGDMomentum spectrum diagnostic (perturb-and-track, corrected)" begin
    println("\n" * "="^100)
    println("MOMENTUM SPECTRUM DIAGNOSTIC (docs/decisions.md #26)")
    println("="^100)

    lr_ctrl = perturb_and_track_renorm(
        InferenceSGDMomentum(; eta_infer=0.01f0, infer_steps=1, latent_decay=0.0, momentum=0.0f0)
    )
    ctrl_rate = exp(sum(lr_ctrl[end-19:end]) / 20)
    println("CONTROL (momentum=0, eta=0.01) converged rate: $(ctrl_rate)  (original docs/decisions.md #24 claim: ~0.977 -- an UNDER-converged short-iteration estimate)")

    lambda_min_true, lambda_max = 0.17, 113.6
    sqrt_max, sqrt_min = sqrt(lambda_max), sqrt(lambda_min_true)
    eta_star = 4.0 / (sqrt_max + sqrt_min)^2
    beta_star = ((sqrt_max - sqrt_min) / (sqrt_max + sqrt_min))^2
    lr_opt = perturb_and_track_renorm(
        InferenceSGDMomentum(;
            eta_infer=Float32(eta_star), infer_steps=1, latent_decay=0.0, momentum=Float32(beta_star)
        )
    )
    opt_rate = exp(sum(lr_opt[end-19:end]) / 20)
    println("MOMENTUM corrected-optimal (eta*=$(eta_star), beta*=$(beta_star)) converged rate: $(opt_rate)")
    println("  naive quadratic-model prediction sqrt(beta*) = $(sqrt(beta_star)) -- NOT expected to match closely (trajectory-dependent Hessian, see test_inference_momentum.jl's file header)")
    println("  what IS the claim: momentum measurably faster than plain SGD's own (corrected) rate, same methodology")

    @test ctrl_rate > 0.99   # confirms the corrected, slower control rate (not #24's original 0.977)
    @test opt_rate < ctrl_rate - 0.003   # real, if modest, speedup -- not noise
end
