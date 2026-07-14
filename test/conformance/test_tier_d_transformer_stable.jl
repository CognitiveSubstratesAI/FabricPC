# Tier D transformer-LM, PRIMARY/UNGATED conformance test (docs/AUDIT_REGISTER.md section 6).
# Runs at eta_infer=0.01, the CONTRACTIVE regime of this graph's PC-relaxation iteration matrix
# (I - eta*H): measured tail per-step amplification ~0.88-0.98 (perturbations shrink), vs the
# EXPANSIVE eta=0.1 production default's measured ~1.24-1.27/step (perturbations grow ~25,000-
# 45,000x over the same 12 steps -- see the DEMOTED test_tier_d_transformer.jl / tier_d_transformer.npz,
# kept as a diagnostic conditioning envelope, not a fidelity gate). Companion measurement:
# e5_amplification_eta01.py (scratchpad) directly measures the per-step amplification factor at
# eta=0.01 via perturb-and-propagate on the real upstream fori_loop-compiled inference_step -- a
# contractive map passing tight tolerance is EXPECTED regardless of residual port bugs being
# masked in principle; the defense against that is Tier B's independent 151/151 node-local
# gradient conformance PLUS this file's own strict per-step (not just endpoint) coverage: a
# genuinely wrong Julia formula would show up as a LARGE, roughly STEP-INVARIANT gap (not
# shrinking with the map's own contraction), unlike the compounding-noise signature this
# experiment was built to distinguish from a bug (see the register's full E1-E5 trail).
#
# Every allclose_dt comparison's ACTUAL max abs diff is also computed and reported (pass/fail
# counts at 1e-4/1e-5/1e-6/1e-7, plus the single WORST required tolerance across all compared
# arrays) via `report()`, in addition to the real @test gating below -- so a future regression
# both FAILS the suite and prints exactly how far off it is, not just pass/fail.
using NPZ
using Test
using FabricPC
using Zygote
using Random

const FIXDIR = joinpath(@__DIR__, "fixtures")
const FIX_SGD = npzread(joinpath(FIXDIR, "tier_d_transformer_stable_sgd.npz"))
const FIX_NC  = npzread(joinpath(FIXDIR, "tier_d_transformer_stable_normclip.npz"))

const B, S, V, E, H = 3, 8, 10, 8, 2
const ETA_INFER = 0.01f0
const INFER_STEPS = 12
const MAX_NORM = 1.0f0
const NC_EPS = 1.0f-8

row_dt(v) = reshape(Float32.(v), 1, :)

const TIER_DT_FULL_NODES = ("embed", "transformer_0", "skip_0", "output")

# ---------------------------------------------------------------------------------------
# Collect (label, julia_array, fixture_array) triples -- no @test, just data -- so we can
# evaluate pass/fail + "tolerance actually needed" at several thresholds after the fact
# without re-running the (cheap but nontrivial) 12-step relaxation multiple times.
# ---------------------------------------------------------------------------------------
function collect_state_pairs!(pairs, prefix::AbstractString, state, FIX)
    for name in TIER_DT_FULL_NODES
        ns = state.nodes[name]
        push!(pairs, ("$(prefix)_$(name)_z_latent", ns.z_latent, FIX["$(prefix)_$(name)_z_latent"]))
        push!(pairs, ("$(prefix)_$(name)_z_mu", ns.z_mu, FIX["$(prefix)_$(name)_z_mu"]))
        push!(pairs, ("$(prefix)_$(name)_error", ns.error, FIX["$(prefix)_$(name)_error"]))
        push!(pairs, ("$(prefix)_$(name)_energy", ns.energy, FIX["$(prefix)_$(name)_energy"]))
        if name != "output"
            push!(pairs, ("$(prefix)_$(name)_pre_activation", ns.pre_activation, FIX["$(prefix)_$(name)_pre_activation"]))
        end
        push!(pairs, ("$(prefix)_$(name)_latent_grad", ns.latent_grad, FIX["$(prefix)_$(name)_latent_grad"]))
    end
    ni = state.nodes["input"]
    push!(pairs, ("$(prefix)_input_z_latent", ni.z_latent, FIX["$(prefix)_input_z_latent"] .+ 1.0f0))
    expected_input_zmu = prefix == "init" ? zeros(Float32, size(ni.z_mu)) :
        (FIX["$(prefix)_input_z_latent"] .+ 1.0f0)
    push!(pairs, ("$(prefix)_input_z_mu", ni.z_mu, expected_input_zmu))
    push!(pairs, ("$(prefix)_input_error", ni.error, FIX["$(prefix)_input_error"]))
    push!(pairs, ("$(prefix)_input_energy", ni.energy, FIX["$(prefix)_input_energy"]))
    push!(pairs, ("$(prefix)_input_pre_activation", ni.pre_activation, FIX["$(prefix)_input_pre_activation"]))
    push!(pairs, ("$(prefix)_input_latent_grad", ni.latent_grad, FIX["$(prefix)_input_latent_grad"]))
    return pairs
end

function collect_params_pairs!(pairs, prefix::AbstractString, gp, FIX)
    push!(pairs, ("$(prefix)_embed_w__embeddings", gp.nodes["embed"].weights["embeddings"], FIX["$(prefix)_embed_w__embeddings"]))
    tb = gp.nodes["transformer_0"]
    for k in ("W_q", "W_k", "W_v", "W_o", "W_ff1", "W_ff2")
        push!(pairs, ("$(prefix)_transformer_0_w__$(k)", tb.weights[k], FIX["$(prefix)_transformer_0_w__$(k)"]))
    end
    for k in ("ln1_gamma", "ln2_gamma")
        push!(pairs, ("$(prefix)_transformer_0_w__$(k)", vec(tb.weights[k]), vec(FIX["$(prefix)_transformer_0_w__$(k)"])))
    end
    for k in ("b_q", "b_k", "b_v", "b_o", "b_ff1", "b_ff2", "ln1_beta", "ln2_beta")
        push!(pairs, ("$(prefix)_transformer_0_b__$(k)", vec(tb.biases[k]), vec(FIX["$(prefix)_transformer_0_b__$(k)"])))
    end
    out = gp.nodes["output"]
    push!(pairs, ("$(prefix)_output_w__W_out", out.weights["W_out"], FIX["$(prefix)_output_w__W_out"]))
    push!(pairs, ("$(prefix)_output_b__b_out", vec(out.biases["b_out"]), vec(FIX["$(prefix)_output_b__b_out"])))
    return pairs
end

allclose_dt(a, b; rtol=1.0f-4, atol=1.0f-4) = all(abs.(a .- b) .<= atol .+ rtol .* abs.(b))

"""Minimum t (used as both rtol AND atol, matching this track's convention) such that
`allclose_dt(a, b; rtol=t, atol=t)` passes: max elementwise |a-b| / (1 + |b|)."""
function required_tol(a, b)
    d = abs.(Float64.(a) .- Float64.(b))
    denom = 1.0 .+ abs.(Float64.(b))
    isempty(d) ? 0.0 : maximum(d ./ denom)
end

function build_params(FIX)
    GraphParams(
        Dict(
            "input" => NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()),
            "embed" => NodeParams(
                Dict("embeddings" => FIX["params0_embed_w__embeddings"]), Dict{String, Matrix{Float32}}()
            ),
            "transformer_0" => NodeParams(
                Dict(
                    "W_q" => FIX["params0_transformer_0_w__W_q"], "W_k" => FIX["params0_transformer_0_w__W_k"],
                    "W_v" => FIX["params0_transformer_0_w__W_v"], "W_o" => FIX["params0_transformer_0_w__W_o"],
                    "W_ff1" => FIX["params0_transformer_0_w__W_ff1"], "W_ff2" => FIX["params0_transformer_0_w__W_ff2"],
                    "ln1_gamma" => row_dt(FIX["params0_transformer_0_w__ln1_gamma"]),
                    "ln2_gamma" => row_dt(FIX["params0_transformer_0_w__ln2_gamma"])
                ),
                Dict(
                    "b_q" => row_dt(FIX["params0_transformer_0_b__b_q"]), "b_k" => row_dt(FIX["params0_transformer_0_b__b_k"]),
                    "b_v" => row_dt(FIX["params0_transformer_0_b__b_v"]), "b_o" => row_dt(FIX["params0_transformer_0_b__b_o"]),
                    "b_ff1" => row_dt(FIX["params0_transformer_0_b__b_ff1"]), "b_ff2" => row_dt(FIX["params0_transformer_0_b__b_ff2"]),
                    "ln1_beta" => row_dt(FIX["params0_transformer_0_b__ln1_beta"]), "ln2_beta" => row_dt(FIX["params0_transformer_0_b__ln2_beta"])
                )
            ),
            "skip_0" => NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()),
            "output" => NodeParams(
                Dict("W_out" => FIX["params0_output_w__W_out"]), Dict("b_out" => row_dt(FIX["params0_output_b__b_out"]))
            )
        )
    )
end

"""Run the full init -> relax01..12 -> converged -> grad -> train_step pipeline for one
`structure` (built with a specific eta_infer / inference algorithm), collect every comparable
(label, julia, fixture) pair against `FIX`."""
function run_pipeline(structure, FIX)
    pairs = Tuple{String, Any, Any}[]
    params = build_params(FIX)
    collect_params_pairs!(pairs, "params0", params, FIX)

    Xb = Float32.(FIX["batch_x"]) .+ 1.0f0
    Yb = FIX["batch_y"]
    clamps = Dict{String, Any}("input" => Xb, "output" => Yb)
    rng = Random.default_rng()

    init_state = initialize_graph_state(structure, B, rng; clamps=clamps, params=params)
    collect_state_pairs!(pairs, "init", init_state, FIX)

    relax_state = init_state
    for step in 1:INFER_STEPS
        relax_state = FabricPC.inference_step(params, relax_state, clamps, structure)
        collect_state_pairs!(pairs, "relax$(lpad(step, 2, '0'))", relax_state, FIX)
    end

    converged = run_inference(params, init_state, clamps, structure)
    collect_state_pairs!(pairs, "converged", converged, FIX)

    batch = Dict{String, Any}("x" => Xb, "y" => Yb)
    grads, energy, grad_final_state = get_graph_param_gradient(params, batch, structure, rng)
    collect_params_pairs!(pairs, "grad", grads, FIX)
    push!(pairs, ("scalars_energy_pretrain", [Float32(energy)], FIX["scalars_energy_pretrain"]))
    collect_state_pairs!(pairs, "grad_final", grad_final_state, FIX)

    LR = 0.05
    new_params, energy2, final_state2 = train_step(params, batch, structure, LR, rng)
    collect_params_pairs!(pairs, "trained", new_params, FIX)
    push!(pairs, ("scalars_energy_trainstep", [Float32(energy2)], FIX["scalars_energy_trainstep"]))
    collect_state_pairs!(pairs, "trained_final", final_state2, FIX)

    return pairs
end

function report(label, pairs)
    println("\n" * "="^100)
    println(label, " -- ", length(pairs), " comparisons")
    println("="^100)
    for tol in (1.0f-4, 1.0f-5, 1.0f-6, 1.0f-7)
        npass = count(p -> allclose_dt(p[2], p[3]; rtol=tol, atol=tol), pairs)
        println("  tol=$(tol): $(npass)/$(length(pairs)) pass")
    end
    worst = 0.0
    worst_label = "none"
    for (name, a, b) in pairs
        t = required_tol(a, b)
        if t > worst
            worst = t
            worst_label = name
        end
    end
    println("  WORST required tolerance across all $(length(pairs)) comparisons: $(worst)  (at $(worst_label))")
    # also show top-5 worst, for visibility into whether failures cluster (e.g. norm-clip
    # threshold-crossing) or spread evenly
    sorted = sort(pairs; by = p -> -required_tol(p[2], p[3]))
    println("  top 5 worst comparisons:")
    for (name, a, b) in sorted[1:min(5, end)]
        println("    $(name): required_tol=$(required_tol(a, b))")
    end
    return worst
end

println("Building structures...")
structure_sgd = transformer_lm(; seq_len=S, vocab_size=V, embed_dim=E, num_heads=H, num_blocks=1, eta_infer=ETA_INFER)

# Hand-built graph for InferenceSGDNormClip (transformer_lm() hardcodes plain InferenceSGD;
# no keyword to swap algorithms) -- mirrors transformer_lm()'s own topology/node-construction
# exactly (src/models/transformer_lm.jl:41-89), just with InferenceSGDNormClip(eta_infer=0.01,
# infer_steps=12, latent_decay=0.0, max_norm=1.0, eps=1e-8) in place of InferenceSGD. C-05's
# own test defaults (test/test_inference_normclip.jl) use max_norm=1.0/eps=1e-8; matched here.
input_nc = IdentityNode((S,), "input")
embed_nc = EmbeddingNode((S, E), "embed"; vocab_size=V, weight_init=NormalInitializer(; std=1.0))
block_nc = TransformerBlock((S, E), "transformer_0"; num_heads=H, ff_dim=nothing, use_rope=true, causal=true)
skip_nc = SkipConnection((S, E), "skip_0")
output_nc = VocabProjectionNode((S, V), "output"; weight_init=NormalInitializer(; std=sqrt(1.0 / E)))
nodes_nc = FabricPC.AbstractNode[input_nc, embed_nc, block_nc, skip_nc, output_nc]
edges_nc = Edge[
    Edge(input_nc, slot(embed_nc, "in")),
    Edge(embed_nc, slot(block_nc, "in")),
    Edge(embed_nc, slot(skip_nc, "in")),
    Edge(block_nc, slot(skip_nc, "in")),
    Edge(skip_nc, slot(output_nc, "in")),
]
inference_nc = InferenceSGDNormClip(;
    eta_infer=ETA_INFER, infer_steps=INFER_STEPS, latent_decay=0.0f0, max_norm=MAX_NORM, eps=NC_EPS
)
structure_nc = graph(nodes_nc, edges_nc, TaskMap(; x=input_nc, y=output_nc), inference_nc)

println("Running SGD (eta=0.01) pipeline vs tier_d_transformer_stable_sgd.npz ...")
pairs_sgd = run_pipeline(structure_sgd, FIX_SGD)
worst_sgd = report("InferenceSGD, eta_infer=0.01, 12 steps", pairs_sgd)

println("\nRunning InferenceSGDNormClip (eta=0.01, max_norm=1.0, eps=1e-8) pipeline vs tier_d_transformer_stable_normclip.npz ...")
pairs_nc = run_pipeline(structure_nc, FIX_NC)
worst_nc = report("InferenceSGDNormClip, eta_infer=0.01, 12 steps", pairs_nc)

println("\n" * "="^100)
println("SUMMARY")
println("="^100)
println("SGD variant:      $(count(p -> allclose_dt(p[2], p[3]), pairs_sgd))/$(length(pairs_sgd)) pass at 1e-4;  worst required tol = $(worst_sgd)")
println("NormClip variant: $(count(p -> allclose_dt(p[2], p[3]), pairs_nc))/$(length(pairs_nc)) pass at 1e-4;  worst required tol = $(worst_nc)")

# ---------------------------------------------------------------------------------------
# ACTUAL GATING: the report() calls above are diagnostics; these @tests are what makes a
# regression fail the suite. One named @test per (label, julia, fixture) triple, at the
# track's tolerance convention (rtol=atol=1f-4). TOLERANCE: looser than Tier C's 1e-5 --
# Float32 BLAS/XLA associativity across multiple chained inference steps, same convention
# every Tier D test in this repo uses; do not weaken further without a measured reason.
@testset "Tier D conformance (upstream JAX FabricPC, end-to-end, transformer-LM, eta=0.01 contractive regime, rtol/atol 1e-4)" begin
    @testset "InferenceSGD" begin
        for (name, a, b) in pairs_sgd
            @test allclose_dt(a, b)
        end
    end
    @testset "InferenceSGDNormClip (C-05 conformance coverage)" begin
        for (name, a, b) in pairs_nc
            @test allclose_dt(a, b)
        end
    end
end
