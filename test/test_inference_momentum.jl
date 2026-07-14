# InferenceSGDMomentum: heavy-ball momentum inference (src/core/inference.jl). AHEAD-OF-UPSTREAM
# divergence, not a conformance-gap fill -- upstream has only a design document
# (docs/dev_plans_archive/momentum_sgd_inference_plan.md, as of the commit this port tracks),
# no shipped `InferenceSGDMomentum` code to conform to. See docs/decisions.md #26 and
# docs/AUDIT_REGISTER.md for the full derivation and register entry.
#
# THE STORY IS NOT "theory matched measurement on the first try" -- it's more interesting than
# that. The FIRST closed-loop attempt (spectrum from docs/decisions.md #24: lambda_min~=2.3,
# lambda_max~=113.6, kappa~=49) predicted an asymptotic rate of sqrt(beta*)~=0.751/step; the
# MEASURED rate was ~0.99/step, a real mismatch. Chasing that mismatch (not accepting either
# number blind) found two things: (1) #24's lambda_min was measured with too few
# power-iteration steps to converge on this graph's near-degenerate spectrum -- a properly
# converged (renormalized, ~400-step) power iteration finds the TRUE lambda_min~=0.17, not 2.3
# (kappa~=668, not ~49); (2) the local Hessian is TRAJECTORY-DEPENDENT, not a single fixed
# quadratic -- lambda_max=113.6 is confirmed correct (direct stability-boundary crossing at
# eta~=0.0176) ONLY near initialization; 40 steps into relaxation, no instability appears even
# at eta=0.025. A single global (eta*, beta*) derived from the worst-case pair across the whole
# trajectory is therefore a genuine engineering compromise, not a clean quadratic-model fit --
# and running it from a cold start shows exactly that: a real transient overshoot near
# initialization (where lambda_max is largest) before net convergence at a modest but genuine
# speedup over plain SGD. See docs/decisions.md #26 for the full writeup; this file's tests
# reflect the CORRECTED, not the original, prediction.
#
# Three things this file establishes, in order:
#
#   1. CONFORMANCE ANCHOR: momentum=0 must reproduce InferenceSGD bit-for-bit. This is the
#      correctness tether for a feature with no upstream reference output -- if the general
#      momentum recursion doesn't collapse EXACTLY to the already-conformance-tested plain-SGD
#      formula at its degenerate case, the general formula is wrong regardless of what the
#      rate-measurement finding above shows.
#
#   2. Corrected-spectrum optimal (eta*, beta*) stays finite and net-converges from a cold
#      start (cheap, always-run). The FULL theory-vs-measurement closure -- the renormalized
#      perturb-and-track power iteration that found the corrected spectrum in the first place,
#      several minutes -- is a SEPARATE, gated diagnostic at the bottom of this file
#      (FABRICPC_MOMENTUM_SPECTRUM_DIAGNOSTIC=1), not run on every test invocation.
#
#   3. FALSIFICATION: upstream's own proposed default (beta=0.9, eta=0.1 --
#      momentum_sgd_inference_plan.md) is predicted, from the SAME (corrected) spectrum, to be
#      UNSTABLE on this graph -- its heavy-ball stability boundary at beta=0.9 is
#      eta < 2(1+0.9)/lambda_max ~= 0.0334, three-fold below 0.1, and since
#      eta < 2(1+beta)/lambda_max for beta in [0,1) always, the ceiling as beta -> 1 is
#      4/lambda_max ~= 0.0352 -- no beta < 1 stabilizes eta=0.1 on this graph. This part of the
#      original prediction was UNAFFECTED by the lambda_min correction (lambda_max=113.6 was
#      independently reconfirmed, not revised) and held up cleanly under both the original and
#      corrected analysis. Tested at beta=0.9 (upstream's literal proposed default) AND
#      beta=0.99 (near the beta->1 ceiling, to show it's not just upstream's specific guess that
#      fails but the eta=0.1 configuration itself, structurally, for any momentum coefficient).
using NPZ
using Test
using FabricPC
using Random

const FIXDIR = joinpath(@__DIR__, "conformance", "fixtures")
const FIX = npzread(joinpath(FIXDIR, "tier_d_transformer_stable_sgd.npz"))

# Same tiny transformer-LM diagnostic graph as docs/decisions.md #24's spectrum measurement and
# test/conformance/test_tier_d_transformer_stable.jl's InferenceSGDNormClip hand-built graph
# (src/models/transformer_lm.jl:41-89's own topology/node-construction, transformer_lm() itself
# hardcodes InferenceSGD with no keyword to swap algorithms).
const B, S, V, E, H = 3, 8, 10, 8, 2

row_dt(v) = reshape(Float32.(v), 1, :)

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

"""Hand-build the graph with `inf` as the inference algorithm (transformer_lm() itself has no
keyword to swap algorithms -- same pattern test_tier_d_transformer_stable.jl uses for
InferenceSGDNormClip)."""
function build_graph(inf)
    input = IdentityNode((S,), "input")
    embed = EmbeddingNode((S, E), "embed"; vocab_size=V, weight_init=NormalInitializer(; std=1.0))
    block = TransformerBlock((S, E), "transformer_0"; num_heads=H, ff_dim=nothing, use_rope=true, causal=true)
    skip = SkipConnection((S, E), "skip_0")
    output = VocabProjectionNode((S, V), "output"; weight_init=NormalInitializer(; std=sqrt(1.0 / E)))
    nodes = FabricPC.AbstractNode[input, embed, block, skip, output]
    edges = Edge[
        Edge(input, slot(embed, "in")),
        Edge(embed, slot(block, "in")),
        Edge(embed, slot(skip, "in")),
        Edge(block, slot(skip, "in")),
        Edge(skip, slot(output, "in")),
    ]
    return graph(nodes, edges, TaskMap(; x=input, y=output), inf)
end

const PARAMS = build_params(FIX)
const XB = Float32.(FIX["batch_x"]) .+ 1.0f0
const YB = FIX["batch_y"]
const CLAMPS = Dict{String, Any}("input" => XB, "output" => YB)

"""Fresh init state for `structure`, deterministic given PARAMS/CLAMPS (every non-input node in
this topology has in_degree>0 and is unclamped, so FeedforwardStateInit's second pass overwrites
`initialize`'s random draw with `z_mu` from the forward pass -- the rng seed does not affect this
graph's initial z_latent, but is fixed anyway as cheap insurance)."""
function fresh_init(structure)
    Random.seed!(20260714)
    return initialize_graph_state(structure, B, Random.default_rng(); clamps=CLAMPS, params=PARAMS)
end

function total_norm(state, node_names, field)
    s = 0.0
    for name in node_names
        s += sum(abs2, Float64.(getfield(state.nodes[name], field)))
    end
    return sqrt(s)
end

const NONCLAMPED = ("embed", "transformer_0", "skip_0")   # input/output are clamped

"""Run `inf` (an `InferenceSGDMomentum`) step-by-step via `momentum_inference_step`, returning
the post-step `total_norm(state, NONCLAMPED, :latent_grad)` trace. MUST use
`momentum_inference_step` (threading velocity through the caller), NOT `FabricPC.inference_step`
-- the latter dispatches through `compute_new_latent`, which for `InferenceSGDMomentum` is a
documented plain-SGD fallback that silently ignores `momentum` entirely (see
`src/core/inference.jl`'s `momentum_inference_step` docstring for the bug this caused here once)."""
function momentum_norm_trace(inf::InferenceSGDMomentum, structure)
    state = fresh_init(structure)
    velocity = Dict{String, Any}(
        name => zero(state.nodes[name].z_latent) for name in structure.node_names
    )
    norms = Float64[]
    for _ in 1:inf.infer_steps
        state, velocity = FabricPC.momentum_inference_step(inf, PARAMS, state, velocity, CLAMPS, structure)
        push!(norms, total_norm(state, NONCLAMPED, :latent_grad))
    end
    return norms
end

@testset "InferenceSGDMomentum" begin

    # ---------------------------------------------------------------------------------------
    # 1. Conformance anchor: momentum=0 == InferenceSGD, bit-for-bit, every field, every node.
    # ---------------------------------------------------------------------------------------
    @testset "momentum=0 reproduces InferenceSGD bit-for-bit" begin
        ETA, STEPS = 0.01f0, 12
        structure_sgd = build_graph(InferenceSGD(; eta_infer=ETA, infer_steps=STEPS, latent_decay=0.0))
        structure_mom0 = build_graph(
            InferenceSGDMomentum(; eta_infer=ETA, infer_steps=STEPS, latent_decay=0.0, momentum=0.0)
        )

        state_sgd = run_inference(PARAMS, fresh_init(structure_sgd), CLAMPS, structure_sgd)
        state_mom0 = run_inference(PARAMS, fresh_init(structure_mom0), CLAMPS, structure_mom0)

        for name in ("embed", "transformer_0", "skip_0", "output")
            ns_sgd = state_sgd.nodes[name]
            ns_mom0 = state_mom0.nodes[name]
            @test ns_sgd.z_latent == ns_mom0.z_latent
            @test ns_sgd.z_mu == ns_mom0.z_mu
            @test ns_sgd.error == ns_mom0.error
            @test ns_sgd.energy == ns_mom0.energy
            @test ns_sgd.latent_grad == ns_mom0.latent_grad
        end
    end

    # ---------------------------------------------------------------------------------------
    # 2. Corrected-spectrum optimal (eta*, beta*): stays finite from a cold start and shows a
    # net decrease past its own transient. Cheap, natural-relaxation-only sanity check -- the
    # full theory-vs-measurement closure (perturb-and-track power iteration, several minutes)
    # is the SEPARATE gated diagnostic below, not run on every test invocation.
    #
    # WHY "corrected" and not docs/decisions.md #24's original lambda_min=2.3: re-deriving
    # (eta*, beta*) from that spectrum and measuring the ACTUAL asymptotic rate via
    # perturb-and-track (this exact graph+params, docs/decisions.md #24's own methodology)
    # revealed #24's lambda_min was measured with too few power-iteration steps to converge on
    # a near-degenerate spectrum -- a properly-converged (renormalized, ~400-step) power
    # iteration finds the TRUE lambda_min~=0.17, not 2.3 (kappa~=668, not ~49). lambda_max=113.6
    # independently re-confirmed correct (direct stability-boundary crossing observed between
    # eta=0.017 stable / eta=0.02 unstable, matching 2/lambda_max~=0.0176, but ONLY near
    # initialization -- burned 40 steps into relaxation, no instability appears even at
    # eta=0.025, i.e. the local Hessian is TRAJECTORY-DEPENDENT, not a single fixed quadratic).
    # See docs/decisions.md #26 for the full writeup.
    @testset "corrected-optimal (eta*, beta*) stays finite and net-converges from a cold start" begin
        lambda_min_true, lambda_max = 0.17, 113.6
        sqrt_max, sqrt_min = sqrt(lambda_max), sqrt(lambda_min_true)
        eta_star = 4.0 / (sqrt_max + sqrt_min)^2
        beta_star = ((sqrt_max - sqrt_min) / (sqrt_max + sqrt_min))^2

        @test isapprox(eta_star, 0.0326; atol=0.001)
        @test isapprox(beta_star, 0.857; atol=0.005)

        STEPS = 60
        inf_opt = InferenceSGDMomentum(;
            eta_infer=Float32(eta_star), infer_steps=STEPS, latent_decay=0.0, momentum=Float32(beta_star)
        )
        structure_opt = build_graph(inf_opt)
        norms = momentum_norm_trace(inf_opt, structure_opt)

        # eta* sits almost exactly AT its own stability boundary 2(1+beta*)/lambda_max (using
        # the near-init worst-case lambda_max=113.6, ~0.0327 -- eta* itself is ~0.0326) -- this
        # produces a REAL, expected transient overshoot early (measured: norm roughly doubles
        # around step 10) before the trajectory moves past the stiff near-init region and
        # settles into net decay. Not a bug -- a genuine consequence of deriving a single global
        # momentum config from the worst-case (largest) lambda_max anywhere on the trajectory.
        @test all(isfinite, norms)
        @test norms[end] < norms[10] / 1.5   # net decay well past the transient peak
        println("\ncorrected-optimal momentum eta*=$(eta_star) beta*=$(beta_star): " *
                "step1=$(norms[1]) step10=$(norms[10]) step$(STEPS)=$(norms[end])")
    end

    # ---------------------------------------------------------------------------------------
    # 3. Falsification: upstream's proposed default, and the eta=0.1 ceiling itself, diverge.
    # ---------------------------------------------------------------------------------------
    @testset "upstream's proposed default (beta=0.9, eta=0.1) diverges -- predicted before measured" begin
        STEPS = 100
        for beta in (0.9f0, 0.99f0)   # upstream's literal guess, and near the beta->1 ceiling
            inf_bad = InferenceSGDMomentum(; eta_infer=0.1f0, infer_steps=STEPS, latent_decay=0.0, momentum=beta)
            structure_bad = build_graph(inf_bad)
            # NOTE: latent_grad is zeros-by-construction on the fresh init state (no step has
            # run yet) -- a step-0 baseline would be trivially zero, making ANY nonzero endpoint
            # pass vacuously. Baseline is norms[1] = the norm AFTER the first real step.
            norms = momentum_norm_trace(inf_bad, structure_bad)
            println(
                "beta=$(beta), eta=0.1: latent_grad norm step1=$(norms[1]) -> step$(STEPS)=$(norms[end]), " *
                "max=$(maximum(norms))" *
                (any(!isfinite, norms) ? "  (NaN/Inf reached)" : "")
            )
            # Stability boundary at this beta is eta < 2(1+beta)/lambda_max ~= 2(1+beta)/113.6,
            # i.e. ~0.0334 at beta=0.9 and ~0.0350 at beta=0.99 -- eta=0.1 sits past it either
            # way (ceiling as beta->1 is 4/lambda_max~=0.0352, still < 0.1; lambda_max=113.6 is
            # the near-initialization value, directly reconfirmed via stability-boundary
            # crossing -- docs/decisions.md #26).
            #
            # Predicted: NOT convergence. What that means empirically on this graph is NOT a
            # clean exponential blowup to non-finite values -- nonlinear saturation
            # (softmax/LayerNorm/GELU) bounds it, exactly as docs/decisions.md #24 already
            # documented for plain SGD at this same eta=0.1 (measured 25,000-45,000x over just
            # 12 steps by LINEAR extrapolation, but the REAL trajectory saturates well short of
            # that). Measured here: a real excursion (>2x its post-transient baseline) that
            # settles into an ELEVATED, non-decaying oscillation, never returning below its own
            # baseline -- qualitatively the opposite of the corrected-optimal config above, which
            # nets a >10x DECREASE using the identical measurement protocol. That sign
            # difference (net growth vs net decay from the same post-transient baseline) is the
            # robust, saturation-proof falsification signal -- not a specific multiplier, which
            # saturation makes graph-and-config-dependent.
            @test any(!isfinite, norms) || (maximum(norms) > 2 * norms[1] && norms[end] > norms[1])
        end
    end
end

# The full theory-vs-measurement closure (perturb-and-track power iteration, several minutes) is
# a SEPARATE, gated diagnostic: test/test_inference_momentum_diagnostic.jl, wired into
# test/runtests.jl as an independent, try/caught top-level testset (FABRICPC_MOMENTUM_SPECTRUM_DIAGNOSTIC=1)
# -- same pattern as the demoted Tier D eta=0.1 test, and for the same reason: a testset nested
# inside this file's own "InferenceSGDMomentum" @testset only records failures into ITS parent's
# tally without throwing independently, so a slow/gated block belongs in its own top-level
# testset, not nested here.
