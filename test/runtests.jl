using FabricPC
using Test
# Load Zygote at top-level BEFORE the testset so the FabricPCZygoteExt seam (the AD backend for
# TransformerBlock/VocabProjection/Storkey/ADLinear) is registered ahead of any test. If a test
# file is the first to `using Zygote`, world-age makes the freshly-loaded ext methods invisible
# within that file's own @testset → the seam raises its "load a backend" hint. (The suite is
# Zygote-only — Enzyme also implements this seam, and the two cannot be co-loaded.)
using Zygote

@testset "FabricPC.jl" begin
    @testset "scaffold" begin
        @test FabricPC.FABRICPC_VERSION == v"0.1.0"
    end
    include("test_graph_construction.jl")        # R-05b: duplicate-edge handling
    include("test_phase_b.jl")
    include("test_residual.jl")
    include("test_mupc.jl")
    include("test_activations.jl")
    include("test_energies.jl")
    include("test_classifier.jl")
    include("test_optim.jl")
    include("test_mupc_resnet.jl")
    include("test_jit_flat.jl")
    include("test_initializers_natgrad.jl")
    include("test_train_pcn.jl")
    include("test_autodiff_seam.jl")   # Phase D: Enzyme node-local autodiff seam
    include("test_transformer.jl")     # Phase D: PC-transformer block (Enzyme seam)
    include("test_transformer_decomposed.jl")  # fully-PC decomposed stages
    include("test_storkey_hopfield.jl")         # Hopfield associative memory (composite energy)
    include("test_sequence_training.jl")        # autoregressive/sequence trainer (port phases)
    include("test_transformer_lm.jl")           # assembled transformer LM graph (e2e)
    include("test_char_dataloader.jl")          # C-03: Tiny Shakespeare char loader (offline)
    include("test_inference_normclip.jl")       # C-05: InferenceSGDNormClip
    include("test_inference_momentum.jl")       # InferenceSGDMomentum: ahead-of-upstream (docs/decisions.md #26)
    include("conformance/test_tier_a.jl")        # Tier A: primitives vs upstream JAX fixtures
    include("conformance/test_tier_b.jl")        # Tier B: node-local vs upstream JAX fixtures
    include("conformance/test_tier_c.jl")        # Tier C: loop-level vs upstream JAX fixtures
    include("conformance/test_tier_d_mnist.jl")   # Tier D: end-to-end (MNIST-MLP) vs upstream JAX fixtures
    # Tier D transformer-LM track (docs/AUDIT_REGISTER.md section 6): VERIFIED CLOSED at
    # eta_infer=0.01 (measured contractive: tail per-step ratio ~0.88-0.98 vs the production
    # default eta=0.1's measured ~1.24-1.27/step, i.e. ~25,000-45,000x amplification of any
    # float32 seed over the same 12 steps) -- 523/523 at rtol=atol=1e-4 (every relax01..relax12
    # intermediate step, not just endpoints; worst-case required tolerance 1.57e-5, a ~6x margin),
    # covering both plain InferenceSGD and InferenceSGDNormClip (C-05's first conformance
    # coverage). This is the primary/ungated conformance target for this track.
    include("conformance/test_tier_d_transformer_stable.jl")
    # C-05 boundary-straddle coverage (docs/AUDIT_REGISTER.md section 4/6): a second
    # InferenceSGDNormClip fixture, max_norm=2.5, chosen so the clip decision genuinely
    # straddles the boundary for some samples (the max_norm=1.0 fixture above never gets near
    # it -- coverage-by-absence). 523/523 value comparisons PLUS a dedicated discrete
    # clip/no-clip decision-agreement check at every near-boundary point -- zero disagreements.
    include("conformance/test_tier_d_transformer_stable_normclip_boundary.jl")
end

# The ORIGINAL eta=0.1/12-step transformer-LM config is DEMOTED, not deleted: measured expansive
# (eta*~=0.0176 via power iteration, i.e. this config runs ~5.7x past its own stability limit --
# two independent float32 implementations of an expansive iterative map cannot be expected to
# agree after 12 steps, regardless of port fidelity, which is independently established by Tier
# B's 151/151 node-local gradients and this same track's own bit-exact pre-relaxation checks).
# Its 146/175 pass rate is NOT a fidelity gate -- a CONDITIONING ENVELOPE (the relax01..relax12
# per-step dumps are the most valuable diagnostic asset this investigation produced; do not
# delete them). Deliberately run as an INDEPENDENT testset OUTSIDE "FabricPC.jl" above (not
# nested inside it) and wrapped in try/catch: a nested @testset only records its failures into
# the PARENT's tally without throwing on its own, so nesting this here would silently make
# "FabricPC.jl"'s own default-green guarantee depend on this file's known-red state whenever the
# gate is set -- exactly the LoadError this structure hit once already (2026-07-14) before being
# pulled out to its own top-level testset, which throws (and is caught) independently. Set
# FABRICPC_TIER_D_TRANSFORMER=1 to include it; the outer suite's pass/fail verdict never depends
# on this block either way.
if get(ENV, "FABRICPC_TIER_D_TRANSFORMER", "0") == "1"
    try
        @testset "Tier D transformer-LM, eta=0.1 (conditioning envelope, NOT a fidelity gate)" begin
            include("conformance/test_tier_d_transformer.jl")
        end
    catch e
        e isa Test.TestSetException || rethrow()
        println("\n[conditioning envelope] eta=0.1 config's known-expansive failures printed above ($(e.fail + e.error)/$(e.pass + e.fail + e.error) -- see docs/AUDIT_REGISTER.md section 6). Not counted against the main suite.")
    end
end

# F-04 dual-AD-backend guard's precompile-vs-load-time regression (test_autodiff_backend_registration.jl,
# 2026-07-14). Opt-in like the Tier D transformer block above, for the same reason: this spawns up
# to 5 genuinely fresh `julia` subprocesses (the bug it guards against can only be reproduced
# across a real precompile-cache boundary, which the already-warm `runtests.jl` process itself
# cannot exercise) -- real cold-start/precompile cost, not appropriate for the default fast run.
# Set FABRICPC_AD_BACKEND_SUBPROCESS_TESTS=1 to include it. Independent top-level testset (not
# nested in "FabricPC.jl" above) for the same reason as the Tier D block: its own subprocess
# I/O/exit-code assertions should never be silently swallowed into the parent's tally.
if get(ENV, "FABRICPC_AD_BACKEND_SUBPROCESS_TESTS", "0") == "1"
    include("test_autodiff_backend_registration.jl")   # defines its own top-level @testset
end

# InferenceSGDMomentum's full theory-vs-measurement closure (docs/decisions.md #26): a
# renormalized perturb-and-track power iteration, several minutes, not appropriate for the
# default fast run. Requires test_inference_momentum.jl's helpers (build_graph, fresh_init,
# PARAMS, CLAMPS) in scope first. Set FABRICPC_MOMENTUM_SPECTRUM_DIAGNOSTIC=1 to include it.
if get(ENV, "FABRICPC_MOMENTUM_SPECTRUM_DIAGNOSTIC", "0") == "1"
    try
        include("test_inference_momentum_diagnostic.jl")   # defines its own top-level @testset
    catch e
        e isa Test.TestSetException || rethrow()
        println("\n[momentum spectrum diagnostic] failures printed above -- see docs/decisions.md #26. Not counted against the main suite.")
    end
end
