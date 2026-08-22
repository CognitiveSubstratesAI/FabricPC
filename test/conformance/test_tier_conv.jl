# Tier-conv conformance (C-01): ConvNode / MaxPool / AvgPool vs upstream JAX.
#
# Fixtures: scripts/generate_tier_conv_fixtures.py -> fixtures/tier_conv.npz (jax 0.10.2).
# This exercises the NODE API through the autodiff seam — forward / forward_and_weight_grads /
# forward_and_latent_grads — which is the integration the standalone kernel probes could not
# reach. Both sides are autodiff (upstream conv/pool implement only `forward` and inherit
# NodeBase's jax.value_and_grad; ours implement only `compute_mu` and inherit the Enzyme/Zygote
# seam), so this compares AD-to-AD across two different stacks.
#
# RNG TRAP (as Tier A/B): params are built DIRECTLY from the fixture arrays. `initialize_params`
# is never called for a compared value — different RNG family entirely.
#
# TOLERANCES, derived BEFORE the kernels ran and both confirmed:
#   conv rtol 1e-5 — our im2col GEMM and XLA's fused conv reassociate differently; measured floor
#                    ~1e-7. A bit-identity gate here would fail on CORRECT code (decisions.md,
#                    the C-01 fold-order note).
#   pool rtol 1e-6 — a pure reduce_window with no GEMM to reassociate; measured BIT-IDENTICAL.
using Test
using FabricPC
using FabricPC: forward, forward_and_weight_grads, forward_and_latent_grads
using FabricPC: to_flat_params, CompiledPlan   # audit regressions: flat-lane scope gate
# OrderedDict via FabricPC (it depends on OrderedCollections; the test target does not, and this
# suite must not need a dep the package already provides). The multi-edge testset NEEDS an ordered
# map: `gather_inputs` produces one, and in_edges order is what the fold must follow.
using FabricPC: OrderedDict
using NPZ
using Random

const FIX_C = npzread(joinpath(@__DIR__, "fixtures", "tier_conv.npz"))
allclose_c(a, b; rtol=1.0f-5, atol=1.0f-5) = all(abs.(a .- b) .<= atol .+ rtol .* abs.(b))

_st(z, shape) = NodeState(
    z, zeros(Float32, size(z)), zeros(Float32, size(z)),
    zeros(Float32, size(z, 1)), zeros(Float32, size(z))
)
_info(name, shape, edges) = NodeInfo(
    name, shape, "conv", Dict{String, SlotInfo}(), length(edges), 1, collect(edges),
    String[], nothing
)

@testset "Tier conv conformance (C-01: ConvNode/MaxPool/AvgPool vs upstream JAX)" begin

    @testset "conv: overlap-heavy backward (3x2 ASYMMETRIC kernel, SAME, stride 1)" begin
        # The case the oracle exists for: every interior pixel takes 9 contributions in the dE/dx
        # scatter. The kernel is non-square so a patch-order transposition cannot hide.
        e = "src->conv:in"
        x, W = FIX_C["overlap_in_x"], FIX_C["overlap_in_W"]
        b, z = FIX_C["overlap_in_b"], FIX_C["overlap_in_z_latent"]
        node = ConvNode((6, 5, 3), "conv", (3, 2); stride=(1, 1), padding="SAME",
            activation=IdentityActivation(), energy=GaussianEnergy())
        params = NodeParams(Dict(e => W), Dict("b" => b))
        inputs = Dict{String, Any}(e => x)
        st, info = _st(z, (6, 5, 3)), _info("conv", (6, 5, 3), [e])

        ns = forward(node, params, inputs, st)
        @test allclose_c(ns.z_mu, FIX_C["overlap_fwd_z_mu"])
        @test allclose_c(ns.error, FIX_C["overlap_fwd_error"])
        @test allclose_c(ns.energy, FIX_C["overlap_fwd_energy"])

        _, gp = forward_and_weight_grads(node, params, inputs, st)
        @test allclose_c(gp.weights[e], FIX_C["overlap_gradw_gW_0"])
        @test allclose_c(gp.biases["b"], FIX_C["overlap_gradw_gb"])

        _, ig, sg = forward_and_latent_grads(node, params, inputs, st, info, true)
        @test allclose_c(ig[e], FIX_C["overlap_gradz_gx_0"])   # the col2im scatter
        @test allclose_c(sg, FIX_C["overlap_gradz_self_grad"])
    end

    @testset "conv: 3 in-edges — fold order is in_edges, in EVERY lane" begin
        # Edge names are hash-order-hostile on purpose. forward folds in in_edges order; the
        # differentiated lanes must too (the `_concrete_inputs` OrderedDict fix, df1d470). An
        # rtol test passes either way for 2 edges; 3 with these names does not.
        edges = split(String(UInt8.(FIX_C["multi_edge_order_ascii"])), "|")
        @test edges == ["zzz->c:in", "aaa->c:in", "mmm->c:in"]     # the order the oracle used
        Ws = Dict(e => FIX_C["multi_in_W_$(i-1)"] for (i, e) in enumerate(edges))
        xs = Dict{String, Any}(
            e => FIX_C["multi_in_x_$(i-1)"] for (i, e) in enumerate(edges)
        )
        z = FIX_C["multi_in_z_latent"]
        node = ConvNode((5, 4, 3), "c", (3, 2); stride=(1, 1), padding="SAME",
            activation=IdentityActivation(), energy=GaussianEnergy())
        params = NodeParams(Ws, Dict("b" => FIX_C["multi_in_b"]))
        # inputs must ITERATE in in_edges order — that is what gather_inputs produces.
        inputs = OrderedDict{String, Any}(e => xs[e] for e in edges)
        st, info = _st(z, (5, 4, 3)), _info("c", (5, 4, 3), edges)

        ns = forward(node, params, inputs, st)
        @test allclose_c(ns.z_mu, FIX_C["multi_fwd_z_mu"])
        _, gp = forward_and_weight_grads(node, params, inputs, st)
        for (i, e) in enumerate(edges)
            @test allclose_c(gp.weights[e], FIX_C["multi_gradw_gW_$(i-1)"])
        end
        _, ig, sg = forward_and_latent_grads(node, params, inputs, st, info, true)
        for (i, e) in enumerate(edges)
            @test allclose_c(ig[e], FIX_C["multi_gradz_gx_$(i-1)"])
        end
        @test allclose_c(sg, FIX_C["multi_gradz_self_grad"])
    end

    @testset "conv: ReLU + strided + VALID — activation is genuinely exercised" begin
        # Was "pre_activation != z_mu (B5)", pinning the compute_pre_and_mu seam hook. Upstream
        # b6f64ad removed pre_activation from NodeState (it is consumed inside one forward and
        # never stored), so that hook and the quantity it reported are both gone — there is no
        # longer a pre to compare against, on either stack.
        #
        # What the testset was FOR survives and is kept: this fixture is the strided/VALID/ReLU
        # conformance case, and the non-vacuity guard below still proves the ReLU actually bites
        # here (z_mu has exact zeros where the convolution went negative) rather than the testset
        # passing on an all-positive fixture where ReLU would be an identity no-op.
        e = "s->c:in"
        x, W, b, z = FIX_C["relu_valid_in_x"], FIX_C["relu_valid_in_W"],
        FIX_C["relu_valid_in_b"], FIX_C["relu_valid_in_z_latent"]
        shape = size(z)[2:end]
        node = ConvNode(shape, "c", (3, 2); stride=(2, 1), padding="VALID",
            activation=ReLUActivation(), energy=GaussianEnergy())
        params = NodeParams(Dict(e => W), Dict("b" => b))
        inputs = Dict{String, Any}(e => x)
        st, info = _st(z, shape), _info("c", shape, [e])

        ns = forward(node, params, inputs, st)
        @test allclose_c(ns.z_mu, FIX_C["relu_valid_fwd_z_mu"])
        # Non-vacuous: ReLU clipped real negatives here (exact zeros), and clipped nothing below 0.
        @test any(ns.z_mu .== 0) && all(ns.z_mu .>= 0)
        _, gp = forward_and_weight_grads(node, params, inputs, st)
        @test allclose_c(gp.weights[e], FIX_C["relu_valid_gradw_gW_0"])
        _, ig, _ = forward_and_latent_grads(node, params, inputs, st, info, true)
        @test allclose_c(ig[e], FIX_C["relu_valid_gradz_gx_0"])
    end

    @testset "conv: rank dispatch — 1D (NLC) and 3D (NDHWC)" begin
        for (tag, ks, st_, pad, shp) in (
            ("conv1d", (3,), (2,), "SAME", size(FIX_C["conv1d_in_z_latent"])[2:end]),
            (
                "conv3d",
                (2, 3, 1),
                (1, 1, 1),
                "SAME",
                size(FIX_C["conv3d_in_z_latent"])[2:end]
            )
        )
            e = tag == "conv1d" ? "s->c1:in" : "s->c3:in"
            node = ConvNode(shp, "c", ks; stride=st_, padding=pad,
                activation=IdentityActivation(), energy=GaussianEnergy())
            params = NodeParams(
                Dict(e => FIX_C["$(tag)_in_W"]), Dict("b" => FIX_C["$(tag)_in_b"])
            )
            inputs = Dict{String, Any}(e => FIX_C["$(tag)_in_x"])
            z = FIX_C["$(tag)_in_z_latent"]
            state, info = _st(z, shp), _info("c", shp, [e])
            ns = forward(node, params, inputs, state)
            @test allclose_c(ns.z_mu, FIX_C["$(tag)_fwd_z_mu"])
            _, gp = forward_and_weight_grads(node, params, inputs, state)
            @test allclose_c(gp.weights[e], FIX_C["$(tag)_gradw_gW_0"])
            _, ig, _ = forward_and_latent_grads(node, params, inputs, state, info, true)
            @test allclose_c(ig[e], FIX_C["$(tag)_gradz_gx_0"])
        end
    end

    # ---- pooling: rtol 1e-6 (no GEMM ⇒ measured bit-identical) --------------------------------
    @testset "pool: MaxPool + AvgPool" begin
        e = "s->p:in"
        x, z = FIX_C["pool_in_x"], FIX_C["pool_in_z_latent"]
        st, info = _st(z, (2, 2, 2)), _info("p", (2, 2, 2), [e])
        for (tag, node) in (
            ("maxpool", MaxPool((2, 2, 2), "p", (2, 2); stride=(2, 2), padding="VALID")),
            (
                "avgpool",
                AvgPool(
                    (2, 2, 2), "p"; window_shape=(2, 2), stride=(2, 2), padding="VALID"
                )
            )
        )
            params = NodeParams(
                Dict{String, Array{Float32}}(), Dict{String, Array{Float32}}()
            )
            inputs = Dict{String, Any}(e => x)
            ns = forward(node, params, inputs, st)
            @test allclose_c(ns.z_mu, FIX_C["$(tag)_fwd_z_mu"]; rtol=1.0f-6, atol=1.0f-6)
            _, ig, sg = forward_and_latent_grads(node, params, inputs, st, info, true)
            @test allclose_c(ig[e], FIX_C["$(tag)_gradz_gx_0"]; rtol=1.0f-6, atol=1.0f-6)
            @test allclose_c(sg, FIX_C["$(tag)_gradz_self_grad"]; rtol=1.0f-6, atol=1.0f-6)
        end
    end

    @testset "pool: MaxPool PARTIAL-tie routing (row-major, measured — NOT reasoned)" begin
        # THE discriminating fixture. An all-tied window passes under BOTH tie-break orders (both
        # start at (0,0)); this one ties (0,1) with (1,0) and (0,0) is strictly smaller, so
        # row-major (JAX, measured) and column-major (our patch layout) MUST disagree.
        e = "s->p:in"
        xp, zp = FIX_C["maxpool_ptie_in_x"], FIX_C["maxpool_ptie_in_z_latent"]
        node = MaxPool((2, 2, 2), "p", (2, 2); stride=(2, 2), padding="VALID")
        params = NodeParams(Dict{String, Array{Float32}}(), Dict{String, Array{Float32}}())
        inputs = Dict{String, Any}(e => xp)
        st, info = _st(zp, (2, 2, 2)), _info("p", (2, 2, 2), [e])
        ns = forward(node, params, inputs, st)
        @test allclose_c(ns.z_mu, FIX_C["maxpool_ptie_fwd_z_mu"]; rtol=1.0f-6, atol=1.0f-6)
        _, ig, _ = forward_and_latent_grads(node, params, inputs, st, info, true)
        @test allclose_c(ig[e], FIX_C["maxpool_ptie_gradz_gx_0"]; rtol=1.0f-6, atol=1.0f-6)
        # Assert on WHICH CELLS, not just the norm — a norm can match while routing differs.
        @test findall(!iszero, ig[e]) == findall(!iszero, FIX_C["maxpool_ptie_gradz_gx_0"])
    end

    @testset "pool: AvgPool count_include_pad — both modes, and NON-VACUOUSLY" begin
        # SAME over an odd extent ⇒ the last window is half-padded ⇒ the two modes MUST differ.
        # If they agreed, padding wouldn't be reaching a window and the test would prove nothing.
        e = "s->p:in"
        xo, zo = FIX_C["avgpool_cip_in_x"], FIX_C["avgpool_cip_in_z_latent"]
        st, info = _st(zo, (2, 2, 2)), _info("p", (2, 2, 2), [e])
        mus = Dict{Bool, Any}()
        for flag in (true, false)
            node = AvgPool((2, 2, 2), "p"; window_shape=(2, 2), stride=(2, 2),
                padding="SAME",
                count_include_pad=flag)
            params = NodeParams(
                Dict{String, Array{Float32}}(), Dict{String, Array{Float32}}()
            )
            inputs = Dict{String, Any}(e => xo)
            ns = forward(node, params, inputs, st)
            @test allclose_c(
                ns.z_mu,
                FIX_C["avgpool_cip_$(Int(flag))_fwd_z_mu"];
                rtol=1.0f-6,
                atol=1.0f-6
            )
            _, ig, _ = forward_and_latent_grads(node, params, inputs, st, info, true)
            @test allclose_c(
                ig[e],
                FIX_C["avgpool_cip_$(Int(flag))_gradz_gx_0"];
                rtol=1.0f-6,
                atol=1.0f-6
            )
            mus[flag] = ns.z_mu
        end
        @test maximum(abs.(mus[true] .- mus[false])) > 1.0f-3    # the divisor IS exercised
    end

    @testset "pool/conv: PARAM-FREE weight-grads (the shape C-14's probe could not fail on)" begin
        # The audit's find: the pool testsets called only forward / forward_and_latent_grads, so
        # the param-free weight-grad lane had NO test in either backend — and C-14 was closed on
        # a probe using ConvNode WITH bias, the one shape that cannot produce an empty NamedTuple.
        # An empty NamedTuple is a GHOST type: Enzyme rejects Duplicated on it outright, so every
        # PoolNode and every ConvNode(use_bias=false) crashed the Enzyme weight-grad lane, and
        # core/learning.jl calls this for every in_degree>0 node ⇒ no graph with a pool could
        # train on Enzyme. Zygote is the backend this suite runs (F-04 forbids co-loading), so
        # these assert the CONTRACT holds here; the Enzyme lane is covered by
        # test/dual_ad_backend_env (see the register's C-14).
        e = "s->p:in"
        x = FIX_C["pool_in_x"]
        z = FIX_C["pool_in_z_latent"]
        st = _st(z, (2, 2, 2))
        empty_p() =
            NodeParams(Dict{String, Array{Float32}}(), Dict{String, Array{Float32}}())
        for (tag, node) in (
            ("maxpool", MaxPool((2, 2, 2), "p", (2, 2); stride=(2, 2), padding="VALID")),
            (
                "avgpool",
                AvgPool(
                    (2, 2, 2), "p"; window_shape=(2, 2), stride=(2, 2), padding="VALID"
                )
            )
        )
            _, gp = forward_and_weight_grads(node, empty_p(), Dict{String, Any}(e => x), st)
            @test isempty(gp.weights.nt)      # param-free ⇒ empty gradient, not an error
            @test isempty(gp.biases.nt)
        end
        # ConvNode(use_bias=false): weights present, biases EMPTY — the other ghost shape.
        ec = "s->c:in"
        W = FIX_C["overlap_in_W"]
        xc, zc = FIX_C["overlap_in_x"], FIX_C["overlap_in_z_latent"]
        nb = ConvNode((6, 5, 3), "c", (3, 2); stride=(1, 1), padding="SAME",
            activation=IdentityActivation(), energy=GaussianEnergy(), use_bias=false)
        pnb = NodeParams(Dict(ec => W), Dict{String, Array{Float32}}())
        _, gnb = forward_and_weight_grads(
            nb, pnb, Dict{String, Any}(ec => xc), _st(zc, (6, 5, 3))
        )
        @test isempty(gnb.biases.nt)                       # the empty-ghost half
        @test !isempty(gnb.weights.nt)                     # non-vacuous: weights DID differentiate
        @test all(isfinite, gnb.weights[ec])
        @test any(!iszero, gnb.weights[ec])
    end

    @testset "pool: AvgPool global (B,spatial…,C) -> (B,C)" begin
        e = "s->p:in"
        xg, zg = FIX_C["avgpool_global_in_x"], FIX_C["avgpool_global_in_z_latent"]
        node = AvgPool((2,), "p"; global_pool=true)
        params = NodeParams(Dict{String, Array{Float32}}(), Dict{String, Array{Float32}}())
        inputs = Dict{String, Any}(e => xg)
        st, info = _st(zg, (2,)), _info("p", (2,), [e])
        ns = forward(node, params, inputs, st)
        @test allclose_c(
            ns.z_mu, FIX_C["avgpool_global_fwd_z_mu"]; rtol=1.0f-6, atol=1.0f-6
        )
        _, ig, _ = forward_and_latent_grads(node, params, inputs, st, info, true)
        @test allclose_c(
            ig[e], FIX_C["avgpool_global_gradz_gx_0"]; rtol=1.0f-6, atol=1.0f-6
        )
        @test size(ns.z_mu) == size(zg)     # rank-1 (C,) output, spatial collapsed
    end

    # ---- structural: the things rtol cannot see -----------------------------------------------
    @testset "structural: floor-vs-truncate, SAME asymmetry, fan_in" begin
        # `n < k` must RAISE (Python `//` floors to o<=0 and raises; Julia `÷` would truncate to
        # o=1 and silently validate a garbage shape — the fail-open that sank one design).
        @test_throws Exception ConvNode((1, 1, 2), "bad", (5, 5); padding="VALID") |>
            n ->
            FabricPC.initialize_params(n, Random.MersenneTwister(1), (1, 1, 2),
                Dict("s->bad:in" => (3, 3, 2)), NormalInitializer())
        # SAME pad arithmetic, against Python-computed truth. Extra goes HIGH.
        for (n, k, s, want) in ((4, 3, 2, (0, 1)), (5, 3, 2, (1, 1)), (4, 2, 1, (0, 1)))
            @test FabricPC._pads((n,), (k,), (s,), FabricPC.SamePad())[1] == want
        end
        # conv-aware fan_in (C-08): C_in * prod(kernel), NOT C_in.
        c = ConvNode((6, 5, 3), "c", (3, 2))
        @test FabricPC.get_weight_fan_in(c, (6, 5, 2)) == 2 * 3 * 2
        @test FabricPC.get_weight_fan_in(MaxPool((2, 2, 2), "m", (2, 2)), (4, 4, 2)) == 1
    end

    # ---- C-01 adversarial-audit regressions -------------------------------------------------
    # EVERY test here feeds the guard THE INPUT IT MUST REJECT. That is the whole lesson of the
    # bug in the first testset below: the in-place guard had three passing @test_throws and was
    # still dead code for conv, because all three fed it rank-2 graphs that could REACH it.
    # Asserting "a valid pool constructs" would prove no false-positive, not that a guard fires.
    @testset "audit: in-place guard is REACHABLE for a conv graph (was pre-empted)" begin
        # to_flat_params (rank-2 only) used to run BEFORE the scope guards, so a ConvNode died on
        # `MethodError: Matrix{Float32}(::Array{Float32,4})` instead of the documented scope error.
        rng = Random.MersenneTwister(4)
        x = Linear((5, 5, 2), "x"; use_bias=false)
        cv = ConvNode((5, 5, 3), "cv", (3, 2); padding="SAME")
        o = Linear((3,), "o"; energy=GaussianEnergy())
        g = AvgPool((3,), "gp"; global_pool=true)
        st = graph([x, cv, g, o], [Edge(x, cv), Edge(cv, g), Edge(g, o)],
            TaskMap(x=x, y=o), InferenceSGD(eta_infer=0.1, infer_steps=2))
        p = initialize_params(st, rng)
        cl = Dict{String, Any}(
            "x" => randn(rng, Float32, 2, 5, 5, 2), "o" => randn(rng, Float32, 2, 3)
        )
        err = try
            FabricPC.prealloc_inference(st, p, cl; batch=2)
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("prealloc_inference", err)         # OUR message, not a MethodError
        @test occursin("outside the in-place lane", err)
        @test !occursin("MethodError", err)               # the pre-emption's signature
    end

    @testset "audit: flat lane fails CLOSED on conv/pool (was a bare MethodError)" begin
        rng = Random.MersenneTwister(4)
        x = Linear((5, 5, 2), "x"; use_bias=false)
        cv = ConvNode((5, 5, 3), "cv", (3, 2); padding="SAME")
        st = graph([x, cv], [Edge(x, cv)], TaskMap(x=x, y=cv), InferenceSGD(infer_steps=2))
        p = initialize_params(st, rng)
        err = try
            to_flat_params(CompiledPlan(st), p)
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("outside the FLAT lane's scope", err)   # names the node AND the lane
        @test occursin("run_inference", err)                   # and points at the right lane
        @test !occursin("MethodError", err)
    end

    @testset "audit: MaxPool/AvgPool spatial-rank guard (ConvNode had one; they did not)" begin
        # Feed the REJECTED input: rank-4 spatial. (A rank-2 pool constructing proves nothing.)
        @test_throws ArgumentError MaxPool((2, 2, 2, 2, 2), "m", (2, 2, 2, 2))
        @test_throws ArgumentError AvgPool((2, 2, 2, 2, 2), "a"; window_shape=(2, 2, 2, 2))
        @test_throws ArgumentError ConvNode((2, 2, 2, 2, 2), "c", (2, 2, 2, 2))   # already had it
        # non-vacuous: the ACCEPTED ranks still construct
        @test MaxPool((2, 2, 2), "m", (2, 2)) isa MaxPool
        @test AvgPool((4, 2), "a"; window_shape=(2,)) isa AvgPool
    end

    @testset "audit: AvgPool(global) ACCEPTS explicit padding (upstream ignores it)" begin
        # Global mode ignores window/stride/padding upstream (pooling.py:288-290). We used to
        # REJECT an explicit pad here — stricter than upstream on a config it accepts silently.
        @test AvgPool((3,), "g"; global_pool=true, padding=[(1, 1), (1, 1)]) isa AvgPool
        @test AvgPool((3,), "g"; global_pool=true, window_shape=(9, 9)) isa AvgPool  # also ignored
        @test_throws ArgumentError AvgPool((3, 3), "g"; global_pool=true)  # rank-1 rule still fires
    end
end
