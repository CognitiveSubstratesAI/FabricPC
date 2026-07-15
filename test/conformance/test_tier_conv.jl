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
    zeros(Float32, size(z, 1)), zeros(Float32, size(z)), zeros(Float32, size(z))
)
_info(name, shape, edges) = NodeInfo(
    name, shape, "conv", Dict{String,SlotInfo}(), length(edges), 1, collect(edges), String[], nothing
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
        inputs = Dict{String,Any}(e => x)
        st, info = _st(z, (6, 5, 3)), _info("conv", (6, 5, 3), [e])

        _, ns = forward(node, params, inputs, st)
        @test allclose_c(ns.z_mu, FIX_C["overlap_fwd_z_mu"])
        @test allclose_c(ns.pre_activation, FIX_C["overlap_fwd_pre_activation"])
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
        xs = Dict{String,Any}(e => FIX_C["multi_in_x_$(i-1)"] for (i, e) in enumerate(edges))
        z = FIX_C["multi_in_z_latent"]
        node = ConvNode((5, 4, 3), "c", (3, 2); stride=(1, 1), padding="SAME",
                        activation=IdentityActivation(), energy=GaussianEnergy())
        params = NodeParams(Ws, Dict("b" => FIX_C["multi_in_b"]))
        # inputs must ITERATE in in_edges order — that is what gather_inputs produces.
        inputs = OrderedDict{String,Any}(e => xs[e] for e in edges)
        st, info = _st(z, (5, 4, 3)), _info("c", (5, 4, 3), edges)

        _, ns = forward(node, params, inputs, st)
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

    @testset "conv: ReLU + strided + VALID — pre_activation is genuinely != z_mu (B5)" begin
        # Pins the compute_pre_and_mu seam hook: with ReLU, `pre` carries negatives that `z_mu`
        # cannot. Before B5 the seam reported pre = z_mu; measured gap on this fixture: 4.81.
        e = "s->c:in"
        x, W, b, z = FIX_C["relu_valid_in_x"], FIX_C["relu_valid_in_W"],
                     FIX_C["relu_valid_in_b"], FIX_C["relu_valid_in_z_latent"]
        shape = size(z)[2:end]
        node = ConvNode(shape, "c", (3, 2); stride=(2, 1), padding="VALID",
                        activation=ReLUActivation(), energy=GaussianEnergy())
        params = NodeParams(Dict(e => W), Dict("b" => b))
        inputs = Dict{String,Any}(e => x)
        st, info = _st(z, shape), _info("c", shape, [e])

        _, ns = forward(node, params, inputs, st)
        @test allclose_c(ns.z_mu, FIX_C["relu_valid_fwd_z_mu"])
        @test allclose_c(ns.pre_activation, FIX_C["relu_valid_fwd_pre_activation"])
        @test any(ns.pre_activation .< 0) && all(ns.z_mu .>= 0)     # non-vacuous: they differ
        _, gp = forward_and_weight_grads(node, params, inputs, st)
        @test allclose_c(gp.weights[e], FIX_C["relu_valid_gradw_gW_0"])
        _, ig, _ = forward_and_latent_grads(node, params, inputs, st, info, true)
        @test allclose_c(ig[e], FIX_C["relu_valid_gradz_gx_0"])
    end

    @testset "conv: rank dispatch — 1D (NLC) and 3D (NDHWC)" begin
        for (tag, ks, st_, pad, shp) in (
            ("conv1d", (3,), (2,), "SAME", size(FIX_C["conv1d_in_z_latent"])[2:end]),
            ("conv3d", (2, 3, 1), (1, 1, 1), "SAME", size(FIX_C["conv3d_in_z_latent"])[2:end]),
        )
            e = tag == "conv1d" ? "s->c1:in" : "s->c3:in"
            node = ConvNode(shp, "c", ks; stride=st_, padding=pad,
                            activation=IdentityActivation(), energy=GaussianEnergy())
            params = NodeParams(Dict(e => FIX_C["$(tag)_in_W"]), Dict("b" => FIX_C["$(tag)_in_b"]))
            inputs = Dict{String,Any}(e => FIX_C["$(tag)_in_x"])
            z = FIX_C["$(tag)_in_z_latent"]
            state, info = _st(z, shp), _info("c", shp, [e])
            _, ns = forward(node, params, inputs, state)
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
            ("avgpool", AvgPool((2, 2, 2), "p"; window_shape=(2, 2), stride=(2, 2), padding="VALID")),
        )
            params = NodeParams(Dict{String,Array{Float32}}(), Dict{String,Array{Float32}}())
            inputs = Dict{String,Any}(e => x)
            _, ns = forward(node, params, inputs, st)
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
        params = NodeParams(Dict{String,Array{Float32}}(), Dict{String,Array{Float32}}())
        inputs = Dict{String,Any}(e => xp)
        st, info = _st(zp, (2, 2, 2)), _info("p", (2, 2, 2), [e])
        _, ns = forward(node, params, inputs, st)
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
        mus = Dict{Bool,Any}()
        for flag in (true, false)
            node = AvgPool((2, 2, 2), "p"; window_shape=(2, 2), stride=(2, 2), padding="SAME",
                           count_include_pad=flag)
            params = NodeParams(Dict{String,Array{Float32}}(), Dict{String,Array{Float32}}())
            inputs = Dict{String,Any}(e => xo)
            _, ns = forward(node, params, inputs, st)
            @test allclose_c(ns.z_mu, FIX_C["avgpool_cip_$(Int(flag))_fwd_z_mu"]; rtol=1.0f-6, atol=1.0f-6)
            _, ig, _ = forward_and_latent_grads(node, params, inputs, st, info, true)
            @test allclose_c(ig[e], FIX_C["avgpool_cip_$(Int(flag))_gradz_gx_0"]; rtol=1.0f-6, atol=1.0f-6)
            mus[flag] = ns.z_mu
        end
        @test maximum(abs.(mus[true] .- mus[false])) > 1.0f-3    # the divisor IS exercised
    end

    @testset "pool: AvgPool global (B,spatial…,C) -> (B,C)" begin
        e = "s->p:in"
        xg, zg = FIX_C["avgpool_global_in_x"], FIX_C["avgpool_global_in_z_latent"]
        node = AvgPool((2,), "p"; global_pool=true)
        params = NodeParams(Dict{String,Array{Float32}}(), Dict{String,Array{Float32}}())
        inputs = Dict{String,Any}(e => xg)
        st, info = _st(zg, (2,)), _info("p", (2,), [e])
        _, ns = forward(node, params, inputs, st)
        @test allclose_c(ns.z_mu, FIX_C["avgpool_global_fwd_z_mu"]; rtol=1.0f-6, atol=1.0f-6)
        _, ig, _ = forward_and_latent_grads(node, params, inputs, st, info, true)
        @test allclose_c(ig[e], FIX_C["avgpool_global_gradz_gx_0"]; rtol=1.0f-6, atol=1.0f-6)
        @test size(ns.z_mu) == size(zg)     # rank-1 (C,) output, spatial collapsed
    end

    # ---- structural: the things rtol cannot see -----------------------------------------------
    @testset "structural: floor-vs-truncate, SAME asymmetry, fan_in" begin
        # `n < k` must RAISE (Python `//` floors to o<=0 and raises; Julia `÷` would truncate to
        # o=1 and silently validate a garbage shape — the fail-open that sank one design).
        @test_throws Exception ConvNode((1, 1, 2), "bad", (5, 5); padding="VALID") |>
            n -> FabricPC.initialize_params(n, Random.MersenneTwister(1), (1, 1, 2),
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
end
