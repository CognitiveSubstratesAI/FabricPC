# HOIST+PRUNE inference optimizations (src/core/inference.jl, `run_inference(...; optimize=true)`).
#
# These are ALGEBRAICALLY-EXACT optimizations, so the oracle here is BIT-IDENTITY, not a
# tolerance: the optimized loop must reproduce the stock loop EXACTLY on every field the rest of
# the system consumes. The single legitimate divergence is the `latent_grad` of CLAMPED nodes —
# a scratch value `update_latents` discards for clamped nodes and `compute_local_weight_gradients`
# never reads — which the optimized loop leaves at zero (that is the whole point of PRUNE). This
# file asserts (a) every OTHER field is bit-identical, (b) clamped `latent_grad` is exactly zero
# under optimization while the stock loop leaves it NONZERO (so the optimization is non-vacuous),
# (c) the hoist set is exactly the expected node (so HOIST actually fires), and (d) the downstream
# local weight gradients are bit-identical (so the training path is unaffected).
using Test
using FabricPC
using Random

# Elementwise bit-exact equality (any difference is a bug, not a tolerance miss).
bitexact(a, b) = size(a) == size(b) && all(a .== b)

# Build an x->h(tanh)->y linear chain with deterministic Float32 params + clamps.
function _opt_fixture(DX, DH, DY, B; seed=0xC0DE, inference)
    rng = MersenneTwister(seed)
    x = Linear((DX,), "x"; use_bias=false)
    h = Linear((DH,), "h"; activation=TanhActivation())
    y = Linear((DY,), "y"; energy=GaussianEnergy())
    structure = graph([x, h, y], [Edge(x, h), Edge(h, y)], TaskMap(x=x, y=y), inference)
    edge_xh, edge_hy = "x->h:in", "h->y:in"
    params = GraphParams(
        Dict(
            "x" => NodeParams(Dict{String,Matrix{Float32}}(), Dict{String,Matrix{Float32}}()),
            "h" => NodeParams(
                Dict(edge_xh => 0.1f0 .* randn(rng, Float32, DX, DH)),
                Dict("b" => 0.1f0 .* randn(rng, Float32, 1, DH)),
            ),
            "y" => NodeParams(
                Dict(edge_hy => 0.1f0 .* randn(rng, Float32, DH, DY)),
                Dict("b" => 0.1f0 .* randn(rng, Float32, 1, DY)),
            ),
        ),
    )
    Xb = randn(rng, Float32, B, DX)
    Yb = randn(rng, Float32, B, DY)
    clamps = Dict{String,Any}("x" => Xb, "y" => Yb)
    init = initialize_graph_state(structure, B, rng; clamps=clamps, params=params)
    return params, init, clamps, structure
end

# Assert optimized run is bit-identical to the stock run on all consumed fields; verify PRUNE and
# HOIST both fired non-vacuously; verify the local weight gradients are bit-identical.
function _assert_bit_identical(params, init, clamps, structure)
    s0 = run_inference(params, init, clamps, structure)                 # stock
    s1 = run_inference(params, init, clamps, structure; optimize=true)  # HOIST+PRUNE

    # (c) HOIST fires exactly on "h" (source x clamped, out_degree>0, unclamped) — non-vacuous.
    @test FabricPC._hoistable_nodes(clamps, structure) == Set(["h"])

    any_clamped_nonzero_stock = false
    for name in structure.node_names
        a, b = s0.nodes[name], s1.nodes[name]
        # (a) every consumed field bit-identical, for EVERY node.
        @test bitexact(a.z_latent, b.z_latent)
        @test bitexact(a.z_mu, b.z_mu)
        @test bitexact(a.error, b.error)
        @test bitexact(a.energy, b.energy)
        if haskey(clamps, name)
            # (b) clamped node's latent_grad: optimized leaves it at zero (pruned dead value)...
            @test all(iszero, b.latent_grad)
            any_clamped_nonzero_stock |= !all(iszero, a.latent_grad)
        else
            # ...unclamped node's latent_grad is LIVE and must be bit-identical.
            @test bitexact(a.latent_grad, b.latent_grad)
        end
    end
    # (b, cont.) the stock loop leaves at least one clamped latent_grad NONZERO — i.e. PRUNE
    # actually skipped real (dead) work, this is not a vacuous "both zero" pass.
    @test any_clamped_nonzero_stock

    # (d) downstream local weight gradients are bit-identical from either converged state
    # (compute_local_weight_gradients recomputes from z_latent/z_mu; it never reads latent_grad).
    g0 = compute_local_weight_gradients(params, s0, structure)
    g1 = compute_local_weight_gradients(params, s1, structure)
    for name in structure.node_names
        for (k, v) in pairs(g0.nodes[name].weights)
            @test bitexact(v, g1.nodes[name].weights[k])
        end
        for (k, v) in pairs(g0.nodes[name].biases)
            @test bitexact(v, g1.nodes[name].biases[k])
        end
    end
    return nothing
end

@testset "inference HOIST+PRUNE — bit-identical on consumed fields" begin
    @testset "InferenceSGD" begin
        inf = InferenceSGD(eta_infer=0.1, infer_steps=20, latent_decay=0.0)
        @testset "Tier-C shape 4->6->3, B=3" begin
            _assert_bit_identical(_opt_fixture(4, 6, 3, 3; inference=inf)...)
        end
        @testset "MNIST shape 784->128->10, B=8" begin
            _assert_bit_identical(_opt_fixture(784, 128, 10, 8; inference=inf)...)
        end
    end
    @testset "InferenceSGD with latent_decay" begin
        inf = InferenceSGD(eta_infer=0.05, infer_steps=15, latent_decay=0.01)
        _assert_bit_identical(_opt_fixture(784, 128, 10, 8; seed=0x1234, inference=inf)...)
    end
    @testset "InferenceSGDNormClip" begin
        inf = InferenceSGDNormClip(eta_infer=0.1, infer_steps=20, max_norm=1.0)
        _assert_bit_identical(_opt_fixture(784, 128, 10, 8; seed=0xBEEF, inference=inf)...)
    end
end
