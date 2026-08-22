# J-06 in-place inference lane (src/inference_inplace.jl): type-stable, pre-allocated, 0-alloc
# PC relaxation that bakes in HOIST+PRUNE. It is an ALGEBRAICALLY-EXACT reimplementation of
# run_inference, so the oracle is BIT-IDENTITY (max|diff| == 0), not a tolerance.
#
# This file asserts:
#   (a) bit-identity of the converged latents vs run_inference on chains, a mixed-source diamond
#       (a node with one clamped + one unclamped source — the case that exercises partial hoist/
#       prune), a 3-in-edge node (multi-input sum ORDER under float non-associativity), and a
#       graph with per-node GaussianEnergy precision != 1 (the precision is read, never assumed 1);
#   (b) the HOIST set fires exactly on the all-clamped-source nodes (non-vacuous, no over-hoisting);
#   (c) the relaxation loop is 0-alloc per step (alloc count invariant to step count);
#   (d) the scope guard FAILS CLOSED: NormClip/Momentum, non-Gaussian energy, muPC scaling, and
#       unclamped sinks are REJECTED at prealloc time rather than silently mis-computed.
using Test
using FabricPC
using FabricPC: prealloc_inference, run_inference!
using Random
using LinearAlgebra: BLAS

bitexact(a, b) = size(a) == size(b) && all(a .== b)
_alloc_count(f) =
    (f(); a=Base.gc_num(); f(); b=Base.gc_num(); Base.gc_alloc_count(Base.GC_Diff(b, a)))

# x -> n1(tanh) -> ... -> out linear chain. pf(k): per-node GaussianEnergy precision (default 1).
# clampout=false leaves the sink unclamped (for the reject test).
function _ip_chain(dims; B=128, T=20, pf=(k -> 1.0f0), clampout=true, seed=7)
    rng = MersenneTwister(seed)
    names = ["n0", (["n$(i-1)" for i in 2:(length(dims) - 1)])..., "out"]
    nodes = FabricPC.AbstractNode[Linear((dims[1],), "n0"; use_bias=false)]
    for i in 2:(length(dims) - 1)
        push!(
            nodes,
            Linear(
                (dims[i],),
                names[i];
                activation=TanhActivation(),
                energy=GaussianEnergy(; precision=pf(i))
            )
        )
    end
    push!(
        nodes,
        Linear((dims[end],), "out"; energy=GaussianEnergy(; precision=pf(length(dims))))
    )
    edges = Edge[Edge(nodes[i], nodes[i + 1]) for i in 1:(length(nodes) - 1)]
    st = graph(nodes, edges, TaskMap(; x=nodes[1], y=nodes[end]),
        InferenceSGD(; eta_infer=0.1, infer_steps=T, latent_decay=0.0))
    pd = Dict{String, NodeParams}(
        "n0" => NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}())
    )
    for k in 2:length(dims)
        ek = "$(names[k-1])->$(names[k]):in"
        pd[names[k]] = NodeParams(
            Dict(ek => 0.1f0 .* randn(rng, Float32, dims[k - 1], dims[k])),
            Dict("b" => zeros(Float32, 1, dims[k])))
    end
    clamps = Dict{String, Any}("n0" => randn(rng, Float32, B, dims[1]))
    clampout && (clamps["out"] = randn(rng, Float32, B, dims[end]))
    init = initialize_graph_state(st, B, rng; clamps=clamps, params=GraphParams(pd))
    return GraphParams(pd), init, clamps, st, B
end

# Mixed-source diamond. out has THREE in-edges {h1,h2,m}; m has MIXED sources {x(clamped),h1};
# h1,h2 are the only all-clamped-source (hoistable) nodes.
function _ip_diamond(; B=96, T=20)
    rng = MersenneTwister(11)
    x = Linear((10,), "x"; use_bias=false)
    h1 = Linear((8,), "h1"; activation=TanhActivation())
    h2 = Linear((8,), "h2"; activation=TanhActivation())
    m = Linear((6,), "m"; activation=TanhActivation())
    out = Linear((4,), "out"; energy=GaussianEnergy())
    nodes = FabricPC.AbstractNode[x, h1, h2, m, out]
    edges = Edge[
        Edge(x, h1),
        Edge(x, h2),
        Edge(x, m),
        Edge(h1, m),
        Edge(h1, out),
        Edge(h2, out),
        Edge(m, out)
    ]
    st = graph(
        nodes,
        edges,
        TaskMap(; x=x, y=out),
        InferenceSGD(; eta_infer=0.1, infer_steps=T, latent_decay=0.0)
    )
    W(a, b) = 0.1f0 .* randn(rng, Float32, a, b)
    pd = Dict{String, NodeParams}(
        "x" => NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()),
        "h1" => NodeParams(Dict("x->h1:in" => W(10, 8)), Dict("b" => zeros(Float32, 1, 8))),
        "h2" => NodeParams(Dict("x->h2:in" => W(10, 8)), Dict("b" => zeros(Float32, 1, 8))),
        "m" => NodeParams(
            Dict("x->m:in" => W(10, 6), "h1->m:in" => W(8, 6)),
            Dict("b" => zeros(Float32, 1, 6))
        ),
        "out" => NodeParams(
            Dict("h1->out:in" => W(8, 4), "h2->out:in" => W(8, 4), "m->out:in" => W(6, 4)),
            Dict("b" => zeros(Float32, 1, 4)))
    )
    clamps = Dict{String, Any}(
        "x" => randn(rng, Float32, B, 10), "out" => randn(rng, Float32, B, 4)
    )
    init = initialize_graph_state(st, B, rng; clamps=clamps, params=GraphParams(pd))
    return GraphParams(pd), init, clamps, st, B
end

# THREE clamped sources feed one UNCLAMPED node H (3 in-edges) -> clamped sink T. The reference
# folds inputs in Dict-iteration order; this catches any mismatch in the in-place fold order
# (float non-associativity for 3+ in-edges). srcnames chosen so the edge-key hashes do NOT match
# insertion order (verified to diverge before the fix).
function _ip_multi3(srcnames=["A", "B", "C"]; B=64, T=20)
    rng = MersenneTwister(1)
    W(a, b) = 0.1f0 .* randn(rng, Float32, a, b)
    a, b, c = (Linear((5,), srcnames[1]; use_bias=false),
        Linear((5,), srcnames[2]; use_bias=false),
        Linear((5,), srcnames[3]; use_bias=false))
    h = Linear((4,), "H")                                    # unclamped, in_degree 3, Identity act
    t = Linear((3,), "T"; energy=GaussianEnergy())
    st = graph(FabricPC.AbstractNode[a, b, c, h, t],
        Edge[Edge(a, h), Edge(b, h), Edge(c, h), Edge(h, t)],
        TaskMap(; x=a, y=t), InferenceSGD(; eta_infer=0.1, infer_steps=T, latent_decay=0.0))
    pd = Dict{String, NodeParams}(
        srcnames[1] =>
            NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()),
        srcnames[2] =>
            NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()),
        srcnames[3] =>
            NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()),
        "H" => NodeParams(
            Dict("$(srcnames[1])->H:in" => W(5, 4), "$(srcnames[2])->H:in" => W(5, 4),
                "$(srcnames[3])->H:in" => W(5, 4)), Dict("b" => zeros(Float32, 1, 4))),
        "T" => NodeParams(Dict("H->T:in" => W(4, 3)), Dict("b" => zeros(Float32, 1, 3))))
    clamps = Dict{String, Any}(srcnames[1] => randn(rng, Float32, B, 5),
        srcnames[2] => randn(rng, Float32, B, 5),
        srcnames[3] => randn(rng, Float32, B, 5), "T" => randn(rng, Float32, B, 3))
    init = initialize_graph_state(st, B, rng; clamps=clamps, params=GraphParams(pd))
    return GraphParams(pd), init, clamps, st, B
end

# x -> mid(IdentityNode | SkipConnection, unclamped) -> out. Exercises the WEIGHTLESS node types
# (they crash prealloc before the fix). Skip variant merges two sources to also test multi-input.
function _ip_weightless(kind; B=32)
    rng = MersenneTwister(2)
    W(a, b) = 0.1f0 .* randn(rng, Float32, a, b)
    if kind == :ident
        x = Linear((6,), "x"; use_bias=false)
        mid = IdentityNode((6,), "mid")
        out = Linear((4,), "out"; energy=GaussianEnergy())
        nodes = FabricPC.AbstractNode[x, mid, out]
        edges = Edge[Edge(x, mid), Edge(mid, out)]
        pd = Dict(
            "x" => NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            ),
            "mid" => NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            ),
            "out" => NodeParams(
                Dict("mid->out:in" => W(6, 4)), Dict("b" => zeros(Float32, 1, 4))
            ))
        clamps = Dict{String, Any}(
            "x" => randn(rng, Float32, B, 6), "out" => randn(rng, Float32, B, 4)
        )
    else  # :skip — two clamped sources merge through a SkipConnection (multi-input, weightless)
        x1 = Linear((6,), "x1"; use_bias=false)
        x2 = Linear((6,), "x2"; use_bias=false)
        mid = SkipConnection((6,), "mid")
        out = Linear((4,), "out"; energy=GaussianEnergy())
        nodes = FabricPC.AbstractNode[x1, x2, mid, out]
        edges = Edge[Edge(x1, mid), Edge(x2, mid), Edge(mid, out)]
        pd = Dict(
            "x1" => NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            ),
            "x2" => NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            ),
            "mid" => NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            ),
            "out" => NodeParams(
                Dict("mid->out:in" => W(6, 4)), Dict("b" => zeros(Float32, 1, 4))
            ))
        clamps = Dict{String, Any}("x1" => randn(rng, Float32, B, 6),
            "x2" => randn(rng, Float32, B, 6),
            "out" => randn(rng, Float32, B, 4))
    end
    st = graph(
        nodes,
        edges,
        TaskMap(; x=nodes[1], y=out),
        InferenceSGD(; eta_infer=0.1, infer_steps=10, latent_decay=0.0)
    )
    init = initialize_graph_state(st, B, rng; clamps=clamps, params=GraphParams(pd))
    return GraphParams(pd), init, clamps, st, B
end

# Small x->h->out graph, perturbing ONE thing that must be rejected fail-closed.
function _ip_bad(kind)
    rng = MersenneTwister(3)
    x = Linear((16,), "x"; use_bias=false)
    h = Linear((12,), "h"; activation=TanhActivation())
    out = Linear((4,), "out"; energy=(kind == :bern ? BernoulliEnergy() : GaussianEnergy()))
    inf = if kind == :normclip
        InferenceSGDNormClip(; eta_infer=0.1, infer_steps=10)
    else
        InferenceSGD(; eta_infer=0.1, infer_steps=10)
    end
    st = graph([x, h, out], [Edge(x, h), Edge(h, out)], TaskMap(; x=x, y=out), inf)
    W(a, b) = 0.1f0 .* randn(rng, Float32, a, b)
    pd = Dict(
        "x" => NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()),
        "h" => NodeParams(Dict("x->h:in" => W(16, 12)), Dict("b" => zeros(Float32, 1, 12))),
        "out" =>
            NodeParams(Dict("h->out:in" => W(12, 4)), Dict("b" => zeros(Float32, 1, 4))))
    clamps = Dict{String, Any}("x" => randn(rng, Float32, 32, 16))
    (kind != :sink) && (clamps["out"] = randn(rng, Float32, 32, 4))
    return GraphParams(pd), clamps, st
end

# Assert run_inference! is bit-identical to run_inference on every node's converged latents.
function _assert_ip_bitexact(params, init, clamps, st, B; expect_hoist=nothing)
    ref = run_inference(params, init, clamps, st)
    ii = prealloc_inference(st, params, clamps; batch=B)
    if expect_hoist !== nothing
        hoisted = Set(ii.plan.names[i] for i in 1:length(ii.plan.names) if ii.hoist[i])
        @test hoisted == expect_hoist                      # (b) HOIST fires exactly here — no over-hoist
    end
    got = run_inference!(ii, init)
    for (i, nm) in enumerate(ii.plan.names)
        @test bitexact(got[i], ref.nodes[nm].z_latent)     # (a) bit-identity
    end
end

@testset "in-place inference (J-06)" begin
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)                                 # determinism across BLAS thread counts
    try
        @testset "bit-identity vs run_inference" begin
            _assert_ip_bitexact(_ip_chain([784, 128, 10]; B=128)...)
            _assert_ip_bitexact(_ip_chain([64, 64, 64, 64, 64, 64, 64]; B=96)...)
            _assert_ip_bitexact(_ip_diamond(; B=96)...; expect_hoist=Set(["h1", "h2"]))
            # precision != 1, distinct per node (C1: precision is READ, not assumed 1)
            _assert_ip_bitexact(
                _ip_chain([256, 96, 10]; B=96, pf=(k -> 0.5f0 + 0.7f0 * k))...
            )
            # 3-in-edge node across edge-key sets: all lanes fold in in_src (== in_edges) order, the
            # canonical upstream order. gather_inputs uses an OrderedDict so run_inference agrees;
            # before that, run_inference (hash-order Dict) diverged from these by ~1 ULP.
            for sn in (
                ["A", "B", "C"],
                ["s1", "s2", "s3"],
                ["p", "q", "r"],
                ["n1", "n2", "n3"],
                ["h1", "h2", "m"]
            )
                _assert_ip_bitexact(_ip_multi3(sn)...)
            end
            # weightless node types (crashed prealloc before the fix)
            _assert_ip_bitexact(_ip_weightless(:ident)...)
            _assert_ip_bitexact(_ip_weightless(:skip)...)
        end

        @testset "0-alloc per step (step-count invariant)" begin
            a20 = let (p, i, c, s, B) = _ip_chain([784, 128, 10]; B=128, T=20)
                ii = prealloc_inference(s, p, c; batch=B)
                _alloc_count(() -> run_inference!(ii, i))
            end
            a200 = let (p, i, c, s, B) = _ip_chain([784, 128, 10]; B=128, T=200)
                ii = prealloc_inference(s, p, c; batch=B)
                _alloc_count(() -> run_inference!(ii, i))
            end
            @test a20 == a200                                # per-step loop allocates nothing
        end

        @testset "scope guard fails CLOSED" begin
            # NormClip / non-Gaussian energy / unclamped sink must THROW at prealloc, not mis-compute.
            @test_throws Exception (
                let (p, c, s) = _ip_bad(:normclip)
                    prealloc_inference(s, p, c; batch=32)
                end
            )
            @test_throws Exception (
                let (p, c, s) = _ip_bad(:bern)
                    prealloc_inference(s, p, c; batch=32)
                end
            )
            @test_throws Exception (
                let (p, c, s) = _ip_bad(:sink)
                    prealloc_inference(s, p, c; batch=32)
                end
            )
            # batch mismatch: buffers baked to B=128; running a B=96 init_state must THROW, not
            # silently copy stale rows into the larger buffers.
            @test_throws Exception let
                p, _, c, s, _ = _ip_chain([64, 32, 10]; B=128)
                ii = prealloc_inference(s, p, c; batch=128)
                _, i96, _, _, _ = _ip_chain([64, 32, 10]; B=96)
                run_inference!(ii, i96)
            end
        end
    finally
        BLAS.set_num_threads(old_blas)
    end
end
