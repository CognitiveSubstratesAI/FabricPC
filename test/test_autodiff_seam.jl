# Phase-D autodiff node seam (FabricPCEnzymeExt). Two guarantees:
#
#  (1) CONFORMANCE — a node defined via `compute_mu` only, routed through the
#      generic Enzyme gradient path, reproduces the closed-form `Linear`
#      gradients (the oracle) to Float32 tolerance. This locks the autodiff seam
#      to the hand-derived path that validated it.
#
#  (2) END-TO-END — a genuinely non-linear node (a 1-hidden-layer MLP, whose
#      local weight gradient has NO closed form in this framework) trains via
#      `train_pcn` purely by predictive coding: energy falls and a separable
#      classification task is learned. This is the capability the seam unlocks —
#      arbitrary node forwards (transformer/attention next) with local PC, no
#      backprop.
#
# `using Zygote` or `using Enzyme` activates the weakdep extension (exactly ONE per
# session — decisions.md #19, F-04); without either loaded these methods raise a
# load hint (so the base package stays autodiff-free, decisions.md #8). This suite
# runs on Zygote (the backend that also handles transformer/attention nodes).

using FabricPC
using FabricPC: NodeParams, NodeState, NodeInfo, SlotInfo, AbstractNode, compute_mu
using Zygote
using Random, Test

const FP = FabricPC

# ---------------------------------------------------------------------------
# Test node A: ADLinear — same math as Linear, but defined ONLY via compute_mu,
# so it exercises the generic autodiff gradient methods (no explicit override).
# ---------------------------------------------------------------------------
struct ADLinear <: AbstractNode
    shape::Tuple
    name::String
    activation::Any
    energy::Any
    weight_init::Any
    latent_init::Any
end
FP.get_slots(::ADLinear) = Dict("in" => FP.SlotSpec("in", true))
function FP.compute_mu(n::ADLinear, params::NodeParams, inputs)
    out = n.shape[end]
    batch = size(first(values(inputs)), 1)
    pre = zeros(Float32, batch, out)
    for (k, xv) in inputs
        pre = pre .+ xv * params.weights[k]
    end
    haskey(params.biases, "b") && (pre = pre .+ params.biases["b"])
    return FP.forward(n.activation, pre)
end

# ---------------------------------------------------------------------------
# Test node B: MLPNode — z_mu = activation(x·W1 + b1)·W2 + b2. A hidden
# nonlinearity ⇒ the local weight gradient is NOT the framework's explicit
# xᵀ·(error·f') form; it genuinely requires autodiff through the inner layer.
# ---------------------------------------------------------------------------
struct MLPNode <: AbstractNode
    shape::Tuple
    name::String
    hidden::Int
    activation::Any
    energy::Any
    weight_init::Any
    latent_init::Any
end
FP.get_slots(::MLPNode) = Dict("in" => FP.SlotSpec("in", true))
function FP.initialize_params(
    node::MLPNode, rng::AbstractRNG, node_shape::Tuple, input_shapes::AbstractDict,
    weight_init
)
    out = node_shape[end]
    h = node.hidden
    weights = Dict{String, Matrix{Float32}}()
    for (edge_key, in_shape) in input_shapes
        weights[edge_key] = FP.initialize(rng, (in_shape[end], h), weight_init)
    end
    weights["W2"] = FP.initialize(rng, (h, out), weight_init)
    biases = Dict{String, Matrix{Float32}}("b1" => zeros(Float32, 1, h),
        "b2" => zeros(Float32, 1, out))
    return NodeParams(weights, biases)
end
function FP.compute_mu(node::MLPNode, params::NodeParams, inputs)
    h = node.hidden
    batch = size(first(values(inputs)), 1)
    pre1 = zeros(Float32, batch, h)
    for (k, xv) in inputs
        pre1 = pre1 .+ xv * params.weights[k]
    end
    pre1 = pre1 .+ params.biases["b1"]
    hid = FP.forward(node.activation, pre1)
    return hid * params.weights["W2"] .+ params.biases["b2"]
end

@testset "Phase-D autodiff node seam (Enzyme)" begin
    @testset "conformance: generic autodiff == closed-form Linear" begin
        rng = MersenneTwister(1)
        batch, in_f, out_f = 5, 4, 3
        ekey = "src->h:in"
        act, en = TanhActivation(), GaussianEnergy()

        ln = Linear((out_f,), "h"; activation=act, energy=en)
        ad = ADLinear((out_f,), "h", act, en, NormalInitializer(), NormalInitializer())

        W = randn(rng, Float32, in_f, out_f)
        b = randn(rng, Float32, 1, out_f)
        x = randn(rng, Float32, batch, in_f)
        z = randn(rng, Float32, batch, out_f)
        params = NodeParams(Dict(ekey => W), Dict("b" => b))
        inputs = Dict{String, Any}(ekey => x)
        st = NodeState(z, zeros(Float32, batch, out_f), zeros(Float32, batch, out_f),
            zeros(Float32, batch),
            zeros(Float32, batch, out_f))
        info = NodeInfo("h", (out_f,), "node", Dict{String, SlotInfo}(),
            1, 1, [ekey], String[], nothing)

        _, cf_w = FP.forward_and_weight_grads(ln, params, inputs, st)
        _, cf_ig, cf_sg = FP.forward_and_latent_grads(ln, params, inputs, st, info, false)
        _, ad_w = FP.forward_and_weight_grads(ad, params, inputs, st)
        _, ad_ig, ad_sg = FP.forward_and_latent_grads(ad, params, inputs, st, info, false)

        @test ad_w.weights[ekey] ≈ cf_w.weights[ekey] rtol = 1e-4
        @test ad_w.biases["b"] ≈ cf_w.biases["b"] rtol = 1e-4
        @test ad_sg ≈ cf_sg rtol = 1e-4
        @test ad_ig[ekey] ≈ cf_ig[ekey] rtol = 1e-4
    end

    @testset "load hint without Enzyme is a clear message" begin
        # The hint constant exists and names the activation path (smoke).
        @test occursin("using Enzyme", FP._ENZYME_HINT)
    end

    @testset "end-to-end: non-linear MLP node trains by PC (no backprop)" begin
        rng = MersenneTwister(2024)
        n_features, n_class, per_class = 6, 3, 20
        batch = n_class * per_class
        centers = 2.5f0 .* randn(rng, Float32, n_class, n_features)
        X = zeros(Float32, batch, n_features)
        Y = zeros(Float32, batch, n_class)
        row = 1
        for c in 1:n_class, _ in 1:per_class
            X[row, :] = centers[c, :] .+ 0.3f0 .* randn(rng, Float32, n_features)
            Y[row, c] = 1.0f0
            row += 1
        end

        xn = Linear((n_features,), "x")
        hn = MLPNode((12,), "h", 16, TanhActivation(), GaussianEnergy(),
            NormalInitializer(), NormalInitializer())
        yn = Linear((n_class,), "y")
        structure = graph(
            [xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)],
            TaskMap(; x=xn, y=yn), InferenceSGD(; eta_infer=0.1, infer_steps=30)
        )
        params = initialize_params(structure, MersenneTwister(99))
        loader = [Dict("x" => X, "y" => Y)]

        ev0 = evaluate_pcn(params, structure, loader; rng=MersenneTwister(7))
        params, iters, _ = train_pcn(
            params, structure, loader, 0.01; num_epochs=150,
            rng=MersenneTwister(7), verbose=false
        )
        ev1 = evaluate_pcn(params, structure, loader; rng=MersenneTwister(7))

        @test iters[end][1] < iters[1][1]        # per-batch energy fell
        @test ev1["energy"] < ev0["energy"]       # converged-state energy dropped
        @test ev1["accuracy"] > ev0["accuracy"]   # learned
        @test ev1["accuracy"] > 0.85              # separable task substantially solved
    end
end
