# Linear residual node. Port of fabricpc/nodes/linear_residual.py.
#
# Fuses a linear transform path and an identity skip path into ONE PC node:
#   z_mu = activation(Σ_in xₑ Wₑ + b) + Σ_skip xₛ
# Two slots:
#   "in"   — variance-scalable, weighted (muPC scales these edges in Phase C)
#   "skip" — non-variance-scalable identity bypass (left at scale 1.0; counts
#            toward muPC depth L). No weights.
# This halves graph depth vs the Linear + SkipConnection pattern (one node per
# residual block). v0 is autodiff-free: explicit Gaussian gradients below.
#
# Edge keys carry the slot: "src->tgt:in" / "src->tgt:skip". v0 supports rank-1
# shapes (flatten_input accepted but inert), like Linear.

"""
    LinearResidual(shape, name; activation = IdentityActivation(),
                   energy = GaussianEnergy(), use_bias = true, flatten_input = false,
                   weight_init = NormalInitializer(), latent_init = NormalInitializer())

Residual block node `z_mu = activation(Σ_in xₑWₑ + b) + Σ_skip xₛ`. Slots
`"in"` (weighted, muPC-scalable) and `"skip"` (identity bypass, unscaled).
"""
struct LinearResidual <: AbstractNode
    shape::Tuple
    name::String
    activation::AbstractActivation
    energy::AbstractEnergy
    use_bias::Bool
    flatten_input::Bool
    weight_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function LinearResidual(
    shape,
    name::AbstractString;
    activation::AbstractActivation=IdentityActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    use_bias::Bool=true,
    flatten_input::Bool=false,
    weight_init::AbstractInitializer=NormalInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    return LinearResidual(
        Tuple(shape),
        String(name),
        activation,
        energy,
        use_bias,
        flatten_input,
        weight_init,
        latent_init
    )
end

get_slots(::LinearResidual) = Dict(
    "in" => SlotSpec("in", true; is_variance_scalable=true),
    "skip" => SlotSpec(
        "skip",
        true;
        is_variance_scalable=false,
        is_skip_connection=true
    )
)

_is_in_edge(k::AbstractString) = endswith(k, ":in")
_is_skip_edge(k::AbstractString) = endswith(k, ":skip")

function initialize_params(
    node::LinearResidual,
    rng::AbstractRNG,
    node_shape::Tuple,
    input_shapes::AbstractDict,
    weight_init::AbstractInitializer
)
    out_features = node_shape[end]
    weights = Dict{String, Matrix{Float32}}()
    # Weight matrices for "in" slot edges only; skip edges are identity.
    for (edge_key, in_shape) in input_shapes
        _is_in_edge(edge_key) || continue
        in_features = in_shape[end]
        weights[edge_key] = initialize(rng, (in_features, out_features), weight_init)
    end

    biases = Dict{String, Matrix{Float32}}()
    if node.use_bias
        biases["b"] = zeros(Float32, 1, out_features)
    end
    return NodeParams(weights, biases)
end

function forward(
    node::LinearResidual,
    params::NodeParams,
    inputs::AbstractDict,
    state::NodeState
)
    batch = size(state.z_latent, 1)
    out_features = node.shape[end]

    # Transform path: Σ_in xₑ Wₑ (+ b), then activation.
    pre = zeros(Float32, batch, out_features)
    for (edge_key, x) in inputs
        _is_in_edge(edge_key) || continue
        pre = pre .+ x * params.weights[edge_key]
    end
    if haskey(params.biases, "b") && length(params.biases["b"]) > 0
        pre = pre .+ params.biases["b"]
    end
    transformed = forward(node.activation, pre)

    # Skip path: Σ_skip xₛ (identity, no transform).
    skip_sum = nothing
    for (edge_key, x) in inputs
        _is_skip_edge(edge_key) || continue
        skip_sum = skip_sum === nothing ? x : skip_sum .+ x
    end

    z_mu = skip_sum === nothing ? transformed : transformed .+ skip_sum
    err = state.z_latent .- z_mu
    new_state = update_state(state; pre_activation=pre, z_mu=z_mu, error=err)
    new_state = energy_functional(node, new_state)
    return sum(new_state.energy), new_state
end

function forward_and_latent_grads(
    node::LinearResidual,
    params::NodeParams,
    inputs::AbstractDict,
    state::NodeState,
    info::NodeInfo,
    is_clamped::Bool
)
    if info.in_degree == 0
        ns = update_state(
            state;
            z_mu=copy(state.z_latent),
            error=zero(state.error),
            pre_activation=zero(state.pre_activation)
        )
        ns = energy_functional(node, ns)
        return ns, Dict{String, Any}(), zero(state.latent_grad)
    elseif info.out_degree == 0 && !is_clamped
        _, ns = forward(node, params, inputs, state)
        ns = update_state(
            ns;
            z_latent=ns.z_mu,
            error=zero(ns.error),
            energy=zero(ns.energy),
            latent_grad=zero(ns.latent_grad)
        )
        input_grads = Dict{String, Any}(k => zero(inputs[k]) for k in keys(inputs))
        return ns, input_grads, zero(state.z_latent)
    else
        _, ns = forward(node, params, inputs, state)
        self_grad = grad_latent(node.energy, ns.z_latent, ns.z_mu)
        dpre = pre_grad(node, ns)        # transform path: ∂E/∂pre
        dmu = mu_grad(node, ns)          # skip path: ∂E/∂z_mu (bypasses activation)
        input_grads = Dict{String, Any}()
        for (edge_key, _) in inputs
            if _is_in_edge(edge_key)
                # Transform path: gradient flows through W and the activation.
                input_grads[edge_key] = dpre * transpose(params.weights[edge_key])
            else
                # Skip path: dz_mu/dxₛ = 1 (bypasses the activation) ⇒ ∂E/∂z_mu.
                input_grads[edge_key] = dmu
            end
        end
        return ns, input_grads, self_grad
    end
end

function forward_and_weight_grads(
    node::LinearResidual,
    params::NodeParams,
    inputs::AbstractDict,
    state::NodeState
)
    _, ns = forward(node, params, inputs, state)
    dpre = pre_grad(node, ns)

    weight_grads = Dict{String, Matrix{Float32}}()
    for (edge_key, x) in inputs
        _is_in_edge(edge_key) || continue   # skip edges have no weights
        weight_grads[edge_key] = transpose(x) * dpre
    end

    bias_grads = Dict{String, Matrix{Float32}}()
    if haskey(params.biases, "b")
        bias_grads["b"] = sum(dpre; dims=1)
    end
    return ns, NodeParams(weight_grads, bias_grads)
end
