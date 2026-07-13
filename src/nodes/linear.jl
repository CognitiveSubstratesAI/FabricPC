# Linear node. Port of fabricpc/nodes/linear.py + linear_explicit_grad.py
# FUSED: v0 is autodiff-free, so our Linear computes explicit closed-form
# Gaussian gradients directly (there is no separate LinearExplicitGrad class).
#
# v0 supports rank-1 shapes (features,) only → per-position last-axis matmul and
# the flatten_input dense path collapse to the same plain 2D matmul on
# (batch, features) arrays. flatten_input is therefore accepted but inert in v0;
# higher-rank tensors (where the two paths diverge and column-major reshape bites)
# are deferred.

"""
    Linear(shape, name; activation = IdentityActivation(), energy = GaussianEnergy(),
           use_bias = true, flatten_input = false,
           weight_init = NormalInitializer(), latent_init = NormalInitializer())

Linear transformation node: `z_mu = activation(Σ_e xₑ Wₑ + b)`. Single
multi-input slot `"in"`; multiple incoming edges sum. Implements local
predictive-coding learning via explicit Gaussian gradients.

The `weight_init` default is `NormalInitializer` (matching upstream
`LinearExplicitGrad`, not `Linear`'s Kaiming) because our node IS the explicit
path. A node with no incoming edges (`in_degree == 0`) is a pure source: it
holds no parameters and contributes no gradients.
"""
struct Linear <: AbstractNode
    shape::Tuple
    name::String
    activation::AbstractActivation
    energy::AbstractEnergy
    use_bias::Bool
    flatten_input::Bool
    weight_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function Linear(
    shape,
    name::AbstractString;
    activation::AbstractActivation=IdentityActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    use_bias::Bool=true,
    flatten_input::Bool=false,
    weight_init::AbstractInitializer=NormalInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    return Linear(
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

get_slots(::Linear) = Dict("in" => SlotSpec("in", true))

function initialize_params(
    node::Linear,
    rng::AbstractRNG,
    node_shape::Tuple,
    input_shapes::AbstractDict,
    weight_init::AbstractInitializer
)
    out_features = node_shape[end]
    weights = Dict{String, Matrix{Float32}}()
    for (edge_key, in_shape) in input_shapes
        occursin(":in", edge_key) || throw(
            ArgumentError("linear node requires 'in' slot; got edge key $edge_key")
        )
        in_features = in_shape[end]
        weights[edge_key] = initialize(rng, (in_features, out_features), weight_init)
    end

    biases = Dict{String, Matrix{Float32}}()
    if node.use_bias
        # (1, out) so it broadcasts over the batch dimension.
        biases["b"] = zeros(Float32, 1, out_features)
    end
    return NodeParams(weights, biases)
end

"""
    forward(node::Linear, params, inputs, state) -> (total_energy, NodeState)

Forward pass: pre = Σ_e xₑ Wₑ (+ b); z_mu = activation(pre); error = z - z_mu;
energy. Leaves `z_latent` and `latent_grad` untouched (matching upstream).
"""
function forward(node::Linear, params::NodeParams, inputs::AbstractDict, state::NodeState)
    batch = size(state.z_latent, 1)
    out_features = node.shape[end]
    pre = zeros(Float32, batch, out_features)
    for (edge_key, x) in inputs
        pre = pre .+ x * params.weights[edge_key]
    end
    if haskey(params.biases, "b") && length(params.biases["b"]) > 0
        pre = pre .+ params.biases["b"]
    end

    z_mu = forward(node.activation, pre)
    err = state.z_latent .- z_mu
    new_state = update_state(state; pre_activation=pre, z_mu=z_mu, error=err)
    new_state = energy_functional(node, new_state)
    return sum(new_state.energy), new_state
end

"""
    forward_and_latent_grads(node::Linear, params, inputs, state, info, is_clamped)

Explicit (non-autodiff) inference-phase gradients. Returns the forwarded
NodeState (with `latent_grad` preserved), the per-edge input gradients
`dE/dxₑ = pre_grad · Wₑᵀ`, and the self-latent gradient `dE/dz =
grad_latent(energy)`. Handles the terminal-source and unclamped-output special
cases (port of `NodeBase.forward_and_latent_grads`). `pre_grad = (∂E/∂z_mu)·f'`
generalizes the Phase B Gaussian `error·f'` to any energy.
"""
function forward_and_latent_grads(
    node::Linear,
    params::NodeParams,
    inputs::AbstractDict,
    state::NodeState,
    info::NodeInfo,
    is_clamped::Bool
)
    if info.in_degree == 0
        # Terminal source node: z_mu = z_latent ⇒ zero error / energy / grads.
        ns = update_state(
            state;
            z_mu=copy(state.z_latent),
            error=zero(state.error),
            pre_activation=zero(state.pre_activation)
        )
        ns = energy_functional(node, ns)
        input_grads = Dict{String, Any}()
        self_grad = zero(state.latent_grad)
        return ns, input_grads, self_grad
    elseif info.out_degree == 0 && !is_clamped
        # Output node in eval/inference mode (no targets, not clamped):
        # keep the projection, contribute no gradient.
        _, ns = forward(node, params, inputs, state)
        ns = update_state(
            ns;
            z_latent=ns.z_mu,
            error=zero(ns.error),
            energy=zero(ns.energy),
            latent_grad=zero(ns.latent_grad)
        )
        input_grads = Dict{String, Any}(k => zero(inputs[k]) for k in keys(inputs))
        self_grad = zero(state.z_latent)
        return ns, input_grads, self_grad
    else
        # Internal or clamped-output node: explicit input + self-latent grads.
        _, ns = forward(node, params, inputs, state)
        self_grad = grad_latent(node.energy, ns.z_latent, ns.z_mu)
        dpre = pre_grad(node, ns)
        input_grads = Dict{String, Any}()
        for (edge_key, _) in inputs
            input_grads[edge_key] = dpre * transpose(params.weights[edge_key])
        end
        return ns, input_grads, self_grad
    end
end

"""
    forward_and_weight_grads(node::Linear, params, inputs, state) -> (NodeState, NodeParams)

Explicit learning-phase gradients: `dWₑ = xₑᵀ · pre_grad`, `db = Σ_batch pre_grad`,
where `pre_grad = (∂E/∂z_mu)·f'(pre)`. Port of `LinearExplicitGrad.forward_and_weight_grads`
generalized to any energy (identical to the Phase B Gaussian form at precision 1).
"""
function forward_and_weight_grads(
    node::Linear,
    params::NodeParams,
    inputs::AbstractDict,
    state::NodeState;
    info=nothing
)
    _, ns = forward(node, params, inputs, state)
    dpre = pre_grad(node, ns)

    weight_grads = Dict{String, Matrix{Float32}}()
    for (edge_key, x) in inputs
        weight_grads[edge_key] = transpose(x) * dpre
    end

    bias_grads = Dict{String, Matrix{Float32}}()
    if haskey(params.biases, "b")
        bias_grads["b"] = sum(dpre; dims=1)
    end
    return ns, NodeParams(weight_grads, bias_grads)
end
