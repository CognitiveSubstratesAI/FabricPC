# Skip-connection node. Port of fabricpc/nodes/skip_connection.py.
#
# Behaves exactly like IdentityNode at scale 1.0 (sums inputs, passes through,
# no learnable parameters) — the difference lives entirely in the slot spec:
# `is_variance_scalable = false`, `is_skip_connection = true`. That tells muPC
# (Phase C) to leave incoming edges at scale 1.0, preserving the identity
# mapping that carries signal through deep residual nets (without it muPC's
# in-degree formula would decay skip signal by 1/√K per hop). In v0 (scaling
# off) SkipConnection and IdentityNode(scale=1) are behaviourally identical.

"""
    SkipConnection(shape, name; activation = IdentityActivation(),
                   energy = GaussianEnergy(), latent_init = NormalInitializer())

Residual/skip summation node: `z_mu = Σ_e xₑ`. Single multi-input slot `"in"`
that is NOT muPC-variance-scaled. No weights or biases.
"""
struct SkipConnection <: AbstractNode
    shape::Tuple
    name::String
    activation::AbstractActivation
    energy::AbstractEnergy
    latent_init::AbstractInitializer
end

function SkipConnection(
    shape,
    name::AbstractString;
    activation::AbstractActivation=IdentityActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    return SkipConnection(Tuple(shape), String(name), activation, energy, latent_init)
end

get_slots(::SkipConnection) = Dict(
    "in" => SlotSpec(
        "in",
        true;
        is_variance_scalable=false,
        is_skip_connection=true
    )
)

get_weight_fan_in(::SkipConnection, ::Tuple) = 1

initialize_params(
    ::SkipConnection,
    ::AbstractRNG,
    ::Tuple,
    ::AbstractDict,
    ::AbstractInitializer
) = NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}())

function forward(node::SkipConnection, ::NodeParams, inputs::AbstractDict, state::NodeState)
    z_mu = nothing
    for (_, x) in inputs
        z_mu = z_mu === nothing ? x : z_mu .+ x
    end
    # No activation, no scale: pre_activation == z_mu (matches upstream).
    err = state.z_latent .- z_mu
    new_state = update_state(state; pre_activation=z_mu, z_mu=z_mu, error=err)
    new_state = energy_functional(node, new_state)
    return sum(new_state.energy), new_state
end

function forward_and_latent_grads(
    node::SkipConnection,
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
        # z_mu = Σx bypasses the activation ⇒ dz_mu/dxₑ = 1 ⇒ input_grad = ∂E/∂z_mu.
        dmu = mu_grad(node, ns)
        input_grads = Dict{String, Any}(k => dmu for k in keys(inputs))
        return ns, input_grads, self_grad
    end
end

function forward_and_weight_grads(
    node::SkipConnection,
    params::NodeParams,
    inputs::AbstractDict,
    state::NodeState
)
    _, ns = forward(node, params, inputs, state)
    return ns, NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}())
end
