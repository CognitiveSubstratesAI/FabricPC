# Identity node. Port of fabricpc/nodes/identity.py.
#
# Passes input through unchanged (sum of inputs × fixed `scale`), no learnable
# parameters. Useful as a passthrough / aggregation conduit. Upstream relies on
# the autodiff default for its gradients; since v0 is explicit we derive them:
# with z_mu = scale·Σ xₑ and GaussianEnergy (precision 1, the default),
#   self_grad   = grad_latent          = z - z_mu
#   dE/dxₑ      = -scale · gain_mod_error   (gain_mod_error = error, f'=1)
# which equals the autodiff result at the default precision.

"""
    IdentityNode(shape, name; activation = IdentityActivation(),
                 energy = GaussianEnergy(), latent_init = NormalInitializer(),
                 scale = 1.0)

Identity passthrough node: `z_mu = scale * Σ_e xₑ`. Single multi-input slot
`"in"`; no weights or biases.
"""
struct IdentityNode <: AbstractNode
    shape::Tuple
    name::String
    activation::AbstractActivation
    energy::AbstractEnergy
    latent_init::AbstractInitializer
    scale::Float32
end

function IdentityNode(
    shape,
    name::AbstractString;
    activation::AbstractActivation=IdentityActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    latent_init::AbstractInitializer=NormalInitializer(),
    scale=1.0
)
    return IdentityNode(
        Tuple(shape),
        String(name),
        activation,
        energy,
        latent_init,
        Float32(scale)
    )
end

get_slots(::IdentityNode) = Dict("in" => SlotSpec("in", true))

# No weight matrix; muPC fan-in of 1 reduces the scaling formula to multi-edge
# summation variance compensation only (Phase C).
get_weight_fan_in(::IdentityNode, ::Tuple) = 1

initialize_params(
    ::IdentityNode,
    ::AbstractRNG,
    ::Tuple,
    ::AbstractDict,
    ::AbstractInitializer
) = NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}())

function forward(
    node::IdentityNode,
    ::NodeParams,
    inputs::AbstractDict,
    state::NodeState
)
    z_mu = nothing
    for (_, x) in inputs
        z_mu = z_mu === nothing ? x : z_mu .+ x
    end
    z_mu = z_mu .* node.scale
    # No activation transform: pre_activation == z_mu (matches upstream).
    err = state.z_latent .- z_mu
    new_state = update_state(state; z_mu=z_mu, error=err)
    new_state = energy_functional(node, new_state)
    return new_state
end

function forward_and_latent_grads(
    node::IdentityNode,
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
            error=zero(state.error)
        )
        ns = energy_functional(node, ns)
        return ns, Dict{String, Any}(), zero(state.latent_grad)
    elseif info.out_degree == 0 && !is_clamped
        ns = forward(node, params, inputs, state)
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
        ns = forward(node, params, inputs, state)
        self_grad = grad_latent(node.energy, ns.z_latent, ns.z_mu)
        # z_mu = scale·Σx bypasses the activation ⇒ dE/dxₑ = scale·(∂E/∂z_mu).
        dmu = mu_grad(node, ns)
        input_grads = Dict{String, Any}()
        for (edge_key, _) in inputs
            input_grads[edge_key] = node.scale .* dmu
        end
        return ns, input_grads, self_grad
    end
end

# No learnable parameters ⇒ empty gradients (skipped by the SGD update).
function forward_and_weight_grads(
    node::IdentityNode,
    params::NodeParams,
    inputs::AbstractDict,
    state::NodeState;
    info=nothing
)
    ns = forward(node, params, inputs, state)
    return ns, NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}())
end
