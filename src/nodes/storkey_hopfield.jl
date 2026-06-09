# StorkeyHopfield — Hopfield associative-memory node with energy-based learning.
# Port of fabricpc/nodes/storkey_hopfield.py. The FIRST node with a COMPOSITE energy:
#
#   E_total = E_pc(z, z_mu)  +  s · E_hop(z, W),   E_hop = (1/2D) zᵀ(W²−W)z
#
# E_pc pulls the latent z toward the upstream prediction z_mu; E_hop pulls z toward
# stored patterns (attractors encoded in W). The equilibrium z* is the PC-optimal
# compromise between top-down expectation and internal memory prior — attractor
# dynamics (denoising / pattern completion) arise from the Hopfield energy gradient
# (s/D)(W²−W)z accumulated to latent_grad during PC inference. Local PC gradients
# (latent: PC pull + attractor pull; weight: ∂(E_pc+E_hop)/∂W) come from the Enzyme
# seam — this node OVERRIDES `energy_kernel` (so the seam sees E_hop) and `forward`
# (so state.energy reports the full energy). Rank-2 (batch, D).

inverse_softplus(x) = log(exp(x) - 1.0f0)             # raw s.t. softplus(raw)=x
_softplus(x) = log1p(exp(-abs(x))) + max(x, 0.0f0)    # numerically-stable softplus

"""
    StorkeyHopfield(shape, name; activation = TanhActivation(), energy = GaussianEnergy(),
                    hopfield_strength = nothing, use_bias = true, enforce_symmetry = true,
                    zero_diagonal = false, weight_init = ZerosInitializer(),
                    latent_init = NormalInitializer())

Hopfield associative-memory PC node. `shape = (D,)`. `hopfield_strength = nothing`
makes the strength learnable (softplus-constrained ≥ 0, init so s=1); a number fixes
it. `W` (D×D, stored under the input edge key) participates in BOTH the prediction
(probe·W) and the attractor energy. Local PC gradients via the Enzyme seam
(`using Enzyme`).
"""
struct StorkeyHopfield <: AbstractNode
    shape::Tuple
    name::String
    activation::AbstractActivation
    energy::AbstractEnergy
    hopfield_strength::Union{Float32, Nothing}
    use_bias::Bool
    enforce_symmetry::Bool
    zero_diagonal::Bool
    weight_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function StorkeyHopfield(
    shape,
    name::AbstractString;
    activation::AbstractActivation=TanhActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    hopfield_strength::Union{Real, Nothing}=nothing,
    use_bias::Bool=true,
    enforce_symmetry::Bool=true,
    zero_diagonal::Bool=false,
    weight_init::AbstractInitializer=ZerosInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    return StorkeyHopfield(
        Tuple(shape), String(name), activation, energy,
        hopfield_strength === nothing ? nothing : Float32(hopfield_strength),
        use_bias, enforce_symmetry, zero_diagonal, weight_init, latent_init
    )
end

get_slots(::StorkeyHopfield) = Dict("in" => SlotSpec("in", false))

# Symmetrize and/or zero the diagonal of W (both differentiable).
function _prepare_W(node::StorkeyHopfield, W::AbstractMatrix)
    D = size(W, 1)
    node.enforce_symmetry && (W = 0.5f0 .* (W .+ transpose(W)))
    node.zero_diagonal && (W = W .* Float32[i == j ? 0.0f0 : 1.0f0 for i in 1:D, j in 1:D])
    return W
end

# Effective strength: learnable (softplus of the raw bias) or fixed.
_strength(node::StorkeyHopfield, params::NodeParams) =
    if node.hopfield_strength === nothing
        _softplus(params.biases["hopfield_strength"][1, 1])
    else
        node.hopfield_strength
    end

function initialize_params(
    node::StorkeyHopfield, rng::AbstractRNG, node_shape::Tuple,
    input_shapes::AbstractDict, weight_init::AbstractInitializer
)
    D = node_shape[end]
    edge_key = first(keys(input_shapes))
    input_shapes[edge_key][end] == D ||
        throw(ArgumentError("Hopfield input dim must match node dim D=$D"))
    W = _prepare_W(node, initialize(rng, (D, D), weight_init))
    weights = Dict{String, Matrix{Float32}}(edge_key => W)   # under edge key ⇒ grad to source
    biases = Dict{String, Matrix{Float32}}()
    node.use_bias && (biases["b"] = zeros(Float32, 1, D))
    node.hopfield_strength === nothing &&
        (biases["hopfield_strength"] = fill(inverse_softplus(1.0f0), 1, 1))
    return NodeParams(weights, biases)
end

# z_mu = activation( probe·blend + (probe·W)·(1−blend) + b ),  blend = 1/(1+s).
# At s=0 pure pass-through activation(probe); at large s ≈ activation(probe·W).
function compute_mu(node::StorkeyHopfield, params::NodeParams, inputs)
    edge_key = first(keys(inputs))
    probe = inputs[edge_key]
    W = _prepare_W(node, params.weights[edge_key])
    s = _strength(node, params)
    blend = 1.0f0 / (1.0f0 + s)
    pre = probe .* blend .+ (probe * W) .* (1.0f0 - blend)
    haskey(params.biases, "b") && (pre = pre .+ params.biases["b"])
    return forward(node.activation, pre)
end

# Per-sample Hopfield attractor energy s·(1/2D) zᵀ(W²−W)z, with wz = z·W → (batch,).
function _hopfield_energy(node::StorkeyHopfield, params::NodeParams, z)
    edge_key = first(k for k in keys(params.weights))
    W = _prepare_W(node, params.weights[edge_key])
    s = _strength(node, params)
    D = size(z, 2)
    wz = z * W
    return s .* (0.5f0 / D) .* vec(sum(wz .* (wz .- z); dims=2))
end

# Seam gradient target: total scalar energy Σ(E_pc + s·E_hop). Overrides the generic
# energy_kernel so Enzyme differentiates the FULL energy (latent: PC + attractor pull;
# weight: ∂E_hop/∂W incl the W²−W term).
function energy_kernel(node::StorkeyHopfield, params::NodeParams, inputs, z_latent)
    e_pc = sum(energy(node.energy, z_latent, compute_mu(node, params, inputs)))
    return e_pc + sum(_hopfield_energy(node, params, z_latent))
end

# forward override so state.energy carries E_pc + E_hop (the generic forward sets
# E_pc only). Gradients come from the seam via the energy_kernel override above.
function forward(
    node::StorkeyHopfield, params::NodeParams, inputs::AbstractDict, state::NodeState
)
    z_mu = compute_mu(node, params, inputs)
    ns = update_state(state; pre_activation=z_mu, z_mu=z_mu, error=state.z_latent .- z_mu)
    ns = energy_functional(node, ns)                                   # E_pc per sample
    ns = update_state(
        ns; energy=ns.energy .+ _hopfield_energy(node, params, state.z_latent)
    )
    return sum(ns.energy), ns
end
