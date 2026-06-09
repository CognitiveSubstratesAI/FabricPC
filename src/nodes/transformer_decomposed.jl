# Decomposed (fully-PC) transformer block — port of the middle stages of
# fabricpc/nodes/transformer_v2.py. Each sub-component is its OWN PC node with its
# own latent + local energy, so predictive coding operates at every stage, not just
# the block boundary:
#
#   x → MhaResidual(z=x+W_o·MHA(LN x)) → LnMlp1(z=GELU(LN·W_ff1)) → Mlp2Residual(z=res+·W_ff2)
#                                    └──────────── skip (residual) ────────────┘
#
# Gradients come from the Phase-D Enzyme seam (each node implements only
# `compute_mu`). Reuses the `_tb_*` helpers in transformer.jl. UNLIKE the monolithic
# `TransformerBlock`, residuals are plain (z = x + mha, z = res + mlp2) — no 1/√2 or
# √S scaling (matches transformer_v2). v1: causal/full self-attention, no mask input.
# Embedding/VocabProjection (discrete-token I/O) are a separate follow-up.

# ============================================================================
# MhaResidualNode — z_mu = x + W_o·MHA(LayerNorm(x))
# ============================================================================
struct MhaResidualNode <: AbstractNode
    shape::Tuple
    name::String
    num_heads::Int
    use_rope::Bool
    causal::Bool
    energy::AbstractEnergy
    weight_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function MhaResidualNode(
    shape,
    name::AbstractString;
    num_heads::Int=4,
    use_rope::Bool=true,
    causal::Bool=true,
    energy::AbstractEnergy=GaussianEnergy(),
    weight_init::AbstractInitializer=NormalInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    shp = Tuple(shape)
    embed = shp[end]
    embed % num_heads == 0 ||
        throw(
            ArgumentError("embed_dim ($embed) must be divisible by num_heads ($num_heads)")
        )
    return MhaResidualNode(
        shp, String(name), num_heads, use_rope, causal, energy, weight_init, latent_init
    )
end

get_slots(::MhaResidualNode) = Dict("in" => SlotSpec("in", false))
get_weight_fan_in(::MhaResidualNode, source_shape::Tuple) = source_shape[end]

function initialize_params(
    node::MhaResidualNode, rng::AbstractRNG, node_shape::Tuple,
    input_shapes::AbstractDict, weight_init::AbstractInitializer
)
    embed = node_shape[end]
    sq = NormalInitializer(; std=1.0 / sqrt(embed))
    W(m, n) = initialize(rng, (m, n), sq)
    weights = Dict{String, Matrix{Float32}}(
        "W_q" => W(embed, embed), "W_k" => W(embed, embed),
        "W_v" => W(embed, embed), "W_o" => W(embed, embed),
        "ln_gamma" => ones(Float32, 1, embed)
    )
    biases = Dict{String, Matrix{Float32}}(
        "b_q" => zeros(Float32, 1, embed), "b_k" => zeros(Float32, 1, embed),
        "b_v" => zeros(Float32, 1, embed), "b_o" => zeros(Float32, 1, embed),
        "ln_beta" => zeros(Float32, 1, embed)
    )
    return NodeParams(weights, biases)
end

function compute_mu(node::MhaResidualNode, params::NodeParams, inputs)
    x = first(values(inputs))
    E = size(x, 3)
    xn = _tb_layernorm(
        x, reshape(params.weights["ln_gamma"], 1, 1, E),
        reshape(params.biases["ln_beta"], 1, 1, E)
    )
    mha = _tb_mha(node.num_heads, node.use_rope, node.causal, params, xn)
    return x .+ mha
end

# ============================================================================
# LnMlp1Node — z_mu = activation(LayerNorm(x) · W_ff1 + b_ff1)   shape (S, ff_dim)
# ============================================================================
struct LnMlp1Node <: AbstractNode
    shape::Tuple
    name::String
    activation::AbstractActivation
    energy::AbstractEnergy
    weight_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function LnMlp1Node(
    shape,
    name::AbstractString;
    activation::AbstractActivation=GeluActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    weight_init::AbstractInitializer=KaimingInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    return LnMlp1Node(
        Tuple(shape), String(name), activation, energy, weight_init, latent_init
    )
end

get_slots(::LnMlp1Node) = Dict("in" => SlotSpec("in", false))

function initialize_params(
    node::LnMlp1Node, rng::AbstractRNG, node_shape::Tuple,
    input_shapes::AbstractDict, weight_init::AbstractInitializer
)
    ff = node_shape[end]
    in_key = first(k for k in keys(input_shapes) if endswith(k, ":in"))
    embed = input_shapes[in_key][end]
    weights = Dict{String, Matrix{Float32}}(
        "ln_gamma" => ones(Float32, 1, embed),
        "W_ff1" => initialize(rng, (embed, ff), weight_init)
    )
    biases = Dict{String, Matrix{Float32}}(
        "ln_beta" => zeros(Float32, 1, embed), "b_ff1" => zeros(Float32, 1, ff)
    )
    return NodeParams(weights, biases)
end

function compute_mu(node::LnMlp1Node, params::NodeParams, inputs)
    x = first(values(inputs))
    E = size(x, 3)
    xn = _tb_layernorm(
        x, reshape(params.weights["ln_gamma"], 1, 1, E),
        reshape(params.biases["ln_beta"], 1, 1, E)
    )
    h = _tb_dense(xn, params.weights["W_ff1"], reshape(params.biases["b_ff1"], 1, 1, :))
    return forward(node.activation, h)
end

# ============================================================================
# Mlp2ResidualNode — z_mu = residual + (mlp1_in · W_ff2 + b_ff2)
# Two slots: "in" (from LnMlp1, (S,ff)) and "residual" (skip from MhaResidual,
# (S,embed)). The skip's input gradient flows back to its source via the seam.
# ============================================================================
struct Mlp2ResidualNode <: AbstractNode
    shape::Tuple
    name::String
    energy::AbstractEnergy
    weight_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function Mlp2ResidualNode(
    shape,
    name::AbstractString;
    energy::AbstractEnergy=GaussianEnergy(),
    weight_init::AbstractInitializer=XavierInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    return Mlp2ResidualNode(Tuple(shape), String(name), energy, weight_init, latent_init)
end

get_slots(::Mlp2ResidualNode) = Dict(
    "in" => SlotSpec("in", false),
    "residual" => SlotSpec(
        "residual", false; is_variance_scalable=false, is_skip_connection=true
    )
)

function initialize_params(
    node::Mlp2ResidualNode, rng::AbstractRNG, node_shape::Tuple,
    input_shapes::AbstractDict, weight_init::AbstractInitializer
)
    embed = node_shape[end]
    in_key = first(k for k in keys(input_shapes) if endswith(k, ":in"))
    ff = input_shapes[in_key][end]
    weights = Dict{String, Matrix{Float32}}(
        "W_ff2" => initialize(rng, (ff, embed), weight_init)
    )
    biases = Dict{String, Matrix{Float32}}("b_ff2" => zeros(Float32, 1, embed))
    return NodeParams(weights, biases)
end

function compute_mu(node::Mlp2ResidualNode, params::NodeParams, inputs)
    in_key = first(k for k in keys(inputs) if endswith(k, ":in"))
    res_key = first(k for k in keys(inputs) if endswith(k, ":residual"))
    mlp1 = inputs[in_key]
    res = inputs[res_key]
    mlp2 = _tb_dense(
        mlp1, params.weights["W_ff2"], reshape(params.biases["b_ff2"], 1, 1, :)
    )
    return res .+ mlp2
end
