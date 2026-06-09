# TransformerBlock node — a predictive-coding transformer (PC, NOT backprop).
# Port of fabricpc/nodes/transformer.py, REDEVELOPED natively in Julia rather than
# transliterated from JAX: column-major-first array ops + Base/LinearAlgebra (no
# NNlib batched_mul). Trained from scratch, so we need a correct, internally
# consistent multi-head-attention-with-RoPE forward — not numpy-byte-compatibility.
#
# The node implements ONLY `compute_mu` (the block forward). Its LOCAL predictive-
# coding gradients — weight (learning) and latent+input (inference) — come for FREE
# from the Phase-D Enzyme seam (src/nodes/autodiff.jl): Enzyme differentiates this
# one node's local energy. This is the PC-transformer — a transformer whose every
# parameter is learned by local PC, with NO backprop through the network.
#
# Forward (rank-3 (batch, seq, embed) latents):
#   x → LayerNorm → MHA(+RoPE) → ·√seq → (x + ·)/√2 → LayerNorm → FFN → (·+ff)/√2
# Validated: vectorized == explicit-loop oracle (math) and Enzyme == finite-diff
# (gradient), both to Float32 tol. See test_transformer.jl.
#
# All parameters are 2D `Matrix{Float32}` so they fit NodeParams unchanged (biases
# and LayerNorm γ/β are stored (1, embed) and reshaped to (1,1,embed) here for the
# broadcast). v1: full (non-causal) attention, single "in" slot, no mask.

"""
    TransformerBlock(shape, name; num_heads = 4, activation = IdentityActivation(),
                     energy = GaussianEnergy(), internal_activation = GeluActivation(),
                     ff_dim = 4*embed_dim, use_rope = true,
                     weight_init = NormalInitializer(), latent_init = NormalInitializer())

A pre-norm transformer block as a single PC node. `shape = (seq_len, embed_dim)`;
`embed_dim` must be divisible by `num_heads`. Single multi-input-free `"in"` slot.
Local PC gradients via the Enzyme autodiff seam (`using Enzyme` to activate).
"""
struct TransformerBlock <: AbstractNode
    shape::Tuple
    name::String
    num_heads::Int
    activation::AbstractActivation
    energy::AbstractEnergy
    internal_activation::AbstractActivation
    ff_dim::Int
    use_rope::Bool
    weight_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function TransformerBlock(
    shape,
    name::AbstractString;
    num_heads::Int=4,
    activation::AbstractActivation=IdentityActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    internal_activation::AbstractActivation=GeluActivation(),
    ff_dim::Union{Int, Nothing}=nothing,
    use_rope::Bool=true,
    weight_init::AbstractInitializer=NormalInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    shp = Tuple(shape)
    embed = shp[end]
    embed % num_heads == 0 ||
        throw(
            ArgumentError("embed_dim ($embed) must be divisible by num_heads ($num_heads)")
        )
    ffd = ff_dim === nothing ? 4 * embed : ff_dim
    return TransformerBlock(
        shp, String(name), num_heads, activation, energy, internal_activation,
        ffd, use_rope, weight_init, latent_init
    )
end

get_slots(::TransformerBlock) = Dict("in" => SlotSpec("in", false))

# muPC fan_in = embed_dim (pre-norm LayerNorm absorbs the external forward_scale).
get_weight_fan_in(::TransformerBlock, source_shape::Tuple) = source_shape[end]

function initialize_params(
    node::TransformerBlock,
    rng::AbstractRNG,
    node_shape::Tuple,
    input_shapes::AbstractDict,
    weight_init::AbstractInitializer
)
    embed = node_shape[end]
    ff = node.ff_dim
    sq = NormalInitializer(; std=1.0 / sqrt(embed))            # 1/√d  (Q/K/V/O)
    sf = NormalInitializer(; std=1.0 / sqrt(ff))               # 1/√d_ff (W_ff2)
    W(init, m, n) = initialize(rng, (m, n), init)
    weights = Dict{String, Matrix{Float32}}(
        "W_q" => W(sq, embed, embed), "W_k" => W(sq, embed, embed),
        "W_v" => W(sq, embed, embed), "W_o" => W(sq, embed, embed),
        "W_ff1" => W(KaimingInitializer(), embed, ff),         # He (GELU)
        "W_ff2" => W(sf, ff, embed),
        "ln1_gamma" => ones(Float32, 1, embed), "ln2_gamma" => ones(Float32, 1, embed)
    )
    biases = Dict{String, Matrix{Float32}}(
        "b_q" => zeros(Float32, 1, embed), "b_k" => zeros(Float32, 1, embed),
        "b_v" => zeros(Float32, 1, embed), "b_o" => zeros(Float32, 1, embed),
        "b_ff1" => zeros(Float32, 1, ff), "b_ff2" => zeros(Float32, 1, embed),
        "ln1_beta" => zeros(Float32, 1, embed), "ln2_beta" => zeros(Float32, 1, embed)
    )
    return NodeParams(weights, biases)
end

# --- native forward helpers (column-major, Enzyme-differentiable) --------------

# Dense over the last (feature) axis: x (B,S,E), W (E,O), b3 (1,1,O) -> (B,S,O).
# reshape(x, B*S, E) is layout-correct in column-major (batch & seq are the leading
# fast axes; the contracted feature axis is last), so this is a plain matmul.
function _tb_dense(x, W, b3)
    B, S, E = size(x)
    return reshape(reshape(x, B * S, E) * W, B, S, size(W, 2)) .+ b3
end

# LayerNorm over the last axis; g3, b3 are (1,1,E).
function _tb_layernorm(x, g3, b3; eps=1.0f-5)
    E = size(x, 3)
    mu = sum(x; dims=3) ./ E
    xc = x .- mu
    var = sum(xc .^ 2; dims=3) ./ E
    return g3 .* (xc ./ sqrt.(var .+ eps)) .+ b3
end

# RoPE cos/sin tables: (seq, head_dim÷2).
function _tb_rope_tables(head_dim, seq; theta=10000.0f0)
    half = head_dim ÷ 2
    freqs = [1.0f0 / theta^(Float32(2 * (i - 1)) / head_dim) for i in 1:half]
    ang = [Float32(p - 1) * freqs[i] for p in 1:seq, i in 1:half]
    return cos.(ang), sin.(ang)
end

# Apply RoPE to a head tensor (B,S,Dh). Adjacent-pair convention (1,2),(3,4),…
# (a valid consistent rotation; from-scratch ⇒ no numpy-layout match required).
function _tb_apply_rope(xh, cosA, sinA)
    B, S, Dh = size(xh)
    half = Dh ÷ 2
    xr = reshape(xh, B, S, 2, half)
    x1 = xr[:, :, 1, :]
    x2 = xr[:, :, 2, :]
    c = reshape(cosA, 1, S, half)
    s = reshape(sinA, 1, S, half)
    r1 = x1 .* c .- x2 .* s
    r2 = x1 .* s .+ x2 .* c
    return reshape(
        cat(reshape(r1, B, S, 1, half), reshape(r2, B, S, 1, half); dims=3), B, S, Dh
    )
end

# Multi-head self-attention (full / non-causal). x (B,S,E) -> (B,S,E).
function _tb_mha(node::TransformerBlock, p, x)
    B, S, E = size(x)
    H = node.num_heads
    Dh = E ÷ H
    Q = _tb_dense(x, p.weights["W_q"], reshape(p.biases["b_q"], 1, 1, E))
    K = _tb_dense(x, p.weights["W_k"], reshape(p.biases["b_k"], 1, 1, E))
    V = _tb_dense(x, p.weights["W_v"], reshape(p.biases["b_v"], 1, 1, E))
    cosA, sinA = _tb_rope_tables(Dh, S)
    scale = sqrt(Float32(Dh))
    heads = map(1:H) do h
        cols = ((h - 1) * Dh + 1):(h * Dh)
        Qh = Q[:, :, cols]
        Kh = K[:, :, cols]
        Vh = V[:, :, cols]
        if node.use_rope
            Qh = _tb_apply_rope(Qh, cosA, sinA)
            Kh = _tb_apply_rope(Kh, cosA, sinA)
        end
        outb = map(1:B) do b
            qb = Qh[b, :, :]
            kb = Kh[b, :, :]
            vb = Vh[b, :, :]
            scores = (qb * transpose(kb)) ./ scale
            m = maximum(scores; dims=2)
            ex = exp.(scores .- m)
            (ex ./ sum(ex; dims=2)) * vb
        end
        stack(outb; dims=1)
    end
    concat = cat(heads...; dims=3)
    return _tb_dense(concat, p.weights["W_o"], reshape(p.biases["b_o"], 1, 1, E))
end

"""
    compute_mu(node::TransformerBlock, params, inputs) -> z_mu (batch, seq, embed)

The transformer block forward (the node's prediction of its own latent). Pure array
ops; Enzyme differentiates it for the local PC gradients. Single `"in"` input edge.
"""
function compute_mu(node::TransformerBlock, params::NodeParams, inputs)
    x = first(values(inputs))                       # (B, S, E) — single "in" edge
    E = size(x, 3)
    inv2 = 1.0f0 / sqrt(2.0f0)
    S = size(x, 2)
    xn1 = _tb_layernorm(
        x, reshape(params.weights["ln1_gamma"], 1, 1, E),
        reshape(params.biases["ln1_beta"], 1, 1, E)
    )
    attn = _tb_mha(node, params, xn1) .* sqrt(Float32(S))     # full-attn variance comp.
    xres1 = inv2 .* (x .+ attn)
    xn2 = _tb_layernorm(
        xres1, reshape(params.weights["ln2_gamma"], 1, 1, E),
        reshape(params.biases["ln2_beta"], 1, 1, E)
    )
    ff1 = _tb_dense(xn2, params.weights["W_ff1"], reshape(params.biases["b_ff1"], 1, 1, :))
    ffa = forward(node.internal_activation, ff1)
    ff2 = _tb_dense(ffa, params.weights["W_ff2"], reshape(params.biases["b_ff2"], 1, 1, E))
    z_mu = inv2 .* (xres1 .+ ff2)
    return forward(node.activation, z_mu)            # output activation (Identity by default)
end

# =============================================================================
# Reactant/XLA JIT path: a Dict-free, positional-array form of the block forward,
# traceable by Reactant.@compile AND differentiable by Enzyme UNDER @compile
# (Reactant+Enzyme) — the perf compiler for the eager forward above.
#
# Differs from `compute_mu` ONLY in mechanics, not math: it takes the parameter
# arrays positionally (no Dict — Reactant traces arrays/tuples), biases/γβ as
# (1,1,·), and unrolls heads+batch with `ntuple(…, Val(N))` so the indices are
# LITERAL ⇒ concrete slices (a `map`/`for` over a range makes the index a traced
# value, which breaks `Q[:,:,(h-1)*Dh+1:h*Dh]`). Internal activation is GELU and
# output activation identity (the default node config). Validated equal to
# `compute_mu` (test_transformer.jl) and JIT==eager + Enzyme(JIT)==Enzyme(eager) to
# ~1e-7 (benchmark/transformer_jit.jl). `B` is the (fixed-per-compile) batch size.
function _tb_block_flat(
    x, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, W_ff1, b_ff1, W_ff2, b_ff2,
    ln1g, ln1b, ln2g, ln2b, ::Val{H}, ::Val{B}, ::Val{ROPE}
) where {H, B, ROPE}
    S = size(x, 2)
    E = size(x, 3)
    Dh = E ÷ H
    inv2 = 1.0f0 / sqrt(2.0f0)
    xn1 = _tb_layernorm(x, ln1g, ln1b)
    Q = _tb_dense(xn1, W_q, b_q)
    K = _tb_dense(xn1, W_k, b_k)
    V = _tb_dense(xn1, W_v, b_v)
    cosA, sinA = _tb_rope_tables(Dh, S)
    scale = sqrt(Float32(Dh))
    heads = ntuple(Val(H)) do h
        cols = ((h - 1) * Dh + 1):(h * Dh)
        Qh = Q[:, :, cols]
        Kh = K[:, :, cols]
        Vh = V[:, :, cols]
        if ROPE
            Qh = _tb_apply_rope(Qh, cosA, sinA)
            Kh = _tb_apply_rope(Kh, cosA, sinA)
        end
        outb = ntuple(Val(B)) do b
            qb = Qh[b, :, :]
            kb = Kh[b, :, :]
            vb = Vh[b, :, :]
            sc = (qb * transpose(kb)) ./ scale
            m = maximum(sc; dims=2)
            ex = exp.(sc .- m)
            (ex ./ sum(ex; dims=2)) * vb
        end
        stack(outb; dims=1)
    end
    attn = _tb_dense(cat(heads...; dims=3), W_o, b_o) .* sqrt(Float32(S))
    xres1 = inv2 .* (x .+ attn)
    xn2 = _tb_layernorm(xres1, ln2g, ln2b)
    ff1 = _tb_dense(xn2, W_ff1, b_ff1)
    ff2 = _tb_dense(forward(GeluActivation(), ff1), W_ff2, b_ff2)
    return inv2 .* (xres1 .+ ff2)
end

"""
    flat_block_args(node, params) -> Tuple

Unpack a `TransformerBlock`'s `NodeParams` (Dict) into the positional array tuple
`_tb_block_flat` expects (weights + (1,1,·) biases/γβ), in order. The bridge from
the Dict-based eager params to the Reactant-traceable flat kernel.
"""
function flat_block_args(node::TransformerBlock, params::NodeParams)
    E = node.shape[end]
    r(a, n) = reshape(a, 1, 1, n)
    w = params.weights
    b = params.biases
    return (
        w["W_q"], w["W_k"], w["W_v"], w["W_o"],
        r(b["b_q"], E), r(b["b_k"], E), r(b["b_v"], E), r(b["b_o"], E),
        w["W_ff1"], r(b["b_ff1"], node.ff_dim), w["W_ff2"], r(b["b_ff2"], E),
        r(w["ln1_gamma"], E), r(b["ln1_beta"], E), r(w["ln2_gamma"], E), r(
            b["ln2_beta"], E
        )
    )
end
