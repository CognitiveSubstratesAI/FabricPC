# MaxPool / AvgPool — C-01. Port of fabricpc/nodes/pooling.py.
#
# Parameter-free windowed reductions. Like upstream's `_PoolBase`, these implement only the
# prediction (sum the in-edges → reduce → activation); both gradient paths come from the generic
# autodiff seam, exactly as upstream inherits them from NodeBase's `jax.value_and_grad`.
#
# The reductions themselves live in nodes/windowed.jl (`_pool_max` / `_pool_avg` /
# `_pool_avg_global`) and are gated against the upstream oracle — including the two traps that
# make this node family treacherous: MaxPool's `-Inf32` fill (never `typemin`) and AvgPool's
# `count_include_pad=false` divisor.

"""Parameter-free pooling nodes ([`MaxPool`](@ref), [`AvgPool`](@ref)). Julia analogue of
upstream's `_PoolBase`: shared slots, `fan_in = 1`, empty params, and the shared forward
template (sum in-edges → subclass reduction → activation)."""
abstract type PoolNode <: AbstractNode end

get_slots(::PoolNode) = Dict("in" => SlotSpec("in", true))

"""No weight matrix ⇒ 1, matching the IdentityNode/SkipConnection convention (upstream
`pooling.py:92-100`). With `fan_in = 1` the muPC formula `a = gain/√(fan_in·K_slot·L)` reduces to
`a = gain/√(K_slot·L)` — compensating only for multi-edge summation variance, not for a
non-existent weight matrix. Returning `C_in` (the NodeBase default) would silently attenuate
every pool by `1/√C_in`."""
get_weight_fan_in(::PoolNode, ::Tuple) = 1

"""Pooling has no learnable parameters. Windowed mode validates HERE — the only place the input
shapes are known (upstream `pooling.py:102-132`); `channels_preserved=true` because pooling
cannot change the channel count. Global mode has no spatial output, so it is skipped (its rank-1
shape is checked at construction instead)."""
function initialize_params(
    node::PoolNode, ::AbstractRNG, node_shape::Tuple, input_shapes::AbstractDict,
    ::Union{AbstractInitializer, Nothing}
)
    _pool_validate(node, node_shape, input_shapes)
    return NodeParams(Dict{String, Array{Float32}}(), Dict{String, Array{Float32}}())
end

"""Sum the in-edges as a STRICT LEFT FOLD in `in_edges` order.

NOT `sum(…)`: upstream's `sum(inputs.values())` (`pooling.py:151`) is a strict left fold, while
Julia's `sum` reassociates pairwise above blocksize 1024. Float `+` is non-associative, so this is
a guarantee, not a coincidence — and the order is `_edge_keys` = `in_edges`, matching upstream's
insertion-order dict iteration.

NOT `foldl` either, despite that being the literal spelling of "strict left fold": Zygote's
`foldl` pullback BoundsErrors at index [0] when the folded collection is EMPTY, which is exactly
the single-in-edge case (`ks[2:end] == []`) — i.e. every pooling node in practice. Caught by the
conformance tier, not by the standalone kernel probes, which call `_pool_max`/`_pool_avg`
directly and never route through here. The explicit loop below is the same strict left fold and
is what `_conv_pre` already proves differentiable.
"""
function _pool_sum(inputs)
    ks = _edge_keys(inputs)
    s = inputs[ks[1]]
    for i in 2:length(ks)          # empty range for a single edge — no pullback edge case
        s = s + inputs[ks[i]]
    end
    return s
end

# ---------------------------------------------------------------------------------------------

"""
    MaxPool(shape, name, window_shape; stride, padding, activation, energy, latent_init)

Windowed max pooling, channels-last `(B, spatial…, C)`. No learnable params.

Upstream defaults, matched: `padding="VALID"` (note ConvNode defaults to `"SAME"`), `stride`
defaults to `window_shape` (non-overlapping), `activation=IdentityActivation()`,
`energy=GaussianEnergy()`, `latent_init=NormalInitializer()`.

Sets `reject_pad_ge_window=true` (`pooling.py:214`): max pooling fills with `-Inf`, so a window
fully covered by explicit padding would output `-Inf`, and the validator rejects that config.
"""
struct MaxPool{R, P <: PadSpec} <: PoolNode
    shape::Tuple
    name::String
    window_shape::NTuple{R, Int}
    stride::NTuple{R, Int}
    padding::P
    activation::AbstractActivation
    energy::AbstractEnergy
    latent_init::AbstractInitializer
end

function MaxPool(
    shape, name::AbstractString, window_shape;
    stride=nothing, padding="VALID",
    activation::AbstractActivation=IdentityActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    shp = Tuple(shape)
    R = length(shp) - 1
    # Same spatial-rank guard ConvNode has. Its ABSENCE here was an asymmetry the C-01
    # adversarial audit caught: ConvNode rejected a rank-4-spatial shape at construction while
    # MaxPool/AvgPool accepted it and failed later (or silently), for no reason other than that
    # nobody wrote the check. `_window_plan`/`_pads` are the same rank-1/2/3 machinery either way.
    _pool_rank_guard(R, shp, name, "MaxPool")
    ws = NTuple{R, Int}(window_shape)
    st = stride === nothing ? ws : NTuple{R, Int}(stride)     # upstream: defaults to the window
    p = padspec(padding, R)
    return MaxPool{R, typeof(p)}(
        shp, String(name), ws, st, p, activation, energy, latent_init
    )
end

"""Spatial-rank guard shared by MaxPool/AvgPool, mirroring ConvNode's. Windowed pooling is
1D/2D/3D like conv; a shape outside that has no `_window_plan` and must fail CLOSED at
construction rather than deeper in."""
_pool_rank_guard(R::Int, shp::Tuple, name, op::String) =
    R in (1, 2, 3) || throw(
        ArgumentError(
            "$op $name: shape must be (spatial…, C) with spatial rank 1, 2 or 3; " *
            "got shape=$shp (spatial rank $R). For global pooling use " *
            "`AvgPool(shape, name; global_pool=true)` with a rank-1 (C,) shape."
        )
    )

_pool_validate(n::MaxPool, node_shape::Tuple, input_shapes::AbstractDict) =
    validate_windowed_output(
        node_shape, input_shapes, n.window_shape, n.stride, n.padding;
        op_name="MaxPool", channels_preserved=true, reject_pad_ge_window=true
    )

_pool_reduce(n::MaxPool, x) = _pool_max(x, n.stride, n.padding, n.window_shape)

# ---------------------------------------------------------------------------------------------

"""
    AvgPool(shape, name; window_shape, stride, padding, global_pool, count_include_pad,
            activation, energy, latent_init)

Average pooling — windowed or global.
* Windowed (`global_pool=false`): `(B, spatial…, C) -> (B, spatial'…, C)`; `window_shape` required.
* Global (`global_pool=true`): averages over ALL spatial dims, `(B, spatial…, C) -> (B, C)`;
  `shape` must be rank-1 `(C,)` and `window_shape` is ignored.

`count_include_pad` (windowed only, default `true` matching PyTorch): `true` divides each window
by the FULL window volume (padded cells count as zeros); `false` divides by the number of REAL
(non-padded) cells. The two differ only where padding reaches a window — i.e. never under the
default VALID padding, which is exactly what makes a wrong divisor invisible. See
`_pool_avg`'s docstring; measured non-vacuous (the modes differ by 0.807 on the oracle's fixture).
"""
struct AvgPool{R, P <: PadSpec} <: PoolNode
    shape::Tuple
    name::String
    window_shape::NTuple{R, Int}
    stride::NTuple{R, Int}
    padding::P
    global_pool::Bool
    count_include_pad::Bool
    activation::AbstractActivation
    energy::AbstractEnergy
    latent_init::AbstractInitializer
end

function AvgPool(
    shape, name::AbstractString;
    window_shape=nothing, stride=nothing, padding="VALID",
    global_pool::Bool=false, count_include_pad::Bool=true,
    activation::AbstractActivation=IdentityActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    shp = Tuple(shape)
    if global_pool
        # Global collapses every spatial dim ⇒ the declared shape must be rank-1 (C,). Not a
        # windowed check, so it stays at construction (upstream `pooling.py:278-290`).
        length(shp) == 1 || throw(
            ArgumentError(
                "AvgPool global_pool=true requires a rank-1 shape (C,), got shape=$shp."
            )
        )
        # Global mode IGNORES window/stride/padding (`pooling.py:288-290`: upstream sets
        # window_shape/stride to () and never reads padding — it means over every spatial axis).
        # So accept-and-ignore, do NOT validate: `padspec(padding, 0)` used to REJECT any explicit
        # pad here ("2 spatial pairs but spatial_rank is 0") on a config upstream accepts silently.
        # Matching upstream means being equally permissive where it is permissive — §29 cuts both
        # ways. Found by the C-01 adversarial audit (upstream-diff lens).
        p = ValidPad()
        return AvgPool{0, typeof(p)}(shp, String(name), (), (), p, true, count_include_pad,
            activation, energy, latent_init)
    end
    window_shape === nothing && throw(ArgumentError(
        "AvgPool: window_shape is required when global_pool=false."
    ))
    R = length(shp) - 1
    _pool_rank_guard(R, shp, name, "AvgPool")     # same guard ConvNode has (audit finding #11)
    ws = NTuple{R, Int}(window_shape)
    st = stride === nothing ? ws : NTuple{R, Int}(stride)
    p = padspec(padding, R)
    return AvgPool{R, typeof(p)}(shp, String(name), ws, st, p, false, count_include_pad,
        activation, energy, latent_init)
end

_pool_validate(n::AvgPool, node_shape::Tuple, input_shapes::AbstractDict) =
    if n.global_pool
        nothing
    else
        validate_windowed_output(
            node_shape, input_shapes, n.window_shape, n.stride, n.padding;
            op_name="AvgPool", channels_preserved=true
        )
    end

_pool_reduce(n::AvgPool, x) =
    if n.global_pool
        _pool_avg_global(x)
    else
        _pool_avg(x, n.stride, n.padding, n.window_shape, n.count_include_pad)
    end

# ---------------------------------------------------------------------------------------------
# The seam: only the prediction. Both gradient paths are generic.

compute_mu(node::PoolNode, params::NodeParams, inputs) =
    forward(node.activation, _pool_reduce(node, _pool_sum(inputs)))
