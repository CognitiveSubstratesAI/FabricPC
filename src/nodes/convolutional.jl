# ConvNode (1D/2D/3D) — C-01. Port of fabricpc/nodes/convolutional.py.
#
# Spatial rank is inferred from `length(shape) - 1`:
#   length(shape)==2 -> 1D (L_out, C_out) | ==3 -> 2D (H,W,C_out) | ==4 -> 3D (D,H,W,C_out)
#
# Like upstream, this node implements ONLY the prediction; every gradient comes from the generic
# autodiff seam (nodes/autodiff.jl). That mirrors upstream exactly: `ConvNode` defines only
# get_slots/get_weight_fan_in/initialize_params/forward, and NodeBase supplies both gradient
# paths via `jax.value_and_grad` (base.py:507-551). So the ONLY math here is `compute_mu`.
#
# The windowed arithmetic (cross-correlation, JAX-exact SAME/VALID padding, the im2col plan)
# lives in nodes/windowed.jl, gated against the upstream oracle — see that file's header for the
# three silent-divergence traps it encodes.

"""
    ConvNode(shape, name, kernel_size; stride, padding, activation, energy, use_bias,
             weight_init, bias_init, latent_init)

Unified convolutional node. `shape` is the output shape EXCLUDING batch, channels-last:
`(spatial…, C_out)`. `kernel_size` is required (upstream has no default either).

Upstream defaults, matched: `activation=ReLUActivation()`, `energy=GaussianEnergy()`,
`use_bias=true`, `weight_init=KaimingInitializer()`, `bias_init=ZerosInitializer()`,
`latent_init=NormalInitializer()`, `padding="SAME"` (note the pooling nodes default to `"VALID"`),
`stride` all-ones. Dilation is NOT supported (upstream always calls `lax.conv_general_dilated`
with the default dilation of 1).

`padding` accepts `"SAME"`/`"VALID"` or a sequence of spatial `(low, high)` pairs, and is
normalised ONCE here into a [`PadSpec`](@ref) — a type, not a string. Two independent reasons:
Enzyme rejects `uppercase(::String)` inside a differentiated kernel, and a normalised-once type
cannot be misspelled at a call site (decisions.md §29).
"""
struct ConvNode{R,P<:PadSpec} <: AbstractNode
    shape::Tuple
    name::String
    kernel_size::NTuple{R,Int}
    stride::NTuple{R,Int}
    padding::P
    activation::AbstractActivation
    energy::AbstractEnergy
    use_bias::Bool
    weight_init::AbstractInitializer
    bias_init::AbstractInitializer
    latent_init::AbstractInitializer
end

function ConvNode(
    shape,
    name::AbstractString,
    kernel_size;
    stride=nothing,
    padding="SAME",
    activation::AbstractActivation=ReLUActivation(),
    energy::AbstractEnergy=GaussianEnergy(),
    use_bias::Bool=true,
    weight_init::AbstractInitializer=KaimingInitializer(),
    bias_init::AbstractInitializer=ZerosInitializer(),
    latent_init::AbstractInitializer=NormalInitializer()
)
    shp = Tuple(shape)
    R = length(shp) - 1
    R in (1, 2, 3) || throw(ArgumentError(
        "ConvNode $name: shape must be (spatial…, C_out) with spatial rank 1, 2 or 3; " *
        "got shape=$shp (spatial rank $R)."
    ))
    ks = NTuple{R,Int}(kernel_size)
    st = stride === nothing ? ntuple(_ -> 1, R) : NTuple{R,Int}(stride)   # upstream: all-ones
    p = padspec(padding, R)
    return ConvNode{R,typeof(p)}(
        shp, String(name), ks, st, p, activation, energy, use_bias,
        weight_init, bias_init, latent_init
    )
end

get_slots(::ConvNode) = Dict("in" => SlotSpec("in", true))

"""fan_in = C_in × ∏(kernel_size) — overrides the NodeBase default (C_in only).
Port of `convolutional.py:137-141`. This is what makes Kaiming (ConvNode's DEFAULT weight_init)
size a conv kernel correctly; see [`_fans`](@ref) / register C-08."""
get_weight_fan_in(n::ConvNode, source_shape::Tuple) = source_shape[end] * prod(n.kernel_size)

"""
    initialize_params(node::ConvNode, rng, node_shape, input_shapes, weight_init) -> NodeParams

Kernels `(kernel_size…, C_in, C_out)` per in-edge; bias `(1,…,1, C_out)` (rank R+2, broadcasting
over batch and every spatial axis). Port of `convolutional.py:143-202`.

Shape validation fires HERE, not at construction — the input shapes it checks against are only
known once the graph is wired (upstream's own reasoning, `convolutional.py:19-23`).
`channels_preserved=false`: a conv sets its own C_out via the kernel.
"""
function initialize_params(
    node::ConvNode{R},
    rng::AbstractRNG,
    node_shape::Tuple,
    input_shapes::AbstractDict,
    weight_init::AbstractInitializer
) where {R}
    validate_windowed_output(
        node_shape, input_shapes, node.kernel_size, node.stride, node.padding;
        op_name="ConvNode", channels_preserved=false
    )
    out_channels = node_shape[end]
    weights = Dict{String,Array{Float32}}()
    for (edge_key, in_shape) in input_shapes
        occursin(":in", edge_key) || throw(
            ArgumentError("ConvNode requires the 'in' slot; got edge key $edge_key")
        )
        in_channels = in_shape[end]
        # (kernel…, C_in, C_out) — upstream's HWIO/LIO/DHWIO layout, used verbatim. The
        # shape-aware Kaiming/Xavier derive fan_in = ∏(kernel)*C_in from it themselves (`_fans`).
        weights[edge_key] = initialize(rng, (node.kernel_size..., in_channels, out_channels), weight_init)
    end
    biases = Dict{String,Array{Float32}}()
    if node.use_bias
        biases["b"] = initialize(rng, (ntuple(_ -> 1, R + 1)..., out_channels), node.bias_init)
    end
    return NodeParams(weights, biases)
end

"""
    _conv_pre(node::ConvNode, params, inputs) -> (B, out_spatial…, C_out)

The pre-activation: `Σ_edges crosscorr(x_e, W_e) (+ b)`. Port of `convolutional.py:218-229`.

Starts from `zeros` and folds the edges in `_edge_keys` order — i.e. `in_edges` order, matching
upstream's `for edge_key, x in inputs.items()` (a Python dict iterates in INSERTION order) and
this port's own canonicalisation (`5663c3f`). Float `+` is non-associative, so that order is
semantics, not convenience.
"""
function _conv_pre(node::ConvNode, params::NodeParams, inputs)
    ks = _edge_keys(inputs)                       # @nograd; indexable for OrderedDict AND SoA
    x1 = inputs[ks[1]]
    pre = zeros(Float32, size(x1, 1), node.shape...)
    for k in ks                                   # in_edges order — see the docstring
        pre = pre + _conv_im2col(inputs[k], params.weights[k], node.stride, node.padding)
    end
    node.use_bias && (pre = pre .+ params.biases["b"])   # (1,…,1,C_out) broadcasts
    return pre
end

# The seam: implement ONLY the prediction; forward/both gradient paths are generic.
compute_mu(node::ConvNode, params::NodeParams, inputs) =
    forward(node.activation, _conv_pre(node, params, inputs))

