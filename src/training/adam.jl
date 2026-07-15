# AdamW optimizer. The muPC training recipe (mupc_demo.py) uses optax.adamw —
# muPC's `weight_grad_scale = 1` ("the optimizer handles magnitude") assumes an
# ADAPTIVE optimizer, so plain SGD does not realize the parameterization. This is
# a hand-rolled AdamW over the GraphParams tree (same walk as sgd_update),
# faithful to optax.adamw: Adam moments + DECOUPLED weight decay.

_zeros_like(np::NodeParams) = NodeParams(
    Dict{String, Array{Float32}}(k => zero(w) for (k, w) in np.weights),
    Dict{String, Array{Float32}}(k => zero(b) for (k, b) in np.biases)
)
_zeros_like(p::GraphParams) =
    GraphParams(Dict{String, NodeParams}(k => _zeros_like(np) for (k, np) in p.nodes))

"""
    AdamW(params; lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)

AdamW optimizer state for a `GraphParams` tree (first/second moments + step
counter). Faithful to `optax.adamw`. Defaults match `mupc_demo.py`
(lr 0.002, weight_decay 0.01 set explicitly there).
"""
mutable struct AdamW
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    weight_decay::Float32
    t::Int
    m::GraphParams
    v::GraphParams
end

function AdamW(
    params::GraphParams;
    lr=0.002,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0
)
    return AdamW(
        Float32(lr),
        Float32(beta1),
        Float32(beta2),
        Float32(eps),
        Float32(weight_decay),
        0,
        _zeros_like(params),
        _zeros_like(params)
    )
end

# In-place moment update + decoupled-weight-decay param step for one dict of
# tensors (weights or biases). Mutates the moment dicts `md`, `vd`; returns new params.
function _adamw_dict!(pd, gd, md, vd, lr, b1, b2, eps, wd, bc1, bc2)
    out = Dict{String, Array{Float32}}()
    for (k, w) in pd
        g = gd[k]
        mk = md[k]
        vk = vd[k]
        mk .= b1 .* mk .+ (1 - b1) .* g
        vk .= b2 .* vk .+ (1 - b2) .* (g .^ 2)
        out[k] = w .- lr .* ((mk ./ bc1) ./ (sqrt.(vk ./ bc2) .+ eps) .+ wd .* w)
    end
    return out
end

"""
    step!(opt::AdamW, params, grads) -> GraphParams

One AdamW update. Mutates `opt` (moments + step); returns the new params.
"""
function step!(opt::AdamW, params::GraphParams, grads::GraphParams)
    opt.t += 1
    bc1 = 1.0f0 - opt.beta1^opt.t
    bc2 = 1.0f0 - opt.beta2^opt.t
    new_nodes = Dict{String, NodeParams}()
    for (name, p) in params.nodes
        g = grads.nodes[name]
        m = opt.m.nodes[name]
        v = opt.v.nodes[name]
        w = _adamw_dict!(
            p.weights, g.weights, m.weights, v.weights,
            opt.lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, bc1, bc2
        )
        b = _adamw_dict!(
            p.biases, g.biases, m.biases, v.biases,
            opt.lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, bc1, bc2
        )
        new_nodes[name] = NodeParams(w, b)
    end
    return GraphParams(new_nodes)
end

"""
    train_step!(opt::AdamW, params, batch, structure, rng) -> (params, energy, final_state)

One PC training step with an AdamW weight update (the muPC training recipe's
optimizer). Same inference + local-gradient path as `train_step`, but the weight
update is AdamW instead of plain SGD.
"""
function train_step!(
    opt::AdamW,
    params::GraphParams,
    batch::AbstractDict,
    structure::GraphStructure,
    rng::AbstractRNG
)
    grads, energy, final_state = get_graph_param_gradient(params, batch, structure, rng)
    params = step!(opt, params, grads)
    return params, energy, final_state
end
