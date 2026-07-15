# Natural-gradient optimizer transforms. Port of fabricpc/training/natural_gradients.py.
#
# Online Fisher-information preconditioners over the GraphParams gradient tree —
# the cheap diagonal / layer-wise approximations from the upstream optax transforms
# `scale_by_natural_gradient_{diag,layerwise}`. (Upstream threads optax state; we
# use a mutable struct holding the Fisher EMA, mirroring AdamW in adam.jl.)
#
# NOTE: this is the classic Amari Fisher-diagonal natural gradient — NOT the
# ActPC-Geom Wasserstein natural gradient (metric tensor G = JᵀL†J). It is upstream
# FabricPC's natural-gradient surface; the Wasserstein/information-geometry layer is
# a separate (unbuilt) effort.

# ── Diagonal Fisher preconditioner ────────────────────────────────────────────

"""
    NaturalGradientDiag(params; fisher_decay = 0.95, damping = 1e-3)

EMA diagonal-Fisher preconditioner state for a `GraphParams` tree. Port of
`scale_by_natural_gradient_diag`.
"""
mutable struct NaturalGradientDiag
    fisher_decay::Float32
    damping::Float32
    fisher::GraphParams           # per-element Fisher EMA, same tree as params
end
function NaturalGradientDiag(params::GraphParams; fisher_decay=0.95, damping=1e-3)
    0.0 <= fisher_decay < 1.0 || error("fisher_decay must be in [0,1), got $fisher_decay")
    damping > 0.0 || error("damping must be > 0, got $damping")
    NaturalGradientDiag(Float32(fisher_decay), Float32(damping), _zeros_like(params))
end

"""
    precondition!(ng::NaturalGradientDiag, grads::GraphParams) -> GraphParams

Update the Fisher EMA (`f ← decay·f + (1−decay)·g²`) and return the preconditioned
gradient (`g / (f + damping)`), elementwise over every weight/bias.
"""
function precondition!(ng::NaturalGradientDiag, grads::GraphParams)
    omd = 1.0f0 - ng.fisher_decay
    out = Dict{String, NodeParams}()
    for (name, g_np) in grads.nodes
        f_np = ng.fisher.nodes[name]
        nf_w = Dict{String, Array{Float32}}()
        out_w = Dict{String, Array{Float32}}()
        for (k, g) in g_np.weights
            f = ng.fisher_decay .* f_np.weights[k] .+ omd .* (g .^ 2)
            nf_w[k] = f
            out_w[k] = g ./ (f .+ ng.damping)
        end
        nf_b = Dict{String, Array{Float32}}()
        out_b = Dict{String, Array{Float32}}()
        for (k, g) in g_np.biases
            f = ng.fisher_decay .* f_np.biases[k] .+ omd .* (g .^ 2)
            nf_b[k] = f
            out_b[k] = g ./ (f .+ ng.damping)
        end
        ng.fisher.nodes[name] = NodeParams(nf_w, nf_b)
        out[name] = NodeParams(out_w, out_b)
    end
    return GraphParams(out)
end

# ── Layer-wise scalar-Fisher preconditioner ───────────────────────────────────

"""
    NaturalGradientLayerwise(; fisher_decay = 0.95, damping = 1e-3)

One scalar Fisher estimate per parameter tensor (cheaper, often more stable than
the diagonal). Port of `scale_by_natural_gradient_layerwise`. The per-leaf scalar
state is built lazily on the first `precondition!`.
"""
mutable struct NaturalGradientLayerwise
    fisher_decay::Float32
    damping::Float32
    fisher::Dict{String, Float32}     # key "node/w/edge" or "node/b/edge" → scalar EMA
end
function NaturalGradientLayerwise(; fisher_decay=0.95, damping=1e-3)
    0.0 <= fisher_decay < 1.0 || error("fisher_decay must be in [0,1), got $fisher_decay")
    damping > 0.0 || error("damping must be > 0, got $damping")
    NaturalGradientLayerwise(
        Float32(fisher_decay), Float32(damping), Dict{String, Float32}()
    )
end

"""
    precondition!(ng::NaturalGradientLayerwise, grads::GraphParams) -> GraphParams

Update each tensor's scalar Fisher (`f ← decay·f + (1−decay)·mean(g²)`) and return
`g / (f + damping)`.
"""
function precondition!(ng::NaturalGradientLayerwise, grads::GraphParams)
    omd = 1.0f0 - ng.fisher_decay
    out = Dict{String, NodeParams}()
    for (name, g_np) in grads.nodes
        out_w = Dict{String, Array{Float32}}()
        for (k, g) in g_np.weights
            key = "$name/w/$k"
            f =
                ng.fisher_decay * get(ng.fisher, key, 0.0f0) +
                omd * Float32(sum(abs2, g) / length(g))
            ng.fisher[key] = f
            out_w[k] = g ./ (f + ng.damping)
        end
        out_b = Dict{String, Array{Float32}}()
        for (k, g) in g_np.biases
            key = "$name/b/$k"
            f =
                ng.fisher_decay * get(ng.fisher, key, 0.0f0) +
                omd * Float32(sum(abs2, g) / length(g))
            ng.fisher[key] = f
            out_b[k] = g ./ (f + ng.damping)
        end
        out[name] = NodeParams(out_w, out_b)
    end
    return GraphParams(out)
end
