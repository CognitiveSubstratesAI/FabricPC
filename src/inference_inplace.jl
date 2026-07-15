# J-06: type-stable, pre-allocated, in-place PC inference (opt-in, isolated).
#
# The eager Dict/NodeState path (run_inference) is type-unstable (NodeState's fields are ::Any;
# 100 runtime-dispatch sites per JET) and allocation-heavy (6565 allocs / 208 MB per call; 40-57%
# GC in a sustained/training loop). This lane fixes BOTH by construction: it works on concrete
# `Matrix{Float32}` buffers held in a persistent pool (no ::Any → no boxing, no dispatch overhead)
# and mutates them in place with `mul!` + fused broadcasts (0 allocations per step). It also bakes
# in the §28 hoist+prune (loop-invariant all-clamped-source forwards computed once; gradients into
# clamped nodes skipped). Measured (benchmark/inplace_inference_poc.jl, hardcoded 3-node PoC):
# bit-identical, 0 allocs, ~5.8x over the already-hoist+pruned path, and eager-fused == XLA on
# per-step compute (decisions.md §28 addendum 3). THIS is the general version.
#
# SCOPE (increment 1) — enforced FAIL-CLOSED by prealloc_inference (`_inplace_supported` +
# explicit guards); anything outside throws at build time, never silently produces wrong numbers:
#   • nodes: Linear / IdentityNode / SkipConnection (any other type is rejected)
#   • energy: GaussianEnergy at ANY precision (read per-node, not assumed 1)
#   • activation: Identity/Tanh on Linear (Identity/Skip bypass their activation in forward)
#   • inference algorithm: InferenceSGD only (NormClip/Momentum change the update rule → rejected)
#   • no muPC scaling (scaling_config must be nothing), no unclamped sinks (reference snaps z:=z_mu)
# TransformerBlock and the non-Gaussian energies stay on run_inference (their backward isn't in the
# flat lane either — see jit_flat.jl / J-01).
#
# Inputs are folded in in_src (== in_edges) order, which matches BOTH upstream JAX (its Python dict
# iterates in insertion order) and — since gather_inputs was switched to an OrderedDict — the
# reference run_inference. Float addition is non-associative, so this shared canonical order is what
# keeps 3+ in-edge nodes bit-identical across all lanes.
#
# VERIFIED (2026-07-14, warm .warm/zygote_env): bit-identical (max|diff|=0.0) to run_inference on
# MNIST-MLP, deep chain, a mixed-source diamond, a 3-in-edge node (multiple key sets), IdentityNode
# and SkipConnection graphs, and per-node precision≠1; per-step 0-alloc proven by step-count
# invariance; 23-27× over stock run_inference. Depends on the LinearAlgebra stdlib (for `mul!`),
# in Project.toml [deps] (UUID 37e2e46d-f89d-539d-b4ee-838fcccc9c8e).
using LinearAlgebra: mul!

"""Per-node concrete work buffers (type-stable; the whole point of J-06)."""
struct NodeBuf
    pre::Matrix{Float32}     # pre-activation
    z_mu::Matrix{Float32}    # prediction  (for a hoisted node this is computed ONCE and reused)
    lg::Matrix{Float32}      # latent_grad accumulator
    dpre::Matrix{Float32}    # pre_grad scratch (for input-grad matmuls)
end

"""A persistent, reusable buffer pool + static plan for in-place inference of one graph shape.
Build once with `prealloc_inference`, run many times with `run_inference!` — 0 allocations per
run except the returned latents."""
struct InplaceInference
    plan::CompiledPlan
    clamped::Vector{Bool}
    hoist::Vector{Bool}                       # nodes whose forward is loop-invariant (all sources clamped)
    weights::Vector{Vector{Matrix{Float32}}}  # per node: concrete edge weights (aligned to in_src)
    biases::Vector{Union{Matrix{Float32},Nothing}}
    bufs::Vector{NodeBuf}
    z::Vector{Matrix{Float32}}                # per-node z_latent (mutated in place)
    src::Vector{Vector{Matrix{Float32}}}      # per node: STABLE refs to source z buffers (in_src = in_edges order)
    prec::Vector{Float32}                     # per node: GaussianEnergy.precision (NOT assumed 1)
    eta::Float32
    decay::Float32
    steps::Int
end

# Scope guard — FAIL-CLOSED. Only the three node types below have a hand-written 0-alloc
# forward+grad here, and each has an exact-math precondition. Anything else (LinearResidual,
# TransformerBlock, Embedding, …) hits this `false` default and is rejected at prealloc time,
# never reaching a MethodError or wrong-math path mid-loop.
_inplace_supported(::AbstractNode) = false
# Linear DOES apply its activation ⇒ must be Identity or Tanh (the only f' we implement).
_inplace_supported(node::Linear) =
    (node.energy isa GaussianEnergy) &&
    (node.activation isa IdentityActivation || node.activation isa TanhActivation)
# IdentityNode (z_mu = scale·Σx) and SkipConnection (z_mu = Σx) BYPASS their activation in
# `forward` (see nodes/identity.jl, skip_connection.jl), so activation is a no-op for them —
# only the Gaussian energy matters for the grad math to be exact.
_inplace_supported(node::IdentityNode) = node.energy isa GaussianEnergy
_inplace_supported(node::SkipConnection) = node.energy isa GaussianEnergy

"""
    prealloc_inference(structure, params, clamps; batch) -> InplaceInference

Build the reusable buffer pool. Extracts weights into concrete `Matrix{Float32}` (type-stable),
sizes every per-node buffer, and precomputes the hoist set. Errors early if any node is outside
increment-1 scope (so it never silently falls to a slow/allocating path mid-loop)."""
function prealloc_inference(
    structure::GraphStructure, params::GraphParams, clamps::AbstractDict; batch::Int
)
    plan = CompiledPlan(structure)
    n = length(plan.names)
    clamped = Bool[nm in keys(clamps) for nm in plan.names]
    fparams = to_flat_params(plan, params)

    # The in-place loop hardcodes the plain-SGD update (z ← z·(1-η·decay) - η·grad). NormClip
    # and Momentum override run_inference itself (clipping / cross-step velocity), so reading only
    # eta/decay/steps off them would silently degrade to plain SGD — reject them (fail-closed).
    plan.inference isa InferenceSGD || error(
        "prealloc_inference: in-place lane supports only InferenceSGD; got $(typeof(plan.inference)). " *
        "InferenceSGDNormClip / InferenceSGDMomentum change the update rule — use run_inference."
    )

    for i in 1:n
        info = plan.infos[i]
        _inplace_supported(plan.nodes[i]) || error(
            "prealloc_inference: node $(plan.names[i]) ($(typeof(plan.nodes[i])), " *
            "energy $(typeof(plan.nodes[i].energy)), activation $(hasproperty(plan.nodes[i],:activation) ? typeof(plan.nodes[i].activation) : "-"))" *
            " is outside the in-place lane's increment-1 scope (Linear/Identity/Skip + GaussianEnergy; Linear also needs Identity/Tanh)."
        )
        # muPC scaling (info.scaling_config != nothing) rescales inputs/self-grad in the reference
        # forward_value_and_grad; the in-place lane applies none. It's `nothing` in v0 — reject
        # anything else rather than silently drop it.
        info.scaling_config === nothing || error(
            "prealloc_inference: node $(plan.names[i]) has non-nothing muPC scaling_config " *
            "($(typeof(info.scaling_config))) — the in-place lane does not apply muPC scaling. Use run_inference."
        )
        # Unclamped sink (out_degree==0 && !clamped): the reference snaps z_latent := z_mu with
        # zero gradient (eval/prediction mode, nodes/linear.jl:139). The in-place loop instead runs
        # the interior SGD relaxation ⇒ divergence. Training always clamps sinks; reject otherwise.
        !(!clamped[i] && info.out_degree == 0 && info.in_degree > 0) || error(
            "prealloc_inference: node $(plan.names[i]) is an UNCLAMPED sink (out_degree==0). The " *
            "reference sets z:=z_mu there (prediction mode), not implemented in the in-place lane " *
            "increment 1 — clamp it, or use run_inference."
        )
    end

    weights = Vector{Vector{Matrix{Float32}}}(undef, n)
    biases = Vector{Union{Matrix{Float32},Nothing}}(undef, n)
    bufs = Vector{NodeBuf}(undef, n)
    z = Vector{Matrix{Float32}}(undef, n)
    # Per-node GaussianEnergy precision (the scope guard above guarantees GaussianEnergy).
    prec = Float32[(plan.nodes[i].energy::GaussianEnergy).precision for i in 1:n]
    for i in 1:n
        feat = plan.infos[i].shape[end]
        # Linear has one Matrix per in-edge; IdentityNode/SkipConnection are WEIGHTLESS (their
        # to_flat_params placeholders are `nothing`) — filter them out (never indexed) so we don't
        # call Matrix{Float32}(nothing). Weights stay in in_src (== in_edges) order, which now
        # matches the reference forward's fold order (gather_inputs uses an OrderedDict).
        weights[i] = Matrix{Float32}[Matrix{Float32}(w) for w in fparams[i].w if w isa AbstractMatrix]
        b = fparams[i].b
        biases[i] = (b !== nothing && length(b) > 0) ? Matrix{Float32}(b) : nothing
        bufs[i] = NodeBuf(
            Matrix{Float32}(undef, batch, feat), Matrix{Float32}(undef, batch, feat),
            Matrix{Float32}(undef, batch, feat), Matrix{Float32}(undef, batch, feat),
        )
        z[i] = Matrix{Float32}(undef, batch, feat)
    end

    # STABLE source-buffer refs (in_src order == the reference's OrderedDict fold order), precomputed
    # once. z[s] is mutated in place (identity never changes across steps), so these vectors stay
    # valid — this is what makes run_inference! 0-alloc per step (vs rebuilding `[z[s] for s in ...]`
    # every node every step).
    src = Vector{Vector{Matrix{Float32}}}(undef, n)
    for i in 1:n
        src[i] = Matrix{Float32}[z[s] for s in plan.in_src[i]]
    end

    hoist = falses(n)
    for i in 1:n
        info = plan.infos[i]
        (info.in_degree > 0 && info.out_degree > 0 && !clamped[i]) || continue
        all(clamped[s] for s in plan.in_src[i]) && (hoist[i] = true)
    end

    inf = plan.inference
    return InplaceInference(
        plan, clamped, hoist, weights, biases, bufs, z, src, prec,
        inf.eta_infer, 1.0f0 - inf.eta_infer * inf.latent_decay, inf.infer_steps,
    )
end

# in-place forward: writes buf.pre and buf.z_mu from the source latents. (pre = Σ x*W (+b))
@inline function _fwd!(node::Linear, buf::NodeBuf, ws, b, zsrc)
    fill!(buf.pre, 0.0f0)
    @inbounds for k in eachindex(zsrc)
        mul!(buf.pre, zsrc[k], ws[k], 1.0f0, 1.0f0)          # pre += x_k * W_k, no temp
    end
    b !== nothing && (buf.pre .+= b)
    if node.activation isa TanhActivation
        buf.z_mu .= tanh.(buf.pre)
    else                                                      # IdentityActivation
        buf.z_mu .= buf.pre
    end
    return nothing
end
@inline function _fwd!(::IdentityNode, buf::NodeBuf, ws, b, zsrc, scale)
    fill!(buf.z_mu, 0.0f0)
    @inbounds for k in eachindex(zsrc)
        buf.z_mu .+= zsrc[k]
    end
    scale != 1.0f0 && (buf.z_mu .*= scale)
    return nothing
end
@inline function _fwd!(::SkipConnection, buf::NodeBuf, ws, b, zsrc)
    fill!(buf.z_mu, 0.0f0)
    @inbounds for k in eachindex(zsrc)
        buf.z_mu .+= zsrc[k]
    end
    return nothing
end

"""
    run_inference!(ii, init_state) -> Vector{Matrix{Float32}}

Run the in-place PC relaxation reusing `ii`'s buffers (0 allocations except the returned copies).
`init_state` supplies the initial z_latents (clamped nodes stay fixed; unclamped relax). Returns
converged per-node z_latents in `ii.plan.names` order — bit-identical to `run_inference`."""
function run_inference!(ii::InplaceInference, init_state::GraphState)
    plan = ii.plan; n = length(plan.names)
    for i in 1:n
        # ::Matrix{Float32} asserts through init_state's type-unstable NodeState.z_latent::Any,
        # so this boundary copy stays statically dispatched + allocation-free.
        z0 = init_state.nodes[plan.names[i]].z_latent::Matrix{Float32}
        # FAIL-CLOSED on a batch/shape mismatch: the buffers are baked to prealloc's `batch`, and
        # copyto! into a larger dest would silently copy stale rows (mirrors compile_inference's
        # "rebuild if batch changes" contract).
        size(z0) == size(ii.z[i]) || error(
            "run_inference!: init_state node $(plan.names[i]) has shape $(size(z0)) but this " *
            "InplaceInference was pre-allocated for $(size(ii.z[i])). Rebuild with prealloc_inference " *
            "at the new batch size."
        )
        copyto!(ii.z[i], z0)                                        # seed latents
    end
    # hoisted forwards: computed ONCE (all sources clamped ⇒ constant across steps)
    @inbounds for i in 1:n
        ii.hoist[i] || continue
        _hoist_fwd!(plan.nodes[i], ii.bufs[i], ii.weights[i], ii.biases[i], ii.src[i])
    end

    @inbounds for _step in 1:ii.steps
        for i in 1:n
            fill!(ii.bufs[i].lg, 0.0f0)                         # Phase 1: zero latent grads
        end
        for i in 1:n                                            # Phase 2: forward + accumulate
            info = plan.infos[i]
            info.in_degree == 0 && continue                     # terminal source: no fwd, no grads
            node = plan.nodes[i]; buf = ii.bufs[i]
            if !ii.hoist[i]
                _forward!(node, buf, ii.weights[i], ii.biases[i], ii.src[i])
            end
            # self_grad = precision*(z - z_mu) into lg[i] (lg is zeroed; unclamped only)
            ii.clamped[i] || (buf.lg .+= ii.prec[i] .* (ii.z[i] .- buf.z_mu))
            ii.hoist[i] && continue                             # all in-sources clamped ⇒ back-edges pruned
            # input grads into sources (skip clamped targets)
            _accum_input_grads!(node, buf, ii, i, ii.prec[i])
        end
        for i in 1:n                                            # Phase 3: SGD update, non-clamped
            ii.clamped[i] && continue
            @. ii.z[i] = ii.z[i] * ii.decay - ii.eta * ii.bufs[i].lg
        end
    end
    return [copy(ii.z[i]) for i in 1:n]
end

_forward!(node::Linear, buf, ws, b, zsrc) = _fwd!(node, buf, ws, b, zsrc)
_forward!(node::IdentityNode, buf, ws, b, zsrc) = _fwd!(node, buf, ws, b, zsrc, node.scale)
_forward!(node::SkipConnection, buf, ws, b, zsrc) = _fwd!(node, buf, ws, b, zsrc)
_hoist_fwd!(node, buf, ws, b, zsrc) = _forward!(node, buf, ws, b, zsrc)

# dpre = grad_mu*deriv = precision*(z_mu - z) .* f'(pre); input_grad to src_k = dpre * W_k'
function _accum_input_grads!(node::Linear, buf::NodeBuf, ii::InplaceInference, i::Int, prec::Float32)
    if node.activation isa TanhActivation
        buf.dpre .= prec .* (buf.z_mu .- ii.z[i]) .* (1.0f0 .- buf.z_mu .^ 2)   # f'(tanh)=1-z_mu^2
    else
        buf.dpre .= prec .* (buf.z_mu .- ii.z[i])                               # f'(identity)=1
    end
    # weights[i][k] pairs with in_src[i][k] (same order). Grad accumulation is per-source, so the
    # iteration order is irrelevant to the result — only the forward fold order (src/weights) matters.
    @inbounds for (k, s) in enumerate(ii.plan.in_src[i])
        ii.clamped[s] && continue                                              # PRUNE clamped target
        mul!(ii.bufs[s].lg, buf.dpre, transpose(ii.weights[i][k]), 1.0f0, 1.0f0)
    end
    return nothing
end
# Identity/Skip: input_grad = scale·grad_mu (Identity) or grad_mu (Skip); grad_mu=precision*(z_mu-z)
function _accum_input_grads!(node::IdentityNode, buf::NodeBuf, ii::InplaceInference, i::Int, prec::Float32)
    buf.dpre .= (prec * node.scale) .* (buf.z_mu .- ii.z[i])
    @inbounds for s in ii.plan.in_src[i]
        ii.clamped[s] && continue
        ii.bufs[s].lg .+= buf.dpre
    end
    return nothing
end
function _accum_input_grads!(::SkipConnection, buf::NodeBuf, ii::InplaceInference, i::Int, prec::Float32)
    buf.dpre .= prec .* (buf.z_mu .- ii.z[i])
    @inbounds for s in ii.plan.in_src[i]
        ii.clamped[s] && continue
        ii.bufs[s].lg .+= buf.dpre
    end
    return nothing
end
