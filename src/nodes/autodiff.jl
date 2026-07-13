# Phase-D autodiff node seam. The Julia analog of upstream's base-class
# `jax.value_and_grad(forward)` default (fabricpc/nodes/base.py): a node that
# implements ONLY the pure numeric prediction `compute_mu(node, params, inputs)`
# gets its local predictive-coding gradients — weight (learning) and latent +
# input (inference) — for FREE via Enzyme reverse-mode autodiff.
#
# This is still PURE PREDICTIVE CODING, NOT backprop. The autodiff is confined to
# ONE node's local energy E_node(params, inputs, z_latent); it never propagates
# through the network or the inference loop (the weight loop does not differentiate
# through the latent relaxation). It only spares the author from hand-deriving a
# complex node's local gradient — exactly what makes transformer / attention /
# Storkey-Hopfield nodes expressible.
#
# Enzyme is a weakdep (FabricPCEnzymeExt), per decisions.md #8 (its precompile +
# CI-JIT cost): the base package stays autodiff-free; `using Enzyme` activates the
# seam. Linear/Identity/Skip/Residual keep their closed-form overrides, which are
# the conformance oracle this path was validated against (test_autodiff_seam.jl):
# Enzyme reproduced the closed-form Linear gradients to ~1e-7.
#
# IMPORTANT: the differentiated function (`energy_kernel` → `compute_mu`) must
# touch ONLY concrete arrays (Matrix{Float32}), never the ::Any-boxed NodeState
# (decision #5) — that boxing forces Enzyme onto its type-unstable path, which
# fails. NodeState bookkeeping therefore stays OUTSIDE the differentiated region.

"""
    compute_mu(node, params, inputs) -> z_mu::Matrix{Float32}

Pure numeric prediction of this node's latent from its parameters and inputs —
concrete arrays only (no NodeState). Custom / non-linear / transformer nodes
implement THIS one method; `forward`, `energy_kernel`, and both gradient methods
are then generic. Must be expressible through Enzyme-differentiable array ops.
Multiple incoming edges are combined here (e.g. summed) exactly as in `forward`.
"""
function compute_mu end

"""
    energy_kernel(node, params, inputs, z_latent) -> Float32

Scalar total node energy `Σ E(z_latent, compute_mu(node, params, inputs))`. This
is the function Enzyme differentiates (w.r.t. `params` for weight grads; w.r.t.
`inputs`/`z_latent` for inference grads). Generic in terms of `compute_mu`. `params`
weights/biases are `SoA` (NamedTuple-backed) so Enzyme can accumulate into them.
"""
energy_kernel(node::AbstractNode, params::NodeParams, inputs, z_latent) =
    sum(energy(node.energy, z_latent, compute_mu(node, params, inputs)))

"""
    forward(node::AbstractNode, params, inputs, state) -> (total_energy, NodeState)

Generic forward for autodiff-backed nodes (no explicit override). Builds `z_mu`
via `compute_mu`, fills `error`/`energy`, and packs the NodeState. Mirrors
`Linear`'s `forward` bookkeeping but defers the prediction math to `compute_mu`.
`pre_activation` is set to `z_mu` (the explicit `pre_grad` path is unused here).
"""
function forward(
    node::AbstractNode, params::NodeParams, inputs::AbstractDict, state::NodeState
)
    z_mu = compute_mu(node, params, inputs)
    err = state.z_latent .- z_mu
    ns = update_state(state; pre_activation=z_mu, z_mu=z_mu, error=err)
    ns = energy_functional(node, ns)
    return sum(ns.energy), ns
end

# --- autodiff hooks: real implementations live in FabricPCEnzymeExt / FabricPCZygoteExt -------
# Two interchangeable backends implement this seam (load ONE — both define the same hooks):
#   • FabricPCZygoteExt (`using Zygote`) — the WORKING backend for transformer/attention nodes.
#     Enzyme aborts on the full multi-head-attention block on Julia 1.12
#     (`addToDiffe: "unhandled accumulate with partial sizes"`); Zygote handles FabricPC's
#     functional forwards cleanly. Node-local gradient FD-validated (test_transformer_lm).
#   • FabricPCEnzymeExt (`using Enzyme`) — original backend; fine for simple/dense nodes
#     (validated to ~1e-7 vs the closed-form Linear oracle in test_autodiff_seam.jl).
const _ENZYME_HINT =
    "FabricPC: this node uses the Phase-D autodiff gradient seam — run `using Zygote` " *
    "(works on transformer/attention) or `using Enzyme` (simple nodes) to activate it."

# F-04 dual-backend load guard: FabricPCZygoteExt and FabricPCEnzymeExt implement the
# IDENTICAL `_ad_param_grads`/`_ad_latent_grads` hooks (see decisions.md #19) — they
# cannot safely coexist, since the second-loaded extension silently overrides the
# first's dispatch with no warning, and Enzyme aborts (opaque LLVM crash, not a
# catchable Julia error) on the multi-head-attention block that Zygote handles fine.
# The varargs fallback above only catches "neither backend loaded yet"; it can never
# fire once either extension has registered a fully-specific method. This registers
# WHICH backend loaded first and raises immediately (at `using X` time, not at the
# first mismatched call) if the other one tries to load afterward in the same session.
const _AD_BACKEND = Ref{Symbol}(:none)

function _register_ad_backend!(name::Symbol)
    if _AD_BACKEND[] !== :none && _AD_BACKEND[] !== name
        error(
            "FabricPC: both Zygote and Enzyme AD backends are loaded in this session " *
            "(loaded $(_AD_BACKEND[]) first, then $name). They implement the SAME " *
            "gradient seam and cannot coexist: the second one silently overrides the " *
            "first's dispatch, and Enzyme aborts with an opaque LLVM crash on nodes " *
            "(e.g. TransformerBlock) that only Zygote handles. Use exactly ONE of " *
            "`using Zygote` / `using Enzyme` per Julia session — restart to switch."
        )
    end
    _AD_BACKEND[] = name
    return nothing
end

# Varargs fallbacks (NOT the typed signature the extension defines): a same-
# signature method in the ext would be a forbidden precompile-time overwrite. The
# ext adds strictly-more-specific `(::AbstractNode, ::NodeParams, inputs, z_latent)`
# methods that win dispatch when Enzyme is loaded; otherwise these raise the hint.

"""Reverse-mode `∂(energy_kernel)/∂params`; real method in FabricPCEnzymeExt."""
_ad_param_grads(args...) = error(_ENZYME_HINT)

"""Reverse-mode `∂(energy_kernel)/∂(inputs, z_latent)`; real method in FabricPCEnzymeExt."""
_ad_latent_grads(args...) = error(_ENZYME_HINT)

# gather_inputs builds Dict{String,Any}; concrete-ify so Enzyme stays type-stable.
# Rank is inferred from the inputs (rank-2 (batch,features) for dense nodes, rank-3
# (batch,seq,embed) for sequence/transformer nodes) and baked into the Dict's
# concrete value type `Array{Float32,N}` — an abstract `Array{Float32}` value type
# would re-trigger Enzyme's type-unstable path. Assumes one rank across input slots
# (true for the current nodes; a mixed-rank node would concrete-ify per slot).
function _concrete_inputs(inputs)
    N = ndims(first(values(inputs)))
    d = Dict{String, Array{Float32, N}}()
    for (k, v) in inputs
        d[k] = Float32.(v)
    end
    return d
end

"""
    forward_and_weight_grads(node::AbstractNode, params, inputs, state; info=nothing) -> (NodeState, NodeParams)

Generic learning-phase weight gradients via autodiff (Julia analog of
`NodeBase.forward_and_weight_grads`: `value_and_grad(forward, argnums=0)`). The
NodeState comes from the (non-differentiated) `forward`; the NodeParams gradient
comes from differentiating `energy_kernel` w.r.t. `params`.

`info` (the node's `NodeInfo`, `nothing` outside a full graph call) is accepted so
that node-specific overrides (e.g. `TransformerBlock`'s LayerNorm muPC-gradient
compensation) can read `info.scaling_config`; the generic path here ignores it.
"""
function forward_and_weight_grads(
    node::AbstractNode, params::NodeParams, inputs::AbstractDict, state::NodeState;
    info=nothing
)
    _, ns = forward(node, params, inputs, state)
    grad_params = _ad_param_grads(node, params, _concrete_inputs(inputs), state.z_latent)
    return ns, grad_params
end

"""
    forward_and_latent_grads(node::AbstractNode, params, inputs, state, info, is_clamped)

Generic inference-phase input + self-latent gradients via autodiff (Julia analog
of `NodeBase.forward_and_latent_grads`). Replicates the three upstream branches:
terminal source (`in_degree == 0`) and unclamped output (`out_degree == 0`) are
analytic zero-gradient cases; the internal/clamped-output case differentiates
`energy_kernel` w.r.t. `(inputs, z_latent)`.
"""
function forward_and_latent_grads(
    node::AbstractNode,
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
        dinputs, self_grad = _ad_latent_grads(
            node, params, _concrete_inputs(inputs), state.z_latent
        )
        input_grads = Dict{String, Any}(k => dinputs[k] for k in keys(dinputs))
        return ns, input_grads, self_grad
    end
end
