# Enzyme reverse-mode autodiff extension for FabricPC (opt-in: loaded when
# `using Enzyme`). Implements the Phase-D autodiff node seam declared in
# src/nodes/autodiff.jl — the Julia analog of upstream's base-class
# `jax.value_and_grad(forward)` (fabricpc/nodes/base.py).
#
# Differentiates `FabricPC.energy_kernel` (→ `compute_mu`, concrete arrays only):
#   • weight grads  — ∂/∂params           (Duplicated over the whole NodeParams)
#   • latent grads  — ∂/∂(inputs, z_latent)
# Both use `set_runtime_activity(Reverse)`: the Dict iteration over input edges
# mixes active/constant entries, which Enzyme's static activity analysis cannot
# resolve (EnzymeRuntimeActivityError) — runtime activity restores correctness at
# a small perf cost. Validated against the closed-form Linear oracle to ~1e-7
# (test_autodiff_seam.jl). This is local PC, NOT backprop: each call differentiates
# one node's local energy only — never through the network or the inference loop.

module FabricPCEnzymeExt

using FabricPC
using FabricPC: AbstractNode, NodeParams, SoA, energy_kernel
import FabricPC: _ad_param_grads, _ad_latent_grads, _register_ad_backend!
using Enzyme

# __init__(), NOT bare top-level: a module-level statement that mutates a DIFFERENT,
# already-loaded module's global state (FabricPC._AD_BACKEND) only takes effect in the
# ephemeral worker process that PRECOMPILES this extension -- it is never replayed when the
# extension is later loaded from its cached .ji in a real session. __init__() is Julia's
# documented mechanism for exactly this: it runs on EVERY load, precompiled or not. Before
# this fix, F-04's guard was a no-op in any normal (post-precompile) session -- confirmed by
# direct probe, 2026-07-14 (see docs/AUDIT_REGISTER.md F-04 / docs/decisions.md #15 update).
function __init__()
    _register_ad_backend!(:enzyme)
end

# The params weights/biases (and, for latent grads, the inputs) are threaded as `NamedTuple`s — the
# `SoA` container's `.nt` — so Enzyme accumulates gradients into NamedTuples, never a `Dict` (a Dict
# shadow trips `addToDiffe: "unhandled accumulate with partial sizes"` on its i64 hash internals once
# a node's `compute_mu` has a multi-head map). `node`/`inputs`/`z` are passed as explicit `Const`
# args (NOT closure captures — a captured mutable Dict triggers EnzymeMutabilityException).

# differentiated kernels: rebuild the SoA from the differentiated NamedTuple, then call energy_kernel.
_ek_w(node, wnt, bnt, inputs, z) =
    energy_kernel(node, NodeParams(SoA(wnt), SoA(bnt)), inputs, z)
_ek_in(node, params, innt, z) = energy_kernel(node, params, SoA(innt), z)

function _ad_param_grads(
    node::AbstractNode, params::NodeParams, inputs, z_latent
)
    wnt = params.weights.nt
    bnt = params.biases.nt
    dwnt = map(zero, wnt)
    dbnt = map(zero, bnt)
    Enzyme.autodiff(
        set_runtime_activity(Reverse), _ek_w, Active,
        Const(node), Duplicated(wnt, dwnt), Duplicated(bnt, dbnt), Const(inputs),
        Const(z_latent)
    )
    return NodeParams(SoA(dwnt), SoA(dbnt))
end

function _ad_latent_grads(
    node::AbstractNode, params::NodeParams, inputs, z_latent
)
    innt = SoA(inputs).nt                         # Dict inputs (eager) → NamedTuple for Enzyme
    dinnt = map(zero, innt)
    dz = zero(z_latent)
    Enzyme.autodiff(
        set_runtime_activity(Reverse), _ek_in, Active,
        Const(node), Const(params), Duplicated(innt, dinnt), Duplicated(z_latent, dz)
    )
    N = ndims(first(values(inputs)))
    dinputs = Dict{String, Array{Float32, N}}(String(k) => dinnt[k] for k in keys(dinnt))
    return dinputs, dz
end

end # module FabricPCEnzymeExt
