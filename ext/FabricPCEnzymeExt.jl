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
using FabricPC: AbstractNode, NodeParams, energy_kernel
import FabricPC: _ad_param_grads, _ad_latent_grads
using Enzyme

function _ad_param_grads(
    node::AbstractNode, params::NodeParams, inputs, z_latent
)
    dparams = NodeParams(
        Dict{String, Matrix{Float32}}(k => zero(v) for (k, v) in params.weights),
        Dict{String, Matrix{Float32}}(k => zero(v) for (k, v) in params.biases)
    )
    Enzyme.autodiff(
        set_runtime_activity(Reverse),
        energy_kernel,
        Active,
        Const(node),
        Duplicated(params, dparams),
        Const(inputs),
        Const(z_latent)
    )
    return dparams
end

function _ad_latent_grads(
    node::AbstractNode, params::NodeParams, inputs, z_latent
)
    dinputs = Dict{String, Matrix{Float32}}(k => zero(v) for (k, v) in inputs)
    dz = zero(z_latent)
    Enzyme.autodiff(
        set_runtime_activity(Reverse),
        energy_kernel,
        Active,
        Const(node),
        Const(params),
        Duplicated(inputs, dinputs),
        Duplicated(z_latent, dz)
    )
    return dinputs, dz
end

end # module FabricPCEnzymeExt
