# muPC scaling. Port of fabricpc/core/scaling.py.
#
# In v0 every node's `scaling_config` is `nothing` (muPC off), so all four
# scaling functions are the identity. Phase C adds `MuPCScalingFactors` plus the
# real per-edge methods (forward_scale / topdown_grad_scale / self_grad_scale /
# weight_grad_scale). Keeping the call-sites wired now means turning muPC on is
# purely additive — the inference + learning loops already invoke these.

scale_inputs(inputs::AbstractDict, ::Nothing) = inputs
scale_input_grads(input_grads::AbstractDict, ::Nothing) = input_grads
scale_self_grad(self_grad, ::Nothing) = self_grad
scale_weight_grads(params_grad::NodeParams, ::Nothing) = params_grad

# Phase C: real muPC scaling. Edges absent from a per-edge dict pass through
# UNCHANGED (a missing key is a no-op, not ×1.0) — this preserves input dtype
# for non-scalable (skip/mask) slots, matching upstream.
#
# Pre-scale inputs: W·(a·x) = a·(W·x), equivalent to scaling the node output but
# keeping forward() scaling-unaware.
scale_inputs(inputs::AbstractDict, sc::MuPCScalingFactors) = Dict(
    k => (haskey(sc.forward_scale, k) ? x .* sc.forward_scale[k] : x) for
    (k, x) in inputs
)

# Post-scale input grads by topdown_grad_scale = a · jacobian_gain (chain-rule
# correction for the pre-scaled inputs + per-hop Jacobian compensation).
scale_input_grads(input_grads::AbstractDict, sc::MuPCScalingFactors) = Dict(
    k => (haskey(sc.topdown_grad_scale, k) ? g .* sc.topdown_grad_scale[k] : g) for
    (k, g) in input_grads
)

scale_self_grad(self_grad, sc::MuPCScalingFactors) = self_grad .* sc.self_grad_scale

# Per-weight scaling only where the weight key matches a scalable edge key;
# others pass at 1.0 (faithful to upstream's get(k, 1.0)).
scale_weight_grads(params_grad::NodeParams, sc::MuPCScalingFactors) = NodeParams(
    Dict(k => v .* get(sc.weight_grad_scale, k, 1.0f0) for (k, v) in params_grad.weights),
    params_grad.biases
)
