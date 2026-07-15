# Node base behaviours. Port of fabricpc/nodes/base.py (the shared parts).
#
# Concrete nodes (Linear, IdentityNode) live in their own files and implement,
# by dispatch on the node type:
#   get_slots(node)                                  -> Dict{String,SlotSpec}
#   initialize_params(node, rng, shape, in_shapes, weight_init) -> NodeParams
#   forward(node, params, inputs, state)             -> NodeState
#   forward_and_latent_grads(node, params, inputs, state, info, is_clamped)
#                                                    -> (NodeState, input_grads, self_grad)
#   forward_and_weight_grads(node, params, inputs, state) -> (NodeState, NodeParams)
#   (gradients use the shared helpers mu_grad / pre_grad below)
#
# This file provides the behaviours shared by all node types.

"""
    get_slots(node::AbstractNode) -> Dict{String,SlotSpec}

Named input slots for this node type (e.g. `"in"`, `"skip"`/`"residual"` for nodes with a
decoupled bypass edge). Dispatches on the concrete node type — every node implements its own;
there is no generic fallback.
"""
function get_slots end

"""
    energy_functional(node, state) -> NodeState

Compute the node's per-sample energy `E(z_latent, z_mu)` and store it in
`state.energy`. The self-latent gradient is NOT computed here (the explicit
gradient path calls `grad_latent` directly). Port of `NodeBase.energy_functional`.
"""
function energy_functional(node::AbstractNode, state::NodeState)
    e = energy(node.energy, state.z_latent, state.z_mu)
    return update_state(state; energy=e)
end

"""
    mu_grad(node, state) -> array

`∂E/∂z_mu` for this node's energy (the prediction-side energy gradient). Drives
the explicit input/weight gradients. For a node that bypasses its activation
(Identity/Skip), the input gradient is `(dz_mu/dx)·mu_grad`. At Gaussian
precision 1, `mu_grad = -(z - z_mu) = -error`.
"""
mu_grad(node::AbstractNode, state::NodeState) =
    grad_mu(node.energy, state.z_latent, state.z_mu)

"""
    pre_grad(node, state, pre) -> array

`∂E/∂pre = (∂E/∂z_mu) · f'(pre)` — the energy gradient w.r.t. the pre-activation,
for nodes whose `z_mu = activation(pre)`. The linear-path input/weight gradients
are built from this (`input_grad = pre_grad·Wᵀ`, `dW = xᵀ·pre_grad`,
`db = Σ pre_grad`). Replaces the Gaussian-only `error·f'` from Phase B (identical
at precision 1, but correct for any energy/precision).

`pre` is passed EXPLICITLY rather than read back from `state.pre_activation`, which no longer
exists: it is produced and consumed inside one forward, via `_forward_with_preact`. Mirrors
upstream `b6f64ad`, where `compute_gain_mod_error(state, node_info)` became
`compute_gain_mod_error(pre_activation, error, node_info)` for exactly this reason.
"""
pre_grad(node::AbstractNode, state::NodeState, pre) =
    _pre_grad(node.activation, node.energy, node, state, pre)

# Generic element-wise case: dE/dpre = (∂E/∂z_mu) · f'(pre).
_pre_grad(act::AbstractActivation, ::AbstractEnergy, node::AbstractNode, state::NodeState, pre) =
    mu_grad(node, state) .* derivative(act, pre)

# Exact Softmax + CrossEntropy coupling: the off-diagonal softmax Jacobian and
# the CE gradient combine to the clean closed form dE/dpre = s - y (= z_mu -
# z_latent). This is exact (no autodiff, no diagonal approximation) and is what
# makes a softmax+CE classifier train properly on hard multi-class tasks — the
# diagonal `s·(1-s)` approximation used elsewhere is too crude there.
_pre_grad(::SoftmaxActivation, ::CrossEntropyEnergy, node::AbstractNode, state::NodeState, _pre) =
    state.z_mu .- state.z_latent

"""
    get_weight_fan_in(node, source_shape) -> Int

Weight-matrix fan-in for muPC scaling (Kaiming convention). Default: the
last-axis feature count of the source. Overridden per node type (e.g.
`IdentityNode` returns 1 — it has no weight matrix). Used in Phase C.
"""
get_weight_fan_in(::AbstractNode, source_shape::Tuple) = source_shape[end]
