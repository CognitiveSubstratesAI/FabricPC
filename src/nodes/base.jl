# Node base behaviours. Port of fabricpc/nodes/base.py (the shared parts).
#
# Concrete nodes (Linear, IdentityNode) live in their own files and implement,
# by dispatch on the node type:
#   get_slots(node)                                  -> Dict{String,SlotSpec}
#   initialize_params(node, rng, shape, in_shapes, weight_init) -> NodeParams
#   forward(node, params, inputs, state)             -> (total_energy, NodeState)
#   forward_and_latent_grads(node, params, inputs, state, info, is_clamped)
#                                                    -> (NodeState, input_grads, self_grad)
#   forward_and_weight_grads(node, params, inputs, state) -> (NodeState, NodeParams)
#   compute_gain_mod_error(node, state)              -> array
#
# This file provides the behaviours shared by all node types.

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
    get_weight_fan_in(node, source_shape) -> Int

Weight-matrix fan-in for muPC scaling (Kaiming convention). Default: the
last-axis feature count of the source. Overridden per node type (e.g.
`IdentityNode` returns 1 — it has no weight matrix). Used in Phase C.
"""
get_weight_fan_in(::AbstractNode, source_shape::Tuple) = source_shape[end]
