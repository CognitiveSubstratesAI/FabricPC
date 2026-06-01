# GraphState / NodeState functional update helpers. Port of
# fabricpc/core/state_ops.py (+ the NamedTuple `_replace` it relies on).
#
# Everything returns a NEW state; nothing is mutated in place. For v0's small
# graphs the per-update Dict copy is negligible; a mutable fast path can be
# added later if profiling demands it.

"""
    update_state(s::NodeState; field = value, …) -> NodeState

Return a copy of `s` with the named fields replaced (upstream `NamedTuple._replace`).
"""
update_state(
    s::NodeState;
    z_latent=s.z_latent,
    z_mu=s.z_mu,
    error=s.error,
    energy=s.energy,
    pre_activation=s.pre_activation,
    latent_grad=s.latent_grad
) = NodeState(z_latent, z_mu, error, energy, pre_activation, latent_grad)

"""Return a new GraphState with `name`'s NodeState replaced by `ns`."""
put_node(state::GraphState, name::AbstractString, ns::NodeState) =
    GraphState(merge(state.nodes, Dict(String(name) => ns)), state.batch_size)

"""
    update_node_in_state(state, name; field = value, …) -> GraphState

Update fields of `name`'s NodeState within `state`. Port of
`update_node_in_state`.
"""
update_node_in_state(state::GraphState, name::AbstractString; kwargs...) =
    put_node(state, name, update_state(state.nodes[String(name)]; kwargs...))

"""
    set_latents_to_clamps(state, clamps) -> GraphState

Set `z_latent` of each clamped node to its clamp value. Port of
`set_latents_to_clamps`.
"""
function set_latents_to_clamps(state::GraphState, clamps::AbstractDict)
    for (name, val) in clamps
        if haskey(state.nodes, name)
            state = update_node_in_state(state, name; z_latent=Float32.(val))
        end
    end
    return state
end
