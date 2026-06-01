# Graph topology primitives. Port of fabricpc/core/topology.py.
#
# `SlotSpec`  — static description of a node's named input slot.
# `SlotRef`   — points at a specific (node, slot) for edge wiring.
# `Edge`      — a directed connection source → target:slot.
#
# Upstream's hierarchical GraphNamespace (thread-local name prefixing) is
# DEFERRED for v0 — node names are flat. Re-add when nested sub-graphs land.

"""
    SlotSpec(name, is_multi_input; is_variance_scalable = true, is_skip_connection = false)

Specification for a node input slot. `is_multi_input` allows several incoming
edges to aggregate into one slot. The muPC flags (`is_variance_scalable`,
`is_skip_connection`) are carried for Phase C; skip-connection slots must NOT be
variance-scalable (they are an identity bypass counting toward muPC depth L).
"""
struct SlotSpec
    name::String
    is_multi_input::Bool
    is_variance_scalable::Bool
    is_skip_connection::Bool
end

function SlotSpec(
    name::AbstractString,
    is_multi_input::Bool;
    is_variance_scalable::Bool=true,
    is_skip_connection::Bool=false
)
    if is_skip_connection && is_variance_scalable
        throw(
            ArgumentError(
                "is_skip_connection and is_variance_scalable were both true; " *
                "skip-connection slots must not be muPC variance-scaled"
            )
        )
    end
    return SlotSpec(String(name), is_multi_input, is_variance_scalable, is_skip_connection)
end

"""
    SlotRef(node, slot)

A reference to a particular input slot of `node`, used when wiring an edge to a
non-default slot. `slot(node, name)` is the ergonomic constructor.
"""
struct SlotRef
    node::AbstractNode
    slot::String
end

"""
    slot(node, name) -> SlotRef

Create a `SlotRef` to `node`'s `name` slot, validating the slot exists.
"""
function slot(node::AbstractNode, name::AbstractString)
    specs = get_slots(node)
    haskey(specs, String(name)) || throw(
        KeyError(
            "node '$(node.name)' has no slot '$name'. Available: $(collect(keys(specs)))"
        )
    )
    return SlotRef(node, String(name))
end

"""
    Edge(source, target)        # wires to target's default "in" slot
    Edge(source, ref::SlotRef)  # wires to a specific slot

A directed edge `source → target:slot`.
"""
struct Edge
    source::AbstractNode
    target_node::AbstractNode
    target_slot::String
end

Edge(source::AbstractNode, target::AbstractNode) = Edge(source, target, "in")
Edge(source::AbstractNode, ref::SlotRef) = Edge(source, ref.node, ref.slot)
