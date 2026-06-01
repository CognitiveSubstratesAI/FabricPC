# Core data types. Port of fabricpc/core/types.py.
#
# All immutable (upstream uses frozen dataclasses / NamedTuples + JAX pytrees).
# We update them functionally via the helpers in state_ops.jl (the analogue of
# upstream's `_replace`). Arrays are batch-FIRST: shape (batch, features...).
#
# v0 supports rank-1 node shapes only (features,) ⇒ arrays are (batch, features)
# matrices. Higher-rank / last-axis-matmul tensors (sequences, images) are
# deferred — see docs/decisions.md on the column-major shape hazard.

"""Metadata for one input slot (resolved at graph-build time)."""
struct SlotInfo
    name::String
    parent_node::String
    is_multi_input::Bool
    is_variance_scalable::Bool
    is_skip_connection::Bool
    in_neighbors::Vector{String}
end

"""Metadata for one edge: key `"source->target:slot"`."""
struct EdgeInfo
    key::String
    source::String
    target::String
    slot::String
end

"""
Topology metadata for one node, attached at graph-build time. Activation /
energy / initializers live on the node descriptor itself (read via dispatch);
`NodeInfo` carries only what the loops need: degrees, edge keys, slots, and the
(Phase C) muPC `scaling_config` (always `nothing` in v0).
"""
struct NodeInfo
    name::String
    shape::Tuple
    node_type::String
    slots::Dict{String, SlotInfo}
    in_degree::Int
    out_degree::Int
    in_edges::Vector{String}
    out_edges::Vector{String}
    scaling_config::Any   # ::MuPCScalingFactors or nothing (Phase C)
end

"""Learnable parameters of one node: named weight matrices + named biases."""
struct NodeParams
    weights::Dict{String, Matrix{Float32}}
    biases::Dict{String, Matrix{Float32}}
end

"""All learnable parameters, keyed by node name."""
struct GraphParams
    nodes::Dict{String, NodeParams}
end

"""
Dynamic per-node state during inference. `energy` is per-sample, shape (batch,);
the rest are (batch, features...). `latent_grad` accumulates dE/dz_latent across
the node's own self-grad and downstream successors' contributions.
"""
struct NodeState
    z_latent::Any
    z_mu::Any
    error::Any
    energy::Any
    pre_activation::Any
    latent_grad::Any
end

"""Dynamic state of the whole graph: per-node NodeState + batch size."""
struct GraphState
    nodes::Dict{String, NodeState}
    batch_size::Int
end

"""
Static graph topology (compile-time constant). `nodes`/`infos` are parallel maps
from node name to descriptor / metadata. `node_names` preserves INSERTION order
(the order the inference + learning loops iterate); `node_order` is the
topological order used only by feedforward state init. `config` is a NamedTuple
`(inference, graph_state_initializer)`.
"""
struct GraphStructure
    nodes::Dict{String, AbstractNode}
    infos::Dict{String, NodeInfo}
    edges::Dict{String, EdgeInfo}
    task_map::Dict{String, String}
    node_order::Vector{String}
    node_names::Vector{String}
    config::NamedTuple
end
