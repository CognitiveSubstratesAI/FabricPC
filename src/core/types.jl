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
# ── Struct-of-Arrays param container (replaces Dict{String,Matrix} in NodeParams) ─────────────
# WHY: Enzyme reverse-mode cannot accumulate gradients into a `Dict`'s shadow — its internal i64
# hash/slot arrays trip `addToDiffe: "unhandled accumulate with partial sizes"` once a node's
# `compute_mu` has a multi-head map (root-caused by bisection). A `NamedTuple` has no such integer
# internals and differentiates cleanly; it is ALSO type-stable (no dynamic Dict lookup), which suits
# the Reactant JIT path. `SoA` is a thin NamedTuple-backed container exposing the Dict-like *read*
# API the codebase already uses (`["W_q"]`, `keys`, `values`, iteration), so node forwards and most
# call sites are UNCHANGED; only construction converts Dict→SoA.
struct SoA{NT<:NamedTuple}
    nt::NT
end
SoA(d::AbstractDict) = SoA(NamedTuple{Tuple(Symbol(k) for k in keys(d))}(Tuple(values(d))))
Base.getindex(s::SoA, k::AbstractString) = s.nt[Symbol(k)]
Base.getindex(s::SoA, k::Symbol)         = s.nt[k]
Base.get(s::SoA, k::AbstractString, default) = haskey(s.nt, Symbol(k)) ? s.nt[Symbol(k)] : default
Base.get(s::SoA, k::Symbol, default)         = haskey(s.nt, k) ? s.nt[k] : default
Base.haskey(s::SoA, k::AbstractString)   = haskey(s.nt, Symbol(k))
# Key→String conversion (jl_cstr_to_string) is a foreigncall Zygote cannot differentiate. SoA keys
# are STRUCTURAL — they never depend on the (differentiated) numeric values — so these helpers are
# marked `Zygote.@nograd` in FabricPCZygoteExt. That lets an autodiff'd `compute_mu` ITERATE an SoA
# (generic nodes that sum over input edges) while gradients still flow through the values (`s.nt[i]`).
_soa_key(nt::NamedTuple, i::Int) = String(keys(nt)[i])
_soa_keys(nt::NamedTuple)        = String[String(k) for k in keys(nt)]
Base.keys(s::SoA)   = _soa_keys(s.nt)
Base.values(s::SoA) = values(s.nt)
Base.length(s::SoA) = length(s.nt)
Base.pairs(s::SoA)  = (_soa_key(s.nt, i) => s.nt[i] for i in 1:length(s.nt))
function Base.iterate(s::SoA, st::Int = 1)
    st > length(s.nt) && return nothing
    return (_soa_key(s.nt, st) => s.nt[st], st + 1)
end

struct NodeParams{W<:SoA, B<:SoA}
    weights::W
    biases::B
end
NodeParams(w::AbstractDict, b::AbstractDict) = NodeParams(SoA(w), SoA(b))
# Mixed construction (e.g. a freshly-built gradient Dict for weights reusing an existing SoA bias):
# coerce whichever arg is still a Dict. The SoA/SoA case uses the struct's own constructor.
NodeParams(w::AbstractDict, b::SoA) = NodeParams(SoA(w), b)
NodeParams(w::SoA, b::AbstractDict) = NodeParams(w, SoA(b))

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
