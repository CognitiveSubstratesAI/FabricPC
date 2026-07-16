# Core data types. Port of fabricpc/core/types.py.
#
# All immutable (upstream uses frozen dataclasses / NamedTuples + JAX pytrees).
# We update them functionally via the helpers in state_ops.jl (the analogue of
# upstream's `_replace`). Arrays are batch-FIRST: shape (batch, features...).
#
# Node shapes are RANK-GENERIC: a node's arrays are (batch, node.shape...). Rank-1
# (features,) gives the (batch, features) dense case; rank-2 (seq, embed) backs the
# TransformerBlock family (decisions.md §13); rank-3 (H, W, C) backs ConvNode/pooling
# (C-01). `NodeState`'s fields are untyped, `NodeInfo.shape` is an unconstrained Tuple,
# `state_initializer.jl` splats (batch_size, info.shape...), and every energy reduces over
# all non-batch dims — so nothing here is rank-limited. (Column-major reshape remains a
# real hazard for last-axis-matmul paths — see docs/decisions.md — but it is a per-node
# implementation concern, not a limit of these types.)

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

# ── Struct-of-Arrays param container (replaces Dict{String,Matrix} in NodeParams) ─────────────
# WHY: Enzyme reverse-mode cannot accumulate gradients into a `Dict`'s shadow — its internal i64
# hash/slot arrays trip `addToDiffe: "unhandled accumulate with partial sizes"` once a node's
# `compute_mu` has a multi-head map (root-caused by bisection). A `NamedTuple` has no such integer
# internals and differentiates cleanly; it is ALSO type-stable (no dynamic Dict lookup), which suits
# the Reactant JIT path. `SoA` is a thin NamedTuple-backed container exposing the Dict-like *read*
# API the codebase already uses (`["W_q"]`, `keys`, `values`, iteration), so node forwards and most
# call sites are UNCHANGED; only construction converts Dict→SoA.
"""
    SoA{NT<:NamedTuple}

Struct-of-Arrays: a thin `NamedTuple`-backed container exposing a `Dict`-like read API
(`getindex`, `get`, `haskey`, `keys`, `values`, `pairs`, iteration). Used for `NodeParams`'s
`weights`/`biases` — a `NamedTuple` has no integer-hash internals for Enzyme reverse-mode to
trip on (unlike `Dict`), and is type-stable (suits the Reactant JIT path). See
`docs/decisions.md` §19.
"""
struct SoA{NT <: NamedTuple}
    nt::NT
end
SoA(d::AbstractDict) = SoA(NamedTuple{Tuple(Symbol(k) for k in keys(d))}(Tuple(values(d))))
Base.getindex(s::SoA, k::AbstractString) = s.nt[Symbol(k)]
Base.getindex(s::SoA, k::Symbol) = s.nt[k]
Base.get(s::SoA, k::AbstractString, default) =
    haskey(s.nt, Symbol(k)) ? s.nt[Symbol(k)] : default
Base.get(s::SoA, k::Symbol, default) = haskey(s.nt, k) ? s.nt[k] : default
Base.haskey(s::SoA, k::AbstractString) = haskey(s.nt, Symbol(k))
# Key→String conversion (jl_cstr_to_string) is a foreigncall Zygote cannot differentiate. SoA keys
# are STRUCTURAL — they never depend on the (differentiated) numeric values — so these helpers are
# marked `Zygote.@nograd` in FabricPCZygoteExt. That lets an autodiff'd `compute_mu` ITERATE an SoA
# (generic nodes that sum over input edges) while gradients still flow through the values (`s.nt[i]`).
_soa_key(nt::NamedTuple, i::Int) = String(keys(nt)[i])
_soa_keys(nt::NamedTuple) = String[String(k) for k in keys(nt)]
Base.keys(s::SoA) = _soa_keys(s.nt)
Base.values(s::SoA) = values(s.nt)
Base.length(s::SoA) = length(s.nt)
Base.pairs(s::SoA) = (_soa_key(s.nt, i) => s.nt[i] for i in 1:length(s.nt))
function Base.iterate(s::SoA, st::Int=1)
    st > length(s.nt) && return nothing
    return (_soa_key(s.nt, st) => s.nt[st], st + 1)
end

"""
    NodeParams{W<:SoA, B<:SoA}

Learnable parameters of one node: named weight matrices ([`SoA`](@ref)) + named biases
([`SoA`](@ref)). `NodeParams(weights::AbstractDict, biases::AbstractDict)` coerces from plain
`Dict`s.
"""
struct NodeParams{W <: SoA, B <: SoA}
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

No `pre_activation` field: it is computed and consumed WITHIN a single forward, never stored.
Only the explicit-gradient path needs it (for `f'(pre)` in `pre_grad`), and it gets it directly
from `_forward_with_preact`. Mirrors upstream `b6f64ad`, which cut per-node inference state from
5 `(batch, features...)` tensors to 4 — a real saving in the in-place lane, which allocates one
of these per node per step.

PARAMETRIC (J-06). Fields were `::Any`, which boxed EVERY latent operand: `@code_warntype` showed
`forward(::Linear)` inferring to `Tuple{Any, NodeState}` even for a concrete node, and
`run_inference` on the MNIST-MLP allocating 6565 allocs / 208 MB per call (`Profile.Allocs`: 360
`Memory{NodeState}`, 320 `NodeState`, 240 boxed `Tuple{...}`). One INDEPENDENT type parameter per
field (not a shared `A`) so every construction site auto-infers a fully-concrete type with no edit,
and the two test sites that build partial states with `nothing` still work (each field free to be
`Nothing`). `::NodeState` dispatch annotations and `Dict{String,NodeState}` fields match the
UnionAll unchanged. `JET.report_call` was 0 errors before this — the code was CORRECT; the
instability was purely perf, so bit-identity against the tiers is the gate.
"""
struct NodeState{Z, M, R, E, L}
    z_latent::Z
    z_mu::M
    error::R
    energy::E
    latent_grad::L
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
