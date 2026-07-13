# Graph builder. Port of fabricpc/graph_assembly/graph_construction.py.
#
# Assembles node descriptors + edges + a task map into an immutable
# `GraphStructure`: resolves edge keys, builds per-node `NodeInfo` (degrees,
# edge keys, slots), validates endpoints/slots, and computes a topological order.
# When a `MuPCConfig` is passed as `scaling`, per-node muPC factors are computed
# (core/mupc.jl) and attached to each `NodeInfo.scaling_config`.

"""
    TaskMap(; task = node_or_name, …)

Maps task names (e.g. `x`, `y`) to node names. Values may be `AbstractNode`
instances or name strings.
"""
struct TaskMap
    map::Dict{String, String}
end

function TaskMap(; kwargs...)
    m = Dict{String, String}()
    for (k, v) in kwargs
        m[String(k)] = v isa AbstractNode ? v.name : String(v)
    end
    return TaskMap(m)
end

to_dict(tm::TaskMap) = copy(tm.map)

function _build_slots(node::AbstractNode, in_edges::Dict{String, EdgeInfo})
    slot_specs = get_slots(node)
    slots = Dict{String, SlotInfo}()
    for (slot_name, spec) in slot_specs
        in_neighbors = [e.source for e in values(in_edges) if e.slot == slot_name]
        if !spec.is_multi_input && length(in_neighbors) > 1
            throw(
                ArgumentError(
                    "slot '$slot_name' in node '$(node.name)' is single-input but " *
                    "has $(length(in_neighbors)) connections"
                )
            )
        end
        slots[slot_name] = SlotInfo(
            slot_name,
            node.name,
            spec.is_multi_input,
            spec.is_variance_scalable,
            spec.is_skip_connection,
            in_neighbors
        )
    end
    return slots
end

"""BFS topological sort over node in-degrees. Port of `_topological_sort`."""
function _topological_sort(
    node_names::Vector{String},
    infos::Dict{String, NodeInfo},
    edges::Dict{String, EdgeInfo}
)
    in_degree = Dict(name => infos[name].in_degree for name in node_names)
    queue = [name for name in node_names if in_degree[name] == 0]
    result = String[]
    while !isempty(queue)
        name = popfirst!(queue)
        push!(result, name)
        for out_key in infos[name].out_edges
            target = edges[out_key].target
            in_degree[target] -= 1
            if in_degree[target] == 0
                push!(queue, target)
            end
        end
    end
    if length(result) != length(node_names)
        @warn "graph contains cycles; using partial topological order"
    end
    return result
end

"""
    graph(nodes, edges, task_map, inference;
          graph_state_initializer = FeedforwardStateInit(), scaling = nothing) -> GraphStructure

Build a `GraphStructure` from node descriptors, `Edge`s, and a `TaskMap`. Port of
`graph()`. Pass a `MuPCConfig` as `scaling` to compute and attach per-node muPC
factors (`NodeInfo.scaling_config`); `nothing` (default) leaves scaling off.
"""
function graph(
    nodes::AbstractVector,
    edges::AbstractVector,
    task_map::TaskMap,
    inference;
    graph_state_initializer=FeedforwardStateInit(),
    scaling::Union{Nothing, MuPCConfig}=nothing
)
    # 1. Build EdgeInfo objects (preserve edge-list order).
    #
    # R-05b (docs/AUDIT_REGISTER.md section 8): `edge_keys_ordered` exists only because
    # Julia's `Dict` doesn't guarantee iteration order the way Python's does (upstream's
    # `edge_infos` dict — graph_construction.py:137-145 — is itself the ordered source of
    # truth: `in_edges = {k: e for k,e in edge_infos.items() if e.target==name}`). A literal
    # duplicate edge (same source/target/slot submitted twice) must therefore behave exactly
    # like a Python dict: the LATER `EdgeInfo` overwrites the value at the key's ORIGINAL
    # position, not append a second position — `push!` only on the key's first occurrence,
    # matching that semantics, so `in_keys`/`out_keys` below (and therefore `NodeInfo.in_edges`
    # /`in_degree`) count each unique (source,target,slot) triple once, not once per submission.
    # A missing `haskey` guard here previously let a duplicate edge inflate `in_degree`/
    # `in_edges`, silently double-counting the edge's forward/gradient contribution in the
    # JIT flat path (`jit_flat.jl`'s `CompiledPlan`, which is NOT dict-rekeyed and so has no
    # accidental dedup safety net the eager Dict-based consumers have).
    edge_infos = Dict{String, EdgeInfo}()
    edge_keys_ordered = String[]
    for edge in edges
        source = edge.source.name
        target = edge.target_node.name
        slot = edge.target_slot
        key = "$(source)->$(target):$(slot)"
        haskey(edge_infos, key) || push!(edge_keys_ordered, key)
        edge_infos[key] = EdgeInfo(key, source, target, slot)
    end

    node_names = String[node.name for node in nodes]
    node_name_set = Set(node_names)

    # 3. Validate edge endpoints.
    for (key, e) in edge_infos
        e.source == e.target && throw(ArgumentError("self-edge not allowed: '$key'"))
        e.source in node_name_set ||
            throw(ArgumentError("edge source node '$(e.source)' does not exist"))
        e.target in node_name_set ||
            throw(ArgumentError("edge target node '$(e.target)' does not exist"))
    end

    # 4. Per-node: slots + NodeInfo. Preserve edge order via edge_keys_ordered.
    node_map = Dict{String, AbstractNode}()
    infos = Dict{String, NodeInfo}()
    for node in nodes
        name = node.name
        haskey(node_map, name) && throw(ArgumentError("duplicate node name '$name'"))

        in_edges = Dict(
            k => edge_infos[k] for k in edge_keys_ordered if edge_infos[k].target == name
        )
        out_edges = Dict(
            k => edge_infos[k] for k in edge_keys_ordered if edge_infos[k].source == name
        )
        in_keys = [k for k in edge_keys_ordered if edge_infos[k].target == name]
        out_keys = [k for k in edge_keys_ordered if edge_infos[k].source == name]

        slots = _build_slots(node, in_edges)
        for (k, e) in in_edges
            haskey(slots, e.slot) || throw(
                ArgumentError(
                    "edge '$k' connects to non-existent slot '$(e.slot)' in node " *
                    "'$name'. Available: $(collect(keys(slots)))"
                )
            )
        end

        infos[name] = NodeInfo(
            name,
            node.shape,
            string(nameof(typeof(node))),
            slots,
            length(in_keys),
            length(out_keys),
            in_keys,
            out_keys,
            nothing
        )
        node_map[name] = node
    end

    node_order = _topological_sort(node_names, infos, edge_infos)

    # Compute + attach muPC scaling factors when requested (NodeInfo is immutable,
    # so reconstruct via Accessors @set). Terminal inputs / excluded outputs get
    # `nothing` (left as the default).
    if scaling !== nothing
        scalings = compute_mupc_scalings(node_map, infos, edge_infos, scaling, node_order)
        for name in keys(infos)
            sc = get(scalings, name, nothing)
            if sc !== nothing
                info = infos[name]
                infos[name] = @set info.scaling_config = sc
            end
        end
    end

    config = (inference=inference, graph_state_initializer=graph_state_initializer)
    return GraphStructure(
        node_map,
        infos,
        edge_infos,
        to_dict(task_map),
        node_order,
        node_names,
        config
    )
end
