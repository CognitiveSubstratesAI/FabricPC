# muPC scaling computation. Port of fabricpc/core/mupc.py.
#
# THE novel research layer: per-edge scaling factors that keep activations,
# prediction errors, and gradients O(1) across networks of arbitrary width and
# topology (Depth-μP, Yang et al.; muPC, Innocenti et al. arXiv:2505.13124).
# Pure scalar/topology math — NO autodiff. The inference + learning + state-init
# loops already invoke scale_inputs/scale_*_grads at their call-sites (no-ops
# when scaling is off), so enabling muPC is purely additive.
#
# Per in-edge, keyed on the target slot's flags:
#   variance-scalable slot:  a = gain / sqrt(fan_in · K_slot · L)   (hidden)
#                            a = gain / (fan_in · sqrt(K_slot · L)) (output, O(1/N))
#   non-scalable slot:       absent from every per-edge dict ⇒ identity
#                            pass-through (NOT ×1.0 — preserves input dtype).
# where K_slot = in-degree of that specific slot, gain = variance_gain(act),
# and L = residual depth (count of skip-connection merge nodes along the longest
# path; 1 for pure chains). Top-down grad scale = a · jacobian_gain(act).

"""
Per-node muPC scaling factors, all per-edge dicts keyed by edge key. Edges into
non-variance-scalable slots are ABSENT (call-sites treat a missing key as a
no-op, preserving dtype). Port of `MuPCScalingFactors`.
"""
struct MuPCScalingFactors
    forward_scale::Dict{String, Float32}       # a_l, applied to inputs pre-forward
    self_grad_scale::Float32                  # dE/dz_self scale (1.0; already O(1))
    topdown_grad_scale::Dict{String, Float32}  # a · jacobian_gain, applied to input grads
    weight_grad_scale::Dict{String, Float32}   # 1.0 (optimizer handles magnitude)
end

"""
    MuPCConfig(; include_output = false, terminal_input_variance = 1.0)

muPC configuration passed as the `scaling` argument to `graph()`. `include_output`
adds output nodes (out_degree 0) to scaling with the O(1/N) output formula — use
`true` with MSE/Gaussian energy, `false` with softmax+CE. Port of `MuPCConfig`
(deprecated depth_metric/min_depth fields dropped). `terminal_input_variance` is
reserved (unused in the simplified formula).
"""
struct MuPCConfig
    include_output::Bool
    terminal_input_variance::Float32
end
MuPCConfig(; include_output=false, terminal_input_variance=1.0) =
    MuPCConfig(include_output, Float32(terminal_input_variance))

"""
Residual depth L: the number of skip-connection merge nodes (nodes with ≥1
`is_skip_connection` slot) along the longest path. 0 for a pure chain, D for a
D-block ResNet. Port of `_count_skip_connections_depth`.
"""
function _count_skip_connections_depth(
    infos::Dict{String, NodeInfo},
    edges::Dict{String, EdgeInfo},
    node_order::Vector{String}
)
    skip_counts = Dict{String, Int}()
    for name in node_order
        info = infos[name]
        if info.in_degree == 0
            skip_counts[name] = 0
            continue
        end
        max_pred = 0
        for edge_key in info.in_edges
            max_pred = max(max_pred, get(skip_counts, edges[edge_key].source, 0))
        end
        has_skip = any(s -> s.is_skip_connection, values(info.slots))
        skip_counts[name] = has_skip ? max_pred + 1 : max_pred
    end
    return isempty(skip_counts) ? 0 : maximum(values(skip_counts))
end

"""
    compute_mupc_scalings(nodes, infos, edges, config, node_order) -> Dict{String,Any}

Per-node `MuPCScalingFactors` from graph topology (or `nothing` for terminal
inputs, and for outputs unless `include_output`). Port of `compute_mupc_scalings`.
"""
function compute_mupc_scalings(
    nodes::Dict{String, AbstractNode},
    infos::Dict{String, NodeInfo},
    edges::Dict{String, EdgeInfo},
    config::MuPCConfig,
    node_order::Vector{String}
)
    output_nodes = Set(name for name in keys(infos) if infos[name].out_degree == 0)
    skip_depth = _count_skip_connections_depth(infos, edges, node_order)
    L = max(skip_depth, 1)

    scalings = Dict{String, Any}()
    for name in node_order
        info = infos[name]
        node = nodes[name]

        if info.in_degree == 0
            scalings[name] = nothing
            continue
        end
        is_output = name in output_nodes
        if is_output && !config.include_output
            scalings[name] = nothing
            continue
        end

        gain = variance_gain(node.activation)
        jac = jacobian_gain(node.activation)

        forward_scale = Dict{String, Float32}()
        topdown_grad_scale = Dict{String, Float32}()
        weight_grad_scale = Dict{String, Float32}()

        for edge_key in info.in_edges
            slot_info = info.slots[edges[edge_key].slot]
            slot_info.is_variance_scalable || continue   # non-scalable ⇒ omit (no-op)

            K_slot = length(slot_info.in_neighbors)
            source_shape = infos[edges[edge_key].source].shape
            fan_in = get_weight_fan_in(node, source_shape)

            a = if is_output
                gain / (fan_in * sqrt(K_slot * L))
            else
                gain / sqrt(fan_in * K_slot * L)
            end
            forward_scale[edge_key] = Float32(a)
            topdown_grad_scale[edge_key] = Float32(a * jac)
            weight_grad_scale[edge_key] = 1.0f0
        end

        scalings[name] = MuPCScalingFactors(
            forward_scale, 1.0f0, topdown_grad_scale, weight_grad_scale
        )
    end
    return scalings
end
