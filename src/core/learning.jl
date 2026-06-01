# Local learning. Port of fabricpc/core/learning.py.
#
# After inference converges, each node computes its weight gradient from its own
# local error signal (Hebbian predictive-coding rule) — no global backprop.
# Source nodes (in_degree 0) have no parameters and contribute empty gradients.

"""
    compute_local_weight_gradients(params, final_state, structure) -> GraphParams

Per-node explicit local weight gradients computed from the converged inference
state. Port of `compute_local_weight_gradients`.
"""
function compute_local_weight_gradients(
    params::GraphParams,
    final_state::GraphState,
    structure::GraphStructure
)
    grads = Dict{String, NodeParams}()
    for name in structure.node_names
        node = structure.nodes[name]
        info = structure.infos[name]
        if info.in_degree == 0
            grads[name] = NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            )
            continue
        end

        in_data = gather_inputs(info, structure, final_state)
        scaled_inputs = scale_inputs(in_data, info.scaling_config)

        _, grad_params = forward_and_weight_grads(
            node,
            params.nodes[name],
            scaled_inputs,
            final_state.nodes[name]
        )
        grad_params = scale_weight_grads(grad_params, info.scaling_config)
        grads[name] = grad_params
    end
    return GraphParams(grads)
end
