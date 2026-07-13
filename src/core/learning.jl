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
            final_state.nodes[name];
            info=info
        )
        # F-03 boundary check: sgd_update/AdamW.step! assume grad and param key
        # sets match exactly (SGD silently drops a missing-in-grads param; AdamW
        # throws an opaque KeyError). Every shipped node's forward_and_weight_grads
        # guarantees this by construction, but AbstractNode is an open extension
        # point — catch a violation HERE, at the one place grads are produced, with
        # a clear message rather than downstream in either optimizer.
        node_params = params.nodes[name]
        Set(keys(grad_params.weights)) == Set(keys(node_params.weights)) || throw(
            ArgumentError(
                "node '$name': forward_and_weight_grads returned weight keys " *
                "$(sort(collect(keys(grad_params.weights)))) but params has " *
                "$(sort(collect(keys(node_params.weights)))) — an optimizer step " *
                "would silently drop or KeyError on the mismatched keys"
            )
        )
        Set(keys(grad_params.biases)) == Set(keys(node_params.biases)) || throw(
            ArgumentError(
                "node '$name': forward_and_weight_grads returned bias keys " *
                "$(sort(collect(keys(grad_params.biases)))) but params has " *
                "$(sort(collect(keys(node_params.biases)))) — an optimizer step " *
                "would silently drop or KeyError on the mismatched keys"
            )
        )
        grad_params = scale_weight_grads(grad_params, info.scaling_config)
        grads[name] = grad_params
    end
    return GraphParams(grads)
end
