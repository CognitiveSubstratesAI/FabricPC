# Parameter initialization. Port of fabricpc/graph_initialization/params_initializer.py.
#
# Each node initializes its own parameters. Source nodes (in_degree 0) hold none.
# Upstream splits a JAX key per parametered node; we thread one RNG sequentially.

"""
    initialize_params(structure, rng) -> GraphParams

Initialize parameters for every node in `structure`. Port of `initialize_params`.
"""
function initialize_params(structure::GraphStructure, rng::AbstractRNG)
    node_params = Dict{String, NodeParams}()
    for name in structure.node_names
        node = structure.nodes[name]
        info = structure.infos[name]
        if info.in_degree == 0
            node_params[name] = NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            )
            continue
        end

        input_shapes = Dict{String, Tuple}()
        for edge_key in info.in_edges
            src = structure.edges[edge_key].source
            input_shapes[edge_key] = structure.infos[src].shape
        end

        node_params[name] = initialize_params(
            node, rng, info.shape, input_shapes, node.weight_init
        )
    end
    return GraphParams(node_params)
end
