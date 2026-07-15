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
                Dict{String, Array{Float32}}(), Dict{String, Array{Float32}}()
            )
            continue
        end

        # ORDERED by `in_edges`: `initialize_params` iterates these shapes to build one weight
        # per edge, drawing sequentially from the shared `rng`, so the iteration order decides
        # WHICH draw lands on WHICH edge. A hash `Dict` made that order arbitrary. Upstream also
        # reports the first offending edge in insertion order (`nodes/base.py:692-707`).
        # (Exact JAX RNG bit-parity is an explicit non-goal — see initializers.jl:4-6 — and the
        # conformance fixtures inject upstream arrays rather than relying on our RNG stream.)
        input_shapes = OrderedDict{String, Tuple}()
        for edge_key in info.in_edges
            src = structure.edges[edge_key].source
            input_shapes[edge_key] = structure.infos[src].shape
        end

        # Parameterless nodes (SkipConnection, IdentityNode) carry no `weight_init`
        # field; their `initialize_params` ignores the init arg, so fall back to a
        # default. Without this, an in-degree>0 SkipConnection (a residual summation
        # in a transformer) would FieldError here.
        wi = hasproperty(node, :weight_init) ? node.weight_init : NormalInitializer()
        node_params[name] = initialize_params(
            node, rng, info.shape, input_shapes, wi
        )
    end
    return GraphParams(node_params)
end
