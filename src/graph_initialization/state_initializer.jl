# State initialization. Port of fabricpc/graph_initialization/state_initializer.py.
#
# v0 ships FeedforwardStateInit (the default): initialize every latent from its
# node's `latent_init`, overlay clamps, then propagate forward in topological
# order setting z_latent ← z_mu for unclamped nodes. Global / NodeDistribution
# strategies are deferred (not needed for the trainable core).

"""Feedforward state initializer: propagate predictions through the DAG. Port of `FeedforwardStateInit`."""
struct FeedforwardStateInit end

"""
    initialize_state(::FeedforwardStateInit, structure, batch_size, rng, clamps, params) -> GraphState

Two passes: (1) initialize all latents from each node's `latent_init` and overlay
clamps; (2) walk `node_order`, run each parametered node's forward pass, and for
unclamped nodes set `z_latent = z_mu`. Requires `params`.
"""
function initialize_state(
    ::FeedforwardStateInit,
    structure::GraphStructure,
    batch_size::Int,
    rng::AbstractRNG,
    clamps::AbstractDict,
    params::GraphParams
)
    node_state_dict = Dict{String, NodeState}()
    for name in structure.node_names
        node = structure.nodes[name]
        info = structure.infos[name]
        shp = (batch_size, info.shape...)
        z = initialize(rng, shp, node.latent_init)
        node_state_dict[name] = NodeState(
            z,                                   # z_latent
            zeros(Float32, shp...),              # z_mu
            zeros(Float32, shp...),              # error
            zeros(Float32, batch_size),          # energy (per-sample)
            zeros(Float32, shp...)              # latent_grad
        )
    end
    state = GraphState(node_state_dict, batch_size)
    state = set_latents_to_clamps(state, clamps)

    # Second pass: feedforward propagation in topological order.
    for name in structure.node_order
        info = structure.infos[name]
        info.in_degree > 0 || continue
        node = structure.nodes[name]
        ns = state.nodes[name]
        in_data = gather_inputs(info, structure, state)
        scaled_inputs = scale_inputs(in_data, info.scaling_config)
        projected = forward(node, params.nodes[name], scaled_inputs, ns)

        if !haskey(clamps, name)
            ns = update_state(ns; z_latent=projected.z_mu, z_mu=projected.z_mu)
        else
            ns = update_state(
                ns;
                z_mu=projected.z_mu,
                error=projected.error,
                energy=projected.energy
            )
        end
        state = put_node(state, name, ns)
    end
    return state
end

"""
    initialize_graph_state(structure, batch_size, rng; clamps, state_init, params) -> GraphState

Initialize the graph state with the chosen (or default) strategy. Port of
`initialize_graph_state` (its float-clamp validation is unnecessary in v0 — all
latents are float).
"""
function initialize_graph_state(
    structure::GraphStructure,
    batch_size::Int,
    rng::AbstractRNG;
    clamps::AbstractDict=Dict{String, Any}(),
    state_init=nothing,
    params::Union{GraphParams, Nothing}=nothing
)
    si = state_init === nothing ? structure.config.graph_state_initializer : state_init
    return initialize_state(si, structure, batch_size, rng, clamps, params)
end
