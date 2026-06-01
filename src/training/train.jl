# Training step. Port of the single-device path in fabricpc/training/train.py.
#
# Full PC training step: clamp data → init state → run inference to convergence
# → compute local weight gradients → apply optimizer update. v0 ships plain SGD
# (`param -= lr·grad`, faithful to optax.sgd); momentum/Adam via Optimisers.jl
# and the pmap multi-device path are deferred. JAX `jit` wrapping → eager here
# (Phase B is eager ground truth; Reactant JIT is a later concern).

"""
    get_graph_param_gradient(params, batch, structure, rng) -> (grads, energy_per_sample, final_state)

Run inference for `batch` and compute local weight gradients without updating
params. `batch` maps task names (e.g. `"x"`, `"y"`) to (batch, features) arrays.
Port of `get_graph_param_gradient`.
"""
function get_graph_param_gradient(
    params::GraphParams,
    batch::AbstractDict,
    structure::GraphStructure,
    rng::AbstractRNG
)
    batch_size = size(first(values(batch)), 1)

    clamps = Dict{String, Any}()
    for (task, value) in batch
        if haskey(structure.task_map, task)
            clamps[structure.task_map[task]] = value
        end
    end

    init_state = initialize_graph_state(
        structure, batch_size, rng; clamps=clamps, params=params
    )
    final_state = run_inference(params, init_state, clamps, structure)

    # Energy over internal/output nodes only (ignore terminal inputs), per sample.
    e = 0.0f0
    for name in structure.node_names
        if structure.infos[name].in_degree > 0
            e += sum(final_state.nodes[name].energy)
        end
    end
    e /= batch_size

    grads = compute_local_weight_gradients(params, final_state, structure)
    return grads, e, final_state
end

"""
    sgd_update(params, grads, lr) -> GraphParams

Plain SGD: `param -= lr · grad`, walking the GraphParams tree. Nodes with no
parameters (empty grad dicts) pass through unchanged. Faithful to `optax.sgd` +
`optax.apply_updates`.
"""
sgd_update(params::GraphParams, grads::GraphParams, lr::Real) = GraphParams(
    Dict(
        name => sgd_update(params.nodes[name], grads.nodes[name], lr) for
        name in keys(params.nodes)
    )
)

function sgd_update(params::NodeParams, grads::NodeParams, lr::Real)
    η = Float32(lr)
    weights = Dict{String, Matrix{Float32}}(
        k => params.weights[k] .- η .* grads.weights[k] for k in keys(grads.weights)
    )
    biases = Dict{String, Matrix{Float32}}(
        k => params.biases[k] .- η .* grads.biases[k] for k in keys(grads.biases)
    )
    return NodeParams(weights, biases)
end

"""
    train_step(params, batch, structure, lr, rng) -> (params, energy_per_sample, final_state)

One PC training step with an SGD weight update. Port of `train_step`.
"""
function train_step(
    params::GraphParams,
    batch::AbstractDict,
    structure::GraphStructure,
    lr::Real,
    rng::AbstractRNG
)
    grads, energy, final_state = get_graph_param_gradient(params, batch, structure, rng)
    params = sgd_update(params, grads, lr)
    return params, energy, final_state
end
