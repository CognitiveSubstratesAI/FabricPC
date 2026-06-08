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

"""
    predict(params, structure, inputs, rng; output_task = "y") -> array

Inference-only forward: clamp the given input task(s) (e.g. `"x"`), relax the
graph to convergence, and return the output task node's prediction (`z_mu`). For
a classifier, `argmax` over the feature axis gives the predicted class. Mirrors
the inference half of upstream `eval_step` (single-device, eager).
"""
function predict(
    params::GraphParams,
    structure::GraphStructure,
    inputs::AbstractDict,
    rng::AbstractRNG;
    output_task::AbstractString="y"
)
    batch_size = size(first(values(inputs)), 1)
    clamps = Dict{String, Any}()
    for (task, value) in inputs
        haskey(structure.task_map, task) && (clamps[structure.task_map[task]] = value)
    end
    init_state = initialize_graph_state(
        structure, batch_size, rng; clamps=clamps, params=params
    )
    final_state = run_inference(params, init_state, clamps, structure)
    return final_state.nodes[structure.task_map[output_task]].z_mu
end

# One local training step, dispatching on the optimizer (AdamW vs plain-SGD lr).
# Both call the LOCAL `train_step` — NO backprop through the network/inference loop.
# Runtime dispatch (not `::AdamW` in the signature) because AdamW lives in a sibling
# file included after this one; the body resolves `train_step!` at call time.
_pcn_step(p::GraphParams, opt, b, s, rng) =
    opt isa Real ? train_step(p, b, s, opt, rng) : train_step!(opt, p, b, s, rng)

"""
    train_pcn(params, structure, train_loader, opt; num_epochs = 10,
              rng = Random.default_rng(), verbose = true,
              epoch_callback = nothing, iter_callback = nothing)
        -> (params, iter_energies, epoch_results)

Train a predictive-coding network with **local learning — no backprop.** Loops
`num_epochs` (fractional supported) × batches over `train_loader` (a re-iterable of
task→array batches), calling the local `train_step!`/`train_step` and accumulating
per-batch energy. `opt` is an `AdamW` optimizer OR a `Real` learning rate (plain
SGD). Returns the trained params, a `[epochs][batches]` energy history, and the
collected `epoch_callback` results. Port of `train_pcn` (single-device path;
pmap/multi-GPU deferred).
"""
function train_pcn(
    params::GraphParams,
    structure::GraphStructure,
    train_loader,
    opt;
    num_epochs::Real=10,
    rng::AbstractRNG=Random.default_rng(),
    verbose::Bool=true,
    epoch_callback=nothing,
    iter_callback=nothing
)
    total_epochs = ceil(Int, num_epochs)
    frac = num_epochs - floor(num_epochs)
    num_batches = length(train_loader)
    iter_results = Vector{Vector{Float32}}()
    epoch_results = Any[]
    for epoch in 1:total_epochs
        max_batches =
            if (epoch == total_epochs && frac > 0.0)
                round(Int, frac * num_batches)
            else
                num_batches
            end
        batch_energies = Float32[]
        for (bidx, batch) in enumerate(train_loader)
            bidx > max_batches && break
            params, energy, _ = _pcn_step(params, opt, batch, structure, rng)
            push!(batch_energies, Float32(energy))
            iter_callback === nothing || iter_callback(epoch, bidx, energy)
        end
        push!(iter_results, batch_energies)
        epoch_callback === nothing ||
            push!(epoch_results, epoch_callback(epoch, params, structure, rng))
        if verbose
            m =
                if isempty(batch_energies)
                    0.0f0
                else
                    sum(batch_energies) / length(batch_energies)
                end
            println(
                "[train_pcn] epoch $epoch/$total_epochs  mean energy $(round(m; digits=5))"
            )
        end
    end
    return params, iter_results, epoch_results
end

"""
    evaluate_pcn(params, structure, test_loader; input_task = "x",
                 output_task = "y", rng = Random.default_rng())
        -> Dict{String,Float64}

Evaluate a PC classifier (**no backprop**): for each batch, clamp the input, relax
the graph, and `argmax` the output node's prediction vs the label's `argmax` →
accuracy; also accumulate mean per-sample energy (with input+label clamped).
Returns `Dict("accuracy" => …, "energy" => …)`. Port of `evaluate_pcn`
(single-device).
"""
function evaluate_pcn(
    params::GraphParams,
    structure::GraphStructure,
    test_loader;
    input_task::AbstractString="x",
    output_task::AbstractString="y",
    rng::AbstractRNG=Random.default_rng()
)
    correct = 0
    total = 0
    energy_sum = 0.0f0
    nbatch = 0
    for batch in test_loader
        labels = batch[output_task]
        pred = predict(
            params, structure, Dict(input_task => batch[input_task]), rng;
            output_task=output_task
        )
        for i in 1:size(pred, 1)
            correct += (argmax(@view pred[i, :]) == argmax(@view labels[i, :])) ? 1 : 0
        end
        total += size(pred, 1)
        _, e, _ = get_graph_param_gradient(params, batch, structure, rng)
        energy_sum += e
        nbatch += 1
    end
    return Dict{String, Float64}(
        "accuracy" => total == 0 ? 0.0 : correct / total,
        "energy" => nbatch == 0 ? 0.0 : energy_sum / nbatch
    )
end
