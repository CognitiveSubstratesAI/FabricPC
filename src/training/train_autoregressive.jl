# Autoregressive / sequence training. Port of fabricpc/training/train_autoregressive.py.
#
# PHASED port. Phase 1 (this commit) = the shared foundation both the PC autoregressive trainer
# and the backprop baseline build on (upstream train_backprop.py imports both of these). Later
# phases add the PC train step + driver (train_step_autoregressive / train_autoregressive),
# generation + sampling (_generation_step / generate_autoregressive), and the eval loop.

"""
    create_causal_mask(seq_len) -> Matrix{Float32}  (seq_len × seq_len)

Lower-triangular causal mask: `mask[i,j] = 1 if j ≤ i else 0` — position i may attend to 0…i.
Port of `train_autoregressive.py:29` (`jnp.tril(jnp.ones((L, L)))`).

NOTE: this is the binary TRAINING/task mask (used via `task_map["causal_mask"]`). It is distinct
from the additive (−1e9) ATTENTION mask `_tb_causal_mask` in nodes/transformer.jl — do not conflate.
"""
create_causal_mask(seq_len::Integer) =
    Float32[j <= i ? 1.0f0 : 0.0f0 for i in 1:seq_len, j in 1:seq_len]

"""
    compute_loss(state, targets, output_node; loss_type=:cross_entropy) -> Float32

Scalar monitoring loss on the output node's prediction (`state.nodes[output_node].z_mu`), averaged
over the batch. Port of `train_autoregressive.py:40`.

- `:cross_entropy` — expects one-hot `targets`: `−mean(Σ_v targets · log(pred + 1e-10))`, the inner
  sum over the last (feature/vocab) axis, the mean over the remaining (batch[, seq]) axes.
- `:mse` — `mean((pred − targets)²)`.
"""
function compute_loss(state::GraphState, targets, output_node::AbstractString;
                      loss_type::Symbol = :cross_entropy)
    pred = state.nodes[output_node].z_mu
    if loss_type === :cross_entropy
        s = sum(targets .* log.(pred .+ 1.0f-10); dims = ndims(pred))   # Σ over vocab axis (kept dim)
        return -Float32(sum(s) / length(s))                            # mean over batch[, seq]
    elseif loss_type === :mse
        d = (pred .- targets) .^ 2
        return Float32(sum(d) / length(d))
    else
        throw(ArgumentError("compute_loss: unknown loss_type $loss_type (use :cross_entropy or :mse)"))
    end
end

"""
    train_step_autoregressive(params, opt, batch, structure, rng) -> (params, energy, ce_loss, final_state)

One PC training step for sequence/autoregressive data: clamp the batch tasks via `task_map`, relax
the graph by inference, apply local weight gradients (`opt` = an `AdamW` or a plain-SGD learning
rate), and additionally return the output cross-entropy (perplexity metric — NOT used for the
update). Port of `train_step_autoregressive` (train_autoregressive.py:76).

DIVERGENCE (intentional — not a 1:1 transplant): upstream clamps a causal-mask NODE
(`task_map["causal_mask"]`, broadcast to (B,1,S,S)). Our `TransformerBlock` masks INLINE via its
`causal` flag (`_tb_causal_mask`, fixed at graph construction), so there is no mask node to clamp.
We also use the stateful `AdamW`/lr optimizer rather than threading an optax `opt_state`.
"""
function train_step_autoregressive(params::GraphParams, opt, batch::AbstractDict,
                                   structure::GraphStructure, rng::AbstractRNG)
    params, energy, final_state = _pcn_step(params, opt, batch, structure, rng)
    ce = compute_loss(final_state, batch["y"], structure.task_map["y"]; loss_type = :cross_entropy)
    return params, energy, ce, final_state
end

"""
    train_autoregressive(params, structure, batches, opt, num_epochs, rng;
                         verbose=true, epoch_callback=nothing) -> (params, iter_energies, epoch_results)

Main PC autoregressive training loop. `batches` is any iterable of task→array `Dict`s (each with at
least `"x"`/`"y"`). Per epoch, runs `train_step_autoregressive` over every batch, accumulating the
per-batch energy and the CE/perplexity metric; `epoch_callback(epoch, params, structure)` runs
after each epoch. Port of `train_autoregressive` (train_autoregressive.py:175). Eager (no
`jax.jit`); fractional epochs and gradient-accumulation (an upstream read-but-unused stub) are not
ported — integer epochs.
"""
function train_autoregressive(params::GraphParams, structure::GraphStructure, batches,
                              opt, num_epochs::Integer, rng::AbstractRNG;
                              verbose::Bool = true, epoch_callback = nothing)
    iter_energies = Vector{Vector{Float32}}()
    epoch_results = Vector{Any}()
    for epoch in 1:num_epochs
        batch_e = Float32[]; tot_e = 0.0f0; tot_ce = 0.0f0; n = 0
        for batch in batches
            params, energy, ce, _ = train_step_autoregressive(params, opt, batch, structure, rng)
            tot_e += energy; tot_ce += ce; n += 1
            push!(batch_e, Float32(energy))
        end
        push!(iter_energies, batch_e)
        push!(epoch_results,
            epoch_callback === nothing ? nothing : epoch_callback(epoch, params, structure))
        if verbose && n > 0
            avg_ce = tot_ce / n
            @info "train_autoregressive" epoch="$epoch/$num_epochs" energy=round(tot_e / n; digits=4) ce=round(avg_ce; digits=4) perplexity=round(exp(avg_ce); digits=2)
        end
    end
    return params, iter_energies, epoch_results
end
