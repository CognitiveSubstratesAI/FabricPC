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
