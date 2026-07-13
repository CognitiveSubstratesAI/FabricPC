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
    loss_type::Symbol=:cross_entropy)
    pred = state.nodes[output_node].z_mu
    if loss_type === :cross_entropy
        d = ndims(pred)
        # `pred` may be probabilities (a softmax-output node) or raw LOGITS (e.g. VocabProjection,
        # which emits unnormalized scores). Use log-softmax when it is not a valid distribution so
        # `log` never sees a negative (DomainError) and the metric is correct either way.
        is_prob = all(>=(0.0f0), pred) && all(<(1.0f-2), abs.(sum(pred; dims=d) .- 1.0f0))
        logp = if is_prob
            log.(pred .+ 1.0f-10)
        else
            m = maximum(pred; dims=d)
            (pred .- m) .- log.(sum(exp.(pred .- m); dims=d))        # numerically-stable log-softmax
        end
        s = sum(targets .* logp; dims=d)                             # Σ over vocab axis (kept dim)
        return -Float32(sum(s) / length(s))                            # mean over batch[, seq]
    elseif loss_type === :mse
        d = (pred .- targets) .^ 2
        return Float32(sum(d) / length(d))
    else
        throw(
            ArgumentError(
                "compute_loss: unknown loss_type $loss_type (use :cross_entropy or :mse)"
            )
        )
    end
end

"""
    train_step_autoregressive(params, opt, batch, structure, rng) -> (params, energy, ce_loss, final_state)

One PC training step for sequence/autoregressive data: clamp the batch tasks via `task_map`, relax
the graph by inference, apply local weight gradients (`opt` = an `AdamW` or a plain-SGD learning
rate), and additionally return the output cross-entropy (perplexity metric — NOT used for the
update). Port of `train_step_autoregressive` (train_autoregressive.py:81).

`batch["y"]` may be raw integer token ids `(B, S)` (the loader contract — see `_TokenSequenceLoader`
docstring, dataloader.py) OR pre-shaped one-hot `(B, S, V)`. Raw ids are one-hot-expanded HERE (via
`_one_hot`, vocab size from `structure.infos[task_map["y"]].shape[end]`) *before* `_pcn_step` builds
clamps, so the clamped output node — not `compute_loss` alone — sees the expanded array. Port of the
`y.ndim`/`jax.nn.one_hot` block (train_autoregressive.py:127-139). DIVERGENCE (intentional): upstream
raises unless `y.ndim == 2` (it always one-hots); this also passes already-one-hot `y` through
unchanged (`ndim` already matching the prediction), so callers that pass ready-made one-hot batches
(e.g. `test_sequence_training.jl`'s plain-classifier step) keep working. `compute_loss` itself keeps
no ndim-check of its own (unlike upstream's, train_autoregressive.py:67-68): expansion happens
exactly once, here, so `compute_loss` can assume already-shaped targets.

DIVERGENCE (intentional — not a 1:1 transplant): upstream clamps a causal-mask NODE
(`task_map["causal_mask"]`, broadcast to (B,1,S,S)). Our `TransformerBlock` masks INLINE via its
`causal` flag (`_tb_causal_mask`, fixed at graph construction), so there is no mask node to clamp.
We also use the stateful `AdamW`/lr optimizer rather than threading an optax `opt_state`.
"""
function train_step_autoregressive(params::GraphParams, opt, batch::AbstractDict,
    structure::GraphStructure, rng::AbstractRNG)
    output_node = structure.task_map["y"]
    onehot_ndims = length(structure.infos[output_node].shape) + 1     # (B, S..., V)
    y = batch["y"]
    if ndims(y) == onehot_ndims - 1
        vocab_size = structure.infos[output_node].shape[end]
        y = _one_hot(y, vocab_size)
        batch = copy(batch)
        batch["y"] = y
    elseif ndims(y) != onehot_ndims
        throw(
            ArgumentError(
                "train_step_autoregressive: batch[\"y\"] has $(ndims(y)) dims; expected " *
                "$(onehot_ndims - 1) (raw token ids) or $onehot_ndims (one-hot)"
            )
        )
    end
    params, energy, final_state = _pcn_step(params, opt, batch, structure, rng)
    ce = compute_loss(final_state, y, output_node; loss_type=:cross_entropy)
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
    verbose::Bool=true, epoch_callback=nothing)
    iter_energies = Vector{Vector{Float32}}()
    epoch_results = Vector{Any}()
    for epoch in 1:num_epochs
        batch_e = Float32[];
        tot_e = 0.0f0;
        tot_ce = 0.0f0;
        n = 0
        for batch in batches
            params, energy, ce, _ = train_step_autoregressive(
                params, opt, batch, structure, rng
            )
            tot_e += energy;
            tot_ce += ce;
            n += 1
            push!(batch_e, Float32(energy))
        end
        push!(iter_energies, batch_e)
        push!(epoch_results,
            epoch_callback === nothing ? nothing : epoch_callback(epoch, params, structure))
        if verbose && n > 0
            avg_ce = tot_ce / n
            @info "train_autoregressive" epoch="$epoch/$num_epochs" energy=round(
                tot_e / n; digits=4
            ) ce=round(avg_ce; digits=4) perplexity=round(exp(avg_ce); digits=2)
        end
    end
    return params, iter_energies, epoch_results
end

"""
    _eval_step_autoregressive(params, structure, batch, rng) -> (ce, predictions, y)

Single evaluation step: clamp ONLY `"x"` (unlike training, `"y"` is never clamped — the network
predicts freely) and relax by inference; `batch["y"]` is one-hot-expanded exactly as
`train_step_autoregressive` does (F-05: raw ids `(B, S)` or pre-shaped one-hot `(B, S, V)`) and
returned alongside so the caller doesn't repeat the ndim-check. Port of
`_eval_step_autoregressive` (train_autoregressive.py:541-592). DIVERGENCE: no external
causal-mask clamp branch — see `train_step_autoregressive`'s docstring (`TransformerBlock` masks
inline, so there is no mask node to clamp here either); no `jax.jit` wrapper (this stack is
eager throughout).
"""
function _eval_step_autoregressive(
    params::GraphParams, structure::GraphStructure, batch::AbstractDict, rng::AbstractRNG
)
    output_node = structure.task_map["y"]
    clamps = Dict{String, Any}(structure.task_map["x"] => batch["x"])
    state = initialize_graph_state(
        structure, size(batch["x"], 1), rng; clamps=clamps, params=params
    )
    final_state = run_inference(params, state, clamps, structure)

    y = batch["y"]
    onehot_ndims = length(structure.infos[output_node].shape) + 1     # (B, S..., V)
    if ndims(y) == onehot_ndims - 1
        y = _one_hot(y, structure.infos[output_node].shape[end])
    elseif ndims(y) != onehot_ndims
        throw(
            ArgumentError(
                "_eval_step_autoregressive: batch[\"y\"] has $(ndims(y)) dims; expected " *
                "$(onehot_ndims - 1) (raw token ids) or $onehot_ndims (one-hot)"
            )
        )
    end
    ce = compute_loss(final_state, y, output_node; loss_type=:cross_entropy)
    predictions = final_state.nodes[output_node].z_mu
    return ce, predictions, y
end

"""
    _count_correct(predictions, y) -> Int

Count next-token predictions where `argmax` over the vocab axis (3rd dim, `(B, S, V)`) agrees
between `predictions` and the (possibly one-hot) target `y`. Factored out of
`evaluate_autoregressive` so the accuracy computation is unit-testable on its own — same rationale
as `_sample_next` above (upstream inlines this, train_autoregressive.py:688-696).
"""
function _count_correct(predictions::AbstractArray{<:Real, 3}, y::AbstractArray{<:Real, 3})
    B, S, _ = size(predictions)
    correct = 0
    for b in 1:B, s in 1:S
        correct += (argmax(@view predictions[b, s, :]) == argmax(@view y[b, s, :])) ? 1 : 0
    end
    return correct
end

"""
    evaluate_autoregressive(params, structure, test_loader, rng; debug=false)
        -> Dict{String,Float64}

Evaluate a trained autoregressive model on held-out data: mean cross-entropy loss, perplexity
(`exp(loss)`), and next-token accuracy, over every batch in `test_loader` (any iterable of
task→array `Dict`s with `"x"`/`"y"` — e.g. `CharDataLoader`, `examples/char_lm_pc.jl`, matching
`train_autoregressive`'s own loader contract). `debug=true` prints the same first-batch
diagnostics as upstream (per-token CE, per-token intrinsic perplexity, probability mass on the
correct token). Port of `evaluate_autoregressive` (train_autoregressive.py:595-710). DIVERGENCE:
no `config`/`use_causal_mask` parameter — the upstream causal-mask clamp branch this fed is dead
code in this port (see `_eval_step_autoregressive` above); `rng` is threaded sequentially
(Julia's mutable-state RNG idiom, matching `train_autoregressive`'s own signature) rather than
pre-split via `jax.random.split` (a JIT-specific pattern with no Julia-eager equivalent); no
`jax.jit` wrapper.
"""
function evaluate_autoregressive(
    params::GraphParams, structure::GraphStructure, test_loader, rng::AbstractRNG;
    debug::Bool=false
)
    haskey(structure.task_map, "y") ||
        throw(
            ArgumentError("evaluate_autoregressive: structure must have \"y\" in task_map")
        )

    total_loss = 0.0f0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for (batch_idx, batch) in enumerate(test_loader)
        ce, predictions, y = _eval_step_autoregressive(params, structure, batch, rng)
        total_loss += ce

        if debug && batch_idx == 1
            logp = log.(predictions .+ 1.0f-10)
            per_token_loss = -dropdims(sum(y .* logp; dims=3); dims=3)
            intrinsic_ppl = exp.(-dropdims(sum(predictions .* logp; dims=3); dims=3))
            correct_probs = dropdims(sum(y .* predictions; dims=3); dims=3)
            @info "evaluate_autoregressive debug (batch 1)" per_token_ce=(
                min=minimum(per_token_loss), max=maximum(per_token_loss),
                mean=sum(per_token_loss) / length(per_token_loss)
            ) intrinsic_perplexity=(
                min=minimum(intrinsic_ppl), max=maximum(intrinsic_ppl),
                mean=sum(intrinsic_ppl) / length(intrinsic_ppl)
            ) correct_token_prob=(
                min=minimum(correct_probs), max=maximum(correct_probs),
                mean=sum(correct_probs) / length(correct_probs)
            ) batch_loss=ce
        end

        total_correct += _count_correct(predictions, y)
        total_tokens += size(predictions, 1) * size(predictions, 2)
        num_batches += 1
    end

    # Promote to Float64 before dividing/exponentiating so the reported "perplexity" is exactly
    # exp("loss") to full Float64 precision, not exp(Float32 avg) rounded into a Float64 slot.
    avg_loss = num_batches > 0 ? Float64(total_loss) / num_batches : 0.0
    return Dict{String, Float64}(
        "loss" => avg_loss,
        "perplexity" => exp(avg_loss),
        "accuracy" => total_tokens > 0 ? total_correct / total_tokens : 0.0,
        "num_batches" => num_batches
    )
end

_softmax_vec(v) = (m=maximum(v); e=exp.(v .- m); e ./ sum(e))
_softmax_rows(m) = (e=exp.(m .- maximum(m; dims=2)); e ./ sum(e; dims=2))   # row-wise softmax (B,V)

"""
    _sample_next(probs, rng; temperature=1.0, top_k=nothing, top_p=nothing) -> Vector{Int}

Sample one next-token index (1-based) per batch row from post-softmax `probs` (B × V), with
temperature scaling + optional top-k and top-p (nucleus) filtering, via Gumbel-max categorical
sampling (≡ `jax.random.categorical`). Factored out of the generation step (upstream inlines it in
`_generation_step`, train_autoregressive.py:299) so the sampling logic is unit-testable on its own.

- temperature: logits = log(p+1e-10)/T (T<1 sharpens → greedier; T→0 ⇒ argmax).
- top_k: keep only the k highest-logit tokens (rest masked to −Inf).
- top_p: keep the smallest set whose cumulative softmax mass ≥ top_p (≥1 token always kept).
"""
function _sample_next(probs::AbstractMatrix, rng::AbstractRNG;
    temperature::Real=1.0, top_k::Union{Nothing, Integer}=nothing,
    top_p::Union{Nothing, Real}=nothing)
    B, V = size(probs)
    logits = log.(probs .+ 1.0f-10) ./ Float32(temperature)
    out = Vector{Int}(undef, B)
    for b in 1:B
        row = collect(@view logits[b, :])
        if top_k !== nothing && top_k < V
            thresh = partialsort(row, top_k; rev=true)        # k-th largest logit
            @inbounds for v in 1:V
                row[v] < thresh && (row[v] = -Inf32)
            end
        end
        if top_p !== nothing
            order = sortperm(row; rev=true)
            c = cumsum(_softmax_vec(row[order]))
            keep = falses(V)
            @inbounds for (rank, idx) in enumerate(order)
                keep[idx] = true
                c[rank] >= Float32(top_p) && break               # stop after crossing the threshold
            end
            @inbounds for v in 1:V
                keep[v] || (row[v] = -Inf32)
            end
        end
        # Gumbel-max categorical: argmax(logits + Gumbel) (= jax.random.categorical)
        best = 0;
        bestval = -Inf32
        @inbounds for v in 1:V
            (row[v] == -Inf32 || isnan(row[v])) && continue          # skip masked AND NaN logits
            g = -log(-log(rand(rng, Float32) + 1.0f-20) + 1.0f-20)
            val = row[v] + g
            val > bestval && (bestval=val; best=v)
        end
        # fallback: if every logit was masked/NaN (e.g. an untrained model emitting NaN), pick the
        # argmax of the row so we always return a valid 1..V token — never 0 (which is out of range
        # for the 1-based EmbeddingNode → BoundsError) and still deterministic for greedy decoding.
        best == 0 && (best = argmax(@view probs[b, :]))
        out[b] = best
    end
    return out
end

_one_hot(idx::AbstractMatrix{<:Real}, V::Integer) =
    Float32[
        idx[b, s] == v ? 1.0f0 : 0.0f0 for b in axes(idx, 1), s in axes(idx, 2), v in 1:V
    ]

"""
    generate_autoregressive(params, structure, prompt, max_new_tokens, rng;
                            temperature=1.0, top_k=nothing, top_p=nothing) -> tokens

Autoregressively generate `max_new_tokens` token indices (1-based) from `prompt` (a 1-D
`(seq,)` or 2-D `(batch, seq)` index array). Per step: format the context window for the input node
(1-D input-node shape ⇒ raw indices for an `EmbeddingNode`; 2-D ⇒ one-hot for a `Linear`), clamp the
input only, relax by inference, take the last-position output probabilities, sample via
`_sample_next`, then slide the context window. Returns prompt ++ generated (unbatched if the prompt
was 1-D). Port of `generate_autoregressive` (train_autoregressive.py:407); eager (upstream uses
`jax.lax.scan` + `jax.jit`), no fixed-size output buffer needed.
"""
function generate_autoregressive(params::GraphParams, structure::GraphStructure, prompt,
    max_new_tokens::Integer, rng::AbstractRNG;
    temperature::Real=1.0, top_k::Union{Nothing, Integer}=nothing,
    top_p::Union{Nothing, Real}=nothing)
    unbatch = ndims(prompt) == 1
    p = unbatch ? reshape(prompt, 1, :) : prompt
    B, prompt_len = size(p)
    input_node = get(structure.task_map, "x", nothing)
    output_node = get(structure.task_map, "y", nothing)
    (input_node === nothing || output_node === nothing) &&
        throw(ArgumentError("generate_autoregressive: task_map must contain 'x' and 'y'"))
    in_shape = structure.infos[input_node].shape
    seq_len = in_shape[1]
    vocab_size = structure.infos[output_node].shape[end]

    context = if prompt_len >= seq_len
        p[:, (end - seq_len + 1):end]
    else
        # left-pad with token 1 (1-based; upstream pads 0 in 0-based JAX)
        hcat(ones(Int, B, seq_len - prompt_len), p)
    end
    generated = Matrix{Int}(undef, B, max_new_tokens)
    for t in 1:max_new_tokens
        input_data = length(in_shape) == 1 ? context : _one_hot(context, vocab_size)
        clamps = Dict{String, Any}(input_node => input_data)
        st = initialize_graph_state(structure, B, rng; clamps=clamps, params=params)
        fs = run_inference(params, st, clamps, structure)
        # VocabProjectionNode now emits SOFTMAX PROBABILITIES (Softmax+CrossEntropy, per upstream
        # transformer_v2), so the last position is already a distribution — feed it straight to
        # _sample_next (no re-softmax, which would double-normalize).
        last_probs = fs.nodes[output_node].z_mu[:, end, :]                  # (B, V) — already probs
        nxt = _sample_next(
            last_probs, rng; temperature=temperature, top_k=top_k, top_p=top_p
        )
        generated[:, t] = nxt
        context = hcat(context[:, 2:end], reshape(nxt, B, 1))      # slide window + append
    end
    result = hcat(p, generated)
    return unbatch ? vec(result) : result
end
