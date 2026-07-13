#!/usr/bin/env julia
# Character-level Tiny Shakespeare LM — real-text exhibit for the assembled `transformer_lm`
# stack (src/models/transformer_lm.jl), trained end-to-end by LOCAL predictive coding.
#
# C-03 (docs/AUDIT_REGISTER.md): port of fabricpc/utils/data/dataloader.py's
# `_TokenSequenceLoader`/`CharDataLoader` — yields RAW integer token ids (B, S) for both
# "x" and "y" (train_step_autoregressive one-hot-expands "y" internally, F-05), matching the
# loader contract's rationale: keep host->device transfer int-sized, not vocab_size-larger.
#
# Corpus: not TFDS (no Julia binding) -- the same raw source TFDS's `tiny_shakespeare`
# builder downloads (github.com/karpathy/char-rnn), split with TFDS's own 90/5/5 formula
# (verified against tensorflow_datasets/datasets/tiny_shakespeare/tiny_shakespeare_dataset_builder.py,
# `_split_generators`: `i = int(len(text)*0.9); train, text = text[:i], text[i:]; i =
# int(len(text)*0.5); validation, test = text[:i], text[i:]`) so vocab/splits match what
# upstream's own loader would see. Dependency-light, following `mnist_pc.jl`'s precedent:
# a plain `Downloads` fetch + cache, no library dataloader module (this project keeps data
# loading example-local, not exported FabricPC API).
#
# DIVERGENCE (intentional): char->id is 1-based here (`i in 1:length(chars)`), not upstream's
# 0-based `enumerate`, because every OTHER token-id consumer in this Julia port (EmbeddingNode,
# _one_hot, generate_autoregressive) is already 1-based -- keeping CharDataLoader 0-based would
# be an internal footgun, not fidelity (nothing downstream reads these ids as Python would).
#
# Run (needs Zygote -- TransformerBlock's autodiff seam; Enzyme crashes on the multi-head block):
#   julia --project=. examples/char_lm_pc.jl
# Tunables via env: FPC_SEQLEN, FPC_BATCH, FPC_EMBED, FPC_HEADS, FPC_BLOCKS, FPC_EPOCHS,
# FPC_STEPS, FPC_ETA, FPC_LR, FPC_MAXSAMPLES (cap train sequences, for a fast smoke run).

using FabricPC
using Zygote   # AD seam for TransformerBlock/VocabProjection (Enzyme crashes on the multi-head block)
using Random
using Printf
using Downloads

const SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
const CACHE = joinpath(homedir(), ".cache", "fabricpc_shakespeare")

# ── data: download + TFDS-matching split + char vocab ──────────────────────────────────────

function _fetch_shakespeare()
    mkpath(CACHE)
    path = joinpath(CACHE, "input.txt")
    if !isfile(path)
        @info "downloading tiny shakespeare corpus"
        Downloads.download(SHAKESPEARE_URL, path)
    end
    return read(path, String)
end

"""
    _tiny_shakespeare_splits(text) -> (train, validation, test)

TFDS `tiny_shakespeare`'s own 90/5/5 split (`tiny_shakespeare_dataset_builder.py:52-56`):
`i = int(len(text)*0.9)` (`int()` truncates -- `floor` for a positive length), first `i`
chars -> train, the rest split in half -> validation/test.
"""
function _tiny_shakespeare_splits(text::AbstractString)
    chars = collect(text)
    n = length(chars)
    i1 = floor(Int, n * 0.9)
    train_chars, rest = chars[1:i1], chars[(i1 + 1):end]
    i2 = floor(Int, length(rest) * 0.5)
    validation_chars, test_chars = rest[1:i2], rest[(i2 + 1):end]
    return String(train_chars), String(validation_chars), String(test_chars)
end

"""
    _char_vocab(train_text) -> (chars, char_to_idx, idx_to_char)

Port of `CharDataLoader`'s vocab construction (dataloader.py:367-373): sorted unique chars of
the TRAIN split only (validation/test reuse this vocab, matching upstream's class-level cache
across split instances -- expressed here as a value threaded through the constructor, not
mutable global state). 1-based ids -- see file header DIVERGENCE note.
"""
function _char_vocab(train_text::AbstractString)
    chars = sort(unique(collect(train_text)))
    char_to_idx = Dict(ch => i for (i, ch) in enumerate(chars))
    idx_to_char = Dict(i => ch for (i, ch) in enumerate(chars))
    return chars, char_to_idx, idx_to_char
end

"""
    CharDataLoader(split_text, chars, char_to_idx, idx_to_char, seq_len, batch_size;
                   shuffle=true, seed=nothing, max_samples=nothing)

Sliding-window next-char batching, re-iterable across epochs. Port of `_TokenSequenceLoader`
+ `CharDataLoader` (dataloader.py:272-390): for each window start `p`, `x = data[p:p+seq_len-1]`,
`y = data[p+1:p+seq_len]` (shifted by one) -- RAW ids, Float32-encoded (matching this codebase's
EmbeddingNode/`_one_hot` contract), one-hot expansion happens inside `train_step_autoregressive`
(F-05), not here. `for batch in loader` reshuffles the window-start order fresh each call
(`Base.iterate(loader)`, no state) via `seed + epoch`, mirroring `self._epoch`/
`np.random.default_rng(seed+epoch)`; the LAST incomplete batch is dropped, matching upstream.
"""
mutable struct CharDataLoader
    data::Vector{Int}
    seq_len::Int
    batch_size::Int
    shuffle::Bool
    seed::Union{Nothing, Int}
    epoch::Int
    num_sequences::Int
    num_batches::Int
    vocab_size::Int
    chars::Vector{Char}
    char_to_idx::Dict{Char, Int}
    idx_to_char::Dict{Int, Char}
end

function CharDataLoader(split_text::AbstractString, chars::Vector{Char},
    char_to_idx::Dict{Char, Int}, idx_to_char::Dict{Int, Char},
    seq_len::Integer, batch_size::Integer;
    shuffle::Bool=true, seed::Union{Nothing, Integer}=nothing,
    max_samples::Union{Nothing, Integer}=nothing)
    data = [char_to_idx[ch] for ch in collect(split_text)]
    num_sequences = length(data) - seq_len
    max_samples !== nothing && (num_sequences = min(num_sequences, max_samples))
    num_batches = num_sequences ÷ batch_size
    return CharDataLoader(data, Int(seq_len), Int(batch_size), shuffle,
        seed === nothing ? nothing : Int(seed), 0, num_sequences, num_batches,
        length(chars), chars, char_to_idx, idx_to_char)
end

Base.length(l::CharDataLoader) = l.num_batches
Base.eltype(::Type{CharDataLoader}) = Dict{String, Any}

function Base.iterate(l::CharDataLoader)
    idx = collect(1:(l.num_sequences))
    if l.shuffle
        rng = l.seed === nothing ? Random.default_rng() : MersenneTwister(l.seed + l.epoch)
        Random.shuffle!(rng, idx)
    end
    l.epoch += 1
    return iterate(l, (idx, 1))
end

function Base.iterate(l::CharDataLoader, (idx, pos))
    stop = pos + l.batch_size - 1
    stop > length(idx) && return nothing               # drop incomplete last batch
    starts = @view idx[pos:stop]
    S = l.seq_len
    x = Float32[l.data[p + s - 1] for p in starts, s in 1:S]
    y = Float32[l.data[p + s] for p in starts, s in 1:S]
    return Dict{String, Any}("x" => x, "y" => y), (idx, stop + 1)
end

"""Convert an array/vector of 1-based char indices back to a `String`."""
decode(l::CharDataLoader, indices) = String([l.idx_to_char[Int(i)] for i in indices])

# ── build + train + generate ────────────────────────────────────────────────────────────────

function main()
    seq_len = parse(Int, get(ENV, "FPC_SEQLEN", "64"))
    batch_size = parse(Int, get(ENV, "FPC_BATCH", "32"))
    embed_dim = parse(Int, get(ENV, "FPC_EMBED", "64"))
    num_heads = parse(Int, get(ENV, "FPC_HEADS", "4"))
    num_blocks = parse(Int, get(ENV, "FPC_BLOCKS", "2"))
    epochs = parse(Int, get(ENV, "FPC_EPOCHS", "5"))
    infer_steps = parse(Int, get(ENV, "FPC_STEPS", "12"))
    eta_infer = parse(Float64, get(ENV, "FPC_ETA", "0.1"))
    lr = parse(Float32, get(ENV, "FPC_LR", "0.001"))
    max_samples = let v = get(ENV, "FPC_MAXSAMPLES", "")
        isempty(v) ? nothing : parse(Int, v)
    end

    text = _fetch_shakespeare()
    train_text, validation_text, test_text = _tiny_shakespeare_splits(text)
    chars, char_to_idx, idx_to_char = _char_vocab(train_text)
    vocab_size = length(chars)
    @info "tiny shakespeare loaded" total_chars = length(text) train = length(train_text) validation = length(
        validation_text
    ) test = length(test_text) vocab_size

    train_loader = CharDataLoader(
        train_text, chars, char_to_idx, idx_to_char, seq_len, batch_size;
        seed=0, max_samples=max_samples
    )
    @info "train loader" num_sequences = train_loader.num_sequences num_batches = length(
        train_loader
    )

    structure = transformer_lm(;
        seq_len=seq_len, vocab_size=vocab_size, embed_dim=embed_dim,
        num_heads=num_heads, num_blocks=num_blocks, infer_steps=infer_steps,
        eta_infer=eta_infer
    )
    params = initialize_params(structure, MersenneTwister(1))
    opt = AdamW(params; lr=lr, weight_decay=0.1f0)

    params, _, _ = train_autoregressive(
        params, structure, train_loader, opt, epochs, MersenneTwister(2);
        epoch_callback=(ep, p, s) -> nothing
    )

    # quick generation demo, seeded from a snippet of the train text
    prompt = [char_to_idx[ch] for ch in first(collect(train_text), seq_len)]
    gen = generate_autoregressive(
        params, structure, prompt, 200, MersenneTwister(7); temperature=0.8
    )
    println("\n--- sample generation ---")
    println(decode(train_loader, gen))
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
