# Assembled transformer language-model graph. Adapts the upstream
# `examples/transformer_demo.py:create_transformer_lm` topology to our node set.
#
# Topology (per the demo, divergences applied for our architecture):
#
#   input  = IdentityNode((S,))                      token indices (task x)
#   embed  = EmbeddingNode((S,E); vocab_size=V)      id → embedding lookup
#   per block i:
#     block_i = TransformerBlock((S,E); num_heads, causal=true)
#     skip_i  = SkipConnection((S,E))                residual sum
#       prev → block_i:in ; prev → skip_i:in ; block_i → skip_i:in ; prev := skip_i
#   output = VocabProjectionNode((S,V))              (B,S,E) → (B,S,V) (task y)
#
# DIVERGENCES from upstream (grounded in our node source):
#   1. INLINE CAUSAL MASK. Our `TransformerBlock` masks future keys inline via its
#      `causal::Bool` flag (`_tb_causal_mask`, transformer.jl) and exposes a SINGLE
#      "in" slot — there is NO "mask" slot. So we set `causal=true` and create NO
#      mask node and NO `.slot("mask")` edge. (Upstream clamps a (1,S,S) mask node.)
#   2. OUTPUT = `VocabProjectionNode`, NOT `Linear`. Our `Linear.forward` is 2D-only
#      and cannot do the per-position (B,S,E)→(B,S,V) projection; `VocabProjectionNode`
#      (transformer_decomposed.jl) is the seq-aware vocab projection (single "in"
#      slot, `_tb_dense` over the feature axis → (B,S,V)). It defaults to Identity
#      activation + GaussianEnergy (rank-3 SoftmaxActivation would normalize over the
#      sequence axis, not vocab — see the node docstring), training next-token logits
#      toward one-hot targets.

"""
    transformer_lm(; seq_len, vocab_size, embed_dim = 32, num_heads = 4,
                   num_blocks = 1, ff_dim = nothing, use_rope = true,
                   infer_steps = nothing, eta_infer = 0.1,
                   scaling = nothing) -> GraphStructure

Assemble a predictive-coding transformer language model and return its
`GraphStructure` (build params with `initialize_params(structure, rng)`).

`embed_dim` must be divisible by `num_heads`. `infer_steps` defaults to the
upstream heuristic `3 * (2*num_blocks + 2)`. Task map is `(x = input, y = output)`:
clamp a `(B, seq_len)` integer token batch on `x`; the output node's `z_mu` is the
`(B, seq_len, vocab_size)` next-token logits.
"""
function transformer_lm(;
    seq_len::Integer,
    vocab_size::Integer,
    embed_dim::Integer=32,
    num_heads::Integer=4,
    num_blocks::Integer=1,
    ff_dim::Union{Integer, Nothing}=nothing,
    use_rope::Bool=true,
    infer_steps::Union{Integer, Nothing}=nothing,
    eta_infer::Real=0.1,
    scaling::Union{Nothing, MuPCConfig}=nothing
)
    embed_dim % num_heads == 0 || throw(
        ArgumentError("embed_dim ($embed_dim) must be divisible by num_heads ($num_heads)")
    )
    steps = infer_steps === nothing ? 3 * (2 * num_blocks + 2) : Int(infer_steps)

    input = IdentityNode((seq_len,), "input")
    embed = EmbeddingNode((seq_len, embed_dim), "embed"; vocab_size=Int(vocab_size),
        weight_init=NormalInitializer(std=1.0))                # upstream transformer_v2: embed std=1.0

    nodes = AbstractNode[input, embed]
    edges = Edge[Edge(input, slot(embed, "in"))]

    prev = embed
    for i in 0:(num_blocks - 1)
        block = TransformerBlock(
            (seq_len, embed_dim), "transformer_$i";
            num_heads=Int(num_heads), ff_dim=ff_dim, use_rope=use_rope, causal=true
        )
        skip = SkipConnection((seq_len, embed_dim), "skip_$i")
        push!(nodes, block)
        push!(nodes, skip)
        push!(edges, Edge(prev, slot(block, "in")))    # prev → block:in
        push!(edges, Edge(prev, slot(skip, "in")))     # prev → skip:in (residual)
        push!(edges, Edge(block, slot(skip, "in")))    # block → skip:in (sum)
        prev = skip
    end

    output = VocabProjectionNode((seq_len, vocab_size), "output";
        weight_init=NormalInitializer(std=sqrt(1.0 / embed_dim)))  # upstream transformer_v2: logits std=√(1/E)
    push!(nodes, output)
    push!(edges, Edge(prev, slot(output, "in")))       # prev → output:in

    inference = InferenceSGD(; eta_infer=eta_infer, infer_steps=steps)
    return graph(
        nodes, edges, TaskMap(; x=input, y=output), inference; scaling=scaling
    )
end
