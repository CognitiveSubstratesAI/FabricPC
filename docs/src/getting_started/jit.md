# JIT with Reactant

FabricPC's inference loop (the inner relaxation loop — not weight training) can
be compiled to XLA via [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl),
bit-exact to the eager path and substantially faster. This is opt-in: Reactant
is a weak dependency, loaded via the `FabricPCReactantExt` package extension.

```julia
using FabricPC
using FabricPC: compile_inference
using Reactant   # triggers FabricPCReactantExt

structure = graph([xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)],
    TaskMap(; x=xn, y=yn), InferenceSGD(; eta_infer=0.1, infer_steps=20))
params = initialize_params(structure, rng)
clamps = Dict{String, Any}("x" => x, "y" => y)

ci = compile_inference(structure, params, clamps; batch=size(x, 1))  # traces + compiles once
z_latents = ci(params, init_state)                                   # run the compiled thunk
```

The returned callable maps `(params, init_state) -> Vector` of converged
per-node `z_latent` arrays (node order = `structure.node_names`). It's
specialized to the param shapes, `batch`, and which nodes are clamped —
recompile if any of those change. See `examples/jit_inference.jl` for a
runnable benchmark against the eager path.

## What's compiled today, and what isn't

- **Forward inference** (the relaxation loop, no weight updates) — works, for
  the node types [`Linear`](@ref), [`IdentityNode`](@ref),
  [`SkipConnection`](@ref), [`LinearResidual`](@ref), and — as of this
  writing — [`TransformerBlock`](@ref)'s forward pass (as the graph's unclamped
  terminal node; using `TransformerBlock` as an interior or clamped node isn't
  compiled yet — see below).
- **Weight-gradient compilation** (compiling the full training step, not just
  inference) — not implemented. `compute_local_weight_gradients` remains
  eager/Dict-based.
- **`TransformerBlock`'s backward pass** in the compiled lane — not
  implemented. The forward kernel (`_tb_block_flat`) is wired in and validated
  bit-identical to eager `compute_mu`, but there's no compiled gradient for it
  yet.
- **Decomposed transformer nodes** (`MhaResidualNode`/`LnMlp1Node`/
  `Mlp2ResidualNode`/`EmbeddingNode`/`VocabProjectionNode`) and
  [`StorkeyHopfield`](@ref) — no compiled-lane support at all yet.

## Why gradients are the hard part

Reactant uses Enzyme-MLIR internally for autodiff under `@compile` — a
different thing from loading the `Enzyme` package eagerly for the Phase-D
seam, despite the shared name (see
[`docs/decisions.md`](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/docs/decisions.md)
§23 for the full backend map). There's a real, reproduced conflict: loading
eager `Enzyme` in the same process as a `Reactant`+`Enzyme` `@compile` poisons
the compile. Working around this — and extending compiled gradients past the
current closed-form node types — is open, tracked as J-01/J-02 in
[`docs/AUDIT_REGISTER.md`](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/docs/AUDIT_REGISTER.md)
section 5.
