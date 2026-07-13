# FabricPC.jl

A Julia port of [FabricPC](https://github.com/NACLab) (upstream Python/JAX by
Matthew Behrend, MIT) — a flexible, graph-based **predictive-coding (PC)**
training framework.

```@meta
CurrentModule = FabricPC
```

This is **Layer 2** of the NGC Julia stack:

| Layer | Package | Role |
|------:|---------|------|
| 0 | [NGCSimLib](https://cognitivesubstratesai.github.io/NGCSimLib/) | substrate — Component / Compartment / Context / Process |
| 1 | NGCLearn | biophysical component zoo |
| **2** | **FabricPC** (this package) | predictive-coding graph training framework + muPC |

FabricPC is a **standalone** package — it does *not* depend on NGCSimLib/NGCLearn.
It is a parallel graph abstraction (Nodes / Edges→slots) rather than NGCSimLib's
Component/Compartment substrate; composition between the two stacks is a deferred,
optional concern.

## What is predictive coding?

PC is a biologically-motivated learning framework that performs **bilevel
optimization**: an inner loop infers latent activations by minimizing local
prediction errors, and an outer loop updates weights via **local (Hebbian-like)
rules**. The outer loop does *not* backpropagate through the inner loop —
gradients are local and analytic.

FabricPC builds PC networks from:
- **Nodes** — each owns a latent `z_latent`, predicts `z_mu` from incoming edges,
  derives `error = z_latent − z_mu`, contributes a local energy.
- **Edges** — connect a source node to a target node's named **slot**.
- **muPC** — μP-style per-edge scaling (from graph topology) that keeps
  activations, errors, and gradients O(1) at arbitrary width and depth.

## Status — substrate essentially complete

Design + phase history are in [Architecture](@ref) and
[decisions.md](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/docs/decisions.md)
(the load-bearing porting decisions, read before extending). Shipped:

- **Core PC engine** — closed-form explicit-gradient nodes ([`Linear`](@ref),
  [`IdentityNode`](@ref), [`SkipConnection`](@ref), [`LinearResidual`](@ref)) need
  **no reverse-mode autodiff**: PC's local learning rule is closed-form, so a
  minimal trainable graph is autodiff-free.
- **muPC scaling** ([`MuPCConfig`](@ref)) — μP-style per-edge scaling that keeps
  activations/errors/gradients O(1) across width and depth (residual-depth-aware).
- **Phase-D autodiff seam** — a node that implements only `compute_mu` gets its
  local PC gradients for free via Zygote or Enzyme (exactly ONE backend per
  session). This is what makes transformer/attention/Storkey-Hopfield nodes
  expressible without hand-derived gradients; it is still pure PC, never backprop
  through the network.
- **Node set**: Linear/Identity/SkipConnection/LinearResidual (closed-form),
  [`TransformerBlock`](@ref) (monolithic PC-transformer with RoPE + causal
  masking), the decomposed transformer_v2 node family
  ([`MhaResidualNode`](@ref)/[`LnMlp1Node`](@ref)/[`Mlp2ResidualNode`](@ref)/
  [`EmbeddingNode`](@ref)/[`VocabProjectionNode`](@ref) — PC at every
  sub-component), [`StorkeyHopfield`](@ref) (composite PC + associative-memory
  energy).
- **Training**: SGD, [`AdamW`](@ref), and [`InferenceSGDNormClip`](@ref) (the
  muPC training recipe needs an adaptive optimizer), causal autoregressive
  next-token training ([`train_autoregressive`](@ref)), held-out evaluation
  ([`evaluate_autoregressive`](@ref)), natural-gradient (Fisher) preconditioners.
- **Reactant/XLA JIT** — the inference path, and `TransformerBlock`'s forward
  pass, compile to XLA via Reactant, validated bit-identical to eager.
- **Conformance-tested against upstream JAX** — a fixture-based harness
  (`docs/AUDIT_REGISTER.md` section 6) verifies primitives (Tier A) and
  node-level forward+gradients (Tier B) numerically match the pinned upstream
  checkout, not just internal self-consistency.
- Not ported (deliberately): a backprop training baseline (`train_backprop.py` —
  PC needs no backprop, so this stays the never-ported anti-thesis reference),
  multi-GPU/pmap, spatial convolution nodes, dashboards/tuning infra. See
  `decisions.md` for the full, current list of what's deferred and why.

## Where to go next

- [Installation](@ref) / [Quickstart](@ref) — get a PC graph training in a few
  lines.
- [Architecture](@ref) — the Node/Edge/slot graph model, the inference/learning
  loop, and how muPC scaling fits in.
- [JIT with Reactant](@ref) — compiling the inference loop to XLA.
- [API Reference](@ref) — full reference.

## License

MIT. A Julia port of `FabricPC` (Matthew Behrend), also MIT. See
[`LICENSE`](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/LICENSE).
