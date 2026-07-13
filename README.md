# FabricPC.jl

A Julia port of [FabricPC](https://github.com/NACLab) (upstream Python/JAX by
Matthew Behrend, MIT) — a flexible, graph-based **predictive-coding (PC)**
training framework.

This is **Layer 2** of the NGC Julia stack:

| Layer | Package | Role |
|------:|---------|------|
| 0 | [NGCSimLib](https://github.com/CognitiveSubstratesAI/NGCSimLib) | substrate — Component / Compartment / Context / Process |
| 1 | [NGCLearn](https://github.com/CognitiveSubstratesAI/NGCLearn) | biophysical component zoo |
| **2** | **FabricPC** (this repo) | predictive-coding graph training framework + muPC |

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

Design + phase history are in [docs/DESIGN.md](docs/DESIGN.md) and
[docs/decisions.md](docs/decisions.md) (the load-bearing porting decisions, read
before extending). Shipped:

- **Core PC engine** — closed-form explicit-gradient nodes (Linear, Identity,
  SkipConnection, LinearResidual) need **no reverse-mode autodiff**: PC's local
  learning rule is closed-form, so a minimal trainable graph is autodiff-free.
- **muPC scaling** (`core/mupc.jl`) — μP-style per-edge scaling that keeps
  activations/errors/gradients O(1) across width and depth (residual-depth-aware).
- **Phase-D autodiff seam** (`nodes/autodiff.jl`) — a node that implements only
  `compute_mu` gets its local PC gradients for free via Zygote or Enzyme
  (exactly ONE backend per session — see decisions.md #19). This is what makes
  transformer/attention/Storkey-Hopfield nodes expressible without hand-derived
  gradients; it is still pure PC, never backprop through the network.
- **Node set**: Linear/Identity/SkipConnection/LinearResidual (closed-form),
  TransformerBlock (monolithic PC-transformer with RoPE + causal masking),
  the decomposed transformer_v2 node family (MhaResidual/LnMlp1/Mlp2Residual/
  Embedding/VocabProjection — PC at every sub-component), StorkeyHopfield
  (composite PC + associative-memory energy).
- **Training**: SGD and AdamW (the muPC training recipe needs an adaptive
  optimizer), causal autoregressive next-token training (`train_autoregressive`),
  natural-gradient (Fisher) preconditioners.
- **Reactant/XLA JIT** — the inference path (and the TransformerBlock forward +
  local gradient) compile to XLA via Reactant, validated bit-identical to eager.
- Not ported (deliberately): a backprop training baseline (`train_backprop.py` —
  PC needs no backprop, so this stays the never-ported anti-thesis reference),
  multi-GPU/pmap, spatial convolution nodes, dashboards/tuning infra. See
  decisions.md for the full, current list of what's deferred and why.

## Development

Julia **1.12+**.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Before committing any `.jl` change, run the Blue formatter (CI enforces it,
pinned to JuliaFormatter 2.5.2):

```bash
julia -e 'using JuliaFormatter; format(".")'
```

## License

MIT. A Julia port of `FabricPC` (Matthew Behrend), also MIT. See `LICENSE`.
