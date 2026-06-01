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

## Status — Phase A (scaffold, v0.1.0)

Design + scope are in [docs/DESIGN.md](docs/DESIGN.md). The phased plan:

- **B** — core types + AD-free linear PC graph + train loop (a 3-layer linear PC
  network learns a small task). *In progress.*
- **C** — muPC scaling (width/depth stability).
- **D** — Enzyme autodiff fallback + non-linear activations.
- **E** — exhibits (MNIST-style PC classifier), then deferred reach.

A defining property of the port: a minimal **trainable** PC graph needs **no
reverse-mode autodiff** — PC's local learning rule is closed-form (the
explicit-gradient Gaussian path). Enzyme is deferred to the non-linear / generic
node path (Phase D).

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
