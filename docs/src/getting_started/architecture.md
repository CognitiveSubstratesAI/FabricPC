# Architecture

## Core abstractions

- **Node** — owns an output latent `z_latent`, computes a top-down prediction
  `z_mu` from incoming edges, derives `error = z_latent − z_mu`, contributes a
  local energy. Single output; named input **slots** (e.g. `"in"`, `"skip"`).
- **Edge** — connects a source node to a target node's slot, keyed
  `"src->tgt:slot"`.
- **`GraphStructure`** (static topology) / **`GraphState`** (dynamic latents) /
  **`GraphParams`** (weights) — the three pytree-equivalents everything else is
  built from.
- **muPC** — μP (maximal-update parameterization) generalized to predictive
  coding over arbitrary DAGs: per-edge forward/gradient scaling factors derived
  from graph topology (slot in-degree, residual depth, fan-in, activation gain)
  keep activations/errors/gradients O(1) at any width/depth. This is the
  research contribution the port is built around (Innocenti et al.,
  arXiv:2505.13124; Yang et al. Depth-μP).

## The key architectural property: no backprop, and v0 needs no autodiff at all

Predictive coding performs **bilevel optimization**:

- **Inference** (inner loop): `z ← z·(1 − η·decay) − η·latent_grad`, where
  `latent_grad` is assembled **locally** — each node pushes its own
  prediction-error gradient back to its in-neighbors only.
- **Learning** (outer loop): local Hebbian-like weight gradients computed from
  the converged (relaxed) state.

Neither loop ever backpropagates a gradient through the whole network. The
closed-form nodes ([`Linear`](@ref), [`IdentityNode`](@ref),
[`SkipConnection`](@ref), [`LinearResidual`](@ref)) compute both gradients
analytically — no reverse-mode AD anywhere in that path:

```
self_grad       = precision · (z_latent − z_mu)
gain_mod_error  = error · f'(pre_activation)
input_grad[e]   = −(gain_mod_error · Wₑᵀ)
dW[e]           = −(inputₑᵀ · gain_mod_error)
db              = −Σ_batch gain_mod_error
```

## The Phase-D autodiff seam

Nodes whose local gradient isn't hand-derivable in closed form (attention,
Storkey-Hopfield's attractor energy) implement **only** `compute_mu` — the
forward prediction — and get `forward_and_latent_grads`/
`forward_and_weight_grads` for free via reverse-mode AD over that one function
(Zygote or Enzyme, [`docs/decisions.md`](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/docs/decisions.md)
§19/§23). This is still **local** PC — the seam differentiates one node's own
`compute_mu`, never the whole network — not a backdoor into backprop.

## Execution layers

Four distinct execution paths run the same PC computation
([`docs/decisions.md`](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/docs/decisions.md)
§22 has the full writeup):

- **Layer 0 — eager, Dict-based.** The correctness reference; what every other
  layer (and upstream conformance testing) validates against.
- **Layer 1 — flat/positional.** Position-indexed `Vector`s, no `Dict`s in the
  hot loop — the intermediate form Reactant traces. Validated bit-identical to
  Layer 0.
- **Layer 2 — Reactant/XLA compiled.** Layer 1 traced through
  `Reactant.@compile`. Forward-inference JIT works today; compiled *gradients*
  are blocked on an architectural conflict (Enzyme eager vs. Reactant-compiled
  Enzyme in one process) — see [JIT with Reactant](@ref).
- **Layer 3 — muPC.** Not a separate execution path — a per-edge scaling
  applied at any of the layers above.

**Rule**: each layer is validated against the one directly below it, never
against upstream directly except Layer 0. A failure in Layer 1 means "the flat
lowering broke," not "the port diverged from upstream."

## Conformance vs. upstream

A fixture-based harness
([`docs/AUDIT_REGISTER.md`](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/docs/AUDIT_REGISTER.md)
section 6) verifies this port numerically against the pinned upstream JAX
checkout at two levels so far: primitives (activations, energies, RoPE,
LayerNorm — "Tier A") and node-level forward + gradients, including the
muPC-scaled case ("Tier B"). Loop-level and end-to-end tiers are planned.
