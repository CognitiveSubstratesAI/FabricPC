# FabricPC.jl — design & port plan

A faithful Julia port of [FabricPC](https://github.com/CognitiveSubstratesAI/FabricPC)
(upstream Python/JAX by Matthew Behrend, MIT) — a flexible, graph-based
**predictive-coding** training framework. This is **Layer 2** of the NGC Julia
stack:

| Layer | Package | Role |
|------:|---------|------|
| 0 | NGCSimLib.jl | substrate (Component/Compartment/Context/Process) |
| 1 | NGCLearn.jl | biophysical component zoo |
| **2** | **FabricPC.jl** (this) | predictive-coding graph training framework + muPC |

> **Relationship to Layers 0/1 (important):** FabricPC is a *parallel* graph
> abstraction, NOT built on NGCSimLib's Component/Compartment. NGCSimLib wires
> Components via Compartments; FabricPC wires **Nodes** via **Edges→slots** with
> its own `GraphStructure`/`GraphState`/`GraphParams` pytrees. Per the layered
> plan, composition between the two is deferred (Layer 3, optional). FabricPC.jl
> is therefore a standalone package that does *not* depend on NGCSimLib/NGCLearn.

## What FabricPC is

Predictive coding performs **bilevel optimization**: an inner loop infers latent
activations by minimizing local prediction errors, an outer loop updates weights
via **local (Hebbian-like) rules**. Crucially, the outer loop does **not**
backprop through the inner loop — weight gradients are local and analytic.

Core abstractions:
- **Node** — owns an output latent `z_latent`, computes a top-down prediction
  `z_mu` from incoming edges, derives `error = z_latent − z_mu`, contributes a
  local energy. Single output; named input **slots**.
- **Edge** — connects a source node to a target node's slot (`"src->tgt:slot"`).
- **GraphStructure** (static) / **GraphState** (dynamic latents) / **GraphParams**
  (weights) — the three pytrees.
- **muPC** — μP (maximal-update parameterization) generalized to PC over
  arbitrary DAGs: per-edge forward/grad scaling factors derived from topology
  (slot in-degree K, residual depth L, fan-in, activation gain) keep
  activations/errors/grads O(1) at any width/depth. **This is the novel research
  contribution** (Innocenti et al., arXiv:2505.13124; Yang et al. Depth-μP).

## The key architectural finding: v0 needs NO autodiff

PC's local learning means a **minimal working, trainable PC graph requires no
reverse-mode AD**:
- **Inference** (inner loop): `z ← z·(1−η·decay) − η·latent_grad`, where
  `latent_grad` is assembled locally by pushing each node's prediction-error
  gradient back to its in-neighbors.
- **Learning** (outer loop): local Hebbian weight gradients from the converged
  state.

Upstream computes these two gradients by `jax.value_and_grad` of node-local
energy *by default*, **but** also ships closed-form explicit gradients
(`LinearExplicitGrad`, GaussianEnergy) that are exact and AD-free:
- `self_grad   = precision·(z_latent − z_mu)`
- `gain_mod_error = error · f'(pre_activation)`
- `input_grad[e]  = −(gain_mod_error · Wₑᵀ)`
- `dW[e] = −(inputₑᵀ · gain_mod_error)`,  `db = −Σ_batch gain_mod_error`

**v0 ports the explicit-grad Gaussian path.** Enzyme (the Julia analog of
`jax.value_and_grad`) is deferred to when non-linear / transformer nodes need the
generic autodiff fallback. This sidesteps the single biggest porting risk
(Enzyme over closures returning struct-of-arrays aux) for the entire v0.

## Data model (already understood, from upstream core/types.py)

- `NodeInfo{name, shape, node_class, node_config, activation, energy, slots,
  in/out_degree, in/out_edges, scaling_config}` — static per-node.
- `SlotInfo{name, is_multi_input, is_variance_scalable, is_skip_connection,
  in_neighbors}`.
- `EdgeInfo{key="src->tgt:slot", source, target, slot}`.
- `NodeState{z_latent, z_mu, error, energy, pre_activation, latent_grad}` —
  dynamic; `GraphState{nodes, batch_size}`.
- `NodeParams{weights::Dict, biases::Dict}`; `GraphParams{nodes::Dict}`.

Julia rendering: immutable structs updated via Accessors.jl `@set`; an **ordered**
node container (Vector + name→index, or OrderedDict) since the inference
gradient-accumulation is **node-order-dependent**. Keep batch as a chosen layout
convention applied consistently (decide: batch-first to match upstream, or
batch-last to match NGCLearn — **decision: batch-first**, matching upstream
exactly to minimize port divergence; energies sum over all non-batch dims).

## Phased plan (each phase gated on tests; faithful-then-verified)

**Phase A — scout + scaffold (this).** Design doc, repo scaffold (Project.toml,
CI with the format-pin fix from day one, docs, MIT LICENSE w/ upstream
attribution).

**Phase B — core types + AD-free linear PC (v0 CORE). ✅ DONE.**
- `core/types.jl`: NodeInfo/SlotInfo/EdgeInfo/NodeState/GraphState/GraphParams/
  GraphStructure as immutable structs. ✅
- `core/topology.jl`: Edge, SlotRef, SlotSpec. ✅ (GraphNamespace deferred — flat names.)
- `nodes/`: AbstractNode contract, `Linear` (explicit-grad fused; `flatten_input`
  inert at rank-1), `IdentityNode`, `SkipConnection`, `LinearResidual` — all with
  **explicit Gaussian gradients**. ✅ (Skip slot non-variance-scalable +
  skip-connection flags carried for Phase C; LinearResidual splits `:in`/`:skip`
  edges by key suffix.)
- `core/energy.jl`: GaussianEnergy + `grad_latent`. ✅ (other energies → Phase D.)
- `core/activations.jl`: IdentityActivation. ✅ (non-linear → Phase D.)
- `core/inference.jl`: gather_inputs, `forward_value_and_grad` accumulation,
  `InferenceSGD`, `run_inference` (eager loop; Reactant-friendly later). ✅
- `core/learning.jl`: `compute_local_weight_gradients`. ✅
- `core/scaling.jl`: no-op stubs (`scaling_config === nothing`); real muPC → Phase C. ✅
- `assembly`: `graph()`, BFS topological sort, slot resolution; `initialize_params`,
  `FeedforwardStateInit`, initializers (Normal/Zeros). ✅ (Xavier/Kaiming/MuPC deferred.)
- `training`: `get_graph_param_gradient`, `train_step`, `sgd_update` (manual SGD,
  faithful to optax.sgd). ✅ (Optimisers.jl Adam/momentum + `train_pcn`/`eval_step`
  + multi-device pmap path deferred.)
- **Acceptance:** ✅ a 3-layer linear PC graph (4→6→3) learns a deterministic
  linear map to machine precision — energy 8.76 → 4.7e-14, prediction
  `max|pred−y| = 4.8e-7` (Float32 floor), weights healthy (no collapse). Plus an
  exact hand-computed gradient check + finite-difference cross-check of all four
  explicit gradients (self / input / weight / bias). 21/21 tests.

**Phase B follow-up:** ✅ `SkipConnection` + `LinearResidual` nodes done
(hand-check + finite-diff + a residual-block learning test; 43/43 total).
`Optimisers.jl` Adam still deferred until a non-trivial exhibit needs it.

**Phase C — muPC (the novel layer). ✅ DONE.**
- `core/mupc.jl` (`MuPCConfig`, `MuPCScalingFactors`, residual-depth L,
  `compute_mupc_scalings` — pure scalar/topology math, no AD). ✅
- `core/scaling.jl` real apply-side methods for `MuPCScalingFactors` — missing-key
  = no-op (dtype-preserving), pre-scale inputs / post-scale top-down + weight
  grads. ✅ Activation `variance_gain`/`jacobian_gain` (default 1; Identity). ✅
  Per-node `get_weight_fan_in` (Identity/Skip = 1). ✅
- `graph(...; scaling = MuPCConfig())` computes + attaches per-node factors. ✅
- **Acceptance:** ✅ width scan reproduces the muPC variance-control result —
  hidden-activation RMS stays ≈1.0 across width 64→4096 (1.045 / 0.965 / 0.995 /
  0.982) with muPC ON, vs the √(fan_in) blow-up OFF (8.4 / 15.4 / 31.8 / 62.9,
  tracking √W exactly). Plus exact forward-scale formula checks, residual-depth
  L=2 detection, unscaled skip edges, and a muPC-on convergence smoke test.
  61/61 tests.

**Phase D — non-linear activations + Enzyme autodiff fallback.** Split into D1/D2.

*Phase D1 — element-wise non-linear activation zoo. ✅ DONE (NO new dep).*
- Key realization: the explicit Linear/LinearResidual path already consumes
  `derivative(activation, pre)` via `gain_mod_error = error·f'(pre)`, which is
  exact for ANY element-wise activation under Gaussian energy. So Sigmoid / Tanh
  / ReLU / LeakyReLU / GELU / HardTanh train through the EXISTING explicit path —
  no autodiff. Each ships `forward`/`derivative` + the muPC `variance_gain`/
  `jacobian_gain` constants (ported verbatim). GELU uses the tanh approximation
  for forward (matching its `derivative`) so f/f' are self-consistent.
- **Acceptance:** ✅ per-activation derivative finite-diff; gain constants;
  explicit Linear path exact with Tanh (finite-diff self/input/weight); a Tanh
  PC graph trains (energy ↓ >2×). 83/83 tests.

*Phase D2 — non-Gaussian energies + Softmax, ANALYTIC (NO Enzyme). ✅ DONE.*
- Decision (user-approved): the explicit path generalizes to all in-scope
  energies analytically, so no autodiff dependency is added. The only genuine
  Enzyme use-case (arbitrary custom forwards, e.g. transformers) is already
  deferred to Phase E+ — so the Enzyme fallback is deferred to when it lands.
- Added an analytic `grad_mu` (∂E/∂z_mu) per energy + generalized the node path
  (`pre_grad = grad_mu·f'`, `mu_grad` for activation-bypassing paths). Bit-identical
  to Phase B at Gaussian precision 1. Energies: Bernoulli, CrossEntropy, Laplacian,
  Huber, KLDivergence. Softmax activation uses the upstream diagonal-Jacobian PC
  approximation (energy not a clean monotone objective ⇒ tested via accuracy).
- **Acceptance:** ✅ per-energy `grad_latent` + `grad_mu` finite-diff (all 6);
  explicit node path exact with the ASYMMETRIC Bernoulli+Sigmoid (finite-diff
  self/input/weight); a Bernoulli binary PC graph trains; Softmax+CE classifier
  improves train accuracy. 104/104 tests.

*Enzyme generic fallback — DEFERRED to when transformers/arbitrary forwards land.*

**Phase E — exhibits + reach (SECONDARY/DEFER).**
- MNIST-style PC classifier exhibit; natural-gradient optimizers; then (deferred)
  transformer nodes / RoPE, Storkey-Hopfield, autoregressive, multi-GPU.

## Explicit defers (do NOT port until needed, exhibit-gated)

- Transformer nodes (`transformer.py`, `transformer_v2.py`, RoPE), Storkey-Hopfield.
- `train_backprop.py` (needs Enzyme; baseline comparator only).
- All pmap/multi-GPU (`multi_gpu.py` is a deprecated shim — skip entirely).
- Dashboards (Aim/trackers), bayesian tuner, A/B experiments.
- Cycle-tolerant partial topological order — treat as an explicit design decision
  if/when recurrent PC graphs are needed (upstream warns + returns partial order).

## Backend & conventions

- **Compute:** plain Julia + Reactant (XLA) for the JIT path, Enzyme for the
  Phase-D autodiff fallback — same backend choices as NGCSimLib/NGCLearn.
- **Optimizers:** Optimisers.jl (Adam/SGD; natural-gradient = ~10-line custom rules).
- **Immutability:** Accessors.jl `@set` over the NamedTuple/struct state.
- **Determinism:** ordered node container; splittable RNG (Xoshiro substreams),
  one subkey per non-source node, replicating upstream key-assignment order.
- **dtype discipline:** only `z_latent` carries clamp dtype (int token indices on
  source nodes); all other state fields stay float (stable Reactant carry).
- **muPC scaling is optional** (`scaling_config === nothing` ⇒ identity), so v0
  runs before Phase C lands.

## Verification discipline

Each phase reproduces behavior against the upstream source (equations + a
runnable check), eager/explicit path is the ground truth, tests gate every
commit, CI green before moving on — same discipline that carried Layers 0 and 1.
