# FabricPC design decisions log

Cross-cutting decisions for the Julia port of FabricPC. Per-file rationale lives
in code comments at the point it's made; this file captures decisions that span
the package. Append as we go; correct in place when a decision changes.

The broad architecture + phasing lives in [DESIGN.md](DESIGN.md); this log is for
the load-bearing per-decision rationale that accrues during the build.

---

## 1. v0 is autodiff-free (explicit-grad Gaussian PC); Enzyme deferred to Phase D

Date: 2026-06-01

Predictive coding's outer (weight) loop does NOT backprop through the inner
(inference) loop — both gradients are local. Upstream computes them by
`jax.value_and_grad` of node-local energy *by default*, but ships closed-form
explicit gradients (`LinearExplicitGrad`, GaussianEnergy):
- `self_grad = precision·(z − z_mu)`,  `gain_mod_error = error · f'(pre_act)`
- `input_grad[e] = −(gain_mod_error · Wₑᵀ)`
- `dW[e] = −(inputₑᵀ · gain_mod_error)`,  `db = −Σ_batch gain_mod_error`

**v0 ports the explicit path** for Linear/Identity/Skip/Residual. This avoids the
single biggest porting risk — Enzyme over closures returning struct-of-arrays aux
(`has_aux`) — for the entire minimal trainable graph. Enzyme (the Julia analog of
`jax.value_and_grad`) lands in Phase D as the generic fallback for non-linear /
transformer nodes, with the explicit path as its conformance oracle.

## 2. Standalone package — no dependency on NGCSimLib/NGCLearn

Date: 2026-06-01

FabricPC's Node/Edge/slot graph is a *parallel* abstraction to NGCSimLib's
Component/Compartment, not built on it (per the layered plan). FabricPC.jl
therefore has no NGC-stack dependency. Cross-stack composition (using NGCLearn
components as FabricPC nodes) is the deferred, optional Layer 3.

## 3. Batch-first layout (match upstream exactly)

Date: 2026-06-01

Arrays are `(batch, *node_shape)`; energies sum over all non-batch dims. Upstream
is batch-first (NHWC/NLC); we match it verbatim to minimize port divergence and
make shape-by-shape verification against the Python source direct. (NGCLearn is
also batch-first, so the stack is consistent.)

## 4. Ordered node container (inference is node-order-dependent)

Date: 2026-06-01

`forward_value_and_grad` accumulates `latent_grad` while iterating nodes in graph
order, relying on already-deposited downstream contributions. A `Dict` is
unordered in iteration — use an ordered structure (Vector of nodes + name→index,
or `OrderedDict`) and a deterministic topological `node_order`, or the PC
gradient assembly is wrong.

## 5. Immutable state via Accessors.jl; dtype discipline on z_latent

Date: 2026-06-01

`NodeState`/`GraphState`/`GraphParams` are immutable structs updated with
Accessors.jl `@set` (the analog of upstream's NamedTuple `_replace`). Only
`z_latent` carries a clamp's dtype (integer token indices on source nodes); every
other state field stays float so a Reactant-compiled inference carry is
type-stable. Port `_validate_clamp_dtypes` (reject int clamps on internal nodes).

## 6. muPC scaling is optional (scaling_config === nothing ⇒ identity)

Date: 2026-06-01

Per-edge scaling factors live behind `NodeInfo.scaling_config`; when `nothing`,
every scaling callsite is an identity no-op — so v0 (Phase B) runs before the
muPC module (Phase C) exists. Critical replicated semantic: edges into
non-variance-scalable slots are ABSENT from the scaling dicts, and callsites treat
a *missing key* as a no-op (NOT multiply-by-1.0), which preserves integer dtype on
embedding/token paths. (`scale_weight_grads` is the one exception — it uses
`get(k, 1.0)`.)

## 7. CI format job is pinned + diff-on-fail from day one

Date: 2026-06-01

NGCSimLib and NGCLearn both hit a cache-drift bug where `julia-actions/cache` +
unpinned `Pkg.add("JuliaFormatter")` restored an older formatter than local,
failing the format gate on clean code. FabricPC's CI pins JuliaFormatter to
=2.5.2, drops the cache in the format job, and prints `git diff` on failure — so
the format gate is reliable and self-diagnosing from the start.

## 8. Phase D2: non-Gaussian energies are analytic — Enzyme deferred

Date: 2026-06-01

Phase D was planned as "Enzyme autodiff fallback + non-linear activations". On
porting it became clear the explicit path already covers everything in scope
WITHOUT autodiff:

- Element-wise activations (Tanh/ReLU/GELU/…) train through the existing explicit
  path — `pre_grad = (∂E/∂z_mu)·f'(pre)` is exact for any element-wise `f` (D1).
- Non-Gaussian energies (Bernoulli/CE/Laplacian/Huber/KL) get an analytic
  `grad_mu = ∂E/∂z_mu` alongside upstream's `grad_latent = ∂E/∂z`, so the node
  path generalizes with no autodiff (D2). Bit-identical to Phase B at Gaussian
  precision 1 (there `grad_mu = -error`).

The ONLY genuine Enzyme use-case is arbitrary custom node forwards (transformers,
attention) — which the DESIGN already defers to Phase E+. So adding Enzyme now
would buy nothing in-scope while adding a heavy dep (minutes-long precompile,
~30-min CI JIT job) and the known Enzyme-through-Dict fragility. Decision
(user-approved): **stay autodiff-free; defer the Enzyme generic fallback to when
transformers/arbitrary forwards are actually ported.** This also keeps the v0
thesis intact — PC needs no backprop.

Softmax is non-element-wise; we ship upstream's diagonal-Jacobian approximation
(`s·(1-s)`, off-diagonal `-sᵢsⱼ` dropped), which upstream itself calls "valid for
element-wise PC gradients". Consequence: with Softmax the energy is NOT a clean
monotone objective, so the Softmax+CrossEntropy classifier is validated by train
ACCURACY improvement, not energy descent.

## 9. muPC on MNIST: hidden-only, and no lr-transfer benefit (yet)

Date: 2026-06-01

Investigated why muPC-on diverged on the MNIST exhibit. Findings:

1. **`include_output=true` is wrong for an MSE one-hot classifier.** It scales the
   output edge by a ≈ 1/N (the muPC O(1/N) output convention), capping `z_mu`
   near 0 — it cannot reach a one-hot target of magnitude 1, so training
   diverges trying to grow the weights ~N×. Fix: `include_output=false`
   (scale hidden layers only); the MSE/Gaussian output keeps standard scaling.
   `include_output=true` is for the softmax+CE setup, not MSE.

2. **muPC-on does not beat muPC-off here.** With `include_output=false`, muPC-on
   trains MNIST (5.70% → 77.45%, hidden-only, eta_infer=0.02, lr=0.02, 5k/2k/6ep)
   but underperforms the plain config (81.65%) and has a *lower* lr ceiling
   (diverges above ~0.03) — the opposite of muP's lr-transfer promise. The
   variance-control property is correctly implemented and verified
   (`test_mupc.jl`: O(1) activations across width 64→4096). The gap is the muPC
   *training* recipe: upstream sets per-layer inference rates / optimizer scaling
   in `train.py` + experiments, not in `mupc.py`. Replicating that is future work.

Decision: ship the exhibit's `FPC_MUPC=1` path (hidden-only, stable inference
settings) as a faithful, honest demonstration that muPC integrates and trains;
do NOT claim a muPC accuracy/transfer win on MNIST. The plain config remains the
default. The full muPC training dynamics are deferred (and noted in DESIGN.md).
