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

## 10. Full muPC training recipe — and the exact Softmax+CE gradient

Date: 2026-06-01

Closed the §9 negative finding by porting the COMPLETE muPC recipe from upstream's
`examples/mupc_demo.py`. The three pieces Phase C lacked:

1. **MuPCInitializer** — weights are unit-variance `gain·N(0,1)`, NOT
   `NormalInitializer(std=0.05)`. muPC's whole point is to DECOUPLE init (unit
   variance) from the per-edge forward scaling; pairing the scaling with a
   small-std init double-shrinks the weights and breaks the parameterization.
   This was the root cause of §9's failure.
2. **AdamW** — muPC's `weight_grad_scale=1` ("optimizer handles magnitude")
   assumes an adaptive optimizer. Plain SGD does not realize the recipe.
3. **Architecture** — deep FC-ResNet: IdentityNode input → Linear stem (MuPCInit)
   → N × LinearResidual(Tanh, MuPCInit) → Linear output (Softmax/CE, Xavier),
   `MuPCConfig(include_output=false)`, `infer_steps = max(20, 3·(N+2))`.

One more fix was needed for hard multi-class tasks: the **exact Softmax+CE
gradient**. Our explicit path used the diagonal-softmax approximation `s·(1-s)`
(faithful to upstream's `SoftmaxActivation.derivative`, but upstream's *plain*
output node uses autodiff → the exact gradient). On easy 3-class synthetic the
diagonal approx trains fine, but on 10-class MNIST it makes the energy ASCEND.
Fix: a `_pre_grad(::SoftmaxActivation, ::CrossEntropyEnergy, …)` dispatch
returning the clean closed form `dE/dpre = s − y` (= z_mu − z_latent), exact for
normalized (one-hot) targets — no autodiff, no off-diagonal Jacobian. Validated
by finite-difference (`test_energies.jl`).

**Result (positive — closes §9):** an 8-block muPC FC-ResNet trains MNIST
5.85% → 81.6% with the energy descending monotonically (0.80 → 0.087), and a
deep 6-block muPC residual classifier is a CI test (`test_mupc_resnet.jl`).
muPC + the full recipe makes deep PC nets trainable. The diagonal-softmax
approximation remains for the general (non-CE) softmax case; only the CE pairing
gets the exact path.

## 11. Reactant/XLA JIT: feasible + ~9× — full integration is the GraphState refactor

Date: 2026-06-01

Feasibility spike (`benchmark/reactant_jit.jl`): Reactant 0.2.262 loads here and
`@compile`s a fixed-step PC relaxation loop (matmul + activation + local-grad +
latent update) over a tuple of per-node arrays. JIT matches eager to ~3e-6 and
runs **8.8× faster** (synced) on the MNIST-shaped MLP inference (50.7 → 5.8 ms).
The async-dispatch trap: timing the thunk WITHOUT `Array(...)` shows a bogus
~1000×; always materialize to time XLA compute.

What blocks dropping `Reactant.@compile` straight onto `run_inference`: the eager
state is `Dict{String,NodeState}` / `Dict{String,Matrix}` with per-step `merge`
(rebuilding Dicts). XLA traces array ops over a STATIC pytree; Dict-keyed,
per-step-rebuilt containers are not traceable. Full integration plan (multi-
session, opt-in — keep the eager Dict path as ground truth):

1. **Traceable state representation.** Lower a `GraphStructure` to a static plan:
   node order as integer positions, edges as `(src_pos, tgt_pos, slot)` ints, and
   per-node positional weight order. Represent state as `NTuple{N,NodeState}` and
   params as `NTuple{N,NodeParams}` (NodeState/NodeParams are already structs of
   arrays — traceable). No Dicts, no mutation in the loop.
2. **Dict-free node forward.** Variants of `forward` / `forward_and_*_grads` that
   take inputs/weights as tuples aligned to the node's in-edges (integer-indexed),
   so the traced region has no string-keyed lookups.
3. **Package as a weakdep + extension** (`FabricPCReactantExt`, Reactant in
   `[weakdeps]`), mirroring NGCLearn — the core package stays Reactant-free; the
   JIT is opt-in via `using Reactant`. Validate JIT == eager in the extension's
   tests; gate the heavy CI job off-by-default.

Decision: feasibility + payoff are PROVEN and committed as a benchmark; the full
GraphState refactor + extension is scoped above as the next deliberate increment.

### §11 update — increment 1 landed (Dict-free inference path)

`src/jit_flat.jl`: `CompiledPlan` (GraphStructure → static integer topology),
positional `FlatNodeParams`, and `flat_run_inference` — the inference hot loop
re-expressed over a position-indexed `Vector{NodeState}` with Dict-free node
forwards (`flat_forward` / `flat_latent_grads` for Linear/Identity/Skip/Residual,
reusing the shared pre_grad/mu_grad/energy helpers). No Dicts, no string-keyed
lookups, no `merge` in the loop. Validated == the eager Dict path bit-for-bit
(`test_jit_flat.jl`: MLP train+predict, residual net). Remaining for the JIT:
swap `Vector` → `NTuple` and wrap in `Reactant.@compile` inside a
`FabricPCReactantExt` weakdep/extension (increment 2); add the flat weight-grad
path for JIT'd training.

### §11 update — increment 2 landed (Reactant extension, 32× in-package)

`ext/FabricPCReactantExt.jl` + Project.toml `[weakdeps]`/`[extensions]`: Reactant
is a WEAKDEP, so the core package stays Reactant-free and the JIT is opt-in via
`using Reactant`. `compile_inference(structure, params, clamps; batch)` traces the
Dict-free `jit_inference_runner` with `Reactant.@compile` and returns a
`CompiledInference` callable mapping `(params, init_state) → converged z_latents`.

Two fixes made the trace work: (a) `FlatNodeParams` made type-PARAMETRIC — concrete
`Matrix`-typed fields broke Reactant reconstruction (it substitutes
`ConcretePJRTArray`s); (b) the `@compile`'d function takes only Tuples of arrays
(params flattened via `flatten_param_arrays`, per-node z_latents), repacking the
struct INSIDE the traced region. The carried state is just z_latents per node
(z_mu/error/… are recomputed each step), so the pytree stays a clean tuple.

Validated (env with FabricPC + Reactant): JIT == eager bit-for-bit (max|Δ| = 0)
and **32× faster than the eager Dict path** (280 ms → 8.7 ms, MNIST-shaped MLP,
batch 256, 20 steps) — larger than the §11 microbenchmark's 8.8× because the real
Dict-based eager path is slower than the naive-array baseline. Core CI stays
Reactant-free (the flat path is tested == eager in `test_jit_flat.jl`); the
extension is exercised via `examples/jit_inference.jl` (needs Reactant).

Remaining: a flat weight-grad path to JIT the full train_step (not just
inference); NTuple-typed state for fully type-stable tracing.
