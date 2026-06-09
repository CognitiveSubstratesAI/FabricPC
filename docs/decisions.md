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

## 12. Phase D activated — Enzyme node-local autodiff seam (PC-transformer enabler)

Date: 2026-06-08

Decision #8 deferred the Enzyme generic-gradient fallback "to when
transformers/arbitrary forwards are actually ported." That trigger has arrived
(the PC-transformer is the goal), so Phase D is now activated — as a weakdep
extension (`FabricPCEnzymeExt`, loaded on `using Enzyme`), honoring #8's cost
concern: the base package stays autodiff-free; only consumers of custom nodes pay
the Enzyme precompile / CI-JIT cost.

**The seam.** A node implements ONLY `compute_mu(node, params, inputs) -> z_mu`
(concrete arrays). `energy_kernel`, `forward`, and the generic
`forward_and_weight_grads` / `forward_and_latent_grads` (on `::AbstractNode`) are
then derived: Enzyme reverse-mode differentiates `energy_kernel` w.r.t. `params`
(learning) and w.r.t. `(inputs, z_latent)` (inference) — the Julia analog of
upstream's base-class `jax.value_and_grad(forward)`. Linear/Identity/Skip/Residual
keep their closed-form overrides (strictly more specific ⇒ win dispatch).

**Still PURE PC, not backprop.** Each autodiff call is confined to ONE node's
local energy `E_node(params, inputs, z_latent)`; it never propagates through the
network or the inference relaxation. Enzyme only spares the hand-derivation of a
complex node's *local* gradient — exactly what makes transformer / attention /
Storkey-Hopfield nodes expressible.

**Two hard lessons (gated against the closed-form Linear oracle, ~1e-7):**
1. Enzyme must differentiate CONCRETE arrays. Differentiating the real `forward`
   failed in `runtime_generic_augfwd` because NodeState's `::Any` fields
   (decision #5) force Enzyme's type-unstable path. Hence `compute_mu` returns
   plain `Matrix{Float32}` and NodeState bookkeeping stays OUTSIDE the
   differentiated region.
2. The input-edge `Dict` iteration mixes active/constant entries →
   `EnzymeRuntimeActivityError`. Fix: `set_runtime_activity(Reverse)` (correct,
   small perf cost). The core stubs are VARARG fallbacks (`_ad_*(args...)`), not
   the ext's typed signature — a same-signature stub is a forbidden
   precompile-time method overwrite.

**Validation (`test_autodiff_seam.jl`):** (1) conformance — a `compute_mu`-only
`ADLinear` reproduces the closed-form Linear weight/latent/input grads to 1e-4;
(2) end-to-end — a 1-hidden-layer non-linear `MLPNode` (no closed-form local grad)
trains a separable task purely by PC (energy falls, accuracy > 0.85). Full suite
161/161. Next: the transformer node (rank-2 sequence shapes + attention forward)
on this proven seam.

## 13. PC-transformer (TransformerBlock) — native Julia, gradients via the seam

Date: 2026-06-09

Built the first transformer block as a single PC node (port of
`fabricpc/nodes/transformer.py`), the flagship of decision #12's seam: a
transformer whose every parameter is learned by LOCAL predictive coding, no
backprop. The node implements only `compute_mu` (LayerNorm → MHA+RoPE → ·√seq →
residual/√2 → LayerNorm → FFN → residual/√2); its weight + latent gradients come
free from the Enzyme seam.

**Native Julia, not a JAX transliteration (user directive).** Two consequences:
1. *Embrace column-major; drop numpy-byte-compatibility.* We train from scratch,
   so any consistent head partition + RoPE pairing is valid — the gate is
   mathematical correctness, not matching numpy's layout. This dissolves most of
   the "column-major hazard." Key fact used: for last-axis contraction,
   `reshape(reshape(x, B*S, E) * W, B, S, :)` IS layout-correct in column-major
   (batch/seq are the leading fast axes). Head split = contiguous embed blocks;
   RoPE = adjacent-pair rotation. Both arbitrary-but-consistent.
2. *No NNlib `batched_mul`.* Julia loops/`map`/`stack`/`cat` are fast and idiomatic;
   attention is a native per-(head, batch) loop. Base + LinearAlgebra only.
   Reactant/XLA (already wired for inference via FabricPCReactantExt) is the future
   perf compiler for this eager forward — Reactant is Julia's native XLA path, the
   right answer rather than mimicking JAX ops.

**Rank-3 support.** Latents are `(batch, seq, embed)`. The framework was already
rank-agnostic (energy sums over all non-batch dims; state init uses `shape...`).
Only the seam's input/latent dicts hardcoded rank-2: generalized `_concrete_inputs`
+ `_ad_latent_grads` to `Array{Float32,N}` with N inferred (kept concrete-typed so
Enzyme stays on its type-stable path). All node PARAMS stay 2D `Matrix` (biases /
LayerNorm γ,β stored `(1,embed)`, reshaped to `(1,1,embed)` in `compute_mu`), so
NodeParams is unchanged.

**Two gates (gate-before-build, no Python — jax/numpy unavailable):**
1. Native vectorized forward == independent explicit-loop oracle, reldiff 0.0
   (RoPE on/off). Locks the reshapes/head-split/RoPE.
2. Enzyme grad == finite-difference (dW 0.009, dx 0.0004) — the native
   `map`/`stack`/`cat`/index forward IS Enzyme-differentiable (`set_runtime_activity`).
Both re-asserted in-repo (`test_transformer.jl`): forward-oracle + an autoencode
that trains by local PC (energy falls, no NaN; AdamW, infer_steps=1 since both
graph endpoints are clamped ⇒ no free latents). Full suite 168/168.

v1 scope: full (non-causal) attention, single `"in"` slot, no mask. DEFER (not
gaps): causal masking, the decomposed per-stage transformer (`transformer_v2.py`:
Embedding/MhaResidual/LnMlp/VocabProjection as separate PC nodes), Storkey-Hopfield,
Reactant+Enzyme JIT of the block.

## 14. Reactant+Enzyme JIT of the PC-transformer block (feasible; perf layer)

Date: 2026-06-09

The transformer node's local PC gradients run eagerly via the Enzyme seam
(#12/#13, correctness-first). This adds the PERFORMANCE path: the block forward
AND its local gradient compile to XLA via Reactant — Reactant being Julia's native
XLA frontend (same backend as JAX), not a JAX transliteration. Reactant intercepts
`Enzyme.autodiff` under `@compile`, so the *same* local-energy gradient is lowered
to XLA differentiation (Reactant+Enzyme). Still pure local PC: the autodiff is
confined to one block's local energy.

**Recipe (gated standalone before building):**
- A Dict-free, positional-array kernel `_tb_block_flat` (Reactant traces arrays/
  tuples, not Dicts) — numerically identical to the eager `compute_mu`
  (test_transformer.jl: reldiff 0.0). `flat_block_args` bridges NodeParams→tuple.
- `ntuple(…, Val(N))` STATIC unrolling of heads + batch. A `map`/`for` over a range
  makes the loop index a *traced* value, so the head-slice range `(h-1)*Dh+1:h*Dh`
  becomes a TracedUnitRange and `Q[:,:,cols]` fails ("non-boolean TracedRNumber in
  boolean context"). `ntuple(Val)` gives literal indices ⇒ concrete slices. Batch
  size `B` is a `Val` (fixed per compile, like the inference JIT).
- Gate A (forward): Reactant == eager, reldiff 4e-8. Gate B (gradient):
  Reactant+Enzyme == eager-Enzyme, reldiff 7e-8 (dx) / 3e-7 (dW).

**Benchmark** (`benchmark/transformer_jit.jl`, benchmark/jit env): forward JIT==eager
(1.9e-6) at ~1.5–1.9× over eager on CPU (XLA's edge is modest on CPU/small blocks;
larger on GPU); Reactant+Enzyme gradient validated by directional finite-difference
(⟨∇,v⟩ reldiff 1.3e-3) at ~7ms. Enzyme + Revise added to the benchmark/jit env.

**Enzyme friction documented (so we don't relearn):**
- Eager Enzyme on the `ntuple` kernel is unstable with MANY active args at once
  (falls into `runtime_generic_rev` → `sum(abs2, Nothing)` / `PrimalErrorThunk`).
  The eager seam uses the `map`-based `compute_mu` and differentiates the full
  NodeParams fine; the `ntuple` kernel is for the Reactant path, where the full
  gradient lowers cleanly. Per-array `Duplicated` (rest `Const`) is the robust form.
- An outer scalar factor on the returned loss (`/2`, `0.5f0*…`) trips Enzyme's
  reverse seeding here (`*(Float32, add_one_in_place)`); use a bare `sum(abs2,…)`.

**Deferred (the real integration, multi-session — the GraphState refactor of #11):**
routing the transformer node's gradients through a compiled+cached kernel during
`train_pcn` (Dict→flat at the training callsite, compile-per-(config,batch) cache).
This benchmark proves the kernel + the Reactant+Enzyme gradient; wiring it into the
eager training loop is the next step. Eager PC-transformer remains the default.

## 15. Causal self-attention (autoregressive PC-transformer) + the JIT-training finding

Date: 2026-06-09

**Causal masking.** `TransformerBlock(...; causal=true)` masks future keys for
autoregressive (next-token) modeling: an additive lower-triangular mask
(0 on/below diag, -1f9 above) is added to the attention scores before softmax, and
the softmax variance compensation becomes per-position √i (query i attends to i
keys) instead of √S. Applied identically in `compute_mu` (eager) and
`_tb_block_flat` (the Reactant JIT kernel, via an optional `Val{CAUSAL}` that
defaults false — existing callers unchanged). No external mask input slot: causality
is a node config flag (training from scratch needs no data-supplied mask), keeping
the single rank-3 `"in"` slot and the rank-N seam intact.

Validated (test_transformer.jl): causal forward == the explicit-loop oracle;
causal flat kernel == compute_mu (reldiff 0.0); and a SEMANTIC no-future-leak test —
perturbing the last input token NON-uniformly (a uniform offset is removed by
LayerNorm's mean-subtraction, which would make the test vacuous) leaves the first
causal output position unchanged (0.0), while the same perturbation changes a
non-causal block's first output (0.65). That contrast proves the test isn't vacuous.

**JIT-training-integration finding (investigated, deferred — see memory).** Routing
the block's compiled Reactant+Enzyme gradient into `train_pcn` was gated and found
FEASIBLE (the full 16-weight gradient compiles under Reactant and matches finite-diff
to 0.014) but BLOCKED by an upstream tooling conflict: **eager `Enzyme.autodiff` and
Reactant+Enzyme `@compile` cannot coexist in one process** — a prior eager autodiff
poisons the subsequent compile (`setfield!: immutable Tuple`). So the "JIT path with
eager fallback" architecture is broken; a working integration needs either an
all-compiled (no eager Enzyme) graph or the on-device GraphState refactor (#11).
Deferred; the eager PC-transformer + the block-JIT benchmark (689de8c) stand.

## 16. Decomposed (fully-PC) transformer stages — PC at every sub-component

Date: 2026-06-09

Ported the middle stages of transformer_v2.py as three separate PC nodes, so
predictive coding operates at EVERY sub-component, not just the block boundary:
  x → MhaResidual(z=x+W_o·MHA(LN x)) → LnMlp1(z=GELU(LN·W_ff1)) → Mlp2Residual(z=res+·W_ff2)
with a skip (residual) edge from MhaResidual into Mlp2Residual. Each node implements
only `compute_mu`; local PC gradients come from the Enzyme seam. Reuses the `_tb_*`
helpers (`_tb_mha` was generalized to take config explicitly — num_heads/use_rope/
causal — so the monolithic block and MhaResidual share it; no regression, 184/184).

Unlike the monolithic `TransformerBlock`, residuals are PLAIN (z = x + mha, z =
res + mlp2) — no 1/√2 or √S scaling (matches transformer_v2). Causal MhaResidual ⇒
autoregressive.

KEY new surface validated: **multi-slot nodes**. `Mlp2Residual` has "in" (from
LnMlp1, (S,ff)) + "residual" (skip from MhaResidual, (S,embed)) — different feature
dims, same rank-3, so `_concrete_inputs` (one inferred rank N=3) handles them, and
the seam returns local PC input gradients for BOTH edges (test asserts the residual
edge's grad == z_mu − z_latent, since the residual passes straight through). The
skip is just another input edge to the autodiff — `is_skip_connection`/
`is_variance_scalable=false` only matter for muPC scaling (off here).

Validated: forward shapes + causal no-leak (CI, fast) + the multi-slot grad smoke
(CI, Enzyme). End-to-end is the heavy example examples/decomposed_transformer_pc.jl
— a fully-PC decomposed transformer doing autoregressive next-token prediction with
MhaResidual/LnMlp1 as FREE latents relaxed by multi-node PC inference (infer_steps>1),
x + output clamped. DEFER: EmbeddingNode + VocabProjectionNode (discrete-token I/O —
embedding lookup + vocab logits + custom latent-grad), a follow-up.

## 17. EmbeddingNode + VocabProjectionNode — transformer_v2 token I/O complete

Date: 2026-06-09

Completed the decomposed transformer_v2 pipeline with its discrete-token endpoints:
  tokens → Embedding → MhaResidual → LnMlp1 → Mlp2Residual → VocabProjection → logits

EmbeddingNode (discrete token ids → embeddings) uses EXPLICIT gradients, NOT the
autodiff seam: the input is integer token indices, so (a) the seam's Float32
concrete-ify would corrupt them and (b) there is no gradient through a discrete
index. forward = row lookup embeddings[idx]; forward_and_weight_grads = scatter-add
of ∂E/∂z_mu into embedding rows by token id (validated == manual scatter, reldiff 0);
forward_and_latent_grads = zero input grad (discrete) + self grad = grad_latent (the
latent is anchored to the lookup). Clamps arrive as Float32 (set_latents_to_clamps
casts), so ids are round()'d back to Int for indexing.

VocabProjectionNode (embeddings → vocab logits) is seam-based (compute_mu =
activation(x·W_out + b_out)). Defaults to Identity + Gaussian (rank-agnostic, trains
next-token to one-hot targets, proven). Upstream's softmax+KL default is DEFERRED:
the port's SoftmaxActivation softmaxes over dim 2, not the rank-3 vocab axis — a
last-axis softmax would be needed (+ the softmax+CE non-monotone-energy caveat #8).

Tests 190/190 (Embedding lookup + scatter weight-grad + discrete latent-grad; Vocab
forward). examples/decomposed_lm_pc.jl: the FULL fully-PC transformer LM (token ids
→ logits), Embedding/Mha/LnMlp1/Mlp2 as free latents relaxed by multi-node PC
inference, trained end-to-end by LOCAL PC (no backprop). transformer_v2 port complete.

WORKFLOW NOTE: warm Revise session hot-reloads EXISTING method/test edits (instant),
but adding NEW structs (these nodes) to a tracked package is NOT picked up by
Revise.revise() — needs a module reload. Validate new-node forwards via a fast cold
FabricPC run (no Enzyme needed for explicit nodes); use Revise for iterate-on-existing.
