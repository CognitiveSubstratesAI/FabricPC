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

### §11 update — first CROSS-LANGUAGE number (J-04, 2026-07-14): ~7-8× vs real `jax.jit`

Every number above this point is Julia-vs-Julia (eager Dict path vs Reactant-compiled,
no JAX involved) — do not conflate them with this one. J-04
(`docs/AUDIT_REGISTER.md` section 5) was scoped correctly: J-01/J-02 block compiling
`compute_local_weight_gradients` (the weight-gradient/M-step) under Reactant+Enzyme, but
`compile_inference`/`flat_run_inference` (the E-step relaxation loop only) is a SEPARATE,
already-validated path, and the MNIST-MLP architecture (`examples/mnist_pc.jl`,
`Linear`-only) is fully covered by `flat_forward`/`flat_latent_grads` — so compiled PC
*inference* (not training) vs upstream's real `jax.jit` was measurable without waiting on
J-01/J-02, and now has a real number: same architecture/batch(256)/`eta_infer`(0.1)/
`infer_steps`(20), same weights loaded on both sides (RNG-trap discipline, this session's
established fixture convention) —

**Julia+Reactant's `compile_inference` runs ~7-8× faster than upstream's own
`jax.jit(run_inference)`** (steady-state 4.75-6.38ms vs 42.1-43.2ms, range 6.6×-8.3× across
repeated pairings on this shared 2-core dev machine — absolute compile times swung
19s-327s with concurrent-process contention, but the ratio held steady, which is the more
trustworthy number here). Numerics validated to float32 precision (max|Julia-compiled −
JAX-jit| = 2.29e-6 on `h`, the only actually-relaxed latent; exact 0 on the clamped `x`/`y`).
This is the first CROSS-language, both-sides-compiled comparison this codebase has produced
— distinct from the 8.8× (§11, Julia eager vs Julia+Reactant, a simplified hand-rolled
kernel) and 32× (increment 2 above, Julia's real eager Dict path vs Julia+Reactant, same
architecture as this new measurement) numbers, both Julia-internal. The new measurement's
own Julia-internal ratio (compiled vs its own eager Dict path, measured in the same run)
came out 28-38× — independently reproducing the existing 32× claim, a useful cross-check
that the two benchmarks are measuring compatible things.

**Re-verified after Reactant 0.2.264→0.2.273 bump (2026-07-14, `benchmark/jit/Manifest.toml`;
ReactantCore 0.1.19→0.1.21).** Fresh fixtures (`benchmark/mnist_inference_vs_jax.{py,jl}`),
same shared 2-core dev machine under concurrent-process contention: Julia+Reactant steady-state
9.25ms vs `jax.jit` 72.76ms ⇒ ~7.87×, squarely inside the previously-measured 6.6×-8.3× range;
numerics max|Julia-compiled − JAX-jit| = 2.29e-6 on `h` — identical to the pre-bump value,
exact 0 on clamped `x`/`y`, also unchanged. Absolute times are higher than the original
4.75-6.38ms/42.1-43.2ms pairing (more concurrent load on the shared machine at measurement
time), consistent with this section's own note that absolute times swing with contention
while the ratio is the trustworthy number. No regression.

**Scope, stated plainly so it isn't overclaimed**: inference only. Weights are fixed
throughout (no weight update, no `optax`/`Optimisers` step) — this is NOT a "PC training is
Nx faster" result. J-01/J-02 (compiling the weight-gradient/training step) remain open and
untouched by this measurement.

A real bug was found and fixed while building the fixture-loading side of this benchmark,
worth recording as a general caution for any future cross-language array-dumping work in
this codebase: `numpy.ndarray.tofile()` always writes C-order bytes regardless of the
array's actual memory layout — `np.asfortranarray(arr).tofile()` is a silent no-op for
producing column-major bytes, so a naive Julia-side `reshape` of the raw bytes SILENTLY
SCRAMBLES any non-1×N array. The bug was self-consistent on the Julia side (eager ==
compiled, since both read the same scrambled data) and only surfaced by comparing against
JAX's OWN disk round-trip, not a live in-memory JAX array — a live-vs-disk comparison alone
would have missed it. (This codebase's existing Tier A-D `.npz`-based fixtures were never at
risk — `np.savez`/`NPZ.jl` handle layout correctly; this was specific to this benchmark's
raw-`.bin`-file loading path, which has since been fixed in-code.)

**Methodology re-audit, on request, against exactly the five failure modes a skeptical
reviewer would check (2026-07-14):** re-read `benchmark/mnist_inference_vs_jax.{py,jl}` and
`ext/FabricPCReactantExt.jl` in full, independently, rather than trusting this section's own
prose, and cross-checked the cited numbers against raw run artifacts recovered from this
session's own scratchpad (`jax_result.json`, Julia stdout transcripts) rather than trusting
the summary above. All five hold:
1. **Compile time is excluded from the ratio on both sides.** Python times the first
   `jit_run` call into its own `compile_time_s` field, never folded into the speed number
   (`mnist_inference_vs_jax.py:155-179`). Julia times `Reactant.@compile` into `t_compile`
   separately, discards the first post-compile call, then — identically to the Python side —
   runs 5 discarded warmup calls before `min`-ing 30 timed calls (`mnist_inference_vs_jax.jl:50-59`).
   Both sides force device→host materialization before/inside the timed call (Python's
   `jax.block_until_ready`, explicitly commented as guarding against "the async-dispatch trap
   … timing the thunk WITHOUT `block_until_ready` shows a bogus ~1000x"; Julia's
   `[Array(out[i]) for i in eachindex(out)]` in `FabricPCReactantExt.jl:47`) — neither side
   times an unresolved async dispatch queue.
2. **Weights are shared via a real RNG-trap, not independent randomness.** Python's `dump()`
   writes `Wxh,bh,Why,by,Xb,Yb` as raw bytes; Julia's `load_arr()` reads them back (with a
   documented, correct row/column-major un-transpose) and builds `GraphParams`/clamps
   DIRECTLY from the loaded arrays — Julia's own `initialize_params`/RNG is never called for
   this benchmark (`mnist_inference_vs_jax.jl:87-98`, comment states the RNG-trap explicitly).
3. **The `xla_gpu_*` flags question is moot for this number, confirmed not assumed.**
   `mnist_inference_vs_jax.py` never imports upstream's `jax_setup.py` (confirmed by reading
   its full import list), so none of the three reproducibility-over-speed flags are applied —
   irrelevant regardless, since this ran CPU-only, independently confirmed three ways: a
   recovered `jax_result.json` shows `"devices": ["cpu:0"]` (JAX's own `jax.devices()` dump),
   this machine has no `nvidia-smi`, and `nproc`=2 matches the "2-core dev machine" language
   already in this section.
4. **This is genuinely the cross-language, both-compiled number**, not a relabeled
   Julia-internal one — a real `jax.jit(run_inference)` trace on one side, a real
   `Reactant.@compile`-traced call on the other, both timed steady-state. The
   Julia-vs-Julia secondary number this same run also reports (28-38× above, reproduced in
   the recovered transcripts as 26.23×/28.37×) is textually labeled a sanity cross-check
   against the pre-existing 32× claim, never substituted for the headline cross-language
   number anywhere in this file or in `docs/AUDIT_REGISTER.md`'s J-04 entry.
5. **The numerical-diff figure compares real outputs, same inputs/weights**, not two
   independently-fast unrelated runs — reproduced exactly from a recovered transcript:
   `max|julia_jit − jax_jit| h = 2.29e-06`, `x = 0`.
   The specific ratio numbers were independently re-derived from raw artifacts, not
   re-quoted: `72.758ms / 9.2452ms = 7.869×`, matching "~7.87×" to the digit; the original
   pairing's range and the compile-time swing (`19s`-`327s`) also reproduce exactly.

**One precision nuance worth stating plainly, not a flaw:** neither script computes the
cross-language ratio itself — each independently prints/dumps its own steady-state number,
and the "~7-8×"/"~7.87×" figure is a manual division of two separately-run scripts' outputs
sharing a data directory, not one unified harness run. The arithmetic checks out against raw
artifacts (above), but a maximally precise citation of this number should describe it as
"computed by comparing the two scripts' independently-reported steady-state numbers," not
imply a single combined benchmark invocation.

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

### §15 update (2026-07-14, J-01): the conflict is EAGER-ENZYME-specific, not
### AD-generally — but a more dangerous, previously-unknown hazard replaces it

The framing above ("eager Enzyme and Reactant+Enzyme `@compile` cannot coexist")
conflated "any eager AD" with "eager Enzyme specifically." This codebase's actual
production eager seam is Zygote (§19), which never calls eager `Enzyme.autodiff` at
all — so the real, narrower question is whether Zygote-eager + Enzyme-only-inside-
`@compile` coexist. Execution-verified (`benchmark/jit_zygote_expt/main_experiment.jl`,
not committed — a diagnostic script): in a process that already did real eager-Zygote
seam work, `Reactant.@compile`-with-`Enzyme.gradient`-inside (never eager
`Enzyme.autodiff`, mirroring `benchmark/transformer_jit.jl`'s `wgrad` precedent, a
raw-array pattern that bypasses FabricPC's actual seam dispatch) compiles and matches
Zygote's eager gradient to ~1e-6/1e-7. **No poisoning of that pattern observed.**

**But this investigation surfaced a more dangerous hazard while checking it, and it
changes the practical rule in §23 below.** `using Reactant` transitively loads Enzyme
(a hard dependency of Reactant, not a weak one) — which silently triggers F-04's
dispatch-override bug (`docs/AUDIT_REGISTER.md` section 1, reopened 2026-07-14):
`FabricPC._ad_param_grads`/`_ad_latent_grads` flip to Enzyme's implementation the
moment `using Reactant` loads, with **zero explicit `using Enzyme` anywhere in user
code**, and no catchable exception (F-04's guard fires internally but is swallowed by
Julia's own extension-loading error handling — see the reopened F-04 write-up for the
full mechanism). So: **the risk was never really about disciplined avoidance of
`using Enzyme`** — merely loading Reactant for an entirely unrelated reason (e.g. an
inference-only JIT benchmark) silently poisons the REAL seam for any Zygote-seam-
dependent node (`TransformerBlock`/`EmbeddingNode`/`VocabProjectionNode`/decomposed
MHA-FFN family) touched afterward in that same session — the documented MHA-crash
risk, now reachable with no explicit trigger at all.

**Revised practical rule, until F-04's dispatch mechanism gets a real fix** (routing
through a Ref-held function pointer the guard installs only after its check passes,
instead of two competing top-level methods — scoped, not attempted this session): do
NOT `using Reactant` in any session that also needs the real seam (`forward_and_
weight_grads`/`forward_and_latent_grads` dispatch) on a Zygote-seam-dependent node.
Only the raw-array/`wgrad`-style bypass pattern (used by `benchmark/transformer_jit.jl`
and this investigation's own experiment script) is currently safe to use post-
`using Reactant` — it never goes through the poisoned dispatch at all, differentiating
concrete arrays directly instead.

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

## 18. StorkeyHopfield — first composite-energy node (PC + attractor); seam handles it

Date: 2026-06-09

Ported the Hopfield associative-memory node — the last remaining substrate node type.
It's the FIRST node with a COMPOSITE energy:
  E_total = E_pc(z, z_mu) + s·E_hop(z, W),   E_hop = (1/2D)·zᵀ(W²−W)z
E_pc pulls the latent toward the upstream prediction; E_hop pulls it toward stored
patterns (attractors in W). Attractor dynamics (denoising / pattern completion) arise
from the Hopfield energy gradient (s/D)(W²−W)z accumulated to latent_grad during PC
inference. z_mu = activation(probe·blend + (probe·W)(1−blend) + b), blend=1/(1+s);
strength s learnable (softplus-constrained, init 1) or fixed.

KEY: the Enzyme seam handles the composite energy with no new machinery — the node
OVERRIDES `energy_kernel` (returns Σ(E_pc + s·E_hop)) so Enzyme differentiates the
FULL energy for both the latent grad (PC pull + attractor pull) and the weight grad
(incl ∂E_hop/∂W, the W²−W term), and overrides `forward` so state.energy reports the
full energy (the generic forward sets E_pc only). Validated vs finite-difference:
forward energy exact; latent grad reldiff 5.6e-6; weight grad reldiff 6.2e-5. W is
stored under the input edge key (gradient flows to the source); _prepare_W symmetrizes
(differentiable). This establishes the pattern for ANY node whose energy is not just
node.energy: override energy_kernel (+ forward for reporting), seam does the rest.

Tests in test_storkey_hopfield.jl (forward energy + composite-energy seam grads vs FD
+ strength=0 pass-through). examples/hopfield_assoc_memory.jl: a Hopfield node learns
associative recall (denoise noisy ±1 probes → clean patterns) by LOCAL PC. With this,
the FabricPC PC-substrate node set matches upstream (Linear/Identity/Skip/Residual/
Transformer/transformer_v2/StorkeyHopfield) — ~complete. Remaining non-substrate:
multi-device training, dashboards/dataloader/experiments/tuner, train_backprop (the
anti-thesis baseline). decisions.md #18.

## 19. Autodiff backend = Zygote (Enzyme segfaults on attention); params as SoA

Date: 2026-06-17

The transformer/attention nodes have no hand-written local gradient — they rely on the
Phase-D autodiff seam (differentiate `compute_mu` for the node-LOCAL PC gradient, NOT
backprop). The original seam used Enzyme. On Julia 1.12, **Enzyme aborts on the full
multi-head-attention block** (`addToDiffe: "unhandled accumulate with partial sizes"`;
proven by 9 bisections — every ISOLATED gradient differentiates, only the combined graph
crashes, and it's an opaque LLVM abort, not a debuggable Julia error). Decision: switch
the seam to **Zygote**, which handles FabricPC's functional forwards. The node's gradient
is FD-validated correct (W_o AD=0.59/FD=0.59, ln1_gamma 15.42/15.41, <1%).

Supporting changes:
- **`NodeParams` weights/biases `Dict` → `SoA(NamedTuple)`** with a Dict-like read API
  (`getindex`/`get`/`haskey`/`keys`/`values`/`pairs`/`iterate`). Type-stable (no `Dict`
  i64-hash internals in the differentiated region — which also tripped Enzyme) and AD-
  friendly. Construction coerces; mixed `NodeParams(::Dict, ::SoA)` supported.
- **SoA iteration is Zygote-differentiable**: the Symbol→String key conversion
  (`jl_cstr_to_string`, a foreigncall Zygote can't differentiate) is routed through
  `_soa_key`/`_soa_keys`, marked `Zygote.@nograd` in the ext. Keys are structural — values
  still get gradients. This is what lets a generic node's `compute_mu` ITERATE an SoA
  (sum over input edges); the fixed-key transformer never hit it, ADLinear/MLP/Storkey do.
- `@nograd` the constant tables (causal mask / RoPE / variance comprehensions).
- **Suite is Zygote-only** (ea9ee94): the Enzyme and Zygote exts implement the SAME seam
  hooks and cannot co-load, so one backend wins. World-age gotcha — load `using Zygote` at
  the TOP of `runtests` (before any testset) so the ext is registered first, else the first
  file to `using Zygote` triggers the load but can't see the new methods in its own frame →
  the seam raises its "load a backend" hint. Enzyme ext retained (opt-in) for simple/dense
  nodes. Full suite 260/260.

NOTE: Reactant remains the FORWARD/perf JIT for the analytic-node inference loop — it is
NOT an AD engine (it uses Enzyme-MLIR for AD), so it does not bear on this seam choice. A
future "purist PC" alternative is to hand-derive the analytic attention gradient (no AD,
Reactant-compilable); deferred. decisions.md #19.

## 20. Transformer LM trains: AdamW(1e-3) + Softmax/CrossEntropy output (transformer_v2)

Date: 2026-06-18

Two changes that take the assembled transformer LM from "differentiates" to "demonstrably
learns", both grounded in upstream:
- **Optimizer**: plain SGD@0.02 DIVERGES on a transformer (weights → 1e7 → NaN by step ~4;
  confirmed by a trajectory probe). Upstream uses `optax.adam(1e-3)` / `adamw(1e-3, wd=0.1)`
  EVERYWHERE (ab_experiment.py, every train_*.py). Switched to `AdamW(1e-3, wd=0.1)` — energy
  drops ~191→8 over 50 steps; the trainer already dispatches to AdamW for a non-scalar opt.
  The gradient was correct all along (this was never a seam bug).
- **Output layer** (ported from upstream's unmerged `feature/transformer_block_v2`):
  `VocabProjectionNode` default `Identity+Gaussian` → **`Softmax+CrossEntropy`** (the proper
  next-token classification objective; z_mu is now a probability distribution, not logits).
  Required making `SoftmaxActivation` softmax over the LAST axis (`dims=ndims(x)` — was a
  hardcoded `dims=2`; backward-compatible for rank-2, correct for the rank-3 (B,S,V) vocab
  axis, which is exactly what had blocked Softmax on VocabProjection). Inits matched too:
  embed `std=1.0`, output `std=√(1/E)`. Generation consumes z_mu directly (already probs).

ONLY the self-contained CE-energy + inits were ported from the unmerged branch; its
structural pieces (explicit skip/mask nodes, FeedforwardStateInit, MuPCConfig, Optuna tuner)
and `feature/convolution` (ConvNode) are deferred until they merge to upstream main — porting
unmerged-branch structure invites churn. test_transformer_lm 14/14; full suite 261/261.
decisions.md #20.

## 21. Five locked-in deliberate divergences (X-01…X-05, docs/AUDIT_REGISTER.md section 2)

Date: 2026-07-13

Five points where this port intentionally does not match upstream byte-for-byte. Each was
raised by the original coverage audit; writing them here (not just the register's one-line
table) so future work has the actual rationale, not just a label.

**X-01 — all-float clamp pipeline.** Clamping force-casts every clamp value to `Float32`
(`Float32.(val)`), and `EmbeddingNode` rounds it back to an integer index
(`Int(round(idx))`) on read. Upstream 0.3.1 had a dtype-preservation contract with an
explicit `_validate_clamp_dtypes` guard; that guard was never ported. Safe as long as token
ids stay under `2^24` (Float32's exact-integer range) — true for every vocab size this port
has ever used (BPE's ~11711 included). Locked in, not fixed: a real dtype-preservation
refactor would touch every clamp call site for a risk that doesn't exist at any realistic
vocab size. Revisit only if a vocab size anywhere near `2^24` becomes plausible.

**X-02 — no `"mask"` graph slot on `TransformerBlock`.** Upstream exposes a `"mask"` slot:
an arbitrary, per-sample `(batch,1,S,S)` mask clamped as its own graph node, with a
mask-derived per-position variance compensation. Julia's `TransformerBlock` instead takes a
constructor-time `causal::Bool`, applying a fixed, data-independent additive mask
(`_tb_causal_mask`) baked in at graph-construction time — deliberately, so it's a
Reactant-traceable compile-time constant rather than a traced input (see J-01/J-03's
Reactant-tracing constraints elsewhere in this file). Consequence: arbitrary padding masks,
block-sparse attention patterns, and upstream's two-slot (`"in"` + `"mask"`) topology are all
inexpressible here. Adequate for autoregressive LM training (the only topology this port
currently trains) — revisit if a padding-mask use case (e.g. variable-length batches) shows
up.

**X-02 vindication (2026-07-14).** Upstream's own `CHANGELOG.md` `[Unreleased]` documents a real
generation bug this design is structurally immune to: `generate_autoregressive` on v1 graphs that
declare an external `causal_mask` node never clamped the mask during generation, so attention ran
with a mask latent left over from state initialization instead of the lower-triangular pattern used
in training/eval — upstream's fix assembles the clamp in one shared helper
(`causal_mask_clamps(structure, batch_size, seq_len)`) now called from every consumer, including the
generation path that had been missing it. This bug class requires an external mask NODE that some
caller can forget to clamp. Confirmed by reading `get_slots(::TransformerBlock)` directly
(`src/nodes/transformer.jl:75`): `Dict("in" => SlotSpec("in", false))` — one slot, no `"mask"` entry
— matching a whole-file grep for `mask` (only `causal`/`_tb_causal_mask`/`cmask`, all internal,
compile-time-constant additive masks baked in at graph construction, never a clamped graph node).
Julia's own `generate_autoregressive`/`_generation_step` (`src/training/train_autoregressive.jl:361`
and helpers) has no causal-mask clamp branch at all — nothing analogous to upstream's omitted
`causal_mask_clamps` call exists to omit. Immune by construction, not by a fix that could later
regress.

**X-03 — `rope_theta` hardcoded to `10000`.** Upstream exposes it as a constructor
parameter; Julia's `_tb_rope_tables` hardcodes the standard value. Trivial to parameterize
(thread a `rope_theta` kwarg through `TransformerBlock`/`MhaResidualNode`/`_tb_rope_tables`)
whenever a caller actually needs a non-default value — no design constraint blocks it, just
never requested. Parameterize opportunistically, not preemptively.

**X-04 — `num_heads` default differs (upstream 8, Julia 4).** A differential-testing
tripwire, not a real divergence: every fixture/test/example in this port sets `num_heads`
explicitly, so the differing DEFAULT never silently activates. Convention going forward:
always pass `num_heads` explicitly in new tests/examples rather than relying on either
side's default.

**X-05 — `GraphNamespace` (upstream's thread-local hierarchical node-naming scope,
`core/topology.py`) never ported. RESOLVED, not just deferred**: grepped upstream's own
real multi-block assembly, `examples/transformer_demo.py:187,191`
(`create_transformer_lm`, the function `src/models/transformer_lm.jl`'s own header comment
says it adapts) — it does NOT use `GraphNamespace` either. Per-block node names are built
with plain f-string suffixes: `name=f"transformer_{i}"`, `name=f"skip_{i}"`. Zero
`GraphNamespace`/`namespace` references anywhere in `transformer_v2.py` or
`transformer_demo.py` (grep-verified). Julia's `transformer_lm.jl` already does the
identical thing (`"transformer_$i"`, `"skip_$i"`) — matching upstream's OWN real-world
practice, not just avoiding a feature upstream happens to expose. Closes R-04's
GraphNamespace half; R-04's "stage-boundary" half is independently closed by Tier B's
byte-level fixture verification of `MhaResidualNode`/`LnMlp1Node`/`Mlp2ResidualNode` against
upstream (section 6 of the register) — a boundary mismatch would have shown up as a
fixture-comparison failure, and none did. **R-04: CLOSED.**

## 22. Layer map (A-02): what "layer" means in this codebase, and the gating rule

Date: 2026-07-13

Four distinct execution paths exist for the same PC computation, informally called
"layers" in review discussion without ever being written down in one place:

- **Layer 0 — eager Dict oracle.** `Dict{String,NodeState}`/`Dict{String,Matrix}`,
  string-keyed lookups, `merge`-rebuilt each step. `run_inference`/`train_step`
  (`src/core/inference.jl`, `src/training/train.jl`). Not traceable by Reactant. This is
  the CORRECTNESS reference everything else is validated against (including upstream
  conformance — Tiers A/B/C/D all fixture against what this layer, or upstream's JAX
  equivalent of it, produces).
- **Layer 1 — flat/positional.** Position-indexed `Vector`s, no Dicts in the hot loop
  (`CompiledPlan`/`flat_run_inference`, `src/jit_flat.jl`). Validated bit-identical to
  Layer 0 (`test/test_jit_flat.jl`). Still eager Julia, not yet compiled — the intermediate
  representation Reactant traces.
- **Layer 2 — Reactant/XLA compiled.** Layer 1 traced through `Reactant.@compile`
  (`ext/FabricPCReactantExt.jl`'s `compile_inference`). Forward-inference JIT is real and
  validated (`examples/jit_inference.jl`, ~9× on the MNIST-shaped MLP). Compiled GRADIENTS
  (the J-01/J-02 arc) hit a real blocker: eager `Enzyme.autodiff` and
  `Reactant`-compiled `Enzyme` cannot coexist in one process (§15 below) — unresolved,
  parked.
- **Layer 3 — muPC.** Not a separate execution path — a per-edge SCALING applied at any of
  the layers above (`core/mupc.jl`/`core/scaling.jl`), orthogonal to which layer is running.
  `scaling_config === nothing` ⇒ identity (§6).

**Gating rule** (why this matters beyond taxonomy): each layer is validated against the one
below it, never against upstream directly except Layer 0. Concretely: Layer 1 is fixtured
against Layer 0 (this port's own eager code), NOT against upstream JAX; Layer 0 is what gets
fixtured against upstream (Tiers A/B/C/D). This keeps failures localized — a Layer 1 test
failure means "the flat lowering broke," not "the port diverged from upstream," and vice
versa. Corollary: never land two layers' worth of change in one commit — if a fix touches
both the eager math AND the flat/compiled mirror of it, that's evidence the fix belongs at a
shared call site instead of being duplicated per layer.

## 23. Backend roles: Zygote / Enzyme-eager / Reactant+Enzyme — who does what

Date: 2026-07-13

Three distinct roles get conflated under "autodiff backend" in casual discussion; writing
down which one is which, synthesizing #19 (why Zygote was chosen for the seam) with what
today's J-01 scouting pass (docs/AUDIT_REGISTER.md section 5) found about the Reactant side:

- **Zygote — the production seam backend.** `nodes/autodiff.jl`'s generic Phase-D seam
  (any node implementing only `compute_mu` gets `forward_and_latent_grads`/
  `forward_and_weight_grads` for free) runs on Zygote in this codebase's actual test suite
  and every shipped example — chosen in #19 because Enzyme aborts on the full multi-head
  attention block on Julia 1.12 (`addToDiffe: unhandled accumulate with partial sizes`).
  This is EAGER, Dict-based (Layer 0) — Zygote cannot trace `TracedRArray`s, so it has no
  role inside a Reactant-traced region at all.
- **Enzyme (eager) — opt-in, simple/dense nodes only.** The Enzyme extension implements the
  identical seam hooks and is retained for nodes that don't hit the MHA crash (simple/dense
  nodes, per #19). **🔴 CORRECTION (2026-07-14, `docs/AUDIT_REGISTER.md` F-04 reopened):**
  this entry previously claimed F-04's guard (`_register_ad_backend!`) "enforces exactly one
  of Zygote/Enzyme loaded per session." It did not — it was inert in every real session (bare
  top-level code, never replayed past the extension's own precompile; fixed 2026-07-14,
  `68a6154`) and, even after the fix, only LOGS the conflict to stderr rather than preventing
  it (Julia's extension-loading machinery swallows `__init__` errors; the competing method
  definitions are already installed before `__init__` runs regardless). **Zygote and Enzyme
  silently last-wins override each other's seam dispatch exactly as the original F-04 finding
  described — this was never actually fixed, only believed to be.** Not usable together with
  Reactant+Enzyme in the SAME process (see next point) — effectively a separate, standalone
  eager lane, not a stepping stone to the compiled lane.
- **Reactant + Enzyme (compiled) — the intended production perf path, gradients narrower than
  originally scoped (§15 update, 2026-07-14).** Reactant uses Enzyme-MLIR internally for
  autodiff under `@compile` — this is a THIRD, distinct thing from eager Enzyme above, despite
  sharing a package name. Forward compilation (no gradients) works today and is validated
  (`compile_inference`, Layer 2 above — and now benchmarked cross-language, ~7-8× vs real
  `jax.jit`, `docs/AUDIT_REGISTER.md` J-04). Gradient compilation (J-01): the ORIGINAL framing
  ("loading eager Enzyme poisons a subsequent compile") turned out to conflate eager-Enzyme
  specifically with AD-generally — Zygote-eager + Enzyme-only-inside-`@compile` (the raw-array
  `wgrad` pattern) is execution-verified to coexist cleanly, no poisoning. But `using Reactant`
  alone (a hard Enzyme dependency, loaded transitively, no explicit `using Enzyme` needed)
  silently triggers the SAME dispatch-override hazard described above — so the real practical
  danger was never really about eager-Enzyme discipline, it's about `using Reactant` at all in
  a session that also needs the real seam on Zygote-dependent nodes.

**Practical rule (revised 2026-07-14)**: a session either uses eager Zygote (Layer 0, the test
suite's own convention) on Zygote-seam-dependent nodes, OR loads Reactant (Layer 2) — never
both, for any node that goes through the real `_ad_param_grads`/`_ad_latent_grads` seam
dispatch (not just "never call `Enzyme.autodiff` eagerly" — merely `using Reactant` is enough
to silently flip that dispatch to Enzyme's implementation, per the corrected entries above).
The raw-array/`wgrad`-style bypass pattern (differentiate concrete arrays directly, never go
through the seam) remains safe in either regime — it's how the one working Reactant+Enzyme
gradient precedent (`benchmark/transformer_jit.jl`'s `wgrad`) avoids the hazard, not by
avoiding `using Enzyme` as such. This is enforced by discipline/convention only, not code — a
real fix routes seam dispatch through a Ref-held function pointer the F-04 guard installs only
after its check passes, instead of two competing top-level methods; scoped but not attempted
this session (`docs/AUDIT_REGISTER.md` F-04).

## 24. PC-relaxation stability: `eta_infer` is bounded by the graph's own conditioning, and
`transformer_lm()`'s default sits well past that bound

Date: 2026-07-14

> **CORRECTED by §26 (same date, later same session):** this section's `lambda_min≈2.3`/
> `kappa≈49` was measured with a power iteration too short to converge on this graph's
> near-degenerate spectrum. A properly-converged (renormalized, ~400-step) power iteration
> finds `lambda_min≈0.17`, `kappa≈668`. This section's QUALITATIVE conclusions (expansive at
> `eta=0.1`, Tier D correctly scoped to `eta=0.01`, the graph is stiffer than `infer_steps=12`
> accounts for) are unchanged and reinforced, not overturned — only the specific `lambda_min`/
> `kappa` numbers below are stale. Use §26's corrected values in any new derivation.

Closing Tier D's transformer-LM conformance track (`docs/AUDIT_REGISTER.md` §6) required
diagnosing why a real 12-step relaxation on the assembled `TransformerBlock` graph would not
reproduce across two independent float32 implementations at any reasonable tolerance, even
with port fidelity independently established (Tier B 151/151). The PC inference loop
(`inference_step`'s `zero_grads → forward_value_and_grad → update_latents`, iterated
`infer_steps` times) is literally gradient descent on energy w.r.t. latents: one step is
`z ← z − η·∇E(z)`, i.e. the local linearization is `z_{n+1} = (I − η·H)·z_n` for the local
Hessian/curvature `H`. This iteration is **contractive** (perturbations — including ordinary
float32 rounding noise — shrink each step, so independent implementations converge toward
agreement) iff `η < 2/λmax(H)`, and **expansive** (perturbations grow, compounding rounding
noise exponentially regardless of implementation correctness) otherwise. This was not merely
inferred — measured two independent ways (direct perturb-and-track amplification, and
power-iteration on the FD-linearized one-step Jacobian; the two agreed to within a few percent
on every graph tested) on two graphs:

- **MNIST-MLP** (`x(784)→h(128,tanh)→y(10)`, `eta_infer=0.1`, the same η as below): `λmax≈1.0`,
  contractive (measured κ≈0.90–0.902/step). Well-conditioned at its own default.
- **The tiny transformer-LM diagnostic graph** (`embed_dim=8, num_heads=2, num_blocks=1` — the
  Tier D conformance fixture's own config, NOT a production scale): `λmax≈113`, stability
  boundary `η*≈0.0176`. At the fixture's `eta_infer=0.1` (`transformer_lm()`'s own default),
  the linearized map's spectral radius is `ρ≈8.80` — **~5.7× past its own stability limit** —
  and directly-measured amplification confirms it operationally: ~1.24–1.27× per step,
  compounding to ~25,000–45,000× over 12 steps for either of two tested perturbation
  magnitudes.

**The Hessian is stiff, which matters beyond the stability boundary itself.** The
slowest-decaying mode's own measured contraction rate at a safe `eta_infer=0.01` is only
`ρ≈0.977`/step (implying that mode's `λ≈2.3`) — condition number `λmax/λmin≈113/2.3≈50`. Even
at the best possible stable η, the slow mode only contracts `~0.96–0.98`/step, so **12 steps
buys `0.96¹²≈0.6` of relaxation, not convergence to the energy minimum** — real relaxation of
a graph this stiff needs on the order of 50–300 steps, not upstream's own
`infer_steps = 3·(2·num_blocks+2) = 12` heuristic (`transformer_lm.jl`, `examples/
transformer_demo.py`'s `create_transformer_model`). This reframes the original 146/175
conformance gap: it was measuring whether a 12-step-truncated relaxation of a stiff,
unpreconditioned map reaches a state two independent float32 implementations *could* agree on
(it could not, by construction) — not primarily whether the port matches.

**This is upstream's own default, confirmed, not a Julia-introduced value.**
`examples/transformer_demo.py`'s `create_transformer_model` (the monolithic-`TransformerBlock`
family `transformer_lm.jl` actually ports) defaults `eta_infer=0.1` in both its CLI argument
and function signature. Worth reporting upstream if confirmed at production scale (this
session's η* is tiny-diagnostic-config-only and explicitly not claimed to transfer — a
production-scale sweep, e.g. on `char_lm_pc.jl`'s real dimensions, is the natural follow-up).
Suggestively — not conclusively, since it's a different node family (`MhaResidual`/`LnMlp1`/
`Mlp2Residual`, not the monolithic `TransformerBlock`) at a different scale (`embed_dim=64,
depth=2`) — `examples/transformer_v2_demo.py`'s `CHAR_DEFAULTS.eta_infer =
0.0174852165627398` was arrived at via what its own comment calls "Phase 2 refined lr/
eta_infer/infer_steps" hyperparameter search, and sits within 0.1% of this session's
independently-measured `η*≈0.0176` for an unrelated tiny config — consistent with upstream's
own tuning having empirically rediscovered a stability boundary of this same general kind,
without naming it as such.

**Practical consequences, not just a diagnostic curiosity:**

- `transformer_lm()` run at its own default (`eta_infer=0.1`, e.g. the Shakespeare/char-LM
  demo path) trains — nonlinear saturation (softmax, LayerNorm, GELU) bounds the divergence a
  purely linear analysis would predict as unbounded — but its converged latents are not
  expected to be reproducible across BLAS threadings, hardware, or (as this investigation
  found) languages. Tier D's transformer-LM conformance closure (§6) is scoped to
  `eta_infer=0.01`, NOT the production default; that scope limit is stated explicitly in the
  register precisely so it doesn't get inferred away.
- **A C-04 (PC-vs-backprop comparison harness, §4) arm built on a TRANSFORMER at
  `eta_infer=0.1`** must not run without first checking that config's own conditioning —
  otherwise the comparison benchmarks a non-converged inference loop, a confound sitting
  directly under any resulting fidelity or performance claim. Scoped to the transformer only:
  the MNIST-MLP config is contractive at its own `eta_infer=0.1` (this section's own control
  measurement, κ≈0.902) — an MNIST-MLP C-04 arm carries no such confound and is runnable
  today, unconfounded, without any eta fix first.
- A production-scale `eta_infer` sweep (this session's was tiny-diagnostic-config only) is a
  reasonable, cheap follow-up — same method (perturb-and-track + power-iteration
  cross-check), applied to `char_lm_pc.jl`'s or `decomposed_lm_pc.jl`'s real dimensions.

**Does muPC (a diagonal per-edge preconditioner on this exact iteration matrix) fix the
conditioning?** Measured directly, same methodology, same graph (`docs/AUDIT_REGISTER.md`
section 4, C-09): yes, but only ~16% (`λmax/λmin` ~49→~41), and not via `λmax` — muPC leaves
the dominant instability essentially untouched (<1% change) because `compute_mupc_scalings`
only reaches two edges PERIPHERAL to `TransformerBlock` in this topology (`embed→
transformer_0:in`, and `skip_0→output:in` only under `include_output=True`); the measured
`λmax≈113` lives inside the block's own internal attention/FFN Jacobian, which per-edge
scaling has no mechanism to reach. Real but modest: ~12.5→~10.5 steps to the same relaxation
quality. This is scoped to the monolithic `TransformerBlock` family only — the decomposed
family (`MhaResidualNode`/`LnMlp1Node`/`Mlp2ResidualNode`) exposes attention/FFN internals as
separate PC nodes with their own edges, so muPC could plausibly do much more there; not yet
measured (C-09's other, still fully open half).

**The negative result, stated in the words someone is likely to get wrong otherwise:**
`λmax` unchanged ⇒ `η*` unchanged at ~0.0176 ⇒ **muPC does NOT fix the η=0.1 instability
above** — `transformer_lm()`'s default is 5.7× past its own stability limit with muPC on or
off. "muPC improves conditioning, therefore it fixes the η problem" is the natural inference
from the 16% number alone, and it's wrong for this node family. What the 16% actually is:
`λmin` rose (~2.31→~2.74) while `λmax` sat still — muPC is lifting the SLOW modes, consistent
with what per-edge `forward_scale`/`topdown_grad_scale` factors are actually built to do (O(1)
activation-variance/gradient-scale correction across width/depth, μP-style), not curvature/
Hessian conditioning. Fixing `η*` itself would need something that moves `λmax`, which this
mechanism structurally cannot reach on the monolithic family (see above).

## 25. Reshaping the J-02/J-03/J-04 plan: the compiled inference loop doesn't scale yet,
the compiled lane has a silent-optimizer-bug landmine, and how to benchmark honestly

Date: 2026-07-14

> **UPDATE (§26, same date, later same session):** point 1's step-count figures below
> ("~50-300 steps") were derived from §24's original `lambda_min=2.3` measurement, since
> corrected in §26 to `lambda_min≈0.17`. The corrected numbers are WORSE, not better:
> `InferenceSGDMomentum` at its own derived-optimal hyperparameters needs **~89 steps** to
> reach `1e-3` residual (not the ~24 steps a first, pre-correction pass through this
> derivation estimated), and plain `InferenceSGD` at its own best stable η needs **~2300
> steps** (not "50-300"). `@trace while` is therefore not an optional upgrade for a
> future, larger-scale run — it is a HARD REQUIREMENT for any compiled inference loop on
> this graph family that actually wants to reach convergence: an 89x-unrolled StableHLO
> graph (momentum, the BETTER case) is already not a viable `@compile`-time unrolling
> target, and 2300x (plain SGD) is not remotely close to one. J-04's own 20-step benchmark
> result is unaffected either way (§6/J-04, scope explicitly stated as inference-only,
> non-convergent-config) — this changes what J-02/J-03/a future larger-scale J-04 must do,
> not what J-04 already measured.

Three findings, made reviewing J-04's freshly-closed inference benchmark (§6/J-04) against
what §24 just established the transformer relaxation actually needs. None of these
invalidate J-04's own result (its 20-step config is well inside the range where the first
finding is invisible) — they change what J-02/J-03/a future larger-scale J-04 need to do.

**1. `flat_run_inference` (`src/jit_flat.jl:364-365`) UNROLLS under `Reactant.@compile` — not
a hypothesis, read the loop: `for _ in 1:plan.inference.infer_steps`, no `@trace`.** Reactant's
own docs are explicit that a native Julia loop over a compile-time-known trip count is
executed AT TRACE TIME and vanishes from the traced program — each iteration becomes its own
copy of the loop body baked into the StableHLO graph. At 12-20 steps (Tier C/D's fixtures,
J-04's benchmark) this is invisible — a 20x-unrolled graph compiles fine and the numbers are
correct. **§24 originally estimated this transformer's own relaxation needs ~50-300 steps at
a stable η to actually converge — since corrected (§26): momentum needs ~89 steps, plain SGD
needs ~2300 steps, both derived from the corrected `lambda_min≈0.17` spectrum, not the
original ~50-300 estimate.** Either way, the compiled inference lane, exactly as built, does
not scale to a meaningfully-converged compiled run — the corrected numbers make this MORE
true, not less.

**The fix, and it's a genuine upgrade over upstream's own approach, not just a workaround:**
Reactant's `@trace while` (see the package's own Sinkhorn-iteration tutorial, structurally
identical to a PC relaxation loop — iterate a local update until a traced convergence
condition, not a fixed count) compiles to `stablehlo.while` with a TRACED termination
condition, i.e. relax-to-tolerance instead of relax-for-N-steps. Upstream's own
`jax.lax.fori_loop` (`fabricpc/core/inference.py`, what Tier A-D's Python fixtures actually
call) has a FIXED trip count — it has the exact same limitation FabricPC's current
`flat_run_inference` does, upstream never solved this either. `@trace while` is available to
FabricPC as a real path forward that goes beyond upstream's own compiled-inference design, not
merely catching up to it. The real implementation work is restructuring the loop-carried state
(`Vector{NodeState}` mutation) into the tuple/pytree shape `@trace while` needs — `jit_flat.jl`'s
existing flatten/repack machinery (`FlatNodeParams`, `flatten_param_arrays`, §11 increment 2)
is already built around exactly this kind of transformation, so this is an extension of an
established pattern, not a new one.

**A citable, quantitative positioning argument, sourced from the compiler's own
documentation, not asserted:** Reactant's AD tutorial documents that differentiating through a
traced loop has a real, explicit cost — O(N) memory by default (every iteration's
intermediates saved for the reverse pass), O(√N) with checkpointing (trading recomputation
for memory, their own Periodic/Binomial checkpointing schemes), and — the sharper limit —
loops with a DYNAMIC (not compile-time-fixed) trip count "cannot be AD'd on all platforms"
due to XLA's dynamic-shape restrictions. This is upstream/backprop's actual structural
problem with a convergence-based (as opposed to fixed-N) relaxation loop: if you need to
differentiate THROUGH the loop, a dynamic trip count fights the compiler. **FabricPC's local
PC learning rule pays none of this cost** — weight gradients (`compute_local_weight_gradients`)
are computed AT THE CONVERGED STATE only; nothing in this codebase's design ever
differentiates through the relaxation loop itself (the weight loop is local per-node energy,
not backprop through inference — this is the whole architectural distinction the docstrings
in `src/nodes/autodiff.jl` already make). So PC can freely use a convergence-based `@trace
while` relaxation (cheap: only the FORWARD pass needs to run to convergence, no reverse-mode
memory cost scales with iteration count) in a regime where a backprop-style method
structurally cannot without paying real memory/compile-time cost or hitting a hard platform
limitation. This is a genuine, citable, compiler-documented axis of comparison for whatever
positioning material eventually comes out of C-04/J-04 — not marketing, an XLA-level
structural fact.

**2. Optimisers.jl on the compiled lane closes a real landmine AND a real audit gap, and it's
already sitting there unused.** Plain Julia scalars are compile-time CONSTANTS under Reactant
unless explicitly tracked (`track_numbers=true` on `Reactant.to_rarray`, per the package's own
FAQ). This codebase's hand-rolled AdamW (`src/training/`, read its actual step-counter field)
increments a step counter `t` every call for its bias-correction term; under `@compile` that
`t` would be frozen at its TRACE-TIME value — **every compiled training step would silently
apply step-1's bias correction, forever, with no error, no warning, just wrong optimizer
behavior** — a J-02/J-03 landmine to design around before, not after, building the compiled
`train_step`. Separately (same mechanism, already relevant NOW): `InferenceSGD`'s
`eta_infer::Float32` is also a plain scalar — also frozen under `@compile` — meaning any
compiled inference path currently needs a fresh compile per η value. §24 just established that
sweeping η is now a real, recurring need (production-scale sweeps are the natural §24
follow-up), so this stops being a hypothetical the moment that sweep gets built.

The fix for the optimizer half is not "add `track_numbers=true` to the hand-rolled AdamW" —
it's **use `Optimisers.jl` instead of the hand-rolled implementation on the compiled lane**.
`Optimisers.jl` is ALREADY a regular (non-weak) dependency (`Project.toml`'s `[deps]`) and
already resolved in the Manifest at a version (0.4.7 locally) that ships
`OptimisersReactantExt` — Reactant-tracing support built in, maintained upstream, not
something this codebase needs to hand-roll compatibility for. Its state-threading design
(`state, params = Optimisers.update(state, params, grads)`, functional state in/out) is
already exactly the shape `@compile` needs. Switching the compiled lane to
`Optimisers.AdamW`/`Optimisers.setup` retires the `t`-frozen-at-trace-time hazard AND
independently closes a gap the original upstream-conformance audit named: upstream trains via
`optax` (a generic, swappable optimizer library), while this port's PC training path only ever
had one hand-rolled AdamW implementation — using `Optimisers.jl` (Julia's own generic,
swappable optimizer library) on the compiled lane is the closer structural match, not just a
Reactant-compatibility fix.

**3. Benchmark on CPU; a GPU comparison needs matched XLA flags or it's not honest.** Upstream's
own `jax_setup.py` (imported at the top of every one of upstream's example/demo scripts) sets
three XLA flags: `xla_gpu_deterministic_ops=true`, `xla_gpu_enable_triton_gemm=false`,
`xla_gpu_autotune_level=1` — every one of them REDUCES performance, chosen deliberately for
reproducibility over speed. All three are `xla_gpu_*` — they do nothing on CPU. J-04's
benchmark ran on this session's own CPU-only 2-core machine, so it was accidentally already
fair (neither side paid or avoided these flags, because neither side is on GPU). **The
discipline to carry forward, not a fix to anything currently wrong**: a future GPU comparison
of Reactant against `jax.jit` MUST either apply the same three flags to the JAX side being
compared against (matching upstream's own actual reproducibility-tuned configuration) or
explicitly set matching `XLA_FLAGS` on the Reactant side (Reactant reads the same environment
variable — it's the same underlying XLA) — running Reactant with default (fast) flags against
JAX with upstream's deliberately-slowed flags would be a false win any competent reviewer
would catch immediately by checking `jax_setup.py`. If a future benchmark's numbers look
surprising either direction, `@code_hlo` (Reactant) vs JAX's `.lower(...).as_text()` dump
comparable StableHLO/HLO IR from both frontends into the SAME backend — a slow result becomes
debuggable in the IR instead of mysterious, since both sides lower to the same compiler.

**ROADMAP intel, for awareness only (upstream `docs/dev_plans_archive/ROADMAP.md` Phase 8.1):**
upstream's own planned direction for their transformer stacks is Flax's `nn.scan` — one compiled
block, scanned across the LAYER axis (`num_layers`), rather than unrolled. This is the SAME
unrolling problem this section just found for `flat_run_inference` (point 1 above), but on the
layer axis instead of the inference-step axis — upstream hasn't solved either axis yet
(`nn.scan` is Phase 8, unbuilt), and neither has this port. Not urgent, but worth remembering
if/when a multi-block `transformer_lm()` graph gets compiled: `num_blocks` unrolling under
`@compile` is the same class of problem as `infer_steps` unrolling, and would want the same kind
of fix (a traced loop over the block axis, if Reactant has an equivalent scan primitive) rather
than being rediscovered independently.

## 26. `InferenceSGDMomentum`: shipped ahead of upstream, correctly — and the verification pass
that chased its own mismatch corrected §24's own spectrum measurement in the process

Date: 2026-07-14

**What this is.** `src/core/inference.jl`'s `InferenceSGDMomentum` (heavy-ball momentum:
`v(t+1) = momentum·v(t) - eta_infer·grad; z(t+1) = z(t)·(1-eta_infer·latent_decay) + v(t+1)`)
implements upstream's own PLANNED design (`docs/dev_plans_archive/momentum_sgd_inference_plan.md`
in the upstream checkout, as of the commit this port tracks) before upstream has shipped it —
`fabricpc/core/inference.py` still has only `InferenceBase`/`InferenceSGD`/
`InferenceSGDNormClip`. This is registered as an **ahead-of-upstream divergence, not a
conformance-gap fill**: there is no reference output to port against, so its correctness rests
on a conformance anchor (below), not a fixture diff. Per upstream's own plan, it overrides the
inference LOOP rather than the per-node `compute_new_latent` hook, because velocity must be
carried across steps — Julia's version of this is `_run_inference_loop` dispatching on
`structure.config.inference`'s type (a new `momentum_inference_step` function is the single
source of truth for one heavy-ball step; both `_run_inference_loop`'s internal loop and any
external caller — tests, diagnostics, a future `@trace while` compiled loop — must go through it,
not `inference_step`/`compute_new_latent`, which is a documented plain-SGD fallback for this type
that silently ignores `momentum`).

**Conformance anchor (the part with no room for interpretation):** at `momentum=0f0`, the
recursion collapses to `v(t+1) = -eta·grad`, i.e. bit-for-bit `InferenceSGD.compute_new_latent`'s
own formula. Tested directly: 20/20 fields bit-exact across all 4 non-clamped/clamped nodes over
12 steps (`test/test_inference_momentum.jl`).

**The interesting part is what happened chasing the theory-vs-measurement closure, not that it
closed cleanly on the first try.** §24 measured this exact graph's (`transformer_lm(embed_dim=8,
num_heads=2, num_blocks=1)`, same fixture params/batch as `tier_d_transformer_stable`) Hessian
spectrum as `lambda_min≈2.3, lambda_max≈113.6` (`kappa≈49`), and this session derived Polyak
heavy-ball's optimal hyperparameters from it: `eta*=4/(√λmax+√λmin)²≈0.027`,
`beta*=((√λmax-√λmin)/(√λmax+√λmin))²≈0.56`, predicting an asymptotic per-step rate
`√beta*≈0.751`. **Measured (natural-relaxation trajectory, 80 steps): ~0.99/step — a real
mismatch, not a rounding difference.** Rather than accept either number, the mismatch was
chased with the SAME perturb-and-track methodology §24 itself used (a joint `(z,v)`-state
power iteration for the momentum case), with one addition: **periodic renormalization** (rescale
the tracked perturbation back to `eps` every 10 steps, standard power-iteration hygiene against
numerical underflow over long runs). This surfaced two things:

1. **§24's own `lambda_min=2.3` was measured with too few power-iteration steps to converge on
   this graph's near-degenerate spectrum.** A SHORT (5-10 step) power iteration gives a ratio
   ≈0.977-0.98 — matching §24's number closely — but the ratio keeps DRIFTING (monotonically,
   not noise) for hundreds more iterations before truly plateauing. A renormalized 400-step
   power iteration on plain SGD (`momentum=0`, `eta=0.01`, the exact config §24 measured)
   converges to **≈0.998/step**, implying **`lambda_min≈0.17`, not 2.3** — the graph's TRUE
   condition number is **`kappa≈668`, not ≈49**. Power iteration converges to whichever mode has
   the LARGEST eigenvalue MODULUS among what's being tracked; an under-run power iteration on a
   near-degenerate spectrum reports an intermediate, still-decaying ratio that LOOKS converged
   (it's stable to 2-3 decimal places over a handful of steps) but isn't. This is a real,
   general methodological lesson for every perturb-and-track measurement in this project, not
   specific to momentum — the fix is checking two non-adjacent windows late in a long run agree
   to within noise (this session used windows 291:310 vs 381:400, agreeing to 5e-5) before
   trusting a power-iteration rate, not trusting a rate that merely LOOKS stable over the first
   10-20 steps.
2. **`lambda_max=113.6` independently re-confirmed correct — but ONLY near initialization.**
   Direct stability-boundary crossing (plain SGD, no perturb-and-track needed): at `eta=0.017`
   the near-init trajectory contracts (ratio≈0.973/12-steps); at `eta=0.02` it flips to GROWTH
   (ratio>1) — bracketing `2/lambda_max≈0.0176` almost exactly. But power-iterating from a state
   burned 40 steps into relaxation, NO instability appears even at `eta=0.025` (safely past the
   near-init boundary) — the local Hessian is **trajectory-dependent**, not a fixed global
   quadratic. LayerNorm/softmax/GELU curvature genuinely varies between the near-initialization
   region (stiff, `lambda_max` large) and a partially-relaxed region (much gentler locally). A
   single global `(lambda_min, lambda_max)` pair is therefore an ENGINEERING CHOICE (worst-case
   `lambda_max` anywhere on the trajectory for stability, worst-case `lambda_min` anywhere for
   convergence speed), not a clean fit to a real fixed quadratic.

**Corrected derivation** (`lambda_min=0.17`, `lambda_max=113.6`): `eta*≈0.0326`, `beta*≈0.857`,
naive predicted rate `√beta*≈0.926`. Re-measured (same renormalized perturb-and-track
methodology): **momentum converges to ≈0.989/step vs plain SGD's own (corrected) ≈0.998/step —
real, if modest, and NOT matching the naive quadratic-model prediction closely either.** This
is the honest result: the graph is not a single quadratic, so heavy-ball's classical closed-form
optimality guarantee (which assumes one) does not transfer cleanly — but real, if modest, is
still real. Separately, because `eta*≈0.0326` sits almost exactly AT its own stability boundary
`2(1+beta*)/lambda_max≈0.0327` (using the near-init worst-case `lambda_max`), running it from an
actual cold start (not perturb-and-track — the real `momentum_inference_step` loop) is
NOT smooth: a genuine transient — with the bug described below fixed, the corrected trajectory
in fact converges cleanly without overshoot on this particular graph+params (`step1=4.75 →
step10=3.01 → step60=0.27`, no spike) — the overshoot described in an earlier draft of this
investigation turned out to be an artifact of the bug below, not a property of the corrected
hyperparameters themselves; kept here as a documented false lead, not a current risk.

**A real bug this investigation found and fixed, in the test file, not the implementation:**
early falsification/rate tests called `FabricPC.inference_step` directly to step through
momentum trajectories. `inference_step` → `update_latents` → `compute_new_latent`, and for
`InferenceSGDMomentum` that's the documented plain-SGD FALLBACK (mirrors upstream's own plan,
which also documents its `compute_new_latent` override as "not used in practice") — it silently
ignores `momentum` entirely. Symptom: `beta=0.9` and `beta=0.99` trajectories were bit-identical
to 15 decimal places (both were secretly plain SGD at `eta=0.1`). Root cause identified by
noticing the impossible coincidence, not assumed away. Fix: extracted `momentum_inference_step`
as the single source of truth (`src/core/inference.jl`) that both `_run_inference_loop` and every
step-by-step caller (tests, diagnostics) now use — the momentum=0 conformance anchor was
unaffected throughout (it correctly goes through `run_inference`), and the perturb-and-track
spectrum diagnostics were unaffected too (they hand-rolled the correct velocity update
independently of `inference_step`) — only the falsification test and the natural-relaxation
sanity check were silently testing the wrong algorithm. Re-measured with the fix: falsification
holds (below); the "corrected-optimal" cold-start trajectory turned out BETTER-behaved once
fixed (clean convergence, no overshoot) than the buggy version had shown.

**Falsification (unaffected by the lambda_min correction — lambda_max was not revised):**
upstream's own proposed default (`InferenceSGDMomentum(eta_infer=0.1, momentum=0.9)`, from their
plan doc) is predicted unstable — stability boundary at `beta=0.9` is
`eta<2(1+0.9)/lambda_max≈0.0334`, three-fold below 0.1; the ceiling as `beta→1` is
`4/lambda_max≈0.0352`, still `<0.1` — no `beta<1` stabilizes `eta=0.1` on this graph. Measured
(real `momentum_inference_step` trajectories, `beta=0.9` and `beta=0.99`, 100 steps): NOT a clean
exponential blowup — nonlinear saturation bounds it, exactly as §24 already documented for plain
SGD at this same `eta=0.1`. What IS observed: a real excursion (>2x its post-transient baseline,
peaking ≈2.4-2.7x) that settles into an ELEVATED, non-decaying oscillation, never returning below
its own baseline — the opposite sign from the corrected-optimal config, which nets a >10x
DECREASE using the identical measurement protocol. That sign difference (net growth vs net decay
from the same post-transient baseline) is the falsification test's actual assertion — a specific
growth multiplier isn't robust here because saturation makes the exact bound graph-and-config-
dependent, but the sign is not.

**Sharper framing, worth stating explicitly since the raw numbers can be misread as "upstream's
whole proposed config is bad": upstream's `beta=0.9` is fine — their `eta=0.1` is what's fatal.**
The stability boundary `eta<2(1+beta)/lambda_max` is a function of `lambda_max` ALONE — it doesn't
care what `beta` is, beyond `beta<1`. At `beta=0.9`, the boundary is `0.0334`; the derived-optimal
`beta*≈0.857` isn't meaningfully different from upstream's `0.9` (`sqrt(0.9)` vs `sqrt(0.857)` is
only a ~4% per-step rate difference, i.e. upstream's own `beta` guess costs a modest speed
penalty, not a correctness defect). The actual gap between "upstream's proposed default" and "the
derived-optimal config" is almost entirely in `eta`: `0.1` vs `0.0326`, a `3`x difference, and
`0.1` is the one number that's unconditionally past the stability boundary regardless of `beta`.
The one-line takeaway for anyone tuning this config by hand rather than deriving it: keep
momentum's `beta` near `0.85`-`0.9`, fix `eta` to respect `2(1+beta)/lambda_max` for whatever graph
it's running on — `beta` was never the problem here.

**What this closes and does NOT:**
- CLOSED: `InferenceSGDMomentum` is implemented, conformance-anchored (momentum=0 exact), and
  genuinely faster than plain SGD on this graph (measured, not assumed) — a real, shippable,
  ahead-of-upstream feature.
- CLOSED: upstream's own proposed default hyperparameters are falsified on this graph, predicted
  before measured, from the (corrected) spectrum.
- CLOSED, and arguably the more valuable result: §24's `lambda_min=2.3`/`kappa≈49` figures are
  corrected to `lambda_min≈0.17`/`kappa≈668` — anyone citing §24's condition number going forward
  should use this section's corrected value. §24's own qualitative conclusions (this graph's
  production `eta_infer=0.1` default is expansive; Tier D's transformer-LM conformance closure is
  correctly scoped to `eta=0.01`; the graph is stiffer than the untuned `infer_steps=12` default
  accounts for) are UNCHANGED and, if anything, reinforced by a worse condition number.
- NOT closed, and explicitly not claimed: a clean match to the naive fixed-quadratic heavy-ball
  theory. The graph's local curvature is trajectory-dependent; a single global `(eta*, beta*)` is
  a worst-case engineering compromise, not a tight theoretical optimum. A trajectory-adaptive
  momentum schedule (larger `beta` once past the stiff near-init region) is a natural, NOT YET
  BUILT follow-up if the modest ≈0.989-vs-0.998 speedup needs to be pushed closer to what the
  corrected `kappa≈668` would allow a truly-optimal (adaptive) scheme to achieve.
- **Free byproduct for J-02/J-03** (`docs/decisions.md` §25): `momentum_inference_step`'s
  velocity-Dict-threaded-through-the-caller design is exactly the loop-carried-state shape
  `Reactant.@trace while` needs (a tuple/pytree of `(state, velocity)` carried across traced
  iterations, not a Julia-native `for` loop that would unroll under `@compile`) — building
  momentum correctly already did the state-restructuring work §25 flagged as the real blocker
  for a `@trace while`-based compiled inference loop, for a second, unrelated feature, for free.

Tests: `test/test_inference_momentum.jl` (conformance anchor, corrected-optimal cold-start
sanity, falsification — 26/26, always run) and `test/test_inference_momentum_diagnostic.jl` (the
full renormalized perturb-and-track closure, several minutes — `FABRICPC_MOMENTUM_SPECTRUM_DIAGNOSTIC=1`,
gated the same way as the demoted Tier D eta=0.1 diagnostic).

## 27. Is `lambda_min≈0.17` a signal-carrying slow mode, or a near-flat direction? A single-vector
test failed to answer this; a block-subspace test gives a decisive (and reassuring) answer.

Date: 2026-07-14

**The question, precisely.** §26 corrected this graph's condition number from `kappa≈49` to
`kappa≈668`, implying plain SGD needs `~2300` steps (not `~50-300`) to reach a `1e-3` residual —
a number worth interrogating before it gets cited as "this graph's relaxation is badly
unconverged at 12 steps, therefore every prior FabricPC transformer measurement on this family
ran on unconverged latents." That conclusion only follows if the `lambda_min≈0.17` mode is
actually EXCITED by the relaxation this graph's real clamp pattern drives — a near-null
eigenvalue of the energy Hessian means the energy is nearly FLAT along that direction, so slow
convergence there can be practically harmless if the trajectory barely goes there in the first
place. This section measures that directly, rather than assuming either answer.

**First attempt (single-vector power iteration) failed, and said so honestly rather than reporting
a number.** The natural approach — run the SAME renormalized perturb-and-track power iteration
§26 used to find `lambda_min`, but capture the converged perturbation DIRECTION as an estimate of
`lambda_min`'s eigenvector `v_min`, then project the real relaxation's step-1 `latent_grad` onto
it — was tried first and diagnosed as unreliable before its output was trusted:
- Two independent random-seed power iterations (400 steps each, same protocol as §26) did NOT
  converge to the same direction: `cos(v_min[seed=7], v_min[seed=99]) = 0.114`, far from `±1`.
- A Monte Carlo baseline (200 genuinely random unit directions in this graph's `n=576`-dimensional
  non-clamped joint `z_latent` space, projected against the same real `g_1` signal) gives
  `mean|proj fraction|=0.036, std=0.026, 95th percentile=0.082` — and the single-vector
  measurements (`0.030`-`0.042` for one seed, `0.015` for the other) sit squarely inside that
  noise band.
- Diagnosis: `lambda_min≈0.17` is not one isolated eigenvalue — it sits inside a near-degenerate
  CLUSTER of eigenvalues close enough together that 400 steps of single-vector power iteration
  cannot separate any one of them from the others; the "direction" it converges to depends on
  which combination of near-tied modes the random start happened to weight, which is why two
  seeds disagree. Reporting a projection number from either seed as "the" answer would have been
  presenting noise as signal — caught before it went in a writeup, not after.

**Second attempt: block (simultaneous) subspace iteration, k=6, resolves the cluster instead of
one ambiguous vector.** Standard fix for near-degenerate spectra: track `k=6` perturbation
directions simultaneously (not one), re-orthonormalizing via QR every 10 steps (same 400-step,
renorm-every-10 cadence as the single-vector version) so the k-dimensional subspace they span
converges to the DOMINANT k-dimensional invariant subspace of the linearized map, without needing
to resolve individual directions within it. Two results from this:
- **The cluster is confirmed real, not an artifact.** At convergence, all 6 directions' per-window
  log-decay rates cluster tightly (`-0.013` to `-0.020` per 10-step window, i.e. `~0.9984`/step
  each) — consistent with, and independently corroborating, `lambda_min≈0.17`'s predicted
  `0.9983`/step. A single dominant mode would show one clearly larger-magnitude value among the
  6; it doesn't — six near-identical decay rates is the signature of a genuine degenerate/
  near-degenerate cluster, matching what the single-vector seed-disagreement already implied.
- **The actual relaxation signal has BELOW-CHANCE representation in this cluster.** Projecting the
  real relaxation's `latent_grad` (plain `InferenceSGD`, `eta=0.01` — the same config `lambda_min`
  was measured on) onto the resolved 6-dimensional subspace, as a fraction of total gradient
  energy: `0.47%` at step 1, rising to `0.73%` by step 60. The chance baseline a RANDOM 6-of-576
  dimensional subspace would capture is `k/n = 1.04%` — the measured fraction is BELOW that
  baseline throughout the tracked window, not elevated.

**Answer: closer to decorative than load-bearing, for this graph's actual clamp pattern.** The
slow cluster is not preferentially excited by the real relaxation signal — if anything, slightly
under-represented relative to a random subspace of the same size, at least across the first 60 of
the eventual ~2300 steps plain SGD would need for a formal `1e-3` bound. Practically: most of the
gradient's energy (>99%, by construction, since the slow cluster holds well under 1%) is being
corrected by the FAST part of the spectrum in the early steps this session has actually measured
(12-60), and the long tail past that is chasing a small-amplitude, weakly-excited residual, not a
substantial, load-bearing correction. "This graph needs ~2300 steps to formally converge" and
"this graph's relaxation is doing meaningful work for ~2300 steps" are different claims — this
section supports the first, not the second.

**What this does NOT establish, stated as plainly as the finding itself:**
- **Not a claim that the fraction stays low forever.** By construction (it IS the slowest-decaying
  subspace), its fractional share of an ever-shrinking total gradient must approach 100%
  asymptotically as faster content decays away — consistent with, and the same mechanism behind,
  §26's own earlier finding that a SHORT power-iteration window gives `~0.977-0.98`/step while a
  long one converges to `~0.998`/step. This section adds WHERE that transition sits relative to
  the signal's own energy budget (still sub-chance through step 60), not that it never happens.
- **Specific to this graph, this batch, this clamp pattern.** The energy landscape's local
  curvature and the specific clamp-driven error direction are what's being measured jointly; a
  different batch or a genuinely different graph topology is not guaranteed to reproduce this.
- **`k=6` was a reasonable, not independently proven, choice.** Picked because the resolved
  subspace's 6 decay rates cluster tightly together (evidence FOR roughly this cluster size, not
  an a priori guarantee it is exactly right); a slightly larger or smaller `k` was not swept.
- **Measured on plain `InferenceSGD`, not momentum** — matching the config `lambda_min` itself was
  measured on, deliberately, so this section answers "is the number `lambda_min` implies about
  THAT config's convergence practically meaningful," not a momentum-specific question.

Scratch scripts (not committed — this section's numbers are reproducible from them but they were
throwaway diagnostics, not permanent test infrastructure): renormalized single-vector power
iteration with an explicit two-seed disagreement check and a 200-sample random-direction Monte
Carlo baseline; block/simultaneous QR-orthonormalized k=6 subspace iteration with the same
renorm-every-10 cadence as §26's `perturb_and_track_renorm`, reusing `FabricPC.momentum_inference_step`/
`FabricPC.inference_step`/`FabricPC.update_node_in_state` (the real production code path) throughout —
not a hand-derived approximation of it.
