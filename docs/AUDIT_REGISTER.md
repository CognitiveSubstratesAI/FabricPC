# FabricPC.jl — Consolidated Audit Register

**Opened:** 2026-07-13
**Sources:** Claude fidelity audit (2 sessions: full-tree comparison vs Python 0.3.1 snapshot;
transformer differential deep-dive) + VS Code IDE agent coverage audit (5 diff agents, 68 items,
vs upstream HEAD incl. PRs #19/#28) + a reflective adversarial cross-check (6 skeptical
verification agents + synthesis) reconciling the two.
**Upstream reference:** `~/JuliaAGI/dev-zone/FabricPC` — confirmed LIVE-SYNCED to `origin/main`
HEAD `316367c` (2026-07-09) at cross-check time (see A-01). The original register's "0.3.1
snapshot" framing was itself the stale baseline — see A-01 resolution.
**Baseline verdict:** Julia port tracks upstream contract semantics (3-tuple gradients,
callsite scaling, missing-key omission). Core fidelity high. F-01/F-02 were real, currently-dormant
gradient-correctness bugs on the muPC+transformer path — both now fixed and gated by regression
tests. Coverage gaps remain concentrated in conv/pooling, JIT breadth, and Python-ecosystem infra.

**Status legend:** OPEN / BLOCKED / DECIDE / DEFERRED / VERIFIED CLOSED / IN PROGRESS

---

## 0. Pre-work

| ID | Item | Detail | Status |
|----|------|--------|--------|
| A-01 | Re-sync Python reference to upstream HEAD | **RESOLVED (no action needed).** The reflective cross-check ran `git log`/`git fetch origin main`/`git merge-base --is-ancestor` against the live checkout: HEAD is `316367c` (PR #28, 2026-07-09), `git log HEAD..origin/main` is empty (fully synced), and both `88b0b7d` (#19) and `316367c` (#28) are confirmed reachable/current commits — not references to a hypothetical future state. `pyproject.toml`'s `version = "0.3.1"` string was set 2026-05-04 and never bumped through six subsequent merged PRs, so that version string is the misleading artifact, not the actual checkout. **The original register's own baseline was ~2 months / 6 PRs stale relative to the tree it was flagging as ahead-of-audit.** | VERIFIED |
| A-02 | Write layer map into docs/decisions.md | Layer 0 eager Dict oracle / Layer 1 flat-positional / Layer 2 Reactant+Enzyme compiled / Layer 3 muPC. Rule: each layer gated by the one below; outermost gate = upstream JAX; never two layers in one commit. | OPEN |

---

## 1. Fidelity findings — correctness bugs

| ID | Sev (final) | Finding | Status |
|----|-----|---------|--------|
| F-01 | HIGH (TransformerBlock) / LOW-informational (MhaResidualNode, LnMlp1Node) | muPC LayerNorm compensation | **VERIFIED CLOSED** (`c8fb541`) |
| F-02 | MEDIUM (revised from claimed HIGH) | MhaResidualNode skip-slot not decoupled | **VERIFIED CLOSED** (`c8fb541`) |
| F-03 | MEDIUM | Optimizer key-set asymmetry | **VERIFIED CLOSED** (`c8fb541`) |
| F-04 | HIGH (revised UP then genuinely resolved, 2026-07-14) | Dual AD-backend footgun | **VERIFIED CLOSED, for real this time (round 2, `9a38178`)** — round 1 (`68a6154`) fixed the registration bug but left the actual named hazard (silent dispatch override) open; round 2 (backend-type dispatch) closes it structurally. See the write-up below (this section) and `docs/decisions.md` §15/§23 updates. |
| F-05 | MEDIUM | `compute_loss`/clamp one-hot contract divergence | **VERIFIED CLOSED** |
| F-06 | LOW (dormant) | `TransformerBlock` output-activation divergence | **OPEN** — found by Tier B (section 6) |

### F-01 — muPC LayerNorm gradient compensation — VERIFIED CLOSED

Adversarial verification confirmed the TransformerBlock claim exactly (Julia's generic
`forward_and_weight_grads` seam structurally had no path for `info.scaling_config` to reach it;
upstream's `transformer.py:392-422` performs the compensating multiply-by-`a`). The claim's
extension to MhaResidualNode/LnMlp1Node was **overstated**: upstream's own decomposed nodes have
the identical gap, explicitly flagged as an unfinished follow-up in
`dev_plans_archive/transformer_mupc_variance_plan.md:135` — not a Julia-introduced divergence, so
left unfixed (informational only).

**Fix**: `forward_and_weight_grads` gained an optional `info=nothing` kwarg (threaded through every
node, ignored by all but one); `TransformerBlock` got a new override that multiplies weight grads
by the "in"-edge `forward_scale` when `info.scaling_config !== nothing`, matching upstream's
rationale and formula. Biases are NOT scaled (matches upstream). Regression test:
`test/test_transformer.jl` "F-01: muPC LayerNorm weight-grad compensation" — asserts
`grad_scaled ≈ a .* grad_unscaled` under a real `MuPCScalingFactors`, and that `info=nothing` /
`scaling_config=nothing` are both no-ops (byte-identical to the pre-fix path).

**Reachability at close time**: dormant — no shipped example/test combined `MuPCConfig` with
`TransformerBlock`. Fixed anyway on proximate-risk grounds: upstream PR #28 (2026-07-09) just
wired `MuPCConfig` into its own transformer demo by default.

### F-02 — MhaResidualNode skip-slot decoupling — VERIFIED CLOSED

Underlying bug **CONFIRMED** end-to-end (traced through `core/mupc.jl`/`core/scaling.jl`'s
pre-scaling pipeline, not just the slot metadata): under muPC, the single `"in"` edge fed both the
LayerNorm'd attention path and the raw residual bypass, so a real `forward_scale` would incorrectly
scale the identity skip too. The **"must fix together with F-01" gating claim was REFUTED**: F-02
is a pure forward-only `compute_mu` property, and F-01 is fully isolable on `LnMlp1Node` (no skip
machinery involved) — independent, targeted fixes and tests dominate a bundled gate. **Severity
revised HIGH → MEDIUM**: the stealth/proximate-risk argument is real but was already weighed by the
original coverage audit's MEDIUM call; nothing new compelled an upgrade.

**Fix**: `MhaResidualNode` gets an optional decoupled `"skip"` slot
(`is_variance_scalable=false, is_skip_connection=true`), falling back to `"in"`'s value when
unwired (byte-identical to pre-fix behavior whenever no `"skip"` edge is wired — matches upstream's
`skip = inputs[skip_key] if skip_key else x`). Regression test:
`test/test_transformer_decomposed.jl` "MhaResidual" testset — asserts the residual comes from a
wired `"skip"` edge (not the scaled `"in"` value) and that the no-skip fallback is unchanged.

### F-03 — optimizer key-set asymmetry — VERIFIED CLOSED

**CONFIRMED-BUT-UNREACHABLE**: `sgd_update` silently drops a param key missing from `grads`;
`_adamw_dict!` throws an opaque `KeyError` on the same scenario — genuinely opposite failure modes.
Traced every producer of a gradient `NodeParams` for all 8 shipped node types: key-set equality is
guaranteed by construction in every shipped path today (for the generic AD-fallback nodes, the
gradient shadow is built by `zero`-ing the params object itself). Real hazard only at the open
`AbstractNode` extension boundary (a future custom node with sparse/lazy/masked params would hit
this with nothing to catch it).

**Fix**: `compute_local_weight_gradients` (`core/learning.jl`) now asserts
`Set(keys(grad.weights)) == Set(keys(params.weights))` (and biases) right where grads are produced,
raising a clear `ArgumentError` naming the node and the mismatched keys — one check point instead
of two divergent downstream failure modes.

### F-04 — dual AD-backend load footgun — VERIFIED CLOSED (round 2, 2026-07-14, `9a38178`)

**Original finding (still accurate)**: the trigger (`using Zygote; using Enzyme` in one session,
or vice versa) has the lowest bar of all four F-* findings, and the failure mode is uniquely
bad — a silent method-table overwrite at load time, then an opaque, non-debuggable LLVM abort at
the first call that hits code (e.g. `TransformerBlock`'s full multi-head-attention block) only
the losing backend could handle.

**The original "VERIFIED CLOSED" fix (`c8fb541`) never actually worked, and had zero test
coverage that would have caught it.** `_register_ad_backend!` (`nodes/autodiff.jl`) was called
as BARE TOP-LEVEL CODE in both `FabricPCZygoteExt`/`FabricPCEnzymeExt`, not inside `__init__()`.
Per Julia's package-extension semantics, a top-level statement mutating a DIFFERENT,
already-loaded module's global state (`FabricPC._AD_BACKEND`) only takes effect in the ephemeral
worker process that PRECOMPILES the extension — it is never replayed when the extension is later
loaded from its cached `.ji` in a normal session. Found while investigating J-01 (below), by a
direct probe in a genuinely fresh, already-precompiled process: `_AD_BACKEND[]` stayed `:none`
forever — even loading a SINGLE backend with no conflict at all never registered it. This means
the guard has likely never functioned in any real (post-initial-setup) session since it landed.

**Round 1 fix (`68a6154`)**: moved both `_register_ad_backend!` calls into
`function __init__() ... end`, Julia's documented mechanism for exactly this (runs on every
load, precompiled or not). Verified across a REAL fresh-process/precompile-cache boundary: a
fresh process loading a single backend now correctly registers it immediately post-cache-load.
**This fixed the Ref-registration bug but NOT the actual named hazard**: in the dual-load case,
the guard's `error(...)` genuinely fired inside `__init__()`, but Julia's own package-loading
machinery catches `__init__` errors and reports them via `@error` to stderr — never as a
catchable exception at the `using X` call site — and the competing extensions' top-level method
definitions were installed during module restoration, which precedes `__init__`, so the silent
override had ALREADY happened by the time the guard's error fired regardless. `methods(...)`
after a dual load showed only 2 methods, with the second-loaded backend's implementation winning
unconditionally. **Round 1 alone would have shipped a guard that logs loudly but doesn't guard.**

**Round 2 fix (`9a38178`) — BACKEND-TYPE DISPATCH, verified to actually close the hazard.**
`src/nodes/autodiff.jl` now defines `abstract type ADBackend end` + marker structs
`NoBackend`/`ZygoteBackend`/`EnzymeBackend`. Each extension's real gradient method is now keyed
on a DISTINCT backend-marker as its first argument (`_ad_param_grads(::ZygoteBackend, node,
params, inputs, z)` vs `_ad_param_grads(::EnzymeBackend, ...)`) instead of the previous identical
signature — a structurally different method, so **collision is now impossible**, not merely
guarded against: both methods coexist harmlessly in the method table regardless of load order.
`_AD_BACKEND[]` (an `ADBackend` instance) selects which one an unwrap-and-dispatch entry point
routes real calls to; `_register_ad_backend!`'s conflict check still can't propagate its
`error()` as a catchable exception (same Julia limitation as round 1) — but that no longer
matters, because the `_AD_BACKEND[] = backend` assignment sits AFTER the `error()` call in the
same function body: on conflict, the error throws (swallowed, logged) BEFORE the Ref is
touched, so it stays on the FIRST-loaded backend — protected by control flow/state, not by
exception propagation reaching the caller. A `set_ad_backend!(backend::ADBackend)` escape hatch
is exported for deliberate switching.

**Verified, not just designed**: `test/test_autodiff_backend_registration.jl` (17/17, opt-in
behind `FABRICPC_AD_BACKEND_SUBPROCESS_TESTS=1` — spawns genuinely fresh Julia subprocesses,
the only way to exercise the precompile-cache boundary this bug lives on) asserts the ACTUAL
safety claim, not just Ref bookkeeping: after a conflicting dual load, the method Julia
dispatches to for the currently-registered backend type is confirmed (via `which`) to genuinely
be `FabricPCZygoteExt`'s, not `FabricPCEnzymeExt`'s — dispatch itself now agrees with the Ref,
closing the exact disagreement round 1 left open. (One bug was found and fixed in the TEST's own
verification code along the way — an overly broad `Any`-typed `which()` call that threw
`MethodError` instead of resolving; fixed narrowly, no assertion weakened, confirmed via direct
`applicable()`/subtype probes that the design itself was never wrong.) Full suite: **2309/2309,
zero regressions** from either round's change to this core seam. **CI**:
`.github/workflows/ad-backend-guard.yml` (new) runs this subprocess test on a nightly schedule +
push-to-`main` + manual dispatch — NOT on every PR (≈7 min, spawns several cold Julia processes)
— so the tripwire built for a bug that hid undetected for over a month doesn't itself rot
unexercised the way the original guard did.

**Root-cause note for future `ext/` additions**: swept all three `ext/*.jl` files for other
top-level statements mutating `FabricPC.*` global state — only these two extensions had the
bug (`FabricPCReactantExt.jl` defines no such state). And confirmed WHY the original bug
survived Tier B/C/D's extensive `TransformerBlock`/Zygote-seam testing undetected: `test/runtests.jl`
and everything it `include`s never `using Reactant` (or `Enzyme` directly) — the dangerous
combination was simply never exercised in the main suite. A green, 2309/2309 test suite gave
false confidence for over a month; this is a real limitation of warm/in-process testing for this
whole CLASS of precompile-cache-boundary bug, not specific to this one instance.

**Severity**: revised MEDIUM → HIGH when reopened (the trigger was broader than originally
scoped — see J-01, `using Reactant` alone was enough, not just an explicit `using Enzyme`
mistake), now genuinely closed at HIGH-severity rigor: structural (type-level) prevention,
executable regression coverage, and CI wired so it can't silently rot again.

### F-05 — `compute_loss`/clamp one-hot contract divergence — VERIFIED CLOSED

**Found during C-03 implementation** (not by either prior audit pass): while scoping the char
dataloader, upstream's own loader contract (`_TokenSequenceLoader` docstring, `dataloader.py`) turned
out to require reading `train_autoregressive.py` directly, which neither the original coverage audit
nor the reflective cross-check had done at the function-body level. Source-verified against
`train_autoregressive.py:127-138` (`train_step_autoregressive`'s `y.ndim`/`jax.nn.one_hot` block) and
`:63-68` (`compute_loss`'s own ndim-check).

**CONFIRMED**: upstream's loaders yield raw integer token ids `(B, S)` for `y` — one-hot expansion to
`(B, S, V)` happens inside the training step, not the loader (host→device transfer stays compact).
The Julia port's `compute_loss` assumed `targets` already one-hot, with no such fallback — so raw-id
batches would silently miscompute the CE metric via shape-broadcasting rather than erroring. Worse:
`train_step_autoregressive` (Julia) delegates clamp-construction to the *shared*
`_pcn_step`/`train_step`/`train_step!` (`train.jl`, common to every non-autoregressive example too),
which has no `y`-specific handling — so a `compute_loss`-only fix would leave the actual clamp
shape-mismatched regardless, never reaching `compute_loss` at all.

**Fix** (scoped to `train_autoregressive.jl` only — `train.jl`'s shared clamp machinery is untouched):
`train_step_autoregressive` now inspects `ndims(batch["y"])` against the output node's declared shape
and, for raw ids, one-hot-expands via the file's existing `_one_hot` helper (widened from
`AbstractMatrix{<:Integer}` to `AbstractMatrix{<:Real}` to accept the Float32-encoded ids used
elsewhere in this codebase) *before* calling `_pcn_step`, so the clamped node — not just the CE
metric — sees the expanded array. Already-one-hot `y` passes through unchanged. DIVERGENCE
(intentional): upstream raises unless `y.ndim == 2`; the Julia port additionally accepts pre-shaped
one-hot `y` so existing non-loader callers (e.g. `test_sequence_training.jl`'s plain-classifier step)
keep working, and drops `compute_loss`'s own now-redundant ndim-check (expansion happens exactly
once, at the clamp site).

Regression tests: `test/test_transformer_lm.jl` — raw-id batch's clamped `z_latent` matches a manually
one-hotted target exactly; raw-id and pre-one-hotted batches produce identical energy/CE/state
(pass-through equivalence); a wrong-ndim batch raises `ArgumentError`. Existing tests
(`test_transformer_lm.jl`) updated to pass raw ids for `"y"` instead of manual pre-one-hotting,
matching the real loader contract this fix unblocks (C-03).

### F-06 — `TransformerBlock` output-activation divergence — OPEN

Found incidentally by Tier B's fixture-drafting workflow (section 6), not by a targeted audit
pass. Upstream's `TransformerBlock.forward()` (`fabricpc/nodes/transformer.py`) never applies
`node_info.activation` anywhere in its body — grep-verified, zero references to
`node_info.activation`/`self.activation` in the whole file — so the `activation:
Optional[ActivationBase] = IdentityActivation()` constructor parameter is dead code upstream.
Julia's `compute_mu` (`src/nodes/transformer.jl`) DOES apply it: `return forward(node.activation,
z_mu)`.

**Not a deliberate divergence** (neither side chose this on purpose — upstream's parameter looks
like unfinished wiring, not a design decision), so this is classified as a latent fidelity gap,
not filed under section 2. **Currently dormant**: every shipped example/test constructs
`TransformerBlock` with the default `IdentityActivation()` (a no-op) on both sides, so Tier B's
byte-level fixture comparisons pass regardless — the divergence has no observable effect yet.
**Would surface** the moment any caller passes a non-Identity `activation` to `TransformerBlock`:
Julia would apply it to the output, upstream would silently ignore it, and the two would diverge
with no error or warning. Left unfixed pending a real caller (per this register's own "measured
need" convention) — fixing now would mean choosing between matching upstream's (arguably buggy)
dead parameter or diverging on purpose, a decision better made when something actually depends on
it. Flagged with a one-line pointer in `compute_mu`'s call site (`src/nodes/transformer.jl`) so a
future reader hits the warning before shipping a non-Identity `TransformerBlock`.

---

## 2. Fidelity findings — deliberate divergences (document, don't fix)

Written up as full decisions.md entries this round (D-04) — full rationale for each now
lives in `docs/decisions.md` §21, not just the one-line table below.

| ID | Item | Status |
|----|------|--------|
| X-01 | All-float clamp pipeline | DOCUMENTED (decisions.md §21) — locked in, safe under 2^24 |
| X-02 | Mask slot removed from TransformerBlock | DOCUMENTED (decisions.md §21) — locked in |
| X-03 | rope_theta hardcoded 10000 | DOCUMENTED (decisions.md §21) — parameterize opportunistically |
| X-04 | num_heads default 8→4 | DOCUMENTED (decisions.md §21) — tripwire only, all callers explicit |
| X-05 | GraphNamespace deferred | **RESOLVED (decisions.md §21).** Upstream's own multi-block assembly (`examples/transformer_demo.py:187,191`) doesn't use `GraphNamespace` either — plain f-string-suffixed names, matching Julia's existing `transformer_lm.jl` exactly. Zero `GraphNamespace` references in `transformer_v2.py`/`transformer_demo.py` (grep-verified). |
| X-06 | `train_autoregressive` driver: 3 upstream control-flow features not ported, one previously MISCHARACTERIZED | **NEW — found by R-01's full-driver comparison, independently re-verified by an adversarial pass.** (1) Fractional epochs (`num_epochs` e.g. `1.5` → 2 loop iterations, the second truncated to `round(frac·num_batches)` batches, train_autoregressive.py:238-263) is a REAL, reachable, upstream-unit-tested feature (`tests/test_fabricpc.py:520-549`) — Julia's prior docstring called it an "upstream read-but-unused stub" grouped with gradient-accumulation, which was simply wrong; `num_epochs::Integer` here has no fractional path at all (a `MethodError` on a non-integer call, not silent misbehavior). (2) `gradient_accumulation_steps` genuinely IS a dead upstream read (config value read once, train_autoregressive.py:236, never referenced again in the function body) — the one part the old docstring got right. (3) `iter_callback` (per-batch, train_autoregressive.py:206,298-301, return value replaces the raw energy pushed to the batch-energy list) has no Julia equivalent in `train_autoregressive.jl` at all — despite the identical pattern already existing elsewhere in this SAME package (`train_pcn`, `src/training/train.jl:128,148,167`, tested in `test/test_train_pcn.jl:65`) — an internal inconsistency, not just a cross-language gap. (4) `epoch_callback` here takes `(epoch, params, structure)` vs upstream's `(epoch_idx, params, structure, config, rng_key; energy, ce_loss)` (train_autoregressive.py:312-320) — a real argument-list mismatch. **Reachability**: all three are dormant in THIS repo (no example/test needs them) but `fabricpc/tuning/bayesian_tuner.py` (upstream, C-10 — already DEFERRED here) is a LIVE consumer of both `iter_callback` and the extended `epoch_callback` signature (`bayesian_tuner.py:182-191`) — porting BayesianTuner is currently structurally blocked on this gap, not merely stylistically divergent from it. This sharpens C-10's own DEFERRED rationale: the blocker is now named. Docstring corrected in `train_autoregressive.jl`; not implemented (no measured need yet — matches this project's "measured need, not a checklist" convention) — implement `iter_callback` + the extended `epoch_callback` signature + fractional-epoch support together, if and when C-10 is picked up. |

| X-07 | `InferenceSGDMomentum` — heavy-ball momentum inference, shipped ahead of upstream | **NEW (2026-07-14, `docs/decisions.md` §26).** Upstream has only a design document (`docs/dev_plans_archive/momentum_sgd_inference_plan.md`) — `fabricpc/core/inference.py` still ships only `InferenceBase`/`InferenceSGD`/`InferenceSGDNormClip`. Julia implements upstream's own planned design (override the inference LOOP, not `compute_new_latent`, since velocity must be carried across steps) ahead of upstream, with hyperparameters DERIVED from a measured Hessian spectrum rather than guessed. Not a C-* coverage item — no upstream code exists to conform to. Conformance anchor: `momentum=0` reproduces `InferenceSGD` bit-for-bit, 20/20 fields exact. Falsifies upstream's own proposed default (`eta_infer=0.1, momentum=0.9`) as unstable on the tiny transformer diagnostic graph, predicted before measured, from the same spectrum. **A verification pass chasing its own theory-vs-measurement mismatch corrected `docs/decisions.md` §24's own `lambda_min=2.3`/`kappa≈49` figures to `lambda_min≈0.17`/`kappa≈668`** (§24's power iteration was too short to converge on this graph's near-degenerate spectrum) — §24's qualitative conclusions stand, only the numbers move. Also found+fixed a real test-file bug (calling `inference_step` directly silently ignores `momentum`, routing through a documented plain-SGD fallback instead) — fixed by extracting `momentum_inference_step` as the single source of truth for one heavy-ball step. Real, measured (not naive-theory-matching) speedup over plain SGD on this graph: ≈0.989/step vs ≈0.998/step, same renormalized perturb-and-track methodology. Free byproduct: the velocity-threaded-through-the-caller design is the loop-carried-state shape J-02/J-03's `Reactant.@trace while` plan (section 5) needs. Tests: `test/test_inference_momentum.jl` (26/26, always run) + `test/test_inference_momentum_diagnostic.jl` (gated, `FABRICPC_MOMENTUM_SPECTRUM_DIAGNOSTIC=1`). |

## 3. Verified-equivalent (no action — evidence on file)

Spot-checked by the reflective cross-check (RoPE pairing/frequencies/rotation, muPC residual-depth
formula, Linear-gradient energy-type coverage): all **CONFIRMED-ACCURATE**. RoPE's underlying math
genuinely matches upstream bit-for-bit; only the `_tb_apply_rope` comment was misleading — see D-02.

## 4. Coverage gaps

| ID | Sev | Item | Status |
|----|-----|------|--------|
| C-01 | HIGH | ConvNode (1D/2D/3D) + pooling + windowed-shape validation | DEFERRED (post-Batch 2) |
| C-02 | HIGH | ResNet-18/CIFAR-10 demo | BLOCKED (C-01) |
| C-03 | HIGH | Real-text LM data (char tokenizer + windowed loader) | **VERIFIED CLOSED.** `examples/char_lm_pc.jl` -- `CharDataLoader`, a port of `_TokenSequenceLoader`/`CharDataLoader` (dataloader.py:272-390): raw-download (not TFDS -- no Julia binding) of the same `karpathy/char-rnn` source TFDS uses, split with TFDS's own verified 90/5/5 formula (`tiny_shakespeare_dataset_builder.py:52-56`), train-only vocab. Kept example-local (no `src/utils/` module), matching `mnist_pc.jl`'s existing precedent -- this port has never had a library-level dataloader. DIVERGENCE (documented, intentional): 1-based char ids, not upstream's 0-based, matching every other token-id consumer in this codebase (EmbeddingNode, `_one_hot`). Offline regression tests (`test/test_char_dataloader.jl`, 12 assertions): TFDS split formula on two boundary cases, vocab sortedness/1-basing, sliding-window shift-by-one invariant + exact ground truth, drop-incomplete-batch, `max_samples` cap, seeded-reproducible + re-iterable-across-epochs shuffling, decode round-trip. Live-smoke-tested against the real corpus (not just synthetic): split sizes exactly 1,003,854/55,770/55,770 chars (bit-exact TFDS-formula match), vocab_size=65, **zero OOV chars in validation/test against the train-only vocab** (relevant to C-06), a short real training run + generation both executed without error. 352/352 green. |
| C-04 | HIGH | PC-vs-backprop comparison harness | **MNIST-MLP arm: VERIFIED CLOSED (2026-07-14).** `benchmark/pc_vs_backprop_mnist.jl` (`ExperimentArm`-shaped, mirroring upstream `ab_experiment.py`/`statistics.py`'s opaque-`params`/`structure` treatment — a Julia backprop arm never touches FabricPC internals, not a `train_backprop.py` port). PC arm = FabricPC's own `train_pcn`/`evaluate_pcn` at the exact contractive config already validated (§6/§24, `eta_infer=0.1`, `infer_steps=30`, plain SGD `lr=0.002` — deliberately NOT retuned, so this measures that specific config, not PC's achievable ceiling). Backprop arm = Lux.jl `Chain(Dense(784=>128,tanh), Dense(128=>10))`, `Optimisers.Adam(1e-3)`. Same architecture, same real MNIST data (not synthetic — this is Julia-vs-Julia, not a cross-language conformance fixture) across 8 seeded trials each. **Result: backprop 88.00%±0.16% vs PC 74.67%±5.80% test accuracy — a 13.33pp gap, paired t-test p=0.0004, Cohen's d=−2.28 (large effect), PC also far noisier across seeds (SD 5.80% vs 0.46%).** Real, un-forced negative result for local PC learning at this untuned config — expected direction (backprop's global credit assignment vs PC's local rule), not a claim about PC's ceiling. Cross-checked against `benchmark/results.md`'s existing 81.65% report: one trial (81.35%) lands in that neighborhood, consistent with legitimate seed-to-seed variance at this config rather than a harness bug (the harness's per-epoch reshuffle scheme differs mechanically from `mnist_pc.jl`'s single-persistent-RNG scheme, so per-seed numbers aren't bit-identical to that report — expected, not a defect). Small supporting infra addition: `examples/mnist_pc.jl`'s `main()` guarded behind `abspath(PROGRAM_FILE) == @__FILE__` so its data-loader helpers are `include`-able without triggering a training run. **Transformer arm remains OPEN**, confound as scoped below. |
| C-05 | MED | InferenceSGDNormClip | **VERIFIED CLOSED.** `src/core/inference.jl` -- port of `InferenceSGDNormClip` (inference.py:312-356): per-sample L2 gradient-norm clipping on `latent_grad` before the SGD update. Upstream dispatches per-algorithm via a Python class hierarchy (`InferenceBase.update_latents` calls `cls.compute_new_latent(...)`, `cls = type(inference_obj)`) -- the Julia port had NO such dispatch point (`update_latents` hardcoded the plain-SGD formula inline, since only `InferenceSGD` existed). Extracted `compute_new_latent(inf, z_latent, latent_grad)` as a Julia-multiple-dispatch analogue of the same override point, byte-identical for the existing `InferenceSGD` arm (verified: `max_norm=Inf` ⇔ plain SGD, exactly). NOTE (scoped, documented in-file): `jit_flat.jl`'s separate Dict-free JIT/Reactant-prep inference path still hardcodes SGD inline -- `InferenceSGDNormClip` is not usable there; out of C-05's scope (a distinct subsystem with its own eager-parity test discipline). Tests (`test/test_inference_normclip.jl`): hand-computed clip-vs-no-clip ground truth, `max_norm=Inf` byte-equivalence to `InferenceSGD`, `latent_decay` parity isolated from clipping, acceptance (small PC graph trains to lower energy under norm-clipped inference). 359/359 green. **Upstream conformance coverage added 2026-07-14** (`docs/AUDIT_REGISTER.md` section 6, Tier D transformer-LM): `test_tier_d_transformer_stable.jl`'s `max_norm=1.0` variant (523/523 vs upstream JAX) exercises always-clipped/never-clipped regimes but never approaches the clip boundary itself — closed with a SECOND, purpose-built fixture (`test_tier_d_transformer_stable_normclip_boundary.jl`, `max_norm=2.5`, chosen after a first naive guess was caught and rejected: clipping one node's gradient cascades through the graph topology and can suppress a DIFFERENT node's gradient by ~30×, so a boundary target can't be read off an unclipped baseline in a multi-node graph sharing one global `max_norm`). Result: 523/523 at 1e-4, PLUS a dedicated discrete clip/no-clip decision check at 2 genuine near-boundary crossings (ratios 1.0095→0.9995 and 1.0058→0.9938, both within 1% of the threshold) — **zero decision disagreements**, verified against upstream's own independently-dumped `latent_grad` arrays (not Julia's self-consistency). The `min(1, max_norm/(norm+eps))` branch is now genuinely exercised, not coverage-by-absence. |
| C-06 | MED | evaluate_autoregressive | **VERIFIED CLOSED.** `src/training/train_autoregressive.jl` -- port of `_eval_step_autoregressive`/`evaluate_autoregressive` (train_autoregressive.py:541-710): mean cross-entropy loss, perplexity, next-token accuracy over a `test_loader` (e.g. `CharDataLoader`, C-03), with `"x"` clamped and `"y"` left FREE (the network predicts, unlike training) -- de-risked by C-03's real-corpus smoke test finding zero train/validation/test vocab mismatch. Applies the same F-05 one-hot handling (`"y"` may be raw ids or one-hot; `compute_loss` still carries no ndim-check of its own -- expansion happens once, in `_eval_step_autoregressive`, and the expanded array is returned to the caller so the accuracy computation doesn't repeat the check). DIVERGENCES (documented in-file): no `config`/`use_causal_mask` param (upstream's external causal-mask clamp branch is dead code in this port, same as `train_step_autoregressive`); `rng` threaded sequentially, no `jax.random.split`/`jax.jit` (this stack is eager throughout, matching the rest of the file). `debug=true` ports upstream's first-batch diagnostics (per-token CE, per-token intrinsic perplexity, probability on the correct token). `_count_correct` factored out (mirrors the existing `_sample_next` precedent) for an isolated, hand-computed-ground-truth test independent of the full graph/PC-inference machinery. Tests (`test/test_transformer_lm.jl`): metric shape/range/`perplexity==exp(loss)`, raw-id vs pre-one-hotted equivalence, `debug=true` doesn't perturb results, empty-loader edge case, `_count_correct` exact ground truth. 371/371 green. |
| C-07 | MED | GlobalStateInit / NodeDistributionStateInit | OPEN |
| C-08 | MED | ND Kaiming/Xavier fan-in math | DEFERRED (tied to C-01) |
| C-09 | MED | muPC never validated on either transformer node family | Umbrella over F-01+F-02, both now closed — test coverage for TransformerBlock landed with F-01's fix; MhaResidualNode/LnMlp1Node/Mlp2Residual under real `MuPCConfig` remains OPEN. **Monolithic `TransformerBlock` half now has real CONTENT (2026-07-14, `docs/decisions.md` §24 follow-up):** measured muPC's effect on the SAME condition number characterized while closing Tier D — `MuPCConfig` reduces `λmax/λmin` from **~49 → ~41 (~16%)**, `include_output=True` vs `False` making no measurable difference. This is NOT the mechanism one might hope for: `λmax` is essentially unchanged (<1%) — muPC does not touch the dominant instability at all. The entire effect is a ~19% increase in `λmin`, because `compute_mupc_scalings` only reaches TWO edges in this topology (`embed→transformer_0:in` always, `skip_0→output:in` only when `include_output=True`) — both PERIPHERAL to the graph — while the measured `λmax≈113` instability lives INSIDE `TransformerBlock`'s own monolithic internal Jacobian (attention softmax + GELU FFN), which per-edge muPC scaling has no mechanism to reach. Translates to real steps saved: ~12.5→~10.5 steps to reach the same 0.6 residual-relaxation fraction (η_opt/r_opt formula, cross-validated against this session's independently-measured baseline). **🔴 THE NEGATIVE RESULT, stated explicitly because it is the one people will get wrong: `λmax` unchanged ⇒ `η*` unchanged at ~0.0176 ⇒ muPC does NOT fix the `transformer_lm()` η=0.1 instability (`docs/decisions.md` §24).** "muPC improves conditioning, therefore it fixes the η problem" is the natural — and wrong — inference from the ~16% κ number alone; `transformer_lm()`'s default is 5.7× past its own stability limit with muPC on *or* off. What the 16% actually is: `λmin` rose (~2.31→~2.74, ~19%) while `λmax` sat still — muPC is lifting the SLOW modes, not suppressing the fast one, which is the correct mechanism for what per-edge `forward_scale`/`topdown_grad_scale` factors actually do (O(1) activation-variance/gradient-scale correction across width/depth, μP-style), not curvature/Hessian conditioning — a different mechanism from what would be needed to move `λmax`/`η*`. **The decomposed family (MhaResidualNode/LnMlp1Node/Mlp2ResidualNode) is a materially different question, still fully OPEN** — that family exposes attention/FFN internals as separate PC nodes with their own inter-node edges, so muPC's per-edge scaling would reach much more of the internal dynamics there and could plausibly show a larger effect, possibly including `λmax` itself; not measured this session. **Independently cross-validated by upstream, different methodology and architecture (2026-07-14, upstream `docs/dev_plans_archive/momentum_sgd_inference_plan.md` §"Status of Previous Work"):** upstream shipped BOTH of muPC's forward-variance-gain (`variance_gain()`) AND Jacobian-compensation (`jacobian_gain()`) corrections — theoretically sound, all tests passing — and measured **zero practical impact on deep-chain MNIST accuracy**: 32-layer 59.9%→56.4% (worse), 64-layer 20.8%→21.0% (flat), only the 8-layer case improved (90.8%→91.6%). Upstream's own diagnosis (energy-distribution probe on a 32-layer chain: middle layers h8-h24 carry **zero energy**) locates the real bottleneck as local-inference SIGNAL PROPAGATION (PC relaxation propagates the output error exactly 1 hop/step; the self-error restoring force damps it faster than it can reach deep layers) — a *different* failure mode than a Hessian-conditioning problem, but the SAME conclusion this section reached by direct spectral measurement on a different architecture (`TransformerBlock`, not a deep MLP chain): **muPC's forward/Jacobian gain-scaling mechanism does not touch whatever the actual bottleneck is.** Two independent methodologies (end-to-end accuracy on a deep MLP chain; direct `λmax/λmin` spectral measurement on a transformer block), two different architectures, converging on the same negative result increases confidence this is a structural property of the muPC gain-scaling mechanism (O(1) activation-variance correction, μP-style) rather than an artifact of either investigation's specific setup. This is also upstream's own stated MOTIVATION for the `InferenceSGDMomentum` design this port has now implemented ahead of upstream (`docs/decisions.md` §26) — momentum targets exactly the signal-propagation mechanism upstream's diagnosis names. **Open caveat, not yet re-verified:** this section's own `λmin` figures (`~2.31→~2.74`) were measured with the SAME power-iteration methodology `docs/decisions.md` §24 used before §26 found it under-converged on this graph's near-degenerate spectrum (§24's `λmin=2.3` corrected to `λmin≈0.17`) — the ~16-19% RELATIVE muPC effect reported here has not been re-checked against a properly-converged (renormalized) power iteration; the qualitative conclusion (`λmax` essentially untouched) is independently supported by the near-init stability-boundary cross-check in §26 and is not expected to move, but the specific `2.31`/`2.74` numbers may be revised by a future re-measurement the same way §24's were. |
| C-10 | LOW | BayesianTuner (537-line #28 rework) | DEFERRED — X-06 names a concrete structural blocker: `bayesian_tuner.py` requires `train_autoregressive`'s `iter_callback` param and `epoch_callback`'s extended signature, neither of which exist in `train_autoregressive.jl` yet. |
| C-11 | LOW | StorkeyHopfield weight_init: Zeros (Julia) vs Xavier (upstream) | **VERIFIED CLOSED** (`52bb6a0`). Reflective cross-check resolved the R-03 read: neither implementation has a classical Storkey/Hebbian accumulation rule — W is 100% PC-gradient-learned in both — so upstream's own stated criterion for "Xavier matters when W learns via PC gradients" applies, confirming this was a port slip (Julia copied the internal `None`-fallback constant, not the constructor's actual default), not a legitimate design choice. Fixed to `XavierInitializer()`. |
| C-12 | LOW | ~17 further API-surface/UX divergences (IDE list) | DEFERRED |

**Known-deferred, confirmed absent, no action:** multi-GPU/pmap · train_backprop.py port (anti-thesis; see C-04) · Aim dashboarding/experiments · tfds loaders · Optuna.

## 5. JIT/performance lane

| ID | Item | Status |
|----|------|--------|
| J-01 | Flat gradient seam: Enzyme-under-Reactant | **DISSOLVED as an architectural blocker — now a resolved design decision (2026-07-14, `docs/decisions.md` §15/§23 updates).** The original framing conflated "any eager AD" with eager-Enzyme specifically. This codebase's actual production eager seam is Zygote (decision #19), which never calls eager `Enzyme.autodiff` at all. Execution-verified (`benchmark/jit_zygote_expt/main_experiment.jl`): Zygote-eager + `Reactant.@compile`-with-`Enzyme.gradient`-inside (never eager `Enzyme.autodiff`) compiles and matches Zygote's eager gradient to ~1e-6/1e-7 — no poisoning. A serious secondary hazard surfaced while checking this — `using Reactant` transitively loads Enzyme, silently flipping FabricPC's REAL seam dispatch (not just the raw-array bypass) to Enzyme's MHA-crashing implementation, with zero explicit `using Enzyme` and zero catchable exception — **but this is exactly F-04's dispatch-override hazard, now VERIFIED CLOSED (section 1) by backend-type dispatch.** With that fix, `using FabricPC, Zygote, Reactant` together is now SAFE: dispatch structurally cannot flip to Enzyme's method table entry regardless of what Reactant transitively loads (`_AD_BACKEND[]` stays on `ZygoteBackend`, verified by `which()` in `test/test_autodiff_backend_registration.jl`). The eager seam (Layer 0/1, `_ad_param_grads` → Zygote) and the compiled lane (Layer 2, `Enzyme.gradient` over raw arrays inside `@compile`) were ALREADY different code paths — they only conflicted before due to the now-fixed accidental method overwrite. **What remains is J-02/J-03's own scope, not an architectural blocker**: a flat weight-gradient path that JITs the FULL `train_step` (not just the raw-array demo pattern `wgrad` already validates on a partial parameter set) — real, unbuilt work, not attempted this round, but no longer blocked on an unresolved conflict. |
| J-02 | Compile full `train_step` | **No longer architecturally blocked (J-01 dissolved, 2026-07-14) — still genuinely OPEN, not started.** `compute_local_weight_gradients` remains entirely eager/Dict-based, no flat/traced counterpart exists yet; building one is real, unattempted work, just no longer gated on an unresolved Enzyme-under-Reactant conflict. **🔴 Design-before-building note (`docs/decisions.md` §25):** the hand-rolled AdamW's step counter is a plain Julia scalar, frozen at trace time under `@compile` unless tracked — every compiled step would silently reapply step-1's bias correction with no error. Use `Optimisers.jl` (already a regular dep, already resolved with `OptimisersReactantExt`, currently unused) on the compiled lane instead of the hand-rolled optimizer — retires this landmine and independently closes the original audit's "generic optimizer injection" gap (upstream trains via swappable `optax`). |
| J-03 | Extend flat lane to seam nodes | **PARTIAL — TransformerBlock forward wired, backward parked.** `flat_forward(::TransformerBlock, ...)` (`src/jit_flat.jl`) wires the already-validated `_tb_block_flat` kernel (`src/nodes/transformer.jl:296-338`, previously a standalone, disconnected kernel + benchmark) into `CompiledPlan`'s dispatch — `to_flat_params` special-cased to bridge via the existing `flat_block_args` (TransformerBlock's weights are keyed by NAME, not by edge key like every other node here). Verified byte-equivalent to eager `compute_mu` (`test/test_jit_flat.jl`, new testset) on a graph where TransformerBlock is the unclamped terminal node (`flat_latent_grads`'s `out_degree==0 && !is_clamped` branch — forward only). **No `_flat_input_grads(::TransformerBlock, ...)` method exists** — the attention+FFN backward pass needed for TransformerBlock as an interior or clamped node — same category of problem as J-01 (needs Enzyme-under-Reactant, or a substantial hand-derived closed-form gradient verified against Zygote's already-upstream-validated gradient); verified this fails loudly with a clean `MethodError`, not silently wrong output, rather than asserting it in CI (a `MethodError` from a private multi-arg dispatch is a brittle thing to `@test_throws` on). Decomposed nodes (`MhaResidualNode`/`LnMlp1Node`/`Mlp2ResidualNode`/`EmbeddingNode`/`VocabProjectionNode`) and `StorkeyHopfield` remain untouched — zero flat-lane work for any of them. |
| J-04 | Benchmark harness | **INFERENCE-ONLY arm VERIFIED CLOSED (2026-07-14, `docs/decisions.md` §11).** The "BLOCKED on J-01/J-02" framing was too broad — that pair blocks compiling `compute_local_weight_gradients` (the weight-gradient/M-step) specifically, but `compile_inference`/`flat_run_inference` (the E-step relaxation loop) is a separate, already-validated path, and the MNIST-MLP architecture (`Linear`-only) is fully covered by `flat_forward`/`flat_latent_grads` — confirmed against `src/jit_flat.jl` before benchmarking, not assumed. `benchmark/mnist_inference_vs_jax.{py,jl}`: same architecture/batch/η/steps, same weights both sides (RNG-trap discipline). **Julia+Reactant's `compile_inference` runs ~7-8× faster than upstream's own `jax.jit(run_inference)`** (the first cross-language, both-compiled number this codebase has produced — do not conflate with §11's existing 8.8×/32× Julia-vs-Julia numbers), numerics validated to float32 precision. Scope: inference only, no weight update — NOT a training-speed claim. **The training/weight-gradient arm remains BLOCKED on J-01/J-02**, unchanged. **🔴 SCOPE CAVEAT for any LARGER-scale rerun (`docs/decisions.md` §25):** `flat_run_inference` (`src/jit_flat.jl:364-365`) UNROLLS its step loop under `@compile` (a native `for`, no `@trace`) — invisible at this benchmark's 20 steps, but does not scale to §24's own finding that this graph family needs 50-300 steps to actually converge. `@trace while` (relax-to-tolerance) is the fix, and — sourced from Reactant's own AD docs, not asserted — a genuine positioning argument: PC's local weight-gradients never differentiate through the relaxation loop, so PC can use a convergence-based `@trace while` in a regime where backprop-style AD structurally cannot without paying real memory cost or hitting a hard dynamic-shape platform limit. Also: benchmark CPU only, or match upstream's `jax_setup.py` XLA flags (all `xla_gpu_*`, all performance-reducing) on any future GPU comparison — this run was accidentally fair since this machine has no GPU. |
| J-05 | Hand-derived attention VJP as explicit Layer-1 gradient | DEFERRED (optional, post-J-03) — unchanged. |
| J-06 | `NodeState` parametric types `{T<:AbstractArray}` | OPEN (post-Batch 3) — unchanged, "a real refactor — schedule deliberately." |
| J-07 | Slot-name resolution out of node forwards | OPEN (with J-03) — unchanged. |

**Batch 3 sequencing note**: the original register's ordering (J-01→J-02→J-03→J-04) assumed J-01/J-02 would land first. In practice J-03's TransformerBlock forward-wiring turned out to be the only piece of this batch achievable without first resolving J-01's architectural blocker — it used already-validated math with no new numerical kernel, whereas J-01 hit a real unresolved process-level conflict and J-02 is blocked on J-01. J-04 remains gated on Tier C regardless of J-01/J-02/J-03's state.

## 6. Conformance harness

**Tier A — VERIFIED CLOSED.** `scripts/generate_tier_a_fixtures.py` (run against an isolated
`jax[cpu]` venv, upstream HEAD `316367c`, no re-sync needed per A-01) dumps
`test/conformance/fixtures/tier_a.npz` (69 arrays, 28 KB): every activation's forward+derivative
(`SoftplusActivation` excluded — grep-verified it has no upstream counterpart at all, a Julia-only
addition), Softmax's full Jacobian + an outlier-input numerical-stability case, every energy's
energy+grad_latent+grad_mu (`grad_mu` checked via `jax.grad` autodiff ground truth — upstream has
no hand-written closed form for it; only Julia does), RoPE's cos/sin tables + rotation output +
a single-nonzero-pair discriminator (pins down both position-indexing and pairing-convention),
and LayerNorm (standard + an outlier case). `test/conformance/test_tier_a.jl` loads it via NPZ.jl
and compares with a per-element `numpy.allclose`-style check (rtol 1e-6, atol 1e-6 — NOT Julia's
built-in array `≈`, which is a single norm-based check, too weak for this purpose). **43/44
assertions passed on the first real run** (one failure was a fixture-generator bug — the softmax
jacobian was accidentally computed from a different random draw than the saved input — fixed,
not a Julia port issue). The one substantive finding: `_tb_apply_rope`'s "GELU" comment (see D-02
sibling) claimed upstream's forward used erf-GELU by default vs Julia's tanh-approximation,
framing it as a resolved inconsistency; `jax.nn.gelu`'s actual default is `approximate=True` (the
tanh form) — there was never an inconsistency. Comment corrected in this commit.

**Tier B — VERIFIED CLOSED.** `scripts/generate_tier_b_fixtures.py` dumps
`test/conformance/fixtures/tier_b.npz` (217 arrays): forward + BOTH gradient paths
(closed-form and/or the Phase-D autodiff seam), isolated node-level (hand-built
NodeParams/NodeState/NodeInfo, no `graph()` — mirroring the existing F-01 regression test's own
pattern), for every node type FabricPC.jl has: Linear (↔ upstream `LinearExplicitGrad`) +
LinearResidual, IdentityNode + SkipConnection, TransformerBlock (forward/latent-grads seam +
the T3-extended muPC-on/off weight-grad case — see below), MhaResidualNode + LnMlp1Node +
Mlp2ResidualNode, EmbeddingNode + VocabProjectionNode, StorkeyHopfield.
`test/conformance/test_tier_b.jl` (rtol/atol 1e-5, its own `allclose_b`/`FIX_B` — kept separate
from Tier A's `allclose`/`FIX` since both files share `Main` scope once `include`d).
**151/151 assertions passed** (verified by an isolated re-run of the file alone, not just the
full-suite aggregate, since a very long full-suite run's output was truncated to its final
summary line — a display/capture artifact, not a correctness question; the isolated run shows
the full per-node breakdown).

Drafted by a 6-way parallel research workflow (one agent per node group, each reading the
upstream Python class and the Julia implementation directly and self-testing its own
fixture-generation code against the real venv before returning it), then assembled and
Julia-verified by hand. One real bug caught during assembly (not by any agent's self-test, which
only exercised the Python side): `TransformerBlock`'s `b_ff1` bias is reshaped internally via
`size(bias, 2)` (transformer.jl), unlike its sibling gamma/beta/other-biases which reshape via a
fixed literal `E` — loading the fixture's raw `(1,1,32)` array directly for `b_ff1` throws a
`DimensionMismatch`; fixed by reshaping to `(1,32)` before constructing `NodeParams`, and
flattening (`vec`) both sides of the bias-gradient comparison to sidestep the resulting
non-matching-ndims broadcast. The same `size(bias,2)` hazard exists for `LnMlp1Node`'s `b_ff1`,
`Mlp2ResidualNode`'s `b_ff2`, and `VocabProjectionNode`'s `b_out` — those three node groups'
own agents independently found and correctly handled it in their first drafts.

Substantive findings:
- **F-01/F-02 re-validated against REAL upstream numbers**, not just internal Julia-side formula
  matching (all the existing regression test could do): the muPC LayerNorm weight-grad
  compensation (`a` applied to weights, NOT biases) now has byte-level upstream ground truth on
  both the muPC-on and muPC-off (negative control) cases.
- New, dormant, low-severity finding (not previously in this register): upstream's
  `TransformerBlock.forward()` never applies `node_info.activation` to its output (grep-verified
  against `fabricpc/nodes/transformer.py` — no reference anywhere in the file); Julia's
  `compute_mu` does apply it. Harmless today (both default to `IdentityActivation`, a no-op) but
  would silently diverge if a future caller ever constructed a `TransformerBlock` with a
  non-identity output activation. Tracked here; not fixed (dormant, no live caller).
- `LinearResidual`: upstream has no closed-form gradient override at all (only `forward()`,
  falling through to `NodeBase`'s generic autodiff) while Julia's `LinearResidual` is
  hand-fused-closed-form — same "autodiff-of-upstream's-own-forward is the legitimate ground
  truth" rationale Tier A used for `grad_mu`, applied here to a whole node's gradients, not just
  one energy function.
- `EmbeddingNode`: confirmed upstream's weight gradient is NOT a hand-written scatter-add (no
  override exists) — it's `jax.value_and_grad` differentiating the fancy-index gather directly,
  which JAX's autodiff turns into a scatter-add automatically. Independently verified upstream's
  output against a from-scratch manual scatter-add (matched), and Julia's own hand-written
  scatter-add matches that same upstream ground truth.

**Tier C — VERIFIED CLOSED.** `scripts/generate_tier_c_fixtures.py` dumps
`test/conformance/fixtures/tier_c.npz` (106 arrays): LOOP-LEVEL conformance — not an isolated
node (Tier B) but a real *assembled* 3-node graph, `x(4) → h(6, tanh) → y(3)`, run through
`initialize_graph_state → inference_step → run_inference (8 steps to convergence) →
get_graph_param_gradient → one train_step` (SGD via `optax.sgd`). Params/batch are
JAX-generated arrays injected directly into `GraphParams`/clamps (RNG trap, same discipline as
Tier A/B); the terminal source node "x" is built as plain upstream `Linear` rather than
`LinearExplicitGrad` specifically because `LinearExplicitGrad.forward_and_latent_grads` does
NOT reimplement `NodeBase`'s in-degree-0 terminal special case and would silently compute a
wrong nonzero energy from an unconnected bias — verified by reading both class bodies in full,
not assumed. `test/conformance/test_tier_c.jl` (rtol/atol 1e-5, its own `allclose_c`/`FIX_C`)
asserts all 106 dumped quantities (every `GraphState` field × 3 nodes × 5 phases, h/y
weights+biases at `params0`/`grad`/`trained`, both scalar energies). **100/100 assertions
passed**; full-suite regression run (all 22 test files, not just conformance) **672/672
green** — the one-line `runtests.jl` include caused zero collateral breakage.

Landed via a 4-phase reflective multi-agent workflow (Verify Draft → Fix & Generate → Julia
Test → Adversarial Re-verify), mirroring Tier B's parallel-agent drafting precedent. Verify
Draft ran 3 independent adversarial reviewers against the draft generator before it was ever
executed: two verdicted CORRECT (the `Linear`-vs-`LinearExplicitGrad` node-class reasoning
above, and the state-init RNG-safety argument — a minor wording nuance was flagged but the
underlying safety claim held under full trace); the third caught a real issue — two scalar
energy dumps (`energy_pretrain`, `energy_trainstep`) were bare 0-d JAX arrays, inconsistent
with Tier B's own documented convention of reshaping every scalar to `(1,)` before `put()`
(a 0-d array round-trips through NPZ.jl as a bare `Float32`, not an `Array`, breaking type
uniformity with every other dumped field) — fixed before the fixture was ever generated.

**The Adversarial Re-verify phase went beyond "the assertions look non-trivial" and empirically
proved discrimination by mutation**: with the harness at 100/100 green, the reviewer changed
`train_step`'s learning rate 0.05→0.06 and re-ran — exactly the 4 "trained"-params assertions
failed and nothing else (96/100), isolating the params-check path as a real, independent
comparison rather than a tautology. It then changed `InferenceSGD`'s `eta_infer` 0.1→0.13 and
re-ran — `initialize_graph_state`'s 18/18 correctly stayed green (state-init doesn't depend on
`eta_infer`) while the failure correctly cascaded through
`inference_step → run_inference → get_graph_param_gradient → train_step` (41/100 fail),
isolating the state-check path (90 of the 100 assertions) as genuinely live rather than
cached/stale. The file was then restored to its original content (verified byte-identical via
diff) and reconfirmed at 100/100 before reporting CORRECT. This is a materially stronger claim
than "100/100 green" alone: the tests were *observed to fail* on the specific code paths they
claim to cover, not merely observed to pass once.

**Tier D — MNIST-MLP track VERIFIED CLOSED, transformer-LM track OPEN.** End-to-end (not
just one loop, that was Tier C's job): a real multi-step training run on two assembled
graphs. Landed via the same reflective multi-agent workflow pattern as Tier C, run as two
independent pipelined tracks (Draft → Verify Draft → Fix & Generate → Julia Test →
Adversarial Re-verify each), so neither track waited on the other.

**MNIST-MLP track — VERIFIED CLOSED.** `scripts/generate_tier_d_mnist_fixtures.py` builds
the `examples/mnist_pc.jl` architecture (`x(784) → h(128, tanh) → y(10)`, plain non-muPC
config) fed a SYNTHETIC batch (not real MNIST — TFDS is unavailable in the fixtures venv;
the fixture's claim is "the PC training loop computes the same thing given the same input,"
not TFDS/real-MNIST parity, documented explicitly in the generator's docstring so a future
reader can't assume otherwise), chained across **5 real train_steps on 5 distinct batches**
— the property that distinguishes this from Tier C's single-step scope.
`test/conformance/fixtures/tier_d_mnist.npz` (41 arrays) →
`test/conformance/test_tier_d_mnist.jl` (rtol/atol 1e-4, `allclose_d`/`FIX_D`, comment citing
"Float32 BLAS associativity across multiple inference steps" as the reason for the looser-
than-Tier-C bound) asserts params0, all 5 step energies, final trained params, and the full
final `GraphState`. **31/31 assertions passed; full-suite regression 703/703 green.**

Verify Draft caught a real, reproduced crash before generation: the terminal "x" node
defaulted to `use_bias=True`, giving `params.nodes["x"]` a non-empty `{"b": ...}` biases
dict, but `compute_local_weight_gradients` unconditionally returns an EMPTY grads dict for
any in-degree-0 node — so `optax.apply_updates` hit a pytree-structure mismatch and crashed
on the first `train_step`. Fixed to `use_bias=False`, matching Tier C's own established
precedent for terminal source nodes exactly. Adversarial Re-verify repeated Tier C's
mutation-proof standard: a 10× learning-rate mutation produced an **11-pass/20-fail split
that was independently predicted from reading `forward_and_latent_grads` before running**
(the 11 passes are exactly node "x"'s 6 state fields + `params0`'s 4 LR-invariant arrays,
mathematically LR-invariant since "x" carries no trainable params) — ruling out both a
vacuous test (nothing failed) and an over-brittle one (everything failed regardless of
relevance). File restored byte-identical (md5-verified) and reconfirmed green afterward.

**Transformer-LM track — VERIFIED CLOSED (2026-07-14), via a stability-conditioning
diagnosis, not a tolerance fit.** `scripts/generate_tier_d_transformer_fixtures.py` assembles
the real `src/models/transformer_lm.jl` topology (`seq_len=8, vocab_size=10, embed_dim=8,
num_heads=2, num_blocks=1`: `input → embed → transformer_0 → skip_0 → output`, all four node
types Tier B already validated in isolation) plus one upstream-only addition — an explicit
mask node (`IdentityNode`, clamped via a third `TaskMap` task `causal_mask`) feeding
`TransformerBlock`'s "mask" slot, since Julia's `TransformerBlock` has no such slot at all
(inline `causal::Bool` instead) — a real, intentional, documented topology divergence, not a
bug. `VocabProjectionNode` is built with upstream's own class defaults
(`SoftmaxActivation`/`CrossEntropyEnergy`), matching `transformer_lm.jl`'s actual behavior —
an earlier draft of this fixture had it backwards (Identity+Gaussian), traced to a STALE
top-of-file comment in `transformer_lm.jl` itself; corrected against the real struct default
and `git blame` (`565f9fb`), not the comment. All three Verify Draft lenses (API fidelity,
tolerance-and-comment discipline, and a 4th lens checking the fixture's topology against
`transformer_lm.jl` line-by-line) verdicted CORRECT before generation.

The argument for closure is built in this order — **read the control before the fix**, or it
reads exactly like tolerance-fitting by another name:

**1. Port fidelity is established independently of anything below.** Tier B's 151/151
isolated node-local gradient conformance, plus this track's own pre-relaxation checks
(`params0` 19/19, `initialize_graph_state` 29/29 — graph construction, embedding lookup,
RoPE, causal masking, LayerNorm, GELU MLP, and the residual/vocab-projection stack all
correct pre-relaxation) are unaffected by anything that follows.

**2. The MNIST control: contractive at η=0.1, passes 1e-4 — same harness, same tolerance,
same η.** The PC inference-relaxation loop is literally gradient descent on energy w.r.t.
latents, iteration matrix `(I − η·H)` — contractive (cross-implementation float32 noise stays
bounded) iff `η < 2/λmax`, expansive (noise compounds exponentially, independent of port
correctness) otherwise. Measured directly on Tier D-MNIST's own graph (`x(784)→h(128,tanh)
→y(10)`, **same eta_infer=0.1** as the transformer config below — the contrast is NOT a
gentler η) via perturb-and-track (κ≈0.902/step) AND power-iteration on the FD-linearized
one-step Jacobian (ρ≈0.90001, agreeing to 4 significant figures) — implying `λmax≈1.0` for
MNIST's one free node, ~100× smaller curvature than the transformer's. MNIST is contractive
because it's well-conditioned at η=0.1, not because it uses a smaller η.

**3. The transformer at η=0.1 is measurably expansive — this is what 146/175 was actually
measuring.** Same two methods on the transformer tiny config: power-iteration crossing point
`η*≈0.0176` (`λmax≈113`, superseding this investigation's own earlier single-point estimate
of λ≈36 — that estimate fit ONE eigenvalue to the endpoint growth rate, which cannot describe
a stiff system; see the condition-number paragraph below for why that mattered). At the
fixture's actual η=0.1, the linearized map's spectral radius is **ρ≈8.80 — ~5.7× past its own
stability limit**, and direct perturb-and-track confirms it operationally: tail per-step
amplification ~1.24–1.27×, compounding to **~25,000–45,000× over the 12-step relaxation** for
either of two tested seed magnitudes. Two independent float32 implementations of a map that
amplifies its own rounding error 25,000× in 12 steps cannot be expected to agree — regardless
of port fidelity — and 146/175's failures localize exactly where this predicts: 100%
post-relaxation, concentrated in `embed.latent_grad` (the node where every downstream
gradient in the graph accumulates each step — independently flagged as the single worst field
by three separate, unrelated measurements: the original per-step trace, the fori_loop-vs-eager
JAX-internal check below, and the float64 anchor).

**Condition number, and why 12 steps was never going to be enough even at a safe η.** The
slowest-decaying mode's measured tail contraction at η=0.01 (ρ≈0.977/step, from live E5
measurement, not the linear estimate) implies that mode's own `λ≈2.3`. Condition number
`λmax/λmin ≈ 113/2.3 ≈ 50` — a stiff Hessian. Even at the best possible stable η, the slow
mode only contracts ~0.96–0.98/step; **12 steps buys `0.96¹²≈0.6` of relaxation, not
convergence** — real relaxation of this graph needs on the order of 50–300 steps (from the
measured 0.875–0.977 per-step range), not upstream's own `infer_steps = 3·(2·blocks+2) = 12`
heuristic. This reframes what 146/175 was measuring: not primarily "does Julia match
upstream" but "does an 12-step-truncated relaxation of a stiff, non-preconditioned map even
reach a state two independent float32 implementations *could* agree on" — it could not, by
construction, independent of any port defect.

**4. The transformer at η=0.01 (still 12 steps — same iteration count, not a longer
relaxation) is contractive, and conforms at the existing tight tolerance with no change.**
Measured tail amplification 0.875–0.977/step (both tested seed magnitudes) vs η=0.1's
1.24–1.27/step — the qualitative flip predicted by η*≈0.0176. `test/conformance/fixtures/tier_d_transformer_stable_{sgd,normclip}.npz`
+ `test/conformance/test_tier_d_transformer_stable.jl` replay the full
`initialize_graph_state → relax01..relax12 → converged → get_graph_param_gradient →
train_step` pipeline and compare **523 arrays — every intermediate relaxation step, not just
endpoints (a strictly harder check than the original 175-comparison, endpoint-only test)** —
at rtol=atol=1e-4. **523/523 pass for both `InferenceSGD` and `InferenceSGDNormClip`** (the
latter giving C-05 its first conformance-harness coverage); worst required tolerance across
all 523 comparisons is 1.57e-5 (SGD) / 1.59e-5 (NormClip) — a ~6× margin, not a near-miss.
This is now the **primary, ungated** Tier D transformer-LM conformance test.

**5. Include the falsified hypothesis — a register that shows its own dead ends is worth
more than one that doesn't.** The first specific mechanism proposed for the pre-η*-diagnosis
146/175 gap was "JAX's `value_and_grad`/`has_aux` autodiff tracing reassociates float32 ops
differently than the plain eager forward used at state-init." Measured directly (E1): diffing
`TransformerBlock.forward(...)` against the `has_aux` channel of `forward_and_latent_grads(...)`,
identical params/inputs/state — **exactly 0.0 difference**, for every node. This is CORRECT
behavior, not a null result: `value_and_grad` on an un-`jit`ted function dispatches its primal
op-by-op, so there is no fusion for it to reassociate — the hypothesized mechanism doesn't
exist at that site. The real mechanism (E2) is one level up, in upstream's own
`jax.lax.fori_loop`-compiled `run_inference` vs. an eager per-step Python loop over the
*identical* `inference_step` body: these diverge by up to 1.86e-5 by step 12, purely within
JAX, no Julia involved — real, but only ~1/49th of the η=0.1 endpoint gap on its own (E2
alone does not explain the magnitude; the stability analysis above does). A float64 anchor
(E3) independently nailed the STEP-1 seed almost exactly (7.6776e-7 measured vs. 7.6e-7
originally observed, same node/fields: `transformer_0.z_mu`/`pre_activation`) — ordinary
float32-vs-float64 rounding noise, present from the very first forward pass. This also
resolves a wording overclaim: `params0`/`initialize_graph_state`'s "bit-exact" language
meant "passes the 1e-4 gate," not literal bitwise equality — Julia's first forward pass sits
7.68e-7 from the float64 "truth," and upstream's own float32 sits the same 7.68e-7 from it.
**Julia is as close to the exact answer as upstream is** — correct to float32 precision,
verified against an independent float64 anchor, which is a stronger and more precise claim
than "bit-exact" ever was.

**What this closes, and what it explicitly does NOT.** The defensible claim: *conformance
holds at tight tolerance (1e-4, with a measured ~6× margin) wherever this relaxation is
well-conditioned; where it is expansive, cross-implementation disagreement measures the
configuration's conditioning, not the port's fidelity* — derived from direct measurement
(perturb-and-track AND power-iteration, cross-checked against each other on both graphs), not
fitted to the discrepancy it explains. What it does **not** license: any claim that
`transformer_lm()` run at ITS OWN production default (`eta_infer=0.1`, e.g.
`examples/char_lm_pc.jl`/the Shakespeare demo) is verified-conformant against upstream —
that config runs in the measured-expansive regime, so its converged latents are not expected
to be reproducible across BLAS threadings or machines, let alone across languages. State this
limit plainly rather than let it be inferred from the closure above.

**Disposition.** `test/conformance/test_tier_d_transformer.jl` / `tier_d_transformer.npz`
(η=0.1) are **demoted, not deleted** — the `relax01..relax12` per-step dumps are the most
valuable diagnostic asset this investigation produced. `test/conformance/test_tier_d_transformer_stable.jl`
(η=0.01, both `InferenceSGD` and `InferenceSGDNormClip`) and
`test_tier_d_transformer_stable_normclip_boundary.jl` (the boundary-straddle C-05 coverage,
section 4) are wired in **ungated**, inside the main `@testset "FabricPC.jl"`, as the track's
real conformance target. The demoted η=0.1 file is kept gated behind
`FABRICPC_TIER_D_TRANSFORMER=1` — but pulled **outside** the main testset into its own
independent, try/caught testset (a first pass nested it inside "FabricPC.jl," which made the
whole suite throw `LoadError` whenever the gate was set — a nested `@testset` only defers
its failures to the parent's tally, it doesn't swallow them; only a non-nested testset's own
`TestSetException` can be caught without also catching genuine regressions elsewhere).
**Verified via a live run**: default suite (gate off) — `FabricPC.jl | 2309 2309` — every
prior tier plus the new stable SGD (1046 = 523×2 variants), NormClip-boundary (560 = 523
value comparisons + 1 straddle-exists check + 36 discrete clip/no-clip decision checks, all
36 near-boundary points, zero mismatches), on top of the 703 pre-existing baseline. With
`FABRICPC_TIER_D_TRANSFORMER=1` set, the demoted testset runs and prints its own 146/175
breakdown (identical to the numbers above, confirming reproducibility) but is caught before
it can affect "FabricPC.jl"'s own verdict — the script still completes without throwing.

**Two findings that outlive this specific closure:**

- **The η=0.1 default is a live problem in the codebase this investigation is about, not
  just a fixture artifact.** `transformer_lm.jl`'s own production default is `eta_infer=0.1`
  against a measured `η*≈0.0176` for the (much smaller) tiny diagnostic config — confirmed
  the SAME default exists upstream too: `examples/transformer_demo.py`'s
  `create_transformer_model` (the monolithic-`TransformerBlock` family `transformer_lm.jl`
  actually ports) defaults `eta_infer=0.1` in both its CLI arg and function signature — this
  is an upstream default, not a Julia-introduced one, and worth reporting back as a real
  ecosystem contribution if confirmed at production scale. Suggestively (not conclusively —
  different node family, different scale, so stated as a parallel, not evidence): the
  *decomposed* model family's own demo (`examples/transformer_v2_demo.py`, `MhaResidual`→
  `LnMlp1`→`Mlp2Residual`, embed_dim=64, depth=2 — a different graph from this track's) ships
  a `CHAR_DEFAULTS.eta_infer = 0.0174852165627398`, empirically arrived at via what its own
  comment calls "Phase 2 refined lr/eta_infer/infer_steps" hyperparameter search — within
  0.1% of this session's independently-measured η*≈0.0176 for an unrelated tiny config.
  **Direct consequence for future work, not just a curiosity: a C-04 (PC-vs-backprop
  comparison) arm built on a TRANSFORMER at η=0.1 would be benchmarking a non-converged
  inference loop** — a confound sitting directly under any eventual fidelity/performance
  claim from that specific arm. Scoped to the transformer only: the MNIST-MLP config is
  contractive at its own η=0.1 (§6's control measurement, κ≈0.902), so an MNIST-MLP C-04 arm
  carries no such confound and needs no eta fix first. A production-scale η-sweep (this session's sweep was tiny-config
  only, explicitly not claimed to transfer) belongs in `docs/decisions.md` as a follow-up.
- **`InferenceSGDNormClip` conformance: the boundary-untested gap is now CLOSED, positively
  (section 4, C-05).** A second fixture (`max_norm=2.5`, re-derived after a naive first guess
  was caught and rejected — clipping one node cascades through the graph and can suppress a
  DIFFERENT node's gradient by ~30×, invalidating a threshold read off an unclipped baseline
  in a multi-node graph sharing one global `max_norm`) produces two genuine near-boundary
  crossings (within 1% of the threshold). 523/523 at 1e-4, PLUS zero discrete clip/no-clip
  decision disagreements at either crossing, verified against upstream's own independently-
  dumped arrays. `test_tier_d_transformer_stable_normclip_boundary.jl` is wired in ungated
  alongside the primary stable test.

**C-09 follow-up (section 4): does muPC — a diagonal preconditioner on this exact iteration
matrix — reduce the condition number?** Measured directly on this graph: yes, but modestly
(~49→~41, ~16%) and by a limited mechanism (`compute_mupc_scalings` only touches two
PERIPHERAL edges in this monolithic-`TransformerBlock` topology; `λmax≈113`, the dominant
instability, lives inside the block's own internal attention/FFN Jacobian, which per-edge
scaling cannot reach). See C-09's own row (section 4) for the full numbers — this gives that
row real content for the first time on the monolithic node family; the decomposed family
(`MhaResidualNode`/`LnMlp1Node`/`Mlp2ResidualNode`) remains fully open and is a materially
different question (more inter-node edges for muPC to reach).

## 7. Documentation debt

| ID | Item | Status |
|----|------|--------|
| D-01 | README.md badly stale ("Phase A scaffold") | **VERIFIED CLOSED** (`67a54da`) — rewritten to reflect actual shipped state (muPC, transformer + decomposed transformer_v2 family, Reactant/XLA JIT, StorkeyHopfield, AdamW, natural-gradient preconditioners). |
| D-02 | `_tb_apply_rope` comment claimed a layout divergence that doesn't exist | **VERIFIED CLOSED** (`c8fb541`) — confirmed by spot-check that the math genuinely matches upstream (see §3); reworded the comment to say the convention coincides with upstream's rather than implying no match was attempted. |
| D-03 | transformer_decomposed.jl header calls Embedding/VocabProjection "a separate follow-up" — they're implemented below it in the same file | **VERIFIED CLOSED.** One-line comment fix, cross-referencing this ID. |
| D-04 | decisions.md entries: X-01…X-05, layer map (A-02), backend roles | **VERIFIED CLOSED.** Three new decisions.md entries: §21 (X-01…X-05, full rationale per item, X-05 additionally resolved not just documented), §22 (layer map — the four execution "layers" and the gating rule for validating each against the one below), §23 (backend roles — Zygote/eager-Enzyme/Reactant+Enzyme, synthesizing §19 with today's J-01 scouting findings). |
| D-05 | FabricPC.jl had no published documentation site (Documenter.jl + GitHub Pages) — every sibling package in the org (Core/MORK/PathMap/NGCSimLib) has one | **VERIFIED CLOSED.** `docs/make.jl`/`docs/Project.toml`/`docs/src/` set up via `Pkg.develop`+`Pkg.add`+`Pkg.compat` (no hand-typed UUIDs), mirroring NGCSimLib's exact pattern (verified against the sibling repo's actual `make.jl`/CI workflow, not guessed). Pages: Home (adapted from README), Getting Started (Installation/Quickstart/Architecture/JIT with Reactant), API Reference (`@docs` blocks grouped to match `src/FabricPC.jl`'s own export sections). `.github/workflows/Documenter.yml` added for the `gh-pages` deploy (mirrors NGCSimLib's workflow). Local build verified clean (`julia --project=docs docs/make.jl`) — zero unresolved `@ref`/missing-docstring errors after adding real docstrings to 4 node types (`MhaResidualNode`/`LnMlp1Node`/`Mlp2ResidualNode`/`EmbeddingNode`/`VocabProjectionNode` had none at all — only section-header comments) and ~12 abstract-type/generic-dispatch symbols that had no docstring anywhere (`AbstractNode`/`AbstractEnergy`/`AbstractActivation`/`AbstractInitializer` + `energy`/`grad_latent`/`grad_mu`/`jacobian`/`jacobian_gain`/`variance_gain`/`initialize`/`get_slots`/`to_flat_params`/`to_flat_state`). Found and fixed one real pre-existing bug along the way: a `"""Learnable parameters of one node..."""` docstring in `core/types.jl` was orphaned onto the wrong struct (`SoA` instead of `NodeParams`) by an intervening comment block breaking Julia's docstring-to-next-expression binding — moved to `NodeParams` and gave `SoA` its own accurate docstring. |

## 8. Residual review queue

| ID | Pair | Status |
|----|------|--------|
| R-01 | train_autoregressive.jl vs .py (full) | **VERIFIED CLOSED — genuinely full this time** (the prior "closed" claim in this row was itself overclaimed and corrected; see git history). Every function pair now covered: formatting-safety (whitespace-only, `ecb031a`); one-hot/clamp contract (F-05); `_eval_step_autoregressive`/`evaluate_autoregressive` (C-06); generation index-cast (int32 upstream / native-Int Julia end-to-end, same divergence X-01 already covers); `_sample_next`'s sampling formula (temperature/top-k/top-p/Gumbel-max all semantically match `_generation_step`'s upstream formula — only measure-zero edge-case tie-breaking differences at exact-equality boundaries, plus a Julia-only defensive all-masked/NaN fallback that's a safety addition, not a regression); and — the piece that was actually still missing — the `train_autoregressive` epoch-loop driver itself, which turned up **X-06** (3 real, dormant-but-named control-flow gaps, one of them a genuine blocker for C-10), independently re-verified by an adversarial pass before being written up. Coverage caveat (stands): `test_transformer_lm.jl`'s "(c) greedy generation" testset is a self-consistency smoke test, not a byte-level fixture against upstream JAX output like Tier A/B — no fixture proposed for generation as part of this item. |
| R-02 | FabricPCZygoteExt / FabricPCEnzymeExt internals | **VERIFIED CLOSED.** Enumerated every SoA structural touch point reachable from `compute_mu`/`energy_kernel` via the generic seam. `_soa_key`/`_soa_keys` (Symbol→String foreigncall) are the only two conversion helpers, both `Zygote.@nograd`'d; `keys`/`pairs`/`iterate` delegate to them. `Base.getindex`/`haskey`/`get` on `SoA` (the reverse, String→Symbol direction) are NOT `@nograd`'d and ARE reached with a dynamic key on the latent-grad path (`MhaResidualNode`/`Mlp2ResidualNode`/`StorkeyHopfield`) — verified this needs no guard via three independent checks: a fresh isolated Zygote probe (literal + non-foldable runtime key, both differentiate cleanly), a pre-existing session artifact confirming the same, and two FD-verified conformance tests (`test_transformer_decomposed.jl`'s Mlp2Residual testset, `test_storkey_hopfield.jl`) already exercising this exact path end-to-end with numeric correctness assertions. Enzyme side applies zero `@nograd`-equivalent markers — correctly so, its LLVM type-based activity analysis treats Symbol/String/Bool-producing code as inactive automatically; the one thing its model does need (`set_runtime_activity(Reverse)`) is present and consistent on both `_ad_param_grads`/`_ad_latent_grads`. The known Enzyme MHA-crash limitation is a distinct LLVM array-accumulation issue, unrelated to SoA key machinery and already tracked (decisions.md §19/§23) — out of R-02's scope. |
| R-03 | storkey_hopfield pair (resolves C-11 init question) | **ADDRESSED** as part of the C-11 close — see C-11. |
| R-04 | transformer_v2.py (upstream HEAD after A-01) | **VERIFIED CLOSED** (decisions.md §21, X-05). GraphNamespace half: upstream's own `transformer_demo.py` doesn't use it either (grep-verified). Stage-boundary half: already closed by Tier B's byte-level fixture verification of `MhaResidualNode`/`LnMlp1Node`/`Mlp2ResidualNode` against upstream (section 6) — a boundary mismatch would show up as a fixture failure, and none did. |
| R-05 | Spot-checks: SoftmaxActivation.jacobian, last-axis softmax on rank-3, graph_construction, natural_gradients pair | **VERIFIED CLOSED.** SoftmaxActivation.jacobian: Tier A's fixture generator calls upstream's `SoftmaxActivation.jacobian` directly (not a re-derivation); `test_tier_a.jl` compares Julia's `jacobian(::SoftmaxActivation,...)` against it byte-level, rtol 1e-6 — genuinely covers the upstream method. Last-axis rank-3 softmax: confirmed `forward(::SoftmaxActivation,x)` genuinely uses `dims=ndims(x)` (not hardcoded); Tier B's `embedding_vocab_vocab_fwd` fixture exercises exactly this via `VocabProjectionNode` on a real rank-3 (B,S,V) tensor against upstream's own output, rtol/atol 1e-5; `test_transformer_decomposed.jl` additionally self-checks the same normalization axis. natural_gradients.jl: read both sides in full — diagonal and layerwise Fisher preconditioners match upstream's formula (`f←decay·f+(1−decay)·g²` or `mean(g²)`, then `g/(f+damping)`) exactly, same update-then-precondition ordering, same hyperparameter validation bounds; corroborated by `test_initializers_natgrad.jl`. **graph_construction.jl: REAL GAP FOUND AND FIXED.** Julia's `graph()` pushed every edge's key into an ordered list (`edge_keys_ordered`) unconditionally, unlike upstream which relies solely on a Python dict's natural key-uniqueness (`graph_construction.py:137-145`) — a literal duplicate `Edge` (same source/target/slot submitted twice) inflated `NodeInfo.in_edges`/`in_degree` to count it twice instead of once. Silently absorbed (harmless) everywhere a consumer re-keys by edge key into a `Dict` (`gather_inputs`, `initialize_params`, `compute_mupc_scalings`), but NOT absorbed in the JIT flat path — `CompiledPlan`'s `in_src`/`in_key` are plain, un-deduped `Vector`s, so `flat_latent_grads` would silently double-count both the forward contribution and the backward gradient of a duplicated edge. Dormant (no shipped model constructs a literal duplicate edge) but a real, reachable silent-corruption landmine specifically in the code J-03 (section 5) just extended this session. **Fixed**: `push!` to the ordered list only on a key's first occurrence (matching Python dict insertion-order semantics — later occurrences overwrite the value at the original position, not append a new one). Verified the fix propagates into `CompiledPlan` (duplicate edge now produces exactly one `in_src` entry) and that flat/eager inference still match bit-for-bit post-fix. Regression test: `test/test_graph_construction.jl` (duplicate edge → `in_degree`/`in_edges` identical to the single-edge case; graph still trains). 572/572 green. |

## 9. Execution sequence

**Batch 1 — correctness gate: DONE, informally.** The reflective cross-check refuted the original
hard-gate design ("F-01+F-02 fixed and green *before* any batch-2 work") — nothing was live-broken,
and the two bugs were independently fixable/testable, not a coupled unit. F-01 (TransformerBlock)
+ F-02 + F-03 + F-04 landed as one batch (`c8fb541`) with independent, targeted regression tests
rather than a combined fixture-gated test. No fixture generator was needed to close Batch 1.

**Batch 2 — validated showcase: DONE.** C-03 (Tiny Shakespeare char dataloader) → C-05
(InferenceSGDNormClip) → C-06 (evaluate_autoregressive) — all **VERIFIED CLOSED**. C-03 landing
first was deliberate: highest ROI (turns the already-mature LM stack from synthetic-only to
demonstrably working on real text) and unblocks C-06 being meaningful (nothing to evaluate
perplexity on without real data) — confirmed: C-03's live smoke test found zero OOV chars between
the train vocab and validation/test splits, so C-06 evaluates on either split with no additional
vocab-handling work. A discovery mid-sequence (F-05, found during C-03) turned out to gate both
C-03 and C-06: neither raw-id dataloader batches nor eval's free-running `"y"` scoring could work
until the one-hot-expansion-at-the-clamp-site fix landed — C-05 was independent (the reflective
cross-check's original ordering had it first) and shipped in between. Per A-01/C-09: this batch did
not turn on `MuPCConfig` together with `TransformerBlock`/the decomposed family (`transformer_lm`
tests are muPC-off) — C-09 (MhaResidualNode/LnMlp1Node under real muPC scaling) remains untested,
unchanged.

**Batch 3 — performance claim: OPEN, not started.** J-01 → J-02 → J-03 (+J-07) → J-04 benchmark vs
`jax.jit`. Precondition for any "surpassing JAX" statement.

**Continuous:** D-01…D-05 all closed (Documenter.jl + GitHub Pages site stood up, decisions.md
§21-23 written). R-01…R-05 all closed. R-01's first "closed" pass was itself an overclaim
(caught, corrected, then genuinely finished — see R-01's own row for the full history) and
turned up X-06 (`train_autoregressive` driver gaps, a named blocker for C-10) on the real pass.
R-05's graph_construction spot-check found and fixed a real (dormant) duplicate-edge
double-counting bug in the JIT flat path. C-04 (reclassified, not blocking) remains as its
inputs arrive.

**Explicitly deprioritized (agreed, both audits, unchanged):** C-01/C-02 conv stack, C-10 tuner,
dashboarding/pmap/tfds.
