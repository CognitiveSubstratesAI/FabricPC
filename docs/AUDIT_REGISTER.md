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
| A-01 | Re-sync Python reference to upstream HEAD | **CLOSED 2026-07-15 — re-synced to `b6f64ad`; fixtures regenerated against it.** The drift A-01 was re-opened for is gone: fixtures and the reference checkout now both sit at upstream HEAD `b6f64ad` (was: fixtures at `5514c91`, HEAD moved ahead by one contract-changing commit). Adaptation in C-15, decision in C-16. **A-01's own lesson held twice over.** (1) *The SHA is the only real marker* — upstream's `pyproject.toml` still says `0.3.1`, unchanged since 2026-05-04 and now several merged PRs stale; only `scripts/requirements-fixtures.txt`'s editable-install line records what actually produced the arrays. It is now updated to `b6f64ad`. (2) *The oracle's source moved under us mid-session*: `~/PRIMUS/dev-zone/FabricPC` symlinks to `~/JuliaAGI/dev-zone/FabricPC`, which is exactly what `.venv-fixtures` imports via its EDITABLE install — so the user's `git pull` silently re-pointed the oracle without touching this repo at all. That is why regeneration is gated on a bit-identity check against the previous fixtures (C-15) rather than on the generator merely running: a moved oracle fails OPEN. |
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
| F-06 | LOW (dormant) | `TransformerBlock` output-activation divergence | **CLOSED 2026-07-16 — matched upstream.** Verified against `transformer.py`'s forward BODY (not our stale comment): upstream finalizes `z_mu = inv_sqrt2*(x_res1 + ff_output)` and returns it WITHOUT applying the output `activation` — the constructor stores `activation` (line 178) but forward never reads it (a genuine dead param upstream). We had `return forward(node.activation, z_mu)`, which matched upstream ONLY under the IdentityActivation default (a no-op) and would silently diverge for any non-Identity output activation. Now `return z_mu`; `node.activation` retained for constructor-signature parity but not applied, matching upstream. **Correspondence check caught a near-mistake:** three OTHER nodes also call `forward(node.activation, ...)` — `LnMlp1Node` (:171), `VocabProjectionNode` (:402), `StorkeyHopfield` (:107) — but upstream DOES apply the output activation in each (`act_obj = node_info.activation`, applied), so those are CORRECT and were left untouched. Only the monolithic `TransformerBlock` (transformer.py) has the dead param. tier_b stays 147/147 (transformer 58/58) — the fix is behaviour-preserving under the Identity default. |

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

**2026-07-14 upstream cross-check — independent validation, no Julia change needed.** Upstream's own
`CHANGELOG.md` `[Unreleased]` documents a related-but-distinct bug: `eval_step_backprop`'s accuracy
computation (`train_backprop.py:475-518` — the backprop anti-thesis baseline, confirmed absent from
this port per section 4's "Known-deferred, confirmed absent" line) used a shape-*value* heuristic
(`targets.ndim > 1 and targets.shape[-1] > 1`) to decide one-hot-vs-raw-id, which misread integer
sequence targets `(batch, seq_len)` as already one-hot whenever `seq_len > 1` and argmaxed over the
wrong axis; the fix replaces it with a rank comparison against `predictions`. Checked Julia's own
`compute_loss`/`train_step_autoregressive`/`_eval_step_autoregressive` bodies
(`src/training/train_autoregressive.jl:30-104,174-199`) for the same bug class: it cannot occur here.
`compute_loss` itself carries no shape/rank check of its own by design (`:74-76`'s docstring:
expansion happens exactly once, at the caller); the actual decision lives in
`train_step_autoregressive` (`:88`, `ndims(y) == onehot_ndims - 1`) and `_eval_step_autoregressive`
(`:186`, identical pattern) — both compare `ndims(y)`, a RANK, against
`onehot_ndims = length(structure.infos[output_node].shape) + 1` (the declared output rank), never a
`size(y, ndims(y))`/`shape[-1]` VALUE. `evaluate_autoregressive`'s accuracy path (`:269`,
`_count_correct(predictions, y)`) only ever receives the already-rank-normalized `y` from
`_eval_step_autoregressive`, and `_count_correct` is itself statically typed
`AbstractArray{<:Real,3}` on both arguments (`:209`) — a shape-value-heuristic regression at that
callsite would be a `MethodError`, not a silent misread. Julia's F-05 fix (closed before this
upstream CHANGELOG entry existed) independently converged on the same rank-based contract upstream's
own fix now canonicalizes — this is confirmation, not a new finding; no action taken.

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
| X-02 | Mask slot removed from TransformerBlock | DOCUMENTED (decisions.md §21) — locked in. **2026-07-14: confirmed IMMUNE BY CONSTRUCTION to a real upstream generation bug** (upstream `CHANGELOG.md` `[Unreleased]`) — see decisions.md §21. |
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
| C-01 | HIGH | ConvNode (1D/2D/3D) + pooling + windowed-shape validation | **VERIFIED CLOSED (2026-07-15).** ConvNode (1D/2D/3D) + MaxPool + AvgPool (windowed & global) ported, wired, and conformance-gated against upstream JAX. `test/conformance/test_tier_conv.jl` **51/51**; **full suite 2538/2538**. Fixtures: `scripts/generate_tier_conv_fixtures.py` -> `fixtures/tier_conv.npz` (121 arrays, jax 0.10.2, pinned venv). Both sides are AUTODIFF — upstream conv/pool implement only `forward` and inherit NodeBase's `jax.value_and_grad` (`base.py:507-551`); ours implement only `compute_mu` and inherit the Enzyme/Zygote seam — so this compares AD-to-AD across two stacks, and BOTH agree with upstream (Zygote via the tier; Enzyme via C-14). **Design: hand-rolled im2col, ZERO new deps** (NNlib ranked third: it would have imported a KernelAbstractions-class precompile into a package that made Enzyme itself a weakdep to avoid exactly that, and shipped a known `nextfloat(typemin(T))` vs `-Inf` divergence — decisions.md §29). **Method (the transferable part): ORACLE BEFORE KERNEL, and the treacherous half first** (decisions.md §30). Fixtures were generated before a line of kernel existed, which let measurement ARBITRATE the design rather than merely check it — three design claims went to the oracle and **one survived**: tie-routing (reasoning said window order was an implementation detail; the oracle said JAX breaks ties ROW-MAJOR — measured on a PARTIAL tie, since an all-tied window passes under both orders — while our im2col patch layout is COLUMN-major and would have shipped wrong), B5 `pre_activation` (called a rounding nit; measured `max|pre - z_mu| = 4.81` for a ReLU conv — **since SUPERSEDED: C-16 adopted upstream b6f64ad, which removes the field entirely, so the `compute_pre_and_mu` seam hook B5 added is deleted. The finding was right (the seam WAS reporting `pre = z_mu`, which is wrong for a ReLU); upstream's answer was to stop storing the quantity rather than to report it correctly. Kept here because the reasoning is what generalizes, not the code it produced**), and "square kernels could hide a transposition" (dissolved: random weights already broke the mask). **Tolerances were PREDICTED before either kernel ran and both landed**: conv at the reassociation floor (~1e-7 — im2col GEMM vs XLA's fused VJP are different algorithms, so rtol 1e-5 is the right instrument and a ULP gate would fail on correct code), pooling BIT-IDENTICAL (pure reduce_window, no GEMM to reassociate) — including the col2im scatter on the maximal-accumulation case (9 contributions/pixel). Prerequisites landed separately while green (`df1d470`): C-08 ND fan-in, rank-4 params through 11 optimizer sites, the `compute_pre_and_mu` hook, and a LATENT seam fold-order bug. Two half-closed loops caught during the build are recorded at A-03(4). **Known scope limits, stated rather than implied:** dilation is unsupported (upstream doesn't support it either — parity, not a gap); im2col's patch matrix costs `∏(window)` memory per edge per node per inference step, which is fine at MNIST scale and **unprofiled at ResNet scale** — batch-blocking is the mitigation and should be scoped from a profile, not from this note (`feedback_no_perf_attribution_without_profiling`). Unblocks C-02 (ResNet-18/CIFAR-10). |
| C-02 | HIGH | ResNet-18/CIFAR-10 demo | **UNBLOCKED (2026-07-15) — C-01 closed.** The node family it needs (ConvNode 1D/2D/3D + MaxPool + AvgPool incl. global) is ported and conformance-gated; upstream's own demo is `examples/resnet18_cifar10_demo.py`. Two things to scope before starting, both already known rather than discovered later: (1) im2col's patch matrix is `∏(window)` memory per edge per node per inference step — fine at MNIST scale, **unprofiled at ResNet scale**; batch-blocking the gather is the mitigation and should be scoped FROM A PROFILE (`feedback_no_perf_attribution_without_profiling`), not from an a-priori guess. (2) C-08's ND fan-in (closed, `df1d470`) is what makes Kaiming size a conv kernel correctly — a ResNet is precisely where a 4.9x-too-wide init would have shown up as a training mystery rather than a test failure. |
| C-15 | HIGH | Upstream `forward()` contract changed (b6f64ad): `pre_activation` dropped from `NodeState`; `forward()` returns `NodeState` only | **CLOSED 2026-07-15 — generators adapted, fixtures regenerated, faithfulness PROVEN by bit-identity.** All 8 `scripts/generate_tier_*.py` now run against upstream HEAD `b6f64ad`. The fix itself was small; **the near-miss is this entry's real content.** My first pass used a broad regex over `.forward(` and it did the exact thing this register exists to catch: (a) it matched ACROSS a kwarg boundary in `generate_tier_a_fixtures.py`, turning `x=x_outlier,` + `fwd=SoftmaxActivation.forward(...)` into `x=fwd = SoftmaxActivation.forward(...)` — **silently deleting the `act_softmax_outlier_x` fixture key**, on an ACTIVATION forward that was never in scope; and (b) it dropped scalars (`total_energy`, `total_e`, `total_ev`) that are USED downstream (`energy=jnp.reshape(total_energy, (1,))`). Caught by READING THE DIFF, not by any test — a corrupted oracle still runs. Reverted and redone with balanced-paren parsing restricted to exact two-name unpacks; discarded scalars are REPLACED with `jnp.sum(state.energy)` (upstream's own post-b6f64ad test idiom), not deleted. **The fail-open hazard this entry was opened for was real and is now gone:** `_, fwd = node_class.forward(...)` does not raise against a bare `NodeState` — NamedTuple unpacking silently binds `_`/`fwd` to the first two FIELDS (`z_latent`/`z_mu`) and writes garbage fixtures. **Faithfulness gate (predicted, then measured):** since the upstream change is pure plumbing it must be numerics-neutral, so regenerated fixtures must be BIT-IDENTICAL to the `5514c91` ones apart from dropped keys. Measured across tier_a/b/c/conv: keys 69→69, 217→213, 106→91, 121→109; **every dropped key is a `*_pre_activation`, zero added, and all 482 shared arrays bit-identical.** That one check proves three things at once: upstream's refactor moved no numbers, my adaptation corrupts nothing, and the old fixtures were valid. |
| C-16 | MED | Adopt upstream's `pre_activation` removal, or diverge deliberately | **CLOSED 2026-07-15 — ADOPTED.** `NodeState` goes 6 fields → 5; `forward()` returns `NodeState` alone and the batch→scalar `sum(ns.energy)` moves to the callers that actually need a scalar (the AD seam, which must hand one to `value_and_grad`) — mirroring upstream 1:1 rather than inventing our own shape. `state.energy` was NEVER the issue: it is per-sample `(batch,)` on both stacks and always was (`core/energy.py` sums `axis=tuple(range(1, ndim))` and b6f64ad did not touch it; ours is `src/core/energy.jl:66-71`). I had flagged 'does our forward return per-sample or summed energy? if summed, every energy comparison passes on the wrong quantity' as load-bearing — **that worry is REFUTED**: every tier asserts `ns.energy` against a per-sample fixture key at `B≥2` with independently-sampled targets, so a scalar-collapsed energy would fail, not pass. The one node that genuinely needs a pre-activation is `Linear` (≡ upstream's `LinearExplicitGrad`, per `feedback_verify_the_correspondence_not_just_the_code`) for `f'(pre)`; it and `LinearResidual` now get it from `_forward_with_preact`, and `pre_grad(node, state, pre)` takes it explicitly — exactly upstream's `compute_gain_mod_error(pre_activation, error, node_info)` change. The seam's `compute_pre_and_mu` hook (B5) is DELETED: it existed only to fill the removed field, and nothing on that lane ever read it. **Payoff beyond conformance:** per-node inference state drops from 5 `(batch, features...)` tensors to 4 — a real allocation saving in the in-place lane, which builds one per node per step. |
| C-14 | **HIGH** | Enzyme seam crashes on every PARAMETER-FREE node (`Duplicated(::@NamedTuple{})` is a ghost type) | **REOPENED then FIXED (2026-07-15). My earlier `VERIFIED CLOSED — AS UNNECESSARY` was WRONG, and HOW it was wrong is this entry's real content.** I closed C-14 claiming *"Enzyme is correct on ConvNode single-edge AND 3-in-edge"* — true, and **verified only on ConvNode WITH bias: the one shape that cannot produce an empty NamedTuple**. The C-01 adversarial audit caught it by quoting that commit message back — the verification was scoped to an input that structurally cannot exhibit the failure. Same class as the J-06 guard (decisions.md §30 corollary 2, verb *guarded*), one level up: **"Enzyme works on ConvNode" hid ON WHICH SHAPES.** **The real bug:** an empty NamedTuple is a zero-size GHOST type Enzyme refuses to mark differentiable — reproducible with zero FabricPC involved: `Enzyme.autodiff(set_runtime_activity(Reverse), a->1.0f0, Active, Duplicated(NamedTuple(), NamedTuple()))` → *"Type of ghost or constant type Duplicated{@NamedTuple{}} is marked as differentiable"*. TWO real shapes hit it: **every `PoolNode`** (parameter-free by construction) and **`ConvNode(...; use_bias=false)`** (empty biases). `core/learning.jl:32` calls `forward_and_weight_grads` for every `in_degree>0` node, so **no graph containing a pool could take a training step on the Enzyme backend** — i.e. every CNN. **Why it shipped green — three stacked blind spots, all mine:** (1) the pool testsets called only `forward`/`forward_and_latent_grads`, never `forward_and_weight_grads`, so the param-free weight-grad lane had NO test in EITHER backend; (2) the suite is Zygote-only (`runtests.jl:15`; F-04 forbids co-loading), so no committed test runs conv/pool through `FabricPCEnzymeExt`; (3) the tier hand-builds isolated NodeParams/NodeState and never calls `graph()`, so `compute_local_weight_gradients → pool → _ad_param_grads` was untested outright. **Fix:** `_act(nt, dnt) = isempty(nt) ? Const(nt) : Duplicated(nt, dnt)` in `ext/FabricPCEnzymeExt.jl`'s `_ad_param_grads`. A param-free node's gradient IS the empty NamedTuple; `map(zero, ::@NamedTuple{})` already returns `@NamedTuple{}`, so the `SoA` repack and F-03's key-set check are unaffected. **Verified on the shapes the original probe could not fail on** (`test/dual_ad_backend_env`, Enzyme without Zygote): MaxPool ✅ AvgPool ✅ AvgPool(global) ✅ ConvNode(use_bias=false) ✅, plus a CONTROL that ConvNode(use_bias=true) — the original probe's own shape — is unregressed ✅. Blind spot (1) is now closed by the `pool/conv: PARAM-FREE weight-grads` testset in `test_tier_conv.jl` (non-vacuous: `use_bias=false`'s WEIGHTS still differentiate). Blind spot (2) remains structural — the Enzyme lane lives in `test/dual_ad_backend_env`, not the main suite. **The trigger design still worked** — ConvNode landing made the hazard live AND testable in the same event, and the deferral's precondition expired on schedule. What failed was not the trigger but the verification it fired: a probe that could not fail. |
| C-03 | HIGH | Real-text LM data (char tokenizer + windowed loader) | **VERIFIED CLOSED.** `examples/char_lm_pc.jl` -- `CharDataLoader`, a port of `_TokenSequenceLoader`/`CharDataLoader` (dataloader.py:272-390): raw-download (not TFDS -- no Julia binding) of the same `karpathy/char-rnn` source TFDS uses, split with TFDS's own verified 90/5/5 formula (`tiny_shakespeare_dataset_builder.py:52-56`), train-only vocab. Kept example-local (no `src/utils/` module), matching `mnist_pc.jl`'s existing precedent -- this port has never had a library-level dataloader. DIVERGENCE (documented, intentional): 1-based char ids, not upstream's 0-based, matching every other token-id consumer in this codebase (EmbeddingNode, `_one_hot`). Offline regression tests (`test/test_char_dataloader.jl`, 12 assertions): TFDS split formula on two boundary cases, vocab sortedness/1-basing, sliding-window shift-by-one invariant + exact ground truth, drop-incomplete-batch, `max_samples` cap, seeded-reproducible + re-iterable-across-epochs shuffling, decode round-trip. Live-smoke-tested against the real corpus (not just synthetic): split sizes exactly 1,003,854/55,770/55,770 chars (bit-exact TFDS-formula match), vocab_size=65, **zero OOV chars in validation/test against the train-only vocab** (relevant to C-06), a short real training run + generation both executed without error. 352/352 green. |
| C-04 | HIGH | PC-vs-backprop comparison harness | **MNIST-MLP arm: VERIFIED CLOSED (2026-07-14).** `benchmark/pc_vs_backprop_mnist.jl` (`ExperimentArm`-shaped, mirroring upstream `ab_experiment.py`/`statistics.py`'s opaque-`params`/`structure` treatment — a Julia backprop arm never touches FabricPC internals, not a `train_backprop.py` port). PC arm = FabricPC's own `train_pcn`/`evaluate_pcn` at the exact contractive config already validated (§6/§24, `eta_infer=0.1`, `infer_steps=30`, plain SGD `lr=0.002` — deliberately NOT retuned, so this measures that specific config, not PC's achievable ceiling). Backprop arm = Lux.jl `Chain(Dense(784=>128,tanh), Dense(128=>10))`, `Optimisers.Adam(1e-3)`. Same architecture, same real MNIST data (not synthetic — this is Julia-vs-Julia, not a cross-language conformance fixture) across 8 seeded trials each. **Result: backprop 88.00%±0.16% vs PC 74.67%±5.80% test accuracy — a 13.33pp gap, paired t-test p=0.0004, Cohen's d=−2.28 (large effect), PC also far noisier across seeds (SD 5.80% vs 0.46%).** Real, un-forced negative result for local PC learning at this untuned config — expected direction (backprop's global credit assignment vs PC's local rule), not a claim about PC's ceiling. Cross-checked against `benchmark/results.md`'s existing 81.65% report: one trial (81.35%) lands in that neighborhood, consistent with legitimate seed-to-seed variance at this config rather than a harness bug (the harness's per-epoch reshuffle scheme differs mechanically from `mnist_pc.jl`'s single-persistent-RNG scheme, so per-seed numbers aren't bit-identical to that report — expected, not a defect). Small supporting infra addition: `examples/mnist_pc.jl`'s `main()` guarded behind `abspath(PROGRAM_FILE) == @__FILE__` so its data-loader helpers are `include`-able without triggering a training run. **Transformer arm remains OPEN**, confound as scoped below. |
| C-05 | MED | InferenceSGDNormClip | **VERIFIED CLOSED.** `src/core/inference.jl` -- port of `InferenceSGDNormClip` (inference.py:312-356): per-sample L2 gradient-norm clipping on `latent_grad` before the SGD update. Upstream dispatches per-algorithm via a Python class hierarchy (`InferenceBase.update_latents` calls `cls.compute_new_latent(...)`, `cls = type(inference_obj)`) -- the Julia port had NO such dispatch point (`update_latents` hardcoded the plain-SGD formula inline, since only `InferenceSGD` existed). Extracted `compute_new_latent(inf, z_latent, latent_grad)` as a Julia-multiple-dispatch analogue of the same override point, byte-identical for the existing `InferenceSGD` arm (verified: `max_norm=Inf` ⇔ plain SGD, exactly). NOTE (scoped, documented in-file): `jit_flat.jl`'s separate Dict-free JIT/Reactant-prep inference path still hardcodes SGD inline -- `InferenceSGDNormClip` is not usable there; out of C-05's scope (a distinct subsystem with its own eager-parity test discipline). Tests (`test/test_inference_normclip.jl`): hand-computed clip-vs-no-clip ground truth, `max_norm=Inf` byte-equivalence to `InferenceSGD`, `latent_decay` parity isolated from clipping, acceptance (small PC graph trains to lower energy under norm-clipped inference). 359/359 green. **Upstream conformance coverage added 2026-07-14** (`docs/AUDIT_REGISTER.md` section 6, Tier D transformer-LM): `test_tier_d_transformer_stable.jl`'s `max_norm=1.0` variant (523/523 vs upstream JAX) exercises always-clipped/never-clipped regimes but never approaches the clip boundary itself — closed with a SECOND, purpose-built fixture (`test_tier_d_transformer_stable_normclip_boundary.jl`, `max_norm=2.5`, chosen after a first naive guess was caught and rejected: clipping one node's gradient cascades through the graph topology and can suppress a DIFFERENT node's gradient by ~30×, so a boundary target can't be read off an unclipped baseline in a multi-node graph sharing one global `max_norm`). Result: 523/523 at 1e-4, PLUS a dedicated discrete clip/no-clip decision check at 2 genuine near-boundary crossings (ratios 1.0095→0.9995 and 1.0058→0.9938, both within 1% of the threshold) — **zero decision disagreements**, verified against upstream's own independently-dumped `latent_grad` arrays (not Julia's self-consistency). The `min(1, max_norm/(norm+eps))` branch is now genuinely exercised, not coverage-by-absence. |
| C-06 | MED | evaluate_autoregressive | **VERIFIED CLOSED.** `src/training/train_autoregressive.jl` -- port of `_eval_step_autoregressive`/`evaluate_autoregressive` (train_autoregressive.py:541-710): mean cross-entropy loss, perplexity, next-token accuracy over a `test_loader` (e.g. `CharDataLoader`, C-03), with `"x"` clamped and `"y"` left FREE (the network predicts, unlike training) -- de-risked by C-03's real-corpus smoke test finding zero train/validation/test vocab mismatch. Applies the same F-05 one-hot handling (`"y"` may be raw ids or one-hot; `compute_loss` still carries no ndim-check of its own -- expansion happens once, in `_eval_step_autoregressive`, and the expanded array is returned to the caller so the accuracy computation doesn't repeat the check). DIVERGENCES (documented in-file): no `config`/`use_causal_mask` param (upstream's external causal-mask clamp branch is dead code in this port, same as `train_step_autoregressive`); `rng` threaded sequentially, no `jax.random.split`/`jax.jit` (this stack is eager throughout, matching the rest of the file). `debug=true` ports upstream's first-batch diagnostics (per-token CE, per-token intrinsic perplexity, probability on the correct token). `_count_correct` factored out (mirrors the existing `_sample_next` precedent) for an isolated, hand-computed-ground-truth test independent of the full graph/PC-inference machinery. Tests (`test/test_transformer_lm.jl`): metric shape/range/`perplexity==exp(loss)`, raw-id vs pre-one-hotted equivalence, `debug=true` doesn't perturb results, empty-loader edge case, `_count_correct` exact ground truth. 371/371 green. |
| C-07 | MED | GlobalStateInit / NodeDistributionStateInit | **CLOSED 2026-07-16.** Upstream `state_initializer.py` has THREE `StateInitBase` subclasses; we had ported only `FeedforwardStateInit`. `GlobalStateInit` (one graph-level initializer for every node, default `NormalInitializer(mean=0,std=0.05)`) and `NodeDistributionStateInit` (per-node `latent_init`) are both independent-per-node, no-forward-pass strategies — now ported 1:1 (shared `_independent_state` body + clamp overlay), exported, and dispatched via the same `initialize_state(::Strategy, ...)` mechanism as Feedforward. Verified behaviourally: NodeDistribution draws per-node + overlays clamps; Global applies one initializer (z_mu stays zero — no propagation) and honours a custom-initializer arg (the config path upstream exposes); Feedforward unchanged (still propagates z_latent←z_mu). RNG-model gap (Julia threads a stateful rng vs upstream's per-node key split) is the same one Feedforward already has and is immaterial to conformance. |
| C-08 | MED | ND Kaiming/Xavier fan-in math | **VERIFIED CLOSED (2026-07-15, `df1d470`).** `_fans(shape)` in `src/core/initializers.jl` is a verbatim port of upstream's formula shared by Xavier and Kaiming (`core/initializers.py:211-215`, `:279-283`): rank≥2 ⇒ `fan_in = prod(shape[1:end-1])`, `fan_out = prod(shape[1:end-2])*shape[end]`; rank-1 ⇒ `fan_in = fan_out = shape[1]`. Was rank-2-only (`fan_in = shape[1]`), so a `(3,3,8,16)` conv kernel took `fan = 3` instead of `72` — a **~4.9× too-wide init**, and Kaiming is ConvNode's *default* `weight_init`, so every conv would have been mis-initialised. Rank-2 is UNCHANGED **by construction**, not by test luck: `prod(shape[1:1]) == shape[1]` and `prod(()) * shape[2] == shape[2]` — a pure extension, zero regression surface. Landed with C-01's prerequisites rather than waiting for C-01 itself; 579/579 incl. Tier A-D-MNIST conformance. |
| C-09 | MED | muPC never validated on either transformer node family | Umbrella over F-01+F-02, both now closed — test coverage for TransformerBlock landed with F-01's fix; MhaResidualNode/LnMlp1Node/Mlp2Residual under real `MuPCConfig` remains OPEN. **Monolithic `TransformerBlock` half now has real CONTENT (2026-07-14, `docs/decisions.md` §24 follow-up):** measured muPC's effect on the SAME condition number characterized while closing Tier D — `MuPCConfig` reduces `λmax/λmin` from **~49 → ~41 (~16%)**, `include_output=True` vs `False` making no measurable difference. This is NOT the mechanism one might hope for: `λmax` is essentially unchanged (<1%) — muPC does not touch the dominant instability at all. The entire effect is a ~19% increase in `λmin`, because `compute_mupc_scalings` only reaches TWO edges in this topology (`embed→transformer_0:in` always, `skip_0→output:in` only when `include_output=True`) — both PERIPHERAL to the graph — while the measured `λmax≈113` instability lives INSIDE `TransformerBlock`'s own monolithic internal Jacobian (attention softmax + GELU FFN), which per-edge muPC scaling has no mechanism to reach. Translates to real steps saved: ~12.5→~10.5 steps to reach the same 0.6 residual-relaxation fraction (η_opt/r_opt formula, cross-validated against this session's independently-measured baseline). **🔴 THE NEGATIVE RESULT, stated explicitly because it is the one people will get wrong: `λmax` unchanged ⇒ `η*` unchanged at ~0.0176 ⇒ muPC does NOT fix the `transformer_lm()` η=0.1 instability (`docs/decisions.md` §24).** "muPC improves conditioning, therefore it fixes the η problem" is the natural — and wrong — inference from the ~16% κ number alone; `transformer_lm()`'s default is 5.7× past its own stability limit with muPC on *or* off. What the 16% actually is: `λmin` rose (~2.31→~2.74, ~19%) while `λmax` sat still — muPC is lifting the SLOW modes, not suppressing the fast one, which is the correct mechanism for what per-edge `forward_scale`/`topdown_grad_scale` factors actually do (O(1) activation-variance/gradient-scale correction across width/depth, μP-style), not curvature/Hessian conditioning — a different mechanism from what would be needed to move `λmax`/`η*`. **The decomposed family (MhaResidualNode/LnMlp1Node/Mlp2ResidualNode) is a materially different question, still fully OPEN** — that family exposes attention/FFN internals as separate PC nodes with their own inter-node edges, so muPC's per-edge scaling would reach much more of the internal dynamics there and could plausibly show a larger effect, possibly including `λmax` itself; not measured this session. **Independently cross-validated by upstream, different methodology and architecture (2026-07-14, upstream `docs/dev_plans_archive/momentum_sgd_inference_plan.md` §"Status of Previous Work"):** upstream shipped BOTH of muPC's forward-variance-gain (`variance_gain()`) AND Jacobian-compensation (`jacobian_gain()`) corrections — theoretically sound, all tests passing — and measured **zero practical impact on deep-chain MNIST accuracy**: 32-layer 59.9%→56.4% (worse), 64-layer 20.8%→21.0% (flat), only the 8-layer case improved (90.8%→91.6%). Upstream's own diagnosis (energy-distribution probe on a 32-layer chain: middle layers h8-h24 carry **zero energy**) locates the real bottleneck as local-inference SIGNAL PROPAGATION (PC relaxation propagates the output error exactly 1 hop/step; the self-error restoring force damps it faster than it can reach deep layers) — a *different* failure mode than a Hessian-conditioning problem, but the SAME conclusion this section reached by direct spectral measurement on a different architecture (`TransformerBlock`, not a deep MLP chain): **muPC's forward/Jacobian gain-scaling mechanism does not touch whatever the actual bottleneck is.** Two independent methodologies (end-to-end accuracy on a deep MLP chain; direct `λmax/λmin` spectral measurement on a transformer block), two different architectures, converging on the same negative result increases confidence this is a structural property of the muPC gain-scaling mechanism (O(1) activation-variance correction, μP-style) rather than an artifact of either investigation's specific setup. This is also upstream's own stated MOTIVATION for the `InferenceSGDMomentum` design this port has now implemented ahead of upstream (`docs/decisions.md` §26) — momentum targets exactly the signal-propagation mechanism upstream's diagnosis names. **Open caveat, not yet re-verified:** this section's own `λmin` figures (`~2.31→~2.74`) were measured with the SAME power-iteration methodology `docs/decisions.md` §24 used before §26 found it under-converged on this graph's near-degenerate spectrum (§24's `λmin=2.3` corrected to `λmin≈0.17`) — the ~16-19% RELATIVE muPC effect reported here has not been re-checked against a properly-converged (renormalized) power iteration; the qualitative conclusion (`λmax` essentially untouched) is independently supported by the near-init stability-boundary cross-check in §26 and is not expected to move, but the specific `2.31`/`2.74` numbers may be revised by a future re-measurement the same way §24's were. |
| C-10 | LOW | BayesianTuner (537-line #28 rework) | **CALLBACK BLOCKER RESOLVED (2ce30fa, 2026-07-16); BayesianTuner-proper needs an HPO-FRAMEWORK DECISION, not a 1:1 port.** X-06's concrete blocker is gone: `train_autoregressive` now has `iter_callback(epoch, batch_idx, energy)` and the full `epoch_callback(epoch, params, structure, config, rng; energy, ce_loss)` signature, matched 1:1 to train_autoregressive.py:341-361 (the one existing caller + the char_lm example updated; test_sequence_training 20/20). **But BayesianTuner itself (418 lines) is fundamentally an `optuna` wrapper** — `optuna.create_study` + `samplers.TPESampler` + `pruners.HyperbandPruner` + `study.optimize` + `trial.suggest_*`/`report`/`should_prune`/`TrialPruned`. Its whole value IS optuna's TPE Bayesian sampling + Hyperband pruning, which has NO faithful Julia drop-in. **DECISION MADE (2026-07-16): when HPO is needed, TreeParzen.jl + the ported divergence guard.** The pivotal constraint is PRIMUS's pure-Julia thesis (no Python across the identity seam), so PyCall→optuna is ruled OUT despite being literally-optuna. The choice among pure-Julia options is not 'Hyperopt vs reimplement': **TreeParzen.jl implements the EXACT TPE algorithm optuna's default sampler uses** — so it is pure-Julia AND algorithm-faithful, dominating Hyperopt.jl (pure-Julia but different samplers/pruners) on fidelity at equal Python-freeness. So the recorded path is TreeParzen.jl (faithful TPE sampling) + `divergence_guard` (the novel scale-free-instability prune, PORTED NOW — `src/tuning/divergence_guard.jl`, optuna-independent, consumes the epoch_callback). BayesianTuner's Hyperband pruner is the one piece with no pure-Julia equivalent; a successive-halving loop or the divergence guard covers it — a documented divergence. Full sampler wrapper deferred (no live consumer). So porting it is a DECISION with three options, none a 1:1 optuna port: (a) PyCall/PythonCall → optuna (calls Python; not a Julia port); (b) Hyperopt.jl (different sampler/pruner algorithms — a divergence, not fidelity); (c) reimplement TPE + Hyperband (research-grade, hundreds of lines). **And it has NO live consumer** — it is tuning/experiment convenience, orthogonal to the model, conformance, and compiled lane. Per the user's own taxonomy, BayesianTuner-proper is a Julia-tooling/ecosystem item, not a model-fidelity gap. LEFT for a deliberate HPO decision; the callbacks (the real, bounded fidelity piece) are done and usable by any Julia HPO consumer. |
| C-11 | LOW | StorkeyHopfield weight_init: Zeros (Julia) vs Xavier (upstream) | **VERIFIED CLOSED** (`52bb6a0`). Reflective cross-check resolved the R-03 read: neither implementation has a classical Storkey/Hebbian accumulation rule — W is 100% PC-gradient-learned in both — so upstream's own stated criterion for "Xavier matters when W learns via PC gradients" applies, confirming this was a port slip (Julia copied the internal `None`-fallback constant, not the constructor's actual default), not a legitimate design choice. Fixed to `XavierInitializer()`. |
| C-12 | LOW | ~17 further API-surface/UX divergences (IDE list) | DEFERRED |
| C-13 | LOW | Cyclic-graph support (upstream `examples/mnist_cyclic_graph.py` / `mnist_lateral_connections.py`) | **VERIFIED — NOT A JULIA-SPECIFIC GAP (parity with upstream); narrow coverage note only.** Read both upstream examples in full, not inferred from the names. `mnist_lateral_connections.py` is a DAG despite the name — multi-*parent* fan-in into `hidden1`/`hidden2` via `hidden1_lateral`/`hidden2_lateral`, no back-edge — already supported today (`Linear`'s `"in"` slot is `is_multi_input=true` and sums multiple in-edges, `src/nodes/linear.jl:58`/`:16`, matching upstream's own `SlotSpec(is_multi_input=True)`, `nodes/linear.py:87`, exactly). `mnist_cyclic_graph.py`'s "Cyclic" arm has a REAL literal cycle (`hidden1→hidden2→hidden2_lateral→hidden1`, the file's own `# Cycle` comment). Read `graph()`/`_topological_sort` (`src/graph_assembly/graph_construction.jl:54-78`) body directly: it does NOT raise on a cycle — it `@warn`s and returns a partial `node_order` (nodes unreachable from an in-degree-0 BFS seed are silently omitted) — a byte-for-byte-equivalent 1:1 port of upstream's own identical behavior (`fabricpc/graph_assembly/graph_construction.py:65-105`, docstring: "If the graph contains cycles, some nodes may be omitted from the order") — not a Julia-introduced divergence. The actual PC relaxation loop these examples run through (`train_pcn`/`run_inference`) does not consume the possibly-truncated `node_order` at all: `zero_grads`/`forward_value_and_grad`/`update_latents` (`src/core/inference.jl:118-195`) all iterate `structure.node_names` (the full, untruncated list), and `forward_value_and_grad` is explicitly documented as order-independent within a step ("Accumulation is additive over the fixed-`z_latent` inputs, hence order-robust", `:130-134`) — so gradient computation for a cyclic subgraph is unaffected. The only consumer of `node_order` is `FeedforwardStateInit`'s one-shot init pass (`src/graph_initialization/state_initializer.jl:45`, the DEFAULT `graph_state_initializer`): cyclic-subgraph nodes are skipped there and start from `latent_init` instead of a feedforward-seeded guess — the same degradation upstream's identical algorithm produces, not a Julia weakness. **What genuinely remains open, narrowly**: (1) zero Julia regression test/example ever builds a literal cycle — this exact path has never actually been executed in this port, read-only code-body analysis only (no Julia process run this session, per this task's CPU-sharing constraint); (2) the JIT flat lane (`src/jit_flat.jl:47`, `order_pos = [posof[n] for n in structure.node_order]`) consumes the possibly-truncated `node_order` directly for positional indexing and was NOT checked here for its behavior on a truncated order (clean crash vs. silent mis-index) — out of scope for "can Julia build these two examples" since both use the eager `train_pcn` path only, not the JIT lane, but worth a pointer for whoever next touches J-02/J-03. |

**Known-deferred, confirmed absent, no action:** multi-GPU/pmap · train_backprop.py port (anti-thesis; see C-04) · Aim dashboarding/experiments · tfds loaders · Optuna.

## 5. JIT/performance lane

| ID | Item | Status |
|----|------|--------|
| J-01 | Flat gradient seam: Enzyme-under-Reactant | **DISSOLVED as an architectural blocker — now a resolved design decision (2026-07-14, `docs/decisions.md` §15/§23 updates).** The original framing conflated "any eager AD" with eager-Enzyme specifically. This codebase's actual production eager seam is Zygote (decision #19), which never calls eager `Enzyme.autodiff` at all. Execution-verified (`benchmark/jit_zygote_expt/main_experiment.jl`): Zygote-eager + `Reactant.@compile`-with-`Enzyme.gradient`-inside (never eager `Enzyme.autodiff`) compiles and matches Zygote's eager gradient to ~1e-6/1e-7 — no poisoning. A serious secondary hazard surfaced while checking this — `using Reactant` transitively loads Enzyme, silently flipping FabricPC's REAL seam dispatch (not just the raw-array bypass) to Enzyme's MHA-crashing implementation, with zero explicit `using Enzyme` and zero catchable exception — **but this is exactly F-04's dispatch-override hazard, now VERIFIED CLOSED (section 1) by backend-type dispatch.** With that fix, `using FabricPC, Zygote, Reactant` together is now SAFE: dispatch structurally cannot flip to Enzyme's method table entry regardless of what Reactant transitively loads (`_AD_BACKEND[]` stays on `ZygoteBackend`, verified by `which()` in `test/test_autodiff_backend_registration.jl`). The eager seam (Layer 0/1, `_ad_param_grads` → Zygote) and the compiled lane (Layer 2, `Enzyme.gradient` over raw arrays inside `@compile`) were ALREADY different code paths — they only conflicted before due to the now-fixed accidental method overwrite. **What remains is J-02/J-03's own scope, not an architectural blocker**: a flat weight-gradient path that JITs the FULL `train_step` (not just the raw-array demo pattern `wgrad` already validates on a partial parameter set) — real, unbuilt work, not attempted this round, but no longer blocked on an unresolved conflict. |
| J-02 | Compile full `train_step` | **No longer architecturally blocked (J-01 dissolved, 2026-07-14) — still genuinely OPEN, not started.** `compute_local_weight_gradients` remains entirely eager/Dict-based, no flat/traced counterpart exists yet; building one is real, unattempted work, just no longer gated on an unresolved Enzyme-under-Reactant conflict. **🔴 Design-before-building note (`docs/decisions.md` §25):** the hand-rolled AdamW's step counter is a plain Julia scalar, frozen at trace time under `@compile` unless tracked — every compiled step would silently reapply step-1's bias correction with no error. Use `Optimisers.jl` (already a regular dep, already resolved with `OptimisersReactantExt`, currently unused) on the compiled lane instead of the hand-rolled optimizer — retires this landmine and independently closes the original audit's "generic optimizer injection" gap (upstream trains via swappable `optax`). |
| J-03 | Extend flat lane to seam nodes | **PARTIAL — TransformerBlock forward wired, backward parked.** `flat_forward(::TransformerBlock, ...)` (`src/jit_flat.jl`) wires the already-validated `_tb_block_flat` kernel (`src/nodes/transformer.jl:296-338`, previously a standalone, disconnected kernel + benchmark) into `CompiledPlan`'s dispatch — `to_flat_params` special-cased to bridge via the existing `flat_block_args` (TransformerBlock's weights are keyed by NAME, not by edge key like every other node here). Verified byte-equivalent to eager `compute_mu` (`test/test_jit_flat.jl`, new testset) on a graph where TransformerBlock is the unclamped terminal node (`flat_latent_grads`'s `out_degree==0 && !is_clamped` branch — forward only). **No `_flat_input_grads(::TransformerBlock, ...)` method exists** — the attention+FFN backward pass needed for TransformerBlock as an interior or clamped node — same category of problem as J-01 (needs Enzyme-under-Reactant, or a substantial hand-derived closed-form gradient verified against Zygote's already-upstream-validated gradient); verified this fails loudly with a clean `MethodError`, not silently wrong output, rather than asserting it in CI (a `MethodError` from a private multi-arg dispatch is a brittle thing to `@test_throws` on). Decomposed nodes (`MhaResidualNode`/`LnMlp1Node`/`Mlp2ResidualNode`/`EmbeddingNode`/`VocabProjectionNode`) and `StorkeyHopfield` remain untouched — zero flat-lane work for any of them. |
| J-04 | Benchmark harness | **INFERENCE-ONLY arm VERIFIED CLOSED as a PORT-CORRECTNESS/NUMERICS result; the PERFORMANCE number (7-8×/7.87×) is very likely LEGITIMATE but still NOT SAFE TO CITE EXTERNALLY as settled (2026-07-14, `docs/decisions.md` §11 update).** Operand-level HLO tracing (not just op counts) found the actual mechanism: Julia's unrolled+XLA-optimized program does ~27× fewer FLOPs than JAX's, from two exact algebraic optimizations XLA found on the fully-unrolled graph — hoisting `h.z_mu` (loop-invariant since its only source, `x`, is clamped) from 20 recomputations to 1 (`Arg_0`/`Arg_4`, i.e. `W1`/`x`, each appear in exactly one `dot` in the whole compiled program), and dead-code-eliminating the gradient into the clamped `x` node entirely (never recomputed even once — confirmed by `Arg_0`/`Arg_4` never appearing again). JAX's `while`-loop body structurally cannot do either (no "outside the loop" to hoist into; `GraphState`'s pytree carries every node's `latent_grad`, clamped or not, so the dead gradient can't be dropped from the carry). Theoretical FLOP ratio ≈26.8×, LARGER than the measured 7.87× — the expected signature of a real effect measured on real hardware, not an unexplained residual. Two real, upstream-contributable PC-inference optimizations fell out of this as a byproduct (hoist clamped-source-only forwards; never compute gradients into clamped nodes), independent of Reactant/Julia entirely. **TOPOLOGY-DEPENDENT (§28, runnable analyzer `scripts/flop_elimination_analysis.jl`, validated to the digit against the HLO ground truth): the win = fraction of per-step FLOPs on clamped-incident edges = 98.7% on THIS MNIST-MLP (clamped input is the widest node), 1/(L+1) on a uniform L-layer chain (3% at L=32), and EXACTLY 0% on a transformer-LM at any scale (clamped input feeds an EmbeddingNode gather — 0 FLOPs to hoist, discrete-zero gradient to prune). 7.87× is one point on this curve and MUST be cited with its topology (MNIST-MLP, B=256, 784→128→10) or it badly misleads — the flagship transformer gets nothing.** **Still contingent, not confirmed**: the number survives `@trace while` only if that implementation deliberately preserves these two optimizations (a rolled loop doesn't get them "for free" the way unrolling did). The decisive re-measurement (rebuild under `@trace while`, re-time) was attempted and BLOCKED on a genuine Reactant/`CompiledPlan` tracing-compatibility gap (`SlotInfo`'s `Bool` field fails a traced-type round-trip when swept into `@trace for`'s auto-detected loop-carried state) — not yet resolved; most promising next attempt is destructuring `CompiledPlan` into plain `Int`/`Tuple` values (with dead edges pruned and invariant forwards separated) before entering the trace, which fixes the blocker and implements both optimizations in one refactor. The correctness/numerics half of this closure (matching JAX's own output to 2.29e-6) is unaffected and stands. The "BLOCKED on J-01/J-02" framing was too broad — that pair blocks compiling `compute_local_weight_gradients` (the weight-gradient/M-step) specifically, but `compile_inference`/`flat_run_inference` (the E-step relaxation loop) is a separate, already-validated path, and the MNIST-MLP architecture (`Linear`-only) is fully covered by `flat_forward`/`flat_latent_grads` — confirmed against `src/jit_flat.jl` before benchmarking, not assumed. `benchmark/mnist_inference_vs_jax.{py,jl}`: same architecture/batch/η/steps, same weights both sides (RNG-trap discipline). **Julia+Reactant's `compile_inference` runs ~7-8× faster than upstream's own `jax.jit(run_inference)`** (the first cross-language, both-compiled number this codebase has produced — do not conflate with §11's existing 8.8×/32× Julia-vs-Julia numbers), numerics validated to float32 precision. Scope: inference only, no weight update — NOT a training-speed claim. **The training/weight-gradient arm remains BLOCKED on J-01/J-02**, unchanged. **🔴 SCOPE CAVEAT for any LARGER-scale rerun (`docs/decisions.md` §25):** `flat_run_inference` (`src/jit_flat.jl:364-365`) UNROLLS its step loop under `@compile` (a native `for`, no `@trace`) — invisible at this benchmark's 20 steps, but does not scale to §24's own finding that this graph family needs 50-300 steps to actually converge. `@trace while` (relax-to-tolerance) is the fix, and — sourced from Reactant's own AD docs, not asserted — a genuine positioning argument: PC's local weight-gradients never differentiate through the relaxation loop, so PC can use a convergence-based `@trace while` in a regime where backprop-style AD structurally cannot without paying real memory cost or hitting a hard dynamic-shape platform limit. Also: benchmark CPU only, or match upstream's `jax_setup.py` XLA flags (all `xla_gpu_*`, all performance-reducing) on any future GPU comparison — this run was accidentally fair since this machine has no GPU. |
| J-04c | Benchmark harness — SAME-STATE re-measure with Reactant's DOCUMENTED barrier | **J-04 = 8.97x (2026-07-16), and BOTH the instrument and the METHOD were wrong before.** (1) **INSTRUMENT:** `@compile sync=true` is Reactant's documented completion barrier (CompileOptions.jl: *"blocking till the computation is complete. This is recommended when benchmarking"*; `ConcreteRArray.jl:160-162` explicitly says *"Prefer `@compile` with `sync=true` ... instead of calling this function"*). We hand-rolled the barrier as `Array(...)` — a device→host copy JAX's `block_until_ready` never pays — and then wrote that asymmetry into J-04b **as unavoidable**. It was not unavoidable; it was not reading the docs. Measured cost of that self-inflicted copy: **0.443 ms**, ~10% of the call. (2) **METHOD — the bigger error:** J-04b compared Julia timings against a JAX number measured HOURS EARLIER in a different machine state. Re-running the JAX side in the SAME state gives **36.79 ms, not 50.88** — the machine was heavily contended before (its eager arm moved 752→455 ms too). **A cross-state ratio is not a measurement.** **SAME-STATE, both sides minutes apart:** `remarshal` 5.080 ms = **7.24x** (reconciles with the originally recorded 7.87x — machine noise, and it was never the confound we thought); `fast` 4.768 = 7.71x (**J-04b's headline 8.35x was cross-state-inflated**); `resident` 4.545 = 8.09x; **`sync=true` 4.103 = 8.97x — the true apples-to-apples**, both sides blocking without a host copy. So the documented instruments are worth **+24%** (7.24x → 8.97x), not the +11% estimated. **#2647 GATE PASSED, not assumed:** upstream #2647 (OPEN, "Buffer synchronisation is sometimes removed") means `sync=true`'s wait can be DCE'd, silently restoring the async artifact with no `Array()` left to expose it. The benchmark now ASSERTS the barrier: measured sync/async_bare = **214x** (a DCE'd wait would read ~1x). The bare async dispatch times at 0.019 ms — decisions.md §11's "bogus ~1000x" trap, reproduced live. Numerics unchanged: `max|julia_jit - jax_jit|` = 0 / 2.29e-06 / 0. **Does NOT revisit J-04's mechanism finding** (the ~27x FLOP ratio from XLA hoisting/DCE on the unrolled graph); that analysis stands — this only replaces a hand-built apparatus with the documented one and measures both sides at once. |
| J-04b | Benchmark harness | **SUPERSEDED by J-04c (2026-07-16) — its 8.35x was cross-state; same-state it is 7.71x, and the documented barrier gives 8.97x.** Original entry: **MEASURED 2026-07-15 — the marshalling confound is REMOVED from the harness, and the number it was hiding is now known.** J-04's 7.87x was measured on `ci(params, init)`, the SLOW path, which `to_rarray`s every weight on EVERY call. JAX's timed loop passes `params`/`init_state` that are ALREADY device-resident `jax.Array`s — passing those to a jitted fn copies nothing. **We were charging Julia for a host->device copy JAX never makes, and one our own API already avoided**: `32b64a6` made `compile_inference` hold the params' ConcreteRArrays (the Lux `ps_ra = ps |> xdev` pattern); the benchmark simply never switched to the fast path. Now measures three variants in one process, so the cost is a NUMBER, not an attribution. **Medians of 3 runs** (min-of-30 each, vs `jax_jit_steady_ms = 50.88`): `remarshal` 6.78 ms = **7.50x** (the old path — reproduces the recorded 7.87x); `fast` (params device-resident, per-batch state marshalled — what a real E-step does) 6.09 ms = **8.35x**; `resident` (params AND state device-resident — JAX's exact setup) 5.57 ms = **9.13x**. Ordering held in all 3 runs; absolute ms drifted ~20% on run 3 (machine contention), so the WITHIN-run ratios are the robust part: re-marshalling costs **+11% / +18% / +18%** of the call. **CORRECTION to a documented attribution:** §28 addendum 3's slope test attributed **~2.5 ms/call** to marshalling; measured directly it is **0.69-1.23 ms/call for params** (median 0.99) plus **0.26-0.52 ms for state** — the slope test over-attributed by ~2x. A benchmark says WHAT, not WHERE (`feedback_no_perf_attribution_without_profiling`); this is the same lesson applied to a fitted decomposition. Residual asymmetry, stated not hidden: all three Julia variants still copy the OUTPUT device->host via `Array(...)`, which JAX's `block_until_ready` does not — so these ratios are conservative for Julia. That read stays because removing it is the documented async-dispatch trap (decisions.md §11: timing the thunk without it shows a bogus ~1000x). Numerics unchanged and tight: `max|julia_jit - jax_jit|` = 0 / 2.29e-06 / 0 (x/h/y). Does NOT revisit J-04's mechanism finding (the ~27x FLOP ratio from XLA hoisting `h.z_mu` and DCE-ing the clamped gradient) — that analysis stands; this only removes a harness artefact sitting on top of it. |
| J-05 | Hand-derived attention VJP as explicit Layer-1 gradient | DEFERRED (optional, post-J-03) — unchanged. |
| J-08 | `@trace for` tracing wall: ROOT-CAUSED to a Reactant bug on abstract `Tuple` fields (2026-07-15) | **The `@skip_rewrite_*` one-liner is REFUTED; the real blocker is now a specific, reportable upstream bug — and the documented destructure plan is NECESSARY BUT NOT SUFFICIENT.** All by execution on Reactant 0.2.273, warm session. **(1) Baseline reproduced:** `@trace for` over `flat_inference_step` still fails `NoFieldMatchError`, `SlotInfo` implicated — exactly as §11/§25 recorded. **(2) One-liner refuted:** `@skip_rewrite_constructor` DOES NOT EXIST (the name in the plan was wrong); the real macro is `Reactant.@skip_rewrite_type`, and registering `Type{<:SlotInfo}` / `Type{<:NodeInfo}` / `Type{<:CompiledPlan}` (list 16→19 entries, registration verified) leaves the error BYTE-IDENTICAL. Wrong subsystem: `@skip_rewrite_type` governs the IR-rewrite/`@reactant_overlay` machinery (`utils.jl`), while the wall is in the traced-type sweep (`Tracing.jl`). **(3) Destructure spiked — clears the FIRST wall, hits a SECOND.** Passing a SlotInfo-free plan into `@trace for` (nodes as a Tuple; `infos` as plain `(in_degree, out_degree)` NamedTuples — `flat_latent_grads`'s `info` arg is untyped, so it duck-types) changes the error from `NoFieldMatchError` to `UndefRefError`, thrown at `traced_type_inner(::Type{Base.RefValue{Tuple{Linear,Linear,Linear}}})` → `collect_tvars_in_type!` (`Tracing.jl:661`). So §11/§25's "destructure `CompiledPlan` into plain values" is real but understated: `@trace` also sweeps the NODE structs, and they break too. **(4) ROOT CAUSE, isolated with zero FabricPC involved:** `Reactant.collect_tvars_in_type!(IdSet{TypeVar}(), Tuple)` throws `UndefRefError`, while `NTuple{2,Int}` is fine. `Tracing.jl:660-661` does `collect_tvars_in_type!(dependencies, t.T)` / `(..., t.N)` on a `Core.TypeofVararg` with NO `isdefined` guard — and `Tuple === Tuple{Vararg{Any}}`, whose bare `Vararg` has no `.N`. **Every FabricPC node struct declares `shape::Tuple`**, so tracing any node walks its field types straight into that crash. **(5) CORRECTION to this entry's own first version (committed `85cc6ff`, corrected the same day): I said the fix was "upstream guard OR concrete `shape::NTuple`". The second half is REFUTED, and there is a THIRD wall behind the second.** I applied the proposed `isdefined` guard to Reactant in a live session and re-ran the spike. It gets PAST the `UndefRefError` and fails with a new `NoFieldMatchError` whose message is explicit: *"Cannot convert type Linear... the type does not capture the fieldtypes that should be converted in its type parameters"*, with the per-field table naming the culprits — `use_bias idx=5 Derived: Reactant.TracedRNumber{Bool} Existing: Bool` and `flatten_input idx=6` the same. **`shape` in that same table reads `Derived: Tuple Existing: Tuple` — i.e. once the Vararg bug is guarded, `shape::Tuple` traces FINE and was never the tracing blocker.** So: **`shape::NTuple{N,Int}` would NOT unblock `@trace`** (its case is now only type-stability/dispatch, which is weaker and separate). **THE ACTUAL WALL, all three instances of it:** `@trace` sweeps loop-invariant structs into traced loop-carried state and traces their `Bool` fields as `TracedRNumber{Bool}`, which a non-parametric struct cannot hold — `SlotInfo.is_multi_input` (wall 1) and `Linear.use_bias`/`flatten_input` (wall 3) are the SAME failure, not two. The only real fix is keeping the node structs OUT of the sweep (they are static config that never needs tracing), which means resolving per-node dispatch OUTSIDE the loop body so the body never references them — a genuine restructure, or an upstream `Const`-style escape hatch. **The upstream Vararg bug is still worth filing on its own merits** (a) — and it is WIDER than us: `collect_tvars_in_type!(Tuple{Int, Vararg{Int}})` also throws, so ANY tuple type with an unbounded Vararg crashes, not just abstract `::Tuple` fields. The `isdefined` guard was verified to fix all four cases (`Tuple`, `NTuple{2,Int}`, `Tuple{Int,Vararg{Int}}`, bare `Vararg`) with no regression. **But it does NOT unblock 3(c) for free** — measured, not assumed. **Next session starts at wall 3 (keep nodes out of the sweep), not at `@skip_rewrite` and not at `shape::NTuple`.** **(6) UPSTREAM REPORT DRAFTED + INDEPENDENTLY VERIFIED (3 agents: upstream-source / prior-art / adversary — all CONFIRM, no blockers): `docs/upstream/reactant_vararg_undefref.md`, ready to file, NOT filed (`gh` is not installed here — needs a browser).** Verification corrected our draft on five points, which is why it was worth running: (i) newest release is **v0.2.274**, not our installed 0.2.273 — the local registry tarball is stale; the code is byte-identical at the v0.2.274 tag and on `main` (raw-fetched), so we file against those, not a stale version. (ii) There is a **SECOND unguarded site**, `Tracing.jl:142` `traced_type_inner(::Core.TypeofVararg)` (`Vararg{traced_type_inner(T.T,...),T.N}`) — confirmed by our own execution, so the one-site patch was incomplete. (iii) **Issue #767 'Tracing VersionNumber' is precedent, not a duplicate**: same root cause, closed 2025-02-19 by PR #773 which BLACKLISTED `VersionNumber` instead of guarding the reads — and **its root cause is still live**, which we demonstrate directly: `traced_type_inner(Tuple{Vararg{Union{UInt64,String}}}, ...)` (VersionNumber's exact field type) still throws `UndefRefError`. Blacklisting cannot fix ours: the trigger is a USER struct, so there is no finite list. (iv) Our claim "any struct with an abstract `::Tuple` field crashes" was **TOO BROAD** — the crash needs `changed == true` (some other field must actually trace) to pass the `if !changed; return T; end` early-return; `RefValue{Tuple{Linear,...}}` under `ConcreteToTraced` returns fine. (v) The fix does **NOT** unblock our `Linear` (it yields the intended `NoFieldMatchError`) — found independently by the adversary AND by our own live patch, agreeing. Filing it claiming otherwise would have cost credibility with a maintainer. |
| J-09 | **`@trace for` WALL BROKEN — compiled PC inference LOOP shipped in `compile_inference(...; loop=true)` (2026-07-15)** | **🔴 CORRECTED, same day, and the correction is the lesson: THE ENTIRE FOUR-WALL ANALYSIS BELOW COLLAPSES TO ONE DOCUMENTED OPTION — `Reactant.@trace track_numbers=false`.** The user asked "are you using any documentation from Reactant.jl/Enzyme.jl/Zygote.jl, or blindly developing?" — the answer was blindly, and reading `ReactantCore`'s `@trace` docstring (line 187: *"track_numbers::Union{Bool,Datatype} — whether Julia numbers should be automatically promoted to traced numbers upon entering the loop"*, default `true`) ended it. Number promotion IS the mechanism behind every wall: it turns `SlotInfo.is_multi_input::Bool` / `Linear.use_bias::Bool` into `TracedRNumber{Bool}` (unholdable by a non-parametric struct ⇒ `NoFieldMatchError`), makes `if !clamped[i]` a traced-Bool branch (⇒ upstream #2441), and drags the walker into `shape::Tuple` (⇒ the upstream `Vararg` `UndefRefError`, J-08). Turn promotion off and all four vanish. **MEASURED minimality:** the RAW, UNTOUCHED `flat_inference_step(plan, fp, fs, clamped)` — full `CompiledPlan`, `SlotInfo`, `NodeInfo`, plain `Vector{Bool}`, no destructure, no hatch, no `Val` — compiles under `@trace track_numbers=false` at max|Δ| = 1.49e-8 vs the eager flat loop. **So of the machinery I built and briefly recorded here as necessary — a `StaticPlan` destructure, a two-part `traced_type_inner`/`make_tracer` hatch, a `Val`-carried clamp mask — NONE is needed; all deleted.** The hatch IS real and IS the documented extension point (Reactant's own `test/core/constructor.jl:22-42`, issue #1595), and it does work — it is simply the wrong tool when a one-word option exists. **SHIPPED:** `compile_inference(structure, params, clamps; batch, loop=true)`, ~15 lines in `ext/FabricPCReactantExt.jl`. Verified on Reactant **0.2.274** + Enzyme **0.13.185**: `loop=false` 78.4s compile / Δ=1.49e-8; `loop=true` **23.6s compile** (~3x faster — body emitted once, not `infer_steps` times) / Δ=1.49e-8. ⚠️ `track_numbers=false` is correct here ONLY because every loop-carried value is an ARRAY; a future dynamic SCALAR (convergence error, step counter — cf. the frozen-`t` AdamW hazard, decisions.md §25) would be SILENTLY frozen at its trace-time value. Documented at the call site. **UNBLOCKS §28's decisive experiment**: J-04's ~27x FLOP claim rests on XLA optimizing the UNROLLED graph in ways a loop body structurally cannot — now measurable by compiling both, not arguable from HLO. Superseded detail follows. | **COMPILED, numerics 1.49e-8 vs the eager flat loop.** The blocker recorded since §11/§25 is DOWN. **How this was found matters more than the fix: the user said "go through the Reactant.jl issues, there are 347 open" — and the answer was already filed.** Our own `decisions.md` asserted *"No `Const`-style escape hatch was found in Reactant's source for excluding a value from this automatic sweep"*. **That claim was FALSE and stood for weeks.** Reactant issue **#1595** (open, "NoFieldMatchError for Oceananigans with RectilinearGrid") shows packages define **`Reactant.traced_type_inner` for their OWN types** (Oceananigans does it for LatitudeLongitudeGrid), and Reactant's `src/Tracing.jl:30-54` is the blessed do-not-trace registration (`DataType, Module, Nothing, Symbol, …, VersionNumber, Sharding.Mesh`) — identity `traced_type_inner`, plus the value-half `make_tracer(...) = prev` at `Tracing.jl:1087`. **FOUR walls, each measured, each cleared:** (1) `SlotInfo` Bool → destructure the plan (never pass `NodeInfo`/`SlotInfo` into the body; `flat_latent_grads`'s `info` arg is untyped so an `(in_degree, out_degree)` NamedTuple duck-types in). (2) `shape::Tuple` → `UndefRefError` from the upstream Vararg bug (J-08; report drafted at `docs/upstream/reactant_vararg_undefref.md`). (3) node / energy / activation / initializer structs → **BOTH halves of the hatch** for each static-config family: `traced_type_inner(::Type{T}, ::Any, ::Reactant.TraceMode, ::Type, ::Any, ::Any) where {T<:S} = T` AND `make_tracer(seen, prev::S, path, mode; kwargs...) = prev`. Type-half alone gets you an `AssertionError` from the value path — you need both. (4) `clamped::Vector{Bool}` → `TypeError: non-boolean (TracedRNumber{Bool}) used in boolean context` (upstream **#2441**, open); fixed by carrying the clamp mask in a **`Val`** — which is ON Reactant's own do-not-trace list — and reading it with `@inline _cl(::Val{C}, i) where {C} = C[i]`, so it resolves to a literal `Bool` at trace time. **What this unblocks:** J-02 (compile `train_step`) loses its stated blocker, 3(c) becomes reachable, and — most valuable — §28's *decisive* experiment is now runnable: J-04's mechanism analysis argues the ~27x FLOP gap comes from XLA hoisting/DCE-ing on a fully-UNROLLED graph, which a `while`-loop body structurally cannot do. We can now BUILD the looped version and measure it, instead of arguing from HLO. Spike lives at `scratchpad/final.jl` (session-local); productionizing = doing the destructure + hatch in `src/`, which is the §11/§25 refactor, now with the recipe known. **NOT YET IN `src/`.** |
| J-10 | **Backend re-derivation after §19 fell — GATES 1 & 3 MEASURED (2026-07-16)** | **§19's premise is dead (see decisions.md §19) and the measurements point the OPPOSITE way from the folklore. GATE 1 — can the conformance suite certify Enzyme? YES, IDENTICALLY.** Ran the tiers in a purpose-built Enzyme-only env (`.warm/enzyme_suite_env`; NOT the pinned `test/dual_ad_backend_env`, which is an F-04 fixture and must not be perturbed): **tier_a 44/44, tier_b 147/147 (incl. TransformerBlock 58/58 — §19's exact subject), tier_c 100/100, tier_conv 73/73, tier_d_mnist 31/31 = 395/395, the SAME counts as Zygote.** `tier_b` needed a variant only because it hardcodes `using Zygote` at :15 to activate the seam — the file is otherwise backend-agnostic (it calls the seam, never Zygote directly); nothing in it failed on Enzyme. **⇒ The strongest argument FOR Zygote — 'switching production splits the certifier from the certified' — is now NEUTRAL: the suite certifies both.** **GATE 3 — the number that actually decides a PRODUCTION backend is steady-state gradient latency, NOT suite wall-clock (which is compile-dominated).** Same machine, one backend per process, minutes apart, min-of-20 after warmup, TransformerBlock (B=4,S=8,E=16,H=2): **param-grads Zygote 2.883 ms vs Enzyme 2.065 ms (Enzyme 1.40x FASTER); latent-grads Zygote 3.090 ms vs Enzyme 1.918 ms (1.61x FASTER)** — on the very node §19 said Enzyme could not differentiate. **COLD/COMPILE cost runs the other way**: same 5 tiers, same state, both cold — Zygote 443s vs Enzyme ~797s (~1.8x), driven by tier_conv (114.6s → 327.2s, 2.9x) and tier_b (292.6s → ~438s, 1.5x); tier_a and tier_d_mnist are actually FASTER on Enzyme. **So the real trade is: Enzyme is faster where it RUNS, slower where you WAIT for it — a developer-ergonomics cost, not a production one.** **GATE 2 — MEASURED, and it is OUR bug, not a library limit.** Question, stated precisely: the flat lane has closed-form backwards for its four dense nodes (`_flat_input_grads` for Linear/Identity/Skip/LinearResidual) and **NO backward for TransformerBlock**; Conv/Pool are `_flat_supported=false` by design. So J-02 needs Enzyme-under-Reactant to supply the transformer backward. J-01 only ever proved this on a TOY `ADLinear`. Tested per Reactant's OWN tutorial idiom (`docs/src/tutorials/automatic-differentiation.md`: `@jit Enzyme.gradient(Reverse, f, x)`): **(a) the tutorial baseline COMPILES in our env** (`sum_squares` → `[2.0,4.0,6.0]`), so the idiom and the env are fine; **(b) our TransformerBlock FAILS with `MethodError: no method matching Float32(::Reactant.TracedRNumber{Float32})` — IDENTICALLY with `track_numbers=false` and `=true`**, so it is not the tracing knob either. **⇒ The blocker is a concrete-`Float32(...)` conversion inside OUR `compute_mu` path that a tracer cannot satisfy** (candidates: `transformer.jl:164` `sqrt.(Float32.(1:S))` / `:179`,`:309` `sqrt(Float32(Dh))`), not a Reactant/Enzyme limitation. **FIXED + GATE 2 CLOSED (fae0e67):** the `Float32(::TracedRNumber)` was a symptom; the cause was `map(1:B) do b ...` — a per-sample attention loop that Reactant overlays (`Base.map(f, ::AbstractArray)`, Overlay.jl:249; a UnitRange IS an AbstractArray) and lowers to a traced while-loop. Replaced with `NNlib.batched_mul` (batched matmul, no loop; head loop → `ntuple(H)`). **Compiled Enzyme-under-Reactant transformer gradient now COMPILES and is CORRECT — worst rel err 3.71e-07 vs the Zygote reference** (cross-checked because eager Enzyme, see below, can't provide a baseline). tier_b stays 147/147 on Zygote. **So J-02's stated blocker (no transformer backward in the flat lane) is GONE**: Enzyme-under-Reactant supplies it, verified correct. **HONEST TRADE (fae0e67):** `batched_mul` BREAKS EAGER Enzyme — `_ad_param_grads(EnzymeBackend(), tb, ...)` throws "Illegal calling convention fixup" (Enzyme 0.13.185 can't differentiate batched_mul's BLAS calling convention eagerly). That is neither the production backend (Zygote) nor the compiled lane (which works), so it is accepted; the per-sample loop is the fallback if eager Enzyme on attention is ever needed. **DECISION (3-of-3 gates in): the evidence supports `eager=Zygote / compiled=Enzyme-under-Reactant` — Zygote certifies identically (395/395) and is the mature/legible eager backend every fixture was authored against; Enzyme-under-Reactant is the compiled lane (gate 2 closed, correct) and is 1.4-1.6x faster steady-state. F-04 narrows from 'load exactly one, they conflict' to 'eager=Zygote, compiled=Enzyme, cleanly separated by lane.' NOT flipping production to eager-Enzyme — no gate argued for that.**: does Enzyme-under-Reactant hold for ALL node types (we have only C-14's ConvNode spike and today's MHA probe)? A two-thirds-tested hypothesis is exactly what §19 was. Remaining arguments for Zygote that these gates do NOT touch: error legibility (Julia exceptions vs opaque LLVM aborts — and §19 was itself one such abort, so the class is real even though this instance healed), and the fact that every fixture/oracle in the repo was authored against it. |
| J-06 | `NodeState` parametric types `{T<:AbstractArray}` | OPEN — **now QUANTIFIED, not speculative (2026-07-14, decisions.md §28).** Type instability CONFIRMED via `@code_warntype`: `NodeState`'s 6 fields are all `::Any` (field type = `Any`), so even for a concrete `Linear`, `forward(::Linear)` infers to `Tuple{Any, NodeState}` and `forward_and_latent_grads(::Linear)` to `Tuple{NodeState, Dict{String,Any}, Any}` — every latent operand is boxed. COST measured (deterministic, contention-robust): `run_inference` on MNIST-MLP (B=256, 784→128→10, 20 steps) allocates **6565 allocations / 208 MB per call** (~328 allocs/step for a 3-node graph doing ~4 matmuls/step — ~10× a type-stable path). Allocation-profiler (`Profile.Allocs`) type breakdown shows the boxing directly: 900 `Memory{Float32}`, 360 `Memory{NodeState}`, 320 `NodeState`, 240 `Tuple{NodeState, Dict{String,Any}, Matrix{Float32}}`. Fix = parametric `NodeState{A}` (`z_latent::A` etc.). "A real refactor — schedule deliberately," but its value is now a measured ~208 MB of GC pressure/call, not an argument from principle. **Elevated to HIGHEST-leverage perf item (decisions.md §28 addenda 2–3): a SUSTAINED loop (40 calls, i.e. training) is 40–57% gc% (8.33 GB garbage), and a PROOF-OF-CONCEPT in-place/type-stable path (`benchmark/inplace_inference_poc.jl`) measures the actual win (clean, one-process, BLAS-pinned): 6565 allocs → 0 (bit-identical), gc% → 0%, and **5.8× faster than the already-hoist+pruned path** (removes GC + boxing + 208 MB bandwidth + Dict/NodeState construction; earlier "~2×"=GC-only, "~4×"=contaminated baseline). The in-place path is FLOP/bandwidth-LIMITED (~6% above its arithmetic floor) — NOT "≈ the FLOP ceiling" (a category error: in-place eliminates 0 FLOPs, an orthogonal axis to hoist+prune's 26.8×). **XLA-vs-eager (decisions.md §28 addendum 3 — took THREE corrections, each an un-exhausted-lane artifact): with the eager loop fully optimized (hoist+prune + in-place + FUSED), eager per-step compute = 0.1239–0.1272 ms/step == XLA's 0.1272 — EXACT PARITY (both memory-bandwidth-bound, ~10 GFLOP/s). XLA's apparent 21% edge was FUSION (unfused eager did 3 passes materializing `lg_h` twice; XLA auto-fuses), matched by hand. So compiled buys NOTHING on compute for this shape. The compiled lane's real value = auto-fusion of COMPLEX chains (the transformer's LN→GEMM→softmax→GEMM, impractical to hand-fuse) + scaling (`@trace while`) + GPU — a PREDICTION, untested (no compiled transformer inference: needs `_flat_input_grads(::TransformerBlock)`). Measure `transformer_lm()` before any roadmap claim; on MNIST the compiled speed advantage is ZERO. Also: `A` (fat-contraction) ≈ 2.0 ms (25.7 GFLOP/s) from `a_eager`, correcting the ΔT-inflated 3.17 ms. Design item: `compile_inference` re-marshals constant `params` every call (~400 KB) — should hold `ConcreteRArray`s across calls.** Productionizing J-06 IS the same refactor as the CompiledPlan destructure (concrete positional plan metadata; buffers as plain arrays, not a struct, to avoid the `SlotInfo` tracing wall). Confirmed by FIVE tools (`@code_warntype` / `JET.report_opt` (100 runtime-dispatch sites whole-graph; `JET.report_call`=0 errors ⇒ code is CORRECT, instability is purely perf) / `AllocCheck`(16 sites where an `::Any` operand forced boxing off the static-dispatch path — the allocation is the box, not the dispatch; multiple dispatch itself is a Julia strength) / `Profile.Allocs` / `BenchmarkTools`) and one falsified mitigation (`--gcthreads=1,1` made GC worse, 63.6% — reduce-allocations is the only lever).** **PARAMETRIC NodeState LANDED (2026-07-16, bit-identical): `struct NodeState{Z,M,R,E,L}` — one independent type param per field, so every construction site auto-infers concrete with no edit and the 2 `nothing`-partial test sites still work; `::NodeState` dispatch + `Dict{String,NodeState}` fields match the UnionAll unchanged. Full suite 2379/2379, tier_b 147/147, flat lane bit-identical. Verified: `st.z_latent` now infers `Matrix{Float32}` (was `::Any`). The bit-identity gate CAUGHT a Julia-invariance regression — `::Vector{NodeState}` does NOT match `Vector{NodeState{concrete}}`; 5 flat-lane signatures relaxed to `Vector{<:NodeState}`. **HONEST SCOPE: this is the type-stability FOUNDATION, NOT the alloc win.** Measured: parametric-alone does NOT reduce `run_inference` allocations — the eager path stores `Dict{String,NodeState}` (abstract-valued) and `update_state` builds a new immutable NodeState per step; `forward(::Linear)` is only PARTIALLY concrete (z_mu/error/energy stay free type-vars, the construction chain doesn't propagate). The PoC's 6565→0 alloc / 5.8× is the IN-PLACE loop (`inplace_inference_poc.jl`: pre-alloc buffers + `mul!`/`.=`, no per-step NodeState), a SEPARATE larger refactor. Parametric NodeState is the prerequisite for it AND for J-02 (Reactant traces concrete types better); the in-place loop is lower-leverage than J-02 given §28's finding that the compiled lane, not eager, is where perf matters.** |
| J-07 | Slot-name resolution + immutable state-Dict churn out of node forwards | OPEN — **QUANTIFIED (2026-07-14).** `Profile.Allocs` attribution of the same 6565 events: the dominant allocation *count* is `put_node` (`state_ops.jl:24`) at **1440 events**, each rebuilding the whole `Dict{String,NodeState}` + `NodeState` + `GraphState` (360 `Dict{String,NodeState}`, 320 `NodeState`, 242 `GraphState` allocated) on every single node update — the immutable-reconstruction Python-ism. The flat/positional path (`jit_flat.jl`, `Vector{NodeState}` indexed by position) already removes THIS Dict-rebuild, but keeps the J-06 boxing — so the two are separable and J-07's Dict-churn is fixed for free once a positional state replaces the `Dict{String,NodeState}` in the eager path. Related NEW finding: `forward_and_latent_grads`'s `in_degree==0` clamped-source branch (`linear.jl:129`) needlessly reallocates `copy(z_latent)` + `zero(error)` + `zero(pre_activation)` for the 256×784 clamped input **every step** — **48.2 MB, the single largest byte allocator** — a hoistable case §28's current opt (in_degree>0 only) does not yet cover. |

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
**151/151 assertions passed** *(count updated 2026-07-15: **147/147** after C-16 adopted upstream b6f64ad, which removes `NodeState.pre_activation`; the 4 dropped assertions compared a field that no longer exists on either stack — coverage is unchanged, the quantity is gone.)* (verified by an isolated re-run of the file alone, not just the
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

**1. Port fidelity is established independently of anything below.** Tier B's 151/151 (now 147/147, C-16)
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
at rtol=atol=1e-4. **523/523 pass for both `InferenceSGD` and `InferenceSGDNormClip`** (2026-07-15: the file now runs 918 assertions, having dropped 96 `pre_activation` comparisons to C-16/b6f64ad; the *value* comparisons this claim is about are untouched and still pass) (the
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

### Future re-validation opportunities (2026-07-14) — new upstream test files, not yet consumed

Upstream now ships real pytest oracles (and one diagnostic script) that didn't exist when several
items above were closed "blind against source" (F-01/F-02's original regression tests could only
assert Julia-internal formula self-consistency; C-03 was closed against a docstring + the TFDS split
formula, not a checked-in upstream test file). Not urgent action items — a catalog for whoever next
touches the relevant register item, so a re-validation pass starts from a known oracle instead of
rediscovering one. Confirmed present in the upstream checkout and read (not inferred from filename)
before writing the "what it tests" column:

| Upstream file | What it actually tests | Relevant to |
|----|----|----|
| `tests/test_transformer_mupc.py` (232 lines) | `TestTransformerFanIn`: `TransformerBlock.get_weight_fan_in` returns `embed_dim`, ignoring `flatten_input`. `TestTransformerMuPCVariance`: forward() under a real `MuPCConfig` produces `z_mu` with `Var≈1.0`; forward() without muPC preserves original (pre-scaling) behavior; scaling-attached/absent bookkeeping. | F-01 (muPC LayerNorm compensation on `TransformerBlock`) — a forward-side variance-control oracle, complementary to F-01's own weight-grad-compensation regression test; also C-09's monolithic-family half. |
| `tests/test_transformer_v2_mupc.py` (71 lines) | muPC residual-depth accounting for the DECOMPOSED v2 transformer (`create_deep_transformer`): residual depth `L == 2·depth` (MHA-skip + MLP2-residual merge points per block), skip slots correctly marked `is_skip_connection`, `forward_scale` shrinks with `L`. | C-09's decomposed-family half ("fully OPEN … MhaResidualNode/LnMlp1Node/Mlp2ResidualNode under real muPC scaling") — close to a ready-made oracle for exactly that gap. |
| `tests/test_rope.py` (145 lines) | `TestRoPE`: pairing/frequency/rotation correctness of `precompute_freqs_cis`/`apply_rotary_emb` in isolation (not inside a transformer block). | General RoPE conformance — already independently confirmed bit-exact (section 3 above; Tier A's RoPE fixture). This file is a live upstream pytest oracle for re-generating that fixture, not a new gap. |
| `tests/test_ndim_shapes.py` (210 lines) | `TestNDimShapes`/`TestNDimTraining`: plain `Linear` nodes with 2D (`(28,28)`) and 3D NHWC shapes flow correctly through `graph()`/`initialize_graph_state`/`train_step` — generic N-dim shape support, no conv-specific node involved. | C-01/C-08 (ConvNode + ND Kaiming/Xavier fan-in, both DEFERRED) — validates the *generic* rank-agnostic graph machinery (decisions.md §13) ahead of any conv-specific work; a re-validation target once C-01 is picked up, not a conv-node test itself. |
| `tests/test_token_loaders.py` (116 lines) | `_ArrayTokenLoader(_TokenSequenceLoader)`: sliding-window batching, `max_samples` cap, drop-incomplete-last-batch, next-token alignment, and directly on point — `test_yields_integer_targets_not_onehot` (the loader contract F-05/C-03 both depend on). BPE-cache test gated behind `tokenizers`. | C-03 (`_TokenSequenceLoader`/`CharDataLoader` port, closed against the docstring + TFDS split formula — no checked-in upstream pytest file existed at the time) — now a real oracle for that contract; secondarily F-05 (integer-vs-one-hot target contract). |
| `scripts/diagnose_deep_mupc.py` (158 lines, diagnostic script, not pytest) | Per-layer forward-scale printout for a 20-layer plain `Linear`+muPC chain: confirms `a=1/sqrt(fan_in)` (no depth factor) on non-skip chains, `z_mu` norms O(1) throughout, nonzero energy/weight-grads post-inference. | None of the open F-*/C-* items directly — overlaps existing `test_mupc.jl`/`test_mupc_resnet.jl` coverage (decisions.md §9/§10) for a plain deep chain; useful as an upstream cross-check script if that coverage is ever revisited, not new-gap-relevant. |

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

**Batch 3 — performance claim: J-04's inference-only arm VERIFIED CLOSED (see section 5); J-01/J-02
(training/weight-gradient compile) remain OPEN.** J-04 vs `jax.jit` no longer waits on J-01→J-02→J-03
end-to-end — its inference-only scope was measurable independently (section 5's own J-04 entry) and
delivered the first real cross-language, both-compiled number (~7-8×, re-verified ~7.87× post-Reactant-bump,
`docs/decisions.md` §11 update). The training/weight-gradient arm remains gated on J-01/J-02, unchanged
— any "surpassing JAX" statement about TRAINING (not inference) still has that precondition.

**Continuous:** D-01…D-05 all closed (Documenter.jl + GitHub Pages site stood up, decisions.md
§21-23 written). R-01…R-05 all closed. R-01's first "closed" pass was itself an overclaim
(caught, corrected, then genuinely finished — see R-01's own row for the full history) and
turned up X-06 (`train_autoregressive` driver gaps, a named blocker for C-10) on the real pass.
R-05's graph_construction spot-check found and fixed a real (dormant) duplicate-edge
double-counting bug in the JIT flat path. C-04 (reclassified, not blocking) remains as its
inputs arrive.

**Explicitly deprioritized (agreed, both audits, unchanged):** C-01/C-02 conv stack, C-10 tuner,
dashboarding/pmap/tfds.
