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
| F-04 | MEDIUM (revised from claimed LOW) | Dual AD-backend footgun | **VERIFIED CLOSED** (`c8fb541`) |
| F-05 | MEDIUM | `compute_loss`/clamp one-hot contract divergence | **VERIFIED CLOSED** |

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

### F-04 — dual AD-backend load footgun — VERIFIED CLOSED

**CONFIRMED**, and **severity revised LOW → MEDIUM**: the trigger (`using Zygote; using Enzyme` in
one session, or vice versa) has the lowest bar of all four findings, and the failure mode is
uniquely bad — fully silent method-table overwrite at load time (empirically verified: no stderr
warning), then an opaque, non-debuggable LLVM abort at the first call that hits code only the
losing backend could handle. Distinct from — and NOT mitigated by — the existing "world-age
gotcha" fallback (`_ENZYME_HINT`), which can only fire when *neither* backend has loaded yet.

**Fix**: `_register_ad_backend!` (new, `nodes/autodiff.jl`) records which backend loaded first;
both `FabricPCZygoteExt` and `FabricPCEnzymeExt` call it at module-load time and it raises
immediately, with an actionable message, if the other backend tries to load afterward in the same
session — catches the mistake at `using X` time instead of at an arbitrary later call site.

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

---

## 2. Fidelity findings — deliberate divergences (document, don't fix)

No action this round. Unchanged from the original register.

| ID | Item | Status |
|----|------|--------|
| X-01 | All-float clamp pipeline | OPEN |
| X-02 | Mask slot removed from TransformerBlock | OPEN |
| X-03 | rope_theta hardcoded 10000 | OPEN |
| X-04 | num_heads default 8→4 | OPEN |
| X-05 | GraphNamespace deferred | DECIDE |

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
| C-04 | HIGH | PC-vs-backprop comparison harness | **DECIDE → OPEN, reclassified.** Reflective cross-check confirmed the new-document framing: NOT structurally blocked by the never-port-backprop decision. `ab_experiment.py`/`statistics.py` treat each arm's `params`/`structure` as fully opaque duck-typed objects; a Julia arm on a plain Lux.jl/Flux.jl MLP never touches FabricPC internals and is not a "port of `train_backprop.py`" in any sense. Real new code still needed (stats module + `ExperimentArm`/`ABExperiment` port + a small Lux arm) — moderate/large effort stands, "blocked" does not. |
| C-05 | MED | InferenceSGDNormClip | **IN PROGRESS** |
| C-06 | MED | evaluate_autoregressive | **IN PROGRESS** |
| C-07 | MED | GlobalStateInit / NodeDistributionStateInit | OPEN |
| C-08 | MED | ND Kaiming/Xavier fan-in math | DEFERRED (tied to C-01) |
| C-09 | MED | muPC never validated on either transformer node family | Umbrella over F-01+F-02, both now closed — test coverage for TransformerBlock landed with F-01's fix; MhaResidualNode/LnMlp1Node/Mlp2Residual under real `MuPCConfig` remains OPEN. |
| C-10 | LOW | BayesianTuner (537-line #28 rework) | DEFERRED |
| C-11 | LOW | StorkeyHopfield weight_init: Zeros (Julia) vs Xavier (upstream) | **VERIFIED CLOSED** (`52bb6a0`). Reflective cross-check resolved the R-03 read: neither implementation has a classical Storkey/Hebbian accumulation rule — W is 100% PC-gradient-learned in both — so upstream's own stated criterion for "Xavier matters when W learns via PC gradients" applies, confirming this was a port slip (Julia copied the internal `None`-fallback constant, not the constructor's actual default), not a legitimate design choice. Fixed to `XavierInitializer()`. |
| C-12 | LOW | ~17 further API-surface/UX divergences (IDE list) | DEFERRED |

**Known-deferred, confirmed absent, no action:** multi-GPU/pmap · train_backprop.py port (anti-thesis; see C-04) · Aim dashboarding/experiments · tfds loaders · Optuna.

## 5. JIT/performance lane

No action this round. Unchanged from the original register (J-01 through J-07, all OPEN/DEFERRED).

## 6. Conformance harness

**Status: parallel track, not yet started.** The reflective cross-check confirmed "fixtures before
fixes" was correctly killed as a *gate* (F-02's property test is analytic, F-01's expected gradient
is closed-form `autodiff_grad × a` — neither needed a JAX fixture to land). The harness remains the
milestone artifact for proving JAX-equivalence and should still be built — fixtures generated from
upstream HEAD (`316367c`), per the A-01 resolution (the tree is already synced; no re-sync step
needed first). Tier A/B/C/D design unchanged from the original register.

## 7. Documentation debt

| ID | Item | Status |
|----|------|--------|
| D-01 | README.md badly stale ("Phase A scaffold") | **VERIFIED CLOSED** (`67a54da`) — rewritten to reflect actual shipped state (muPC, transformer + decomposed transformer_v2 family, Reactant/XLA JIT, StorkeyHopfield, AdamW, natural-gradient preconditioners). |
| D-02 | `_tb_apply_rope` comment claimed a layout divergence that doesn't exist | **VERIFIED CLOSED** (`c8fb541`) — confirmed by spot-check that the math genuinely matches upstream (see §3); reworded the comment to say the convention coincides with upstream's rather than implying no match was attempted. |
| D-03 | transformer_decomposed.jl header calls Embedding/VocabProjection "a separate follow-up" — they're implemented below it in the same file | OPEN (not touched this round) |
| D-04 | decisions.md entries: X-01…X-05, layer map (A-02), backend roles | OPEN |

## 8. Residual review queue

| ID | Pair | Status |
|----|------|--------|
| R-01 | train_autoregressive.jl vs .py (full) | Partially addressed as a side effect: full diff review during commit prep confirmed the Blue-reformat pass on this file was whitespace-only except for one formatter-introduced comment defect (cleaned up, `ecb031a`). The X-01 dtype-preservation question and the upstream `_generation_step` index-cast comparison itself are still OPEN — this was a formatting-safety check, not the R-01 review. |
| R-02 | FabricPCZygoteExt / FabricPCEnzymeExt internals | OPEN |
| R-03 | storkey_hopfield pair (resolves C-11 init question) | **ADDRESSED** as part of the C-11 close — see C-11. |
| R-04 | transformer_v2.py (upstream HEAD after A-01) | OPEN (A-01 sync itself is done; the stage-boundary/GraphNamespace read is not) |
| R-05 | Spot-checks: SoftmaxActivation.jacobian, last-axis softmax on rank-3, graph_construction, natural_gradients pair | OPEN |

## 9. Execution sequence

**Batch 1 — correctness gate: DONE, informally.** The reflective cross-check refuted the original
hard-gate design ("F-01+F-02 fixed and green *before* any batch-2 work") — nothing was live-broken,
and the two bugs were independently fixable/testable, not a coupled unit. F-01 (TransformerBlock)
+ F-02 + F-03 + F-04 landed as one batch (`c8fb541`) with independent, targeted regression tests
rather than a combined fixture-gated test. No fixture generator was needed to close Batch 1.

**Batch 2 — validated showcase: IN PROGRESS.** C-03 (Tiny Shakespeare char dataloader) —
**VERIFIED CLOSED**, real-corpus smoke-tested. C-05 (InferenceSGDNormClip) → C-06
(evaluate_autoregressive) queued next, proceeding in parallel with (not blocked by) the
conformance-harness track. C-03 landing first was deliberate: highest ROI (turns the
already-mature LM stack from synthetic-only to demonstrably working on real text) and unblocks C-06
being meaningful (nothing to evaluate perplexity on without real data) — confirmed: C-03's live
smoke test found zero OOV chars between the train vocab and validation/test splits, so C-06 can
evaluate on either split with no additional vocab-handling work. Per A-01/C-09: before assuming no
correctness exposure, confirm whether this work turns on `MuPCConfig` together with
`TransformerBlock`/the decomposed family — if it does, F-01/F-02's fix already covers
TransformerBlock; MhaResidualNode/LnMlp1Node under real muPC scaling remains untested (C-09).

**Batch 3 — performance claim: OPEN, not started.** J-01 → J-02 → J-03 (+J-07) → J-04 benchmark vs
`jax.jit`. Precondition for any "surpassing JAX" statement.

**Continuous:** D-01/D-02 closed; D-03/D-04 remain for the next documentation pass. R-01/R-03
partially/fully addressed; R-02/R-04/R-05 open. Decisions on C-04 (now reclassified, not blocking),
C-11 (closed), X-05 remain as their inputs arrive.

**Explicitly deprioritized (agreed, both audits, unchanged):** C-01/C-02 conv stack, C-10 tuner,
dashboarding/pmap/tfds.
