# PC inference: hoist all-clamped-source forwards and prune gradients into clamped nodes

**Target:** `trueagi-io/FabricPC`
**File touched:** `fabricpc/core/inference.py` (`InferenceBase`)
**Kind:** two *exact* (bit-identical) FLOP reductions to `run_inference`; no numerical change, no API change.

## Summary

The inference loop `InferenceBase.run_inference` (`inference.py:245-270`) drives a
`jax.lax.fori_loop` (`inference.py:269`) whose body `inference_step`
(`inference.py:86-112`) runs three phases every step: `zero_grads` →
`forward_value_and_grad` → `update_latents`. Two categories of per-step work are
provably redundant when some nodes are **clamped** (their `z_latent` is held fixed
by the caller and is skipped by `update_latents`, `inference.py:219`):

1. **HOIST** — For any node whose *every* in-edge source is clamped, the node's
   prediction `z_mu` is a function of clamped-constant inputs only, so it is
   **identical on every one of the `T` inference steps**. The prediction (the
   forward mat-mul) can be computed **once before the loop** instead of `T` times
   inside it.

2. **PRUNE** — The input-gradient that `forward_value_and_grad` accumulates *into*
   a clamped source node (`inference.py:192-197`) is **never read**: the only
   consumer, `update_latents`, skips clamped nodes (`inference.py:219`). Computing
   that gradient (the backward mat-mul transpose) and accumulating it is dead work.

Both are exact by construction — see *Correctness* below. Together they remove, per
step, both the forward mat-mul *and* the backward mat-mul-transpose of every
all-clamped-source node, leaving only the O(node-size) elementwise self-gradient.

The win is **topology dependent**, not a single speedup number. It is large for an
MLP whose clamped input is the widest node (MNIST-MLP: 98.7% of per-step FLOPs are
clamped-incident, ~27× FLOP ceiling), modest for a uniform deep chain (exactly
`1/(L+1)`), and **exactly zero for a transformer LM** whose clamp feeds an
`EmbeddingNode`. This document states the transformation, quantifies each regime,
and gives the one correctness caveat that keeps the change bit-identical.

---

## 1. The two optimizations, against the current upstream code

### Where the redundant work lives

`inference_step` (`inference.py:86-112`) per step:

```
state = cls.zero_grads(...)               # inference.py:104  → inference.py:114-132
state = cls.forward_value_and_grad(...)   # inference.py:107  → inference.py:134-199
state = cls.update_latents(...)           # inference.py:110  → inference.py:201-227
```

Inside `forward_value_and_grad` (`inference.py:154-199`), for **every** node each
step:

- inputs are gathered from the sources' `z_latent` via `gather_inputs`
  (`inference.py:47-63`, the read is `state.nodes[node].z_latent` at
  `inference.py:59-61`);
- `node_class.forward_and_latent_grads(...)` (`inference.py:171-177`) runs the
  forward and, in the default autodiff path, differentiates it w.r.t. **both**
  inputs and `z_latent` (`base.py:507-518`: `jax.value_and_grad(energy_fn,
  argnums=(0, 1))` returning `(input_grads, self_grad)`);
- `self_grad` is scaled and added to the node's own `latent_grad`
  (`inference.py:182-186`);
- each per-edge `input_grad` is accumulated **into the source node's**
  `latent_grad` (`inference.py:191-197`).

For a node whose sources are all clamped, `gather_inputs` returns the *same*
constant arrays every step (clamped `z_latent` is never written, because
`update_latents` skips clamped nodes at `inference.py:219`). Hence in
`Linear.forward` (`linear.py:157-224`) the prediction `pre_activation = Σ x·W (+b)`
and `z_mu = activation(pre_activation)` (`linear.py:198-214`) are **step-invariant**
— they depend only on inputs and params, never on `state.z_latent`
(`z_latent` enters only at `error = z_latent - z_mu`, `linear.py:214`).

### Optimization 1 — HOIST the all-clamped-source forward out of `fori_loop`

**Rule.** Let `C = clamps.keys()`. Precompute, once, the set

```
H = { n ∈ structure.nodes : in_edges(n) ≠ ∅  and  every source(in_edges(n)) ∈ C }
```

For `n ∈ H`, compute its prediction (`z_mu`, `pre_activation`) **before** the
`fori_loop` and store it in the initial carry. Inside `forward_value_and_grad`,
for `n ∈ H`, **skip the forward mat-mul** and reuse the cached `z_mu`; only the
`z_latent`-dependent residual is recomputed each step (`error = z_latent - z_mu`,
the Gaussian `self_grad = z_latent - z_mu`, and the energy). Those residuals are
O(node-size) elementwise; the eliminated mat-mul is O(in-features × out-features).

- **Touch point:** `forward_value_and_grad`, `inference.py:171-177` — branch nodes
  in `H` to a path that consumes the cached prediction instead of calling the
  full autodiff forward. The pre-loop computation is a new step added to
  `run_inference` (`inference.py:245-270`) before `jax.lax.fori_loop`
  (`inference.py:269`).
- The node's own latent update is *not* hoisted: `n ∈ H` is itself usually an
  unclamped latent (e.g. the MNIST hidden layer), so its `z_latent` still updates
  each step (`inference.py:219-225`) and it still receives downstream successors'
  gradients (`inference.py:191-197`). Only its **prediction of itself from clamped
  inputs** is constant and hence hoisted.

### Optimization 2 — PRUNE input-gradient accumulation into clamped nodes

**Rule.** In the accumulation loop `inference.py:191-197`:

```python
for edge_key, grad in inedge_grads.items():
    source_name = structure.edges[edge_key].source
    latent_grad = state.nodes[source_name].latent_grad + grad   # inference.py:194
    state = update_node_in_state(state, source_name, latent_grad=latent_grad)
```

**skip the iterations whose `source_name ∈ clamps`.** That accumulated value is
read only by `update_latents`, which computes a new latent solely for
`node_name not in clamps` (`inference.py:219`). A clamped source's `latent_grad`
is therefore write-only and dead.

- **Touch point:** `forward_value_and_grad`, `inference.py:191-197`.
- The matching **producer-side** win: when *all* of a node's in-edges point at
  clamped sources, none of its `input_grads` are read, so the node need not
  compute `input_grads` at all — differentiate only w.r.t. `z_latent`
  (`base.py:514-516`, drop `argnums=0`). This removes the **backward
  mat-mul-transpose** `Wᵀ · dE/dz_mu`. `EmbeddingNode` already hand-writes this
  pruning: `input_grads = {edge_key: jnp.zeros_like(inp) ...}` with the comment
  *"Discrete indices: no gradient flows back through the input edge."*
  (`transformer_v2.py:117-130`, zeros at `:121-123`).

### What remains in the loop for an all-clamped-source node

After HOIST + PRUNE, per step, an all-clamped-source node contributes only: zero
its `latent_grad` (kept — see *Correctness*), read cached `z_mu`, compute
`error`/energy/`self_grad` (all O(node-size) elementwise), add `self_grad`
(`inference.py:182-186`), and receive any downstream successors' gradients. Both
mat-muls (forward and input-grad backward) are gone.

---

## 2. The win is a static, topology-dependent FLOP fraction

Define `f` = fraction of a single `inference_step`'s mat-mul/MAC FLOPs that are
**clamped-incident** = (forward MACs of all-clamped-source nodes, hoistable) +
(input-grad backward MACs feeding clamped sources, prunable). The realizable FLOP
ceiling additionally depends on `infer_steps = T` (default `T = 20`,
`InferenceSGD.__init__`, `inference.py:289`), because the hoisted forward is not
deleted but *amortized* — it runs once outside the loop — and the O(size)
self-grad residual stays in the loop.

Verified regimes (measured this session via operand-level HLO analysis of the
compiled graphs; cited, not recomputed here):

| Model | Clamped node | `f` (clamped-incident per-step FLOPs) | FLOP ceiling |
|---|---|---|---|
| **MNIST-MLP** `x[784]→h[128]→y[10]`, `x`,`y` clamped | input `x` is the **widest** node | **98.7%** | **~27×** (at `T=20`) |
| **Uniform `L`-layer chain**, clamped input | input | **exactly `1/(L+1)`** (3.03% at `L=32`) | small, → 0 as `L`→∞ |
| **Transformer LM**, clamped tokens | tokens → `EmbeddingNode` | **exactly 0.0%** | **1× (nothing)** |

### Why MNIST-MLP wins big
`h`'s only source is the clamped input `x[784]`, so `h.z_mu = x·W`
(`x[batch,784] @ W[784,128]`, `linear.py:198-203`) is hoistable, and the
input-grad `dE_h/dz_x = dE/dz_mu · Wᵀ` (`128×784` MACs) accumulated into clamped
`x` (`inference.py:191-197`) is prunable. Because the clamped input is the widest
tensor in the net, these two mat-muls are 98.7% of the step's MACs; amortizing the
forward over `T = 20` and dropping the backward yields the ~27× FLOP ceiling.

### Why a deep chain wins little
In a uniform `L`-layer chain only the *first* hidden layer has an all-clamped
source (the input); the other `L` layers have unclamped sources and are untouched.
Clamped-incident FLOPs are one layer out of `L+1` symmetric layers → exactly
`1/(L+1)` (3.03% at `L=32`). The optimization is **structurally irrelevant** to
model depth.

### Why the transformer gets exactly nothing
The clamp feeds `EmbeddingNode` (`transformer_v2.py:46-130`), not a `Linear`:

- Its forward is a **gather**, `z_mu = params.weights["embeddings"][indices]`
  (`transformer_v2.py:109`) — **0 multiply-adds**. Hoisting a 0-MAC op saves 0.
- Its input-gradient is **discrete-zeros by construction**:
  `input_grads = {edge_key: jnp.zeros_like(inp) ...}` (`transformer_v2.py:121-123`).
  Pruning a zeros-accumulate saves 0.

So the flagship model gets **exactly zero** benefit. This must be stated plainly in
the PR: the optimization targets *dense-mat-mul nodes fed by wide clamped inputs*,
which is the MLP-on-images regime, not language models. It is a real, sometimes
large win where it applies and a provable no-op elsewhere — never a regression.

---

## 3. Correctness caveat that keeps PRUNE bit-identical (do not remove the allocation)

`zero_grads` (`inference.py:114-132`) iterates **every** node, clamped included,
and resets `latent_grad` from its own dtype/shape:

```python
# inference.py:127-130
for node_name in structure.nodes:
    node_state = state.nodes[node_name]
    zero_grad = jnp.zeros_like(node_state.latent_grad)   # inference.py:129
    state = update_node_in_state(state, node_name, latent_grad=zero_grad)
```

with this verbatim upstream comment (`inference.py:125-126`):

> Use latent_grad (not z_latent) as the dtype/shape source: z_latent may
> carry an integer clamp dtype on source nodes, but gradients are float.

**Therefore PRUNE must skip the accumulation *write*, not the field.** The
`latent_grad` array must stay **allocated as a float zero** for clamped nodes,
because `zero_grads` uses it — not `z_latent` — as the dtype/shape source, exactly
because a clamped source's `z_latent` may be an **integer** clamp (e.g. token
indices). Deleting or narrowing `latent_grad` on clamped nodes would either break
`zero_grads` (nothing to `zeros_like`) or, if it fell back to `z_latent`, produce
an **integer** "gradient" buffer — a real bug the current comment is guarding
against. The compute path is likewise already gated:
`InferenceSGD.compute_new_latent` documents that it only ever runs on non-clamped
(float) nodes (`inference.py:294-309`, esp. the precondition comment at
`inference.py:296-301`).

**Net:** PRUNE removes only the `+= grad` at `inference.py:192-197` for clamped
targets. The clamped node's `latent_grad` remains the all-zeros float array that
`zero_grads` produced — the same shape and dtype as before.

---

## 4. Why both are EXACT (bit-identical, not approximations)

**HOIST is bit-identical.** For a node in `H`, the inputs read by `gather_inputs`
(`inference.py:59-61`) are the identical constant arrays on every step (clamped
`z_latent` is never written; `update_latents` skips it at `inference.py:219`).
JAX forward is a pure, deterministic function of `(params, inputs)`
(`linear.py:157-224`); the same operands through the same op yield the same bits.
Computing `z_mu` once vs. `T` times gives **the identical `z_mu`**, so every
downstream value (`error`, energy, `self_grad`, and thus all latent updates) is
unchanged to the bit.

**PRUNE is bit-identical.** The only value that differs between the original and
pruned runs is a clamped node's own `latent_grad` (original: `zeros +` accumulated
input-grads; pruned: `zeros`). That value is **provably never read**:
`update_latents` computes a new `z_latent` only for `node_name not in clamps`
(`inference.py:219`), and the clamped node's `z_latent` is held fixed by the
caller. No observable output — the final `z_latent` of any node — depends on it.
The divergent buffer is dead; every returned array is bit-identical.

Both optimizations are pure dead-code / loop-invariant-code elimination guided by
the clamp set, not numerical approximations. There is no tolerance to tune and no
accuracy trade-off.

---

## 5. Suggested implementation shape (upstream Python)

- In `run_inference` (`inference.py:245-270`), before `jax.lax.fori_loop`
  (`inference.py:269`): compute `H` from `structure` + `clamps`, run each `n ∈ H`'s
  forward once, and seed the initial carry's `NodeState.z_mu`/`pre_activation`.
- In `forward_value_and_grad` (`inference.py:154-199`): (a) for `n ∈ H`, reuse the
  cached `z_mu` and take a `z_latent`-only gradient (`base.py:514-516` with
  `argnums=1`), skipping the forward and backward mat-muls; (b) in the accumulation
  loop (`inference.py:191-197`), guard `if source_name not in clamps` before the
  `+=`.
- Leave `zero_grads` (`inference.py:114-132`) and the `latent_grad` allocation
  untouched (see §3).
- `H` and the clamped-source mask are static given `(structure, clamps.keys())`, so
  they can be computed at trace time and closed over — no per-step Python control
  flow inside the `fori_loop`.

A differential test is trivial and should gate the PR: run `run_inference`
before/after on MNIST-MLP, deep chain, and transformer fixtures and assert
`jnp.array_equal` (not `allclose`) on every node's final `z_latent`.

---

## Verified upstream citations

All line numbers are `fabricpc/core/inference.py` unless noted, verified against
the code bodies (not names/comments) in this contribution:

- `run_inference` outer loop: `inference.py:245-270`; `jax.lax.fori_loop`:
  `inference.py:269`; `body_fn`: `inference.py:263-266`.
- `inference_step` three phases: `inference.py:86-112` (calls at `:104`, `:107`,
  `:110`).
- `zero_grads`: `inference.py:114-132`; verbatim int-dtype comment:
  `inference.py:125-126`; `jnp.zeros_like(node_state.latent_grad)`:
  `inference.py:129`.
- `forward_value_and_grad`: `inference.py:134-199`; per-node forward call:
  `inference.py:171-177`; `self_grad` accumulation: `inference.py:182-186`;
  **input-grad accumulation into sources (PRUNE site):** `inference.py:191-197`
  (`source_name = structure.edges[edge_key].source`, `:193`;
  `latent_grad = ... + grad`, `:194`).
- `update_latents` clamp skip (the sole consumer): `inference.py:215-225`, gate at
  `inference.py:219` (`if node_name not in clamps:`).
- `gather_inputs` reads sources' `z_latent`: `inference.py:47-63` (read at `:59-61`).
- `InferenceSGD` default `infer_steps=20`: `inference.py:289`;
  `compute_new_latent` float-only precondition: `inference.py:294-309` (comment
  `:296-301`).
- `Linear.forward` prediction is input-only (`z_mu` independent of `z_latent`):
  `fabricpc/nodes/linear.py:157-224` (mat-mul `:198-203`, bias `:206-207`,
  activation `:210-211`, `error = z_latent - z_mu` `:214`).
- Base autodiff computing `(input_grads, self_grad)` via
  `value_and_grad(argnums=(0,1))`: `fabricpc/nodes/base.py:507-518`; terminal
  (in-degree-0 source) node returns zero input/self grads:
  `fabricpc/nodes/base.py:465-483`.
- `EmbeddingNode` (transformer 0.0% case): gather forward `z_mu =
  params.weights["embeddings"][indices]`: `fabricpc/nodes/transformer_v2.py:109`;
  discrete-zeros input-grad: `fabricpc/nodes/transformer_v2.py:117-130` (zeros at
  `:121-123`, comment `:120`).
