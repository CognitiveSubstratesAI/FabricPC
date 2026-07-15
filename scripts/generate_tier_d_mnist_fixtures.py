#!/usr/bin/env python3
"""
Tier D conformance fixture generator, MNIST-MLP track (docs/AUDIT_REGISTER.md section 6):
END-TO-END -- a short real multi-step TRAINING RUN on an assembled graph, proving the whole
training loop (not just one step -- that was Tier C's job) computes what upstream computes.
Dumped from the PINNED upstream JAX FabricPC checkout into one .npz consumed by
test/conformance/test_tier_d_mnist.jl.

TOLERANCE REQUIREMENT FOR test/conformance/test_tier_d_mnist.jl (MUST, checked in Adversarial
Re-verify): that Julia test file MUST use rtol=atol=1e-4 (looser than Tier C's 1e-5) when
comparing against this fixture's arrays, and MUST carry an in-file comment at the tolerance
definition that explicitly cites, VERBATIM, the reason: "Float32 BLAS associativity across
multiple inference steps". This is not a bare-number tolerance -- it must be justified in the
test file's own text, at the point the tolerance is defined, using that exact phrase (this
fixture chains 5 train_step calls x 30 infer_steps = 150 inner relaxation iterations with
real SGD updates threaded between steps, so floating-point summation order can legitimately
differ by more than Tier C's single-step 1e-5 bound without indicating a real bug).

RNG TRAP (same discipline as Tier A/B/C): fixtures carry JAX-GENERATED ARRAYS, never seeds.
`params0` and all 5 batches are generated here via jax.random and dumped as concrete arrays;
the Julia side must build GraphParams/batches DIRECTLY from the loaded arrays, never calling
its own initialize_params or its own RNG for any compared value. This script fixes ITS OWN
jax.random.PRNGKey purely so re-running it is reproducible (stable diffs), not to hand a seed
to Julia.

STATE-INIT RNG NON-ISSUE (same argument as generate_tier_c_fixtures.py, re-verified for this
graph): initialize_graph_state's default FeedforwardStateInit only uses its rng_key as a
fallback placeholder for in-degree-0 non-clamped / recurrent nodes; every in-degree>0 node's
z_latent is overwritten by a deterministic forward pass, and every clamped node's z_latent is
overwritten by set_latents_to_clamps. This graph's only in-degree-0 node is "x", and both
task-map nodes ("x" AND "y") are clamped on EVERY train_step call below -- so the rng_key
argument threaded into each of the 5 train_step calls never affects a single dumped array. Any
concrete key works; no cross-language seed-reproducibility claim is made or needed for it
(contrast the genuine RNG trap on params0/batches above).

ARCHITECTURE SCOPE (explicit user decision, NOT downsized): mirrors examples/mnist_pc.jl's
"plain" (non-muPC) configuration exactly -- x(784) -> h(128, tanh) -> y(10, Identity+Gaussian),
a Linear/Linear/Linear chain, InferenceSGD(eta_infer=0.1, infer_steps=30), plain SGD lr=0.002
(mnist_pc.jl's FPC_ETA/FPC_STEPS/FPC_LR defaults for the mupc=False branch -- see that file's
lines 85-87). The full 784/128/10 shape was kept (not shrunk): the weight matrices are tiny
(784*128 + 128*10 = 101,632 scalars total) and the trajectory is short (5 steps * 30
infer_steps = 150 inner relaxation iterations over a batch of 8), so generation is fast even
at the real architecture size -- no need to invoke the "downsize if impractical" escape hatch.

NODE-CLASS CHOICE (mirrors generate_tier_c_fixtures.py's rationale, re-verified by reading both
class bodies again for THIS graph): "x" (in_degree=0, the terminal source) uses plain `Linear`
-- its generic/autodiff `forward_and_latent_grads` (nodes/base.py:435-453) has the terminal-node
special case (z_mu <- z_latent, zero error/pre_activation/energy/grads) that Julia's fused
`Linear` implements for EVERY node. `LinearExplicitGrad.forward_and_latent_grads`
(linear_explicit_grad.py:49-107) does NOT reimplement that special case -- it unconditionally
calls `node_class.forward` (line 66) + `energy.grad_latent` (line 70) -- so using it for an
in-degree=0 node would silently compute a wrong, nonzero energy from an unconnected bias.
"h"/"y" (in_degree>0) use `LinearExplicitGrad`, matching Julia's fused `Linear` exactly
(Tier B already validated
`LinearExplicitGrad`'s closed forms == Julia's `Linear` bit-for-bit at the isolated-node level;
Tier C validated the loop-level extension of that same correspondence; this fixture is the
multi-step-trajectory extension of Tier C's single-step case).

"x" IS BUILT WITH use_bias=False, MATCHING Tier C's choice for its own "x" node (this
docstring previously argued for a deliberate divergence to use_bias=True, mirroring
examples/mnist_pc.jl's literal `xn = Linear((784,), "x")` call -- that argument was WRONG
and has been corrected after an adversarial re-verify pass actually ran the generator and
hit a crash, not just read the source: `compute_local_weight_gradients` (learning.py:31-33,
learning.jl:22-26) unconditionally returns an EMPTY gradient dict
(`NodeParams(weights={}, biases={})`) for any in_degree==0 node, but with use_bias=True
`params.nodes["x"]` itself has a NON-empty biases dict (`{"b": bx}`) from
`initialize_params`. `optax.sgd(...).update()` derives its `updates` pytree from the
(empty-for-x) grads structure, so `optax.apply_updates(params, updates)` inside `train_step`
(training/train.py:206) raises a pytree-structure `ValueError` (symmetric key-set difference
`{'b'}` vs `{}`) on the very first `train_step` call below -- confirmed by actually running
this script before the fix. With use_bias=False, `params.nodes["x"] ==
NodeParams(weights={}, biases={})` matches the empty grads structure exactly, so no bias for
"x" is generated, initialized, or dumped at all (no `params0_x_*` / `final_params_x_*`
arrays appear in the fixture -- there is nothing dead to keep around).

DATASET (explicit user decision): upstream's real MNIST loader uses TFDS, unavailable in this
venv, so this fixture uses a SYNTHETIC batch instead -- shape (8, 784) float input in [0, 1)
(pixel-scale-like but NOT real pixels) and one-hot (8, 10) labels, generated fresh via
jax.random for each of the 5 steps (5 distinct batches, never reused). The claim this fixture
tests is narrowly "the PC training loop computes the same thing given the same input, chained
across multiple steps" -- NOT "the data comes from the same source as upstream's real MNIST
pipeline". A reader must not assume TFDS/real-MNIST parity from this fixture.

STEP COUNT (explicit user scoping decision): unlike Tier C (exactly one train_step) and unlike
the Tier D transformer track (also exactly one train_step, per that track's own explicit
instruction), MNIST-MLP is the one Tier D fixture that is a genuine short TRAINING RUN: 5
train_step calls on 5 distinct synthetic batches (B=8), threading params AND the optax
opt_state through sequentially so step i's inference genuinely runs on step (i-1)'s trained
weights. This is what distinguishes this track's "end-to-end" claim from Tier C's "one step"
claim -- a params-only or single-step check could not catch a bug that only manifests after
several chained SGD updates (e.g. an accumulation-order or in-place-mutation bug that happens
to cancel out on step 1).

DUMPED CONTENTS (kept intentionally light per the "keep it small" scoping note -- NOT a
phase-by-phase dump like Tier C's 5-phases-per-fixture, since here the interesting claim is the
5-step TRAJECTORY, not any single phase's internals):
  - params0_*        : initial weights/biases (x/h/y), before any training.
  - batch{i}_x/y      : the 5 distinct synthetic batches, i in 1..5 (1-indexed to match the
                        "step i" language used throughout this docstring and the companion
                        Julia test).
  - scalars_energy_step{i} : train_step's returned per-sample energy at each of the 5 steps
                        (computed on the PRE-update params for that step, matching both
                        upstream train_step's and Julia's train_step's documented return
                        semantics -- verified by reading both bodies: energy/final_state come
                        from get_graph_param_gradient, called BEFORE the SGD update).
  - final_params_*    : h/y weights+biases after all 5 SGD updates ("x" has NO params0_x_*/
                        final_params_x_* arrays at all -- use_bias=False, see NODE-CLASS
                        CHOICE above).
  - final_{node}_*    : the full GraphState (z_latent/z_mu/error/energy/pre_activation/
                        latent_grad, x/h/y) returned by the 5th (last) train_step call -- the
                        converged state from batch 5's inference under step-4's trained params,
                        i.e. the deepest, most-accumulated-error checkpoint in the trajectory.

Run with the upstream fabricpc package importable (see generate_tier_a_fixtures.py's
docstring for the venv activation command -- run from THIS repo as CWD):
    cd <this repo> && source /home/shivaji1012/JuliaAGI/dev-zone/FabricPC/.venv-fixtures/bin/activate
    python3 scripts/generate_tier_d_mnist_fixtures.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import KaimingInitializer
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly.graph_construction import TaskMap, graph
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.linear_explicit_grad import LinearExplicitGrad
from fabricpc.training.train import train_step

OUT = {}
key = jax.random.PRNGKey(4242)


def split():
    global key
    key, sub = jax.random.split(key)
    return sub


def put(prefix, **arrays):
    for name, arr in arrays.items():
        OUT[f"{prefix}_{name}"] = np.asarray(arr)


def dump_state(prefix, state):
    for name, ns in state.nodes.items():
        put(
            f"{prefix}_{name}",
            z_latent=ns.z_latent,
            z_mu=ns.z_mu,
            error=ns.error,
            energy=ns.energy,
            latent_grad=ns.latent_grad,
        )


def dump_params(prefix, params):
    for name, np_ in params.nodes.items():
        for edge_key, w in np_.weights.items():
            put(f"{prefix}_{name}", **{f"w__{edge_key}": w})
        for bname, b in np_.biases.items():
            put(f"{prefix}_{name}", **{f"b__{bname}": b})


# ---------------------------------------------------------------------------------------
# Graph: x(784) -> h(128, tanh) -> y(10, Identity+Gaussian). Same shape/hyperparameters as
# examples/mnist_pc.jl's "plain" (non-muPC) configuration -- see ARCHITECTURE SCOPE above.
# ---------------------------------------------------------------------------------------
B, DX, DH, DY = 8, 784, 128, 10
N_STEPS = 5
LR = 0.002
ETA_INFER = 0.1
INFER_STEPS = 30

x = Linear(shape=(DX,), name="x", use_bias=False)                   # NODE-CLASS CHOICE: plain
h = LinearExplicitGrad(shape=(DH,), name="h", activation=TanhActivation())
y = LinearExplicitGrad(shape=(DY,), name="y", energy=GaussianEnergy())

inference = InferenceSGD(eta_infer=ETA_INFER, infer_steps=INFER_STEPS, latent_decay=0.0)
structure = graph(
    nodes=[x, h, y],
    edges=[Edge(x, h), Edge(h, y)],
    task_map=TaskMap(x=x, y=y),
    inference=inference,
)

edge_xh = "x->h:in"
edge_hy = "h->y:in"

# ---------------------------------------------------------------------------------------
# params0: Kaiming-normal weights (matching Linear's own default `weight_init=
# KaimingInitializer()` -- called directly here, not hand-derived, to avoid a
# formula-transcription bug), zero biases (matching Linear.initialize_params, which
# unconditionally zero-inits biases regardless of weight_init). "x" has NO bias entry at all
# (use_bias=False -- see NODE-CLASS CHOICE above; a non-empty bias here would crash
# optax.apply_updates against the empty grads structure for in_degree==0 nodes).
# ---------------------------------------------------------------------------------------
Wxh = KaimingInitializer.initialize(split(), (DX, DH))
bh = jnp.zeros((1, DH))
Why = KaimingInitializer.initialize(split(), (DH, DY))
by = jnp.zeros((1, DY))

params = GraphParams(
    nodes={
        "x": NodeParams(weights={}, biases={}),
        "h": NodeParams(weights={edge_xh: Wxh}, biases={"b": bh}),
        "y": NodeParams(weights={edge_hy: Why}, biases={"b": by}),
    }
)
dump_params("params0", params)

# ---------------------------------------------------------------------------------------
# 5-step training trajectory: fresh synthetic batch each step, params/opt_state threaded
# through sequentially (see STEP COUNT above for why this -- not a single step -- is the
# point of this fixture).
# ---------------------------------------------------------------------------------------
optimizer = optax.sgd(learning_rate=LR)
opt_state = optimizer.init(params)

final_state = None
for i in range(1, N_STEPS + 1):
    Xb = jax.random.uniform(split(), (B, DX), minval=0.0, maxval=1.0)
    labels = jax.random.randint(split(), (B,), 0, DY)
    Yb = jax.nn.one_hot(labels, DY)
    put(f"batch{i}", x=Xb, y=Yb)

    batch = {"x": Xb, "y": Yb}
    rng_key_i = split()
    params, opt_state, energy_i, final_state = train_step(
        params, opt_state, batch, structure, optimizer, rng_key_i
    )
    # NOTE: reshaped to a 1-element 1-D array before put() -- matches Tier B/C's established
    # convention: a bare 0-d jax array round-trips through np.savez/NPZ.jl ambiguously (loads
    # back as a bare scalar, not an Array), a shape-(1,) array does not.
    put("scalars", **{f"energy_step{i}": jnp.reshape(jnp.asarray(energy_i), (1,))})

dump_params("final_params", params)
dump_state("final", final_state)

# ---------------------------------------------------------------------------------------
np.savez("test/conformance/fixtures/tier_d_mnist.npz", **OUT)
print(f"wrote {len(OUT)} arrays to test/conformance/fixtures/tier_d_mnist.npz")
