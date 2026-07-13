#!/usr/bin/env python3
"""
Tier C conformance fixture generator (docs/AUDIT_REGISTER.md section 6): loop-level --
inference_step -> run_inference -> one train_step, on a real *assembled graph* (not an
isolated node like Tier B), dumped from the PINNED upstream JAX FabricPC checkout into
one .npz consumed by test/conformance/test_tier_c.jl (rtol/atol 1e-5, matching Tier B).

RNG TRAP (same as Tier A/B): fixtures carry JAX-GENERATED ARRAYS, never seeds. Params and
batch data are generated here via jax.random and dumped as arrays; the Julia side builds
GraphParams DIRECTLY from the loaded arrays (never calls its own initialize_params).

STATE-INIT RNG NON-ISSUE (verified by reading both state_initializer.py and .jl): the
default FeedforwardStateInit draws z_latent from `rng_key` only as a FALLBACK placeholder
for in-degree-0 non-clamped / recurrent nodes; every in-degree>0 node's z_latent is then
OVERWRITTEN by a deterministic forward pass (z_latent <- z_mu), and every clamped node's
z_latent is OVERWRITTEN by set_latents_to_clamps. This fixture's graph has exactly one
in-degree-0 node (the "x" source) and it is always clamped in every call below -> the
rng_key argument to initialize_graph_state never affects a single output array. Both sides
pass an arbitrary fixed key purely to satisfy the API; no seed-reproducibility claim is
made or needed for it (contrast the genuine RNG trap on `params`/`batch` above).

NODE-CLASS CHOICE: "x" (in_degree=0, the terminal source) uses plain `Linear` -- its
generic/autodiff `forward_and_latent_grads` (nodes/base.py:436-454) has the terminal-node
special case (z_mu <- z_latent, zero error/energy/grads) that Julia's fused `Linear` ports
for EVERY node. `LinearExplicitGrad.forward_and_latent_grads` (linear_explicit_grad.py)
does NOT reimplement that special case (verified by reading its full body) -- using it for
an in-degree=0 node would silently compute a nonzero, WRONG energy from an unconnected
bias. "h"/"y" (in_degree>0) use `LinearExplicitGrad`, matching Julia's Linear exactly (Tier
B already validated LinearExplicitGrad's closed forms == Julia's Linear bit-for-bit at the
isolated-node level; this fixture is the loop-level extension of that same correspondence).
"x" is built with use_bias=False (a source node's bias is dead weight -- never read by the
in-degree=0 forward_and_latent_grads special case -- so leaving it out avoids the unrelated,
harmless "gradient dict for a param the optimizer never touches" bookkeeping quirk in
compute_local_weight_gradients/learning.py:31-33 entirely).

Run with the upstream fabricpc package importable (see generate_tier_a_fixtures.py's
docstring for the venv activation command):
    python3 <this repo>/scripts/generate_tier_c_fixtures.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly.graph_construction import TaskMap, graph
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.linear_explicit_grad import LinearExplicitGrad
from fabricpc.training.train import get_graph_param_gradient, train_step

OUT = {}
key = jax.random.PRNGKey(42)


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
            pre_activation=ns.pre_activation,
            latent_grad=ns.latent_grad,
        )


def dump_params(prefix, params):
    for name, np_ in params.nodes.items():
        for edge_key, w in np_.weights.items():
            put(f"{prefix}_{name}", **{f"w__{edge_key}": w})
        for bname, b in np_.biases.items():
            put(f"{prefix}_{name}", **{f"b__{bname}": b})


# ---------------------------------------------------------------------------------------
# Graph: x(4) -> h(6, tanh) -> y(3, Identity+Gaussian). Small 3-node chain exercising every
# phase of the loop (zero_grads / forward_value_and_grad / update_latents, multiple
# iterations, local weight-gradient learning, one SGD update) -- not just one node in
# isolation (that's Tier B's job).
# ---------------------------------------------------------------------------------------
B, DX, DH, DY = 3, 4, 6, 3

x = Linear(shape=(DX,), name="x", use_bias=False)
h = LinearExplicitGrad(shape=(DH,), name="h", activation=TanhActivation())
y = LinearExplicitGrad(shape=(DY,), name="y", energy=GaussianEnergy())

inference = InferenceSGD(eta_infer=0.1, infer_steps=8, latent_decay=0.0)
structure = graph(
    nodes=[x, h, y],
    edges=[Edge(x, h), Edge(h, y)],
    task_map=TaskMap(x=x, y=y),
    inference=inference,
)

edge_xh = "x->h:in"
edge_hy = "h->y:in"

Wh = jax.random.normal(split(), (DX, DH)) * 0.3
bh = jax.random.normal(split(), (1, DH)) * 0.1
Wy = jax.random.normal(split(), (DH, DY)) * 0.3
by = jax.random.normal(split(), (1, DY)) * 0.1

params = GraphParams(
    nodes={
        "x": NodeParams(weights={}, biases={}),
        "h": NodeParams(weights={edge_xh: Wh}, biases={"b": bh}),
        "y": NodeParams(weights={edge_hy: Wy}, biases={"b": by}),
    }
)
dump_params("params0", params)

Xb = jax.random.normal(split(), (B, DX))
labels = jax.random.randint(split(), (B,), 0, DY)
Yb = jax.nn.one_hot(labels, DY)
put("batch", x=Xb, y=Yb)

clamps = {"x": Xb, "y": Yb}
dummy_key = jax.random.PRNGKey(999)  # see STATE-INIT RNG NON-ISSUE above -- value inert

init_state = initialize_graph_state(structure, B, dummy_key, clamps=clamps, params=params)
dump_state("init", init_state)

# --- one inference_step -----------------------------------------------------------------
state1 = InferenceSGD.inference_step(params, init_state, clamps, structure, inference.config)
dump_state("step1", state1)

# --- run_inference to convergence (structure's own infer_steps=8) -----------------------
converged = run_inference(params, init_state, clamps, structure)
dump_state("converged", converged)

# --- get_graph_param_gradient (local weight grads + energy on the converged state) ------
batch = {"x": Xb, "y": Yb}
rng_key2 = jax.random.PRNGKey(777)
grads, energy, grad_final_state = get_graph_param_gradient(params, batch, structure, rng_key2)
dump_params("grad", grads)
# NOTE: reshaped to a 1-element 1-D array before put() -- matches Tier B's established
# convention (generate_tier_b_fixtures.py:295-297): a bare 0-d jax array round-trips
# through np.savez/NPZ.jl ambiguously (loads back as a bare scalar, not an Array), a
# shape-(1,) array does not.
put("scalars", energy_pretrain=jnp.reshape(jnp.asarray(energy), (1,)))
dump_state("grad_final", grad_final_state)

# --- one train_step (SGD via optax, lr matches Julia's plain-lr train_step) -------------
LR = 0.05
optimizer = optax.sgd(learning_rate=LR)
opt_state = optimizer.init(params)
new_params, new_opt_state, energy2, final_state2 = train_step(
    params, opt_state, batch, structure, optimizer, rng_key2
)
dump_params("trained", new_params)
put("scalars", energy_trainstep=jnp.reshape(jnp.asarray(energy2), (1,)))
dump_state("trained_final", final_state2)

# ---------------------------------------------------------------------------------------
np.savez("test/conformance/fixtures/tier_c.npz", **OUT)
print(f"wrote {len(OUT)} arrays to test/conformance/fixtures/tier_c.npz")
