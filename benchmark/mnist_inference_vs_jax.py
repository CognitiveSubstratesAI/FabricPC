#!/usr/bin/env python3
"""
Compiled PC INFERENCE (relaxation loop only, no weight-gradient training) vs upstream's own
jax.jit — the MNIST-MLP architecture (x(784) -> h(128,tanh) -> y(10), same shapes/config as
examples/jit_inference.jl and Tier D's mnist track, docs/AUDIT_REGISTER.md section 6).

SCOPE (read docs/AUDIT_REGISTER.md section 5 J-01/J-02/J-04 + section 6 Tier D, and
docs/decisions.md sections 11/22 before touching this file): J-01/J-02 block compiling
`compute_local_weight_gradients` (the WEIGHT-gradient training step) under Reactant+Enzyme --
that is NOT what this benchmarks. This benchmarks `run_inference` (the E-step latent
relaxation loop) only: params and clamps ("x" the input, "y" the target -- train-style
clamping, matching examples/jit_inference.jl) are FIXED; only "h" relaxes. No weight update,
no optax, no gradient w.r.t. params anywhere in this file.

RNG TRAP (same discipline as Tier A-D, docs/AUDIT_REGISTER.md section 6): this script is the
SOLE source of the weights/batch -- KaimingInitializer-initialized via jax.random, dumped as
raw Fortran-order Float32 binaries (+ a plain-text .shapes sidecar) so the Julia side
(benchmark/mnist_inference_vs_jax.jl) builds its GraphParams/clamps DIRECTLY from these
concrete arrays, never calling its own initialize_params/RNG. Apples-to-apples numerically,
not just a timing comparison.

Run (pinned upstream venv, per this repo's CLAUDE.md JIT-lane discipline):
    /home/shivaji1012/dev-zone/FabricPC/.venv-fixtures/bin/python3 \
        benchmark/mnist_inference_vs_jax.py [outdir]
"""

import os
import sys
import time
import json

import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import KaimingInitializer
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly.graph_construction import TaskMap, graph
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.linear_explicit_grad import LinearExplicitGrad

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------------------
# Same architecture as examples/jit_inference.jl (already-validated Layer-2 example) and
# Tier D's mnist track: x(784) -> h(128, tanh) -> y(10, Identity+Gaussian). NODE-CLASS CHOICE
# mirrors scripts/generate_tier_d_mnist_fixtures.py exactly: "x" (in_degree=0) = plain Linear
# (use_bias=False -- LinearExplicitGrad's forward_and_latent_grads has no terminal-node
# special case, see that script's docstring); "h"/"y" = LinearExplicitGrad, matching Julia's
# fused `Linear` bit-for-bit (Tier B/C/D already established this).
# ---------------------------------------------------------------------------------------
B, DX, DH, DY = 256, 784, 128, 10
ETA_INFER = 0.1
INFER_STEPS = 20

key = jax.random.PRNGKey(7)


def split():
    global key
    key, sub = jax.random.split(key)
    return sub


x_node = Linear(shape=(DX,), name="x", use_bias=False)
h_node = LinearExplicitGrad(shape=(DH,), name="h", activation=TanhActivation())
y_node = LinearExplicitGrad(shape=(DY,), name="y", energy=GaussianEnergy())

inference = InferenceSGD(eta_infer=ETA_INFER, infer_steps=INFER_STEPS, latent_decay=0.0)
structure = graph(
    nodes=[x_node, h_node, y_node],
    edges=[Edge(x_node, h_node), Edge(h_node, y_node)],
    task_map=TaskMap(x=x_node, y=y_node),
    inference=inference,
)

edge_xh = "x->h:in"
edge_hy = "h->y:in"

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

Xb = jax.random.normal(split(), (B, DX))
Yb = jax.random.normal(split(), (B, DY))
clamps = {"x": Xb, "y": Yb}

init_state = initialize_graph_state(
    structure, B, split(), clamps=clamps, params=params
)


# ---------------------------------------------------------------------------------------
# Dump shared inputs for the Julia side (RNG trap): raw C-order (row-major) Float32 binaries
# + a plain-text shapes sidecar (avoids depending on NPZ.jl in the benchmark/jit Julia env).
# NOTE: `ndarray.tofile()` ALWAYS serializes in C order regardless of the array's actual
# memory layout (confirmed empirically -- `np.asfortranarray(a).tofile()` writes IDENTICAL
# bytes to `a.tofile()`; numpy's own docs say as much). An earlier version of this script
# called `np.asfortranarray(...).tofile()` believing it produced column-major bytes for a
# cheap Julia `reshape` -- it did not, silently scrambling every non-1xN array (Wxh/Why/Xb/Yb)
# on the Julia read side (row-major bytes read back as column-major). The companion .jl loader
# now explicitly un-transposes (`permutedims(reshape(data, reverse(shape)), ndims:-1:1)`) to
# correctly invert a genuine C-order dump instead. Caught by comparing a live in-memory JAX
# array against its own disk round-trip (see benchmark git history / session notes) --
# textbook lesson: verify a serialization round-trip numerically, don't assume tofile()
# respects `asfortranarray`.
# ---------------------------------------------------------------------------------------
def dump(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    a.tofile(os.path.join(OUTDIR, f"{name}.bin"))
    return name, list(a.shape)


shapes = dict(
    [
        dump("Wxh", Wxh), dump("bh", bh),
        dump("Why", Why), dump("by", by),
        dump("Xb", Xb), dump("Yb", Yb),
    ]
)
with open(os.path.join(OUTDIR, "shapes.txt"), "w") as f:
    for name, shp in shapes.items():
        f.write(f"{name} {' '.join(str(s) for s in shp)}\n")

# ---------------------------------------------------------------------------------------
# Eager reference (correctness oracle, not timed for the headline number).
# ---------------------------------------------------------------------------------------
eager_final = run_inference(params, init_state, clamps, structure)
eager_final = jax.block_until_ready(eager_final)

# ---------------------------------------------------------------------------------------
# jax.jit-compiled inference. Closes over `clamps`/`structure` (Python objects), matching
# upstream's OWN idiom for this exact call (fabricpc/training/train.py:352,
# `step_fn = jax.jit(lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k))` --
# structure/optimizer closed over, never passed as a traced pytree arg) -- not a benchmark
# invention.
# ---------------------------------------------------------------------------------------
jit_run = jax.jit(lambda p, s: run_inference(p, s, clamps, structure))

t_compile0 = time.perf_counter()
jit_final = jax.block_until_ready(jit_run(params, init_state))
t_compile1 = time.perf_counter()
compile_time_s = t_compile1 - t_compile0

# Numerics check: JIT must match eager (both computing the SAME thing) before timing means
# anything.
max_abs_diff = 0.0
for name in ("x", "h", "y"):
    d = float(jnp.max(jnp.abs(jit_final.nodes[name].z_latent - eager_final.nodes[name].z_latent)))
    max_abs_diff = max(max_abs_diff, d)

# Steady-state (warm/compiled) wall-clock: min-of-N, block_until_ready each call (avoids the
# async-dispatch trap documented in decisions.md section 11 -- "timing the thunk WITHOUT
# Array(...)/block_until_ready shows a bogus ~1000x").
N = 30
for _ in range(5):
    jax.block_until_ready(jit_run(params, init_state))
times = []
for _ in range(N):
    t0 = time.perf_counter()
    jax.block_until_ready(jit_run(params, init_state))
    t1 = time.perf_counter()
    times.append(t1 - t0)
jit_steady_ms = min(times) * 1e3

# Eager steady-state, for reference (NOT the headline comparison -- Julia's own eager Dict
# path is a materially different implementation; this is just upstream-internal context).
eager_times = []
for _ in range(5):
    jax.block_until_ready(run_inference(params, init_state, clamps, structure))
for _ in range(10):
    t0 = time.perf_counter()
    jax.block_until_ready(run_inference(params, init_state, clamps, structure))
    t1 = time.perf_counter()
    eager_times.append(t1 - t0)
eager_steady_ms = min(eager_times) * 1e3

# Dump JIT-path converged z_latents for the Julia-side numerics cross-check, appending their
# shapes to the sidecar (written above, before these existed).
with open(os.path.join(OUTDIR, "shapes.txt"), "a") as f:
    for name in ("x", "h", "y"):
        n1, s1 = dump(f"jax_jit_{name}_z_latent", jit_final.nodes[name].z_latent)
        f.write(f"{n1} {' '.join(str(s) for s in s1)}\n")
        n2, s2 = dump(f"jax_eager_{name}_z_latent", eager_final.nodes[name].z_latent)
        f.write(f"{n2} {' '.join(str(s) for s in s2)}\n")

result = dict(
    batch=B, dx=DX, dh=DH, dy=DY, eta_infer=ETA_INFER, infer_steps=INFER_STEPS,
    jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
    compile_time_s=compile_time_s,
    jit_steady_ms=jit_steady_ms,
    eager_steady_ms=eager_steady_ms,
    jit_vs_eager_speedup=eager_steady_ms / jit_steady_ms,
    max_abs_diff_jit_vs_eager=max_abs_diff,
)
with open(os.path.join(OUTDIR, "jax_result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
