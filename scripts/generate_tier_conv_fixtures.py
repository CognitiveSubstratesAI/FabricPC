#!/usr/bin/env python3
"""
Tier-conv conformance fixture generator (C-01: ConvNode / MaxPool / AvgPool).

BACKWARD FIRST — deliberately. The forward is the easy half: a conv forward that matches
upstream while the BACKWARD folds in a different order is exactly the invisible-divergence
pattern this conformance effort exists to catch. Every input position under overlapping
receptive fields accumulates up to prod(kernel)/prod(stride) contributions, and that
scatter-add is pure floating-point associativity: upstream's order is whatever XLA's conv
VJP kernel chose, ours is whatever Zygote/Enzyme generate by reversing an im2col
gather->GEMM. Those are structurally different computations, so the gradients are pinned
here as first-class fixtures (not an afterthought), and the FIRST group below is the
overlap-heavy stride-1 case where the accumulation depth is maximal.

Upstream conv/pool grads are NOT hand-written: ConvNode/_PoolBase implement only `forward`
and inherit NodeBase.forward_and_latent_grads / forward_and_weight_grads, which are
jax.value_and_grad over the forward (base.py:507-551). The Julia port mirrors this exactly
via its compute_mu/energy_kernel Enzyme/Zygote seam, so these fixtures compare
autodiff-to-autodiff across two different AD stacks.

RNG TRAP (same as Tier A/B): fixtures carry JAX-GENERATED ARRAYS, never seeds. The Julia
side must build NodeParams DIRECTLY from the loaded arrays and never call its own
initialize_params for compared values (different RNG family entirely).

ISOLATED NODE-LEVEL: every fixture calls the node's forward / forward_and_weight_grads /
forward_and_latent_grads directly with a hand-built NodeParams/NodeState/NodeInfo -- no
graph() construction.

TOLERANCE NOTE: conv compares at rtol 1e-5, NOT bit-identity. Our im2col GEMM and XLA's
fused conv reassociate differently; that is expected and accepted. Pooling is exact-ish
(1e-6) since it is a pure reduce_window with no GEMM reassociation. A divergence LARGER
than these is a structural bug (wrong flip / layout / pad / fold order), which is the
entire point of pinning them.

Provenance: generated against the local upstream checkout; the fixture-venv pin is recorded
in scripts/requirements-fixtures.txt. conv/pool forward math is identical between the pinned
commit 316367c6 and HEAD 5514c91 (the only deltas there are Python default-object plumbing:
`weight_init: Optional[...] = None` -> `weight_init: "InitializerBase"`), so these fixtures
are commit-stable across that range.
"""

import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.nodes.convolutional import ConvNode
from fabricpc.nodes.pooling import MaxPool, AvgPool

OUT = {}
key = jax.random.PRNGKey(20260715)


def split():
    global key
    key, sub = jax.random.split(key)
    return sub


def put(prefix, **arrays):
    for name, arr in arrays.items():
        OUT[f"{prefix}_{name}"] = np.asarray(arr)


def _state(z_latent, shape):
    B = z_latent.shape[0]
    return NodeState(
        z_latent=z_latent,
        z_mu=jnp.zeros((B, *shape)),
        error=jnp.zeros((B, *shape)),
        energy=jnp.zeros((B,)),
        pre_activation=jnp.zeros((B, *shape)),
        latent_grad=jnp.zeros((B, *shape)),
    )


def _conv_info(node, shape, in_edges, act, en):
    """NodeInfo fields ConvNode.forward + NodeBase's grads actually read."""
    return NodeInfo(
        name=node.name,
        shape=shape,
        node_type="conv",
        node_class=ConvNode,
        # `node.node_info` is None until the graph builder finalizes it; we bypass graph()
        # per the isolated-node contract, so read the config the constructor stashed.
        node_config=dict(node._extra_config),
        activation=act,
        energy=en,
        latent_init=None,
        weight_init=None,
        slots={},
        in_degree=len(in_edges),
        out_degree=1,
        in_edges=tuple(in_edges),
        scaling_config=None,
        out_edges=(),
    )


def _dump(prefix, node_class, params, inputs, state, info, edges):
    """forward + BOTH autodiff grad paths for one configuration."""
    _, fwd = node_class.forward(params, inputs, state, info)
    put(prefix + "_fwd", z_mu=fwd.z_mu, pre_activation=fwd.pre_activation,
        error=fwd.error, energy=fwd.energy)
    # ---- BACKWARD (the point of this file) ----
    _, gp = node_class.forward_and_weight_grads(params, inputs, state, info)
    gw = {f"gW_{i}": gp.weights[e] for i, e in enumerate(edges) if e in gp.weights}
    put(prefix + "_gradw", **gw, **({"gb": gp.biases["b"]} if "b" in gp.biases else {}))
    _, in_grads, self_grad = node_class.forward_and_latent_grads(
        params, inputs, state, info, True
    )
    gi = {f"gx_{i}": in_grads[e] for i, e in enumerate(edges)}
    put(prefix + "_gradz", **gi, self_grad=self_grad)


# =====================================================================================
# GROUP 1 (THE ONE THAT MATTERS MOST): overlap-heavy backward.
# stride=1, kernel=3x3, SAME padding => every interior input pixel receives 9 separate
# contributions in the dE/dx scatter-add. Maximal accumulation depth => maximal exposure
# to a fold-order divergence. Kernel is ASYMMETRIC (3x2) so a flip is also detectable.
# =====================================================================================
def gen_overlap_backward():
    B, H, W_, CIN, COUT = 3, 6, 5, 2, 3
    kh, kw = 3, 2                      # asymmetric => flip-sensitive
    x = jax.random.normal(split(), (B, H, W_, CIN))
    Wk = jax.random.normal(split(), (kh, kw, CIN, COUT)) * 0.1
    b = jax.random.normal(split(), (1, 1, 1, COUT)) * 0.1
    shape = (H, W_, COUT)              # SAME padding, stride 1 => spatial preserved
    zlat = jax.random.normal(split(), (B, *shape))
    e = "src->conv:in"
    node = ConvNode(shape=shape, name="conv", kernel_size=(kh, kw), stride=(1, 1),
                    padding="SAME", activation=IdentityActivation(), energy=GaussianEnergy())
    info = _conv_info(node, shape, [e], IdentityActivation(), GaussianEnergy())
    _dump("overlap", ConvNode, NodeParams(weights={e: Wk}, biases={"b": b}),
          {e: x}, _state(zlat, shape), info, [e])
    put("overlap_in", x=x, W=Wk, b=b, z_latent=zlat)
    # Identity activation on purpose: isolates the conv/scatter arithmetic from ReLU's
    # gate, so a backward mismatch here can only be fold order / flip / pad — not masking.


# =====================================================================================
# GROUP 2: multi-edge fold order — THREE in-edges into one ConvNode.
# The forward folds `for edge_key, x in inputs.items()` (insertion order); the port must
# fold in in_edges order in BOTH the forward and the differentiated lanes. Edge names are
# chosen to be hash-order-hostile.
# =====================================================================================
def gen_multi_edge():
    # EVERY extent distinct (H!=W, kh!=kw, CIN!=COUT): a kernel/axis transposition then
    # produces a SHAPE error (loud) instead of plausible-but-wrong numbers. A square kernel
    # with CIN==COUT can hide a k1/k2 or channel transposition entirely.
    B, H, W_, CIN, COUT = 2, 5, 4, 2, 3
    kh, kw = 3, 2
    shape = (H, W_, COUT)
    edges = ["zzz->c:in", "aaa->c:in", "mmm->c:in"]     # insertion != sorted != hash
    xs = {e: jax.random.normal(split(), (B, H, W_, CIN)) for e in edges}
    Ws = {e: jax.random.normal(split(), (kh, kw, CIN, COUT)) * 0.1 for e in edges}
    b = jax.random.normal(split(), (1, 1, 1, COUT)) * 0.1
    zlat = jax.random.normal(split(), (B, *shape))
    node = ConvNode(shape=shape, name="c", kernel_size=(kh, kw), stride=(1, 1),
                    padding="SAME", activation=IdentityActivation(), energy=GaussianEnergy())
    info = _conv_info(node, shape, edges, IdentityActivation(), GaussianEnergy())
    _dump("multi", ConvNode, NodeParams(weights=Ws, biases={"b": b}),
          {e: xs[e] for e in edges}, _state(zlat, shape), info, edges)
    put("multi_in", **{f"x_{i}": xs[e] for i, e in enumerate(edges)},
        **{f"W_{i}": Ws[e] for i, e in enumerate(edges)}, b=b, z_latent=zlat)
    OUT["multi_edge_order"] = np.array("|".join(edges))   # the in_edges order to reproduce


# =====================================================================================
# GROUP 3: ReLU + strided + VALID (the ConvNode defaults path) — pre_activation is
# genuinely distinct from z_mu here (pre can be negative), which pins the B5 seam hook.
# =====================================================================================
def gen_relu_strided_valid():
    # Distinct extents throughout (H!=W, kh!=kw, sh!=sw, CIN!=COUT) — see gen_multi_edge.
    # The previous (3,3,3,4) kernel had kh==kw==CIN==3, which could mask a transposition.
    B, H, W_, CIN, COUT = 2, 7, 6, 2, 4
    kh, kw, sh, sw = 3, 2, 2, 1
    oh, ow = (H - kh) // sh + 1, (W_ - kw) // sw + 1     # VALID
    shape = (oh, ow, COUT)
    x = jax.random.normal(split(), (B, H, W_, CIN))
    Wk = jax.random.normal(split(), (kh, kw, CIN, COUT)) * 0.3
    b = jax.random.normal(split(), (1, 1, 1, COUT)) * 0.1
    zlat = jax.random.normal(split(), (B, *shape))
    e = "s->c:in"
    node = ConvNode(shape=shape, name="c", kernel_size=(kh, kw), stride=(sh, sw),
                    padding="VALID", activation=ReLUActivation(), energy=GaussianEnergy())
    info = _conv_info(node, shape, [e], ReLUActivation(), GaussianEnergy())
    _dump("relu_valid", ConvNode, NodeParams(weights={e: Wk}, biases={"b": b}),
          {e: x}, _state(zlat, shape), info, [e])
    put("relu_valid_in", x=x, W=Wk, b=b, z_latent=zlat)


# =====================================================================================
# GROUP 4: rank dispatch — 1D (NLC) and 3D (NDHWC), SAME + stride>1.
# =====================================================================================
def gen_rank_1d_3d():
    # ---- 1D ----
    B, L, CIN, COUT = 2, 9, 2, 3
    shape1 = (int(np.ceil(L / 2)), COUT)                 # SAME, stride 2 => ceil(L/s)
    x1 = jax.random.normal(split(), (B, L, CIN))
    W1 = jax.random.normal(split(), (3, CIN, COUT)) * 0.1
    b1 = jax.random.normal(split(), (1, 1, COUT)) * 0.1
    z1 = jax.random.normal(split(), (B, *shape1))
    e1 = "s->c1:in"
    n1 = ConvNode(shape=shape1, name="c1", kernel_size=(3,), stride=(2,), padding="SAME",
                  activation=IdentityActivation(), energy=GaussianEnergy())
    _dump("conv1d", ConvNode, NodeParams(weights={e1: W1}, biases={"b": b1}), {e1: x1},
          _state(z1, shape1), _conv_info(n1, shape1, [e1], IdentityActivation(), GaussianEnergy()), [e1])
    put("conv1d_in", x=x1, W=W1, b=b1, z_latent=z1)
    # ---- 3D ----
    # The previous fixture was a (2,2,2,2,2) kernel over D=H=4: EVERY extent equal, so any
    # axis confusion (D/H/W or k/C_in/C_out) stayed shape-compatible and invisible. Now every
    # extent is distinct — D!=H!=W, kd!=kh!=kw, CIN!=COUT — so a transposition is a shape error.
    B, D, H, W_, CIN, COUT = 2, 4, 3, 5, 2, 3
    kd, kh, kw = 2, 3, 1
    shape3 = (D, H, W_, COUT)                            # SAME, stride 1 => spatial preserved
    x3 = jax.random.normal(split(), (B, D, H, W_, CIN))
    W3 = jax.random.normal(split(), (kd, kh, kw, CIN, COUT)) * 0.1
    b3 = jax.random.normal(split(), (1, 1, 1, 1, COUT)) * 0.1
    z3 = jax.random.normal(split(), (B, *shape3))
    e3 = "s->c3:in"
    n3 = ConvNode(shape=shape3, name="c3", kernel_size=(kd, kh, kw), stride=(1, 1, 1),
                  padding="SAME", activation=IdentityActivation(), energy=GaussianEnergy())
    _dump("conv3d", ConvNode, NodeParams(weights={e3: W3}, biases={"b": b3}), {e3: x3},
          _state(z3, shape3), _conv_info(n3, shape3, [e3], IdentityActivation(), GaussianEnergy()), [e3])
    put("conv3d_in", x=x3, W=W3, b=b3, z_latent=z3)


# =====================================================================================
# GROUP 5: pooling. MaxPool tie routing (all-equal window => which cell gets the grad?)
# is a FIXTURE question, not something to reason about. AvgPool pins count_include_pad
# on BOTH settings + the global mode.
# =====================================================================================
def _pool_info(node, shape, edges, cls, act, en):
    return NodeInfo(
        name=node.name, shape=shape, node_type="pool", node_class=cls,
        # `node.node_info` is None until the graph builder finalizes it; we bypass graph()
        # per the isolated-node contract, so read the config the constructor stashed.
        node_config=dict(node._extra_config), activation=act, energy=en,
        latent_init=None, weight_init=None, slots={}, in_degree=len(edges),
        out_degree=1, in_edges=tuple(edges), out_edges=(), scaling_config=None,
    )


def gen_pooling():
    B, H, W_, C = 2, 4, 4, 2
    e = "s->p:in"
    shape = (2, 2, C)
    x = jax.random.normal(split(), (B, H, W_, C))
    zlat = jax.random.normal(split(), (B, *shape))
    for cls, name, kw in (
        (MaxPool, "maxpool", dict(window_shape=(2, 2), stride=(2, 2), padding="VALID")),
        (AvgPool, "avgpool", dict(window_shape=(2, 2), stride=(2, 2), padding="VALID")),
    ):
        node = cls(shape=shape, name="p", activation=IdentityActivation(),
                   energy=GaussianEnergy(), **kw)
        _dump(name, cls, NodeParams(weights={}, biases={}), {e: x},
              _state(zlat, shape), _pool_info(node, shape, [e], cls, IdentityActivation(), GaussianEnergy()), [e])
    put("pool_in", x=x, z_latent=zlat)

    # --- MaxPool TIE ROUTING (all-equal window). Not exotic in a ReLU PC net: dead units
    # give all-zero windows. Which cell receives the gradient is XLA's call; the fixture
    # adjudicates, never our reasoning.
    xt = jnp.zeros((B, H, W_, C))
    node = MaxPool(shape=shape, name="p", window_shape=(2, 2), stride=(2, 2),
                   padding="VALID", activation=IdentityActivation(), energy=GaussianEnergy())
    _dump("maxpool_tie", MaxPool, NodeParams(weights={}, biases={}), {e: xt},
          _state(zlat, shape), _pool_info(node, shape, [e], MaxPool, IdentityActivation(), GaussianEnergy()), [e])
    put("maxpool_tie_in", x=xt, z_latent=zlat)

    # --- MaxPool PARTIAL-TIE: the DISCRIMINATING case. The all-tied window above cannot
    # tell row-major from column-major tie-breaking — both orders visit (0,0) first, so a
    # wrong implementation passes it. Here (0,1) and (1,0) tie for the max while (0,0) is
    # strictly smaller, so the two orders MUST disagree:
    #     row-major / NHWC (what JAX does, measured) -> (0,1)
    #     column-major (what an im2col patch vector unrolls to, k1 fastest) -> (1,0)
    # Our patch layout is column-major, so MaxPool's reduction must iterate each window in
    # ROW-MAJOR order to match. Without this fixture that divergence is invisible.
    win = np.zeros((B, H, W_, C), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            for oh in range(0, H, 2):
                for ow in range(0, W_, 2):
                    win[b, oh, ow, c] = 0.0            # (0,0): strictly smaller
                    win[b, oh, ow + 1, c] = 1.0        # (0,1): tied max
                    win[b, oh + 1, ow, c] = 1.0        # (1,0): tied max
                    win[b, oh + 1, ow + 1, c] = 0.5
    xp = jnp.asarray(win)
    _dump("maxpool_ptie", MaxPool, NodeParams(weights={}, biases={}), {e: xp},
          _state(zlat, shape), _pool_info(node, shape, [e], MaxPool, IdentityActivation(), GaussianEnergy()), [e])
    put("maxpool_ptie_in", x=xp, z_latent=zlat)

    # --- AvgPool count_include_pad: SAME padding over an ODD extent so the last window
    # is genuinely half-padded => True and False MUST differ (if they agree the test is
    # vacuous and padding isn't being counted at all).
    Ho, Wo = 3, 3
    xo = jax.random.normal(split(), (B, Ho, Wo, C))
    shp = (2, 2, C)
    zo = jax.random.normal(split(), (B, *shp))
    for flag in (True, False):
        node = AvgPool(shape=shp, name="p", window_shape=(2, 2), stride=(2, 2),
                       padding="SAME", count_include_pad=flag,
                       activation=IdentityActivation(), energy=GaussianEnergy())
        _dump(f"avgpool_cip_{int(flag)}", AvgPool, NodeParams(weights={}, biases={}),
              {e: xo}, _state(zo, shp),
              _pool_info(node, shp, [e], AvgPool, IdentityActivation(), GaussianEnergy()), [e])
    put("avgpool_cip_in", x=xo, z_latent=zo)

    # --- AvgPool GLOBAL: (B, spatial..., C) -> (B, C); rank-1 shape (C,).
    shg = (C,)
    zg = jax.random.normal(split(), (B, C))
    node = AvgPool(shape=shg, name="p", global_pool=True,
                   activation=IdentityActivation(), energy=GaussianEnergy())
    _dump("avgpool_global", AvgPool, NodeParams(weights={}, biases={}), {e: x},
          _state(zg, shg), _pool_info(node, shg, [e], AvgPool, IdentityActivation(), GaussianEnergy()), [e])
    put("avgpool_global_in", x=x, z_latent=zg)


for g in (gen_overlap_backward, gen_multi_edge, gen_relu_strided_valid,
          gen_rank_1d_3d, gen_pooling):
    g()
    print(f"  {g.__name__}: ok")

np.savez("test/conformance/fixtures/tier_conv.npz", **OUT)
print(f"wrote {len(OUT)} arrays to test/conformance/fixtures/tier_conv.npz")
