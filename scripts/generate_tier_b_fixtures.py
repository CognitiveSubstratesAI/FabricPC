#!/usr/bin/env python3
"""
Tier B conformance fixture generator (docs/AUDIT_REGISTER.md section 6): node-local --
per-node forward + BOTH gradient paths (closed-form and/or the Phase-D autodiff seam,
whichever the Julia node uses), isolated (no full graph construction), rtol/atol 1e-5.

Drafted by a 6-way parallel research workflow (one agent per node group, each reading
the upstream Python node class AND the Julia node implementation directly, then
self-testing its own fixture code against this venv before returning it) and assembled
by hand afterward -- see git log for the workflow's node-group split.

RNG TRAP (same as Tier A): fixtures carry JAX-GENERATED ARRAYS, never seeds. Every node's
params are built via jax.random (or the node's own initialize_params, itself fed by
jax.random through split()) and dumped as arrays; the Julia side must construct NodeParams
DIRECTLY from those loaded arrays, never call its own initialize_params for the compared
values (different RNG family entirely).

ISOLATED NODE-LEVEL TESTING: every fixture calls a node's forward/forward_and_weight_grads/
forward_and_latent_grads directly with a hand-built NodeParams/NodeState/NodeInfo -- no
graph() construction, mirroring the existing test_transformer.jl F-01 regression test's
own pattern (see that testset for the Julia-side analogue).
"""

import numpy as np
import jax

OUT = {}
key = jax.random.PRNGKey(42)


def split():
    global key
    key, sub = jax.random.split(key)
    return sub


def put(prefix, **arrays):
    for name, arr in arrays.items():
        OUT[f"{prefix}_{name}"] = np.asarray(arr)



# ==============================================================================
# Node group: linear_family
# ==============================================================================
def generate_linear_family_fixtures(put, split):
    """Tier B fixtures for the Linear-family nodes (both CLOSED-FORM gradients on the
    Julia side).

    Linear  -> fixtured against upstream LinearExplicitGrad (linear_explicit_grad.py),
               whose forward_and_latent_grads/forward_and_weight_grads ARE the explicit
               closed forms Julia's fused Linear node ports (verified by reading both:
               gain_mod_error = error * f'(pre); dW = -inᵀ·gain_mod_error;
               db = -Σ_batch gain_mod_error; d_input = -gain_mod_error·Wᵀ — signs match
               Julia's dpre = mu_grad·f' = -error·f' formulation exactly since
               mu_grad = grad_mu(Gaussian, z, z_mu) = -(z - z_mu) = -error at precision 1).

    LinearResidual -> upstream's LinearResidual (linear_residual.py) subclasses NodeBase
               directly (NOT LinearExplicitGrad) and defines ONLY forward(); it has no
               explicit-grad override at all, so forward_and_latent_grads/
               forward_and_weight_grads fall through to NodeBase's jax.value_and_grad
               autodiff defaults (base.py). Julia's LinearResidual node, by contrast, IS
               fused-closed-form (like Linear). So this fixture's ground truth for
               LinearResidual's gradients is autodiff of the SAME forward() math, exactly
               the same rationale Tier A used for grad_mu (jax.grad checks a hand-derived
               Julia closed form where upstream itself has none) -- not a transcription of
               an upstream closed form, since there isn't one to transcribe for this node.

    RNG TRAP: weight/bias/input arrays are generated here via jax.random (through
    `split()`) and dumped as arrays; Julia must build NodeParams directly from the loaded
    fixture arrays, never call its own initialize_params for these.
    """
    import jax
    import jax.numpy as jnp
    from fabricpc.core.types import NodeParams, NodeState, NodeInfo
    from fabricpc.core.activations import IdentityActivation
    from fabricpc.core.energy import GaussianEnergy
    from fabricpc.nodes.linear_explicit_grad import LinearExplicitGrad
    from fabricpc.nodes.linear_residual import LinearResidual

    act = IdentityActivation()
    en = GaussianEnergy()

    # -----------------------------------------------------------------------------------
    # Linear (LinearExplicitGrad): batch=3, in=4, out=5. Single "in" slot, one edge.
    # -----------------------------------------------------------------------------------
    B, IN, OUT = 3, 4, 5
    edge_in = "x->linear:in"

    x = jax.random.normal(split(), (B, IN))
    W = jax.random.normal(split(), (IN, OUT)) * 0.1
    b = jax.random.normal(split(), (1, OUT)) * 0.1
    zlat = jax.random.normal(split(), (B, OUT))

    params = NodeParams(weights={edge_in: W}, biases={"b": b})
    inputs = {edge_in: x}
    state = NodeState(
        z_latent=zlat,
        z_mu=jnp.zeros((B, OUT)),
        error=jnp.zeros((B, OUT)),
        energy=jnp.zeros((B,)),
        latent_grad=jnp.zeros((B, OUT)),
    )
    # NodeInfo fields actually READ by Linear/LinearExplicitGrad's forward/grad methods
    # (grep-verified): node_class, node_config, activation, energy, shape, in_degree,
    # out_degree. name/slots/in_edges/out_edges/latent_init/weight_init are unread here
    # (only used by the graph builder / initialize_params, which we bypass per the RNG
    # trap) -- filled with reasonable dummies.
    info = NodeInfo(
        name="linear",
        shape=(OUT,),
        node_type="linear",
        node_class=LinearExplicitGrad,
        node_config={"flatten_input": False, "use_bias": True},
        activation=act,
        energy=en,
        latent_init=None,
        weight_init=None,
        slots={},
        in_degree=1,
        out_degree=1,
        in_edges=(edge_in,),
        out_edges=(),
        scaling_config=None,
    )

    fwd_state = LinearExplicitGrad.forward(params, inputs, state, info)
    put(
        "linear_fwd",
        x=x,
        W=W,
        b=b,
        z_latent=zlat,
        z_mu=fwd_state.z_mu,
        error=fwd_state.error,
        energy=fwd_state.energy,
    )

    _, gp = LinearExplicitGrad.forward_and_weight_grads(params, inputs, state, info)
    put("linear_gradw", W=gp.weights[edge_in], b=gp.biases["b"])

    # is_clamped=True is irrelevant to which branch fires here (in_degree=1, out_degree=1
    # ⇒ always the "internal or clamped output node" branch regardless of is_clamped) but
    # is passed explicitly since the signature requires it.
    _, in_grads, self_grad = LinearExplicitGrad.forward_and_latent_grads(
        params, inputs, state, info, True
    )
    put("linear_gradz", x=in_grads[edge_in], self_grad=self_grad)

    # -----------------------------------------------------------------------------------
    # LinearResidual: batch=3, in=4, out=5. "in" slot (weighted) + "skip" slot (identity,
    # shape == node output shape (B, OUT), no weight matrix).
    # -----------------------------------------------------------------------------------
    edge_in_r = "a->linresid:in"
    edge_skip_r = "b->linresid:skip"

    xr = jax.random.normal(split(), (B, IN))
    Wr = jax.random.normal(split(), (IN, OUT)) * 0.1
    br = jax.random.normal(split(), (1, OUT)) * 0.1
    xskip = jax.random.normal(split(), (B, OUT))
    zlat_r = jax.random.normal(split(), (B, OUT))

    params_r = NodeParams(weights={edge_in_r: Wr}, biases={"b": br})
    inputs_r = {edge_in_r: xr, edge_skip_r: xskip}
    state_r = NodeState(
        z_latent=zlat_r,
        z_mu=jnp.zeros((B, OUT)),
        error=jnp.zeros((B, OUT)),
        energy=jnp.zeros((B,)),
        latent_grad=jnp.zeros((B, OUT)),
    )
    info_r = NodeInfo(
        name="linresid",
        shape=(OUT,),
        node_type="linear_residual",
        node_class=LinearResidual,
        node_config={"flatten_input": False, "use_bias": True},
        activation=act,
        energy=en,
        latent_init=None,
        weight_init=None,
        slots={},
        in_degree=2,
        out_degree=1,
        in_edges=(edge_in_r, edge_skip_r),
        out_edges=(),
        scaling_config=None,
    )

    fwd_state_r = LinearResidual.forward(params_r, inputs_r, state_r, info_r)
    put(
        "linresid_fwd",
        x=xr,
        W=Wr,
        b=br,
        skip=xskip,
        z_latent=zlat_r,
        z_mu=fwd_state_r.z_mu,
        error=fwd_state_r.error,
        energy=fwd_state_r.energy,
    )

    _, gp_r = LinearResidual.forward_and_weight_grads(params_r, inputs_r, state_r, info_r)
    put("linresid_gradw", W=gp_r.weights[edge_in_r], b=gp_r.biases["b"])

    _, in_grads_r, self_grad_r = LinearResidual.forward_and_latent_grads(
        params_r, inputs_r, state_r, info_r, True
    )
    put(
        "linresid_gradz",
        x=in_grads_r[edge_in_r],
        skip=in_grads_r[edge_skip_r],
        self_grad=self_grad_r,
    )


# ==============================================================================
# Node group: identity_skip
# ==============================================================================
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeParams, NodeState, NodeInfo, SlotInfo
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.activations import IdentityActivation
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.skip_connection import SkipConnection


def generate_identity_skip_fixtures(put, split):
    """Tier B fixtures for IdentityNode + SkipConnection (closed-form gradients).

    Both nodes: multi-input sum node, no learnable params. Isolated node-level
    calls (no graph() construction) -- forward, forward_and_latent_grads (the
    general "else" branch: in_degree>0 and out_degree>0, so upstream's autodiff
    generic path exercises the same branch as Julia's explicit closed form),
    and forward_and_weight_grads (trivially empty weights/biases on both sides,
    included for structural parity only).
    """
    energy = GaussianEnergy()  # precision=1.0 default, matches Julia's default
    activation = IdentityActivation()

    # =====================================================================
    # IdentityNode: z_mu = scale * sum(inputs), 2 input edges, non-trivial scale.
    # =====================================================================
    B, D = 3, 5
    scale = 1.7

    x1 = jax.random.normal(split(), (B, D))
    x2 = jax.random.normal(split(), (B, D))
    z_latent = jax.random.normal(split(), (B, D))

    inputs = {"e1": x1, "e2": x2}
    params = NodeParams(weights={}, biases={})
    state = NodeState(
        z_latent=z_latent,
        z_mu=jnp.zeros((B, D)),
        error=jnp.zeros((B, D)),
        energy=jnp.zeros((B,)),
        latent_grad=jnp.zeros((B, D)),
    )
    slots = {
        "in": SlotInfo(
            name="in",
            parent_node="id",
            is_multi_input=True,
            is_variance_scalable=True,
            is_skip_connection=False,
            in_neighbors=("src1", "src2"),
        )
    }
    node_info = NodeInfo(
        name="id",
        shape=(D,),
        node_type="identity",
        node_class=IdentityNode,
        node_config={"scale": scale},
        activation=activation,
        energy=energy,
        latent_init=None,
        weight_init=None,
        slots=slots,
        in_degree=2,
        out_degree=1,
        in_edges=("e1", "e2"),
        out_edges=("id->out:in",),
        scaling_config=None,
    )

    # NOTE: every scalar (total_energy, scale) is reshaped to a 1-element 1-D
    # array before put() -- a bare 0-d jax array round-trips through
    # np.savez/NPZ.jl ambiguously, a shape-(1,) array does not.
    put(
        "identity_skip_identity_fwd",
        x1=x1,
        x2=x2,
        z_latent=z_latent,
        scale=jnp.reshape(jnp.asarray(scale, dtype=jnp.float32), (1,)),
    )
    new_state = IdentityNode.forward(params, inputs, state, node_info)
    total_energy = jnp.sum(new_state.energy)
    put(
        "identity_skip_identity_fwd",
        energy=jnp.reshape(total_energy, (1,)),
        z_mu=new_state.z_mu,
        error=new_state.error,
    )

    ns_g, input_grads, self_grad = IdentityNode.forward_and_latent_grads(
        params, inputs, state, node_info, is_clamped=False
    )
    put(
        "identity_skip_identity_gradz",
        e1_grad=input_grads["e1"],
        e2_grad=input_grads["e2"],
        self_grad=self_grad,
        energy=ns_g.energy,
        z_mu=ns_g.z_mu,
    )

    ns_w, params_grad = IdentityNode.forward_and_weight_grads(
        params, inputs, state, node_info
    )
    # weights/biases are ALWAYS {} for both nodes (no params exist to
    # differentiate w.r.t.), on both upstream and Julia -- no fixture data
    # needed for that; only the returned state's per-sample energy is
    # meaningful to compare.
    put("identity_skip_identity_gradw", energy=ns_w.energy)

    # =====================================================================
    # SkipConnection: z_mu = sum(inputs) (no scale), 2 input edges.
    # =====================================================================
    Bs, Ds = 3, 4

    y1 = jax.random.normal(split(), (Bs, Ds))
    y2 = jax.random.normal(split(), (Bs, Ds))
    z_latent_s = jax.random.normal(split(), (Bs, Ds))

    inputs_s = {"f1": y1, "f2": y2}
    params_s = NodeParams(weights={}, biases={})
    state_s = NodeState(
        z_latent=z_latent_s,
        z_mu=jnp.zeros((Bs, Ds)),
        error=jnp.zeros((Bs, Ds)),
        energy=jnp.zeros((Bs,)),
        latent_grad=jnp.zeros((Bs, Ds)),
    )
    slots_s = {
        "in": SlotInfo(
            name="in",
            parent_node="skip",
            is_multi_input=True,
            is_variance_scalable=False,
            is_skip_connection=True,
            in_neighbors=("src1", "src2"),
        )
    }
    node_info_s = NodeInfo(
        name="skip",
        shape=(Ds,),
        node_type="skip_connection",
        node_class=SkipConnection,
        node_config={},
        activation=activation,
        energy=energy,
        latent_init=None,
        weight_init=None,
        slots=slots_s,
        in_degree=2,
        out_degree=1,
        in_edges=("f1", "f2"),
        out_edges=("skip->out:in",),
        scaling_config=None,
    )

    put("identity_skip_skip_fwd", x1=y1, x2=y2, z_latent=z_latent_s)
    new_state_s = SkipConnection.forward(
        params_s, inputs_s, state_s, node_info_s
    )
    total_energy_s = jnp.sum(new_state_s.energy)
    put(
        "identity_skip_skip_fwd",
        energy=jnp.reshape(total_energy_s, (1,)),
        z_mu=new_state_s.z_mu,
        error=new_state_s.error,
    )

    ns_g_s, input_grads_s, self_grad_s = SkipConnection.forward_and_latent_grads(
        params_s, inputs_s, state_s, node_info_s, is_clamped=False
    )
    put(
        "identity_skip_skip_gradz",
        f1_grad=input_grads_s["f1"],
        f2_grad=input_grads_s["f2"],
        self_grad=self_grad_s,
        energy=ns_g_s.energy,
        z_mu=ns_g_s.z_mu,
    )

    ns_w_s, params_grad_s = SkipConnection.forward_and_weight_grads(
        params_s, inputs_s, state_s, node_info_s
    )
    put("identity_skip_skip_gradw", energy=ns_w_s.energy)


# ==============================================================================
# Node group: transformer_block
# ==============================================================================
import jax
import jax.numpy as jnp

from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import GeluActivation, IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.mupc import MuPCScalingFactors


def generate_transformer_block_fixtures(put, split):
    """Tier B fixtures for TransformerBlock: forward + BOTH gradient paths.

    HYBRID node: forward / forward_and_latent_grads use the generic Phase-D
    autodiff seam (no override, upstream or Julia); forward_and_weight_grads has
    an EXPLICIT muPC LayerNorm-compensation override on both sides (F-01).

    Cases dumped:
      tb_shared_*     - one shared NodeParams (weights/biases) + input x +
                        z_latent, reused by the muPC-off/on weight-grad cases so
                        the ONLY difference between them is the scaling_config.
      tb_muPCoff_*    - forward_and_weight_grads with scaling_config=None
                        (negative control: unscaled weight grads).
      tb_muPCon_*     - forward_and_weight_grads with a real MuPCScalingFactors,
                        forward_scale[edge]=a=2.5 on the "in" edge. Weight grads
                        must equal a * muPCoff's; bias grads must be UNCHANGED
                        -- confirmed directly from transformer.py:392-422, which
                        only rescales params_grad.weights, never .biases.
      tb_latentgrad_* - one plain forward_and_latent_grads call (generic seam,
                        no override; in_degree=1, out_degree=1 hits the
                        internal-node autodiff branch of
                        NodeBase.forward_and_latent_grads).
      tb_t2_*         - causal vs full forward on the SAME params; base vs a
                        future-position-perturbed input. Causal position 0's
                        output is EXACTLY invariant to the perturbation (the
                        causal mask excludes the future key entirely from the
                        softmax -- confirmed empirically: diff == 0.0, not just
                        small); full (non-causal) position 0 visibly changes
                        (diff ~1.25). Both causal/full outputs on both
                        (base, pert) inputs are dumped for byte-level
                        conformance on top of the property check.
    """
    B, S, E, H = 2, 4, 8, 2
    ff_dim = 4 * E
    node_shape = (S, E)
    edge_key = "x->tb:in"
    mask_key = "m->tb:mask"

    node_config = {
        "num_heads": H,
        "use_rope": True,
        "rope_theta": 10000.0,
        "internal_activation": GeluActivation(),
        "ff_dim": ff_dim,
    }

    def make_node_info(scaling_config, in_degree=1, out_degree=1):
        return NodeInfo(
            name="tb",
            shape=node_shape,
            node_type="transformer",
            node_class=TransformerBlock,
            node_config=node_config,
            activation=IdentityActivation(),
            energy=GaussianEnergy(precision=1.0),
            latent_init=None,
            weight_init=None,
            slots={},
            in_degree=in_degree,
            out_degree=out_degree,
            in_edges=(edge_key,),
            out_edges=(),
            scaling_config=scaling_config,
        )

    # --- shared params: real init-scale weights (upstream's OWN initializer,
    # still JAX-generated -- no seed crosses to Julia, see module RNG-trap note)
    # + randomized (non-zero) biases/LayerNorm affine params so the fixture
    # isn't degenerate (initialize_params zeros all biases/beta by default). ---
    params0 = TransformerBlock.initialize_params(
        split(), node_shape, {edge_key: (S, E)}, weight_init=None, config=node_config
    )
    weights = dict(params0.weights)
    biases = dict(params0.biases)
    weights["ln1_gamma"] = 1.0 + 0.05 * jax.random.normal(split(), (1, 1, E))
    weights["ln2_gamma"] = 1.0 + 0.05 * jax.random.normal(split(), (1, 1, E))
    biases["ln1_beta"] = 0.1 * jax.random.normal(split(), (1, 1, E))
    biases["ln2_beta"] = 0.1 * jax.random.normal(split(), (1, 1, E))
    for bk, dim in [
        ("b_q", E), ("b_k", E), ("b_v", E), ("b_o", E),
        ("b_ff1", ff_dim), ("b_ff2", E),
    ]:
        biases[bk] = 0.05 * jax.random.normal(split(), (1, 1, dim))
    params = NodeParams(weights=weights, biases=biases)

    x = jax.random.normal(split(), (B, S, E))
    zlat = jax.random.normal(split(), (B, S, E))

    put("tb_shared_w", **weights)
    put("tb_shared_b", **biases)
    put("tb_shared", x=x, zlatent=zlat)

    state = NodeState(
        z_latent=zlat,
        z_mu=jnp.zeros((B, S, E)),
        error=jnp.zeros((B, S, E)),
        energy=jnp.zeros((B,)),
        latent_grad=jnp.zeros((B, S, E)),
    )
    inputs = {edge_key: x}

    # --- Case 1: muPC OFF (scaling_config=None) -- negative control. ---------
    info_off = make_node_info(None)
    new_state_off, gradw_off = TransformerBlock.forward_and_weight_grads(
        params, inputs, state, info_off
    )
    put("tb_muPCoff_gradw", **gradw_off.weights, **gradw_off.biases)
    put("tb_muPCoff", zmu=new_state_off.z_mu)

    # --- Case 2: muPC ON, a=2.5 on the "in" edge's forward_scale. -------------
    a = 2.5
    sc = MuPCScalingFactors(
        forward_scale={edge_key: a},
        self_grad_scale=1.0,
        topdown_grad_scale={edge_key: a * 1.0},
        weight_grad_scale={edge_key: 1.0},
    )
    info_on = make_node_info(sc)
    new_state_on, gradw_on = TransformerBlock.forward_and_weight_grads(
        params, inputs, state, info_on
    )
    put("tb_muPCon_gradw", **gradw_on.weights, **gradw_on.biases)
    put("tb_muPCon", zmu=new_state_on.z_mu, a=jnp.asarray(a, dtype=jnp.float32))

    # --- Case 3: plain forward_and_latent_grads (generic seam, no override). -
    info_generic = make_node_info(None, in_degree=1, out_degree=1)
    is_clamped = False
    new_state_lg, input_grads, self_grad = TransformerBlock.forward_and_latent_grads(
        params, inputs, state, info_generic, is_clamped
    )
    put(
        "tb_latentgrad",
        dinput=input_grads[edge_key],
        dself=self_grad,
        zmu=new_state_lg.z_mu,
    )

    # --- T2: causal vs full forward, base vs future-perturbed input. ---------
    x_base = jax.random.normal(split(), (B, S, E))
    # Perturb the LAST (future) position with a fresh random VECTOR (NOT a
    # uniform per-token scalar shift -- LayerNorm subtracts the per-token mean,
    # so a constant shift across all E dims of one token is invisible post-LN
    # and would make this property check vacuous for BOTH causal and full).
    pert_noise = 5.0 * jax.random.normal(split(), (B, E))
    x_pert = x_base.at[:, S - 1, :].add(pert_noise)
    causal_mask = jnp.tril(jnp.ones((S, S)))[None, None, :, :]  # (1,1,S,S)

    info_fwd = make_node_info(None)
    st_causal_base = TransformerBlock.forward(
        params, {edge_key: x_base, mask_key: causal_mask}, state, info_fwd
    )
    st_causal_pert = TransformerBlock.forward(
        params, {edge_key: x_pert, mask_key: causal_mask}, state, info_fwd
    )
    st_full_base = TransformerBlock.forward(
        params, {edge_key: x_base}, state, info_fwd
    )
    st_full_pert = TransformerBlock.forward(
        params, {edge_key: x_pert}, state, info_fwd
    )

    put(
        "tb_t2",
        x_base=x_base,
        x_pert=x_pert,
        causal_out_base=st_causal_base.z_mu,
        causal_out_pert=st_causal_pert.z_mu,
        full_out_base=st_full_base.z_mu,
        full_out_pert=st_full_pert.z_mu,
    )


# ==============================================================================
# Node group: decomposed_mha_mlp
# ==============================================================================
def generate_decomposed_mha_mlp_fixtures(put, split):
    """
    Node-level Tier B fixtures for MhaResidualNode, LnMlp1Node, Mlp2ResidualNode
    (fabricpc/nodes/transformer_v2.py) -- all three are SEAM-based on the Julia side
    (compute_mu only; forward_and_latent_grads/forward_and_weight_grads come from the
    generic Phase-D autodiff seam). Ground truth here is upstream's own
    NodeBase.forward / .forward_and_weight_grads / .forward_and_latent_grads
    (jax.value_and_grad), called on hand-built NodeInfo/NodeState/NodeParams -- no
    graph() construction, matching the isolated node-level testing pattern (F-01 /
    test_transformer_decomposed.jl's Mlp2Residual testset on the Julia side).

    RNG TRAP: all weight/bias/input arrays below are JAX-GENERATED via `split()`
    (jax.random) -- never a seed handed to Julia. The Julia side must build
    NodeParams DIRECTLY from these loaded arrays, not call its own initialize_params.
    """
    from fabricpc.core.types import NodeParams, NodeState, NodeInfo
    from fabricpc.core.activations import IdentityActivation, GeluActivation
    from fabricpc.core.energy import GaussianEnergy
    from fabricpc.nodes.transformer_v2 import (
        MhaResidualNode,
        LnMlp1Node,
        Mlp2ResidualNode,
    )
    import jax
    import jax.numpy as jnp

    B, S, E, H, ff = 2, 3, 8, 2, 16

    def zeros_state(z_latent):
        shp = z_latent.shape
        return NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros(shp),
            error=jnp.zeros(shp),
            energy=jnp.zeros((shp[0],)),
            latent_grad=jnp.zeros(shp),
        )

    # =====================================================================
    # MhaResidualNode: z_mu = skip + W_o . MHA(LayerNorm(x)); slots "in"/"skip".
    # in_degree=2/out_degree=1 in NodeInfo forces the generic seam's "internal
    # node" branch (real autodiff), matching what the Julia snippet exercises.
    # =====================================================================
    ln_gamma = 1.0 + 0.1 * jax.random.normal(split(), (E,))
    ln_beta = 0.1 * jax.random.normal(split(), (E,))
    W_q = 0.1 * jax.random.normal(split(), (E, E))
    W_k = 0.1 * jax.random.normal(split(), (E, E))
    W_v = 0.1 * jax.random.normal(split(), (E, E))
    W_o = 0.1 * jax.random.normal(split(), (E, E))
    b_q = 0.05 * jax.random.normal(split(), (E,))
    b_k = 0.05 * jax.random.normal(split(), (E,))
    b_v = 0.05 * jax.random.normal(split(), (E,))
    b_o = 0.05 * jax.random.normal(split(), (E,))

    mha_params = NodeParams(
        weights={"ln_gamma": ln_gamma, "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o},
        biases={"ln_beta": ln_beta, "b_q": b_q, "b_k": b_k, "b_v": b_v, "b_o": b_o},
    )
    mha_x = jax.random.normal(split(), (B, S, E))
    mha_skip = jax.random.normal(split(), (B, S, E))
    mha_inputs = {"x->mha:in": mha_x, "skip->mha:skip": mha_skip}
    mha_zlat = jax.random.normal(split(), (B, S, E))
    mha_state = zeros_state(mha_zlat)
    mha_info = NodeInfo(
        name="mha",
        shape=(S, E),
        node_type="MhaResidualNode",
        node_class=MhaResidualNode,
        node_config={
            "num_heads": H,
            "use_rope": True,
            "is_causal": True,
            "rope_theta": 10000.0,
        },
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=None,
        weight_init=None,
        slots={},
        in_degree=2,
        out_degree=1,
        in_edges=("x->mha:in", "skip->mha:skip"),
        out_edges=(),
        scaling_config=None,
    )

    mha_fwd_state = MhaResidualNode.forward(mha_params, mha_inputs, mha_state, mha_info)
    put(
        "dmm_mha_fwd",
        ln_gamma=ln_gamma, ln_beta=ln_beta, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o,
        b_q=b_q, b_k=b_k, b_v=b_v, b_o=b_o,
        x=mha_x, skip=mha_skip, out=mha_fwd_state.z_mu,
    )

    _, mha_gradw = MhaResidualNode.forward_and_weight_grads(
        mha_params, mha_inputs, mha_state, mha_info
    )
    put(
        "dmm_mha_gradw",
        z_latent=mha_zlat,
        gW_q=mha_gradw.weights["W_q"], gW_k=mha_gradw.weights["W_k"],
        gW_v=mha_gradw.weights["W_v"], gW_o=mha_gradw.weights["W_o"],
        gln_gamma=mha_gradw.weights["ln_gamma"],
        gb_q=mha_gradw.biases["b_q"], gb_k=mha_gradw.biases["b_k"],
        gb_v=mha_gradw.biases["b_v"], gb_o=mha_gradw.biases["b_o"],
        gln_beta=mha_gradw.biases["ln_beta"],
    )

    _, mha_igrads, mha_sg = MhaResidualNode.forward_and_latent_grads(
        mha_params, mha_inputs, mha_state, mha_info, False
    )
    put(
        "dmm_mha_gradlatent",
        z_latent=mha_zlat,
        grad_in=mha_igrads["x->mha:in"], grad_skip=mha_igrads["skip->mha:skip"],
        self_grad=mha_sg,
    )

    # =====================================================================
    # LnMlp1Node: z_mu = GELU(LayerNorm(x) . W_ff1 + b_ff1), shape (S, ff)
    # =====================================================================
    ln1_gamma = 1.0 + 0.1 * jax.random.normal(split(), (E,))
    ln1_beta = 0.1 * jax.random.normal(split(), (E,))
    W_ff1 = 0.1 * jax.random.normal(split(), (E, ff))
    b_ff1 = 0.05 * jax.random.normal(split(), (ff,))

    lnmlp1_params = NodeParams(
        weights={"ln_gamma": ln1_gamma, "W_ff1": W_ff1},
        biases={"ln_beta": ln1_beta, "b_ff1": b_ff1},
    )
    lnmlp1_x = jax.random.normal(split(), (B, S, E))
    lnmlp1_inputs = {"mha->mlp1:in": lnmlp1_x}
    lnmlp1_zlat = jax.random.normal(split(), (B, S, ff))
    lnmlp1_state = zeros_state(lnmlp1_zlat)
    lnmlp1_info = NodeInfo(
        name="mlp1",
        shape=(S, ff),
        node_type="LnMlp1Node",
        node_class=LnMlp1Node,
        node_config={},
        activation=GeluActivation(),
        energy=GaussianEnergy(),
        latent_init=None,
        weight_init=None,
        slots={},
        in_degree=1,
        out_degree=1,
        in_edges=("mha->mlp1:in",),
        out_edges=(),
        scaling_config=None,
    )

    lnmlp1_fwd_state = LnMlp1Node.forward(
        lnmlp1_params, lnmlp1_inputs, lnmlp1_state, lnmlp1_info
    )
    put(
        "dmm_lnmlp1_fwd",
        ln_gamma=ln1_gamma, ln_beta=ln1_beta, W_ff1=W_ff1, b_ff1=b_ff1,
        x=lnmlp1_x, out=lnmlp1_fwd_state.z_mu,
    )

    _, lnmlp1_gradw = LnMlp1Node.forward_and_weight_grads(
        lnmlp1_params, lnmlp1_inputs, lnmlp1_state, lnmlp1_info
    )
    put(
        "dmm_lnmlp1_gradw",
        z_latent=lnmlp1_zlat,
        gW_ff1=lnmlp1_gradw.weights["W_ff1"], gln_gamma=lnmlp1_gradw.weights["ln_gamma"],
        gb_ff1=lnmlp1_gradw.biases["b_ff1"], gln_beta=lnmlp1_gradw.biases["ln_beta"],
    )

    _, lnmlp1_igrads, lnmlp1_sg = LnMlp1Node.forward_and_latent_grads(
        lnmlp1_params, lnmlp1_inputs, lnmlp1_state, lnmlp1_info, False
    )
    put(
        "dmm_lnmlp1_gradlatent",
        z_latent=lnmlp1_zlat,
        grad_in=lnmlp1_igrads["mha->mlp1:in"], self_grad=lnmlp1_sg,
    )

    # =====================================================================
    # Mlp2ResidualNode: z_mu = residual + (mlp1_in . W_ff2 + b_ff2). Two slots:
    # "in" (from LnMlp1, (S,ff)) and "residual" (skip from MhaResidual, (S,E)).
    # =====================================================================
    W_ff2 = 0.1 * jax.random.normal(split(), (ff, E))
    b_ff2 = 0.05 * jax.random.normal(split(), (E,))

    mlp2_params = NodeParams(weights={"W_ff2": W_ff2}, biases={"b_ff2": b_ff2})
    mlp2_mlp1 = jax.random.normal(split(), (B, S, ff))
    mlp2_res = jax.random.normal(split(), (B, S, E))
    mlp2_inputs = {"m1->mlp2:in": mlp2_mlp1, "mha->mlp2:residual": mlp2_res}
    mlp2_zlat = jax.random.normal(split(), (B, S, E))
    mlp2_state = zeros_state(mlp2_zlat)
    mlp2_info = NodeInfo(
        name="mlp2",
        shape=(S, E),
        node_type="Mlp2ResidualNode",
        node_class=Mlp2ResidualNode,
        node_config={},
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=None,
        weight_init=None,
        slots={},
        in_degree=2,
        out_degree=1,
        in_edges=("m1->mlp2:in", "mha->mlp2:residual"),
        out_edges=(),
        scaling_config=None,
    )

    mlp2_fwd_state = Mlp2ResidualNode.forward(
        mlp2_params, mlp2_inputs, mlp2_state, mlp2_info
    )
    put(
        "dmm_mlp2_fwd",
        W_ff2=W_ff2, b_ff2=b_ff2,
        mlp1=mlp2_mlp1, residual=mlp2_res, out=mlp2_fwd_state.z_mu,
    )

    _, mlp2_gradw = Mlp2ResidualNode.forward_and_weight_grads(
        mlp2_params, mlp2_inputs, mlp2_state, mlp2_info
    )
    put(
        "dmm_mlp2_gradw",
        z_latent=mlp2_zlat,
        gW_ff2=mlp2_gradw.weights["W_ff2"], gb_ff2=mlp2_gradw.biases["b_ff2"],
    )

    _, mlp2_igrads, mlp2_sg = Mlp2ResidualNode.forward_and_latent_grads(
        mlp2_params, mlp2_inputs, mlp2_state, mlp2_info, False
    )
    put(
        "dmm_mlp2_gradlatent",
        z_latent=mlp2_zlat,
        grad_in=mlp2_igrads["m1->mlp2:in"], grad_residual=mlp2_igrads["mha->mlp2:residual"],
        self_grad=mlp2_sg,
    )


# ==============================================================================
# Node group: embedding_vocab
# ==============================================================================
def generate_embedding_vocab_fixtures(put, split):
    """
    Tier B (docs/AUDIT_REGISTER.md section 6) fixtures for EmbeddingNode +
    VocabProjectionNode (fabricpc/nodes/transformer_v2.py). B=3, S=4, V=6, E=5
    (embed's vocab/output dim), Vout=6 (vocab-projection's output dim).

    EmbeddingNode is bespoke (both sides): scatter-add weight grad by token id,
    zero/discrete input grad. Ground truth here still calls the class's OWN
    forward_and_weight_grads/forward_and_latent_grads (which upstream does NOT
    override -- it inherits NodeBase's generic `jax.value_and_grad(forward)`, and
    JAX's autodiff of a `params.weights["embeddings"][indices]` gather naturally
    produces the scatter-add gradient) rather than hand-computing scatter-add in
    this script, so the fixture is the actual upstream computation, not a
    re-derivation of it.

    VocabProjectionNode is seam-based (compute_mu only): defaults verified by
    reading the class (DEFAULT_ACTIVATION=SoftmaxActivation, DEFAULT_ENERGY=
    CrossEntropyEnergy) -- forward/forward_and_weight_grads/forward_and_latent_grads
    all come from NodeBase's generic jax.value_and_grad seam (no override), which
    is the exact analogue of Julia's Phase-D Zygote/Enzyme seam.

    ID CONVENTION (RNG trap sibling issue -- upstream 0-based vs Julia 1-based
    token ids, this codebase's documented divergence for EmbeddingNode/_one_hot):
    took the "keep the embedding table IDENTICAL, feed each side its own native
    id array" approach -- idx dumped here is upstream's native 0-based int array;
    the Julia snippet adds 1 and Float32-encodes it itself. The SAME `embeddings`
    array (unchanged) is used on both sides, so row `idx0[b,s]` (Python, 0-based)
    and row `idx0[b,s]+1` (Julia, 1-based) address the identical embedding row.
    """
    from fabricpc.nodes.transformer_v2 import EmbeddingNode, VocabProjectionNode
    from fabricpc.core.types import NodeParams, NodeState, NodeInfo
    from fabricpc.core.energy import GaussianEnergy, CrossEntropyEnergy
    from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
    import jax
    import jax.numpy as jnp

    B, S, V, E, Vout = 3, 4, 6, 5, 6

    # ---------------- EmbeddingNode ----------------
    embeddings = jax.random.normal(split(), (V, E)).astype(jnp.float32)
    idx0 = jax.random.randint(split(), (B, S), 0, V)  # upstream: 0-based ids
    z_latent_emb = jax.random.normal(split(), (B, S, E)).astype(jnp.float32)

    emb_params = NodeParams(weights={"embeddings": embeddings}, biases={})
    emb_state = NodeState(
        z_latent=z_latent_emb,
        z_mu=jnp.zeros((B, S, E), dtype=jnp.float32),
        error=jnp.zeros((B, S, E), dtype=jnp.float32),
        energy=jnp.zeros((B,), dtype=jnp.float32),
        latent_grad=jnp.zeros((B, S, E), dtype=jnp.float32),
    )
    emb_energy_obj = GaussianEnergy()
    emb_info = NodeInfo(
        name="emb",
        shape=(S, E),
        node_type="EmbeddingNode",
        node_class=EmbeddingNode,
        node_config={"vocab_size": V, "embed_dim": E},
        activation=IdentityActivation(),
        energy=emb_energy_obj,
        latent_init=None,
        weight_init=None,
        slots=EmbeddingNode.get_slots(),
        in_degree=1,
        out_degree=1,
        in_edges=("tok->emb:in",),
        out_edges=(),
        scaling_config=None,
    )
    inputs_emb = {"tok->emb:in": idx0}

    new_state = EmbeddingNode.forward(emb_params, inputs_emb, emb_state, emb_info)
    total_e = jnp.sum(new_state.energy)
    put(
        "embedding_vocab_embed_fwd",
        idx=idx0,
        embeddings=embeddings,
        z_latent=z_latent_emb,
        mu=new_state.z_mu,
        energy=new_state.energy,
        total_energy=jnp.asarray(total_e),
    )

    _, params_grad = EmbeddingNode.forward_and_weight_grads(
        emb_params, inputs_emb, emb_state, emb_info
    )
    put(
        "embedding_vocab_embed_gradw",
        idx=idx0,
        embeddings=embeddings,
        z_latent=z_latent_emb,
        gradembeddings=params_grad.weights["embeddings"],
    )

    _, input_grads, self_grad = EmbeddingNode.forward_and_latent_grads(
        emb_params, inputs_emb, emb_state, emb_info, False
    )
    put(
        "embedding_vocab_embed_gradz",
        idx=idx0,
        embeddings=embeddings,
        z_latent=z_latent_emb,
        gradidx=input_grads["tok->emb:in"],
        selfgrad=self_grad,
    )

    # ---------------- VocabProjectionNode ----------------
    x_vocab = jax.random.normal(split(), (B, S, E)).astype(jnp.float32)
    W_out = (jax.random.normal(split(), (E, Vout)) * 0.1).astype(jnp.float32)
    b_out = (jax.random.normal(split(), (Vout,)) * 0.1).astype(jnp.float32)
    labels = jax.random.randint(split(), (B, S), 0, Vout)
    z_latent_vocab = jax.nn.one_hot(labels, Vout).astype(jnp.float32)

    vocab_params = NodeParams(weights={"W_out": W_out}, biases={"b_out": b_out})
    vocab_state = NodeState(
        z_latent=z_latent_vocab,
        z_mu=jnp.zeros((B, S, Vout), dtype=jnp.float32),
        error=jnp.zeros((B, S, Vout), dtype=jnp.float32),
        energy=jnp.zeros((B,), dtype=jnp.float32),
        latent_grad=jnp.zeros((B, S, Vout), dtype=jnp.float32),
    )
    vocab_energy_obj = CrossEntropyEnergy()
    vocab_act_obj = SoftmaxActivation()
    vocab_info = NodeInfo(
        name="vocab",
        shape=(S, Vout),
        node_type="VocabProjectionNode",
        node_class=VocabProjectionNode,
        node_config={"vocab_size": Vout, "embed_dim": E},
        activation=vocab_act_obj,
        energy=vocab_energy_obj,
        latent_init=None,
        weight_init=None,
        slots=VocabProjectionNode.get_slots(),
        in_degree=1,
        out_degree=1,
        in_edges=("h->vocab:in",),
        out_edges=(),
        scaling_config=None,
    )
    inputs_vocab = {"h->vocab:in": x_vocab}

    new_state_v = VocabProjectionNode.forward(
        vocab_params, inputs_vocab, vocab_state, vocab_info
    )
    total_ev = jnp.sum(new_state_v.energy)
    put(
        "embedding_vocab_vocab_fwd",
        x=x_vocab,
        W_out=W_out,
        b_out=b_out,
        z_latent=z_latent_vocab,
        mu=new_state_v.z_mu,
        energy=new_state_v.energy,
        total_energy=jnp.asarray(total_ev),
    )

    _, params_grad_v = VocabProjectionNode.forward_and_weight_grads(
        vocab_params, inputs_vocab, vocab_state, vocab_info
    )
    put(
        "embedding_vocab_vocab_gradw",
        x=x_vocab,
        W_out=W_out,
        b_out=b_out,
        z_latent=z_latent_vocab,
        gradW_out=params_grad_v.weights["W_out"],
        gradb_out=params_grad_v.biases["b_out"],
    )

    _, input_grads_v, self_grad_v = VocabProjectionNode.forward_and_latent_grads(
        vocab_params, inputs_vocab, vocab_state, vocab_info, False
    )
    put(
        "embedding_vocab_vocab_gradz",
        x=x_vocab,
        W_out=W_out,
        b_out=b_out,
        z_latent=z_latent_vocab,
        gradx=input_grads_v["h->vocab:in"],
        selfgrad=self_grad_v,
    )


# ==============================================================================
# Node group: storkey
# ==============================================================================
import jax
import jax.numpy as jnp

from fabricpc.nodes.storkey_hopfield import StorkeyHopfield
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy


def generate_storkey_fixtures(put, split):
    """StorkeyHopfield (composite PC + attractor energy) -- forward + BOTH grad
    seams (weight grads via forward_and_weight_grads, latent/input grads via
    forward_and_latent_grads), isolated node-level (no graph() call). This node
    only overrides `forward`/`accumulate_hopfield_energy`/`_prepare_W`/
    `initialize_params`/`get_slots` upstream -- forward_and_weight_grads and
    forward_and_latent_grads are the INHERITED NodeBase autodiff defaults
    (jax.value_and_grad of node_class.forward), so this is exactly the ground
    truth Julia's generic Phase-D seam (autodiff.jl) must match once it
    dispatches to StorkeyHopfield's `forward`/`energy_kernel` overrides.

    W is the RAW (unprepared) stored matrix: forward()/compute_mu re-applies
    `_prepare_W(params.weights[edge_key], config)` on every call (idempotent
    once symmetric -- applying it twice is a no-op), and the gradient w.r.t.
    this raw stored matrix already has the symmetrization folded in by
    autodiff -- no pre-symmetrization needed here, and none should be applied
    on the Julia side either when loading this fixture.
    """
    D, B = 4, 3
    edge_key = "p->hop:in"

    W = 0.5 * jax.random.normal(split(), (D, D))            # raw Hopfield weight (D,D)
    b = 0.1 * jax.random.normal(split(), (1, D))              # bias (1,D)
    # hopfield_strength: upstream stores this as a 0-d scalar jnp.array (see
    # StorkeyHopfield.initialize_params: inverse_softplus(jnp.array(1.0))) and
    # uses it UNINDEXED in forward() (`jax.nn.softplus(params.biases["hopfield_strength"])`).
    # Julia stores the SAME scalar value as a (1,1) Matrix{Float32} and indexes
    # [1,1] in `_strength`. Keep the JAX-side param genuinely 0-d for the actual
    # computation (a (1,1) array here would silently broadcast E_hopfield (B,)
    # against strength (1,1) -> (1,B), corrupting the energy shape -- caught by
    # self-test during development) and only reshape to (1,1) when writing to
    # the fixture file, for Julia to load into its own (1,1) convention.
    s_raw_scalar = jax.random.normal(split(), ()) * 0.5       # raw hopfield_strength, 0-d
    probe = jax.random.normal(split(), (B, D))                 # upstream "in" edge input
    zlat = jax.random.normal(split(), (B, D))                   # this node's z_latent

    params = NodeParams(
        weights={edge_key: W},
        biases={"b": b, "hopfield_strength": s_raw_scalar},
    )
    state = NodeState(
        z_latent=zlat,
        z_mu=jnp.zeros((B, D)),
        error=jnp.zeros((B, D)),
        energy=jnp.zeros((B,)),
        latent_grad=jnp.zeros((B, D)),
    )
    node_info = NodeInfo(
        name="hop",
        shape=(D,),
        node_type="StorkeyHopfield",
        node_class=StorkeyHopfield,
        node_config={"enforce_symmetry": True, "zero_diagonal": False, "use_bias": True},
        activation=TanhActivation(),
        energy=GaussianEnergy(),
        latent_init=None,
        weight_init=None,
        slots={},
        in_degree=1,
        out_degree=1,
        in_edges=(edge_key,),
        out_edges=(),
        scaling_config=None,
    )

    # --- forward ---
    new_state = StorkeyHopfield.forward(params, {edge_key: probe}, state, node_info)
    total_energy = jnp.sum(new_state.energy)
    put(
        "storkey_fwd",
        w=W, b=b, s_raw=jnp.reshape(s_raw_scalar, (1, 1)), probe=probe, zlat=zlat,
        zmu=new_state.z_mu, error=new_state.error, energy=new_state.energy,
        # 1-element vector, not a true 0-d array -- more portable across NPZ.jl.
        total_energy=jnp.reshape(jnp.asarray(total_energy), (1,)),
    )

    # --- weight grads (learning phase): dE/dparams, incl W, b, hopfield_strength ---
    _, params_grad = StorkeyHopfield.forward_and_weight_grads(
        params, {edge_key: probe}, state, node_info
    )
    put(
        "storkey_gradw",
        dw=params_grad.weights[edge_key],
        db=params_grad.biases["b"],
        ds=jnp.reshape(params_grad.biases["hopfield_strength"], (1, 1)),
    )

    # --- latent/input grads (inference phase): internal node (in_degree=1,
    # out_degree=1, is_clamped=False) -> autodiff branch, dE/d(probe) + dE/dz_latent.
    _, input_grads, self_grad = StorkeyHopfield.forward_and_latent_grads(
        params, {edge_key: probe}, state, node_info, False
    )
    put(
        "storkey_gradlatent",
        dprobe=input_grads[edge_key],
        dz=self_grad,
    )


# ---------------------------------------------------------------------------------------
generate_linear_family_fixtures(put, split)
generate_identity_skip_fixtures(put, split)
generate_transformer_block_fixtures(put, split)
generate_decomposed_mha_mlp_fixtures(put, split)
generate_embedding_vocab_fixtures(put, split)
generate_storkey_fixtures(put, split)

np.savez("test/conformance/fixtures/tier_b.npz", **OUT)
print(f"wrote {len(OUT)} arrays to test/conformance/fixtures/tier_b.npz")
