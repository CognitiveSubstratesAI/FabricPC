#!/usr/bin/env python3
"""
Tier D transformer-LM track -- C-05 NORM-CLIP BOUNDARY-STRADDLE VARIANT (2026-07-14).

BACKGROUND: generate_tier_d_transformer_stable_fixtures.py's own InferenceSGDNormClip fixture
(tier_d_transformer_stable_normclip.npz, max_norm=1.0, C-05's own test default) passed
523/523 at 1e-4 -- but a post-hoc check of that run's ACTUAL gradient L2 norms found
embed/transformer_0 sitting around ~0.10-0.27 and skip_0/output sitting around 2.4-33.4,
i.e. embed/transformer_0 NEVER approach max_norm=1.0 (never clip) and skip_0/output are
ALWAYS far above it (always clip) -- an "all-or-nothing" split with no sample ever landing
near the `min(1, max_norm/(norm+eps))` threshold. That means the norm-clip's actual
boundary-decision branch (where float32 implementations could plausibly disagree about
whether a given sample clips or not) was never exercised by that fixture: "passes" there is
coverage-BY-ABSENCE of the risky code path, not a positive test of it.

THIS SCRIPT closes that gap. It reuses generate_tier_d_transformer_stable_fixtures.py's EXACT
setup (same seed=2026, same architecture: B=3,S=8,V=10,E=8,num_heads=2,num_blocks=1, same
un-forked params0/batch RNG-split order, same ETA_INFER=0.01/INFER_STEPS=12 contractive
regime -- see that script's own docstring for the full E5 contractive-regime rationale) so
its params0/batch are BYTE-IDENTICAL to tier_d_transformer_stable_sgd.npz /
tier_d_transformer_stable_normclip.npz's own params0_*/batch_* arrays. The ONLY change is
max_norm: 1.0 -> 2.5 (see the derivation below for why 2.5 and not the naive 0.05-0.10
guess aimed at embed/transformer_0's own raw-grad range).

WHY 2.5, MEASURED (not guessed) -- INCLUDING A FALSIFIED FIRST HYPOTHESIS, kept here because
it is the reason 2.5 (not the naive 0.05-0.10 guess aimed at embed/transformer_0's own raw
range) is correct:

Step 1 -- a throwaway diagnostic (measure_gradnorms.py, run against this exact
seed/architecture/batch with plain InferenceSGD, UNCLIPPED) dumped every non-clamped node's
per-sample latent_grad L2 norm at all 12 relax steps: embed/transformer_0 grow smoothly from
0.0 to ~0.26-0.27; skip_0 sits in ~2.2-2.98, *decreasing* over the 12 steps; output (a
CLAMPED node -- see below) sits in ~28.6-33.4. The naive plan was max_norm~=0.10, aimed at
embed/transformer_0's own range.

Step 2 -- FALSIFIED BY A SELF-CONSISTENCY CHECK: `output` is clamped (in `clamps`), and
`update_latents` (inference.py:202-227) only calls `compute_new_latent`/applies the clip for
NON-clamped nodes -- so `output`'s clip is never actually exercised by the relaxation loop at
all (its grad is dumped but never clipped-and-applied); only embed/transformer_0/skip_0 are
in play. Actually running InferenceSGDNormClip(max_norm=0.10) and re-measuring latent_grad
INSIDE that run (not the unclipped baseline) showed embed/transformer_0's raw norms
CRATERING to ~0.001-0.006 through step 6 -- ~30x smaller than the unclipped baseline predicts,
NEVER approaching 0.10 at all (no straddle). Root cause: skip_0's raw grad (~2.2-2.98) is
~25x above max_norm=0.10, so skip_0 gets clipped almost to a standstill from step 1
(clip_factor~=0.10/2.7~=0.037, an ~96% magnitude cut) -- since skip_0 sits structurally
between embed/transformer_0 and the output error signal (SkipConnection sums
embed+transformer_0 -> output), its near-frozen z_latent starves embed/transformer_0's OWN
backward-pushed error signal every subsequent step. Global max_norm applies to ALL non-clamped
nodes at once, so a value chosen to matter for embed/transformer_0 alone, without checking
what it does to skip_0 (which is ALREADY always-clipping at max_norm=1.0 too, just gently),
silently changes skip_0's clip severity from "mild" to "near-total", which cascades and
invalidates the very distribution the choice was based on. Lesson: pick max_norm by
self-consistently re-measuring INSIDE an actual clipped run at the candidate value, not by
reading off the unclipped baseline.

Step 3 -- CORRECTED, self-consistently verified: target skip_0's OWN natural raw-grad range
instead (2.2-2.98, decreasing over the 12 steps) with a max_norm inside that range, so the
clip stays gentle (clip_factor close to 1) rather than catastrophic, and the resulting
trajectory stays close to the unclipped baseline (verified: re-running with
max_norm=2.5 reproduces embed/transformer_0 raw norms within ~5-6% of the unclipped
baseline at every step, e.g. embed step5 = [0.0940,0.0917,0.0944] vs unclipped
[0.1002,0.0918,0.1104] -- NOT the 30x collapse above). At max_norm=2.5, skip_0's own raw
per-sample grad norm (measured INSIDE this exact clipped run, self-consistently) straddles
the threshold genuinely for 2 of 3 batch samples, with near-exact ties:

    sample0: step8=2.5237 (+0.95%, CLIP) | step9=2.4987 (-0.05%, no-clip) -- crosses AT step 8->9,
             ratio changes from 1.0095 to 0.9995 (essentially an exact tie at step 9)
    sample1: step1=2.5145 (+0.58%, CLIP) | step2=2.4846 (-0.62%, no-clip) -- crosses AT step 1->2,
             an even tighter tie (ratio 1.0058 -> 0.9938)
    sample2: 2.68-2.98 throughout (ratio 1.07-1.19) -- ALWAYS clips, no crossing (that's fine;
             "at least one node...some samples clip and some don't" is satisfied by sample0/1)

Both crossings are hairline (<1% from the boundary on the clipping side), exactly the
scenario where two independent float32 ports of `min(1, max_norm/(norm+eps))` could plausibly
land on opposite sides of the discrete clip/no-clip decision. embed/transformer_0 (raw norms
capped ~0.09-0.24 throughout at max_norm=2.5, per the self-consistency check above) and
output (clamped, clip never applied) do not themselves straddle at this max_norm -- "at least
one node" (skip_0) suffices per the task, and is satisfied with TWO independent near-exact
crossings across the batch.

latent_decay=0.0 / eps=1e-8 UNCHANGED from the max_norm=1.0 fixture (C-05's own test
defaults for everything except max_norm itself).

This is an ADDITIONAL fixture -- it does NOT overwrite or regenerate
tier_d_transformer_stable_sgd.npz / tier_d_transformer_stable_normclip.npz, which stay
exactly as generate_tier_d_transformer_stable_fixtures.py (untouched) left them.

Run with the upstream fabricpc package importable (same venv as every other Tier fixture):
    cd <this repo> && source /home/shivaji1012/dev-zone/FabricPC/.venv-fixtures/bin/activate
    python3 scripts/generate_tier_d_transformer_stable_normclip_boundary_fixtures.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.activations import GeluActivation
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.inference import InferenceSGDNormClip, run_inference
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly.graph_construction import TaskMap, graph
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.skip_connection import SkipConnection
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.nodes.transformer_v2 import EmbeddingNode, VocabProjectionNode
from fabricpc.training.train import get_graph_param_gradient, train_step
from fabricpc.training.train_autoregressive import create_causal_mask  # VALUE helper only

key = jax.random.PRNGKey(2026)  # SAME seed as generate_tier_d_transformer_stable_fixtures.py


def split():
    global key
    key, sub = jax.random.split(key)
    return sub


def put(OUT, prefix, **arrays):
    for name, arr in arrays.items():
        OUT[f"{prefix}_{name}"] = np.asarray(arr)


def dump_state(OUT, prefix, state):
    for name, ns in state.nodes.items():
        put(
            OUT, f"{prefix}_{name}",
            z_latent=ns.z_latent, z_mu=ns.z_mu, error=ns.error, energy=ns.energy,
            latent_grad=ns.latent_grad,
        )


def dump_params(OUT, prefix, params):
    for name, np_ in params.nodes.items():
        for wname, w in np_.weights.items():
            put(OUT, f"{prefix}_{name}", **{f"w__{wname}": w})
        for bname, b in np_.biases.items():
            put(OUT, f"{prefix}_{name}", **{f"b__{bname}": b})


# =========================================================================================
# TINY CONFIG -- IDENTICAL to generate_tier_d_transformer_stable_fixtures.py except MAX_NORM.
# =========================================================================================
B, S, V, E, H = 3, 8, 10, 8, 2
INFER_STEPS = 3 * (2 * 1 + 2)  # = 12 -- UNCHANGED
ETA_INFER = 0.01               # UNCHANGED (contractive regime, same as sibling fixtures)
MAX_NORM = 2.5    # <-- THE ONLY CHANGE FROM tier_d_transformer_stable_normclip.npz (was 1.0);
                  # targets skip_0's OWN natural raw-grad range (~2.2-2.98), self-consistently
                  # verified to produce a genuine clip/no-clip straddle -- see docstring above.
NC_EPS = 1.0e-8    # UNCHANGED -- matches C-05's own test default

input_node = IdentityNode(shape=(S,), name="input")
mask_node = IdentityNode(shape=(1, S, S), name="mask")
embed_node = EmbeddingNode(
    shape=(S, E), name="embed", vocab_size=V, embed_dim=E,
    weight_init=NormalInitializer(std=1.0),
)
block = TransformerBlock(
    shape=(S, E), name="transformer_0", num_heads=H, ff_dim=None,
    use_rope=True, internal_activation=GeluActivation(), rope_theta=10000.0,
)
skip_node = SkipConnection(shape=(S, E), name="skip_0")
output_node = VocabProjectionNode(
    shape=(S, V), name="output", vocab_size=V, embed_dim=E,
    weight_init=NormalInitializer(std=float(np.sqrt(1.0 / E))),
    # activation/energy left at class default (Softmax + CrossEntropy) -- matches Julia's
    # current default, same divergence-handling decision as the sibling stable fixtures.
)

nodes = [input_node, mask_node, embed_node, block, skip_node, output_node]
edges = [
    Edge(source=input_node, target=embed_node.slot("in")),
    Edge(source=embed_node, target=block.slot("in")),
    Edge(source=mask_node, target=block.slot("mask")),
    Edge(source=embed_node, target=skip_node.slot("in")),
    Edge(source=block, target=skip_node.slot("in")),
    Edge(source=skip_node, target=output_node.slot("in")),
]

# =========================================================================================
# params0 / batch -- SAME un-forked split() stream/order as generate_tier_d_transformer_
# stable_fixtures.py's own params0/batch section -- byte-reproducible against that script's
# (and hence both sibling fixtures') params0_*/batch_* arrays.
# =========================================================================================
embed_params = EmbeddingNode.initialize_params(
    split(), (S, E), {}, weight_init=NormalInitializer(std=1.0), config={"vocab_size": V, "embed_dim": E}
)
node_config_tb = {"num_heads": H, "ff_dim": None}
tb_params = TransformerBlock.initialize_params(
    split(), (S, E), {}, weight_init=None, config=node_config_tb
)
output_params = VocabProjectionNode.initialize_params(
    split(), (S, V), {}, weight_init=NormalInitializer(std=float(np.sqrt(1.0 / E))),
    config={"vocab_size": V, "embed_dim": E},
)

params = GraphParams(
    nodes={
        "input": NodeParams(weights={}, biases={}),
        "mask": NodeParams(weights={}, biases={}),
        "embed": embed_params,
        "transformer_0": tb_params,
        "skip_0": NodeParams(weights={}, biases={}),
        "output": output_params,
    }
)

Xb = jax.random.randint(split(), (B, S), 0, V)  # upstream-native 0-based ids, int32
labels = jax.random.randint(split(), (B, S), 0, V)
Yb = jax.nn.one_hot(labels, V)  # (B, S, V) one-hot next-token targets
mask_val = create_causal_mask(S)[None, None, :, :]
mask_val = jnp.broadcast_to(mask_val, (B, 1, S, S))

clamps = {"input": Xb, "output": Yb, "mask": mask_val}  # NODE-name-keyed
batch = {"x": Xb, "y": Yb, "causal_mask": mask_val}     # TASK-name-keyed

dummy_key = jax.random.PRNGKey(999)   # state-init RNG non-issue (see baseline script)
rng_key2 = jax.random.PRNGKey(777)    # get_graph_param_gradient/train_step RNG
LR = 0.05


def run_variant(algo_name, inference_obj, out_path):
    OUT = {}
    structure = graph(
        nodes=nodes, edges=edges,
        task_map=TaskMap(x=input_node, y=output_node, causal_mask=mask_node),
        inference=inference_obj,
    )
    dump_params(OUT, "params0", params)
    put(OUT, "batch", x=Xb, y=Yb, causal_mask=mask_val)

    init_state = initialize_graph_state(structure, B, dummy_key, clamps=clamps, params=params)
    dump_state(OUT, "init", init_state)

    inference_cls = type(inference_obj)
    relax_state = init_state
    # Also record the RAW (unclipped-visibility) latent_grad L2 norm per sample at every step
    # for the four non-clamped nodes, so the Julia side (and this report) can identify EXACTLY
    # which (step, node, sample) triples are near the max_norm boundary without re-deriving it.
    NONCLAMPED = ("embed", "transformer_0", "skip_0", "output")
    norm_log = {name: [] for name in NONCLAMPED}
    for step in range(1, INFER_STEPS + 1):
        relax_state = inference_cls.inference_step(
            params, relax_state, clamps, structure, inference_obj.config
        )
        dump_state(OUT, f"relax{step:02d}", relax_state)
        for name in NONCLAMPED:
            grad = relax_state.nodes[name].latent_grad
            # NOTE: this is the POST-update state's latent_grad, i.e. it reflects the gradient
            # computed and (for NormClip) clipped AT this step -- recompute the pre-clip norm
            # is not directly recoverable here; norms are reported by the separate measurement
            # script (plain InferenceSGD run) which is exactly what this fixture's own
            # docstring/report cites. This log is diagnostic-only, not asserted by the Julia test.
            norm = np.asarray(jnp.sqrt(jnp.sum(grad.conj() * grad, axis=tuple(range(1, grad.ndim)), keepdims=False)))
            norm_log[name].append((step, norm))

    converged = run_inference(params, init_state, clamps, structure)
    dump_state(OUT, "converged", converged)

    _max_sanity_diff = 0.0
    for _name, _ns in converged.nodes.items():
        _rns = relax_state.nodes[_name]
        for _field in ("z_latent", "z_mu", "error", "energy", "latent_grad"):
            _a = np.asarray(getattr(_ns, _field))
            _b = np.asarray(getattr(_rns, _field))
            _d = float(np.max(np.abs(_a - _b))) if _a.size else 0.0
            _max_sanity_diff = max(_max_sanity_diff, _d)
            assert np.allclose(_a, _b, rtol=1e-3, atol=1e-3), (
                f"[{algo_name}] INSTRUMENTATION SANITY FAILED: relax12 vs converged at "
                f"node={_name} field={_field} max|diff|={_d}"
            )
    print(f"[{algo_name}] SANITY OK: relax12 ~= converged (max|diff|={_max_sanity_diff:.3e})")

    grads, energy, grad_final_state = get_graph_param_gradient(params, batch, structure, rng_key2)
    dump_params(OUT, "grad", grads)
    put(OUT, "scalars", energy_pretrain=jnp.reshape(jnp.asarray(energy), (1,)))
    dump_state(OUT, "grad_final", grad_final_state)

    optimizer = optax.sgd(learning_rate=LR)
    opt_state = optimizer.init(params)
    new_params, new_opt_state, energy2, final_state2 = train_step(
        params, opt_state, batch, structure, optimizer, rng_key2
    )
    dump_params(OUT, "trained", new_params)
    put(OUT, "scalars", energy_trainstep=jnp.reshape(jnp.asarray(energy2), (1,)))
    dump_state(OUT, "trained_final", final_state2)

    np.savez(out_path, **OUT)
    print(f"[{algo_name}] wrote {len(OUT)} arrays to {out_path}")

    print(f"[{algo_name}] post-clip latent_grad L2 norms (embed/transformer_0, steps 3-8):")
    for name in ("embed", "transformer_0"):
        for step, norm in norm_log[name]:
            if 3 <= step <= 8:
                print(f"    {name} step={step} norms={norm}")


run_variant(
    "stable_normclip_boundary",
    InferenceSGDNormClip(
        eta_infer=ETA_INFER, infer_steps=INFER_STEPS, latent_decay=0.0,
        max_norm=MAX_NORM, eps=NC_EPS,
    ),
    "test/conformance/fixtures/tier_d_transformer_stable_normclip_boundary.npz",
)
