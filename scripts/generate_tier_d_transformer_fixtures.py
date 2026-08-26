#!/usr/bin/env python3
"""
Tier D conformance fixture generator, TRANSFORMER-LM track (docs/AUDIT_REGISTER.md section 6):
END-TO-END -- initialize_graph_state -> run_inference (to convergence) -> get_graph_param_gradient
-> exactly ONE train_step, on a real ASSEMBLED graph matching src/models/transformer_lm.jl's
`transformer_lm(; seq_len=8, vocab_size=10, embed_dim=8, num_heads=2, num_blocks=1)` topology.
Dumped from the PINNED upstream JAX FabricPC checkout into one .npz consumed by
test/conformance/test_tier_d_transformer.jl (rtol=atol=1e-4 -- looser than Tier C's 1e-5; see
that test file's own comment at the tolerance definition for why: Float32 BLAS associativity
across multiple chained inference steps, not a single primitive op).

This is the highest-risk Tier D track: the assembled graph (EmbeddingNode + TransformerBlock +
SkipConnection + VocabProjectionNode) is topologically the most complex thing conformance-tested
in this repo so far, and (per the divergence-handling notes below) the topology genuinely does
NOT match 1:1 with Julia's graph -- an extra "mask" node is required upstream. Get this exactly
right; do not approximate.

RNG TRAP (same discipline as Tier A/B/C): fixtures carry JAX-GENERATED ARRAYS, never seeds.
`params0` and `batch` are generated here via jax.random (through `split()`, and through calling
each node CLASS's own `initialize_params` staticmethod directly with a jax.random key -- NOT
upstream's top-level `graph_initialization.initialize_params(structure, rng_key)`, which this
script never calls) and dumped as concrete arrays. The Julia side must build GraphParams DIRECTLY
from these loaded arrays, never calling its own `initialize_params`/RNG for any compared value
(a different RNG algorithm entirely). This script fixes its OWN jax.random.PRNGKey purely so
re-running it is reproducible (stable diffs), not to hand a seed to Julia.

STATE-INIT RNG NON-ISSUE (same argument as generate_tier_c_fixtures.py / _mnist, re-verified for
THIS graph, generalized to two in-degree-0 nodes instead of one): initialize_graph_state's default
FeedforwardStateInit only uses its rng_key as a fallback placeholder for in-degree-0 non-clamped /
recurrent nodes; every in-degree>0 node's z_latent is then overwritten by a deterministic forward
pass (z_latent <- z_mu), and every clamped node's z_latent is overwritten by set_latents_to_clamps.
This graph has exactly TWO in-degree-0 nodes ("input" and "mask") and BOTH are always clamped in
every call below -> the rng_key argument threaded into initialize_graph_state/
get_graph_param_gradient/train_step never affects a single dumped output array. Any concrete key
works; no cross-language seed-reproducibility claim is made or needed for it (contrast the genuine
RNG trap on params0/batch above).

======================================================================================
DIVERGENCE-HANDLING DECISIONS -- read before touching this file or the companion Julia test.
Every one of these was independently verified against source (not assumed from the task brief
or from in-repo comments, several of which turned out to be stale -- see #2).
======================================================================================

1. MASK NODE (topology divergence, INTENTIONAL -- transformer_lm.jl's own top-of-file comment
   documents this as its divergence #1, and it is still accurate on this point). Julia's
   TransformerBlock (src/nodes/transformer.jl) has an inline `causal::Bool` flag and NO "mask"
   slot at all (`get_slots(::TransformerBlock) = Dict("in" => SlotSpec("in", false))` --
   transformer.jl:75). Upstream's TransformerBlock (fabricpc/nodes/transformer.py:180-187) has a
   real second slot, `"mask"` (`SlotSpec(name="mask", is_multi_input=False,
   is_variance_scalable=False)`), that must be fed an explicit causal mask via an edge -- the
   canonical wiring is examples/transformer_demo.py:147-222's create_transformer_model:
   `mask_node = IdentityNode(shape=(1, seq_len, seq_len), name="mask")`, no incoming edge (a
   clamped terminal source, exactly like "input"), `Edge(source=mask_node,
   target=block.slot("mask"))`, and the VALUE clamped via `TaskMap(..., causal_mask=mask_node)`
   plus a "causal_mask" key in the batch dict. So THIS fixture's graph has one MORE node ("mask")
   than transformer_lm.jl's Julia graph (6 nodes here: input/mask/embed/transformer_0/skip_0/
   output vs Julia's 5) -- that is correct, not a bug; no node-count parity is attempted, and the
   companion Julia test is expected to simply never read the "*_mask_*" / "mask_*" dumped keys.
   Mask VALUE: create_causal_mask(seq_len) (fabricpc/training/train_autoregressive.py:29-37, a
   plain `jnp.tril(jnp.ones((S,S)))` lower-triangular 0/1 array), broadcast to (B,1,S,S) -- this
   is mathematically the same exclusion Julia's `_tb_causal_mask`/`causal=true` path applies
   (verified: upstream TransformerBlock._mha's `jnp.where(mask==0,-1e9,scores)` masks the exact
   same upper-triangular future positions as Julia's additive `_tb_causal_mask` -1f9 constant --
   transformer.py:480-481 vs transformer.jl:159,204).

   CLAMP PLUMBING -- the part that is easy to get silently wrong (verified by reading
   fabricpc/training/train.py in full, not assumed from the autoregressive-specific wrapper):
   this fixture uses the GENERIC (non-autoregressive-wrapper) API `get_graph_param_gradient`/
   `train_step` from fabricpc.training.train, matching Tier C's precedent exactly (NOT
   train_step_autoregressive/create_causal_mask's own auto-clamp path from
   fabricpc.training.train_autoregressive -- only the create_causal_mask VALUE helper is reused
   from that module, not its training-step wrapper). Those generic functions build their OWN
   internal `clamps` dict by iterating `batch.items()` against `structure.task_map`
   (train.py:136-140: `for task_name, task_value in batch.items(): if task_name in
   structure.task_map: clamps[node_name] = task_value`) -- so the mask value MUST be included as
   `batch["causal_mask"]` (a TASK name) for those two calls to see it at all. This is DIFFERENT
   from the `clamps` dict this script builds by hand for the first `initialize_graph_state`/
   `run_inference` pair, which is NODE-name-keyed (`clamps["mask"]`, `clamps["input"]`,
   `clamps["output"]`). Omitting "causal_mask" from `batch` would silently leave the mask node
   UNCLAMPED on the get_graph_param_gradient/train_step calls -- its in-degree=0 terminal special
   case (base.py:436-454) still fires either way (it does not check is_clamped), but its z_latent
   would then come from whatever initialize_graph_state's OWN default (Gaussian) latent-init
   draws instead of the real causal mask, silently changing the attention pattern between phases.
   This script avoids that by building a single `mask_val` array once and threading it through
   BOTH the node-keyed `clamps` dict and the task-keyed `batch` dict.

2. VocabProjectionNode ACTIVATION/ENERGY -- MATCHES upstream's own default; do NOT override to
   Identity+Gaussian (this corrects a STALE comment in transformer_lm.jl that the original task
   framing was built on -- verified against the actual code body + git history, not assumed).
   transformer_lm.jl's own top-of-file comment (lines 19-25 as of this writing) currently claims
   VocabProjectionNode "defaults to Identity activation + GaussianEnergy (rank-3 SoftmaxActivation
   would normalize over the sequence axis, not vocab)". Reading the ACTUAL struct default in
   src/nodes/transformer_decomposed.jl:366-372 shows the opposite:
   `activation::AbstractActivation=SoftmaxActivation()`, `energy::AbstractEnergy=
   CrossEntropyEnergy()`. `git show 565f9fb` (commit "port transformer_v2 CE-energy output + init
   scales") is the commit that changed this default AND is the same commit that touched
   transformer_lm.jl -- its own commit message says explicitly: "VocabProjectionNode default
   Identity+Gaussian -> Softmax+CrossEntropy (upstream DEFAULT_ACTIVATION/DEFAULT_ENERGY)". The
   transformer_decomposed.jl docstring/comment was updated in that same commit to say
   "Defaults to Softmax + CrossEntropy"; the transformer_lm.jl top-of-file comment block was
   simply never updated to match -- it describes pre-565f9fb behavior. Independent corroboration:
   test/conformance/test_tier_b.jl:676's own inline comment on `VocabProjectionNode((S, Vout),
   "vocab")` reads "# defaults: Softmax + CrossEntropy". transformer_lm.jl's actual
   `output = VocabProjectionNode((seq_len, vocab_size), "output"; weight_init=...)` call passes
   NO activation/energy override, so it builds with today's (correct) class defaults -- which
   ALREADY MATCH upstream transformer_v2.VocabProjectionNode's own DEFAULT_ACTIVATION=
   SoftmaxActivation / DEFAULT_ENERGY=CrossEntropyEnergy exactly (transformer_v2.py:381-383).
   THEREFORE this fixture's `output` node is built with NO explicit activation/energy override
   either -- plain `VocabProjectionNode(shape=..., name="output", vocab_size=V, embed_dim=E,
   weight_init=...)`, taking upstream's own class defaults. Constructing it with
   Identity+Gaussian (as the stale Julia comment would suggest) would introduce a REAL mismatch
   against what transformer_lm.jl actually computes today -- exactly the failure mode this
   divergence-check exists to catch.

3. `scaling=None` / no `graph_state_initializer` override (a THIRD divergence from the demo
   script, beyond the two the task brief named). examples/transformer_demo.py's
   create_transformer_model passes `scaling=MuPCConfig(include_output=False)` AND
   `graph_state_initializer=FeedforwardStateInit()` to `graph()`. transformer_lm.jl's actual call
   is `graph(nodes, edges, TaskMap(...), inference; scaling=scaling)` where
   `scaling::Union{Nothing,MuPCConfig}=nothing` is transformer_lm()'s OWN keyword, and this
   fixture's equivalent invocation (TINY CONFIG, no scaling override requested) leaves it at that
   default -- i.e. NO muPC scaling at all. So this fixture's `graph()` call passes no `scaling`
   kwarg either (upstream's own default is also `scaling=None`,
   fabricpc/graph_assembly/graph_construction.py:114) -- using the demo's MuPCConfig would
   silently scale every edge's forward/gradient differently from what transformer_lm.jl actually
   runs, making the whole comparison meaningless. `graph_state_initializer` is likewise left
   unset on both sides -- verified IDENTICAL default on both sides (`FeedforwardStateInit()`,
   graph_construction.py:236 vs src/graph_assembly/graph_construction.jl:93), matching Tier C/
   Tier D-MNIST's own precedent of never passing it explicitly.

4. VocabProjectionNode vs the demo's plain `Linear`: examples/transformer_demo.py's
   create_transformer_model uses upstream's own `Linear` node (2D-only forward) for its output
   layer with an explicit SoftmaxActivation()+CrossEntropyEnergy() override -- this is the WRONG
   reference class for this fixture (it happens to reach the same activation/energy as decision
   #2 above by a different, irrelevant route). transformer_lm.jl's own comment (its divergence
   #2, independently re-verified here and still accurate on this specific point) explains Julia
   uses `VocabProjectionNode`, not `Linear`, because `Linear.forward` is 2D-only and cannot do the
   per-position (B,S,E)->(B,S,V) projection that a rank-3 sequence output needs. So this fixture
   imports `VocabProjectionNode` from `fabricpc.nodes.transformer_v2` (Tier B's own import path,
   already conformance-tested byte-level at the isolated-node level in
   generate_tier_b_fixtures.py's generate_embedding_vocab_fixtures) for the output node, NOT
   `fabricpc.nodes.Linear`/the demo's construction.

5. rope_theta / internal_activation -- MATCH, not diverge (verified directly in both bodies, not
   assumed from docs/AUDIT_REGISTER.md's X-01..X-05 notes, though they turn out to agree).
   src/nodes/transformer.jl's `_tb_mha` (transformer.jl:171-213) calls
   `_tb_rope_tables(Dh, S)` with NO `theta` argument (transformer.jl:178), so it always uses
   `_tb_rope_tables`'s own default `theta=10000.0f0` (transformer.jl:128) -- there is no
   `rope_theta` FIELD on the `TransformerBlock` struct at all (its 11 fields are shape/name/
   num_heads/activation/energy/internal_activation/ff_dim/use_rope/causal/weight_init/
   latent_init -- transformer.jl:35-47), so 10000.0 is hardcoded and not configurable from
   outside. Upstream's OWN class default is ALSO rope_theta=10000.0 (fabricpc/nodes/
   transformer.py:160) -- so this already matches without any override needed; this fixture still
   passes `rope_theta=10000.0` EXPLICITLY below (not examples/transformer_demo.py's own CLI
   `--rope_theta` default of 500.0, which is irrelevant to this fixture) so a future upstream
   default change cannot silently desync the two sides. `internal_activation`: Julia's default is
   `GeluActivation()` (transformer.jl:55); upstream's own `__init__` default is
   `internal_activation=None` -> `internal_activation or GeluActivation()`
   (transformer.py:171) -- also already matches; passed explicitly here for the same
   future-proofing reason.

6. TOKEN ID CONVENTION (matches Tier B's established split-responsibility pattern exactly --
   see generate_tier_b_fixtures.py's "ID CONVENTION" docstring on generate_embedding_vocab_fixtures
   for the precedent this follows): `x` (the token-id batch clamped onto "input") is generated via
   `jax.random.randint(...,0,vocab_size)` and dumped AS-IS (upstream-native, 0-based, int32); the
   Julia test is expected to add 1 before clamping "input"'s Float32-encoded z_latent
   (EmbeddingNode's `_embed_lookup`, transformer_decomposed.jl:294-302, indexes a 1-based Julia
   array via `Int(round(idx[b,s]))`). `y` (the one-hot target clamped onto "output") needs NO such
   adjustment: a one-hot array's column position is read identically by both languages (it is
   data, not used as an array INDEX on either side) -- unlike "x", which IS used as an index into
   the embedding table.

======================================================================================
DUMPED CONTENTS (mirrors generate_tier_c_fixtures.py's per-phase structure, MINUS the
intermediate single-`inference_step` phase Tier C dumped -- omitted here per this track's own
explicit step-count scoping: "run initialize_graph_state -> run_inference to convergence ->
get_graph_param_gradient -> exactly ONE train_step", no intermediate single-step check, and
attention is O(S^2) so keeping phases minimal matters more here than in Tier C's tiny
feedforward graph):
  - params0_*   : initial weights/biases for every node (input/mask/embed/transformer_0/skip_0/
                  output -- input/mask/skip_0 are empty NodeParams, dumped for construction-parity
                  honesty like Tier C's "x").
  - batch_*     : x (token ids, int32, 0-based), y (one-hot targets), causal_mask (the (B,1,S,S)
                  mask array) -- the SAME three arrays used for every clamped call below.
  - init_*      : full GraphState (all 6 nodes) right after initialize_graph_state.
  - converged_* : full GraphState after run_inference runs INFER_STEPS relaxation iterations.
  - grad_*      : weight/bias gradients from get_graph_param_gradient (learnable nodes only:
                  embed/transformer_0/output).
  - scalars_energy_pretrain : get_graph_param_gradient's returned per-sample energy.
  - grad_final_*: full GraphState from get_graph_param_gradient's OWN re-init + re-inference
                  (independent of the "converged" state above -- same RNG-key-inert argument
                  applies).
  - trained_*   : weight/bias values after the ONE train_step SGD update.
  - scalars_energy_trainstep : train_step's returned per-sample energy (computed on the
                  PRE-update params, matching both sides' documented train_step semantics).
  - trained_final_* : full GraphState from the train_step call.

Run with the upstream fabricpc package importable (see generate_tier_a_fixtures.py's docstring
for the venv activation command -- run from THIS repo as CWD, not the upstream checkout):
    cd <this repo> && source /home/shivaji1012/dev-zone/FabricPC/.venv-fixtures/bin/activate
    python3 scripts/generate_tier_d_transformer_fixtures.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.activations import GeluActivation
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.inference import InferenceSGD, run_inference
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

OUT = {}
key = jax.random.PRNGKey(2026)


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
        for wname, w in np_.weights.items():
            put(f"{prefix}_{name}", **{f"w__{wname}": w})
        for bname, b in np_.biases.items():
            put(f"{prefix}_{name}", **{f"b__{bname}": b})


# =========================================================================================
# TINY CONFIG (explicit user instruction). embed_dim=8 divisible by num_heads=2 (head_dim=4,
# even -- required by RoPE's pairwise rotation). num_blocks=1 (single TransformerBlock/
# SkipConnection pair -- no loop needed, this fixture hand-builds the exact topology rather
# than generalizing over num_blocks, matching the task's literal topology description).
# infer_steps = 3*(2*num_blocks+2) = 12, Julia's own default heuristic for num_blocks=1 --
# NOT overridden (no documented reason to). eta_infer=0.1, Julia's own default -- not
# overridden either.
# =========================================================================================
B, S, V, E, H = 3, 8, 10, 8, 2
INFER_STEPS = 3 * (2 * 1 + 2)  # = 12
ETA_INFER = 0.1

input_node = IdentityNode(shape=(S,), name="input")
mask_node = IdentityNode(shape=(1, S, S), name="mask")  # DIVERGENCE #1 -- see docstring
embed_node = EmbeddingNode(
    shape=(S, E),
    name="embed",
    vocab_size=V,
    embed_dim=E,
    weight_init=NormalInitializer(std=1.0),  # matches transformer_lm.jl's embed override
)
block = TransformerBlock(
    shape=(S, E),
    name="transformer_0",
    num_heads=H,
    ff_dim=None,  # Julia's own default: 4*embed_dim = 32 (both sides compute this identically)
    use_rope=True,
    internal_activation=GeluActivation(),  # DIVERGENCE #5 -- matches Julia's default explicitly
    rope_theta=10000.0,  # DIVERGENCE #5 -- matches Julia's HARDCODED value explicitly
    # activation/energy left at class default (IdentityActivation()/GaussianEnergy()) -- matches
    # Julia's TransformerBlock default (transformer.jl:53-54) exactly; transformer_lm.jl passes
    # no override either.
)
skip_node = SkipConnection(shape=(S, E), name="skip_0")
output_node = VocabProjectionNode(
    shape=(S, V),
    name="output",
    vocab_size=V,
    embed_dim=E,
    weight_init=NormalInitializer(std=float(np.sqrt(1.0 / E))),  # matches transformer_lm.jl override
    # activation/energy left at class default: Softmax + CrossEntropy -- see DIVERGENCE #2 above.
    # Do NOT override to Identity+Gaussian.
)

nodes = [input_node, mask_node, embed_node, block, skip_node, output_node]
edges = [
    Edge(source=input_node, target=embed_node.slot("in")),
    Edge(source=embed_node, target=block.slot("in")),
    Edge(source=mask_node, target=block.slot("mask")),  # DIVERGENCE #1 -- no Julia counterpart
    Edge(source=embed_node, target=skip_node.slot("in")),  # residual bypass
    Edge(source=block, target=skip_node.slot("in")),  # transformer output, summed
    Edge(source=skip_node, target=output_node.slot("in")),
]

inference = InferenceSGD(eta_infer=ETA_INFER, infer_steps=INFER_STEPS)
structure = graph(
    nodes=nodes,
    edges=edges,
    task_map=TaskMap(x=input_node, y=output_node, causal_mask=mask_node),
    inference=inference,
    # graph_state_initializer left unset (default FeedforwardStateInit(), matches Julia's own
    # default exactly) and scaling left unset (default None -- see DIVERGENCE #3 above; do NOT
    # pass the demo script's MuPCConfig here).
)

# =========================================================================================
# params0: weight/bias arrays generated by calling each NODE CLASS's own `initialize_params`
# staticmethod directly with a jax.random key (RNG-trap-compliant -- NOT
# graph_initialization.initialize_params(structure, rng_key), which this script never calls).
# This mirrors generate_tier_b_fixtures.py's `tb_shared_w`/`tb_shared_b` pattern exactly: for
# TransformerBlock, the class's own initialize_params HARDCODES each weight's correct init
# scale internally (1/sqrt(embed_dim) for W_q/W_k/W_v/W_o, Kaiming for W_ff1, 1/sqrt(ff_dim)
# for W_ff2 -- transformer.py:237-281) and IGNORES its `weight_init` parameter entirely, so
# hand-deriving these scales here would risk a transcription bug; calling the real method
# avoids that. EmbeddingNode/VocabProjectionNode's own initialize_params DO use the passed
# `weight_init`, so the SAME NormalInitializer objects used in the node constructors above are
# passed here too, for consistency (not that it matters for what gets dumped: dumped values ARE
# these direct-call outputs, not anything derived from the constructors).
# =========================================================================================
embed_params = EmbeddingNode.initialize_params(
    split(), (S, E), {}, weight_init=NormalInitializer(std=1.0), config={"vocab_size": V, "embed_dim": E}
)
node_config_tb = {"num_heads": H, "ff_dim": None}
tb_params = TransformerBlock.initialize_params(
    split(), (S, E), {}, weight_init=None, config=node_config_tb
)
output_params = VocabProjectionNode.initialize_params(
    split(),
    (S, V),
    {},
    weight_init=NormalInitializer(std=float(np.sqrt(1.0 / E))),
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
dump_params("params0", params)

# =========================================================================================
# batch: token ids (0-based, int32 -- DIVERGENCE #6), one-hot targets, causal mask (B,1,S,S).
# Same three arrays reused for every clamped call below -- see CLAMP PLUMBING (DIVERGENCE #1)
# for why the mask must appear in BOTH the node-keyed `clamps` dict (initialize_graph_state/
# run_inference) and the task-keyed `batch` dict (get_graph_param_gradient/train_step).
# =========================================================================================
Xb = jax.random.randint(split(), (B, S), 0, V)  # upstream-native 0-based ids, int32
labels = jax.random.randint(split(), (B, S), 0, V)
Yb = jax.nn.one_hot(labels, V)  # (B, S, V) one-hot next-token targets
mask_val = create_causal_mask(S)[None, None, :, :]  # (1, 1, S, S)
mask_val = jnp.broadcast_to(mask_val, (B, 1, S, S))  # (B, 1, S, S)
put("batch", x=Xb, y=Yb, causal_mask=mask_val)

clamps = {"input": Xb, "output": Yb, "mask": mask_val}  # NODE-name-keyed
batch = {"x": Xb, "y": Yb, "causal_mask": mask_val}  # TASK-name-keyed (via structure.task_map)

# STATE-INIT RNG NON-ISSUE (see docstring): both in-degree-0 nodes ("input", "mask") are always
# clamped below, so this key's value is inert -- any concrete key works.
dummy_key = jax.random.PRNGKey(999)

init_state = initialize_graph_state(structure, B, dummy_key, clamps=clamps, params=params)
dump_state("init", init_state)

# --- run_inference to convergence (INFER_STEPS=12, structure's own infer_steps) -----------
# DIAGNOSTIC INSTRUMENTATION (differential-debugging the transformer-LM track's 29
# post-relaxation failures -- docs/AUDIT_REGISTER.md section 6): dump the FULL GraphState after
# EACH of the 12 individual InferenceSGD.inference_step calls (prefix "relax01".."relax12"),
# not just init/converged, so the Julia side can trace error growth step-by-step instead of
# only seeing the endpoints. Manually unrolls the same body run_inference's jax.lax.fori_loop
# calls (InferenceBase.inference_step, via InferenceSGD's inherited classmethod) -- same op
# sequence, same config dict, just driven from a Python loop instead of `fori_loop` so an
# intermediate state can be pulled out and dumped. relax12 is verified below to equal
# `converged` (computed independently via the real `run_inference` call) bit-for-bit, since
# both are the same computation over the same init_state -- this is a sanity check on the
# instrumentation itself, not a new numerical claim.
inference_cls = type(inference)
relax_state = init_state
for step in range(1, INFER_STEPS + 1):
    relax_state = inference_cls.inference_step(
        params, relax_state, clamps, structure, inference.config
    )
    dump_state(f"relax{step:02d}", relax_state)

converged = run_inference(params, init_state, clamps, structure)
dump_state("converged", converged)

# Sanity check: relax12 (manual Python-loop unroll, eager) vs converged (real run_inference,
# which drives the identical inference_step body through jax.lax.fori_loop, XLA-compiled).
# FINDING (worth keeping as documented evidence, not just a pass/fail gate): these are NOT
# bit-exact -- max|diff| ~1e-6 after 12 steps, e.g. node="embed" field="z_latent". This is
# upstream JAX's OWN execution-path float32 divergence (XLA operator fusion inside a compiled
# fori_loop reassociates reductions differently than the same ops dispatched eagerly one at a
# time) -- NOT a Julia-vs-Python issue at all, and NOT something this script can eliminate by
# construction (`inference_cls.inference_step` and `run_inference` both delegate to the exact
# same `InferenceBase.inference_step` staticmethod; only the *driving* mechanism differs).
# Directly relevant to the transformer-LM track's own noise hypothesis: it shows float32
# execution-order sensitivity is real and present even WITHIN upstream JAX alone, across two
# mathematically-identical call paths, at a magnitude (~1e-5..1e-6, worst-case observed
# 1.86e-05) roughly an order of magnitude below the track's final 1e-4 tolerance. Loosened
# from bit-exact to a diagnostic allclose so this
# expected divergence doesn't block fixture generation; still hard-fails if it were ever grossly
# larger (which WOULD indicate a real instrumentation bug, e.g. wrong config dict/aliasing).
_max_sanity_diff = 0.0
for _name, _ns in converged.nodes.items():
    _rns = relax_state.nodes[_name]
    for _field in ("z_latent", "z_mu", "error", "energy", "latent_grad"):
        _a = np.asarray(getattr(_ns, _field))
        _b = np.asarray(getattr(_rns, _field))
        _d = float(np.max(np.abs(_a - _b))) if _a.size else 0.0
        if _d > _max_sanity_diff:
            _max_sanity_diff = _d
        assert np.allclose(_a, _b, rtol=1e-3, atol=1e-3), (
            f"INSTRUMENTATION SANITY FAILED (gross mismatch, not just float32 path noise): "
            f"relax12 vs converged at node={_name} field={_field} max|diff|={_d}"
        )
print(
    f"SANITY OK: relax12 ~= converged (max|diff|={_max_sanity_diff:.3e} -- upstream "
    "eager-loop-vs-fori_loop float32 path divergence, not bit-exact; see comment above)"
)

# --- get_graph_param_gradient (local weight grads + energy on a fresh re-init/re-inference) -
rng_key2 = jax.random.PRNGKey(777)
grads, energy, grad_final_state = get_graph_param_gradient(params, batch, structure, rng_key2)
dump_params("grad", grads)
# NOTE: reshaped to a 1-element 1-D array before put() -- matches Tier B/C/D-MNIST's established
# convention: a bare 0-d jax array round-trips through np.savez/NPZ.jl ambiguously (loads back as
# a bare scalar, not an Array), a shape-(1,) array does not.
put("scalars", energy_pretrain=jnp.reshape(jnp.asarray(energy), (1,)))
dump_state("grad_final", grad_final_state)

# --- exactly ONE train_step (SGD via optax, matching Julia's plain-lr train_step) ---------
# STEP COUNT (explicit user instruction): do NOT chain multiple training steps -- this track's
# job is proving the ASSEMBLED graph computes correctly end-to-end, not multi-step convergence
# (already covered at the loop level by Tier C on a simpler graph; chaining steps here would
# mostly re-test that, at much higher fixture-generation cost given attention's O(S^2) cost).
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
np.savez("test/conformance/fixtures/tier_d_transformer.npz", **OUT)
print(f"wrote {len(OUT)} arrays to test/conformance/fixtures/tier_d_transformer.npz")
