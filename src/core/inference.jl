# Inference dynamics. Port of fabricpc/core/inference.py.
#
# The inner predictive-coding loop: iterate `infer_steps` times, each step
#   1. zero latent gradients
#   2. forward pass → predictions (z_mu), errors, energy; accumulate dE/dz_latent
#   3. update each non-clamped latent via the inference algorithm's `compute_new_latent`
#
# muPC scaling is applied at the call-sites here (identity in v0), keeping the node methods
# scaling-unaware. `compute_new_latent(inf, z_latent, latent_grad)` is the pluggable-algorithm
# dispatch point (Julia multiple dispatch, the analogue of upstream's `cls.compute_new_latent`
# class-hierarchy override — InferenceBase, inference.py:229-243) — `update_latents` calls it
# generically on whatever concrete algorithm `structure.config.inference` holds.
#
# NOTE: `jit_flat.jl`'s Dict-free flat inference path (`flat_inference_step`) hardcodes the
# plain-SGD update formula inline rather than dispatching through `compute_new_latent` — it is a
# separate, deliberately SGD-only JIT/Reactant-prep lane (C-05 scope is this eager path only;
# `InferenceSGDNormClip` is not usable there yet).

"""
    InferenceSGD(; eta_infer = 0.1, infer_steps = 20, latent_decay = 0.0)

Standard SGD inference: `z -= eta_infer · latent_grad` (with optional weight
decay `z *= 1 - eta_infer·latent_decay`).
"""
struct InferenceSGD
    eta_infer::Float32
    infer_steps::Int
    latent_decay::Float32
end
InferenceSGD(; eta_infer=0.1, infer_steps=20, latent_decay=0.0) =
    InferenceSGD(Float32(eta_infer), Int(infer_steps), Float32(latent_decay))

"""Port of `InferenceSGD.compute_new_latent` (inference.py:294-309)."""
compute_new_latent(inf::InferenceSGD, z_latent, latent_grad) =
    z_latent .* (1.0f0 - inf.eta_infer * inf.latent_decay) .- inf.eta_infer .* latent_grad

"""
    InferenceSGDNormClip(; eta_infer = 0.1, infer_steps = 20, latent_decay = 0.0,
                          max_norm = 1.0, eps = 1e-8)

SGD inference with per-sample L2 gradient-norm clipping: `z -= eta_infer · clip(latent_grad)`,
clipping each node's latent gradient independently per batch row (`||grad|| > max_norm` scales it
down to `max_norm`; `eps` guards the division against a zero norm). Port of
`InferenceSGDNormClip` (inference.py:312-356).
"""
struct InferenceSGDNormClip
    eta_infer::Float32
    infer_steps::Int
    latent_decay::Float32
    max_norm::Float32
    eps::Float32
end
InferenceSGDNormClip(;
    eta_infer=0.1, infer_steps=20, latent_decay=0.0, max_norm=1.0, eps=1.0f-8
) = InferenceSGDNormClip(
    Float32(eta_infer), Int(infer_steps), Float32(latent_decay), Float32(max_norm),
    Float32(eps)
)

"""Port of `InferenceSGDNormClip.compute_new_latent` (inference.py:338-356)."""
function compute_new_latent(inf::InferenceSGDNormClip, z_latent, latent_grad)
    dims = 2:ndims(latent_grad)                       # per-sample: all but the batch (1st) dim
    grad_norm = sqrt.(sum(abs2, latent_grad; dims=dims))
    clip_factor = min.(1.0f0, inf.max_norm ./ (grad_norm .+ inf.eps))
    clipped_grad = latent_grad .* clip_factor
    return z_latent .* (1.0f0 - inf.eta_infer * inf.latent_decay) .-
           inf.eta_infer .* clipped_grad
end

"""
    InferenceSGDMomentum(; eta_infer = 0.1, infer_steps = 20, latent_decay = 0.0, momentum = 0.9)

Heavy-ball momentum inference: `v(t+1) = momentum·v(t) - eta_infer·latent_grad;
z(t+1) = z(t)·(1 - eta_infer·latent_decay) + v(t+1)`.

Upstream has not implemented this — it exists only as a design document
(`docs/dev_plans_archive/momentum_sgd_inference_plan.md`, not yet in `inference.py` as of the
commit this port tracks). This is therefore a deliberate ahead-of-upstream divergence, not a
conformance-gap fill: there is no reference output to port against, so its correctness rests on
the momentum=0 conformance anchor (below) plus the theory-vs-measurement closure documented in
`docs/decisions.md`.

Per the upstream plan, this overrides `run_inference` rather than `compute_new_latent`: velocity
must be carried across inference steps, which the per-node `compute_new_latent(inf, z_latent,
latent_grad)` interface (called once per step, no memory of prior steps) has no access to. See
`_run_inference_loop(inf::InferenceSGDMomentum, ...)` below for the actual update; the
`compute_new_latent` method on this type is a plain-SGD fallback only (mirrors upstream's own
documented "not used in practice" fallback) for any caller that invokes it directly outside the
`run_inference` loop.
"""
struct InferenceSGDMomentum
    eta_infer::Float32
    infer_steps::Int
    latent_decay::Float32
    momentum::Float32
end
InferenceSGDMomentum(; eta_infer=0.1, infer_steps=20, latent_decay=0.0, momentum=0.9) =
    InferenceSGDMomentum(
        Float32(eta_infer), Int(infer_steps), Float32(latent_decay), Float32(momentum)
    )

"""Plain-SGD fallback (momentum not applied) — see the type docstring; the real update path is
`_run_inference_loop(inf::InferenceSGDMomentum, ...)`, which carries velocity across steps."""
compute_new_latent(inf::InferenceSGDMomentum, z_latent, latent_grad) =
    z_latent .* (1.0f0 - inf.eta_infer * inf.latent_decay) .- inf.eta_infer .* latent_grad

"""Gather each in-edge's source `z_latent` into an ORDERED map keyed by edge key. Port of
`gather_inputs`. Uses `OrderedDict` (insertion == `in_edges` order) NOT a hash `Dict`: `forward`
folds these inputs (`pre += x·W`), and float addition is non-associative, so the fold order must
match upstream JAX — whose Python `dict` iterates in insertion order (= `in_edges` order). A plain
Julia `Dict` iterates in hash order, silently diverging by ~1 ULP on 3+ in-edge nodes (masked by
the rtol-1e-6 conformance, but it broke bit-identity with the flat / in-place lanes, which fold in
`in_src` = `in_edges` order)."""
function gather_inputs(info::NodeInfo, structure::GraphStructure, state::GraphState)
    in_data = OrderedDict{String, Any}()
    for edge_key in info.in_edges
        src = structure.edges[edge_key].source
        in_data[edge_key] = state.nodes[src].z_latent
    end
    return in_data
end

"""Phase 1: zero every node's `latent_grad` (shape/dtype from latent_grad, which is always float)."""
function zero_grads(state::GraphState, structure::GraphStructure)
    for name in structure.node_names
        state = update_node_in_state(
            state,
            name;
            latent_grad=zero(state.nodes[name].latent_grad)
        )
    end
    return state
end

"""
Phase 2: forward pass; accumulate latent gradients. The universal PC mechanics.
Iterates nodes in insertion order; per node, adds its scaled self-grad to its own
accumulator (preserving earlier downstream contributions) and pushes scaled input
gradients onto its source nodes. Accumulation is additive over the fixed-`z_latent`
inputs, hence order-robust. Port of `forward_value_and_grad`.
"""
function forward_value_and_grad(
    params::GraphParams,
    state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    for name in structure.node_names
        node = structure.nodes[name]
        info = structure.infos[name]
        in_data = gather_inputs(info, structure, state)
        scaled_inputs = scale_inputs(in_data, info.scaling_config)

        node_state, inedge_grads, self_grad = forward_and_latent_grads(
            node,
            params.nodes[name],
            scaled_inputs,
            state.nodes[name],
            info,
            haskey(clamps, name)
        )

        inedge_grads = scale_input_grads(inedge_grads, info.scaling_config)
        self_grad = scale_self_grad(self_grad, info.scaling_config)
        node_state = update_state(
            node_state; latent_grad=node_state.latent_grad .+ self_grad
        )
        state = put_node(state, name, node_state)

        for (edge_key, grad) in inedge_grads
            src = structure.edges[edge_key].source
            state = update_node_in_state(
                state,
                src;
                latent_grad=state.nodes[src].latent_grad .+ grad
            )
        end
    end
    return state
end

"""
Phase 3: latent update on non-clamped nodes, dispatching to the inference algorithm's
`compute_new_latent` (`structure.config.inference`'s concrete type). Port of `update_latents`
(inference.py:201-227); the per-algorithm formula itself is `compute_new_latent` above.
"""
function update_latents(
    state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    inf = structure.config.inference
    for name in structure.node_names
        if !haskey(clamps, name)
            ns = state.nodes[name]
            new_z = compute_new_latent(inf, ns.z_latent, ns.latent_grad)
            state = update_node_in_state(state, name; z_latent=new_z)
        end
    end
    return state
end

"""One inference step: zero grads → forward+accumulate → latent update. Port of `inference_step`."""
function inference_step(
    params::GraphParams,
    state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    state = zero_grads(state, structure)
    state = forward_value_and_grad(params, state, clamps, structure)
    state = update_latents(state, clamps, structure)
    return state
end

"""
Run the inference loop `infer_steps` times to convergence. Port of `run_inference`. Dispatches
on `structure.config.inference`'s type via `_run_inference_loop`: the generic per-step
`inference_step`/`compute_new_latent` path below for algorithms with no cross-step state
(`InferenceSGD`, `InferenceSGDNormClip`), or a specialized override for algorithms that need to
carry extra state across steps (`InferenceSGDMomentum`'s velocity — see its `_run_inference_loop`
method further down). `structure::GraphStructure` is not itself parametric on `config`'s type
(`config::NamedTuple`, checked at runtime), so the dispatch is on the extracted `inference` value,
not on `structure` directly — this is Julia's analogue of upstream's per-algorithm subclass
override of `run_inference` (inference.py's `InferenceBase` hierarchy).

`optimize=false` (default) runs the exact stock loop, byte-for-byte unchanged — this is what
every conformance test and the whole training path use, so their behaviour is untouched. Passing
`optimize=true` selects the algebraically-equivalent HOIST+PRUNE loop (`_run_inference_loop_opt`,
below) for the two stateless SGD algorithms. That loop is bit-identical on every field the rest of
the system ever CONSUMES — all unclamped nodes' `z_latent` (the actual inference result) and every
node's `z_mu`/`error`/`energy`, hence identical downstream weight gradients and
energy (`compute_local_weight_gradients`/`get_graph_param_gradient` recompute those from the
converged state and never read `latent_grad`). Its ONLY observable divergence is the `latent_grad`
of *clamped* nodes, which it leaves at zero instead of accumulating the provably-dead value that
`update_latents` skips for clamped nodes anyway. Because a raw conformance dump (Tier C/D) DOES
compare that scratch field, the optimization is deliberately opt-in rather than default-on — see
`test/optimizations/test_inference_optimizations.jl` for the bit-identity oracle and
`_hoistable_nodes`/`_run_inference_loop_opt` for the mechanics. `InferenceSGDMomentum` ignores the
flag (falls through to its own velocity-carrying loop).
"""
function run_inference(
    params::GraphParams,
    initial_state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure;
    optimize::Bool=false
)
    inf = structure.config.inference
    if optimize && (inf isa InferenceSGD || inf isa InferenceSGDNormClip)
        return _run_inference_loop_opt(inf, params, initial_state, clamps, structure)
    end
    return _run_inference_loop(inf, params, initial_state, clamps, structure)
end

"""Generic loop: `inference_step` (which dispatches `compute_new_latent` per node) `infer_steps`
times. Used by any inference algorithm with no state carried across steps."""
function _run_inference_loop(
    inf,
    params::GraphParams,
    initial_state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    state = initial_state
    for _ in 1:inf.infer_steps
        state = inference_step(params, state, clamps, structure)
    end
    return state
end

# ─────────────────────────────────────────────────────────────────────────────
# HOIST + PRUNE — algebraically-exact inference optimizations (opt-in via
# `run_inference(...; optimize=true)`). Both are derived from two facts about the
# stock loop above:
#   PRUNE  update_latents (Phase 3) NEVER reads a clamped node's `latent_grad`
#          (it `continue`s past clamped nodes), and compute_local_weight_gradients
#          recomputes from the converged state without reading `latent_grad` at all.
#          So every accumulation WRITE into a clamped node's `latent_grad` — a
#          back-edge grad flowing into a clamped source (e.g. dE_h/dz_x → clamped x),
#          or a clamped node's own self-grad into its own accumulator — is dead work.
#   HOIST  a node whose in-edge sources are ALL clamped sees constant inputs across
#          every step (clamped z_latent never changes), so its matmul-heavy forward
#          (`z_mu`) is loop-invariant and can be computed ONCE. Its
#          `error`/`energy`/`self_grad` still depend on its OWN (relaxing) z_latent, so
#          those are recomputed each step from the hoisted z_mu — cheap elementwise, no
#          matmul. And every one of its input-grads flows to a clamped source, so by
#          PRUNE the entire back-projection (`dpre * Wᵀ`) is skipped outright.
# Bit-identity is by construction: the hoisted `pre`/`z_mu` are the SAME operands the
# stock loop recomputes each step (Float32 BLAS is deterministic for identical
# operands), self-grad is recomputed from the identical (z_latent, z_mu), and the
# accumulation ORDER into unclamped nodes is preserved (self-grad added at the node's
# own iteration, downstream back-edges at the successor's — node_names order unchanged).

"""
    _hoistable_nodes(clamps, structure) -> Set{String}

Nodes whose forward (`z_mu`) is loop-invariant AND take the plain
`else` branch of `forward_and_latent_grads`: `in_degree > 0` (not a terminal source),
`out_degree > 0` and unclamped (so NOT the eval/`unclamped-output` special case), and
every in-edge source clamped (so inputs are constant and all back-edges prune away).
"""
function _hoistable_nodes(clamps::AbstractDict, structure::GraphStructure)
    hoist = Set{String}()
    for name in structure.node_names
        info = structure.infos[name]
        info.in_degree > 0 || continue          # terminal source: nothing to hoist
        info.out_degree > 0 || continue          # unclamped/clamped-output branch: leave to stock path
        haskey(clamps, name) && continue          # clamped node: handled by the prune branch, not hoisted
        all_clamped = true
        for edge_key in info.in_edges
            if !haskey(clamps, structure.edges[edge_key].source)
                all_clamped = false
                break
            end
        end
        all_clamped && push!(hoist, name)
    end
    return hoist
end

"""
    _static_sources(clamps, structure) -> Set{String}

Clamped terminal sources (`in_degree == 0` AND clamped) — e.g. the MNIST `"x"` input. Their state
is FULLY loop-invariant: `z_latent` is clamped (Phase 3 never updates it), so `z_mu = z_latent`,
`error`/`energy`/`latent_grad` are all constant across every step. `_hoistable_nodes` excludes them
(`in_degree > 0 || continue`), so the stock loop re-ran `forward_and_latent_grads` for them every
step — and its `in_degree==0` branch reallocates `copy(z_latent) + zero(error)` for the whole
`(B, features)` clamped input each step (J-07: ~48 MB/call for the 256×784 MNIST input, the single
largest byte allocator). Computing them ONCE and skipping is bit-identical (their state never
changes) and removes that reallocation.
"""
function _static_sources(clamps::AbstractDict, structure::GraphStructure)
    static = Set{String}()
    for name in structure.node_names
        info = structure.infos[name]
        info.in_degree == 0 && haskey(clamps, name) && push!(static, name)
    end
    return static
end

"""
Phase-2 forward+accumulate with HOIST+PRUNE. `hoisted` maps each hoistable node to its
loop-invariant forwarded `NodeState` (source of `z_mu`); `hoist` is that
node set. Bit-identical to `forward_value_and_grad` on all consumed fields (see the block
comment above): hoisted nodes reuse cached `z_mu`/`pre` and recompute `error`/`energy`/
`self_grad` from the current z_latent (skipping their all-pruned back-edges); every other
node runs the stock `forward_and_latent_grads` but skips accumulation WRITES into clamped
targets (self-grad into a clamped node's own accumulator, and back-edge grads into clamped
sources) — the values `update_latents` would discard.
"""
function _forward_value_and_grad_opt(
    params::GraphParams,
    state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure,
    hoist::AbstractSet,
    hoisted::AbstractDict,
    static::AbstractSet=Set{String}()
)
    for name in structure.node_names
        node = structure.nodes[name]
        info = structure.infos[name]

        # STATIC (clamped terminal source): its full state was computed once before the loop and is
        # loop-invariant — skip re-running its in_degree==0 forward (J-07: the ~48 MB/step
        # reallocation of `copy(z_latent) + zero(error)` for the clamped input). `zero_grads`
        # already re-zeroed its (never-read) latent_grad; z_mu/error/energy are unchanged.
        name in static && continue

        if name in hoist
            # HOIST: reuse cached pre/z_mu; recompute the z_latent-dependent parts only.
            cached = hoisted[name]
            ns = state.nodes[name]
            err = ns.z_latent .- cached.z_mu
            ns = update_state(
                ns; z_mu=cached.z_mu, error=err
            )
            ns = energy_functional(node, ns)
            self_grad = grad_latent(node.energy, ns.z_latent, ns.z_mu)
            self_grad = scale_self_grad(self_grad, info.scaling_config)
            # Hoistable ⇒ unclamped, so its own self-grad accumulation is LIVE.
            ns = update_state(ns; latent_grad=ns.latent_grad .+ self_grad)
            state = put_node(state, name, ns)
            # PRUNE: all in-edge sources clamped ⇒ every back-edge grad is dead ⇒ skip entirely.
            continue
        end

        in_data = gather_inputs(info, structure, state)
        scaled_inputs = scale_inputs(in_data, info.scaling_config)

        node_state, inedge_grads, self_grad = forward_and_latent_grads(
            node,
            params.nodes[name],
            scaled_inputs,
            state.nodes[name],
            info,
            haskey(clamps, name)
        )

        inedge_grads = scale_input_grads(inedge_grads, info.scaling_config)
        self_grad = scale_self_grad(self_grad, info.scaling_config)
        # PRUNE: a clamped node's own self-grad accumulation is never read — skip the write.
        if !haskey(clamps, name)
            node_state = update_state(
                node_state; latent_grad=node_state.latent_grad .+ self_grad
            )
        end
        state = put_node(state, name, node_state)

        for (edge_key, grad) in inedge_grads
            src = structure.edges[edge_key].source
            # PRUNE: a back-edge grad flowing into a clamped source is never read — skip the write.
            haskey(clamps, src) && continue
            state = update_node_in_state(
                state, src; latent_grad=state.nodes[src].latent_grad .+ grad
            )
        end
    end
    return state
end

"""Optimized generic loop (`InferenceSGD`/`InferenceSGDNormClip`): compute the hoistable nodes'
loop-invariant forward ONCE, then run `infer_steps` of zero_grads → `_forward_value_and_grad_opt`
→ `update_latents` (Phase 3 is unchanged; it already skips clamped nodes). See `run_inference`'s
`optimize` kwarg and the HOIST+PRUNE block comment for the bit-identity argument."""
function _run_inference_loop_opt(
    inf,
    params::GraphParams,
    initial_state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    hoist = _hoistable_nodes(clamps, structure)
    static = _static_sources(clamps, structure)
    state = initial_state
    # Hoisted forward computed ONCE from the clamp-set initial state. Each hoistable node's
    # inputs are all clamped ⇒ constant for the whole loop ⇒ this pre/z_mu is exactly what the
    # stock loop would recompute on every step.
    hoisted = Dict{String, NodeState}()
    for name in hoist
        info = structure.infos[name]
        node = structure.nodes[name]
        in_data = gather_inputs(info, structure, state)
        scaled_inputs = scale_inputs(in_data, info.scaling_config)
        ns_fwd = forward(node, params.nodes[name], scaled_inputs, state.nodes[name])
        hoisted[name] = ns_fwd
    end
    # Static (clamped-terminal-source) forward computed ONCE — bit-identical to what every step's
    # stock `forward_and_latent_grads` in_degree==0 branch would produce, then skipped in the loop.
    for name in static
        node = structure.nodes[name]
        ns_fwd, _, _ = forward_and_latent_grads(
            node, params.nodes[name], Dict{String, Any}(), state.nodes[name],
            structure.infos[name], true
        )
        state = put_node(state, name, ns_fwd)
    end
    for _ in 1:inf.infer_steps
        state = zero_grads(state, structure)
        state = _forward_value_and_grad_opt(
            params, state, clamps, structure, hoist, hoisted, static
        )
        state = update_latents(state, clamps, structure)
    end
    return state
end

"""
    momentum_inference_step(inf::InferenceSGDMomentum, params, state, velocity, clamps, structure)
        -> (new_state, new_velocity)

One heavy-ball step: reuses the shared `zero_grads`/`forward_value_and_grad` phases
(bit-identical to `inference_step`'s own Phase 1/2), then applies the momentum update directly,
threading a per-node velocity `Dict` through the CALLER (not internal loop state) — this is the
single source of truth for the momentum recursion; `_run_inference_loop` below is a thin driver
over repeated calls to this function, and any other caller that needs step-by-step access
(diagnostics, a future `@trace while`-based compiled loop, tests) MUST go through this function
too, NOT `inference_step`/`update_latents`/`compute_new_latent` — those dispatch to this type's
plain-SGD FALLBACK (see its docstring), which silently ignores `momentum` entirely. (A version of
`test/test_inference_momentum.jl` called `inference_step` directly by mistake and got
bit-identical trajectories for momentum=0.9 and momentum=0.99 at the same eta — the fallback's
silent momentum-blindness, not saturation. Fixed by routing every step-by-step caller through
this function instead.)

At `momentum == 0f0`, `v(t+1) = 0·v(t) - eta·grad = -eta·grad` exactly (IEEE 754: `0f0 * finite =
0f0` in every case that can arise here — `v(t)` is `NaN`/`Inf`-free during a converging inference
run, and startup `v(0) = zero(z_latent)`), giving `new_z = z·(1 - eta·latent_decay) - eta·grad` —
bit-for-bit `InferenceSGD.compute_new_latent`'s formula. This is the momentum=0 conformance
anchor asserted in `test/test_inference_momentum.jl` (via `run_inference`, which correctly
dispatches through this function).
"""
function momentum_inference_step(
    inf::InferenceSGDMomentum,
    params::GraphParams,
    state::GraphState,
    velocity::AbstractDict,
    clamps::AbstractDict,
    structure::GraphStructure
)
    state = zero_grads(state, structure)
    state = forward_value_and_grad(params, state, clamps, structure)
    new_velocity = copy(velocity)
    for name in structure.node_names
        if !haskey(clamps, name)
            ns = state.nodes[name]
            v = inf.momentum .* velocity[name] .- inf.eta_infer .* ns.latent_grad
            new_velocity[name] = v
            new_z = ns.z_latent .* (1.0f0 - inf.eta_infer * inf.latent_decay) .+ v
            state = update_node_in_state(state, name; z_latent=new_z)
        end
    end
    return state, new_velocity
end

"""Momentum loop: `momentum_inference_step` `infer_steps` times, carrying velocity across steps
(velocity is internal loop state here — NOT dispatched through `compute_new_latent`/
`update_latents`, which is the generic path used by algorithms with no cross-step state)."""
function _run_inference_loop(
    inf::InferenceSGDMomentum,
    params::GraphParams,
    initial_state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    velocity = Dict{String, Any}(
        name => zero(initial_state.nodes[name].z_latent) for name in structure.node_names
    )
    state = initial_state
    for _ in 1:inf.infer_steps
        state, velocity = momentum_inference_step(
            inf, params, state, velocity, clamps, structure
        )
    end
    return state
end
