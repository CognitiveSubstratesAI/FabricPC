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

"""Gather each in-edge's source `z_latent` into a Dict keyed by edge key. Port of `gather_inputs`."""
function gather_inputs(info::NodeInfo, structure::GraphStructure, state::GraphState)
    in_data = Dict{String, Any}()
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

"""Run the inference loop `infer_steps` times to convergence. Port of `run_inference`."""
function run_inference(
    params::GraphParams,
    initial_state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    state = initial_state
    for _ in 1:structure.config.inference.infer_steps
        state = inference_step(params, state, clamps, structure)
    end
    return state
end
