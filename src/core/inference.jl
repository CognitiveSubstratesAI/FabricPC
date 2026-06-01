# Inference dynamics. Port of fabricpc/core/inference.py.
#
# The inner predictive-coding loop: iterate `infer_steps` times, each step
#   1. zero latent gradients
#   2. forward pass → predictions (z_mu), errors, energy; accumulate dE/dz_latent
#   3. update each non-clamped latent: z -= eta · grad   (SGD)
#
# muPC scaling is applied at the call-sites here (identity in v0), keeping the
# node methods scaling-unaware. v0 ships InferenceSGD only.

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

"""Phase 3: SGD latent update on non-clamped nodes. Port of `update_latents` + `InferenceSGD.compute_new_latent`."""
function update_latents(
    state::GraphState,
    clamps::AbstractDict,
    structure::GraphStructure
)
    inf = structure.config.inference
    decay = 1.0f0 - inf.eta_infer * inf.latent_decay
    for name in structure.node_names
        if !haskey(clamps, name)
            ns = state.nodes[name]
            new_z = ns.z_latent .* decay .- inf.eta_infer .* ns.latent_grad
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
