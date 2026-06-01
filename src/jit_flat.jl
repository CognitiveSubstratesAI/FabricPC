# Traceable (Dict-free) inference path — foundation for the Reactant/XLA JIT.
#
# The eager path stores state as Dict{String,NodeState} / Dict{String,Matrix} and
# rebuilds Dicts each step (`merge`), which XLA cannot trace. This file lowers a
# GraphStructure to a STATIC integer plan and re-expresses `run_inference` over a
# POSITION-INDEXED container (no Dicts, no string-keyed lookups in the loop), with
# Dict-free node-forward variants. NodeState/NodeParams are already structs of
# arrays (traceable); only the outer containers change.
#
# Increment 1 (this file): the inference hot loop, validated == the eager Dict
# path. Increment 2 swaps the Vector for an NTuple and wraps `flat_run_inference`
# in Reactant.@compile (a weakdep/extension). See docs/decisions.md §11.

"""
    CompiledPlan(structure)

Static lowering of a `GraphStructure` for the Dict-free path. Node `position` =
insertion order (the order the eager loops iterate). Per position it stores the
node, its `NodeInfo`, and its in-edges as parallel vectors of (source position,
slot). `order_pos` is the topological order (positions) for feedforward init.
"""
struct CompiledPlan
    names::Vector{String}
    nodes::Vector{AbstractNode}
    infos::Vector{NodeInfo}
    in_src::Vector{Vector{Int}}      # per position: source positions of its in-edges
    in_slot::Vector{Vector{String}}  # per position: slot of each in-edge
    in_key::Vector{Vector{String}}   # per position: edge key of each in-edge
    order_pos::Vector{Int}           # topological order, as positions
    inference::Any
end

function CompiledPlan(structure::GraphStructure)
    names = structure.node_names
    posof = Dict(n => i for (i, n) in enumerate(names))
    nodes = AbstractNode[structure.nodes[n] for n in names]
    infos = NodeInfo[structure.infos[n] for n in names]
    in_src = Vector{Int}[]
    in_slot = Vector{String}[]
    in_key = Vector{String}[]
    for n in names
        info = structure.infos[n]
        push!(in_src, [posof[structure.edges[k].source] for k in info.in_edges])
        push!(in_slot, [structure.edges[k].slot for k in info.in_edges])
        push!(in_key, collect(info.in_edges))
    end
    order_pos = [posof[n] for n in structure.node_order]
    return CompiledPlan(
        names, nodes, infos, in_src, in_slot, in_key, order_pos, structure.config.inference
    )
end

# ── flat params: per position, weights aligned to in-edges + optional bias ──────

"""Per-node parameters in positional (in-edge-aligned) form: `w[k]` is the weight
for the k-th in-edge (or `nothing` for weightless edges, e.g. skip/identity)."""
struct FlatNodeParams
    w::Vector{Union{Nothing, Matrix{Float32}}}
    b::Union{Nothing, Matrix{Float32}}
end

function to_flat_params(plan::CompiledPlan, params::GraphParams)
    out = FlatNodeParams[]
    for (i, name) in enumerate(plan.names)
        np = params.nodes[name]
        w = Union{Nothing, Matrix{Float32}}[
            get(np.weights, k, nothing) for k in plan.in_key[i]
        ]
        b = get(np.biases, "b", nothing)
        push!(out, FlatNodeParams(w, b))
    end
    return out
end

to_flat_state(plan::CompiledPlan, state::GraphState) =
    NodeState[state.nodes[n] for n in plan.names]

# ── Dict-free node forward (positional inputs/weights) ──────────────────────────
# `ins` / `slots` are aligned to the node's in-edges (plan.in_src / plan.in_slot);
# `fp.w` is aligned likewise. Each mirrors the corresponding eager `forward`.

function flat_forward(node::Linear, fp::FlatNodeParams, ins, slots, state::NodeState)
    batch = size(state.z_latent, 1)
    pre = zeros(Float32, batch, node.shape[end])
    for k in eachindex(ins)
        pre = pre .+ ins[k] * fp.w[k]
    end
    fp.b !== nothing && length(fp.b) > 0 && (pre = pre .+ fp.b)
    z_mu = forward(node.activation, pre)
    ns = update_state(state; pre_activation=pre, z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns)
end

function flat_forward(node::IdentityNode, ::FlatNodeParams, ins, slots, state::NodeState)
    z_mu = nothing
    for x in ins
        z_mu = z_mu === nothing ? x : z_mu .+ x
    end
    z_mu = z_mu .* node.scale
    ns = update_state(state; pre_activation=z_mu, z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns)
end

function flat_forward(node::SkipConnection, ::FlatNodeParams, ins, slots, state::NodeState)
    z_mu = nothing
    for x in ins
        z_mu = z_mu === nothing ? x : z_mu .+ x
    end
    ns = update_state(state; pre_activation=z_mu, z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns)
end

function flat_forward(
    node::LinearResidual, fp::FlatNodeParams, ins, slots, state::NodeState
)
    batch = size(state.z_latent, 1)
    pre = zeros(Float32, batch, node.shape[end])
    skip = nothing
    for k in eachindex(ins)
        if slots[k] == "in"
            pre = pre .+ ins[k] * fp.w[k]
        else
            skip = skip === nothing ? ins[k] : skip .+ ins[k]
        end
    end
    fp.b !== nothing && length(fp.b) > 0 && (pre = pre .+ fp.b)
    transformed = forward(node.activation, pre)
    z_mu = skip === nothing ? transformed : transformed .+ skip
    ns = update_state(state; pre_activation=pre, z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns)
end

# ── Dict-free forward + latent gradients ────────────────────────────────────────
# Returns (NodeState, input_grads::Vector aligned to in-edges, self_grad).

function flat_latent_grads(node::AbstractNode, fp, ins, slots, state, info, is_clamped)
    if info.in_degree == 0
        ns = update_state(
            state; z_mu=copy(state.z_latent),
            error=zero(state.error), pre_activation=zero(state.pre_activation)
        )
        ns = energy_functional(node, ns)
        return ns, Any[], zero(state.latent_grad)
    elseif info.out_degree == 0 && !is_clamped
        ns = flat_forward(node, fp, ins, slots, state)
        ns = update_state(
            ns; z_latent=ns.z_mu, error=zero(ns.error),
            energy=zero(ns.energy), latent_grad=zero(ns.latent_grad)
        )
        return ns, Any[zero(ins[k]) for k in eachindex(ins)], zero(state.z_latent)
    else
        ns = flat_forward(node, fp, ins, slots, state)
        self_grad = grad_latent(node.energy, ns.z_latent, ns.z_mu)
        igrads = _flat_input_grads(node, fp, ins, slots, ns)
        return ns, igrads, self_grad
    end
end

# Per-node input gradients (positional), reusing the shared pre_grad / mu_grad.
function _flat_input_grads(node::Linear, fp, ins, slots, ns)
    dpre = pre_grad(node, ns)
    return Any[dpre * transpose(fp.w[k]) for k in eachindex(ins)]
end
function _flat_input_grads(node::IdentityNode, fp, ins, slots, ns)
    dmu = mu_grad(node, ns)
    return Any[node.scale .* dmu for _ in eachindex(ins)]
end
function _flat_input_grads(node::SkipConnection, fp, ins, slots, ns)
    dmu = mu_grad(node, ns)
    return Any[dmu for _ in eachindex(ins)]
end
function _flat_input_grads(node::LinearResidual, fp, ins, slots, ns)
    dpre = pre_grad(node, ns)
    dmu = mu_grad(node, ns)
    return Any[slots[k] == "in" ? dpre * transpose(fp.w[k]) : dmu for k in eachindex(ins)]
end

# ── flat inference loop (reproduces run_inference, position-indexed) ─────────────

function flat_inference_step(
    plan, fparams, fstate::Vector{NodeState}, clamped::Vector{Bool}
)
    n = length(fstate)
    # Phase 1: zero latent grads.
    fstate = NodeState[update_state(s; latent_grad=zero(s.latent_grad)) for s in fstate]
    # Phase 2: forward + accumulate latent grads (insertion order).
    for i in 1:n
        ins = [fstate[s].z_latent for s in plan.in_src[i]]
        ns, igrads, self_grad = flat_latent_grads(
            plan.nodes[i],
            fparams[i],
            ins,
            plan.in_slot[i],
            fstate[i],
            plan.infos[i],
            clamped[i]
        )
        fstate[i] = update_state(ns; latent_grad=ns.latent_grad .+ self_grad)
        for (k, src) in enumerate(plan.in_src[i])
            fstate[src] = update_state(
                fstate[src]; latent_grad=fstate[src].latent_grad .+ igrads[k]
            )
        end
    end
    # Phase 3: SGD latent update on non-clamped nodes.
    inf = plan.inference
    decay = 1.0f0 - inf.eta_infer * inf.latent_decay
    for i in 1:n
        if !clamped[i]
            s = fstate[i]
            fstate[i] = update_state(
                s; z_latent=s.z_latent .* decay .- inf.eta_infer .* s.latent_grad
            )
        end
    end
    return fstate
end

"""
    flat_run_inference(plan, fparams, fstate, clamped) -> Vector{NodeState}

Dict-free `run_inference`: `infer_steps` flat inference steps. `clamped[i]` marks
clamped positions. Reproduces the eager loop exactly (validated in test_jit_flat).
"""
function flat_run_inference(plan, fparams, fstate::Vector{NodeState}, clamped::Vector{Bool})
    for _ in 1:plan.inference.infer_steps
        fstate = flat_inference_step(plan, fparams, fstate, clamped)
    end
    return fstate
end
