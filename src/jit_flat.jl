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
for the k-th in-edge (or `nothing` for weightless edges, e.g. skip/identity).

PARAMETRIC on the field types so the same struct can hold eager `Matrix`es OR
Reactant tracer arrays — concrete `Matrix`-typed fields break Reactant
reconstruction (it substitutes `ConcretePJRTArray`s for the leaves)."""
struct FlatNodeParams{W, B}
    w::W
    b::B
end

"""
    to_flat_params(plan::CompiledPlan, params::GraphParams) -> Vector{FlatNodeParams}

Bridge `GraphParams` (Dict-of-Dicts) to position-indexed `FlatNodeParams`, one per
`plan.names` entry. Eager-only (builds `Union{Nothing,Matrix{Float32}}` vectors — not itself
traceable; run before `Reactant.@compile`, not inside it — see [`flatten_param_arrays`](@ref)/
[`repack_params`](@ref) for the traced bridge).

SCOPE: the flat lane is RANK-2 (dense) only — `w` below is `Matrix{Float32}`. That is the lane's
long-standing scope (J-03: Linear/Identity/Skip/LinearResidual; the transformer flat backward is
J-01/J-02 architecturally-blocked), NOT an oversight: C-01's ConvNode/MaxPool/AvgPool ship on the
EAGER seam, which was the claim, and were never claimed for this lane. But scope has to FAIL
CLOSED to be a scope: without `_flat_supported` below, a rank-4 conv kernel died on
`MethodError: Matrix{Float32}(::Array{Float32,4})` — a divergence that happens to crash, with a
message naming neither the node nor the lane (decisions.md §29: match the property; do not leave a
divergence defended by the fact that it happens to blow up). Found by the C-01 adversarial audit.
"""
# Rank-2/dense only — see the scope note in `to_flat_params`. Fail-CLOSED default.
_flat_supported(::AbstractNode) = true          # dense nodes + TransformerBlock's own bridge
_flat_supported(::ConvNode) = false
_flat_supported(::PoolNode) = false

function to_flat_params(plan::CompiledPlan, params::GraphParams)
    for (i, name) in enumerate(plan.names)
        _flat_supported(plan.nodes[i]) || error(
            "to_flat_params: node '$name' ($(typeof(plan.nodes[i]))) is outside the FLAT lane's " *
            "scope. The flat/JIT lane is rank-2 (dense) only — Linear/IdentityNode/" *
            "SkipConnection/LinearResidual (+ TransformerBlock via its own bridge). Conv/pool " *
            "nodes are rank-3+ and ship on the EAGER seam instead: use `run_inference` (or " *
            "`run_inference(...; optimize=true)`), not `flat_run_inference`/`prealloc_inference`."
        )
    end
    out = FlatNodeParams[]
    for (i, name) in enumerate(plan.names)
        np = params.nodes[name]
        if plan.nodes[i] isa TransformerBlock
            # TransformerBlock's weights are keyed by NAME ("W_q", "ln1_gamma", ...), not by
            # edge key like every other node here — the generic per-in-edge lookup below
            # doesn't apply. Stash the pre-bridged `flat_block_args` tuple as the sole `w`
            # entry; `flat_forward(::TransformerBlock, ...)` unpacks it directly.
            push!(out, FlatNodeParams(Any[flat_block_args(plan.nodes[i], np)], nothing))
        else
            w = Union{Nothing, Matrix{Float32}}[
                get(np.weights, k, nothing) for k in plan.in_key[i]
            ]
            b = get(np.biases, "b", nothing)
            push!(out, FlatNodeParams(w, b))
        end
    end
    return out
end

"""
    to_flat_state(plan::CompiledPlan, state::GraphState) -> Vector{NodeState}

Bridge `GraphState` (a `Dict{String,NodeState}`) to a position-indexed `Vector{NodeState}`,
ordered to match `plan.names`.
"""
to_flat_state(plan::CompiledPlan, state::GraphState) =
    [state.nodes[n] for n in plan.names]

# ── Reactant bridge: flatten params to a plain array tuple (+ static layout),
#    repack inside the traced region, rebuild states from z_latents. The
#    @compile'd function takes only Tuples of arrays — never concrete-typed
#    structs (which break Reactant reconstruction). ────────────────────────────

"""Flatten per-node params to a flat `Tuple` of arrays + a static layout that
`repack_params` uses to rebuild them. The tuple is what `Reactant.@compile`
traces; the layout (integer indices) is a compile-time constant."""
function flatten_param_arrays(fparams::Vector{<:FlatNodeParams})
    arrs = Any[]
    layout = Tuple{Vector{Int}, Int}[]
    for fp in fparams
        widx = Int[]
        for w in fp.w
            if w === nothing
                push!(widx, 0)
            else
                push!(arrs, w)
                push!(widx, length(arrs))
            end
        end
        bidx = 0
        if fp.b !== nothing
            push!(arrs, fp.b)
            bidx = length(arrs)
        end
        push!(layout, (widx, bidx))
    end
    return Tuple(arrs), layout
end

"""Rebuild `Vector{FlatNodeParams}` from a flat array tuple + layout. Built inside
the traced region, so the (parametric) FlatNodeParams hold tracer arrays."""
repack_params(arr_tuple, layout) = [
    FlatNodeParams(
        Any[wi == 0 ? nothing : arr_tuple[wi] for wi in widx],
        bidx == 0 ? nothing : arr_tuple[bidx]
    ) for (widx, bidx) in layout
]

"""Build a `Vector{NodeState}` from per-node z_latents; other fields are zero
placeholders (recomputed in inference step 1, so unused — matches the eager init)."""
state_from_latents(zl) = [
    NodeState(
        zl[i], zero(zl[i]), zero(zl[i]),
        zeros(eltype(zl[i]), size(zl[i], 1)), zero(zl[i])
    ) for i in eachindex(zl)
]

"""
    jit_inference_runner(arr_tuple, zl, plan, layout, clamped) -> Tuple of z_latents

Reactant-traceable entry point: takes only Tuples of arrays (params flattened via
`flatten_param_arrays`, plus per-node z_latents), runs `flat_run_inference`, and
returns the new z_latents tuple. `plan`/`layout`/`clamped` are compile-time-static
— capture them in a closure before `Reactant.@compile` (see `FabricPCReactantExt`)."""
function jit_inference_runner(
    arr_tuple, zl, plan::CompiledPlan, layout, clamped::Vector{Bool}
)
    fparams = repack_params(arr_tuple, layout)
    fstate = flat_run_inference(plan, fparams, state_from_latents(zl), clamped)
    return ntuple(i -> fstate[i].z_latent, length(fstate))
end

"""
    compile_inference(structure, params, clamps; batch) -> callable

JIT-compile the predictive-coding inference loop via Reactant/XLA (≈9× faster
than eager on the MNIST-shaped MLP). The returned object maps
`(params, init_state) -> Vector` of converged per-node `z_latent` arrays (node
order = `structure.node_names`). The compiled thunk is specialized to the param
shapes, `batch`, and which nodes are clamped — recompile if those change.

Requires `using Reactant` (implemented by the `FabricPCReactantExt` extension).
"""
function compile_inference end
compile_inference(args...; kwargs...) =
    error(
        "compile_inference requires Reactant — run `using Reactant` to load FabricPCReactantExt"
    )

# ── Dict-free node forward (positional inputs/weights) ──────────────────────────
# `ins` / `slots` are aligned to the node's in-edges (plan.in_src / plan.in_slot);
# `fp.w` is aligned likewise. Each mirrors the corresponding eager `forward`.

function _flat_fwd_pre(node::Linear, fp::FlatNodeParams, ins, slots, state::NodeState)
    batch = size(state.z_latent, 1)
    pre = zeros(Float32, batch, node.shape[end])
    for k in eachindex(ins)
        pre = pre .+ ins[k] * fp.w[k]
    end
    fp.b !== nothing && length(fp.b) > 0 && (pre = pre .+ fp.b)
    z_mu = forward(node.activation, pre)
    ns = update_state(state; z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns), pre
end

flat_forward(node::Linear, fp::FlatNodeParams, ins, slots, state::NodeState) =
    first(_flat_fwd_pre(node, fp, ins, slots, state))

function flat_forward(node::IdentityNode, ::FlatNodeParams, ins, slots, state::NodeState)
    z_mu = nothing
    for x in ins
        z_mu = z_mu === nothing ? x : z_mu .+ x
    end
    z_mu = z_mu .* node.scale
    ns = update_state(state; z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns)
end

function flat_forward(node::SkipConnection, ::FlatNodeParams, ins, slots, state::NodeState)
    z_mu = nothing
    for x in ins
        z_mu = z_mu === nothing ? x : z_mu .+ x
    end
    ns = update_state(state; z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns)
end

function _flat_fwd_pre(
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
    ns = update_state(state; z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns), pre
end

flat_forward(node::LinearResidual, fp::FlatNodeParams, ins, slots, state::NodeState) =
    first(_flat_fwd_pre(node, fp, ins, slots, state))

# TransformerBlock (J-03, docs/AUDIT_REGISTER.md section 5): wires the already-validated
# `_tb_block_flat` kernel (transformer.jl) into this file's dispatch — FORWARD ONLY. No
# `_flat_input_grads(::TransformerBlock, ...)` method exists (the attention+FFN backward
# pass, deliberately out of scope here — needs either Enzyme-under-Reactant, blocked per
# J-01/decisions.md §15, or a from-scratch hand-derived gradient). `flat_latent_grads`'s
# `out_degree == 0 && !is_clamped` branch (this file, below) calls ONLY `flat_forward` — so
# this works wherever TransformerBlock is the graph's unclamped terminal/output node (e.g.
# JIT-compiled prediction on fixed params). Using it as an interior or clamped node hits a
# MethodError on `_flat_input_grads` — a deliberate, honest failure, not silently wrong math.
#
# DIVERGENCE from `_tb_block_flat` itself: that kernel returns the raw block output with no
# activation applied (matching upstream, which never applies `node_info.activation` either —
# see F-06). Eager `compute_mu` DOES apply `node.activation`. This wrapper matches eager
# `compute_mu` (this file's stated contract — every `flat_forward` "mirrors the corresponding
# eager forward", not upstream), applying `node.activation` here rather than inside the
# shared, separately-benchmarked kernel.
function flat_forward(
    node::TransformerBlock, fp::FlatNodeParams, ins, slots, state::NodeState
)
    x = only(ins)
    args = fp.w[1]
    B = size(x, 1)
    z_mu = forward(
        node.activation,
        _tb_block_flat(
            x, args..., Val(node.num_heads), Val(B), Val(node.use_rope), Val(node.causal)
        )
    )
    ns = update_state(state; z_mu=z_mu, error=state.z_latent .- z_mu)
    return energy_functional(node, ns)
end

# ── Dict-free forward + latent gradients ────────────────────────────────────────
# Returns (NodeState, input_grads::Vector aligned to in-edges, self_grad).

function flat_latent_grads(node::AbstractNode, fp, ins, slots, state, info, is_clamped)
    if info.in_degree == 0
        ns = update_state(state; z_mu=copy(state.z_latent), error=zero(state.error))
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
        ns, pre = _flat_fwd_for_grads(node, fp, ins, slots, state)
        self_grad = grad_latent(node.energy, ns.z_latent, ns.z_mu)
        igrads = _flat_input_grads(node, fp, ins, slots, ns, pre)
        return ns, igrads, self_grad
    end
end

# Forward for the gradient branch, additionally handing back the pre-activation the explicit
# `f'(pre)` path needs — NodeState no longer carries it (upstream b6f64ad). Only Linear and
# LinearResidual have a distinct pre; for everything else it is `nothing`, and each method is
# separately specialized, so `pre` stays concretely typed rather than boxed on this lane.
_flat_fwd_for_grads(node::AbstractNode, fp, ins, slots, state) =
    (flat_forward(node, fp, ins, slots, state), nothing)
_flat_fwd_for_grads(node::Linear, fp, ins, slots, state) =
    _flat_fwd_pre(node, fp, ins, slots, state)
_flat_fwd_for_grads(node::LinearResidual, fp, ins, slots, state) =
    _flat_fwd_pre(node, fp, ins, slots, state)

# Per-node input gradients (positional), reusing the shared pre_grad / mu_grad.
function _flat_input_grads(node::Linear, fp, ins, slots, ns, pre)
    dpre = pre_grad(node, ns, pre)
    return Any[dpre * transpose(fp.w[k]) for k in eachindex(ins)]
end
function _flat_input_grads(node::IdentityNode, fp, ins, slots, ns, _pre)
    dmu = mu_grad(node, ns)
    return Any[node.scale .* dmu for _ in eachindex(ins)]
end
function _flat_input_grads(node::SkipConnection, fp, ins, slots, ns, _pre)
    dmu = mu_grad(node, ns)
    return Any[dmu for _ in eachindex(ins)]
end
function _flat_input_grads(node::LinearResidual, fp, ins, slots, ns, pre)
    dpre = pre_grad(node, ns, pre)
    dmu = mu_grad(node, ns)
    return Any[slots[k] == "in" ? dpre * transpose(fp.w[k]) : dmu for k in eachindex(ins)]
end

# ── flat inference loop (reproduces run_inference, position-indexed) ─────────────

function flat_inference_step(
    plan, fparams, fstate::Vector{<:NodeState}, clamped::Vector{Bool}
)
    n = length(fstate)
    # Phase 1: zero latent grads.
    fstate = [update_state(s; latent_grad=zero(s.latent_grad)) for s in fstate]
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
    flat_run_inference(plan, fparams, fstate, clamped; optimize=false) -> Vector{NodeState}

Dict-free `run_inference`: `infer_steps` flat inference steps. `clamped[i]` marks
clamped positions. Reproduces the eager loop exactly (validated in test_jit_flat).

`optimize=true` selects the positional HOIST+PRUNE loop (`_flat_run_inference_opt`),
the flat-lane twin of `run_inference(...; optimize=true)` in `core/inference.jl` §28 —
same two exact optimizations (hoist a position whose in-sources are all clamped; skip
gradient accumulation into clamped positions), same bit-identity on all consumed fields.
Bundled here to measure the optimization on the DICT-FREE eager path (isolating the
algorithmic win from the Dict/string-key overhead the core eager path carries — see
docs/decisions.md §28's three-way decomposition)."""
function flat_run_inference(
    plan, fparams, fstate::Vector{<:NodeState}, clamped::Vector{Bool}; optimize::Bool=false
)
    optimize && return _flat_run_inference_opt(plan, fparams, fstate, clamped)
    for _ in 1:plan.inference.infer_steps
        fstate = flat_inference_step(plan, fparams, fstate, clamped)
    end
    return fstate
end

"""Positions whose forward is loop-invariant AND take the plain `else` branch of
`flat_latent_grads`: `in_degree > 0`, `out_degree > 0` and unclamped, every in-edge source
clamped. Position-indexed twin of `core/inference.jl`'s `_hoistable_nodes`."""
function _flat_hoistable(plan, clamped::Vector{Bool})
    n = length(plan.nodes)
    hoist = falses(n)
    for i in 1:n
        info = plan.infos[i]
        (info.in_degree > 0 && info.out_degree > 0 && !clamped[i]) || continue
        all(clamped[s] for s in plan.in_src[i]) && (hoist[i] = true)
    end
    return hoist
end

"""HOIST+PRUNE flat step. `hoist[i]` marks hoistable positions; `hoisted[i]` caches their
loop-invariant forwarded `NodeState` (`z_mu`). Bit-identical to
`flat_inference_step` on every consumed field: hoisted positions reuse the cached forward and
recompute only the `z_latent`-dependent `error`/`energy`/`self_grad`; all back-edge grads into
clamped sources and self-grads into clamped positions (the values Phase 3 discards) are skipped."""
function _flat_inference_step_opt(
    plan, fparams, fstate::Vector{<:NodeState}, clamped::Vector{Bool},
    hoist::AbstractVector{Bool}, hoisted::Vector{<:NodeState}
)
    n = length(fstate)
    fstate = [update_state(s; latent_grad=zero(s.latent_grad)) for s in fstate]
    for i in 1:n
        if hoist[i]
            node = plan.nodes[i]
            cached = hoisted[i]
            s = fstate[i]
            err = s.z_latent .- cached.z_mu
            ns = update_state(s; z_mu=cached.z_mu, error=err)
            ns = energy_functional(node, ns)
            self_grad = grad_latent(node.energy, ns.z_latent, ns.z_mu)
            fstate[i] = update_state(ns; latent_grad=ns.latent_grad .+ self_grad)
            continue                                   # all in-sources clamped ⇒ all back-edges pruned
        end
        ins = [fstate[s].z_latent for s in plan.in_src[i]]
        ns, igrads, self_grad = flat_latent_grads(
            plan.nodes[i], fparams[i], ins, plan.in_slot[i], fstate[i], plan.infos[i], clamped[i]
        )
        # PRUNE: a clamped position's own self-grad accumulation is dead — skip it.
        fstate[i] = clamped[i] ? ns : update_state(ns; latent_grad=ns.latent_grad .+ self_grad)
        for (k, src) in enumerate(plan.in_src[i])
            clamped[src] && continue                   # PRUNE: back-edge grad into a clamped source is dead
            fstate[src] = update_state(
                fstate[src]; latent_grad=fstate[src].latent_grad .+ igrads[k]
            )
        end
    end
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

"""Optimized flat loop: compute hoistable positions' loop-invariant forward ONCE, then run
`infer_steps` of `_flat_inference_step_opt`. See `flat_run_inference`'s `optimize` kwarg."""
function _flat_run_inference_opt(plan, fparams, fstate::Vector{<:NodeState}, clamped::Vector{Bool})
    hoist = _flat_hoistable(plan, clamped)
    # Vector{NodeState} (NOT Vector{Any}) — unfilled (non-hoistable) slots stay #undef and are
    # never indexed (only read where hoist[i]); keeping the eltype concrete avoids boxing.
    hoisted = Vector{NodeState}(undef, length(fstate))
    for i in 1:length(fstate)
        hoist[i] || continue
        ins = [fstate[s].z_latent for s in plan.in_src[i]]
        hoisted[i] = flat_forward(plan.nodes[i], fparams[i], ins, plan.in_slot[i], fstate[i])
    end
    for _ in 1:plan.inference.infer_steps
        fstate = _flat_inference_step_opt(plan, fparams, fstate, clamped, hoist, hoisted)
    end
    return fstate
end

# ── Dict-free CLOSED-FORM weight gradients (the M-step, position-indexed) ─────────
# Positional twin of `forward_and_weight_grads` (src/nodes/*.jl) / `compute_local_weight_gradients`
# (core/learning.jl). Returns `(dw, db)` where `dw` is aligned to `fp.w` (`nothing` for weightless
# edges, so it drops straight into the same layout) and `db` is the bias grad or `nothing`. Every
# method here is the SAME closed-form local rule its eager counterpart uses — no autodiff, all
# matmuls — so the whole flat learning step traces under `Reactant.@compile`. SCOPE: dense/rank-2
# and v0 (`scaling_config === nothing`) only, matching the flat lane (J-03); muPC weight-grad
# scaling (`scale_weight_grads`) and the TransformerBlock/Conv weight backward are NOT covered —
# those learn on the eager seam (`compute_local_weight_gradients`).
_flat_weight_grads(node::AbstractNode, fp, ins, slots, state) = error(
    "_flat_weight_grads: no flat weight-grad for $(typeof(node)). The flat learning lane covers " *
    "Linear/IdentityNode/SkipConnection/LinearResidual (dense, v0). TransformerBlock/Conv learn " *
    "on the eager seam — use `train_step`/`compute_local_weight_gradients`, not the compiled lane."
)

# Linear: dWₖ = insₖᵀ·dpre, db = Σ_batch dpre, dpre = (∂E/∂z_mu)·f'(pre). Mirrors linear.jl:188.
function _flat_weight_grads(node::Linear, fp, ins, slots, state)
    ns, pre = _flat_fwd_pre(node, fp, ins, slots, state)
    dpre = pre_grad(node, ns, pre)
    dw = Any[transpose(ins[k]) * dpre for k in eachindex(ins)]
    db = fp.b === nothing ? nothing : sum(dpre; dims=1)
    return dw, db
end

# LinearResidual: only the "in" slot carries a weight; the skip slot has none. Mirrors
# linear_residual.jl:189 (`_is_in_edge(edge_key) || continue`).
function _flat_weight_grads(node::LinearResidual, fp, ins, slots, state)
    ns, pre = _flat_fwd_pre(node, fp, ins, slots, state)
    dpre = pre_grad(node, ns, pre)
    dw = Any[slots[k] == "in" ? transpose(ins[k]) * dpre : nothing for k in eachindex(ins)]
    db = fp.b === nothing ? nothing : sum(dpre; dims=1)
    return dw, db
end

# IdentityNode / SkipConnection are weightless — empty grads (identity.jl:119, skip_connection.jl:104).
_flat_weight_grads(::IdentityNode, fp, ins, slots, state) =
    (Any[nothing for _ in eachindex(ins)], nothing)
_flat_weight_grads(::SkipConnection, fp, ins, slots, state) =
    (Any[nothing for _ in eachindex(ins)], nothing)

"""
    flat_sgd_step(plan, fparams, fstate, clamped, lr) -> Vector{FlatNodeParams}

Local SGD M-step on the SETTLED inference state `fstate`: for each dense node compute its
closed-form local weight grad and apply `w -= lr·dW`, `b -= lr·db`. Position-indexed twin of
`get_graph_param_gradient` + `sgd_update` (core/learning.jl + training/train.jl) — the same
per-node local rule, no backprop through inference. Source nodes (`in_degree == 0`) and weightless
edges pass through unchanged. All arrays ⇒ Reactant-traceable."""
function flat_sgd_step(plan, fparams, fstate::Vector{<:NodeState}, clamped::Vector{Bool}, lr)
    n = length(fparams)
    out = Vector{FlatNodeParams}(undef, n)
    η = Float32(lr)
    for i in 1:n
        fp = fparams[i]
        if plan.infos[i].in_degree == 0
            out[i] = fp                                  # source node: no params
            continue
        end
        ins = [fstate[s].z_latent for s in plan.in_src[i]]
        dw, db = _flat_weight_grads(plan.nodes[i], fp, ins, plan.in_slot[i], fstate[i])
        new_w = Any[
            if fp.w[k] === nothing || dw[k] === nothing
                fp.w[k]
            elseif fp.w[k] isa Tuple            # TransformerBlock stashed weight tuple (fp.w[1])
                map((wi, di) -> wi .- η .* di, fp.w[k], dw[k])
            else
                fp.w[k] .- η .* dw[k]
            end
            for k in eachindex(fp.w)
        ]
        new_b = (fp.b === nothing || db === nothing) ? fp.b : fp.b .- η .* db
        out[i] = FlatNodeParams(new_w, new_b)
    end
    return out
end

"""
    flat_train_step(plan, fparams, fstate_init, clamped, lr) -> Vector{FlatNodeParams}

One full PC training step, Dict-free: settle the latents (`flat_run_inference`, the E-step) then
apply the local SGD weight update (`flat_sgd_step`, the M-step). Position-indexed twin of eager
`train_step` (training/train.jl). Eager reference for the compiled `compile_train_step`; also the
whole thing that gets traced inside `@compile`."""
function flat_train_step(plan, fparams, fstate_init::Vector{<:NodeState}, clamped::Vector{Bool}, lr)
    fstate = flat_run_inference(plan, fparams, fstate_init, clamped)
    return flat_sgd_step(plan, fparams, fstate, clamped, lr)
end

"""
    graph_params_from_flat(plan, fparams) -> GraphParams

Inverse of `to_flat_params`: rebuild `GraphParams` (Dict-of-Dicts) from position-indexed
`FlatNodeParams`, reattaching each node's edge-keyed weights (`fp.w[k]` → `weights[in_key[k]]`, a
`nothing` entry meaning a weightless edge) and its bias (`fp.b` → `biases["b"]`). Eager (runs on
the compiled train step's Array output, OUTSIDE the traced region). Dense/v0 only — the same rank-2
scope as `to_flat_params` (a TransformerBlock's stashed-tuple `w` is not handled)."""
function graph_params_from_flat(plan::CompiledPlan, fparams::Vector{<:FlatNodeParams})
    nodes = Dict{String, NodeParams}()
    for (i, name) in enumerate(plan.names)
        node = plan.nodes[i]
        fp = fparams[i]
        if node isa TransformerBlock
            # TransformerBlock stashes its whole weight set as the tuple `fp.w[1]` (flat_block_args);
            # unstash it back to named weights/biases (J-02b).
            nodes[name] = unflat_block_params(node, fp.w[1])
            continue
        end
        w = Dict{String, Array{Float32}}()
        for (k, key) in enumerate(plan.in_key[i])
            fp.w[k] === nothing || (w[key] = fp.w[k])
        end
        b = Dict{String, Array{Float32}}()
        fp.b === nothing || (b["b"] = fp.b)
        nodes[name] = NodeParams(w, b)
    end
    return GraphParams(nodes)
end

"""
    jit_train_step_runner(arr_tuple, zl, plan, layout, clamped, lr) -> Tuple of arrays

Reactant-traceable entry point for `compile_train_step`: repack params, run `flat_train_step`
(E-step + local SGD M-step), and flatten the UPDATED params back to an array tuple in the SAME
layout as the input (weightless edges stay `nothing`, so the layout is invariant). The caller
rebuilds `GraphParams` from the returned arrays."""
function jit_train_step_runner(
    arr_tuple, zl, plan::CompiledPlan, layout, clamped::Vector{Bool}, lr
)
    fparams = repack_params(arr_tuple, layout)
    new_fparams = flat_train_step(plan, fparams, state_from_latents(zl), clamped, lr)
    return flatten_param_arrays(new_fparams)[1]
end

"""
    compile_train_step(structure, params, clamps; batch, lr, loop) -> callable

JIT-compile the full PC training step (inference E-step + local SGD weight update M-step) via
Reactant/XLA. The returned object maps `(params, init_state) -> GraphParams` (the updated params
after one step). Dense/v0 only (see `flat_train_step`); the learning rate is baked into the
compiled thunk (recompile to change it). Requires `using Reactant` (the `FabricPCReactantExt`
extension)."""
function compile_train_step end
compile_train_step(args...; kwargs...) =
    error(
        "compile_train_step requires Reactant — run `using Reactant` to load FabricPCReactantExt"
    )
