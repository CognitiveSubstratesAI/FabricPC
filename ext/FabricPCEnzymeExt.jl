# Enzyme reverse-mode autodiff extension for FabricPC (opt-in: loaded when
# `using Enzyme`). Implements the Phase-D autodiff node seam declared in
# src/nodes/autodiff.jl — the Julia analog of upstream's base-class
# `jax.value_and_grad(forward)` (fabricpc/nodes/base.py).
#
# Differentiates `FabricPC.energy_kernel` (→ `compute_mu`, concrete arrays only):
#   • weight grads  — ∂/∂params           (Duplicated over the whole NodeParams)
#   • latent grads  — ∂/∂(inputs, z_latent)
# Both use `set_runtime_activity(Reverse)`: the Dict iteration over input edges
# mixes active/constant entries, which Enzyme's static activity analysis cannot
# resolve (EnzymeRuntimeActivityError) — runtime activity restores correctness at
# a small perf cost. Validated against the closed-form Linear oracle to ~1e-7
# (test_autodiff_seam.jl). This is local PC, NOT backprop: each call differentiates
# one node's local energy only — never through the network or the inference loop.

module FabricPCEnzymeExt

using FabricPC
using FabricPC: AbstractNode, NodeParams, SoA, energy_kernel, EnzymeBackend,
    TransformerBlock, FlatNodeParams, NodeState, _tb_block_flat, forward, energy
import FabricPC: _ad_param_grads, _ad_latent_grads, _register_ad_backend!,
    _flat_input_grads, _flat_weight_grads
using Enzyme

# __init__(), NOT bare top-level: a module-level statement that mutates a DIFFERENT,
# already-loaded module's global state (FabricPC._AD_BACKEND) only takes effect in the
# ephemeral worker process that PRECOMPILES this extension -- it is never replayed when the
# extension is later loaded from its cached .ji in a real session. __init__() is Julia's
# documented mechanism for exactly this: it runs on EVERY load, precompiled or not. Before
# this fix, F-04's guard was a no-op in any normal (post-precompile) session -- confirmed by
# direct probe, 2026-07-14 (see docs/AUDIT_REGISTER.md F-04 / docs/decisions.md #15 update).
# Backend-TYPE dispatch (same date, same reopened F-04): this extension's real
# `_ad_param_grads`/`_ad_latent_grads` methods below are keyed on `::EnzymeBackend` as their
# FIRST argument, a distinct type from FabricPCZygoteExt's `::ZygoteBackend` -- the two
# extensions' methods coexist in the method table with no possibility of one overwriting the
# other; `_AD_BACKEND[]` (set here) only selects WHICH one the unwrap-and-dispatch entry
# point in src/nodes/autodiff.jl routes to.
function __init__()
    _register_ad_backend!(EnzymeBackend())
end

# The params weights/biases (and, for latent grads, the inputs) are threaded as `NamedTuple`s — the
# `SoA` container's `.nt` — so Enzyme accumulates gradients into NamedTuples, never a `Dict` (a Dict
# shadow trips `addToDiffe: "unhandled accumulate with partial sizes"` on its i64 hash internals once
# a node's `compute_mu` has a multi-head map). `node`/`inputs`/`z` are passed as explicit `Const`
# args (NOT closure captures — a captured mutable Dict triggers EnzymeMutabilityException).

# differentiated kernels: rebuild the SoA from the differentiated NamedTuple, then call energy_kernel.
_ek_w(node, wnt, bnt, inputs, z) =
    energy_kernel(node, NodeParams(SoA(wnt), SoA(bnt)), inputs, z)
_ek_in(node, params, innt, z) = energy_kernel(node, params, SoA(innt), z)

"""Activity marker for a param NamedTuple. An EMPTY NamedTuple is a GHOST type (zero-size), and
Enzyme rejects `Duplicated` on one outright: *"Type of ghost or constant type
Duplicated{@NamedTuple{}} is marked as differentiable."* Reproducible with no FabricPC involved:

    Enzyme.autodiff(set_runtime_activity(Reverse), a->1.0f0, Active,
                    Duplicated(NamedTuple(), NamedTuple()))   # same error

Two shapes produce an empty NamedTuple, and BOTH are real: every `PoolNode` (parameter-free by
construction) and `ConvNode(...; use_bias=false)` (empty biases). Since `core/learning.jl` calls
`forward_and_weight_grads` for every `in_degree>0` node, an unguarded `Duplicated` means NO graph
containing a pool can take a training step on the Enzyme backend.

Found by the C-01 adversarial audit, which also caught why C-14 was closed too early: that
verification only exercised ConvNode WITH bias — the one shape that cannot produce an empty
NamedTuple. `map(zero, ::@NamedTuple{})` returns `@NamedTuple{}`, so the `SoA(...)` repack and
`core/learning.jl`'s F-03 key-set check are unaffected."""
_act(nt, dnt) = isempty(nt) ? Const(nt) : Duplicated(nt, dnt)

function _ad_param_grads(
    ::EnzymeBackend, node::AbstractNode, params::NodeParams, inputs, z_latent
)
    wnt = params.weights.nt
    bnt = params.biases.nt
    dwnt = map(zero, wnt)
    dbnt = map(zero, bnt)
    # `_act`, not a bare Duplicated: an empty NamedTuple is a ghost type Enzyme refuses to mark
    # differentiable (see above). A param-free node's gradient is correctly the empty NamedTuple.
    Enzyme.autodiff(
        set_runtime_activity(Reverse), _ek_w, Active,
        Const(node), _act(wnt, dwnt), _act(bnt, dbnt), Const(inputs),
        Const(z_latent)
    )
    return NodeParams(SoA(dwnt), SoA(dbnt))
end

function _ad_latent_grads(
    ::EnzymeBackend, node::AbstractNode, params::NodeParams, inputs, z_latent
)
    innt = SoA(inputs).nt                         # Dict inputs (eager) → NamedTuple for Enzyme
    dinnt = map(zero, innt)
    dz = zero(z_latent)
    Enzyme.autodiff(
        set_runtime_activity(Reverse), _ek_in, Active,
        Const(node), Const(params), Duplicated(innt, dinnt), Duplicated(z_latent, dz)
    )
    N = ndims(first(values(inputs)))
    dinputs = Dict{String, Array{Float32, N}}(String(k) => dinnt[k] for k in keys(dinnt))
    return dinputs, dz
end

# ── Flat-lane TransformerBlock gradients (J-02b): positional Enzyme grads for the compiled lane ───
# The dense flat nodes (Linear/Identity/Skip/LinearResidual) have closed-form `_flat_input_grads`/
# `_flat_weight_grads` (src/jit_flat.jl); TransformerBlock has none — its local PC gradients need
# autodiff. These positional twins of the eager Enzyme seam differentiate the SAME node energy the
# flat forward produces (`0.5·prec·Σ(z_latent − z_mu)²`, z_mu = the activated `_tb_block_flat`), so
# the whole flat inference/train step composes under `Reactant.@compile` (verified: Enzyme.autodiff
# COMPOSES inside `@trace track_numbers=false for`, max|Δ|=4.77e-7 vs eager). This is still LOCAL PC:
# each call differentiates ONE block's local energy, never the network or the inference loop.

# Scalar local energy of a TransformerBlock as a function of its input `x` and weight tuple `args`
# (`fp.w[1]` = flat_block_args), matching `flat_forward(::TransformerBlock)`: activation ∘ block,
# then the node's own `energy` summed to a scalar for Enzyme's `Active` return. `z_latent` const.
function _tb_energy_scalar(x, z_latent, node::TransformerBlock, args)
    z_mu = forward(
        node.activation,
        _tb_block_flat(x, args..., Val(node.num_heads), Val(size(x, 1)),
            Val(node.use_rope), Val(node.causal)),
    )
    return sum(energy(node.energy, z_latent, z_mu))
end

# ∂E/∂x — the input grad that flows back to the transformer's source (E-step). x Duplicated, all
# else Const (so no self-grad contamination — self_grad = grad_latent is added separately upstream).
function _flat_input_grads(node::TransformerBlock, fp::FlatNodeParams, ins, slots, ns::NodeState, _pre)
    x = only(ins)
    args = fp.w[1]
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(
        set_runtime_activity(Reverse), _tb_energy_scalar, Active,
        Duplicated(x, dx), Const(ns.z_latent), Const(node), Const(args),
    )
    return Any[dx]                                  # aligned to the single "in" edge
end

# ∂E/∂weights for the whole flat_block_args tuple (M-step). x Const, the weight tuple Duplicated.
# Returns the grad in the SAME stashed-tuple shape `fp.w[1]` holds, so `flat_sgd_step`'s
# TransformerBlock branch can apply it element-wise. muPC `a`-scaling (transformer.jl
# forward_and_weight_grads) is v0-off (scaling_config===nothing) and out of the flat lane's scope.
function _flat_weight_grads(node::TransformerBlock, fp::FlatNodeParams, ins, slots, state::NodeState)
    x = only(ins)
    args = fp.w[1]
    dargs = map(Enzyme.make_zero, args)
    Enzyme.autodiff(
        set_runtime_activity(Reverse), _tb_energy_scalar, Active,
        Const(x), Const(state.z_latent), Const(node), Duplicated(args, dargs),
    )
    return Any[dargs], nothing                      # (dw aligned to fp.w — the stash — , no separate bias)
end

end # module FabricPCEnzymeExt
