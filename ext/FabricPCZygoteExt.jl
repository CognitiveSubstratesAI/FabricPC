# Zygote reverse-mode AD for the node-local PC gradients — a robust alternative to the Enzyme seam.
# Enzyme aborts on the FULL multi-head-attention block on Julia 1.12 (every isolated gradient —
# weights, biases, input — differentiates fine, but the combined graph trips
# `addToDiffe: "unhandled accumulate with partial sizes"`; proved by 9 bisections). FabricPC's node
# forwards are FUNCTIONAL (no in-place mutation), which is exactly Zygote's sweet spot. We
# differentiate w.r.t. the `SoA` `.nt` NamedTuples (rebuilding SoA/NodeParams inside the closure), so
# the gradient repacks cleanly and no `Dict` enters the differentiated region. Same local-PC
# contract as the Enzyme seam (∂/∂params for learning, ∂/∂(inputs,z) for inference) — NOT backprop
# through the network.
module FabricPCZygoteExt

using FabricPC
using FabricPC: AbstractNode, NodeParams, SoA, energy_kernel, ZygoteBackend,
    _tb_causal_mask, _tb_rope_tables, _tb_varcomp, _soa_key, _soa_keys
import FabricPC: _ad_param_grads, _ad_latent_grads, _register_ad_backend!
using Zygote

# __init__(), NOT bare top-level -- see the identical note in FabricPCEnzymeExt.jl. A
# module-level statement mutating FabricPC._AD_BACKEND only ran once, in this extension's own
# precompile worker process; it was never replayed on a normal load from the .ji cache, so
# F-04's guard was a no-op in real sessions for BOTH backends, not just Enzyme. Backend-TYPE
# dispatch (2026-07-14, docs/AUDIT_REGISTER.md F-04 reopened): this extension's real
# `_ad_param_grads`/`_ad_latent_grads` methods below are keyed on `::ZygoteBackend` as their
# FIRST argument, a distinct type from FabricPCEnzymeExt's `::EnzymeBackend` -- the two
# extensions' methods therefore coexist in the method table with no possibility of one
# overwriting the other; `_AD_BACKEND[]` (set here) only selects WHICH one `_ad_param_grads`/
# `_ad_latent_grads`'s unwrap-and-dispatch entry point routes to.
function __init__()
    _register_ad_backend!(ZygoteBackend())
end

# SoA key strings are structural (independent of the differentiated values); the Symbol->String
# foreigncall (jl_cstr_to_string) is not differentiable, so skip tracing it. This is what lets a
# generic node's compute_mu iterate `inputs`/`params` (an SoA) inside the autodiff'd seam.
Zygote.@nograd _soa_key
Zygote.@nograd _soa_keys

# These helpers build CONSTANT tables (functions of S/head_dim only — no differentiable input) via
# comprehensions, which lower to `setindex!`; Zygote's tracer rejects that ("Mutating arrays not
# supported") even though the result is constant. `@nograd` tells Zygote to skip tracing them (their
# gradient is zero), so the comprehensions never enter the tape.
Zygote.@nograd _tb_causal_mask
Zygote.@nograd _tb_rope_tables
Zygote.@nograd _tb_varcomp

# Zygote returns `nothing` (whole or per-field) for inputs that don't affect the output; coerce to
# zeros matching the reference NamedTuple so repack/optimizer code is uniform.
_fill_zero(::Nothing, ref) = map(zero, ref)
_fill_zero(g::NamedTuple, ref) =
    NamedTuple{keys(ref)}(
        map(k -> getproperty(g, k) === nothing ? zero(ref[k]) : getproperty(g, k),
            keys(ref))
    )

function _ad_param_grads(
    ::ZygoteBackend, node::AbstractNode, params::NodeParams, inputs, z_latent
)
    wnt = params.weights.nt
    bnt = params.biases.nt
    gw, gb = Zygote.gradient(
        (w, b) -> energy_kernel(node, NodeParams(SoA(w), SoA(b)), inputs, z_latent),
        wnt, bnt
    )
    return NodeParams(SoA(_fill_zero(gw, wnt)), SoA(_fill_zero(gb, bnt)))
end

function _ad_latent_grads(
    ::ZygoteBackend, node::AbstractNode, params::NodeParams, inputs, z_latent
)
    innt = SoA(inputs).nt
    gin, gz = Zygote.gradient(
        (inn, z) -> energy_kernel(node, params, SoA(inn), z),
        innt, z_latent
    )
    gin = _fill_zero(gin, innt)
    N = ndims(first(values(inputs)))
    dinputs = Dict{String, Array{Float32, N}}(String(k) => gin[k] for k in keys(gin))
    return dinputs, (gz === nothing ? zero(z_latent) : gz)
end

end # module FabricPCZygoteExt
