# Reactant/XLA JIT extension for FabricPC (opt-in: loaded when `using Reactant`).
#
# Implements `compile_inference` by tracing the Dict-free `jit_inference_runner`
# (src/jit_flat.jl) with `Reactant.@compile`. The traced function takes only
# Tuples of arrays (params flattened + per-node z_latents); the static topology
# (plan, layout, clamp mask) is captured in a closure. See docs/decisions.md §11.

module FabricPCReactantExt

using FabricPC
using FabricPC:
    CompiledPlan, to_flat_params, flatten_param_arrays, jit_inference_runner,
    GraphParams, GraphStructure, GraphState
import FabricPC: compile_inference
using Reactant

"""A compiled inference thunk + the static data needed to feed/read it.
`params_r` is the params' `ConcreteRArray`s, marshalled ONCE at compile time: weights are
CONSTANT across inference (E-step) calls, so re-marshalling them per call (~400 KB of
host→device copy for the MNIST-MLP) is pure waste. It's held here and reused; only the
per-batch state is marshalled per call.

Cost, MEASURED directly (J-04b, 2026-07-15, medians of 3 runs): re-marshalling the params costs
0.69-1.23 ms/call (~11-18% of the call) and the per-batch state 0.26-0.52 ms. Note that
`decisions.md` §28 addendum 3's slope test attributed ~2.5 ms/call to this marshalling — about
2x too high. A fitted decomposition is a hypothesis; these numbers come from timing the three
variants against each other in one process."""
struct CompiledInference
    plan::CompiledPlan
    layout::Any
    clamped::Vector{Bool}
    thunk::Any
    batch::Int
    params_r::Any     # params' ConcreteRArrays, marshalled once (weights are inference-constant)
end

function compile_inference(
    structure::GraphStructure, params::GraphParams, clamps::AbstractDict; batch::Int
)
    plan = CompiledPlan(structure)
    arr_tuple, layout = flatten_param_arrays(to_flat_params(plan, params))
    clamped = Bool[n in keys(clamps) for n in plan.names]
    # Sample z_latents (zeros of each node's (batch, features) shape) to trace shapes.
    zl = ntuple(i -> zeros(Float32, batch, plan.infos[i].shape...), length(plan.names))

    runner(at, z) = jit_inference_runner(at, z, plan, layout, clamped)
    params_r = Reactant.to_rarray(arr_tuple)                       # marshalled ONCE
    thunk = Reactant.@compile runner(params_r, Reactant.to_rarray(zl))
    return CompiledInference(plan, layout, clamped, thunk, batch, params_r)
end

"""
    (ci::CompiledInference)(init_state) -> Vector{Matrix{Float32}}

Run the compiled inference reusing the compile-time params (the fast path): marshals ONLY the
per-batch state, executes the XLA thunk, returns the converged per-node z_latents (node order =
`ci.plan.names`) as plain Julia arrays. Correct whenever params are unchanged since `compile_inference`
— i.e. every inference/E-step call; after a weight update, recompile (or use the `(params, state)`
method below, which re-marshals)."""
function (ci::CompiledInference)(init_state::GraphState)
    zl = ntuple(i -> init_state.nodes[ci.plan.names[i]].z_latent, length(ci.plan.names))
    out = ci.thunk(ci.params_r, Reactant.to_rarray(zl))
    return [Array(out[i]) for i in eachindex(out)]
end

"""
    (ci::CompiledInference)(params, init_state) -> Vector{Matrix{Float32}}

Re-marshal `params` too (the slow path) — use only when weights changed since compile. For the
inference E-step (constant weights) prefer `ci(init_state)` above, which reuses `ci.params_r`."""
function (ci::CompiledInference)(params::GraphParams, init_state::GraphState)
    arr_tuple, _ = flatten_param_arrays(to_flat_params(ci.plan, params))
    zl = ntuple(i -> init_state.nodes[ci.plan.names[i]].z_latent, length(ci.plan.names))
    out = ci.thunk(Reactant.to_rarray(arr_tuple), Reactant.to_rarray(zl))
    return [Array(out[i]) for i in eachindex(out)]
end

end # module FabricPCReactantExt
