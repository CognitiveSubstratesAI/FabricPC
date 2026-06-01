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

"""A compiled inference thunk + the static data needed to feed/read it."""
struct CompiledInference
    plan::CompiledPlan
    layout::Any
    clamped::Vector{Bool}
    thunk::Any
    batch::Int
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
    thunk = Reactant.@compile runner(Reactant.to_rarray(arr_tuple), Reactant.to_rarray(zl))
    return CompiledInference(plan, layout, clamped, thunk, batch)
end

"""
    (ci::CompiledInference)(params, init_state) -> Vector{Matrix{Float32}}

Run the compiled inference: converts `params` + `init_state` z_latents to Reactant
arrays, executes the XLA thunk, returns the converged per-node z_latents (node
order = `ci.plan.names`) as plain Julia arrays.
"""
function (ci::CompiledInference)(params::GraphParams, init_state::GraphState)
    arr_tuple, _ = flatten_param_arrays(to_flat_params(ci.plan, params))
    zl = ntuple(i -> init_state.nodes[ci.plan.names[i]].z_latent, length(ci.plan.names))
    out = ci.thunk(Reactant.to_rarray(arr_tuple), Reactant.to_rarray(zl))
    return [Array(out[i]) for i in eachindex(out)]
end

end # module FabricPCReactantExt
