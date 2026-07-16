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
    GraphParams, GraphStructure, GraphState, repack_params, state_from_latents,
    flat_inference_step, flat_sgd_step, jit_train_step_runner, graph_params_from_flat
import FabricPC: compile_inference, compile_train_step
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

"""
    compile_inference(structure, params, clamps; batch, loop=false)

`loop=false` (default): the `infer_steps` are UNROLLED into the traced graph — long-standing
behaviour, and what J-04's numbers were measured on.

`loop=true`: emit a real `@trace for` loop (J-09). Both compile and agree with the eager path to
1.49e-8 (float32 reassociation noise). They are different PROGRAMS, which is the point: §28 argues
XLA's ~27x FLOP advantage over `jax.jit` comes from optimizing the fully-UNROLLED graph (hoisting a
clamped-source-only forward out of every step; DCE-ing the gradient into a clamped node) — neither
of which a loop body can structurally do. That was an argument from HLO; `loop=true` makes it a
measurement. Measured bonus: it compiles ~2.8x faster (the body is emitted once, not `infer_steps`
times), which matters more the more steps you run.

`track_numbers=false` IS THE WHOLE TRICK, and it is Reactant's own documented `@trace` option
(`ReactantCore`'s docstring: "whether Julia numbers should be automatically promoted to traced
numbers upon entering the loop"). Default `true` promotes every Julia number the loop body touches,
which is what made `SlotInfo.is_multi_input::Bool` and `Linear.use_bias::Bool` derive as
`TracedRNumber{Bool}` — unholdable by a non-parametric struct (`NoFieldMatchError`) — and made
`if !clamped[i]` fail with "non-boolean (TracedRNumber{Bool}) used in boolean context" (upstream
#2441). It also keeps Reactant from walking into `shape::Tuple`, dodging the upstream `Vararg`
`UndefRefError` we reported. With it, NO destructure, NO custom `traced_type_inner`/`make_tracer`,
and NO `Val`-wrapped mask are needed: the plain `CompiledPlan`/`SlotInfo`/`Vector{Bool}` trace fine.

⚠️ It is correct HERE because every loop-carried value in this loop is an ARRAY (`NodeState`
fields); `eta`/`decay`/topology are compile-time constants. If a future dynamic SCALAR ever joins
the loop-carried state (a convergence error, a step counter — cf. the frozen-`t` AdamW hazard in
decisions.md §25), `track_numbers=false` would silently freeze it at its trace-time value rather
than erroring. Re-check this flag if the loop's carried state stops being arrays-only.
"""
function compile_inference(
    structure::GraphStructure, params::GraphParams, clamps::AbstractDict; batch::Int,
    loop::Bool=false, sync::Bool=false
)
    plan = CompiledPlan(structure)
    arr_tuple, layout = flatten_param_arrays(to_flat_params(plan, params))
    clamped = Bool[n in keys(clamps) for n in plan.names]
    # Sample z_latents (zeros of each node's (batch, features) shape) to trace shapes.
    zl = ntuple(i -> zeros(Float32, batch, plan.infos[i].shape...), length(plan.names))

    params_r = Reactant.to_rarray(arr_tuple)                       # marshalled ONCE
    steps = plan.inference.infer_steps
    function looped(at, z)
        fp = repack_params(at, layout)
        fs = state_from_latents(z)
        Reactant.@trace track_numbers=false for _ in 1:steps
            fs = flat_inference_step(plan, fp, fs, clamped)
        end
        return ntuple(i -> fs[i].z_latent, length(fs))
    end
    runner(at, z) = jit_inference_runner(at, z, plan, layout, clamped)
    # `sync=true` is Reactant's documented completion barrier — CompileOptions.jl: "Reactant
    # computations are asynchronous by default. If `true`, the computation will be executed
    # synchronously, blocking till the computation is complete. This is recommended when
    # benchmarking." Default FALSE here on purpose: the production path `ci(init_state)` returns
    # plain Julia arrays by contract, and `Array(...)` already syncs, so a barrier buys nothing
    # there. It exists for the BENCHMARK, which must time compute without paying a device->host
    # copy JAX never pays.
    # ⚠️ VERIFY, do not trust: upstream #2647 (OPEN) "Buffer synchronisation is sometimes removed"
    # — the emitted wait can be dead-code-eliminated, which would silently restore the async
    # artifact (decisions.md §11's bogus ~1000x) with no `Array(...)` left to expose it.
    thunk = if loop
        sync ? Reactant.@compile(sync = true, looped(params_r, Reactant.to_rarray(zl))) :
               Reactant.@compile(looped(params_r, Reactant.to_rarray(zl)))
    else
        sync ? Reactant.@compile(sync = true, runner(params_r, Reactant.to_rarray(zl))) :
               Reactant.@compile(runner(params_r, Reactant.to_rarray(zl)))
    end
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

# ── Compiled full training step (J-02): E-step inference + local SGD M-step in one XLA graph ─────

"""A compiled training-step thunk + the static data to feed/read it. Unlike `CompiledInference`,
the params are NOT held resident: a training step CONSUMES the current weights and PRODUCES new
ones, so they are marshalled per call (the caller may instead keep the output device-resident and
feed it back — the efficient training loop; not done in this correctness-first callable). `lr` is
baked into the thunk (recompile to change it)."""
struct CompiledTrainStep
    plan::CompiledPlan
    layout::Any
    clamped::Vector{Bool}
    thunk::Any
    batch::Int
    lr::Float32
end

"""
    compile_train_step(structure, params, clamps; batch, lr, loop=true) -> CompiledTrainStep

JIT-compile one full PC training step — the inference E-step (`flat_run_inference`) AND the local
SGD weight-update M-step (`flat_sgd_step`) — into a single Reactant/XLA graph. The returned callable
maps `(params, init_state) -> GraphParams` (updated params after one step). Dense/v0 only (see
`flat_train_step`): Linear/Identity/Skip/LinearResidual, no muPC scaling, no TransformerBlock/Conv
weight backward. `lr` is compile-time constant.

`loop=true` (default) emits the E-step as a `@trace track_numbers=false for` loop (J-09) — the M-step
runs once after it. `loop=false` unrolls (via `jit_train_step_runner`). Both trace under the same
`track_numbers=false` contract as `compile_inference`, and for the same reason: every loop-carried
value is an array; `lr`/topology are compile-time constants. The M-step's weight grads are the
closed-form local rule (no autodiff), so nothing here needs Enzyme — that is only the transformer
lane (J-10 gate 2), out of this dense scope.

Requires `using Reactant` (this `FabricPCReactantExt` extension)."""
function compile_train_step(
    structure::GraphStructure, params::GraphParams, clamps::AbstractDict;
    batch::Int, lr::Real, loop::Bool=true
)
    plan = CompiledPlan(structure)
    arr_tuple, layout = flatten_param_arrays(to_flat_params(plan, params))
    clamped = Bool[n in keys(clamps) for n in plan.names]
    zl = ntuple(i -> zeros(Float32, batch, plan.infos[i].shape...), length(plan.names))
    η = Float32(lr)
    steps = plan.inference.infer_steps

    function looped_train(at, z)
        fp = repack_params(at, layout)
        fs = state_from_latents(z)
        Reactant.@trace track_numbers=false for _ in 1:steps
            fs = flat_inference_step(plan, fp, fs, clamped)
        end
        newfp = flat_sgd_step(plan, fp, fs, clamped, η)
        return flatten_param_arrays(newfp)[1]
    end
    unrolled(at, z) = jit_train_step_runner(at, z, plan, layout, clamped, η)

    thunk = loop ?
        Reactant.@compile(looped_train(Reactant.to_rarray(arr_tuple), Reactant.to_rarray(zl))) :
        Reactant.@compile(unrolled(Reactant.to_rarray(arr_tuple), Reactant.to_rarray(zl)))
    return CompiledTrainStep(plan, layout, clamped, thunk, batch, η)
end

"""
    (ct::CompiledTrainStep)(params, init_state) -> GraphParams

Run one compiled training step: marshal the CURRENT params + per-batch init state, execute the XLA
thunk (inference + SGD update), and rebuild `GraphParams` from the updated flat arrays. Params are
re-marshalled every call because training changes them (contrast `CompiledInference`, which holds
inference-constant weights resident)."""
# A flattened output element is normally one device array, but a TransformerBlock's whole weight set
# is stashed as ONE nested Tuple (flat_block_args), so materialize recursively: `Array` each leaf,
# `map` into each nested tuple.
_materialize(x) = x isa Tuple ? map(_materialize, x) : Array(x)

function (ct::CompiledTrainStep)(params::GraphParams, init_state::GraphState)
    arr_tuple, _ = flatten_param_arrays(to_flat_params(ct.plan, params))
    zl = ntuple(i -> init_state.nodes[ct.plan.names[i]].z_latent, length(ct.plan.names))
    out = ct.thunk(Reactant.to_rarray(arr_tuple), Reactant.to_rarray(zl))
    new_fp = repack_params(ntuple(i -> _materialize(out[i]), length(out)), ct.layout)
    return graph_params_from_flat(ct.plan, new_fp)
end

end # module FabricPCReactantExt
