# EXPERIMENT (uncommitted, diagnostic only): does `using Reactant` alone trip
# FabricPC's F-04 dual-AD-backend guard (`_register_ad_backend!`,
# src/nodes/autodiff.jl) when Zygote has already registered `:zygote` in this
# session? And if so/if worked around, does a Reactant+Enzyme compiled gradient
# (never calling eager Enzyme.autodiff) match Zygote's eager gradient on the
# same math? See docs/decisions.md §15/§19/§23, docs/AUDIT_REGISTER.md J-01.
#
# Does NOT modify src/nodes/autodiff.jl or any other committed file.

using Printf

function hr(title)
    println()
    println("="^78)
    println(title)
    println("="^78)
end

hr("STEP 1: using FabricPC; using Zygote -- one real eager-Zygote seam call")

using FabricPC
using Zygote
using Random

println("[1] FabricPC._AD_BACKEND[] after `using Zygote` = ", FabricPC._AD_BACKEND[])

# ADLinear pattern, verbatim from test/test_autodiff_seam.jl: implements ONLY
# compute_mu, so it genuinely routes through the generic autodiff seam
# (_ad_param_grads/_ad_latent_grads) -- the REAL `Linear` node has hand-written
# closed-form gradient overrides and would bypass the seam entirely.
struct ADLinear <: FabricPC.AbstractNode
    shape::Tuple
    name::String
    activation::Any
    energy::Any
    weight_init::Any
    latent_init::Any
end
FabricPC.get_slots(::ADLinear) = Dict("in" => FabricPC.SlotSpec("in", true))
function FabricPC.compute_mu(n::ADLinear, params::FabricPC.NodeParams, inputs)
    out = n.shape[end]
    batch = size(first(values(inputs)), 1)
    pre = zeros(Float32, batch, out)
    for (k, xv) in inputs
        pre = pre .+ xv * params.weights[k]
    end
    haskey(params.biases, "b") && (pre = pre .+ params.biases["b"])
    return FabricPC.forward(n.activation, pre)
end

rng = MersenneTwister(1)
batch, in_f, out_f = 5, 4, 3
ekey = "src->h:in"
act, en = FabricPC.TanhActivation(), FabricPC.GaussianEnergy()
ad = ADLinear(
    (out_f,), "h", act, en, FabricPC.NormalInitializer(), FabricPC.NormalInitializer()
)
W = randn(rng, Float32, in_f, out_f)
b = randn(rng, Float32, 1, out_f)
x = randn(rng, Float32, batch, in_f)
z = randn(rng, Float32, batch, out_f)
params = FabricPC.NodeParams(Dict(ekey => W), Dict("b" => b))
inputs = Dict{String, Any}(ekey => x)
st = FabricPC.NodeState(z, zeros(Float32, batch, out_f), zeros(Float32, batch, out_f),
    zeros(Float32, batch), zeros(Float32, batch, out_f), zeros(Float32, batch, out_f))
info = FabricPC.NodeInfo("h", (out_f,), "node", Dict{String, FabricPC.SlotInfo}(),
    1, 1, [ekey], String[], nothing)

_, zyg_w = FabricPC.forward_and_weight_grads(ad, params, inputs, st)
_, zyg_ig, zyg_sg = FabricPC.forward_and_latent_grads(ad, params, inputs, st, info, false)
println(
    "[1] Zygote-eager FabricPC seam call SUCCEEDED (genuinely dispatched via _ad_param_grads/_ad_latent_grads)."
)
@printf("    |weight grad|^2 = %.6g   |bias grad|^2 = %.6g   |latent self-grad|^2 = %.6g\n",
    sum(abs2, zyg_w.weights[ekey]), sum(abs2, zyg_w.biases["b"]), sum(abs2, zyg_sg))
println("[1] AD backend after eager call = ", FabricPC._AD_BACKEND[])

# Independent, portable ground truth (same math: tanh(x*W+b), Gaussian energy)
# as a raw function -- used in step 4 to check a Reactant+Enzyme compiled
# gradient against Zygote's eager gradient on IDENTICAL math.
raw_mu(W, b, x) = tanh.(x * W .+ b)
raw_energy(W, b, x, z) = sum(abs2, z .- raw_mu(W, b, x)) * 0.5f0
gW_zyg, gb_zyg, gx_zyg, _ = Zygote.gradient(raw_energy, W, b, x, z)
@printf("[1] Raw-function Zygote ground truth: |gW|^2=%.6g |gb|^2=%.6g |gx|^2=%.6g\n",
    sum(abs2, gW_zyg), sum(abs2, gb_zyg), sum(abs2, gx_zyg))

hr("STEP 2: using Reactant -- does the F-04 guard throw?")

enzyme_uuid = Base.UUID("7da242da-08ed-463a-9acd-ee780be4f1d9")
reactant_uuid = Base.UUID("3c362404-f566-11ee-1572-e11a4b42c853")

enzyme_loaded_before = any(pkgid.uuid == enzyme_uuid for pkgid in keys(Base.loaded_modules))
ext_before = Base.get_extension(FabricPC, :FabricPCEnzymeExt)
println("[2] Enzyme module loaded BEFORE `using Reactant`?      ", enzyme_loaded_before)
println("[2] FabricPCEnzymeExt loaded BEFORE `using Reactant`?  ", ext_before !== nothing)

guard_threw = false
local guard_err, guard_bt
try
    Core.eval(Main, :(using Reactant))
    global guard_threw = false
    println(
        "[2] `using Reactant` returned WITHOUT a catchable exception reaching the caller."
    )
catch e
    global guard_threw = true
    global guard_err = e
    global guard_bt = catch_backtrace()
    println("[2] `using Reactant` THREW a catchable exception reaching the caller:")
    println(sprint(showerror, e))
end

enzyme_loaded_after = any(pkgid.uuid == enzyme_uuid for pkgid in keys(Base.loaded_modules))
reactant_loaded_after = any(
    pkgid.uuid == reactant_uuid for pkgid in keys(Base.loaded_modules)
)
ext_after = Base.get_extension(FabricPC, :FabricPCEnzymeExt)

println("[2] Enzyme module loaded AFTER attempt?     ", enzyme_loaded_after)
println("[2] Reactant module loaded AFTER attempt?   ", reactant_loaded_after)
println("[2] FabricPCEnzymeExt loaded AFTER attempt? ", ext_after !== nothing)
println("[2] FabricPC._AD_BACKEND[] AFTER attempt =  ", FabricPC._AD_BACKEND[])
println("[2] guard_threw (caught at the `using Reactant` call site) = ", guard_threw)

if guard_threw
    println("[2] stacktrace (up to 25 frames):")
    for (i, fr) in enumerate(stacktrace(guard_bt))
        i > 25 && break
        println("    [$i] ", fr)
    end
end

# Regardless of how [2] resolved, check whether the Enzyme-specific methods for
# the seam hooks actually got installed (i.e. whether FabricPCEnzymeExt fully
# executed past its `_register_ad_backend!(:enzyme)` call).
ms = methods(FabricPC._ad_param_grads)
println("[2] methods(FabricPC._ad_param_grads) count = ", length(ms))
for m in ms
    println("      ", m)
end

hr("STEP 3/4: Reactant+Enzyme compiled-gradient kernel")
println(
    "(never calling eager Enzyme.autodiff in this process -- only Zygote did eager AD above)"
)

if !reactant_loaded_after
    println(
        "[3/4] SKIPPED -- Reactant module itself did not end up loaded in this process; cannot proceed."
    )
else
    try
        Core.eval(Main, :(using Enzyme))
        Wr = Reactant.to_rarray(W)
        br = Reactant.to_rarray(b)
        xr = Reactant.to_rarray(x)
        zr = Reactant.to_rarray(z)

        function wgrad(W, b, x, z)
            dW = Enzyme.make_zero(W)
            db = Enzyme.make_zero(b)
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse), raw_energy, Enzyme.Active,
                Enzyme.Duplicated(W, dW), Enzyme.Duplicated(b, db),
                Enzyme.Const(x), Enzyme.Const(z)
            )
            return (dW, db)
        end

        @info "[3/4] compiling Reactant+Enzyme gradient kernel via Reactant.@compile ..."
        thunk = Reactant.@compile wgrad(Wr, br, xr, zr)
        println("[3/4] COMPILE SUCCEEDED.")
        gW_r, gb_r = thunk(Wr, br, xr, zr)
        gW_re = Array(gW_r)
        gb_re = Array(gb_r)

        maxW = maximum(abs.(gW_zyg))
        maxb = maximum(abs.(gb_zyg))
        reldiff_W = maximum(abs.(gW_re .- gW_zyg)) / max(1.0f-6, maxW)
        reldiff_b = maximum(abs.(gb_re .- gb_zyg)) / max(1.0f-6, maxb)

        println("[4] Reactant+Enzyme compiled gradient EXECUTED.")
        @printf(
            "    max|Δ gW| = %.3g   reldiff = %.3g\n",
            maximum(abs.(gW_re .- gW_zyg)),
            reldiff_W
        )
        @printf(
            "    max|Δ gb| = %.3g   reldiff = %.3g\n",
            maximum(abs.(gb_re .- gb_zyg)),
            reldiff_b
        )
        println(
            "    MATCHES Zygote eager gradient to ~1e-7? ",
            reldiff_W < 1.0f-5 && reldiff_b < 1.0f-5
        )
    catch e
        println("[3/4] Reactant+Enzyme compiled gradient FAILED:")
        println(sprint(showerror, e))
        for (i, fr) in enumerate(stacktrace(catch_backtrace()))
            i > 25 && break
            println("    [$i] ", fr)
        end
    end
end

hr("EXPERIMENT COMPLETE")
