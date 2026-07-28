# Decisive, minimal, isolated test of the F-04 guard's LIVE behavior in a
# process where FabricPCZygoteExt/FabricPCEnzymeExt are already precompiled
# (the normal case for any warm/repeated-use session).
using FabricPC
println("[probe] _AD_BACKEND before anything = ", FabricPC._AD_BACKEND[])
using Zygote
println("[probe] _AD_BACKEND after `using Zygote`  = ", FabricPC._AD_BACKEND[])
println(
    "[probe] Base.get_extension(FabricPC, :FabricPCZygoteExt) = ",
    Base.get_extension(FabricPC, :FabricPCZygoteExt) !== nothing
)
threw = false
try
    using Enzyme
    println("[probe] `using Enzyme` did NOT throw.")
catch e
    threw = true
    println("[probe] `using Enzyme` THREW: ", sprint(showerror, e))
end
println("[probe] _AD_BACKEND after `using Enzyme` attempt = ", FabricPC._AD_BACKEND[])
println(
    "[probe] Base.get_extension(FabricPC, :FabricPCEnzymeExt) = ",
    Base.get_extension(FabricPC, :FabricPCEnzymeExt) !== nothing
)
println("[probe] threw = ", threw)
println(
    "[probe] methods(FabricPC._ad_param_grads) = ",
    length(methods(FabricPC._ad_param_grads))
)
for m in methods(FabricPC._ad_param_grads)
    println("           ", m)
end
println("[probe] DONE")
