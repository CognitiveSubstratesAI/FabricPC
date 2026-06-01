using FabricPC
using Test

@testset "FabricPC.jl" begin
    @testset "scaffold" begin
        @test FabricPC.FABRICPC_VERSION == v"0.1.0"
    end
    include("test_phase_b.jl")
    include("test_residual.jl")
    include("test_mupc.jl")
    include("test_activations.jl")
    include("test_energies.jl")
end
