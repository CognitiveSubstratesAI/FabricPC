using FabricPC
using Test

@testset "FabricPC.jl" begin
    @testset "scaffold" begin
        @test FabricPC.FABRICPC_VERSION == v"0.1.0"
    end
    include("test_phase_b.jl")
end
