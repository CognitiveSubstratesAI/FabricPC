using FabricPC
using Test

@testset "FabricPC.jl" begin
    @testset "scaffold" begin
        @test FabricPC.FABRICPC_VERSION == v"0.1.0"
    end
    # Phase B onward appends per-area test files here.
end
