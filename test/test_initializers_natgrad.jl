using FabricPC
using Random, Statistics, Test

# Upstream-parity additions: Kaiming/He + Xavier-uniform + Ones + Uniform
# initializers, and the Fisher natural-gradient optimizer transforms.

@testset "Initializers + natural-gradient transforms (upstream parity)" begin
    @testset "Kaiming/He init" begin
        rng = MersenneTwister(7)
        # normal, fan_in, relu: std = gain·√2/√fan_in (fan_in=200 ⇒ 0.1)
        W = initialize(rng, (200, 100), KaimingInitializer())
        @test std(W) ≈ sqrt(2.0f0) / sqrt(200.0f0) atol = 5e-3
        @test abs(mean(W)) < 0.01
        # uniform variant: U(−limit, limit), limit = √2·√(3/fan_in)
        Wu = initialize(rng, (200, 100), KaimingInitializer(distribution=:uniform))
        limit = sqrt(2.0f0) * sqrt(3.0f0 / 200.0f0)
        @test maximum(Wu) <= limit && minimum(Wu) >= -limit
        @test maximum(Wu) > 0.9 * limit
        # leaky-relu with a=0 ⇒ gain √2 ⇒ same std as relu
        Wl = initialize(
            rng, (200, 100), KaimingInitializer(nonlinearity=:leaky_relu, a=0.0)
        )
        @test std(Wl) ≈ sqrt(2.0f0) / sqrt(200.0f0) atol = 5e-3
    end

    @testset "Xavier-uniform + Ones + Uniform" begin
        rng = MersenneTwister(7)
        # Xavier uniform: limit = √(6/(fan_in+fan_out))
        Wx = initialize(rng, (100, 50), XavierInitializer(distribution=:uniform))
        limit = sqrt(6.0f0 / (100.0f0 + 50.0f0))
        @test maximum(Wx) <= limit && minimum(Wx) >= -limit
        @test all(initialize(rng, (3, 4), OnesInitializer()) .== 1.0f0)
        Wun = initialize(rng, (2000,), UniformInitializer(minval=-2.0, maxval=3.0))
        @test minimum(Wun) >= -2.0f0 && maximum(Wun) <= 3.0f0
        @test mean(Wun) ≈ 0.5f0 atol = 0.1                 # midpoint of [−2, 3]
        # Xavier normal still defaults correctly (back-compat)
        Wxn = initialize(rng, (100, 50), XavierInitializer())
        @test std(Wxn) ≈ sqrt(2.0f0 / (100.0f0 + 50.0f0)) atol = 5e-3
    end

    @testset "NaturalGradientDiag (EMA Fisher diagonal)" begin
        gp = GraphParams(
            Dict(
                "n" => NodeParams(
                    Dict("e" => Float32[2.0 2.0]), Dict{String, Matrix{Float32}}())
            )
        )
        ng = NaturalGradientDiag(gp; fisher_decay=0.95, damping=1e-3)
        out = precondition!(ng, gp)
        # f = 0.95·0 + 0.05·g² = 0.2 ; out = g/(f+damping) = 2/0.201
        @test out.nodes["n"].weights["e"][1] ≈ 2.0f0 / (0.2f0 + 1.0f-3) atol = 1e-3
        @test ng.fisher.nodes["n"].weights["e"][1] ≈ 0.2f0 atol = 1e-4
        # second step: f = 0.95·0.2 + 0.05·4 = 0.39 ; out = 2/0.391
        out2 = precondition!(ng, gp)
        @test out2.nodes["n"].weights["e"][1] ≈ 2.0f0 / (0.39f0 + 1.0f-3) atol = 1e-3
        @test_throws Exception NaturalGradientDiag(gp; fisher_decay=1.0)
        @test_throws Exception NaturalGradientDiag(gp; damping=0.0)
    end

    @testset "NaturalGradientLayerwise (scalar Fisher per tensor)" begin
        gp = GraphParams(
            Dict(
                "n" => NodeParams(
                    Dict("e" => Float32[1.0 3.0]), Dict{String, Matrix{Float32}}())
            )
        )
        ng = NaturalGradientLayerwise(; fisher_decay=0.95, damping=1e-3)
        out = precondition!(ng, gp)
        # scalar f = 0.05·mean([1,9]) = 0.25 ; out = g/(0.25+0.001)
        @test out.nodes["n"].weights["e"][1] ≈ 1.0f0 / (0.25f0 + 1.0f-3) atol = 1e-3
        @test out.nodes["n"].weights["e"][2] ≈ 3.0f0 / (0.25f0 + 1.0f-3) atol = 1e-3
    end
end
