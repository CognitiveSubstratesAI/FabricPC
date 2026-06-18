# muPC training-recipe primitives (Phase F1): MuPCInitializer / XavierInitializer
# / AdamW. These are the pieces mupc_demo.py uses that Phase B/C lacked.

using FabricPC: initialize, step!, _zeros_like

@testset "MuPCInitializer is unit-variance (× gain)" begin
    W = initialize(MersenneTwister(0), (800, 800), MuPCInitializer())
    @test abs(sum(W) / length(W)) < 0.01           # ~zero mean
    @test isapprox(sqrt(sum(abs2, W) / length(W)), 1.0; rtol=0.05)   # unit std

    W2 = initialize(MersenneTwister(0), (800, 800), MuPCInitializer(; gain=2.5))
    @test isapprox(sqrt(sum(abs2, W2) / length(W2)), 2.5; rtol=0.05)
end

@testset "XavierInitializer std = √(2/(fan_in+fan_out))" begin
    fan_in, fan_out = 100, 300
    W = initialize(MersenneTwister(0), (fan_in, fan_out), XavierInitializer())
    expected = sqrt(2 / (fan_in + fan_out))
    @test isapprox(sqrt(sum(abs2, W) / length(W)), expected; rtol=0.05)
end

@testset "AdamW single step matches hand computation" begin
    # First step from zero moments: m̂ = g, v̂ = g² ⇒ update ≈ sign(g), plus
    # decoupled weight decay. new = w - lr·(g/(|g|+eps) + wd·w).
    W = Float32[1 2; -3 4]
    b = Float32[0.5 -0.5]
    g = Float32[0.5 -0.5; 2 -1]
    gb = Float32[1 -2]
    params = GraphParams(
        Dict("y" => NodeParams(Dict("e" => copy(W)), Dict("b" => copy(b))))
    )
    grads = GraphParams(Dict("y" => NodeParams(Dict("e" => g), Dict("b" => gb))))

    lr, wd, eps = 0.01f0, 0.1f0, 1.0f-8
    opt = AdamW(params; lr=lr, weight_decay=wd, eps=eps)
    newp = step!(opt, params, grads)

    expW = W .- lr .* (g ./ (abs.(g) .+ eps) .+ wd .* W)
    expb = b .- lr .* (gb ./ (abs.(gb) .+ eps) .+ wd .* b)
    @test newp.nodes["y"].weights["e"] ≈ expW rtol = 1e-4
    @test newp.nodes["y"].biases["b"] ≈ expb rtol = 1e-4
    @test opt.t == 1

    # A node with no params (source) passes through cleanly.
    empty = GraphParams(
        Dict(
            "x" => NodeParams(
                Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}()
            )
        )
    )
    o2 = AdamW(empty)
    @test length(step!(o2, empty, _zeros_like(empty)).nodes["x"].weights) == 0   # SoA: no params
end

@testset "AdamW descends a quadratic" begin
    # Minimize 0.5‖w − target‖² ⇒ grad = w − target. AdamW should drive w → target.
    target = Float32[2 -1 3]
    w = zeros(Float32, 1, 3)
    params = GraphParams(
        Dict("y" => NodeParams(Dict("e" => copy(w)), Dict{String, Matrix{Float32}}()))
    )
    opt = AdamW(params; lr=0.05)
    for _ in 1:500
        cur = params.nodes["y"].weights["e"]
        g = cur .- target
        grads = GraphParams(
            Dict("y" => NodeParams(Dict("e" => g), Dict{String, Matrix{Float32}}()))
        )
        params = step!(opt, params, grads)
    end
    @test params.nodes["y"].weights["e"] ≈ target rtol = 1e-2
end
