# Non-linear activation tests (Phase D1).
#
# Reuses bare_state / local_energy / internal_info from test_phase_b.jl.
# 1. derivative(f, x) matches a finite-difference slope of forward(f, x).
# 2. muPC variance_gain / jacobian_gain constants match the upstream tables.
# 3. The EXISTING explicit Linear path is exact with a non-linear activation
#    (finite-diff cross-check of self / input / weight grads with Tanh).
# 4. A non-linear (Tanh) PC graph trains (energy decreases).

using FabricPC:
    forward,
    derivative,
    variance_gain,
    jacobian_gain,
    jacobian,
    forward_and_latent_grads,
    forward_and_weight_grads

@testset "activation derivative matches finite-difference slope" begin
    ε = 1.0f-3
    # x points chosen away from kinks (ReLU@0, HardTanh@±1, LeakyReLU@0).
    cases = [
        (IdentityActivation(), Float32[-2.0, -0.3, 0.7, 3.1]),
        (SigmoidActivation(), Float32[-2.0, -0.5, 0.5, 2.0]),
        (TanhActivation(), Float32[-2.0, -0.5, 0.5, 2.0]),
        (ReLUActivation(), Float32[-2.0, -1.0, 1.0, 2.0]),
        (LeakyReLUActivation(; alpha=0.1), Float32[-2.0, -1.0, 1.0, 2.0]),
        (GeluActivation(), Float32[-2.0, -0.5, 0.5, 2.0]),
        (HardTanhActivation(), Float32[-2.0, -0.5, 0.5, 2.0])
    ]
    for (act, xs) in cases
        fd = (forward(act, xs .+ ε) .- forward(act, xs .- ε)) ./ (2ε)
        d = derivative(act, xs)
        # IdentityActivation returns a scalar 1; broadcast-compare handles it.
        @test all(abs.(d .- fd) .< 1.0f-2)
    end
end

@testset "muPC gain constants" begin
    @test variance_gain(TanhActivation()) ≈ Float32(sqrt(5 / 3))
    @test jacobian_gain(TanhActivation()) == 1.261f0
    @test variance_gain(ReLUActivation()) ≈ Float32(sqrt(2))
    @test jacobian_gain(ReLUActivation()) == 1.0f0          # default
    @test variance_gain(LeakyReLUActivation(; alpha=0.0)) ≈ Float32(sqrt(2))
    @test variance_gain(GeluActivation()) ≈ Float32(sqrt(2))
    @test jacobian_gain(GeluActivation()) == 1.168f0
    @test variance_gain(HardTanhActivation()) ≈ Float32(sqrt(5 / 3))
    @test jacobian_gain(HardTanhActivation()) == 1.035f0
    @test variance_gain(SigmoidActivation()) == 1.0f0       # default
end

@testset "explicit Linear path is exact with Tanh activation" begin
    rng = MersenneTwister(123)
    ek = "x->y:in"
    in_f, out_f, batch = 4, 3, 5
    node = Linear((out_f,), "y"; activation=TanhActivation(), use_bias=true)
    W = randn(rng, Float32, in_f, out_f)
    b = randn(rng, Float32, 1, out_f)
    params = NodeParams(Dict(ek => copy(W)), Dict("b" => copy(b)))
    x = randn(rng, Float32, batch, in_f)
    z = randn(rng, Float32, batch, out_f)
    inputs = Dict(ek => x)
    info = internal_info("y", (out_f,), ek)

    _, input_grads, self_grad = forward_and_latent_grads(
        node, params, inputs, bare_state(z), info, true
    )
    _, gp = forward_and_weight_grads(node, params, inputs, bare_state(z))

    ε = 1.0f-2
    fd(f) = (f(ε) - f(-ε)) / (2ε)

    num_self = similar(self_grad)
    for i in eachindex(z)
        num_self[i] = fd() do δ
            zz = copy(z);
            zz[i] += δ
            local_energy(node, params, inputs, zz)
        end
    end
    @test self_grad ≈ num_self rtol = 2e-2

    num_in = similar(input_grads[ek])
    for i in eachindex(x)
        num_in[i] = fd() do δ
            xx = copy(x);
            xx[i] += δ
            local_energy(node, params, Dict(ek => xx), z)
        end
    end
    @test input_grads[ek] ≈ num_in rtol = 2e-2

    num_W = similar(W)
    for i in eachindex(W)
        num_W[i] = fd() do δ
            Wp = copy(W);
            Wp[i] += δ
            local_energy(node, NodeParams(Dict(ek => Wp), Dict("b" => b)), inputs, z)
        end
    end
    @test gp.weights[ek] ≈ num_W rtol = 2e-2
end

@testset "non-linear (Tanh) PC graph trains" begin
    data_rng = MersenneTwister(21)
    param_rng = MersenneTwister(31)
    in_f, hid_f, out_f, batch = 4, 8, 3, 16

    x = randn(data_rng, Float32, batch, in_f)
    # A mildly non-linear target so a Tanh hidden layer has something to fit.
    A = randn(data_rng, Float32, in_f, out_f)
    y = tanh.(x * A) .* 0.5f0

    xn = Linear((in_f,), "x")
    hn = Linear((hid_f,), "h"; activation=TanhActivation())
    yn = Linear((out_f,), "y")
    inference = InferenceSGD(; eta_infer=0.1, infer_steps=30)
    structure = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        inference
    )
    params = initialize_params(structure, param_rng)
    batch_dict = Dict("x" => x, "y" => y)

    energies = Float32[]
    for _ in 1:200
        params, e, _ = train_step(params, batch_dict, structure, 0.02, param_rng)
        push!(energies, e)
    end
    @test all(isfinite, energies)
    @test energies[end] < 0.5f0 * energies[1]
end

@testset "full jacobian() — softmax cross-checked vs upstream (activations.py:348)" begin
    # Reference computed from upstream's EXACT formula J_ij = s_i(δ_ij − s_j) on x = [1 2 3]
    # (s = [0.090031, 0.244728, 0.665241]). Layout: batch-first (B,D) → (B,D,D).
    x = reshape(Float32[1, 2, 3], 1, 3)
    J = jacobian(SoftmaxActivation(), x)
    @test size(J) == (1, 3, 3)
    ref = Float32[0.081925 -0.022033 -0.059892
                  -0.022033 0.184836 -0.162803
                  -0.059892 -0.162803 0.222695]
    @test isapprox(J[1, :, :], ref; atol=1f-5)            # numeric parity with upstream
    @test all(abs.(sum(J; dims=3)) .< 1f-5)               # softmax-Jacobian invariant: row sums 0
    # diagonal must equal the diagonal `derivative` (s·(1−s)) the PC path uses
    d = derivative(SoftmaxActivation(), x)
    @test isapprox(Float32[J[1, i, i] for i in 1:3], vec(d); atol=1f-6)

    # element-wise activations: jacobian = diag(derivative), off-diagonals exactly 0
    for act in (IdentityActivation(), TanhActivation(), ReLUActivation())
        xe = reshape(Float32[-1.5, 0.4, 2.0, -0.7], 2, 2)
        Je = jacobian(act, xe)
        @test size(Je) == (2, 2, 2)
        de = derivative(act, xe) .* ones(Float32, 2, 2)
        for b in 1:2, i in 1:2, j in 1:2
            @test isapprox(Je[b, i, j], i == j ? de[b, i] : 0.0f0; atol=1f-6)
        end
    end
end
