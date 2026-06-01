# Non-Gaussian energy tests (Phase D2).
#
# Reuses bare_state / local_energy / internal_info from test_phase_b.jl.
# 1. Per-energy grad_latent (∂E/∂z) AND grad_mu (∂E/∂z_mu) vs finite differences.
# 2. The explicit node path with an ASYMMETRIC energy (Bernoulli + Sigmoid):
#    end-to-end finite-diff of self / input / weight gradients.
# 3. A Bernoulli (binary) PC graph trains.
# 4. Softmax: forward sums to 1; a CrossEntropy + Softmax classifier reduces
#    energy (diagonal-Jacobian approximation — see SoftmaxActivation docstring).

using FabricPC: forward, derivative, energy, grad_latent, grad_mu,
    forward_and_latent_grads, forward_and_weight_grads, initialize_graph_state,
    run_inference

# Central finite-difference of total energy w.r.t. z and w.r.t. z_mu.
function _energy_grad_fd(e, z, mu; ε=1.0f-3)
    nz = similar(z)
    for i in eachindex(z)
        zp = copy(z);
        zp[i] += ε
        zm = copy(z);
        zm[i] -= ε
        nz[i] = (sum(energy(e, zp, mu)) - sum(energy(e, zm, mu))) / (2ε)
    end
    nm = similar(mu)
    for i in eachindex(mu)
        mp = copy(mu);
        mp[i] += ε
        mm = copy(mu);
        mm[i] -= ε
        nm[i] = (sum(energy(e, z, mp)) - sum(energy(e, z, mm))) / (2ε)
    end
    return nz, nm
end

@testset "energy gradients vs finite differences (z and z_mu)" begin
    rng = MersenneTwister(7)
    rand01(n...) = 0.2f0 .+ 0.6f0 .* rand(rng, Float32, n...)   # in (0.2, 0.8)

    # (energy, z, z_mu) chosen to stay off non-smooth points.
    z = randn(rng, Float32, 4, 3)
    mu = randn(rng, Float32, 4, 3)
    cases = [
        (GaussianEnergy(; precision=1.5), z, mu),
        (BernoulliEnergy(), rand01(4, 3), rand01(4, 3)),
        (CrossEntropyEnergy(), rand01(4, 3), rand01(4, 3)),
        # Laplacian: keep |z-mu| well away from 0 so sign is stable under ±ε.
        (LaplacianEnergy(; scale=2.0), fill(2.0f0, 4, 3), fill(0.5f0, 4, 3)),
        # Huber(δ=1): keep |z-mu| < δ (quadratic region, smooth).
        (HuberEnergy(; delta=1.0), z .* 0.1f0, mu .* 0.1f0),
        (KLDivergenceEnergy(), rand01(4, 3), rand01(4, 3))
    ]
    for (e, zz, mm) in cases
        nz, nm = _energy_grad_fd(e, zz, mm)
        @test grad_latent(e, zz, mm) ≈ nz rtol = 2e-2 atol = 1e-3
        @test grad_mu(e, zz, mm) ≈ nm rtol = 2e-2 atol = 1e-3
    end
end

@testset "explicit node path exact with Bernoulli energy + Sigmoid" begin
    rng = MersenneTwister(17)
    ek = "x->y:in"
    in_f, out_f, batch = 4, 3, 5
    node = Linear(
        (out_f,),
        "y";
        activation=SigmoidActivation(),
        energy=BernoulliEnergy(),
        use_bias=true
    )
    W = randn(rng, Float32, in_f, out_f) .* 0.5f0
    b = randn(rng, Float32, 1, out_f) .* 0.5f0
    params = NodeParams(Dict(ek => copy(W)), Dict("b" => copy(b)))
    x = randn(rng, Float32, batch, in_f)
    z = 0.2f0 .+ 0.6f0 .* rand(rng, Float32, batch, out_f)   # binary-ish target in (0.2,0.8)
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
        num_self[i] = fd(
            δ -> (zz=copy(z); zz[i] += δ; local_energy(node, params, inputs, zz))
        )
    end
    @test self_grad ≈ num_self rtol = 2e-2 atol = 1e-3

    num_in = similar(input_grads[ek])
    for i in eachindex(x)
        num_in[i] = fd(
            δ -> (xx=copy(x); xx[i] += δ; local_energy(node, params, Dict(ek => xx), z))
        )
    end
    @test input_grads[ek] ≈ num_in rtol = 2e-2 atol = 1e-3

    num_W = similar(W)
    for i in eachindex(W)
        num_W[i] = fd(
            δ -> (Wp=copy(W); Wp[i] += δ;
                local_energy(node, NodeParams(Dict(ek => Wp), Dict("b" => b)), inputs, z))
        )
    end
    @test gp.weights[ek] ≈ num_W rtol = 2e-2 atol = 1e-3
end

@testset "Bernoulli (binary) PC graph trains" begin
    data_rng = MersenneTwister(23)
    param_rng = MersenneTwister(29)
    in_f, hid_f, out_f, batch = 4, 8, 3, 16

    x = randn(data_rng, Float32, batch, in_f)
    # Binary targets from a fixed random rule.
    y = Float32.((x * randn(data_rng, Float32, in_f, out_f)) .> 0.0f0)

    xn = Linear((in_f,), "x")
    hn = Linear((hid_f,), "h"; activation=TanhActivation())
    yn = Linear((out_f,), "y"; activation=SigmoidActivation(), energy=BernoulliEnergy())
    structure = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=30)
    )
    params = initialize_params(structure, param_rng)
    batch_dict = Dict("x" => x, "y" => y)

    energies = Float32[]
    for _ in 1:200
        params, e, _ = train_step(params, batch_dict, structure, 0.02, param_rng)
        push!(energies, e)
    end
    @test all(isfinite, energies)
    @test energies[end] < energies[1]
end

@testset "Softmax forward + CrossEntropy classifier improves accuracy" begin
    # Softmax sums to 1 along the feature axis.
    s = forward(SoftmaxActivation(), randn(MersenneTwister(1), Float32, 6, 4))
    @test all(abs.(vec(sum(s; dims=2)) .- 1.0f0) .< 1.0f-5)
    @test all(s .>= 0.0f0)

    # NOTE: SoftmaxActivation uses the diagonal-Jacobian PC approximation
    # (off-diagonal -sᵢsⱼ terms dropped), so the energy is NOT a clean monotone
    # objective here — we check the actual task metric (train accuracy) instead.
    data_rng = MersenneTwister(41)
    param_rng = MersenneTwister(43)
    in_f, hid_f, n_class, batch = 5, 10, 3, 24
    x = randn(data_rng, Float32, batch, in_f)
    labels = rand(data_rng, 1:n_class, batch)
    y = zeros(Float32, batch, n_class)             # one-hot
    for i in 1:batch
        y[i, labels[i]] = 1.0f0
    end

    xn = Linear((in_f,), "x")
    hn = Linear((hid_f,), "h"; activation=TanhActivation())
    yn = Linear(
        (n_class,),
        "y";
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy()
    )
    structure = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=30)
    )
    params = initialize_params(structure, param_rng)
    batch_dict = Dict("x" => x, "y" => y)

    function accuracy(p)
        clamps = Dict{String, Any}("x" => x)
        st = initialize_graph_state(structure, batch, param_rng; clamps=clamps, params=p)
        pred = run_inference(p, st, clamps, structure).nodes["y"].z_mu
        return sum(argmax(pred[i, :]) == labels[i] for i in 1:batch) / batch
    end

    acc0 = accuracy(params)
    energies = Float32[]
    for _ in 1:300
        params, e, _ = train_step(params, batch_dict, structure, 0.05, param_rng)
        push!(energies, e)
    end
    @test all(isfinite, energies)
    @test accuracy(params) > acc0          # diagonal-softmax PC still improves accuracy
end
