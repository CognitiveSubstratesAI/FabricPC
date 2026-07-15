# SkipConnection + LinearResidual tests.
#
# Reuses bare_state / local_energy from test_phase_b.jl (included first). Covers:
# 1. SkipConnection forward + explicit grads (hand-check + finite-diff).
# 2. LinearResidual forward + explicit grads, both slots (hand-check + finite-diff).
# 3. Acceptance: a residual-block graph (x → LinearResidual → y) learns a map.

using FabricPC: slot

# NodeInfo with arbitrary in-edges (in_degree from the edge list, out_degree=1 ⇒
# explicit "else" branch).
multi_info(name, shape, in_edges) = NodeInfo(
    name,
    shape,
    "Node",
    Dict{String, SlotInfo}(),
    length(in_edges),
    1,
    collect(in_edges),
    String[],
    nothing
)

@testset "SkipConnection forward + explicit gradients" begin
    ek = "a->s:in"
    node = SkipConnection((3,), "s")
    params = NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}())
    x = Float32[1 2 3]
    z = Float32[0 0 0]
    inputs = Dict(ek => x)

    st = forward(node, params, inputs, bare_state(z))
    @test st.z_mu ≈ Float32[1 2 3]            # pure sum, no scale
    @test st.error ≈ Float32[-1 -2 -3]

    info = multi_info("s", (3,), [ek])
    _, input_grads, self_grad = forward_and_latent_grads(
        node, params, inputs, bare_state(z), info, true
    )
    @test self_grad ≈ Float32[-1 -2 -3]       # z - z_mu
    @test input_grads[ek] ≈ Float32[1 2 3]    # -gme = -error

    # slot flags: non-variance-scalable skip slot
    sp = get_slots(node)["in"]
    @test sp.is_skip_connection
    @test !sp.is_variance_scalable
end

@testset "LinearResidual forward + explicit gradients — hand-check" begin
    in_ek, skip_ek = "a->r:in", "b->r:skip"
    node = LinearResidual((2,), "r"; use_bias=true)
    W = Float32[1 0; 0 1; 1 1]                # (3, 2)
    b = Float32[0 0]
    params = NodeParams(Dict(in_ek => W), Dict("b" => b))
    x_in = Float32[1 2 3]                     # (1, 3)
    x_skip = Float32[10 20]                   # (1, 2)
    z = Float32[0 0]
    inputs = Dict(in_ek => x_in, skip_ek => x_skip)

    # transform = x_in·W = [4 5]; z_mu = [4 5] + [10 20] = [14 25]
    st = forward(node, params, inputs, bare_state(z))
    @test st.z_mu ≈ Float32[14 25]
    @test st.error ≈ Float32[-14 -25]
    @test sum(forward(node, params, inputs, bare_state(z)).energy) ≈ 410.5f0  # 0.5*(196+625)

    info = multi_info("r", (2,), [in_ek, skip_ek])
    _, input_grads, self_grad = forward_and_latent_grads(
        node, params, inputs, bare_state(z), info, true
    )
    @test self_grad ≈ Float32[-14 -25]
    @test input_grads[in_ek] ≈ Float32[14 25 39]    # -(gme · Wᵀ)
    @test input_grads[skip_ek] ≈ Float32[14 25]      # -error (skip bypasses activation)

    _, gp = forward_and_weight_grads(node, params, inputs, bare_state(z))
    @test gp.weights[in_ek] ≈ Float32[14 25; 28 50; 42 75]  # -(x_inᵀ · gme)
    @test !haskey(gp.weights, skip_ek)               # skip slot has no weights
    @test gp.biases["b"] ≈ Float32[14 25]
end

@testset "LinearResidual gradients — finite-difference cross-check" begin
    rng = MersenneTwister(99)
    in_ek, skip_ek = "a->r:in", "b->r:skip"
    in_f, out_f, batch = 4, 3, 5
    node = LinearResidual((out_f,), "r"; use_bias=true)
    W = randn(rng, Float32, in_f, out_f)
    b = randn(rng, Float32, 1, out_f)
    params = NodeParams(Dict(in_ek => copy(W)), Dict("b" => copy(b)))
    x_in = randn(rng, Float32, batch, in_f)
    x_skip = randn(rng, Float32, batch, out_f)
    z = randn(rng, Float32, batch, out_f)
    inputs = Dict(in_ek => x_in, skip_ek => x_skip)
    info = multi_info("r", (out_f,), [in_ek, skip_ek])

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

    num_in = similar(input_grads[in_ek])
    for i in eachindex(x_in)
        num_in[i] = fd() do δ
            xx = copy(x_in);
            xx[i] += δ
            local_energy(node, params, Dict(in_ek => xx, skip_ek => x_skip), z)
        end
    end
    @test input_grads[in_ek] ≈ num_in rtol = 2e-2

    num_skip = similar(input_grads[skip_ek])
    for i in eachindex(x_skip)
        num_skip[i] = fd() do δ
            xx = copy(x_skip);
            xx[i] += δ
            local_energy(node, params, Dict(in_ek => x_in, skip_ek => xx), z)
        end
    end
    @test input_grads[skip_ek] ≈ num_skip rtol = 2e-2

    num_W = similar(W)
    for i in eachindex(W)
        num_W[i] = fd() do δ
            Wp = copy(W);
            Wp[i] += δ
            local_energy(node, NodeParams(Dict(in_ek => Wp), Dict("b" => b)), inputs, z)
        end
    end
    @test gp.weights[in_ek] ≈ num_W rtol = 2e-2
end

@testset "Acceptance: residual block learns a linear map" begin
    data_rng = MersenneTwister(3)
    param_rng = MersenneTwister(11)
    d, out_f, batch = 4, 3, 16

    x = randn(data_rng, Float32, batch, d)
    A = randn(data_rng, Float32, d, out_f)
    y = x * A

    xn = Linear((d,), "x")
    rn = LinearResidual((d,), "r")           # skip path needs source dim == r dim
    yn = Linear((out_f,), "y")
    inference = InferenceSGD(; eta_infer=0.1, infer_steps=30)
    structure = graph(
        [xn, rn, yn],
        [Edge(xn, rn), Edge(xn, slot(rn, "skip")), Edge(rn, yn)],
        TaskMap(; x=xn, y=yn),
        inference
    )

    params = initialize_params(structure, param_rng)
    batch_dict = Dict("x" => x, "y" => y)

    function predict(p)
        clamps = Dict{String, Any}("x" => x)
        st = initialize_graph_state(structure, batch, param_rng; clamps=clamps, params=p)
        fs = run_inference(p, st, clamps, structure)
        return fs.nodes["y"].z_mu
    end
    mse(pred) = sum(abs2.(pred .- y)) / length(y)

    mse_before = mse(predict(params))
    energies = Float32[]
    for _ in 1:200
        params, e, _ = train_step(params, batch_dict, structure, 0.02, param_rng)
        push!(energies, e)
    end
    mse_after = mse(predict(params))

    @test all(isfinite, energies)
    @test energies[end] < 0.3f0 * energies[1]
    @test mse_after < 0.5f0 * mse_before
end
