# Phase B acceptance + correctness tests.
#
# 1. Exact hand-computed gradient check (FP-robust integer fixture) for Linear's
#    explicit Gaussian self-grad / input-grad / weight-grad / bias-grad.
# 2. Finite-difference cross-check of all four gradients on a random fixture.
# 3. IdentityNode forward + explicit gradients.
# 4. Acceptance: a 3-layer linear PC graph learns a deterministic linear map
#    (energy ↓, prediction MSE ↓).

using FabricPC
using Test
using Random
using FabricPC:
    forward,
    forward_and_latent_grads,
    forward_and_weight_grads,
    NodeState,
    NodeInfo,
    SlotInfo,
    gather_inputs,
    initialize_graph_state,
    run_inference

# Helper: build a bare NodeState whose non-latent fields are zeroed placeholders
# (forward overwrites z_mu/error/energy; latent_grad is unused for
# energy). batch × features inferred from z_latent.
function bare_state(z_latent::AbstractMatrix)
    b, f = size(z_latent)
    return NodeState(
        z_latent,
        zeros(Float32, b, f),
        zeros(Float32, b, f),
        zeros(Float32, b),
        zeros(Float32, b, f)
    )
end

# Minimal internal-node NodeInfo (in_degree=out_degree=1 ⇒ explicit "else" branch).
internal_info(name, shape, edge_key) = NodeInfo(
    name,
    shape,
    "Linear",
    Dict{String, SlotInfo}(),
    1,
    1,
    [edge_key],
    String[],
    nothing
)

# `sum(...energy)`, not `first(forward(...))`: forward returns the NodeState alone now
# (upstream b6f64ad) and the scalar is the caller's to take — `state.energy` is per-sample.
local_energy(node, params, inputs, z_latent) =
    sum(forward(node, params, inputs, bare_state(z_latent)).energy)

@testset "Linear explicit gradients — exact hand-check" begin
    ek = "x->y:in"
    node = Linear((2,), "y"; use_bias=true)
    W = Float32[1 0; 0 1; 1 1]          # (3, 2)
    b = Float32[0 0]                    # (1, 2)
    params = NodeParams(Dict(ek => W), Dict("b" => b))
    x = Float32[1 2 3]                  # (1, 3)
    inputs = Dict(ek => x)
    z_target = Float32[0 0]             # clamped target latent

    # z_mu = x·W = [4 5]; error = z - z_mu = [-4 -5]; E = 0.5*(16+25) = 20.5
    st = forward(node, params, inputs, bare_state(z_target))
    @test st.z_mu ≈ Float32[4 5]
    @test st.error ≈ Float32[-4 -5]
    @test sum(forward(node, params, inputs, bare_state(z_target)).energy) ≈ 20.5f0

    info = internal_info("y", (2,), ek)
    _, input_grads, self_grad = forward_and_latent_grads(
        node, params, inputs, bare_state(z_target), info, true
    )
    # self_grad = precision*(z - z_mu) = [-4 -5]
    @test self_grad ≈ Float32[-4 -5]
    # input_grad = -(gme · Wᵀ); gme = error = [-4 -5]
    #   -([-4 -5]·[[1 0 1];[0 1 1]]) = [4 5 9]
    @test input_grads[ek] ≈ Float32[4 5 9]

    _, gp = forward_and_weight_grads(node, params, inputs, bare_state(z_target))
    # dW = -(xᵀ · gme) = [[4 5];[8 10];[12 15]]
    @test gp.weights[ek] ≈ Float32[4 5; 8 10; 12 15]
    # db = -Σ_batch gme = [4 5]
    @test gp.biases["b"] ≈ Float32[4 5]
end

@testset "Linear explicit gradients — finite-difference cross-check" begin
    rng = MersenneTwister(42)
    ek = "x->y:in"
    in_f, out_f, batch = 4, 3, 5
    node = Linear((out_f,), "y"; use_bias=true)
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

    # dE/dz_latent
    num_self = similar(self_grad)
    for i in eachindex(z)
        num_self[i] = fd() do δ
            zz = copy(z);
            zz[i] += δ
            local_energy(node, params, inputs, zz)
        end
    end
    @test self_grad ≈ num_self rtol = 2e-2

    # dE/dx
    num_in = similar(input_grads[ek])
    for i in eachindex(x)
        num_in[i] = fd() do δ
            xx = copy(x);
            xx[i] += δ
            local_energy(node, params, Dict(ek => xx), z)
        end
    end
    @test input_grads[ek] ≈ num_in rtol = 2e-2

    # dE/dW
    num_W = similar(W)
    for i in eachindex(W)
        num_W[i] = fd() do δ
            Wp = copy(W);
            Wp[i] += δ
            local_energy(node, NodeParams(Dict(ek => Wp), Dict("b" => b)), inputs, z)
        end
    end
    @test gp.weights[ek] ≈ num_W rtol = 2e-2

    # dE/db
    num_b = similar(b)
    for i in eachindex(b)
        num_b[i] = fd() do δ
            bp = copy(b);
            bp[i] += δ
            local_energy(node, NodeParams(Dict(ek => W), Dict("b" => bp)), inputs, z)
        end
    end
    @test gp.biases["b"] ≈ num_b rtol = 2e-2
end

@testset "IdentityNode forward + explicit gradients" begin
    ek = "a->id:in"
    node = IdentityNode((3,), "id"; scale=2.0)
    params = NodeParams(Dict{String, Matrix{Float32}}(), Dict{String, Matrix{Float32}}())
    x = Float32[1 2 3]
    z = Float32[0 0 0]
    inputs = Dict(ek => x)

    st = forward(node, params, inputs, bare_state(z))
    @test st.z_mu ≈ Float32[2 4 6]            # scale * sum(inputs)
    @test st.error ≈ Float32[-2 -4 -6]

    info = internal_info("id", (3,), ek)
    _, input_grads, self_grad = forward_and_latent_grads(
        node, params, inputs, bare_state(z), info, true
    )
    @test self_grad ≈ Float32[-2 -4 -6]       # grad_latent = z - z_mu
    @test input_grads[ek] ≈ Float32[4 8 12]   # -scale * gme = -2 * [-2 -4 -6]
end

@testset "Acceptance: 3-layer linear PC learns a linear map" begin
    data_rng = MersenneTwister(1)
    param_rng = MersenneTwister(7)
    in_f, hid_f, out_f, batch = 4, 6, 3, 16

    x = randn(data_rng, Float32, batch, in_f)
    A = randn(data_rng, Float32, in_f, out_f)
    y = x * A                                   # deterministic linear target

    xn = Linear((in_f,), "x")
    hn = Linear((hid_f,), "h")
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

    @test energies[end] < energies[1]
    @test energies[end] < 0.3f0 * energies[1]   # substantial energy reduction
    @test mse_after < mse_before
    @test mse_after < 0.5f0 * mse_before        # learns the map
    @test all(isfinite, energies)
end
