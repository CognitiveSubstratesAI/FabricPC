# StorkeyHopfield — the first node with a COMPOSITE energy (E_pc + s·E_hop). Validate
# (1) forward reports the full energy, and (2) the Enzyme seam differentiates the
# composite energy correctly for BOTH the latent grad (PC pull + attractor pull
# (s/D)(W²−W)z) and the weight grad (incl ∂E_hop/∂W) — vs finite difference.

using FabricPC
using FabricPC: NodeParams, NodeState, NodeInfo, SlotInfo, compute_mu, energy_kernel
using Enzyme
using Random, Test

const SH = FabricPC

@testset "StorkeyHopfield (composite PC + attractor energy)" begin
    rng = MersenneTwister(1)
    B, D = 4, 6
    node = StorkeyHopfield((D,), "hop"; enforce_symmetry=true)
    ek = "p->hop:in"
    params = SH.initialize_params(
        node, rng, (D,), Dict(ek => (D,)), NormalInitializer(; std=0.3)
    )
    probe = randn(rng, Float32, B, D)
    zlat = randn(rng, Float32, B, D)
    inputs = Dict{String, Any}(ek => probe)
    st = NodeState(zlat, zeros(Float32, B, D), zeros(Float32, B, D),
        zeros(Float32, B), zeros(Float32, B, D), zeros(Float32, B, D))

    @testset "forward energy = E_pc + E_hop" begin
        tot, _ = SH.forward(node, params, inputs, st)
        z_mu = compute_mu(node, params, inputs)
        Wp = SH._prepare_W(node, params.weights[ek])
        s = SH._strength(node, params)
        wz = zlat * Wp
        ehop = s * (0.5f0 / D) * sum(wz .* (wz .- zlat))
        epc = sum(SH.energy(node.energy, zlat, z_mu))
        @test tot ≈ epc + ehop rtol = 1e-4
        @test s ≈ 1.0f0 rtol = 1e-4                     # softplus(inverse_softplus(1)) = 1
    end

    @testset "seam differentiates the composite energy (vs finite-diff)" begin
        info = NodeInfo("hop", (D,), "StorkeyHopfield", Dict{String, SlotInfo}(),
            1, 1, [ek], String[], nothing)
        _, _, sg = SH.forward_and_latent_grads(node, params, inputs, st, info, false)
        _, gp = SH.forward_and_weight_grads(node, params, inputs, st)
        eps = 1.0f-2
        # latent: directional ⟨self_grad, v⟩ == central FD of energy_kernel(z)
        ekf(z) = energy_kernel(node, params, inputs, z)
        vz = randn(rng, Float32, B, D)
        fdz = (ekf(zlat .+ eps .* vz) - ekf(zlat .- eps .* vz)) / (2eps)
        @test sum(sg .* vz) ≈ fdz rtol = 2e-2
        # weight: directional ⟨dW, vW⟩ == central FD of energy_kernel(W)  (incl W²−W term)
        vW = randn(rng, Float32, D, D)
        ekW(c) = energy_kernel(
            node, NodeParams(Dict(ek => params.weights[ek] .+ c .* vW), params.biases),
            inputs, zlat
        )
        fdW = (ekW(eps) - ekW(-eps)) / (2eps)
        @test sum(gp.weights[ek] .* vW) ≈ fdW rtol = 2e-2
    end

    @testset "strength=0 ⇒ pure pass-through, no attractor energy" begin
        n0 = StorkeyHopfield((D,), "h0"; hopfield_strength=0.0, use_bias=false)
        p0 = SH.initialize_params(
            n0, rng, (D,), Dict(ek => (D,)), NormalInitializer(; std=0.3)
        )
        zmu = compute_mu(n0, p0, inputs)
        @test zmu ≈ SH.forward(n0.activation, probe) rtol = 1e-5   # blend=1 ⇒ activation(probe)
        @test all(iszero, SH._hopfield_energy(n0, p0, zlat))        # s=0 ⇒ E_hop=0
    end
end
