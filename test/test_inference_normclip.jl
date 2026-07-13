# InferenceSGDNormClip (C-05, port of inference.py:312-356) + the `compute_new_latent`
# dispatch-point refactor of `update_latents` that makes it pluggable (inference.py:201-243).
using Test
using Random
using FabricPC

@testset "InferenceSGDNormClip (C-05, inference.py:312-356)" begin
    @testset "compute_new_latent: matches hand computation, both under and over max_norm" begin
        inf = InferenceSGDNormClip(;
            eta_infer=0.1, latent_decay=0.0, max_norm=1.0, eps=1.0f-8
        )
        # row 1: ||g|| = sqrt(3²+4²) = 5 > max_norm=1 ⇒ clip_factor = 1/(5+eps)
        # row 2: ||g|| = sqrt(0.3²+0.4²) = 0.5 < max_norm=1 ⇒ clip_factor = 1 (unclipped)
        z = zeros(Float32, 2, 2)
        g = Float32[3.0 4.0; 0.3 0.4]
        new_z = FabricPC.compute_new_latent(inf, z, g)
        clip1 = 1.0f0 / (5.0f0 + 1.0f-8)
        @test new_z[1, :] ≈ -0.1f0 .* (g[1, :] .* clip1) atol = 1.0f-6
        @test new_z[2, :] ≈ -0.1f0 .* g[2, :] atol = 1.0f-6
    end

    @testset "max_norm = Inf (never clips) ⇔ byte-equivalent to plain InferenceSGD" begin
        rng = MersenneTwister(3)
        z = randn(rng, Float32, 4, 5)
        g = randn(rng, Float32, 4, 5) .* 10          # large grads, would clip at any finite max_norm
        sgd = InferenceSGD(; eta_infer=0.05, latent_decay=0.01)
        clip = InferenceSGDNormClip(;
            eta_infer=0.05, latent_decay=0.01, max_norm=Float32(Inf), eps=1.0f-8
        )
        @test FabricPC.compute_new_latent(sgd, z, g) ≈
            FabricPC.compute_new_latent(clip, z, g)
    end

    @testset "latent_decay applied identically to both algorithms (isolated via zero grad)" begin
        rng = MersenneTwister(4)
        z = randn(rng, Float32, 2, 3)
        g = zeros(Float32, 2, 3)                     # zero grad ⇒ clip_factor irrelevant
        sgd = InferenceSGD(; eta_infer=0.1, latent_decay=0.2)
        clip = InferenceSGDNormClip(; eta_infer=0.1, latent_decay=0.2, max_norm=1.0)
        @test FabricPC.compute_new_latent(sgd, z, g) ≈
            FabricPC.compute_new_latent(clip, z, g)
        @test FabricPC.compute_new_latent(sgd, z, g) ≈ z .* (1.0f0 - 0.1f0 * 0.2f0)
    end

    @testset "acceptance: a small PC graph trains (energy decreases) under norm-clipped inference" begin
        rng = MersenneTwister(5)
        in_f, hid_f, out_f, B = 4, 6, 3, 8
        x = randn(rng, Float32, B, in_f)
        labels = rand(rng, 1:out_f, B)
        y = Float32[labels[b] == c ? 1 : 0 for b in 1:B, c in 1:out_f]
        xn = Linear((in_f,), "x")
        hn = Linear((hid_f,), "h"; activation=TanhActivation())
        yn = Linear(
            (out_f,), "y"; activation=SoftmaxActivation(), energy=CrossEntropyEnergy()
        )
        structure = graph(
            [xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)], TaskMap(; x=xn, y=yn),
            InferenceSGDNormClip(; eta_infer=0.1, infer_steps=30, max_norm=1.0)
        )
        params = initialize_params(structure, MersenneTwister(6))
        batch = Dict("x" => x, "y" => y)
        energies = Float32[]
        for _ in 1:80
            params, e, _ = train_step(params, batch, structure, 0.05, rng)
            push!(energies, e)
        end
        @test all(isfinite, energies)
        @test energies[end] < energies[1]
    end
end
