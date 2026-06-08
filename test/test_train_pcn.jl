# train_pcn / evaluate_pcn orchestration harness — backprop-free.
#
# Reuses the separable-blob classifier from test_classifier.jl but drives it
# through the train_pcn epoch/batch loop and the evaluate_pcn accuracy/energy
# harness (instead of a hand-rolled loop), confirming both are faithful to the
# local PC training step (no backprop).

using FabricPC
using Random, Test

@testset "train_pcn loop + evaluate_pcn harness (backprop-free)" begin
    rng = MersenneTwister(2024)
    n_features, n_class, per_class = 6, 3, 24
    batch = n_class * per_class
    centers = 2.5f0 .* randn(rng, Float32, n_class, n_features)
    X = zeros(Float32, batch, n_features)
    Y = zeros(Float32, batch, n_class)            # one-hot targets
    row = 1
    for c in 1:n_class, _ in 1:per_class
        X[row, :] = centers[c, :] .+ 0.3f0 .* randn(rng, Float32, n_features)
        Y[row, c] = 1.0f0
        row += 1
    end

    xn = Linear((n_features,), "x")
    hn = Linear((16,), "h"; activation=TanhActivation())
    yn = Linear((n_class,), "y")
    structure = graph(
        [xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn), InferenceSGD(; eta_infer=0.1, infer_steps=30)
    )
    params = initialize_params(structure, MersenneTwister(99))

    loader = [Dict("x" => X, "y" => Y)]           # one re-iterable batch

    ev0 = evaluate_pcn(params, structure, loader; rng=MersenneTwister(7))
    @test haskey(ev0, "accuracy") && haskey(ev0, "energy")

    # Train via the loop (plain-SGD path, lr=0.01 — matches the classifier test).
    params, iters, _ = train_pcn(
        params, structure, loader, 0.01; num_epochs=200, rng=MersenneTwister(7),
        verbose=false
    )
    ev1 = evaluate_pcn(params, structure, loader; rng=MersenneTwister(7))

    @test length(iters) == 200                    # 200 epochs
    @test length(iters[end]) == 1                 # one batch/epoch
    @test iters[end][1] < iters[1][1]             # per-batch energy fell over training
    @test ev1["accuracy"] > ev0["accuracy"]       # learned
    @test ev1["accuracy"] > 0.9                   # separable task nailed
    @test ev1["energy"] < ev0["energy"]           # energy dropped

    # Fractional num_epochs + AdamW optimizer path smoke.
    p2 = initialize_params(structure, MersenneTwister(99))
    opt = AdamW(p2; lr=0.01)
    p2, it2, _ = train_pcn(
        p2, structure, loader, opt; num_epochs=2, rng=MersenneTwister(7), verbose=false
    )
    @test length(it2) == 2

    # iter_callback is invoked per batch.
    seen = Int[]
    train_pcn(
        params, structure, loader, 0.01; num_epochs=3, rng=MersenneTwister(7),
        verbose=false, iter_callback=(ep, b, e) -> push!(seen, ep)
    )
    @test seen == [1, 2, 3]
end
