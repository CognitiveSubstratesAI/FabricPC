# Classifier harness test (Phase E) — self-contained, no data dependency.
#
# Trains a PC classifier (Gaussian one-hot output, the standard PC-classification
# setup) on a deterministic, well-separated synthetic dataset and checks that
# `predict` + argmax recovers the classes. This exercises the same pipeline the
# MNIST exhibit (examples/mnist_pc.jl) uses, but runs in CI in well under a second.

using FabricPC: predict

@testset "PC classifier learns a separable synthetic task" begin
    rng = MersenneTwister(2024)
    n_features, n_class, per_class = 6, 3, 24
    batch = n_class * per_class

    # Well-separated Gaussian blobs: one random center per class + small noise.
    centers = 2.5f0 .* randn(rng, Float32, n_class, n_features)
    X = zeros(Float32, batch, n_features)
    labels = zeros(Int, batch)
    Y = zeros(Float32, batch, n_class)            # one-hot targets
    row = 1
    for c in 1:n_class, _ in 1:per_class
        X[row, :] = centers[c, :] .+ 0.3f0 .* randn(rng, Float32, n_features)
        labels[row] = c
        Y[row, c] = 1.0f0
        row += 1
    end

    xn = Linear((n_features,), "x")
    hn = Linear((16,), "h"; activation=TanhActivation())
    yn = Linear((n_class,), "y")                  # Identity + Gaussian (MSE-to-one-hot)
    structure = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=30)
    )
    params = initialize_params(structure, MersenneTwister(99))

    accuracy(p) = begin
        pred = predict(p, structure, Dict("x" => X), MersenneTwister(7))
        sum(argmax(pred[i, :]) == labels[i] for i in 1:batch) / batch
    end

    acc_before = accuracy(params)
    batch_dict = Dict("x" => X, "y" => Y)
    # lr = 0.01: a Tanh-hidden PC classifier diverges to NaN around lr ≳ 0.05 here,
    # but converges cleanly at 0.01 (energy → ~5e-4, accuracy → 1.0).
    for _ in 1:200
        params, _, _ = train_step(params, batch_dict, structure, 0.01, MersenneTwister(7))
    end
    acc_after = accuracy(params)

    @test acc_after > acc_before
    @test acc_after > 0.9          # separable task ⇒ a small PC MLP nails it
end
