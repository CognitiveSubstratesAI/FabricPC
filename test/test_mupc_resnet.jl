# Full muPC training recipe (Phase F): a DEEP FC-ResNet trains end-to-end.
#
# This is the positive result closing the decisions.md §9 negative finding: with
# the COMPLETE recipe — MuPCInitializer (unit-variance) + per-edge muPC scaling +
# AdamW — a deep residual PC network trains to high accuracy. (The earlier muPC
# failure was a NormalInitializer(std=0.05) / muPC-scaling mismatch + plain SGD.)
#
# Self-contained, CI-safe: a separable synthetic task, ~1s. Reproduces the depth
# regime on a controlled problem; the exact MNIST accuracy table needs further
# input-normalization tuning (examples/mupc_resnet.jl, deferred).

using FabricPC: slot, train_step!, predict

function _build_mupc_resnet(num_blocks, hidden, d, n_class; mupc::Bool)
    winit = mupc ? MuPCInitializer() : NormalInitializer(; std=0.05)
    input = IdentityNode((d,), "input")
    stem = Linear((hidden,), "stem"; weight_init=winit)
    nodes = Any[input, stem]
    edges = Any[Edge(input, stem)]
    prev = stem
    for i in 1:num_blocks
        res = LinearResidual(
            (hidden,), "res$i"; activation=TanhActivation(), weight_init=winit
        )
        push!(nodes, res)
        push!(edges, Edge(prev, res))               # transform ("in") path
        push!(edges, Edge(prev, slot(res, "skip")))  # identity skip path
        prev = res
    end
    out = Linear((n_class,), "y"; weight_init=XavierInitializer())  # Gaussian one-hot
    push!(nodes, out)
    push!(edges, Edge(prev, out))
    steps = max(20, 3 * (num_blocks + 2))
    return graph(
        nodes, edges, TaskMap(; x=input, y=out),
        InferenceSGD(; eta_infer=0.1, infer_steps=steps);
        scaling=mupc ? MuPCConfig(; include_output=false) : nothing
    )
end

@testset "full muPC recipe: deep FC-ResNet trains" begin
    rng = MersenneTwister(1)
    d, n_class, per_class = 8, 3, 40
    batch = n_class * per_class
    centers = 2.5f0 .* randn(rng, Float32, n_class, d)
    X = zeros(Float32, batch, d)
    labels = zeros(Int, batch)
    Y = zeros(Float32, batch, n_class)
    row = 1
    for c in 1:n_class, _ in 1:per_class
        X[row, :] = centers[c, :] .+ 0.3f0 .* randn(rng, Float32, d)
        labels[row] = c
        Y[row, c] = 1.0f0
        row += 1
    end

    # 6 residual blocks — deep enough that the parameterization matters.
    structure = _build_mupc_resnet(6, 16, d, n_class; mupc=true)
    params = initialize_params(structure, MersenneTwister(7))
    opt = AdamW(params; lr=0.01, weight_decay=0.0)

    accuracy(p) = begin
        pred = predict(p, structure, Dict("x" => X), MersenneTwister(3))
        sum(argmax(pred[i, :]) == labels[i] for i in 1:batch) / batch
    end

    acc0 = accuracy(params)
    energies = Float32[]
    for _ in 1:150
        params, e, _ = train_step!(
            opt, params, Dict("x" => X, "y" => Y), structure, MersenneTwister(3)
        )
        push!(energies, e)
    end
    acc_final = accuracy(params)

    @test all(isfinite, energies)
    @test energies[end] < 0.1f0 * energies[1]   # energy descends (not the §9 ascent)
    @test acc_final > acc0
    @test acc_final > 0.9                        # deep muPC net learns the task
end
