#!/usr/bin/env julia
# MNIST predictive-coding classifier exhibit (Phase E).
#
# A self-contained demonstration of the FabricPC stack on real MNIST:
#   x(784) → h(128, Tanh) → y(10, Gaussian/MSE-to-one-hot)
# trained with local PC learning (inference relaxation + local weight updates),
# NO autodiff. Uses the standard PC-classification setup (Gaussian energy on a
# one-hot target — exact gradients, clean energy descent), which sidesteps the
# diagonal-Softmax approximation.
#
# Dependency-light: downloads the MNIST IDX files via the `Downloads` stdlib and
# `gunzip`, then parses the IDX format directly — no MLDatasets / CodecZlib.
#
# Run (from the repo root):
#   julia --project=. examples/mnist_pc.jl
# Tunables via env: FPC_NTRAIN, FPC_NTEST, FPC_EPOCHS, FPC_LR, FPC_BATCH.

using FabricPC
using Random
using Printf
using Downloads

const MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
const CACHE = joinpath(homedir(), ".cache", "fabricpc_mnist")

# ── data: download + gunzip + IDX parse ────────────────────────────────────

function _fetch(name)
    mkpath(CACHE)
    raw = joinpath(CACHE, name)            # decompressed IDX
    if !isfile(raw)
        gz = raw * ".gz"
        @info "downloading $name"
        Downloads.download("$MNIST_BASE/$name.gz", gz)
        run(`gunzip -f $gz`)               # → writes `raw`, removes `gz`
    end
    return read(raw)
end

_u32(b, i) =
    (UInt32(b[i]) << 24) | (UInt32(b[i + 1]) << 16) | (UInt32(b[i + 2]) << 8) |
    UInt32(b[i + 3])

function load_images(name)
    b = _fetch(name)
    n, rows, cols = Int(_u32(b, 5)), Int(_u32(b, 9)), Int(_u32(b, 13))
    px = rows * cols
    # Batch-first (n, 784), normalized to [0,1].
    X = Matrix{Float32}(undef, n, px)
    @inbounds for i in 1:n, j in 1:px
        X[i, j] = b[16 + (i - 1) * px + j] / 255.0f0
    end
    return X
end

function load_labels(name)
    b = _fetch(name)
    n = Int(_u32(b, 5))
    return Int.(b[9:(8 + n)])                 # 0-9
end

onehot(labels, n_class) = begin
    Y = zeros(Float32, length(labels), n_class)
    for (i, c) in enumerate(labels)
        Y[i, c + 1] = 1.0f0
    end
    Y
end

# ── build + train + eval ────────────────────────────────────────────────────

function main()
    # Defaults tuned for stability: with 784 fan-in, lr ≳ 0.005 or eta_infer too
    # large diverges to NaN (the regime muPC is meant to fix — but muPC's PC
    # inference needs different relaxation settings than these eager defaults, so
    # the exhibit ships the plain, stable config). lr=0.002 / eta_infer=0.1.
    n_train = parse(Int, get(ENV, "FPC_NTRAIN", "5000"))
    n_test = parse(Int, get(ENV, "FPC_NTEST", "2000"))
    epochs = parse(Int, get(ENV, "FPC_EPOCHS", "6"))
    lr = parse(Float32, get(ENV, "FPC_LR", "0.002"))
    bs = parse(Int, get(ENV, "FPC_BATCH", "64"))
    rng = MersenneTwister(0)

    Xtr = load_images("train-images-idx3-ubyte")[1:n_train, :]
    ytr = load_labels("train-labels-idx1-ubyte")[1:n_train]
    Xte = load_images("t10k-images-idx3-ubyte")[1:n_test, :]
    yte = load_labels("t10k-labels-idx1-ubyte")[1:n_test]
    Ytr = onehot(ytr, 10)
    @info "MNIST loaded" n_train n_test epochs lr bs

    xn = Linear((784,), "x")
    hn = Linear((128,), "h"; activation=TanhActivation())
    yn = Linear((10,), "y")                 # Identity + Gaussian (MSE to one-hot)
    structure = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=30)
    )
    params = initialize_params(structure, MersenneTwister(1))

    function test_accuracy(p)
        correct = 0
        for s in 1:bs:n_test
            e = min(s + bs - 1, n_test)
            pred = predict(p, structure, Dict("x" => Xte[s:e, :]), rng)
            for (k, i) in enumerate(s:e)
                correct += (argmax(pred[k, :]) - 1) == yte[i]
            end
        end
        return correct / n_test
    end

    @printf("epoch 0  test_acc=%.4f  (random init)\n", test_accuracy(params))
    nb = 0
    for ep in 1:epochs
        order = randperm(rng, n_train)
        ep_energy = 0.0f0
        for s in 1:bs:n_train
            idx = order[s:min(s + bs - 1, n_train)]
            batch = Dict("x" => Xtr[idx, :], "y" => Ytr[idx, :])
            params, e, _ = train_step(params, batch, structure, lr, rng)
            ep_energy += e
            nb += 1
        end
        @printf("epoch %d  mean_energy=%.4f  test_acc=%.4f\n", ep,
            ep_energy / cld(n_train, bs), test_accuracy(params))
    end
    @printf("\nDONE  %d train steps  final test_acc=%.4f\n", nb, test_accuracy(params))
end

main()
