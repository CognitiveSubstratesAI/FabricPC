#!/usr/bin/env julia
# muPC FC-ResNet on MNIST (Phase F) — the full muPC training recipe.
#
# Deep fully-connected residual predictive-coding net, autodiff-free, with the
# COMPLETE muPC recipe (ports examples/mupc_demo.py):
#   input(784) → stem(H) → [N × LinearResidual(H, Tanh)] → output(10, Softmax/CE)
#   - MuPCInitializer (UNIT-variance weights) on stem + blocks
#   - per-edge muPC scaling (MuPCConfig(include_output=false)): hidden edges keep
#     activations O(1); the Softmax/CE output (Xavier init) is left unscaled
#   - AdamW (the adaptive optimizer muPC's weight_grad_scale=1 assumes)
#   - infer_steps = max(20, 3·(N+2)) — relaxation budget grows with depth
#   - exact Softmax+CrossEntropy gradient (dE/dpre = s − y) for the output
#
# This is the recipe that makes deep PC nets trainable (Innocenti et al.,
# arXiv:2505.13124). See docs/decisions.md §10.
#
# Run:  julia --project=. examples/mupc_resnet.jl
# Env:  FPC_BLOCKS (8), FPC_HID (64), FPC_NTRAIN (8000), FPC_NTEST (2000),
#       FPC_EPOCHS (5), FPC_LR (0.002), FPC_BATCH (256).

using FabricPC
using Random
using Printf
using Downloads

const MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
const CACHE = joinpath(homedir(), ".cache", "fabricpc_mnist")

function _fetch(name)
    mkpath(CACHE)
    raw = joinpath(CACHE, name)
    if !isfile(raw)
        gz = raw * ".gz"
        Downloads.download("$MNIST_BASE/$name.gz", gz)
        run(`gunzip -f $gz`)
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
    X = Matrix{Float32}(undef, n, px)
    @inbounds for i in 1:n, j in 1:px
        X[i, j] = b[16 + (i - 1) * px + j] / 255.0f0
    end
    return X
end
load_labels(name) = (b=_fetch(name); n=Int(_u32(b, 5)); Int.(b[9:(8 + n)]))

function build_resnet(num_blocks, hidden)
    winit = MuPCInitializer()
    input = IdentityNode((784,), "input")
    stem = Linear((hidden,), "stem"; weight_init=winit)
    nodes = Any[input, stem]
    edges = Any[Edge(input, stem)]
    prev = stem
    for i in 1:num_blocks
        res = LinearResidual(
            (hidden,), "res$i"; activation=TanhActivation(), weight_init=winit
        )
        push!(nodes, res)
        push!(edges, Edge(prev, res))
        push!(edges, Edge(prev, slot(res, "skip")))
        prev = res
    end
    out = Linear(
        (10,), "y";
        activation=SoftmaxActivation(), energy=CrossEntropyEnergy(),
        weight_init=XavierInitializer()
    )
    push!(nodes, out)
    push!(edges, Edge(prev, out))
    steps = max(20, 3 * (num_blocks + 2))
    return graph(
        nodes, edges, TaskMap(; x=input, y=out),
        InferenceSGD(; eta_infer=0.1, infer_steps=steps);
        scaling=MuPCConfig(; include_output=false)
    )
end

function main()
    num_blocks = parse(Int, get(ENV, "FPC_BLOCKS", "8"))
    hidden = parse(Int, get(ENV, "FPC_HID", "64"))
    n_train = parse(Int, get(ENV, "FPC_NTRAIN", "8000"))
    n_test = parse(Int, get(ENV, "FPC_NTEST", "2000"))
    epochs = parse(Int, get(ENV, "FPC_EPOCHS", "5"))
    lr = parse(Float32, get(ENV, "FPC_LR", "0.002"))
    bs = parse(Int, get(ENV, "FPC_BATCH", "256"))
    rng = MersenneTwister(0)

    Xtr = load_images("train-images-idx3-ubyte")[1:n_train, :]
    ytr = load_labels("train-labels-idx1-ubyte")[1:n_train]
    Xte = load_images("t10k-images-idx3-ubyte")[1:n_test, :]
    yte = load_labels("t10k-labels-idx1-ubyte")[1:n_test]
    Ytr = zeros(Float32, n_train, 10)
    for i in 1:n_train
        Ytr[i, ytr[i] + 1] = 1.0f0
    end

    structure = build_resnet(num_blocks, hidden)
    params = initialize_params(structure, MersenneTwister(1))
    opt = AdamW(params; lr=lr, weight_decay=0.01)
    @info "muPC FC-ResNet" num_blocks hidden n_train n_test epochs lr nodes = length(
        structure.nodes
    )

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
    for ep in 1:epochs
        order = randperm(rng, n_train)
        ep_e = 0.0f0
        nb = 0
        for s in 1:bs:n_train
            idx = order[s:min(s + bs - 1, n_train)]
            params, e, _ = train_step!(
                opt, params, Dict("x" => Xtr[idx, :], "y" => Ytr[idx, :]), structure, rng
            )
            ep_e += e
            nb += 1
        end
        @printf(
            "epoch %d  mean_energy=%.4f  test_acc=%.4f\n",
            ep,
            ep_e / nb,
            test_accuracy(params)
        )
    end
end

main()
