#!/usr/bin/env julia
# PC-vs-backprop A/B comparison on the MNIST-MLP architecture (docs/AUDIT_REGISTER.md C-04).
#
# C-04's own framing (see the register row + docs/decisions.md §24 in full before touching
# this file): upstream's `ab_experiment.py`/`statistics.py` treat each arm's `params`/
# `structure` as fully OPAQUE duck-typed objects -- a Julia arm on a plain backprop-trained
# MLP never touches FabricPC internals and is not a "port of `train_backprop.py`" in any
# sense that conflicts with this project's own no-backprop-port stance for the PC engine
# itself. This file is new code (loosely mirroring `ExperimentArm`/`ABExperiment`'s shape,
# not a line-by-line port), not a conformance port.
#
# CONFOUND SCOPE (decisions.md §24): a PC arm built on a TRANSFORMER at its own
# `eta_infer=0.1` default would benchmark a non-converged relaxation (~5.7x past its own
# stability limit), not the algorithm. The MNIST-MLP config used here is DIFFERENT --
# §6/§24's own control measurement found it CONTRACTIVE at its actual `eta_infer=0.1`
# (kappa~=0.902/step) -- so this specific comparison carries no such confound. The PC arm's
# hyperparameters below are FIXED at `examples/mnist_pc.jl`'s own "plain" (non-muPC) defaults
# (lr=0.002, eta_infer=0.1, infer_steps=30) -- deliberately not re-tuned for this comparison.
#
# Arms:
#   - PC:        FabricPC's own `train_pcn`/`evaluate_pcn` (src/training/train.jl), plain SGD,
#                x(784) -> h(128,tanh) -> y(10, Identity+Gaussian/MSE-to-one-hot).
#   - Backprop:  a plain Lux.jl MLP with the IDENTICAL shapes/activation (Dense(784=>128,tanh),
#                Dense(128=>10)), MSE-to-one-hot loss (matching the PC arm's Gaussian energy
#                exactly, so both arms optimize the SAME objective under different learning
#                rules), trained via ordinary backprop (Zygote autodiff through the whole
#                network) with Adam.
#
# Same real MNIST data for both arms (examples/mnist_pc.jl's download/IDX-parse code, reused
# via `include` -- that file guards its own `main()` call so including it here only defines
# the loader helpers). Each arm is run over N_TRIALS seeds (paired: trial i uses the SAME
# batch-shuffle order for both arms, matching upstream's `data_loader_factory(seed)` calling
# convention), and compared via a paired t-test + Cohen's d (HypothesisTests.jl) on final test
# accuracy, mirroring `fabricpc/experiments/statistics.py`.
#
# Own dedicated environment (Lux.jl/HypothesisTests.jl don't belong in FabricPC's own
# Project.toml -- they are comparison-harness-only, like `benchmark/jit/Project.toml`'s
# existing precedent for scoped benchmark deps):
#   julia --project=benchmark/pc_vs_backprop benchmark/pc_vs_backprop_mnist.jl
# Or drive interactively via the dedicated warm session (tools/warm_session_pcvb.jl /
# tools/warm_send_pcvb.sh).
#
# Tunables via env: PCVB_NTRAIN, PCVB_NTEST, PCVB_EPOCHS, PCVB_BATCH, PCVB_TRIALS, PCVB_SEED0.

using FabricPC
using Lux
using Zygote
using Optimisers
using Random
using Printf
using Statistics
using HypothesisTests

# Reuse mnist_pc.jl's data-loading helpers (`_fetch`/`load_images`/`load_labels`/`onehot`)
# without triggering its own `main()` (guarded by `abspath(PROGRAM_FILE) == @__FILE__`).
include(joinpath(@__DIR__, "..", "examples", "mnist_pc.jl"))

# ── config ───────────────────────────────────────────────────────────────────────────────

const N_TRAIN = parse(Int, get(ENV, "PCVB_NTRAIN", "5000"))
const N_TEST = parse(Int, get(ENV, "PCVB_NTEST", "2000"))
const EPOCHS = parse(Int, get(ENV, "PCVB_EPOCHS", "6"))
const BATCH = parse(Int, get(ENV, "PCVB_BATCH", "64"))
const N_TRIALS = parse(Int, get(ENV, "PCVB_TRIALS", "8"))
const SEED0 = parse(Int, get(ENV, "PCVB_SEED0", "0"))

# PC arm -- FIXED at mnist_pc.jl's own plain (non-muPC) default. NOT re-tuned here (see
# header comment): measured contractive at this exact eta_infer this session.
const PC_LR = 0.002f0
const PC_ETA_INFER = 0.1
const PC_INFER_STEPS = 30

# Backprop arm -- Adam, a standard choice for a small MLP; not swept/tuned, documented here
# so the comparison is reproducible rather than silently cherry-picked.
const BP_LR = 1.0f-3

# ── shared data loader (same shapes/contract for both arms) ────────────────────────────────
#
# Mirrors `CharDataLoader`'s (examples/char_lm_pc.jl) reshuffle-on-each-`iterate()` pattern,
# for dense (batch, features) MNIST arrays instead of token sequences. Drops the last
# incomplete batch, matching that same precedent.

mutable struct MNISTLoader
    X::Matrix{Float32}
    Y::Matrix{Float32}
    batch_size::Int
    shuffle::Bool
    seed::Union{Nothing, Int}
    epoch::Int
    n::Int
    num_batches::Int
end

function MNISTLoader(X, Y, batch_size::Integer; shuffle::Bool=true,
    seed::Union{Nothing, Integer}=nothing)
    n = size(X, 1)
    return MNISTLoader(X, Y, Int(batch_size), shuffle,
        seed === nothing ? nothing : Int(seed), 0, n, n ÷ batch_size)
end

Base.length(l::MNISTLoader) = l.num_batches
Base.eltype(::Type{MNISTLoader}) = Dict{String, Any}

function Base.iterate(l::MNISTLoader)
    idx = collect(1:(l.n))
    if l.shuffle
        rng = l.seed === nothing ? Random.default_rng() : MersenneTwister(l.seed + l.epoch)
        Random.shuffle!(rng, idx)
    end
    l.epoch += 1
    return iterate(l, (idx, 1))
end

function Base.iterate(l::MNISTLoader, (idx, pos))
    stop = pos + l.batch_size - 1
    stop > length(idx) && return nothing               # drop incomplete last batch
    rows = @view idx[pos:stop]
    return Dict{String, Any}("x" => l.X[rows, :], "y" => l.Y[rows, :]), (idx, stop + 1)
end

# `DataLoaderFactory`: seed -> (train_loader, test_loader). Same signature/contract as
# upstream's `ab_experiment.py` (`data_loader_factory(seed)`), called once per arm per trial
# "for isolation" -- but since both calls pass the SAME `trial_seed`, `MNISTLoader`'s
# `seed`-derived shuffling reproduces the IDENTICAL batch order for both arms (paired design).
function mnist_loaders(seed::Integer, Xtr, Ytr, Xte, Yte)
    train_loader = MNISTLoader(Xtr, Ytr, BATCH; shuffle=true, seed=seed)
    test_loader = MNISTLoader(Xte, Yte, BATCH; shuffle=false)
    return train_loader, test_loader
end

# ── PC arm ───────────────────────────────────────────────────────────────────────────────

function pc_model_factory(rng::AbstractRNG)
    xn = Linear((784,), "x")
    hn = Linear((128,), "h"; activation=TanhActivation())
    yn = Linear((10,), "y")                             # Identity + Gaussian (MSE to one-hot)
    structure = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=PC_ETA_INFER, infer_steps=PC_INFER_STEPS)
    )
    params = initialize_params(structure, rng)
    return params, structure
end

function run_pc_arm(trial_seed::Integer, train_loader, test_loader)
    params, structure = pc_model_factory(MersenneTwister(trial_seed))
    t0 = time()
    trained, _, _ = train_pcn(
        params, structure, train_loader, PC_LR;
        num_epochs=EPOCHS, rng=MersenneTwister(trial_seed + 1), verbose=false
    )
    train_time = time() - t0
    metrics = evaluate_pcn(trained, structure, test_loader; rng=MersenneTwister(trial_seed + 2))
    return metrics, train_time
end

# ── backprop arm (Lux.jl, identical architecture, MSE-to-one-hot loss) ─────────────────────

bp_model() = Lux.Chain(Lux.Dense(784 => 128, tanh), Lux.Dense(128 => 10))

# Objective for Lux.Training: (model, ps, st, data) -> (loss, updated_state, stats). Same
# Gaussian/MSE energy as the PC arm's output node, so both arms train toward the SAME
# objective under different learning rules.
function bp_loss(model, ps, st, (x, y))
    ŷ, st2 = model(x, ps, st)
    loss = mean(abs2, ŷ .- y)
    return loss, st2, (;)
end

function run_bp_arm(trial_seed::Integer, train_loader, test_loader)
    rng = MersenneTwister(trial_seed)
    model = bp_model()
    ps, st = Lux.setup(rng, model)
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(BP_LR))

    t0 = time()
    for _ in 1:EPOCHS
        for batch in train_loader
            xb = permutedims(Float32.(batch["x"]))      # Lux: (features, batch)
            yb = permutedims(Float32.(batch["y"]))
            _, _, _, train_state = Lux.Training.single_train_step!(
                AutoZygote(), bp_loss, (xb, yb), train_state
            )
        end
    end
    train_time = time() - t0

    st_eval = Lux.testmode(train_state.states)
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in test_loader
        xb = permutedims(Float32.(batch["x"]))
        yb = permutedims(Float32.(batch["y"]))
        ŷ, _ = train_state.model(xb, train_state.parameters, st_eval)
        for i in 1:size(ŷ, 2)
            correct += (argmax(@view ŷ[:, i]) == argmax(@view yb[:, i])) ? 1 : 0
        end
        total += size(ŷ, 2)
        loss_sum += sum(abs2, ŷ .- yb)
    end
    metrics = Dict("accuracy" => correct / total, "energy" => loss_sum / total)
    return metrics, train_time
end

# ── A/B trial loop (loosely mirrors ABExperiment.run / ExperimentArm) ──────────────────────

struct TrialResult
    metric_value::Float64
    train_time::Float64
    all_metrics::Dict{String, Float64}
end

function main()
    @info "loading MNIST" N_TRAIN N_TEST EPOCHS BATCH N_TRIALS
    Xtr = load_images("train-images-idx3-ubyte")[1:N_TRAIN, :]
    ytr = load_labels("train-labels-idx1-ubyte")[1:N_TRAIN]
    Xte = load_images("t10k-images-idx3-ubyte")[1:N_TEST, :]
    yte = load_labels("t10k-labels-idx1-ubyte")[1:N_TEST]
    Ytr = onehot(ytr, 10)
    Yte = onehot(yte, 10)

    pc_trials = TrialResult[]
    bp_trials = TrialResult[]
    seeds = Int[]

    total_t0 = time()
    for trial in 1:N_TRIALS
        trial_seed = SEED0 + (trial - 1) * 1000
        push!(seeds, trial_seed)
        @printf("--- Trial %d/%d (seed=%d) ---\n", trial, N_TRIALS, trial_seed)

        train_loader, test_loader = mnist_loaders(trial_seed, Xtr, Ytr, Xte, Yte)
        pc_metrics, pc_time = run_pc_arm(trial_seed, train_loader, test_loader)
        push!(pc_trials, TrialResult(pc_metrics["accuracy"], pc_time, pc_metrics))
        @printf("  PC:        accuracy=%.4f  energy=%.4f  (train: %.1fs)\n",
            pc_metrics["accuracy"], pc_metrics["energy"], pc_time)

        train_loader, test_loader = mnist_loaders(trial_seed, Xtr, Ytr, Xte, Yte)  # fresh, isolated
        bp_metrics, bp_time = run_bp_arm(trial_seed, train_loader, test_loader)
        push!(bp_trials, TrialResult(bp_metrics["accuracy"], bp_time, bp_metrics))
        @printf("  Backprop:  accuracy=%.4f  energy=%.4f  (train: %.1fs)\n",
            bp_metrics["accuracy"], bp_metrics["energy"], bp_time)
    end
    total_time = time() - total_t0

    pc_acc = [t.metric_value for t in pc_trials]
    bp_acc = [t.metric_value for t in bp_trials]

    println()
    println("=" ^ 70)
    println("A/B Experiment: PC vs Backprop  (MNIST-MLP, 784->128(tanh)->10)")
    println("=" ^ 70)
    @printf("Trials: %d   Epochs/trial: %d   Design: paired (same seed/batch-order per trial)\n",
        N_TRIALS, EPOCHS)
    @printf("PC arm:       lr=%.4f  eta_infer=%.3f  infer_steps=%d  (plain SGD)\n",
        PC_LR, PC_ETA_INFER, PC_INFER_STEPS)
    @printf("Backprop arm: lr=%.4f  optimizer=Adam\n", BP_LR)
    println()
    println("Trial   Seed    PC acc(%)   Backprop acc(%)   Diff(%)")
    for i in 1:N_TRIALS
        diff = (pc_acc[i] - bp_acc[i]) * 100
        @printf("%-7d %-7d %-11.2f %-17.2f %+.2f\n", i, seeds[i], pc_acc[i] * 100,
            bp_acc[i] * 100, diff)
    end
    println()
    @printf("PC:       %.2f +/- %.2f%%  (mean +/- SE, SD=%.2f%%)\n",
        mean(pc_acc) * 100, (std(pc_acc) / sqrt(N_TRIALS)) * 100, std(pc_acc) * 100)
    @printf("Backprop: %.2f +/- %.2f%%  (mean +/- SE, SD=%.2f%%)\n",
        mean(bp_acc) * 100, (std(bp_acc) / sqrt(N_TRIALS)) * 100, std(bp_acc) * 100)

    if N_TRIALS >= 2
        tt = OneSampleTTest(pc_acc, bp_acc)               # paired t-test on the differences
        diff = pc_acc .- bp_acc
        d = mean(diff) / std(diff)                        # Cohen's d (paired), SD of differences
        magnitude = abs(d) >= 0.8 ? "large" : abs(d) >= 0.5 ? "medium" :
                    abs(d) >= 0.2 ? "small" : "negligible"

        println()
        println("--- Paired t-test (HypothesisTests.jl OneSampleTTest on the differences) ---")
        @printf("Mean difference (PC - Backprop): %+.2f%%\n", mean(diff) * 100)
        @printf("t-statistic: %.4f\n", tt.t)
        @printf("p-value: %.4f, N = %d\n", pvalue(tt), N_TRIALS)
        @printf("Significant at p<0.05: %s\n", pvalue(tt) < 0.05 ? "YES" : "NO")

        println()
        println("--- Effect size ---")
        @printf("Cohen's d (paired): %.4f\n", d)
        println("Interpretation: $magnitude")
    else
        println()
        println("--- Statistical tests require N_TRIALS >= 2 ---")
    end

    println()
    @printf("Total wall time: %.1fs\n", total_time)
    println("=" ^ 70)

    return (; pc_trials, bp_trials, seeds, total_time)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
