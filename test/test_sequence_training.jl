# Sequence / autoregressive training (port of fabricpc/training/train_autoregressive.py).
# Phase 1: the shared foundation — create_causal_mask + compute_loss — cross-checked vs upstream.
using Test
using Random
using FabricPC
import FabricPC: GraphState, NodeState

@testset "sequence foundation (train_autoregressive.py:29,40)" begin
    # create_causal_mask: lower-triangular ones, mask[i,j] = 1 iff j ≤ i
    @test create_causal_mask(3) == Float32[1 0 0; 1 1 0; 1 1 1]
    @test create_causal_mask(1) == reshape(Float32[1], 1, 1)
    m = create_causal_mask(5)
    @test all(m[i, j] == (j <= i ? 1.0f0 : 0.0f0) for i in 1:5, j in 1:5)

    # compute_loss CE vs upstream formula: one-hot target [1 0 0], pred [0.7 0.2 0.1] → −log(0.7)
    pred = Float32[0.7 0.2 0.1]
    tgt = Float32[1 0 0]
    st = GraphState(Dict("y" => NodeState(nothing, pred, nothing, nothing, nothing, nothing)), 1)
    @test compute_loss(st, tgt, "y") ≈ -log(0.7f0) atol = 1.0f-5
    @test compute_loss(st, tgt, "y"; loss_type = :mse) ≈ (0.3f0^2 + 0.2f0^2 + 0.1f0^2) / 3 atol = 1.0f-6

    # batch mean: two identical rows → same scalar loss
    st2 = GraphState(
        Dict("y" => NodeState(nothing, Float32[0.7 0.2 0.1; 0.7 0.2 0.1],
            nothing, nothing, nothing, nothing)), 2)
    @test compute_loss(st2, Float32[1 0 0; 1 0 0], "y") ≈ -log(0.7f0) atol = 1.0f-5

    @test_throws ArgumentError compute_loss(st, tgt, "y"; loss_type = :bogus)
end

@testset "autoregressive trainer reduces energy + CE (step + driver)" begin
    data_rng = MersenneTwister(7)
    param_rng = MersenneTwister(11)
    in_f, hid_f, out_f, B = 4, 8, 3, 16
    x = randn(data_rng, Float32, B, in_f)
    labels = rand(data_rng, 1:out_f, B)                       # a learnable classification map
    y = Float32[labels[b] == c ? 1 : 0 for b in 1:B, c in 1:out_f]   # one-hot targets

    xn = Linear((in_f,), "x")
    hn = Linear((hid_f,), "h"; activation = TanhActivation())
    yn = Linear((out_f,), "y"; activation = SoftmaxActivation(), energy = CrossEntropyEnergy())
    inference = InferenceSGD(; eta_infer = 0.1, infer_steps = 30)
    structure = graph([xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)], TaskMap(; x = xn, y = yn), inference)
    params = initialize_params(structure, param_rng)
    batch = Dict("x" => x, "y" => y)

    # train_step_autoregressive returns (params, energy, ce, state); both fall over training
    energies = Float32[]; ces = Float32[]
    for _ in 1:150
        params, e, ce, _ = train_step_autoregressive(params, 0.05, batch, structure, param_rng)
        push!(energies, e); push!(ces, ce)
    end
    @test all(isfinite, energies) && all(isfinite, ces)
    @test energies[end] < 0.5f0 * energies[1]                 # energy decreased (PC learning)
    @test ces[end] < ces[1]                                   # CE / perplexity metric decreased

    # driver: runs, returns (params, iter_energies, epoch_results) of the right shape
    _, iters, eres = train_autoregressive(params, structure, [batch], 0.05, 3, param_rng; verbose = false)
    @test length(iters) == 3 && length(iters[1]) == 1
    @test length(eres) == 3
    # epoch_callback fires once per epoch with the epoch index
    _, _, eres2 = train_autoregressive(params, structure, [batch], 0.05, 2, param_rng;
        verbose = false, epoch_callback = (ep, p, s) -> ep)
    @test eres2 == [1, 2]
end
