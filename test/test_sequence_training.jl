# Sequence / autoregressive training (port of fabricpc/training/train_autoregressive.py).
# Phase 1: the shared foundation — create_causal_mask + compute_loss — cross-checked vs upstream.
using Test
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
