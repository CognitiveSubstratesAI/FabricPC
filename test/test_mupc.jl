# muPC scaling tests (Phase C).
#
# 1. Forward-scale formula on a plain chain (a = gain/√(fan_in·K·L), L=1).
# 2. Output-node scaling (include_output ⇒ O(1/N) formula) + exclusion default.
# 3. Residual depth L from skip-connection merge nodes; skip edges unscaled.
# 4. HEADLINE: O(1) activation variance under width scaling with muPC on vs the
#    √(fan_in) blow-up with muPC off.
# 5. muPC-on training still converges (the scaling wiring is self-consistent).

using FabricPC
using Test
using Random
using FabricPC: initialize_graph_state, _count_skip_connections_depth

@testset "muPC forward-scale formula — plain chain" begin
    xn = Linear((8,), "x")
    hn = Linear((16,), "h")
    yn = Linear((4,), "y")
    st = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD();
        scaling=MuPCConfig()       # include_output = false
    )

    @test st.infos["x"].scaling_config === nothing   # terminal input
    @test st.infos["y"].scaling_config === nothing   # output excluded by default

    sc = st.infos["h"].scaling_config
    @test sc isa MuPCScalingFactors
    @test sc.forward_scale["x->h:in"] ≈ 1.0f0 / sqrt(8.0f0)   # gain=1, K=1, L=1
    @test sc.self_grad_scale == 1.0f0
    @test sc.topdown_grad_scale["x->h:in"] ≈ sc.forward_scale["x->h:in"]  # jac=1
    @test sc.weight_grad_scale["x->h:in"] == 1.0f0
end

@testset "muPC output scaling — include_output" begin
    xn = Linear((8,), "x")
    hn = Linear((16,), "h")
    yn = Linear((4,), "y")
    st = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD();
        scaling=MuPCConfig(; include_output=true)
    )
    # Output formula a = gain / (fan_in · √(K·L)) = 1/(16·1)
    @test st.infos["y"].scaling_config.forward_scale["h->y:in"] ≈ 1.0f0 / 16.0f0
end

@testset "muPC residual depth + unscaled skip edges" begin
    xn = Linear((8,), "x")
    r1 = LinearResidual((8,), "r1")
    r2 = LinearResidual((8,), "r2")
    yn = Linear((4,), "y")
    edges = [
        Edge(xn, r1),
        Edge(xn, slot(r1, "skip")),
        Edge(r1, r2),
        Edge(r1, slot(r2, "skip")),
        Edge(r2, yn)
    ]
    st = graph([xn, r1, r2, yn], edges, TaskMap(; x=xn, y=yn), InferenceSGD())

    # Two skip-merge nodes (r1, r2) along the path ⇒ L = 2.
    @test _count_skip_connections_depth(st.infos, st.edges, st.node_order) == 2

    st2 = graph(
        [xn, r1, r2, yn],
        edges,
        TaskMap(; x=xn, y=yn),
        InferenceSGD();
        scaling=MuPCConfig()
    )
    sc = st2.infos["r1"].scaling_config
    # a = gain/√(fan_in·K·L) = 1/√(8·1·2) = 0.25
    @test sc.forward_scale["x->r1:in"] ≈ 0.25f0
    @test !haskey(sc.forward_scale, "x->r1:skip")   # skip slot unscaled (omitted)
end

@testset "muPC keeps activation variance O(1) under width scaling" begin
    rms(a) = sqrt(sum(abs2.(a)) / length(a))

    function hidden_pre_rms(W; mupc::Bool)
        batch = 64
        xn = Linear((W,), "x")
        hn = Linear((32,), "h"; weight_init=NormalInitializer(; std=1.0))
        yn = Linear((4,), "y")
        st = graph(
            [xn, hn, yn],
            [Edge(xn, hn), Edge(hn, yn)],
            TaskMap(; x=xn, y=yn),
            InferenceSGD();
            scaling=mupc ? MuPCConfig() : nothing
        )
        params = initialize_params(st, MersenneTwister(4))
        xdata = randn(MersenneTwister(5), Float32, batch, W)
        state = initialize_graph_state(
            st,
            batch,
            MersenneTwister(6);
            clamps=Dict{String, Any}("x" => xdata),
            params=params
        )
        # FeedforwardStateInit stores z_mu (not pre_activation); for identity
        # activation z_mu == pre_activation, the quantity whose variance muPC controls.
        return rms(state.nodes["h"].z_mu)
    end

    on_64 = hidden_pre_rms(64; mupc=true)
    on_1024 = hidden_pre_rms(1024; mupc=true)
    off_64 = hidden_pre_rms(64; mupc=false)
    off_1024 = hidden_pre_rms(1024; mupc=false)

    # muPC ON: activation RMS ~ 1 at both widths, near width-invariant.
    @test 0.5 < on_64 < 2.0
    @test 0.5 < on_1024 < 2.0
    @test on_1024 / on_64 < 1.5

    # muPC OFF: RMS grows ~ √(fan_in) — a clear blow-up the scaling prevents.
    @test off_1024 / off_64 > 2.5          # expect ~√(1024/64) = 4
    @test off_1024 > 5.0 * on_1024
end

@testset "muPC-on training converges (wiring is self-consistent)" begin
    data_rng = MersenneTwister(2)
    param_rng = MersenneTwister(8)
    in_f, hid_f, out_f, batch = 4, 6, 3, 16
    x = randn(data_rng, Float32, batch, in_f)
    y = x * randn(data_rng, Float32, in_f, out_f)

    xn = Linear((in_f,), "x")
    hn = Linear((hid_f,), "h")
    yn = Linear((out_f,), "y")
    st = graph(
        [xn, hn, yn],
        [Edge(xn, hn), Edge(hn, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=0.1, infer_steps=30);
        scaling=MuPCConfig(; include_output=true)
    )
    params = initialize_params(st, param_rng)
    batch_dict = Dict("x" => x, "y" => y)

    energies = Float32[]
    for _ in 1:200
        params, e, _ = train_step(params, batch_dict, st, 0.05, param_rng)
        push!(energies, e)
    end
    @test all(isfinite, energies)
    @test energies[end] < energies[1]
end
