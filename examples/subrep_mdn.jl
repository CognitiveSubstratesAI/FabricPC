# subrep_mdn.jl — the SubRep Motive Decomposition Network (MDN) on FabricPC.
#
# The MDN co-learns the "motive geometry" — the cone Wₓ of plausible motive weights
# — from context, so SubRep's CDS/PDS gates screen options on a LEARNED slice that
# tightens with experience (SubRep paper §2.6; whitepaper §5.9). Per the whitepaper
# §2.4 this is the "outside" integration path: a Neural Space (FabricPC) whose output
# cone is exposed AS ATOMS to the symbolic CDS/PDS gates (lib/subrep/*.metta).
#
# Architecture mirrors the iCog reference generator/mdn.py: a shared trunk
#   Linear(ctx, h) → ReLU → Linear(h, h) → ReLU
# then a `support_head` Linear(h, num_obj) → Softplus producing the per-objective cone
# bounds (support values > 0, matching the iCog generator/mdn.py softplus head). Trained
# by LOCAL predictive coding (train_pcn) — no backprop — exactly the §3 "PC implements
# the MDN locally" requirement.
#
# Run: julia --project=. examples/subrep_mdn.jl

using FabricPC
using Random

# ── Build the MDN as a predictive-coding graph ───────────────────────────────
function mdn_graph(; context_dim::Int=2, num_obj::Int=2, hidden::Int=16,
                   infer_steps::Int=30, eta_infer::Float64=0.1)
    xn = Linear((context_dim,), "x")                                   # context input (source)
    h1 = Linear((hidden,), "h1"; activation=ReLUActivation())          # trunk layer 1
    h2 = Linear((hidden,), "h2"; activation=ReLUActivation())          # trunk layer 2
    yn = Linear((num_obj,), "y"; activation=SoftplusActivation())      # support_head → cone bounds >0 (iCog softplus)
    graph(
        [xn, h1, h2, yn],
        [Edge(xn, h1), Edge(h1, h2), Edge(h2, yn)],
        TaskMap(; x=xn, y=yn),
        InferenceSGD(; eta_infer=eta_infer, infer_steps=infer_steps),
    )
end

# ── A tiny synthetic motive-context task ─────────────────────────────────────
# Context type-1 (e.g. "raid imminent") should yield a cone emphasizing motive-1
# (allow its weight up to 1.0, motive-2 only to 0.2); type-2 ("trade window") the
# reverse. The MDN must learn context → cone-upper-bounds u (with ℓ=0 ⟹ box [0,u]).
const _CTX1 = Float32[1, 0]
const _CTX2 = Float32[0, 1]
const _CONE1 = Float32[1.0, 0.2]      # emphasize motive-1
const _CONE2 = Float32[0.2, 1.0]      # emphasize motive-2

function make_data(n::Int)
    X = zeros(Float32, 2n, 2)
    Y = zeros(Float32, 2n, 2)
    for i in 1:n
        X[i, :] .= _CTX1; Y[i, :] .= _CONE1
        X[n + i, :] .= _CTX2; Y[n + i, :] .= _CONE2
    end
    X, Y
end

# ── Expose the learned cone AS ATOMS (whitepaper §2.4: Neural Space → atoms) ──
# The MDN's per-objective bounds become a MeTTa atom the CDS/PDS box gate consumes:
#   (motive-cone <ctx> (lo …) (hi …))  →  fed to (cds-margin-box Δr Δn lo hi) in
#   lib/subrep/cds.metta. ℓ=0 ⟹ box [0, u]; the support head supplies u.
function cone_to_atom(ctx::AbstractString, lo::AbstractVector, hi::AbstractVector)
    fmt(v) = join(string.(round.(Float64.(v); digits=3)), " ")
    "(motive-cone $ctx (lo $(fmt(lo))) (hi $(fmt(hi))))"
end

# ── Train + demonstrate the co-learned motive geometry ───────────────────────
function run_mdn_demo(; n::Int=32, epochs::Int=200, lr::Float64=0.02, seed::Int=0)
    X, Y = make_data(n)
    structure = mdn_graph()
    params = initialize_params(structure, MersenneTwister(seed))
    loader = [Dict("x" => X, "y" => Y)]

    ev0 = evaluate_pcn(params, structure, loader; rng=MersenneTwister(7))
    params, iters, _ = train_pcn(
        params, structure, loader, lr; num_epochs=epochs,
        rng=MersenneTwister(7), verbose=false,
    )
    ev1 = evaluate_pcn(params, structure, loader; rng=MersenneTwister(7))

    # Predict the cone for each prototype context (this is what gets exposed as atoms).
    proto = permutedims(hcat(_CTX1, _CTX2))                # (2, 2): row1=ctx1, row2=ctx2
    cones = predict(params, structure, Dict("x" => proto), MersenneTwister(2); output_task="y")

    println("── SubRep MDN (FabricPC, PC-trained) ──")
    println("  energy:  ", round(ev0["energy"]; digits=4), " → ", round(ev1["energy"]; digits=4))
    println("  cone | ctx-1 (raid):  ", round.(cones[1, :]; digits=3), "  (want ≈ ", _CONE1, ")")
    println("  cone | ctx-2 (trade): ", round.(cones[2, :]; digits=3), "  (want ≈ ", _CONE2, ")")
    # Exposed as atoms for the symbolic gate (§2.4) — ℓ=0, u = learned support values.
    lo = zeros(Float32, size(cones, 2))
    println("  atom | ", cone_to_atom("raid", lo, cones[1, :]))
    println("  atom | ", cone_to_atom("trade", lo, cones[2, :]))
    return (; ev0, ev1, iters, cones)
end

# ── Admission-COUPLED training: the cone is shaped by the gate (§2.5 / §2.6) ──
# The iCog support head trains (MSE) on Wₓ targets derived from CERTIFIED WEIGHTS —
# the cone under which the gate admitted an option (utils/mdn_support_pipeline.py
# observe_and_train_support; §2.5 Wₓ^{t+1}=conv(Wₓ^t ∪ W_new)). So the cone target
# is DERIVED from the admission requirement, not hand-set. `cone_to_admit` is that
# derivation for a box cone: drop weight (u_i→0) on objectives the option HURTS
# (Δn_i<0), keep full weight where it helps — the minimal cone that admits it.
# (The paper §2.6 L_CDS hinge is the principled equivalent of this supervised target.)
cone_to_admit(dn::AbstractVector; hi::Float32=1.0f0) =
    Float32[dn[i] >= 0 ? hi : 0.0f0 for i in eachindex(dn)]

# box-cone CDS margin (matches lib/subrep/cds.metta cds-margin-box, ℓ=0) and the
# full-simplex margin, so we can verify admission under the LEARNED vs naive cone.
function box_margin(dr::Real, dn::AbstractVector, u::AbstractVector)
    m = Float32(dr)
    for i in eachindex(dn)
        m += dn[i] >= 0 ? 0.0f0 : u[i] * dn[i]
    end
    m
end
simplex_margin(dr::Real, dn::AbstractVector) = Float32(dr) + minimum(dn)

const _OPT1 = Float32[1.0, -0.3]    # ctx-1 option: helps motive-1, hurts motive-2
const _OPT2 = Float32[-0.3, 1.0]    # ctx-2 option: the mirror

function run_coupled_demo(; n::Int=32, epochs::Int=300, lr::Float64=0.02, seed::Int=0, eps::Float32=0.05f0)
    cone1 = cone_to_admit(_OPT1)            # gate-derived target [1, 0]
    cone2 = cone_to_admit(_OPT2)            # gate-derived target [0, 1]
    X = zeros(Float32, 2n, 2); Y = zeros(Float32, 2n, 2)
    for i in 1:n
        X[i, :] .= _CTX1; Y[i, :] .= cone1
        X[n + i, :] .= _CTX2; Y[n + i, :] .= cone2
    end
    structure = mdn_graph()
    params = initialize_params(structure, MersenneTwister(seed))
    loader = [Dict("x" => X, "y" => Y)]
    params, iters, _ = train_pcn(
        params, structure, loader, lr; num_epochs=epochs, rng=MersenneTwister(7), verbose=false
    )
    proto = permutedims(hcat(_CTX1, _CTX2))
    cones = predict(params, structure, Dict("x" => proto), MersenneTwister(2); output_task="y")
    u1 = cones[1, :]
    m_simplex = simplex_margin(0.0, _OPT1)               # naive full-simplex cone
    m_learned = box_margin(0.0, _OPT1, u1)               # the MDN's co-learned cone
    println("── Admission-COUPLED MDN (cone derived from the gate) ──")
    println("  ctx-1 learned cone u = ", round.(u1; digits=3), "  (gate-derived target ", cone1, ")")
    println("  option Δn=", _OPT1, ":")
    println("    full-simplex margin ", round(m_simplex; digits=3), "  → REJECT")
    println("    learned-cone margin ", round(m_learned; digits=3),
            m_learned >= -eps ? "  → ADMIT (co-learned geometry enables it)" : "  → reject")
    return (; iters, cones, m_learned, m_simplex, eps)
end

# ── PDS-CVaR gate: distribution-aware admission over w ~ Dirichlet(α) (§3.2 Def 3) ──
# The MDN's distribution_head emits Dirichlet concentration α; the CVaR gate admits an
# option iff the worst-tail (CVaR_α) of Δr + wᵀΔn over w ~ Dirichlet(α) is ≥ 0 (iCog
# certification/cvar_test.py). Dirichlet sampled via normalized Gammas (Marsaglia–Tsang;
# no Distributions.jl dep).
function _gamma_sample(a::Float32, rng)
    a < 1.0f0 && return _gamma_sample(a + 1.0f0, rng) * rand(rng, Float32)^(1.0f0 / a)
    d = a - 1.0f0 / 3.0f0
    c = 1.0f0 / sqrt(9.0f0 * d)
    while true
        x = randn(rng, Float32)
        v = (1.0f0 + c * x)^3
        v <= 0.0f0 && continue
        u = rand(rng, Float32)
        (log(u) < 0.5f0 * x^2 + d - d * v + d * log(v)) && return d * v
    end
end
function dirichlet_sample(alphas::AbstractVector, rng)
    g = Float32[_gamma_sample(Float32(a), rng) for a in alphas]
    g ./ sum(g)
end

function pds_cvar(dr::Real, dn::AbstractVector, alphas::AbstractVector;
                  confidence::Float64=0.1, nsamples::Int=4000, rng=MersenneTwister(0))
    vals = Vector{Float32}(undef, nsamples)
    for s in 1:nsamples
        w = dirichlet_sample(alphas, rng)
        vals[s] = Float32(dr) + sum(w .* dn)
    end
    sort!(vals)                                   # ascending → worst draws first
    k = max(1, ceil(Int, confidence * nsamples))  # the α-tail
    sum(@view vals[1:k]) / k                       # CVaR = mean of the worst tail
end
cvar_admit(dr, dn, alphas; kw...) = pds_cvar(dr, dn, alphas; kw...) >= 0.0f0

function run_cvar_demo()
    opt = Float32[1.0, -0.3]                       # helps motive-1, hurts motive-2
    a1 = Float32[10, 1]                            # MDN α: motive-1 dominant
    a2 = Float32[1, 10]                            # MDN α: motive-2 dominant
    c1 = pds_cvar(0.0, opt, a1; rng=MersenneTwister(1))
    c2 = pds_cvar(0.0, opt, a2; rng=MersenneTwister(1))
    println("── PDS-CVaR gate (distribution-aware over the MDN's α) ──")
    println("  option Δn=", opt, ":")
    println("    α=", a1, " (motive-1 weighted): CVaR ", round(c1; digits=3), c1 >= 0 ? "  → ADMIT" : "  → reject")
    println("    α=", a2, " (motive-2 weighted): CVaR ", round(c2; digits=3), c2 >= 0 ? "  → ADMIT" : "  → reject")
    return (; c1, c2)
end

# ── PC-native gate: the L_CDS margin gradient AS a local error node (§3) ──────
# The paper §3 wants CDS realized as LOCAL predictive-coding error nodes, not a
# separate solver. The CDS margin over a box cone is m = Δr + Σ_{i:Δn_i<0} u_i·Δn_i,
# and L_CDS = softplus(ε − m). Its gradient w.r.t. the support head's cone bounds u is
# LOCAL and closed-form: ∂L/∂u_i = σ(ε−m)·(−Δn_i) for objectives the option hurts, 0
# else. That gradient is exactly the error signal the PC support-head node follows —
# so the gate is a PC error node, and descending it makes the cone satisfy admission.
function lcds_grad(dr::Real, dn::AbstractVector, u::AbstractVector, eps::Float32)
    m = box_margin(dr, dn, u)
    sig = 1.0f0 / (1.0f0 + exp(-(eps - m)))          # σ(ε−m) = softplus'(ε−m)
    Float32[dn[i] < 0 ? sig * (-dn[i]) : 0.0f0 for i in eachindex(dn)]
end

function run_pc_native_gate_demo(; eps::Float32=0.05f0, lr::Float32=0.5f0, steps::Int=60)
    dn = Float32[1.0, -0.3]
    u = Float32[1.0, 1.0]                             # naive full cone → margin −0.3 (reject)
    m0 = box_margin(0.0, dn, u)
    for _ in 1:steps
        u = max.(u .- lr .* lcds_grad(0.0, dn, u, eps), 0.0f0)   # follow the local gate error
    end
    m1 = box_margin(0.0, dn, u)
    println("── PC-native gate: follow the local L_CDS error (§3) ──")
    println("  cone u: [1.0, 1.0] → ", round.(u; digits=3))
    println("  margin: ", round(m0; digits=3), " (reject) → ", round(m1; digits=3),
            m1 >= -eps ? "  → ADMIT (gate satisfied by local descent)" : "  → reject")
    return (; m0, m1, u, eps)
end

# ── §2.6 full admission-coupled co-training: learn the cone FROM the gate loss ──
# Deeper than run_coupled_demo (which fit hand-derived MSE cone targets): here the MDN
# co-trains directly on the paper's L_CDS hinge — there are NO cone targets, the
# supervision is the gate's own gradient. Each step predicts the cone, forms a
# PC pseudo-target u − η·∂L_CDS/∂u (a local descent step on softplus(ε−margin) via the
# Danskin gradient lcds_grad), and runs one local-PC update toward it. The motive
# geometry emerges purely from admission decisions (§2.6).
# §2.6 regularizer: a log-VOLUME reward keeps the cone the WEAKEST ADEQUATE slice,
# preventing the degenerate collapse to a point that admits everything. Combined
# objective L = L_CDS − λ·Σ log(u_i); ∂/∂u_i = ∂L_CDS/∂u_i − λ/u_i (the reward pushes
# every bound UP, away from 0 — so the cone stays as LARGE as admission allows).
function lcds_grad_reg(dr::Real, dn::AbstractVector, u::AbstractVector, eps::Float32, lam::Float32)
    lcds_grad(dr, dn, u, eps) .- lam ./ (u .+ 1.0f-3)
end

function run_lcds_cotrain_demo(; epochs::Int=250, lr_pc::Float64=0.3, eta::Float32=1.0f0,
                               inner::Int=3, lam::Float32=0.03f0, eps::Float32=0.07f0, seed::Int=0)
    X = permutedims(hcat(_CTX1, _CTX2))              # (2,2): two contexts
    opts = (_OPT1, _OPT2)                            # each context's should-admit option
    structure = mdn_graph()
    params = initialize_params(structure, MersenneTwister(seed))
    for ep in 1:epochs
        u = predict(params, structure, Dict("x" => X), MersenneTwister(ep); output_task="y")
        T = similar(u)
        for i in 1:size(u, 1)                        # pseudo-target = L_CDS + volume-reward descent
            T[i, :] = clamp.(u[i, :] .- eta .* lcds_grad_reg(0.0, opts[i], u[i, :], eps, lam), 0.0f0, 1.0f0)
        end
        params, _, _ = train_pcn(params, structure, [Dict("x" => X, "y" => T)], lr_pc;
                                 num_epochs=inner, rng=MersenneTwister(ep), verbose=false)
    end
    cones = predict(params, structure, Dict("x" => X), MersenneTwister(99); output_task="y")
    m1 = box_margin(0.0, _OPT1, cones[1, :])
    m2 = box_margin(0.0, _OPT2, cones[2, :])
    println("── §2.6 L_CDS admission-coupled co-training (NO cone targets — gate loss only) ──")
    println("  ctx-1 cone ", round.(cones[1, :]; digits=3), "  margin ", round(m1; digits=3),
            m1 >= -eps ? "  → ADMIT" : "  → reject")
    println("  ctx-2 cone ", round.(cones[2, :]; digits=3), "  margin ", round(m2; digits=3),
            m2 >= -eps ? "  → ADMIT" : "  → reject")
    return (; cones, m1, m2, eps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    r = run_mdn_demo()
    # Co-learned-geometry checks: energy fell, and each context's cone emphasises the
    # right objective (ctx-1 → motive-1 bound larger; ctx-2 → motive-2 bound larger).
    @assert r.iters[end][1] < r.iters[1][1] "energy did not fall"
    @assert r.cones[1, 1] > r.cones[1, 2] "ctx-1 cone should emphasise motive-1"
    @assert r.cones[2, 2] > r.cones[2, 1] "ctx-2 cone should emphasise motive-2"
    println("SUBREP MDN OK — context → motive cone learned by local PC")

    println()
    c = run_coupled_demo()
    # Admission-coupling: the MDN's gate-derived cone must ADMIT (margin ≥ −ε) the
    # option that the naive full-simplex cone REJECTS (margin < 0) — the co-learned
    # geometry is what enables the admission. This closes the gate→cone→MDN→gate loop.
    @assert c.m_simplex < 0 "simplex should reject the complementary option"
    @assert c.m_learned >= -c.eps "the co-learned cone should admit it"
    println("SUBREP MDN COUPLED OK — gate-derived cone admits what the simplex rejects")

    println()
    v = run_cvar_demo()
    # Distribution-aware admission: the SAME option is CVaR-admitted when the MDN's α
    # weights motive-1 (which it helps) and CVaR-rejected when α weights motive-2.
    @assert v.c1 >= 0 "should CVaR-admit under motive-1-weighted α"
    @assert v.c2 < 0 "should CVaR-reject under motive-2-weighted α"
    println("SUBREP CVaR OK — distribution-aware gate flips with the MDN's motive weights")

    println()
    g = run_pc_native_gate_demo()
    # The gate as a local PC error node: starting from the naive full cone (reject),
    # descending the local L_CDS gradient drives the cone to satisfy admission.
    @assert g.m0 < 0 "naive cone should reject"
    @assert g.m1 >= -g.eps "descending the local gate error should reach admission"
    println("SUBREP PC-NATIVE GATE OK — local L_CDS error drives the cone to admit")

    println()
    l = run_lcds_cotrain_demo()
    # The MDN learned each context's cone PURELY from the L_CDS gate loss + volume reward
    # (no cone targets): each context's option is admitted, AND the cone is the WEAKEST
    # ADEQUATE slice (NOT collapsed) — ctx-1 keeps the helped motive's weight and shrinks
    # the hurt one's, ctx-2 the mirror. Differentiated geometry, not a degenerate point.
    @assert l.m1 >= -l.eps "ctx-1 option should be admitted"
    @assert l.m2 >= -l.eps "ctx-2 option should be admitted"
    @assert l.cones[1, 1] > l.cones[1, 2] "ctx-1 cone should keep motive-1 weight (weakest-adequate)"
    @assert l.cones[2, 2] > l.cones[2, 1] "ctx-2 cone should keep motive-2 weight (weakest-adequate)"
    println("SUBREP L_CDS CO-TRAIN OK — weakest-adequate geometry learned from admission decisions")
end
