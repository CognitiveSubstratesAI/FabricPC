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

if abspath(PROGRAM_FILE) == @__FILE__
    r = run_mdn_demo()
    # Co-learned-geometry checks: energy fell, and each context's cone emphasises the
    # right objective (ctx-1 → motive-1 bound larger; ctx-2 → motive-2 bound larger).
    @assert r.iters[end][1] < r.iters[1][1] "energy did not fall"
    @assert r.cones[1, 1] > r.cones[1, 2] "ctx-1 cone should emphasise motive-1"
    @assert r.cones[2, 2] > r.cones[2, 1] "ctx-2 cone should emphasise motive-2"
    println("SUBREP MDN OK — context → motive cone learned by local PC")
end
