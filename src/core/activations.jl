# Activation functions. Port of fabricpc/core/activations.py.
#
# Each activation is a small struct; `forward` / `derivative` dispatch on it.
# Upstream computes `derivative` only on the explicit-gradient path (the
# autodiff path never calls it). Since FabricPC.jl v0 is fully explicit, the
# inference + learning loops DO call `derivative` (via the node `pre_grad` helper).
#
# Phase B shipped IdentityActivation; Phase D1 added the element-wise non-linear
# zoo (Sigmoid, Tanh, ReLU, LeakyReLU, GELU, HardTanh); Phase D2 added Softmax.
# Element-wise activations work through the EXISTING explicit Linear/LinearResidual
# path — the node `pre_grad` helper computes `(∂E/∂z_mu)·f'(pre)`, exact for any
# element-wise activation, so no autodiff is needed. Softmax is non-element-wise
# and uses the diagonal-Jacobian PC approximation (see its docstring).

abstract type AbstractActivation end

"""
    IdentityActivation()

Identity activation: f(x) = x, f'(x) = 1.
"""
struct IdentityActivation <: AbstractActivation end

forward(::IdentityActivation, x) = x
# f'(x) = 1 everywhere; return a scalar one so `error .* derivative` broadcasts
# without allocating a full ones-array.
derivative(::IdentityActivation, x) = one(eltype(x))

# muPC gains (Phase C). `variance_gain` = Kaiming-style g such that pre-activation
# Var = g² gives post-activation Var ≈ 1; `jacobian_gain` = 1/(g·rms(f')) to
# normalize per-hop top-down gradient propagation to ~1. Both default to 1
# (exact for identity / ReLU / LeakyReLU); non-linear activations override.
variance_gain(::AbstractActivation) = 1.0f0
jacobian_gain(::AbstractActivation) = 1.0f0

# Full Jacobian J[b,i,j] = ∂f(x)_i/∂x_j over the feature axis (dim 2; batch-first), shape
# (B, D, D). For ELEMENT-WISE activations this is the diagonal embedding of `derivative`
# (off-diagonals 0) — upstream's stated element-wise contract (activations.py:144, which raises
# and instructs callers to "use derivative() as the diagonal"; here we materialize it so the API
# is uniform across activations rather than raising — idiomatic for Julia dispatch). NON-element-
# wise activations (Softmax) OVERRIDE with their off-diagonal terms. Convenience for explicit
# (non-autodiff) gradient paths needing the full matrix. Mirrors upstream ActivationBase.jacobian.
function jacobian(a::AbstractActivation, x::AbstractMatrix)
    B, D = size(x)
    d = derivative(a, x) .* ones(eltype(x), B, D)             # (B,D); materializes scalar (Identity)
    eye = reshape(eltype(x)[i == j for i in 1:D, j in 1:D], 1, D, D)
    return reshape(d, B, D, 1) .* eye                         # J[b,i,j] = δ_ij · d[b,i]
end

# =============================================================================
# Element-wise non-linear activations (Phase D1).
#
# Each `derivative` returns an array (same shape as x); `gain_mod_error =
# error .* derivative(...)` broadcasts element-wise. variance_gain /
# jacobian_gain values are ported verbatim from upstream (the muPC tables).
# =============================================================================

"""Sigmoid: σ(x) = 1/(1+e⁻ˣ); σ'(x) = σ(1-σ)."""
struct SigmoidActivation <: AbstractActivation end
_sigmoid(x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))
forward(::SigmoidActivation, x) = _sigmoid(x)
function derivative(::SigmoidActivation, x)
    s = _sigmoid(x)
    return s .* (1.0f0 .- s)
end

"""Tanh: f(x) = tanh(x); f'(x) = 1 - tanh²(x)."""
struct TanhActivation <: AbstractActivation end
forward(::TanhActivation, x) = tanh.(x)
derivative(::TanhActivation, x) = 1.0f0 .- tanh.(x) .^ 2
variance_gain(::TanhActivation) = Float32(sqrt(5.0 / 3.0))
jacobian_gain(::TanhActivation) = 1.261f0   # 1/(√(5/3)·rms(tanh'(z))), z~N(0,5/3)

"""ReLU: f(x) = max(0, x); f'(x) = 1[x>0]."""
struct ReLUActivation <: AbstractActivation end
forward(::ReLUActivation, x) = max.(x, 0.0f0)
derivative(::ReLUActivation, x) = Float32.(x .> 0.0f0)
variance_gain(::ReLUActivation) = Float32(sqrt(2.0))

"""Leaky ReLU: f(x) = x>0 ? x : αx; f'(x) = x>0 ? 1 : α."""
struct LeakyReLUActivation <: AbstractActivation
    alpha::Float32
end
LeakyReLUActivation(; alpha=0.01) = LeakyReLUActivation(Float32(alpha))
forward(a::LeakyReLUActivation, x) = ifelse.(x .> 0.0f0, x, a.alpha .* x)
derivative(a::LeakyReLUActivation, x) = ifelse.(x .> 0.0f0, 1.0f0, a.alpha)
variance_gain(a::LeakyReLUActivation) = Float32(sqrt(2.0 / (1.0 + a.alpha^2)))

"""
Softplus: f(x) = log(1 + eˣ) — a smooth, strictly-positive (≥0) activation;
f'(x) = σ(x) = 1/(1 + e⁻ˣ). Numerically stable via `log1p(exp(-|x|)) + max(x, 0)`.
Use for outputs that must stay positive — e.g. the SubRep MDN support head's cone
bounds (`examples/subrep_mdn.jl`), matching the iCog `generator/mdn.py` softplus head
(strict positivity, unlike ReLU which can pin a cone bound to exactly 0).
"""
struct SoftplusActivation <: AbstractActivation end
forward(::SoftplusActivation, x) = log1p.(exp.(-abs.(x))) .+ max.(x, 0.0f0)
derivative(::SoftplusActivation, x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))

"""
GELU (tanh approximation): f(x) = 0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³))).
Upstream's `derivative` is for this tanh approximation, so we use the tanh form
for `forward` too — keeping f and f' self-consistent for the explicit path
(upstream's forward calls erf-GELU, a minor inconsistency we resolve here).
"""
struct GeluActivation <: AbstractActivation end
const _GELU_C = Float32(sqrt(2.0 / π))
function forward(::GeluActivation, x)
    return 0.5f0 .* x .* (1.0f0 .+ tanh.(_GELU_C .* (x .+ 0.044715f0 .* x .^ 3)))
end
function derivative(::GeluActivation, x)
    inner = _GELU_C .* (x .+ 0.044715f0 .* x .^ 3)
    t = tanh.(inner)
    cdf = 0.5f0 .* (1.0f0 .+ t)
    cdf_prime =
        (0.5f0 .* _GELU_C .* (1.0f0 .+ 3.0f0 .* 0.044715f0 .* x .^ 2)) .* (1.0f0 .- t .^ 2)
    return cdf .+ x .* cdf_prime
end
variance_gain(::GeluActivation) = Float32(sqrt(2.0))
jacobian_gain(::GeluActivation) = 1.168f0   # 1/(√2·rms(gelu'(z))), z~N(0,2)

"""
Softmax over the last axis: `f(x)_i = e^{x_i} / Σ_j e^{x_j}`.

NON-element-wise. `derivative` returns the DIAGONAL of the Jacobian, `s·(1-s)`
— upstream's sanctioned approximation "valid for element-wise PC gradients"
(the off-diagonal `-s_i s_j` terms are dropped). Pair with `CrossEntropyEnergy`
for classification. The explicit gradient is therefore approximate for Softmax
(unlike the exact element-wise activations); see docs/decisions.md.
"""
struct SoftmaxActivation <: AbstractActivation end
function forward(::SoftmaxActivation, x)
    # Stable softmax along the LAST axis: the feature axis for rank-2 (B, V) — dims=2, so this stays
    # backward-compatible — and the vocab axis for rank-3 (B, S, V). The rank-3 case is what lets
    # VocabProjectionNode use Softmax+CrossEntropy (per upstream transformer_v2).
    d = ndims(x)
    m = maximum(x; dims=d)
    ex = exp.(x .- m)
    return ex ./ sum(ex; dims=d)
end
function derivative(a::SoftmaxActivation, x)
    s = forward(a, x)
    return s .* (1.0f0 .- s)   # diagonal of diag(s) - s·sᵀ
end
# Full softmax Jacobian (the off-diagonal terms `derivative` drops): J[b,i,j] = s_i·(δ_ij − s_j),
# shape (B, D, D). Mirrors upstream SoftmaxActivation.jacobian (activations.py:348). Use where the
# exact (non-diagonal) softmax gradient is needed; the PC path keeps the diagonal `derivative`.
function jacobian(a::SoftmaxActivation, x::AbstractMatrix)
    s = forward(a, x)                                         # (B, D)
    B, D = size(x)
    eye = reshape(eltype(x)[i == j for i in 1:D, j in 1:D], 1, D, D)
    return reshape(s, B, D, 1) .* (eye .- reshape(s, B, 1, D))  # s_i·(δ_ij − s_j)
end

"""Hard tanh: f(x) = clamp(x, lo, hi); f'(x) = 1[lo<x<hi]."""
struct HardTanhActivation <: AbstractActivation
    min_val::Float32
    max_val::Float32
end
HardTanhActivation(; min_val=-1.0, max_val=1.0) =
    HardTanhActivation(Float32(min_val), Float32(max_val))
forward(a::HardTanhActivation, x) = clamp.(x, a.min_val, a.max_val)
derivative(a::HardTanhActivation, x) = Float32.((x .> a.min_val) .& (x .< a.max_val))
variance_gain(::HardTanhActivation) = Float32(sqrt(5.0 / 3.0))
jacobian_gain(::HardTanhActivation) = 1.035f0   # 1/(√(5/3)·rms(hardtanh'(z)))
