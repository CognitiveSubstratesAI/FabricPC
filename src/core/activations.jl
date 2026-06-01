# Activation functions. Port of fabricpc/core/activations.py.
#
# Each activation is a small struct; `forward` / `derivative` dispatch on it.
# Upstream computes `derivative` only on the explicit-gradient path (the
# autodiff path never calls it). Since FabricPC.jl v0 is fully explicit, the
# inference + learning loops DO call `derivative` (via compute_gain_mod_error).
#
# Phase B shipped IdentityActivation; Phase D1 adds the element-wise non-linear
# zoo (Sigmoid, Tanh, ReLU, LeakyReLU, GELU, HardTanh). These work through the
# EXISTING explicit Linear/LinearResidual gradient path — `gain_mod_error =
# error · f'(pre)` is exact for any element-wise activation under Gaussian energy,
# so no autodiff is needed. Softmax (non-element-wise) + the Enzyme generic
# fallback are Phase D2.

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
