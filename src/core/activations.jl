# Activation functions. Port of fabricpc/core/activations.py.
#
# Each activation is a small struct; `forward` / `derivative` dispatch on it.
# Upstream computes `derivative` only on the explicit-gradient path (the
# autodiff path never calls it). Since FabricPC.jl v0 is fully explicit, the
# inference + learning loops DO call `derivative` (via compute_gain_mod_error).
#
# Phase B ships IdentityActivation only; non-linear activations (Sigmoid, Tanh,
# ReLU, GELU, …) arrive in Phase D alongside the Enzyme fallback.

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
