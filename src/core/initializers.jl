# Tensor initializers. Port of fabricpc/core/initializers.py.
#
# Initializers are context-agnostic (weights vs latent states): the caller picks
# the shape. Upstream threads a JAX PRNGKey; we thread an `AbstractRNG` instead
# (exact JAX bit-parity is neither possible nor a goal — within-Julia
# determinism under a seeded RNG is what we test against).
#
# Phase B ships Zeros + Normal. Kaiming / Xavier / Uniform arrive when a node
# needs them (Linear's v0 default is Normal, matching upstream LinearExplicitGrad).

abstract type AbstractInitializer end

"""
    ZerosInitializer()

Initialize with zeros. Used for biases.
"""
struct ZerosInitializer <: AbstractInitializer end

initialize(::AbstractRNG, shape::Tuple, ::ZerosInitializer) = zeros(Float32, shape...)

"""
    NormalInitializer(; mean = 0.0, std = 0.05, gain = 1.0)

Draw from `N(mean, (gain·std)²)`: `mean + gain·std·randn`.
"""
struct NormalInitializer <: AbstractInitializer
    mean::Float32
    std::Float32
    gain::Float32
end
NormalInitializer(; mean=0.0, std=0.05, gain=1.0) =
    NormalInitializer(Float32(mean), Float32(std), Float32(gain))

function initialize(rng::AbstractRNG, shape::Tuple, init::NormalInitializer)
    return init.mean .+ (init.gain * init.std) .* randn(rng, Float32, shape...)
end

"""
    MuPCInitializer(; gain = 1.0)

muPC weight init: `W ~ gain · N(0, 1)` — UNIT variance. The width/depth scaling
is deliberately NOT baked into the weights; it is applied in the forward pass via
the per-edge muPC factors (core/mupc.jl). This decoupling of init from
forward-scaling is the core of muPC (Yang et al.; Innocenti et al.). Port of
`MuPCInitializer`.

This is the init muPC REQUIRES — pairing the muPC scaling with a small-std init
(e.g. NormalInitializer(std=0.05)) double-shrinks the effective weights and breaks
the parameterization.
"""
struct MuPCInitializer <: AbstractInitializer
    gain::Float32
end
MuPCInitializer(; gain=1.0) = MuPCInitializer(Float32(gain))

initialize(rng::AbstractRNG, shape::Tuple, init::MuPCInitializer) =
    init.gain .* randn(rng, Float32, shape...)

"""
    XavierInitializer(; gain = 1.0)

Xavier/Glorot normal init: `W ~ N(0, std²)`, `std = gain·√(2/(fan_in+fan_out))`,
for a weight of shape `(fan_in, fan_out)`. Used for the (unscaled) output layer of
a muPC net. Port of `XavierInitializer` (normal variant; uniform deferred).
"""
struct XavierInitializer <: AbstractInitializer
    gain::Float32
end
XavierInitializer(; gain=1.0) = XavierInitializer(Float32(gain))

function initialize(rng::AbstractRNG, shape::Tuple, init::XavierInitializer)
    fan_in = shape[1]
    fan_out = length(shape) > 1 ? shape[2] : shape[1]
    std = init.gain * sqrt(2.0f0 / (fan_in + fan_out))
    return std .* randn(rng, Float32, shape...)
end
