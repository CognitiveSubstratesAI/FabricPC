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
