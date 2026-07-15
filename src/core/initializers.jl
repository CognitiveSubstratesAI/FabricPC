# Tensor initializers. Port of fabricpc/core/initializers.py.
#
# Initializers are context-agnostic (weights vs latent states): the caller picks
# the shape. Upstream threads a JAX PRNGKey; we thread an `AbstractRNG` instead
# (exact JAX bit-parity is neither possible nor a goal — within-Julia
# determinism under a seeded RNG is what we test against).
#
# Full roster (upstream parity): Zeros, Normal, MuPC, Xavier (normal+uniform),
# Kaiming/He (normal+uniform), Ones, Uniform. (Deferred: GlobalState /
# NodeDistribution state inits — those are state, not weight, initializers.)

"""
    AbstractInitializer

Context-agnostic tensor initializer (weights or latent states — the caller picks the shape).
Every concrete subtype implements `initialize(rng, shape, init)` via dispatch. Threads a plain
`AbstractRNG` (not a JAX-style `PRNGKey`) — bit-parity with upstream's RNG stream is neither
possible nor a goal; within-Julia determinism under a seeded RNG is what's tested.
"""
abstract type AbstractInitializer end

"""
    initialize(rng::AbstractRNG, shape::Tuple, init::AbstractInitializer) -> Array{Float32}

Sample an array of the given `shape` under the given initializer. Dispatches on the concrete
`AbstractInitializer` subtype ([`ZerosInitializer`](@ref), [`NormalInitializer`](@ref),
[`MuPCInitializer`](@ref), [`XavierInitializer`](@ref), [`KaimingInitializer`](@ref),
[`OnesInitializer`](@ref), [`UniformInitializer`](@ref)).
"""
function initialize end

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
    XavierInitializer(; gain = 1.0, distribution = :normal)

Xavier/Glorot init, shape-aware over ANY rank — Linear `(in, out)` or an ND conv kernel
`(*spatial, C_in, C_out)`; see [`_fans`](@ref). Port of `XavierInitializer` (both variants):
- `:normal`  — `N(0, std²)`, `std = gain·√(2/(fan_in+fan_out))`
- `:uniform` — `U(−limit, limit)`, `limit = gain·√(6/(fan_in+fan_out))`
"""
struct XavierInitializer <: AbstractInitializer
    gain::Float32
    distribution::Symbol
end
XavierInitializer(; gain=1.0, distribution=:normal) =
    XavierInitializer(Float32(gain), distribution)

"""
    _fans(shape) -> (fan_in, fan_out)

Shape-aware fans, shared by Xavier and Kaiming. Verbatim port of the upstream formula
(`core/initializers.py:211-215` Xavier, `:279-283` Kaiming), which is rank-generic over
FabricPC's HWIO/LIO/DHWIO kernel layout:

    rank >= 2 :  fan_in  = prod(shape[1:end-1])            # (*spatial, C_in)
                 fan_out = prod(shape[1:end-2]) * shape[end]  # (*spatial) * C_out
    rank == 1 :  fan_in  = fan_out = shape[1]

Linear `(in, out)` is UNCHANGED by construction — `prod(shape[1:1]) == shape[1]` and
`prod(()) * shape[2] == shape[2]` — so this is a pure extension, not a behaviour change for
any rank-2 weight. A Conv2D kernel `(kH, kW, C_in, C_out)` correctly gives
`fan_in = kH*kW*C_in` (C-08): before this, a `(3,3,8,16)` kernel took `fan = shape[1] = 3`
instead of `72`, initialising ~4.9x too wide under Kaiming (ConvNode's default).
"""
function _fans(shape::Tuple)
    length(shape) >= 2 || return (Int(shape[1]), Int(shape[1]))
    return (Int(prod(shape[1:end-1])), Int(prod(shape[1:end-2]) * shape[end]))
end

function initialize(rng::AbstractRNG, shape::Tuple, init::XavierInitializer)
    fan_in, fan_out = _fans(shape)
    if init.distribution == :uniform
        limit = init.gain * sqrt(6.0f0 / (fan_in + fan_out))
        return (-limit) .+ (2.0f0 * limit) .* rand(rng, Float32, shape...)
    else
        std = init.gain * sqrt(2.0f0 / (fan_in + fan_out))
        return std .* randn(rng, Float32, shape...)
    end
end

"""
    KaimingInitializer(; mode = :fan_in, nonlinearity = :relu,
                       distribution = :normal, a = 0.01, gain = 1.0)

Kaiming/He init optimized for ReLU networks (preserves activation variance).
Port of `KaimingInitializer`. Shape-aware over ANY rank — Linear `(in, out)` or an ND conv
kernel `(*spatial, C_in, C_out)`, e.g. Conv2D `(kH, kW, C_in, C_out)` ⇒ `fan_in = kH*kW*C_in`
(see [`_fans`](@ref)):
- `fan = fan_in` (`:fan_in`) or `fan_out` (`:fan_out`)
- ReLU gain `√2`; leaky-ReLU gain `√(2/(1+a²))`; scaled by `gain`
- `:normal`  — `std = gain·g/√fan`
- `:uniform` — `limit = gain·g·√(3/fan)`
"""
struct KaimingInitializer <: AbstractInitializer
    mode::Symbol
    nonlinearity::Symbol
    distribution::Symbol
    a::Float32
    gain::Float32
end
KaimingInitializer(; mode=:fan_in, nonlinearity=:relu, distribution=:normal,
    a=0.01, gain=1.0) =
    KaimingInitializer(mode, nonlinearity, distribution, Float32(a), Float32(gain))

function initialize(rng::AbstractRNG, shape::Tuple, init::KaimingInitializer)
    fan_in, fan_out = _fans(shape)                 # C-08: ND/conv-aware (see `_fans`)
    fan = init.mode == :fan_out ? fan_out : fan_in
    g = init.nonlinearity == :leaky_relu ? sqrt(2.0f0 / (1.0f0 + init.a^2)) : sqrt(2.0f0)
    if init.distribution == :uniform
        limit = init.gain * g * sqrt(3.0f0 / fan)
        return (-limit) .+ (2.0f0 * limit) .* rand(rng, Float32, shape...)
    else
        std = init.gain * g / sqrt(Float32(fan))
        return std .* randn(rng, Float32, shape...)
    end
end

"""
    OnesInitializer()

Initialize with ones. Port of `OnesInitializer`.
"""
struct OnesInitializer <: AbstractInitializer end
initialize(::AbstractRNG, shape::Tuple, ::OnesInitializer) = ones(Float32, shape...)

"""
    UniformInitializer(; minval = -0.05, maxval = 0.05)

Uniform init `U(minval, maxval)`. Port of `UniformInitializer`.
"""
struct UniformInitializer <: AbstractInitializer
    minval::Float32
    maxval::Float32
end
UniformInitializer(; minval=-0.05, maxval=0.05) =
    UniformInitializer(Float32(minval), Float32(maxval))

function initialize(rng::AbstractRNG, shape::Tuple, init::UniformInitializer)
    return init.minval .+ (init.maxval - init.minval) .* rand(rng, Float32, shape...)
end
