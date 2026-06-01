# Energy functionals. Port of fabricpc/core/energy.py.
#
# An energy E(z_latent, z_mu) scores the local prediction error at a node.
# `energy` returns a per-sample vector (shape (batch,)); `grad_latent` returns
# dE/dz_latent (the node's self-latent gradient, same shape as z_latent).
#
# Phase B shipped GaussianEnergy (the default, == MSE). Phase D2 adds Bernoulli /
# CrossEntropy / Laplacian / Huber / KL, each with analytic `grad_latent` (∂E/∂z)
# AND `grad_mu` (∂E/∂z_mu) so the explicit node path generalizes with no autodiff.

abstract type AbstractEnergy end

"""
    GaussianEnergy(; precision = 1.0)

Gaussian (quadratic) energy: `E = 0.5 * precision * Σ (z - μ)²`, summed over all
non-batch dimensions. `precision = 1/σ²`. This is the standard MSE-based energy
and the default for every node type.

`grad_latent` returns `precision * (z - μ)`.

Note (faithful to upstream): the explicit per-edge input/weight gradients in
`Linear` use `gain_mod_error = error * f'(pre)` WITHOUT a precision factor — they
implicitly assume `precision = 1` (the default). `grad_latent` keeps the factor.
At the default precision the two are consistent; non-unit precision in the
explicit path is a known upstream simplification we replicate verbatim.
"""
struct GaussianEnergy <: AbstractEnergy
    precision::Float32
end
GaussianEnergy(; precision=1.0) = GaussianEnergy(Float32(precision))

function energy(e::GaussianEnergy, z_latent, z_mu)
    diff = z_latent .- z_mu
    # Sum over all non-batch dims (dim 1 is batch in our batch-first layout).
    nonbatch = ntuple(i -> i + 1, ndims(diff) - 1)
    return (0.5f0 * e.precision) .* vec(sum(abs2.(diff); dims=nonbatch))
end

grad_latent(e::GaussianEnergy, z_latent, z_mu) = e.precision .* (z_latent .- z_mu)

# =============================================================================
# Phase D2 — non-Gaussian energies + the ∂E/∂z_mu axis.
#
# Upstream ships `energy` + `grad_latent` (∂E/∂z) only; its explicit Linear path
# is Gaussian-only and falls back to autodiff otherwise. We instead add an
# analytic `grad_mu` (∂E/∂z_mu) per energy, which lets the explicit node path
# generalize to ALL these energies with no autodiff (the v0 thesis). The node
# gradients use dE/dpre = grad_mu · f'(pre); at Gaussian precision 1,
# grad_mu = -(z - z_mu) = -error, reproducing the Phase B formulas bit-for-bit.
# =============================================================================

# Sum over all non-batch dims → per-sample (batch,) vector.
_sum_nonbatch(a) = vec(sum(a; dims=ntuple(i -> i + 1, ndims(a) - 1)))

"""∂E/∂z_mu for the Gaussian energy: `precision·(z_mu - z)` (= -grad_latent)."""
grad_mu(e::GaussianEnergy, z_latent, z_mu) = e.precision .* (z_mu .- z_latent)

"""
    BernoulliEnergy(; eps = 1e-7)

Binary cross-entropy energy `E = -Σ[z·log μ + (1-z)·log(1-μ)]` for binary targets
with μ ∈ (0,1) (pair with `SigmoidActivation`). Port of `BernoulliEnergy`.
"""
struct BernoulliEnergy <: AbstractEnergy
    eps::Float32
end
BernoulliEnergy(; eps=1e-7) = BernoulliEnergy(Float32(eps))

function energy(e::BernoulliEnergy, z_latent, z_mu)
    mu = clamp.(z_mu, e.eps, 1.0f0 - e.eps)
    bce = -(z_latent .* log.(mu) .+ (1.0f0 .- z_latent) .* log.(1.0f0 .- mu))
    return _sum_nonbatch(bce)
end
grad_latent(e::BernoulliEnergy, z_latent, z_mu) =
    (mu=clamp.(z_mu, e.eps, 1.0f0 - e.eps); -log.(mu) .+ log.(1.0f0 .- mu))
grad_mu(e::BernoulliEnergy, z_latent, z_mu) =
    (
        mu=clamp.(z_mu, e.eps, 1.0f0 - e.eps);
        -z_latent ./ mu .+ (1.0f0 .- z_latent) ./ (1.0f0 .- mu)
    )

"""
    CrossEntropyEnergy(; eps = 1e-7)

Categorical cross-entropy `E = -Σ z_i·log μ_i` for one-hot targets with μ a
probability vector (pair with `SoftmaxActivation`). Port of `CrossEntropyEnergy`.
"""
struct CrossEntropyEnergy <: AbstractEnergy
    eps::Float32
end
CrossEntropyEnergy(; eps=1e-7) = CrossEntropyEnergy(Float32(eps))

energy(e::CrossEntropyEnergy, z_latent, z_mu) =
    _sum_nonbatch(-z_latent .* log.(clamp.(z_mu, e.eps, 1.0f0)))
grad_latent(e::CrossEntropyEnergy, z_latent, z_mu) = -log.(clamp.(z_mu, e.eps, 1.0f0))
grad_mu(e::CrossEntropyEnergy, z_latent, z_mu) = -z_latent ./ clamp.(z_mu, e.eps, 1.0f0)

"""
    LaplacianEnergy(; scale = 1.0)

L1 energy `E = (1/b)·Σ|z - μ|` (robust to outliers). Port of `LaplacianEnergy`.
"""
struct LaplacianEnergy <: AbstractEnergy
    scale::Float32
end
LaplacianEnergy(; scale=1.0) = LaplacianEnergy(Float32(scale))

energy(e::LaplacianEnergy, z_latent, z_mu) =
    _sum_nonbatch(abs.(z_latent .- z_mu)) ./ e.scale
grad_latent(e::LaplacianEnergy, z_latent, z_mu) = sign.(z_latent .- z_mu) ./ e.scale
grad_mu(e::LaplacianEnergy, z_latent, z_mu) = -sign.(z_latent .- z_mu) ./ e.scale

"""
    HuberEnergy(; delta = 1.0)

Smooth-L1 (Huber) energy: quadratic within `|z-μ| ≤ delta`, linear beyond. Port of
`HuberEnergy`.
"""
struct HuberEnergy <: AbstractEnergy
    delta::Float32
end
HuberEnergy(; delta=1.0) = HuberEnergy(Float32(delta))

function energy(e::HuberEnergy, z_latent, z_mu)
    diff = z_latent .- z_mu
    ad = abs.(diff)
    h = ifelse.(ad .<= e.delta, 0.5f0 .* diff .^ 2, e.delta .* (ad .- 0.5f0 * e.delta))
    return _sum_nonbatch(h)
end
grad_latent(e::HuberEnergy, z_latent, z_mu) = clamp.(z_latent .- z_mu, -e.delta, e.delta)
grad_mu(e::HuberEnergy, z_latent, z_mu) = -clamp.(z_latent .- z_mu, -e.delta, e.delta)

"""
    KLDivergenceEnergy(; eps = 1e-7)

`E = D_KL(z ‖ μ) = Σ z·log(z/μ)` for probability-vector z and μ. Port of
`KLDivergenceEnergy`.
"""
struct KLDivergenceEnergy <: AbstractEnergy
    eps::Float32
end
KLDivergenceEnergy(; eps=1e-7) = KLDivergenceEnergy(Float32(eps))

function energy(e::KLDivergenceEnergy, z_latent, z_mu)
    zs = clamp.(z_latent, e.eps, 1.0f0)
    mu = clamp.(z_mu, e.eps, 1.0f0)
    kl = ifelse.(z_latent .< e.eps, 0.0f0, zs .* (log.(zs) .- log.(mu)))
    return _sum_nonbatch(kl)
end
function grad_latent(e::KLDivergenceEnergy, z_latent, z_mu)
    zs = clamp.(z_latent, e.eps, 1.0f0)
    mu = clamp.(z_mu, e.eps, 1.0f0)
    g = log.(zs) .- log.(mu) .+ 1.0f0
    return ifelse.(z_latent .< e.eps, -log.(mu), g)
end
grad_mu(e::KLDivergenceEnergy, z_latent, z_mu) =
    (
        mu=clamp.(z_mu, e.eps, 1.0f0);
        ifelse.(z_latent .< e.eps, 0.0f0, -clamp.(z_latent, e.eps, 1.0f0) ./ mu)
    )
