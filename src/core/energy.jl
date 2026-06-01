# Energy functionals. Port of fabricpc/core/energy.py.
#
# An energy E(z_latent, z_mu) scores the local prediction error at a node.
# `energy` returns a per-sample vector (shape (batch,)); `grad_latent` returns
# dE/dz_latent (the node's self-latent gradient, same shape as z_latent).
#
# Phase B ships GaussianEnergy only (the default, == MSE). Bernoulli / CE /
# Laplacian / Huber / KL energies arrive in Phase D.

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
