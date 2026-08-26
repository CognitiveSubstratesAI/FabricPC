#!/usr/bin/env python3
"""
Tier A conformance fixture generator (docs/AUDIT_REGISTER.md section 6): primitives —
RoPE, layernorm, every activation's forward+derivative, every energy's
energy+grad_latent+grad_mu — dumped from the PINNED upstream JAX FabricPC checkout into
one .npz consumed by test/conformance/test_tier_a.jl (rtol 1e-6).

RNG TRAP (register section 6): fixtures carry JAX-GENERATED ARRAYS, never seeds — Julia's
RNG is a different algorithm than jax.random, so a shared seed would not reproduce the same
numbers. This script fixes ITS OWN jax.random.PRNGKey purely so re-running it is
reproducible (stable diffs), not to hand the seed to Julia.

grad_mu NOTE: upstream's EnergyFunctional only defines energy()/grad_latent() as closed
forms (energy.py) — there is no hand-written grad_mu() anywhere upstream; the framework
gets dE/dz_mu via JAX autodiff at the node call site. Julia's energy.jl DOES carry an
explicit closed-form grad_mu per energy (an addition beyond a straight port). So grad_mu's
ground truth here is `jax.grad` of `energy(z, mu).sum()` w.r.t. `mu` — an independent
autodiff check of Julia's hand-derived closed form, not a transcription of an upstream
closed form (there isn't one to transcribe).

Run with the upstream fabricpc package importable, e.g.:
    cd ~/dev-zone/FabricPC && source .venv-fixtures/bin/activate  (or: source
    <this repo>/.venv-fixtures/bin/activate if generated there)
    python3 <this repo>/scripts/generate_tier_a_fixtures.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.core.activations import (
    IdentityActivation,
    SigmoidActivation,
    TanhActivation,
    ReLUActivation,
    LeakyReLUActivation,
    GeluActivation,
    SoftmaxActivation,
    HardTanhActivation,
)
from fabricpc.core.energy import (
    GaussianEnergy,
    BernoulliEnergy,
    CrossEntropyEnergy,
    LaplacianEnergy,
    HuberEnergy,
    KLDivergenceEnergy,
)
from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb
from fabricpc.utils.helpers import layernorm

OUT = {}
key = jax.random.PRNGKey(42)


def split():
    global key
    key, sub = jax.random.split(key)
    return sub


def put(prefix, **arrays):
    for name, arr in arrays.items():
        OUT[f"{prefix}_{name}"] = np.asarray(arr)


# ---------------------------------------------------------------------------------------
# Activations: forward + derivative on a standard-normal input; softmax additionally gets
# the full jacobian and an outlier-augmented input (adversarial: exercises max-subtraction).
# NOTE: SoftplusActivation has NO upstream counterpart (grep-verified against
# fabricpc/core/activations.py) — it is a Julia-only addition, excluded here (nothing to
# conform against).
# ---------------------------------------------------------------------------------------
ACTIVATIONS = {
    "identity": IdentityActivation,
    "sigmoid": SigmoidActivation,
    "tanh": TanhActivation,
    "relu": ReLUActivation,
    "leaky_relu": LeakyReLUActivation,
    "gelu": GeluActivation,
    "softmax": SoftmaxActivation,
    "hard_tanh": HardTanhActivation,
}

for name, cls in ACTIVATIONS.items():
    inst = cls()
    x = jax.random.normal(split(), (4, 6))
    fwd = cls.forward(x, inst.config)
    deriv = cls.derivative(x, inst.config)
    put(f"act_{name}", x=x, fwd=fwd, deriv=deriv)
    if name == "softmax":
        # full jacobian on the SAME x as fwd/deriv above (not a fresh draw) so a Julia
        # test can compare all three against one shared input.
        put("act_softmax", jac=cls.jacobian(x, inst.config))

# softmax: an adversarial outlier row (one huge value in an otherwise standard-normal
# row — exercises the max-subtraction numerical-stability path).
sm = SoftmaxActivation()
x_outlier = jax.random.normal(split(), (4, 6))
x_outlier = x_outlier.at[0, 2].set(50.0)  # one large-outlier sample
put(
    "act_softmax_outlier",
    x=x_outlier,
    fwd=SoftmaxActivation.forward(x_outlier, sm.config),
)

# ---------------------------------------------------------------------------------------
# Energies: domain-appropriate (z_latent, z_mu) pairs per energy; energy + grad_latent are
# upstream closed forms, grad_mu is jax.grad ground truth (see module docstring).
# ---------------------------------------------------------------------------------------
B, D = 5, 4


def grad_mu_autodiff(energy_cls, z_latent, z_mu, config):
    f = lambda mu: jnp.sum(energy_cls.energy(z_latent, mu, config))
    return jax.grad(f)(z_mu)


# Gaussian: any reals.
g = GaussianEnergy(precision=1.0)
z = jax.random.normal(split(), (B, D))
mu = jax.random.normal(split(), (B, D))
put(
    "energy_gaussian",
    z=z,
    mu=mu,
    E=GaussianEnergy.energy(z, mu, g.config),
    gradz=GaussianEnergy.grad_latent(z, mu, g.config),
    gradmu=grad_mu_autodiff(GaussianEnergy, z, mu, g.config),
)

# Bernoulli: z binary {0,1}, mu a valid probability (sigmoid of a random logit).
be = BernoulliEnergy(eps=1e-7)
z = jax.random.bernoulli(split(), 0.5, (B, D)).astype(jnp.float32)
mu = jax.nn.sigmoid(jax.random.normal(split(), (B, D)))
put(
    "energy_bernoulli",
    z=z,
    mu=mu,
    E=BernoulliEnergy.energy(z, mu, be.config),
    gradz=BernoulliEnergy.grad_latent(z, mu, be.config),
    gradmu=grad_mu_autodiff(BernoulliEnergy, z, mu, be.config),
)

# CrossEntropy: z one-hot, mu a valid distribution (softmax of random logits).
ce = CrossEntropyEnergy(eps=1e-7)
labels = jax.random.randint(split(), (B,), 0, D)
z = jax.nn.one_hot(labels, D)
mu = jax.nn.softmax(jax.random.normal(split(), (B, D)), axis=-1)
put(
    "energy_crossentropy",
    z=z,
    mu=mu,
    E=CrossEntropyEnergy.energy(z, mu, ce.config),
    gradz=CrossEntropyEnergy.grad_latent(z, mu, ce.config),
    gradmu=grad_mu_autodiff(CrossEntropyEnergy, z, mu, ce.config),
)

# Laplacian: any reals.
la = LaplacianEnergy(scale=1.0)
z = jax.random.normal(split(), (B, D))
mu = jax.random.normal(split(), (B, D))
put(
    "energy_laplacian",
    z=z,
    mu=mu,
    E=LaplacianEnergy.energy(z, mu, la.config),
    gradz=LaplacianEnergy.grad_latent(z, mu, la.config),
    gradmu=grad_mu_autodiff(LaplacianEnergy, z, mu, la.config),
)

# Huber: any reals, scaled up so |diff| straddles delta=1.0 (exercises BOTH branches).
hu = HuberEnergy(delta=1.0)
z = jax.random.normal(split(), (B, D)) * 2.0
mu = jax.random.normal(split(), (B, D)) * 2.0
put(
    "energy_huber",
    z=z,
    mu=mu,
    E=HuberEnergy.energy(z, mu, hu.config),
    gradz=HuberEnergy.grad_latent(z, mu, hu.config),
    gradmu=grad_mu_autodiff(HuberEnergy, z, mu, hu.config),
)

# KLDivergence: both z and mu valid distributions (softmax of random logits).
kl = KLDivergenceEnergy(eps=1e-7)
z = jax.nn.softmax(jax.random.normal(split(), (B, D)), axis=-1)
mu = jax.nn.softmax(jax.random.normal(split(), (B, D)), axis=-1)
put(
    "energy_kldivergence",
    z=z,
    mu=mu,
    E=KLDivergenceEnergy.energy(z, mu, kl.config),
    gradz=KLDivergenceEnergy.grad_latent(z, mu, kl.config),
    gradmu=grad_mu_autodiff(KLDivergenceEnergy, z, mu, kl.config),
)

# ---------------------------------------------------------------------------------------
# RoPE: a real query tensor + the single-nonzero-pair discriminator (pins down BOTH the
# position-index convention and the adjacent-pair grouping convention unambiguously — a
# generic random tensor could pass with either pairing/ordering by coincidence).
# ---------------------------------------------------------------------------------------
Br, S, H, Dh = 2, 5, 1, 8
freqs_cis = precompute_freqs_cis(Dh, S)
put("rope", cos=jnp.real(freqs_cis), sin=jnp.imag(freqs_cis))

xq = jax.random.normal(split(), (Br, S, H, Dh))
xq_out, _ = apply_rotary_emb(xq, xq, freqs_cis)
put("rope_random", x=xq[:, :, 0, :], out=xq_out[:, :, 0, :])

# single-nonzero-pair discriminator: position s=2 (0-based), pair index p=1 (0-based,
# elements Dh indices 2,3), values (x1,x2) = (1.0, 0.0).
xq_pair = jnp.zeros((1, S, 1, Dh))
xq_pair = xq_pair.at[0, 2, 0, 2].set(1.0)  # x1 of pair 1 at position 2
xq_pair = xq_pair.at[0, 2, 0, 3].set(0.0)  # x2 of pair 1 at position 2 (already 0, explicit)
xq_pair_out, _ = apply_rotary_emb(xq_pair, xq_pair, freqs_cis)
put("rope_single_pair", x=xq_pair[:, :, 0, :], out=xq_pair_out[:, :, 0, :])

# ---------------------------------------------------------------------------------------
# LayerNorm: standard-normal + an adversarial outlier row (exercises the eps/sqrt(var)
# path when one feature dominates the row's variance).
# ---------------------------------------------------------------------------------------
Bl, Sl, E = 3, 4, 6
gamma = 1.0 + 0.1 * jax.random.normal(split(), (E,))
beta = 0.1 * jax.random.normal(split(), (E,))
x_ln = jax.random.normal(split(), (Bl, Sl, E))
put(
    "ln",
    x=x_ln,
    gamma=gamma,
    beta=beta,
    out=layernorm(x_ln, gamma, beta),
)
x_ln_outlier = x_ln.at[0, 0, 3].set(1.0e4)
put(
    "ln_outlier",
    x=x_ln_outlier,
    out=layernorm(x_ln_outlier, gamma, beta),
)

# ---------------------------------------------------------------------------------------
np.savez("test/conformance/fixtures/tier_a.npz", **OUT)
print(f"wrote {len(OUT)} arrays to test/conformance/fixtures/tier_a.npz")
