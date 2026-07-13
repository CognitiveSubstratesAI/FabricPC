# Installation

FabricPC.jl requires **Julia 1.12+**. It isn't registered yet — install it as a
dev/path dependency from the GitHub repo:

```julia
using Pkg
Pkg.develop(url="https://github.com/CognitiveSubstratesAI/FabricPC")
```

or, working from a local clone:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Optional autodiff backends

The core closed-form nodes ([`Linear`](@ref), [`IdentityNode`](@ref),
[`SkipConnection`](@ref), [`LinearResidual`](@ref)) need no autodiff package at
all. Nodes that use the Phase-D autodiff seam (transformer/attention,
[`StorkeyHopfield`](@ref)) need **exactly one** of:

```julia
using Zygote   # the production seam backend — works on the full attention block
```

or

```julia
using Enzyme   # opt-in, simple/dense nodes only — see docs/decisions.md §19
```

Loading both in the same Julia session raises an error (`_register_ad_backend!`,
see F-04 in `docs/AUDIT_REGISTER.md`) rather than silently letting one backend's
methods shadow the other's.

## JIT (optional)

The Reactant/XLA-compiled inference path needs:

```julia
using Reactant
```

See [JIT with Reactant](@ref).

## Development

Before committing any `.jl` change, run the Blue formatter (CI enforces it,
pinned to JuliaFormatter 2.5.2):

```bash
julia -e 'using JuliaFormatter; format(".")'
```
