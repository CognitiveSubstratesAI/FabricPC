# Upstream bug report — Reactant.jl: `UndefRefError` on unbounded `Vararg`

**Status:** drafted 2026-07-15, NOT YET FILED (`gh` CLI is not installed on this box; filing needs a
browser or an authenticated `gh`).
**File at:** <https://github.com/EnzymeAD/Reactant.jl/issues/new>
**Verified before drafting** by three independent checks (upstream-source / prior-art / adversary),
all CONFIRM, no blockers. Every claim below was executed, not inferred — the caveats section exists
because two of our first-draft claims were **wrong** and got corrected.

Found while trying to put FabricPC's PC-inference loop through `Reactant.@trace for`
(`docs/AUDIT_REGISTER.md` J-08).

---

## Title

`collect_tvars_in_type!` and `traced_type_inner(::Core.TypeofVararg)` throw `UndefRefError` on an unbounded `Vararg` (undefined `.N`)

## Body

Two functions in `src/Tracing.jl` read `.T`/`.N` off a `Core.TypeofVararg` without an `isdefined`
guard. An **unbounded** `Vararg` has no `.N`, and a **bare** `Vararg` has neither field — so both
sites throw `UndefRefError` instead of Reactant's own intended, typed error. This is reachable from
entirely ordinary Julia: `Tuple === Tuple{Vararg{Any}}`, and `Tuple{Int,Vararg{Int}}` is a
completely normal varargs tuple type.

**Versions:** reproduced on Reactant **v0.2.273**, Julia 1.12.6. The relevant code is
**byte-identical** at the **v0.2.274** tag (newest release) and on **`main`** — verified by fetching
`https://raw.githubusercontent.com/EnzymeAD/Reactant.jl/{main,v0.2.274}/src/Tracing.jl`.

### Minimal reproduction (no downstream package)

```julia
using Reactant

# Site 1 — collect_tvars_in_type!
Reactant.collect_tvars_in_type!(Base.IdSet{TypeVar}(), Tuple)                   # UndefRefError
Reactant.collect_tvars_in_type!(Base.IdSet{TypeVar}(), Tuple{Int,Vararg{Int}})  # UndefRefError
Reactant.collect_tvars_in_type!(Base.IdSet{TypeVar}(), Vararg)                  # UndefRefError
Reactant.collect_tvars_in_type!(Base.IdSet{TypeVar}(), NTuple{2,Int})           # ok  <- control

# Site 2 — traced_type_inner(::Core.TypeofVararg)
Reactant.traced_type_inner(Tuple{Int,Vararg{Int}}, Base.IdDict(),
                           Reactant.TracedTrack, Union{}, Val(1), nothing)      # UndefRefError
```

The `NTuple{2,Int}` control is the point: the branch is fine when `.N` is defined, which isolates the
fault to the undefined-field case rather than to tuples generally.

### Why it's a bug and not an intended refusal

Two structs differing **only** in the tuple field's type:

```julia
struct S;  shape::Tuple;             flag::Bool; end   # -> UndefRefError          (internal crash)
struct Sc; shape::Tuple{Int,Int};    flag::Bool; end   # -> NoFieldMatchError      (clean, intended)
struct Sv; shape::Tuple{Int,Vararg{Int}}; flag::Bool; end  # -> UndefRefError      (internal crash)
```

Reactant already has a correct, typed error path for this situation; the unguarded read preempts it
with an internal crash. And the crash blocks a case Reactant otherwise handles correctly — with the
guard applied, a *parametric* struct traces properly rather than erroring:

```julia
struct P{W}; w::W; shape::Tuple; end
Reactant.traced_type_inner(P{Bool}, Base.IdDict(), Reactant.TracedTrack, Number, Val(1), nothing)
# before: UndefRefError
# after : P{Reactant.TracedRNumber{Bool}}   <- correct
```

### Relationship to #767 (precedent, not a duplicate) — its root cause is still live

[#767 "Tracing `VersionNumber`"](https://github.com/EnzymeAD/Reactant.jl/issues/767) (closed
2025-02-19) is the **same root cause**: `VersionNumber`'s `prerelease`/`build` fields are
`Tuple{Vararg{Union{UInt64,String}}}` — an unbounded `Vararg` with no `.N` — and it produced the same
`UndefRefError` from the same file. It was closed by
[#773](https://github.com/EnzymeAD/Reactant.jl/pull/773), which added `VersionNumber` to a
do-not-trace list rather than guarding the reads. The underlying defect was never fixed, and is
directly demonstrable today on that exact field type:

```julia
Reactant.traced_type_inner(Tuple{Vararg{Union{UInt64,String}}}, Base.IdDict(),
                           Reactant.TracedTrack, Union{}, Val(1), nothing)   # UndefRefError
```

Blacklisting can't cover this case: the trigger is a *user-defined* struct with an abstract `::Tuple`
field, so there's no finite list of types to add to. Relatedly, `traced_tuple_type_inner` already
carries a `if T === Tuple; return T; end` special-case (`src/Tracing.jl:100-102`) — the bare-`Tuple`
shape is already known to bite; the type-var walker just never got the same treatment, and our crash
path routes around that guard.

### Proposed fix

**Site 1** — `src/Tracing.jl:659-662` (`collect_tvars_in_type!`). Verified locally; an unbounded
`Vararg` genuinely depends on no length type-var, so the empty set is the correct answer:

```julia
elseif t isa Core.TypeofVararg
    isdefined(t, :T) && collect_tvars_in_type!(dependencies, t.T)
    isdefined(t, :N) && collect_tvars_in_type!(dependencies, t.N)
end
```

Type-vars are still collected where they exist — checked: `Tuple{Vararg{Any,N}} where N` → `[N]`,
`Tuple{Vararg{T}} where T` → `[T]`, `NTuple{N,T} where {N,T}` → `[T,N]`; and `Tuple{Int,Float32}`
→ `[]` unchanged.

**Site 2** — `src/Tracing.jl:142` (`traced_type_inner(::Core.TypeofVararg)`), currently
`return Vararg{traced_type_inner(T.T, ...), T.N}`. This one must *reconstruct* a `Vararg`, so we'd
appreciate a maintainer's view on the intended shape. What we validated:

```julia
isdefined(T, :T) || return Vararg
TT = traced_type_inner(T.T, seen, mode, track_numbers, ndevices, runtime)
return isdefined(T, :N) ? Vararg{TT,T.N} : Vararg{TT}
```

Happy to open a PR for both sites with the repro lines as tests.

### Caveats, stated up front

* **The crash is conditional, not universal.** It needs `changed == true` — i.e. some *other* field
  of the struct must actually trace — to get past the `if !changed; return T; end` early-return.
  Our reproductions use `mode = TracedTrack` with `track_numbers = Number`. It would be wrong to say
  "any struct with an abstract `::Tuple` field crashes"; e.g.
  `RefValue{Tuple{Linear,Linear,Linear}}` under `ConcreteToTraced` returns fine.
* **This fix does not make our own use case work, and we're not asking it to.** With the guard
  applied, our node struct correctly yields `NoFieldMatchError` — it's non-parametric with `Bool`
  fields, so no traced version is constructible. That's a separate, legitimate limitation. We're
  reporting the crash-instead-of-diagnostic, nothing more.

### How we hit it in practice

Sweeping a loop-invariant struct into `@trace for` loop-carried state:

```
traced_type_inner(::Type{Base.RefValue{Tuple{Linear,Linear,Linear}}})   Tracing.jl:581
  -> traced_tuple_type_inner                                            Tracing.jl:115
  -> traced_type_inner                                                  Tracing.jl:807
  -> collect_tvars_in_type!                                             Tracing.jl:650
  -> collect_tvars_in_type!                                             Tracing.jl:661  <- UndefRefError
```
