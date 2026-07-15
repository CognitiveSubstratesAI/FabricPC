# Shared machinery for windowed nodes (ConvNode, MaxPool, AvgPool) — C-01.
# Port of the helpers in fabricpc/nodes/base.py (`compute_windowed_output_shape`,
# `validate_windowed_output`) plus the index arithmetic that replaces JAX's
# `lax.conv_general_dilated` / `lax.reduce_window`.
#
# THREE things here are load-bearing for upstream fidelity; each is a silent-divergence
# trap (wrong numbers, no error) and each has a dedicated test:
#
#  1. CROSS-CORRELATION, NOT CONVOLUTION. `lax.conv_general_dilated` does NOT flip the
#     kernel (measured against jax 0.10.2: taps [1,10] over [1,2,3,4] give [21,32,43], not
#     the flipped [12,23,34]). Our window index runs FORWARD with the kernel — there is no
#     flip flag to get wrong.
#  2. `cld`/`fld`, NEVER `÷`. Python's `//` FLOORS; Julia's `÷`/`div` TRUNCATES. They differ
#     exactly where it matters: for a window larger than the input, `(n-k)//s + 1` floors to
#     `o <= 0` and upstream RAISES, whereas `÷` truncates to `o = 1` and would silently
#     "validate" a garbage shape. Fail-open is worse than wrong.
#  3. SAME PADDING IS ASYMMETRIC AND THE EXTRA GOES HIGH. `total = max((o-1)*s + k - n, 0)`,
#     `lo = total ÷ 2` (floor), `hi = total - lo` (measured: total=1 ⇒ (lo,hi)=(0,1)).
#     This is the TF/JAX convention; PyTorch differs.
#
# Padding is a TYPE, not a string, for two independent reasons: Enzyme rejects
# `uppercase(::String)` inside a differentiated kernel (`jl_string_to_genericmemory`
# foreigncall), and a normalized-once type cannot be misspelled at a call site
# (guarantee-not-convention).

"""Padding mode for a windowed op. Normalised ONCE at node construction by [`padspec`](@ref)."""
abstract type PadSpec end

"""`"SAME"`: output `ceil(n/stride)`; pad total split with the EXTRA ON THE HIGH side."""
struct SamePad <: PadSpec end

"""`"VALID"`: no padding; output `fld(n - k, stride) + 1`."""
struct ValidPad <: PadSpec end

"""Explicit per-spatial-axis `(low, high)` pads. `R` = spatial rank."""
struct ExplicitPad{R} <: PadSpec
    pads::NTuple{R,Tuple{Int,Int}}
end

"""
    padspec(padding, spatial_rank) -> PadSpec

Normalise a user-supplied padding (`"SAME"`/`"VALID"`, case-insensitive, or a sequence of
`(low, high)` spatial pairs) into a [`PadSpec`](@ref). Deliberate, flagged deviation from
upstream: an unknown padding string raises HERE (construction) rather than inside
`initialize_params`; the message text is upstream's verbatim, only the timing differs.
"""
function padspec(padding, spatial_rank::Int)
    if padding isa AbstractString
        p = uppercase(String(padding))
        p == "SAME" && return SamePad()
        p == "VALID" && return ValidPad()
        throw(ArgumentError(
            "Unknown padding string $(repr(String(padding))); expected 'SAME' or 'VALID'."
        ))
    end
    padding isa PadSpec && return padding
    pairs = Tuple((Int(lo), Int(hi)) for (lo, hi) in padding)
    length(pairs) == spatial_rank || throw(ArgumentError(
        "Explicit padding has $(length(pairs)) spatial pairs but spatial_rank is $spatial_rank."
    ))
    return ExplicitPad{spatial_rank}(pairs)
end

"""
    _pads(in_spatial, window, stride, p::PadSpec) -> NTuple{R,Tuple{Int,Int}}

Per-axis `(low, high)` pad amounts — the port of JAX's `lax.padtype_to_pads`. SAME is
ASYMMETRIC with the extra on the HIGH side (see the module header, trap 3).
"""
_pads(in_sp::NTuple{R,Int}, win, stride, ::ValidPad) where {R} = ntuple(_ -> (0, 0), Val(R))
_pads(in_sp::NTuple{R,Int}, win, stride, p::ExplicitPad{R}) where {R} = p.pads
function _pads(in_sp::NTuple{R,Int}, win, stride, ::SamePad) where {R}
    ntuple(Val(R)) do i
        o = cld(in_sp[i], stride[i])                        # == Python -(-n // s)
        t = max((o - 1) * stride[i] + win[i] - in_sp[i], 0)
        lo = t >> 1                                         # floor(t/2)
        (lo, t - lo)                                        # extra goes HIGH
    end
end

"""
    compute_windowed_output_shape(in_spatial, window, stride, p::PadSpec) -> NTuple{R,Int}

Literal port of `nodes/base.py:602-625`. Raises on a non-positive output size, with upstream's
message. Uses `cld`/`fld` — never `÷` (module header, trap 2).
"""
function compute_windowed_output_shape(
    in_sp::NTuple{R,Int}, win, stride, p::PadSpec
) where {R}
    pd = _pads(in_sp, win, stride, p)
    return ntuple(Val(R)) do i
        n, k, s = in_sp[i], win[i], stride[i]
        o = p isa SamePad ? cld(n, s) : fld(n + pd[i][1] + pd[i][2] - k, s) + 1   # fld == Python //
        o <= 0 && throw(ArgumentError(
            "Windowed op produces a non-positive output size ($o) on spatial axis $(i - 1): " *
            "input=$n, kernel/window=$k, stride=$s, padding=$(_padrepr(p)). The kernel/window " *
            "is larger than the (padded) input along this axis."
        ))
        o
    end
end

_padrepr(::SamePad) = "'SAME'"
_padrepr(::ValidPad) = "'VALID'"
_padrepr(p::ExplicitPad) = string(p.pads)

"""
    validate_windowed_output(node_shape, input_shapes, window, stride, p; op_name,
                             channels_preserved, reject_pad_ge_window=false)

Port of `nodes/base.py:628-707`. Called from `initialize_params` (the only place input shapes
are known), so windowed validation fires there rather than at construction.

THE CHECK ORDER IS LOAD-BEARING (first failure wins) and is asserted by tests: spatial rank →
window/stride present → window length → stride length → `reject_pad_ge_window` (explicit pads
only; `>=` tested on `lo` and `hi` INDEPENDENTLY) → then, per edge IN INSERTION ORDER, the
channel check BEFORE that edge's spatial check.

`input_shapes` must be insertion-ordered (an `OrderedDict`) so the "first offending edge" is
upstream's first offending edge, not a hash-order accident.
"""
function validate_windowed_output(
    node_shape::Tuple, input_shapes::AbstractDict, window, stride, p::PadSpec;
    op_name::String, channels_preserved::Bool, reject_pad_ge_window::Bool=false
)
    spatial_rank = length(node_shape) - 1
    spatial_rank in (1, 2, 3) || throw(ArgumentError(
        "$op_name shape must have 2-4 elements (spatial_rank 1/2/3). Got shape=$(node_shape)."
    ))
    (window === nothing || stride === nothing) && throw(ArgumentError(
        "$op_name: both window/kernel and stride are required for output shape validation; " *
        "got window=$(repr(window)), stride=$(repr(stride))."
    ))
    length(window) == spatial_rank || throw(ArgumentError(
        "$op_name: window/kernel length $(length(window)) must equal spatial_rank " *
        "$spatial_rank inferred from shape=$(node_shape)."
    ))
    length(stride) == spatial_rank || throw(ArgumentError(
        "$op_name: stride length $(length(stride)) must equal spatial_rank $spatial_rank " *
        "inferred from shape=$(node_shape)."
    ))
    if reject_pad_ge_window && p isa ExplicitPad      # skipped for SAME/VALID, as upstream
        for (axis, ((lo, hi), k)) in enumerate(zip(p.pads, window))
            (lo >= k || hi >= k) && throw(ArgumentError(
                "$op_name: explicit padding $((lo, hi)) on spatial axis $(axis - 1) is >= the " *
                "window size $k. A fully-padded max-pool window would output -inf; reduce the " *
                "padding below the window size."
            ))
        end
    end

    declared_spatial = node_shape[1:end-1]
    declared_channels = node_shape[end]
    for (edge_key, in_shape) in input_shapes                # insertion order (see docstring)
        if channels_preserved && in_shape[end] != declared_channels
            throw(ArgumentError(
                "$op_name preserves channels but declares $declared_channels while edge " *
                "'$edge_key' supplies $(in_shape[end]) (input shape $(Tuple(in_shape))). " *
                "Pooling cannot change the channel count; set the node's channel dimension " *
                "to match its input."
            ))
        end
        in_sp = Tuple(in_shape[1:end-1])
        expected_spatial = compute_windowed_output_shape(in_sp, window, stride, p)
        expected_spatial == declared_spatial || throw(ArgumentError(
            "$op_name declared output spatial shape $declared_spatial but edge '$edge_key' " *
            "(input spatial $(in_sp), window $(Tuple(window)), stride $(Tuple(stride)), " *
            "padding $(_padrepr(p))) produces $expected_spatial."
        ))
    end
    return nothing
end

"""
    _edge_keys(inputs) -> Vector{String}

The in-edge keys, in `in_edges` order, for BOTH lanes: `inputs` is an `OrderedDict` on the
eager path and an `SoA` (NOT `<:AbstractDict`) inside the differentiated region. `keys(::Dict)`
returns a non-indexable `KeySet`, so collect. Float `+` is non-associative, so this ORDER is
semantics, not convenience — it must match `forward`'s fold. Marked `@nograd` in the Zygote ext
(it mutates via `grow_to!` and carries no differentiable value).
"""
_edge_keys(inputs) = collect(keys(inputs))

# =========================================================================================
# Window index plans.
#
# 🔴 THE TWO PLANS ENUMERATE A WINDOW IN DIFFERENT ORDERS, ON PURPOSE. This looks like an
# inconsistency and is not — "unifying" them silently breaks upstream fidelity in a way only
# `maxpool_ptie` catches. Read both notes before touching either:
#
#   `_im2col_plan` (conv)  -> COLUMN-MAJOR window order (k1 fastest).
#       Forced by Julia's column-major `reshape`: the conv contracts the patch vector against
#       `reshape(W, prod(window)*C_in, C_out)`, whose leading index unrolls (k1,…,kR,C_in)
#       with k1 fastest. The patch axis MUST unroll identically or we contract mismatched
#       pairs — a right-shaped, entirely wrong tensor.
#
#   `_pool_plan` (max/avg) -> ROW-MAJOR window order (last spatial axis fastest).
#       Forced by TIE-BREAKING, and MEASURED, not reasoned: `lax.reduce_window`'s max VJP
#       routes the gradient to the first maximal element in NHWC (row-major) window order.
#       Probed against jax 0.10.2 with a partial tie — window [[0,1],[1,0.5]], where (0,1) and
#       (1,0) tie — and the gradient lands on (0,1), i.e. ROW-MAJOR-first; a column-major
#       reduction picks (1,0) and diverges. `maximum`/`argmax` take the FIRST maximum in
#       array order, so the plan's order IS the tie-break policy.
#       NB an ALL-tied window cannot distinguish the two (both start at (0,0)) — that is why
#       the fixture is a PARTIAL tie (`maxpool_ptie` in tier_conv.npz). Pooling has no weight
#       matrix, so it is free of the reshape constraint that pins conv.
#
# Both return `J[k, o]::Int`: the COLUMN-MAJOR linear index into the input's flattened spatial
# dims for window-cell `k` of output position `o`, or 0 for a padded cell. Int-only and derived
# from shapes alone (no differentiable value) => `@nograd` in the Zygote ext.
# =========================================================================================

# Column-major unravel: index -> subscript, first axis fastest.
@inline function _unravel_col(i::Int, dims::NTuple{R,Int}) where {R}
    r = i - 1
    ntuple(Val(R)) do d
        s = r % dims[d]
        r = r ÷ dims[d]
        s + 1
    end
end

# Row-major unravel: index -> subscript, LAST axis fastest (C order / NHWC window order).
@inline function _unravel_row(i::Int, dims::NTuple{R,Int}) where {R}
    r = i - 1
    sub = ntuple(Val(R)) do d
        rd = R - d + 1                      # walk axes back-to-front
        s = r % dims[rd]
        r = r ÷ dims[rd]
        (rd, s + 1)
    end
    ntuple(d -> sub[R - d + 1][2], Val(R))
end

function _window_plan(
    in_sp::NTuple{R,Int}, out_sp::NTuple{R,Int}, win::NTuple{R,Int},
    stride::NTuple{R,Int}, pads::NTuple{R,Tuple{Int,Int}}, rowmajor::Bool
) where {R}
    K, O = prod(win), prod(out_sp)
    J = zeros(Int, K, O)                      # 0 == padded cell (sentinel; see _gather)
    @inbounds for oi in 1:O
        osub = _unravel_col(oi, out_sp)       # output positions always column-major (Julia layout)
        for ki in 1:K
            ksub = rowmajor ? _unravel_row(ki, win) : _unravel_col(ki, win)
            lin, acc, ok = 0, 1, true
            for d in 1:R
                # FORWARD with the window: cross-correlation, no flip (header trap 1).
                p = (osub[d] - 1) * stride[d] + ksub[d] - pads[d][1]
                if p < 1 || p > in_sp[d]
                    ok = false
                    break
                end
                lin += (p - 1) * acc
                acc *= in_sp[d]
            end
            J[ki, oi] = ok ? lin + 1 : 0
        end
    end
    return J
end

"""Conv window plan — COLUMN-MAJOR window order (pairs with `reshape(W,…)`). See the block note."""
_im2col_plan(in_sp, out_sp, win, stride, pads) =
    _window_plan(in_sp, out_sp, win, stride, pads, false)

"""Pool window plan — ROW-MAJOR window order (pins JAX's max tie-break). See the block note."""
_pool_plan(in_sp, out_sp, win, stride, pads) =
    _window_plan(in_sp, out_sp, win, stride, pads, true)

"""
    _gather(x, J, C) -> (B, K, O, C)

Gather each window cell for every output position. `x` is `(B, spatial…, C)`; the returned
array is indexed `[b, k, o, c]`.

The padded cells use a ZERO-SENTINEL ROW rather than an explicit padded copy of `x`: we prepend
one zero row to the flattened spatial axis and shift every index by 1, so `J == 0` reads that
row. Two reasons: it allocates one row instead of a full padded tensor per axis, and it keeps
the whole thing a single `getindex` — which Zygote/Enzyme differentiate directly, with no
mutation on the tape.

🔴 THE SENTINEL IS ZERO, AND ZERO IS ONLY CORRECT FOR SOME CONSUMERS. Two live traps, both
of the "passes the easy fixture, fails the real one" kind:

 1. `MaxPool` must NOT reduce over the zero sentinel. Zero BEATS a genuinely negative
    activation, so a padded window would report 0 instead of the true (negative) max —
    invisible on non-negative test data, wrong the moment a ReLU-free conv feeds negatives.
    Upstream fills with `-jnp.inf` (`pooling.py:228`), so `_pool_max` must use `-Inf32`, a
    TRUE IEEE infinity — NOT `typemin(Float32)` (== -3.4f38, FINITE). They diverge on an
    all-padding window: `-Inf32` gives `-Inf` (degenerate but well-defined and matching JAX),
    `typemin` gives a large finite number that then participates in argmax/tie-breaking
    differently. Upstream guards this with `reject_pad_ge_window=true` for MaxPool
    (`pooling.py:213`), which our validator ports — but match the fill anyway rather than
    resting bit-exactness on a validator's reachability argument.

 2. `AvgPool` with `count_include_pad=false` needs the REAL-CELL COUNT, threaded separately.
    The zero sentinel is correct for the NUMERATOR (padded cells add nothing, exactly like
    upstream's `reduce_window(add)`), but the DENOMINATOR must count only real cells. Dividing
    by the full window volume there computes the wrong mean for every padded window — and
    since padding only reaches a window under SAME/explicit, it passes VALID silently and
    fails SAME. Upstream derives the count by summing ones through the same window and guards
    `count == 0` → 0.0 (`pooling.py:337-352`); our equivalent is `count_nonzero(J[:,o] .!= 0)`.
    `count_include_pad=true` (the DEFAULT) needs no count: it divides by `prod(window)`.
    Non-vacuous by measurement — the two settings differ by 0.807 on `avgpool_cip_*`.
    ➤ WRITE THE DENOMINATOR FIRST and gate it on `avgpool_cip_0` before the numerator is even
      correct — invert the usual order deliberately. This divisor is the highest-risk single
      line in the pool kernels: it is the one place here whose failure is SILENT (right
      numerator, wrong divisor, green on every VALID case) rather than loud. Same reasoning as
      oracle-before-kernel and backward-before-forward: do the treacherous part first, against
      the oracle, while you are still suspicious of it. The numerator is the part that is
      obviously right, so it can wait.

Zero IS the correct fill for conv (a padded tap contributes nothing) and for AvgPool's
numerator. Gate both traps on the fixtures the moment the pool kernels exist:
`maxpool_ptie` + `avgpool_cip_0` (the padded, count_include_pad=false case) FIRST — same
class of fill/sentinel bug, and cheaper to hit now than after a green VALID-only forward.
"""
function _gather(
    x::AbstractArray{Float32,N}, J::Matrix{Int}, C::Int, padfill::Float32=0.0f0
) where {N}
    B = size(x, 1)
    S = prod(ntuple(d -> size(x, d + 1), Val(N - 2)))     # flattened spatial extent
    xs = reshape(x, B, S, C)
    xz = cat(fill(padfill, B, 1, C), xs; dims=2)          # sentinel at flat index 1
    G = xz[:, vec(J) .+ 1, :]                             # (B, K*O, C) — differentiable getindex
    return reshape(G, B, size(J, 1), size(J, 2), C)       # vec(J) unrolls k fastest ⇒ (B,K,O,C)
end

"""
    _pool_counts(J) -> (1, 1, O, 1) Float32

Number of REAL (non-padded) cells per output window — `J`'s zero entries are the padded ones.
This is `AvgPool`'s `count_include_pad=false` DIVISOR, and it is the reason that mode cannot
reuse the numerator's zero sentinel: zeros are right for the SUM and wrong for the COUNT.
Upstream derives the identical quantity by summing an all-ones array through the same window
(`pooling.py:348-350`); counting `J`'s non-sentinel entries is the same number, exactly, without
a second reduce. Int-derived, no differentiable value ⇒ `@nograd`-able with the plans.
"""
_pool_counts(J::Matrix{Int}) =
    reshape(Float32[count(!iszero, view(J, :, o)) for o in axes(J, 2)], 1, 1, size(J, 2), 1)

"""
    _pool_max(x, stride, pad, win) -> (B, out_spatial…, C)

Windowed max. Port of `lax.reduce_window(x, -inf, lax.max, …)` (`pooling.py:228`).

TWO fidelity constraints, both MEASURED against jax 0.10.2, neither obvious from the code:

 1. **Fill is `-Inf32`, a TRUE IEEE infinity — never `typemin(Float32)`** (== -3.4f38, finite).
    Zero would be catastrophic (it beats any negative activation); `typemin` is subtler — it
    only diverges on an all-padding window, which upstream's `reject_pad_ge_window=true` guard
    makes unreachable. We match the fill anyway rather than rest bit-exactness on a validator's
    reachability argument — see decisions.md §29.
 2. **The window is reduced in `_pool_plan`'s ROW-MAJOR order, and that IS the tie-break policy.**
    `maximum` takes the FIRST maximal element in array order, and Zygote's ∇maximum routes the
    gradient to exactly that one (one-hot — measured). JAX ties row-major/NHWC: probed with a
    partial tie (`[[0,1],[1,0.5]]`) the gradient lands on `(0,1)`; a column-major reduction picks
    `(1,0)`. An ALL-tied window cannot tell these apart (both start at `(0,0)`), which is why the
    gate is `maxpool_ptie`, a PARTIAL tie.
"""
function _pool_max(
    x::AbstractArray{Float32,N}, stride::NTuple{R,Int}, pad::PadSpec, win::NTuple{R,Int}
) where {N,R}
    B, C = size(x, 1), size(x, N)
    in_sp = ntuple(i -> size(x, i + 1), Val(R))
    pads = _pads(in_sp, win, stride, pad)
    out_sp = compute_windowed_output_shape(in_sp, win, stride, pad)
    J = _pool_plan(in_sp, out_sp, win, stride, pads)         # ROW-MAJOR window order (tie policy)
    G = _gather(x, J, C, -Inf32)                             # (B,K,O,C); -Inf on padded cells
    M = maximum(G; dims=2)                                   # first max in K order == row-major
    return reshape(M, B, out_sp..., C)
end

"""
    _pool_avg(x, stride, pad, win, count_include_pad) -> (B, out_spatial…, C)

Windowed mean. Port of `lax.reduce_window(x, 0, lax.add, …)` + the divisor (`pooling.py:333-352`).

🔴 THE DIVISOR IS THE WHOLE FUNCTION. The numerator is trivially right — padded cells gather the
zero sentinel and add nothing, exactly like upstream's `reduce_window(add)`. The DIVISOR is where
the only silent failure in the pool kernels lives, so it is written first and gated first
(decisions.md §30: the part that fails silently goes first):

 * `count_include_pad=true` (the DEFAULT, matching PyTorch) divides by the FULL window volume —
   padded cells count as zeros. Exact under VALID, where no padding exists.
 * `count_include_pad=false` divides each window by its REAL cell count (`_pool_counts`).

Get this wrong and it is invisible on every VALID case (no padding ⇒ count == prod(win) ⇒ both
modes agree) and wrong on every SAME/explicit case where padding reaches a window. Measured
non-vacuous: the two modes differ by 0.807 on `avgpool_cip_*`.

The `count == 0` guard (fully-padded window) mirrors upstream's `jnp.where(counts > 0, …, 0.0)`:
it yields 0.0 rather than NaN, and the `safe_counts` divisor keeps the NaN out of the GRADIENT
too (0/0 differentiates to NaN even where `where` masks the forward value).
"""
function _pool_avg(
    x::AbstractArray{Float32,N}, stride::NTuple{R,Int}, pad::PadSpec, win::NTuple{R,Int},
    count_include_pad::Bool
) where {N,R}
    B, C = size(x, 1), size(x, N)
    in_sp = ntuple(i -> size(x, i + 1), Val(R))
    pads = _pads(in_sp, win, stride, pad)
    out_sp = compute_windowed_output_shape(in_sp, win, stride, pad)
    J = _pool_plan(in_sp, out_sp, win, stride, pads)
    S = sum(_gather(x, J, C, 0.0f0); dims=2)                 # numerator: zeros add nothing
    if count_include_pad
        return reshape(S ./ Float32(prod(win)), B, out_sp..., C)
    end
    cnt = _pool_counts(J)                                    # (1,1,O,1) real-cell counts
    safe = max.(cnt, 1.0f0)                                  # keep 0/0 out of value AND gradient
    A = ifelse.(cnt .> 0.0f0, S ./ safe, 0.0f0)
    return reshape(A, B, out_sp..., C)
end

"""
    _pool_avg_global(x) -> (B, C)

Global mean over ALL spatial axes (`pooling.py:321-324`). NB the axis set comes from the INPUT's
rank, not the node's declared `shape` — a global-pool node's shape is `(C,)`, i.e. spatial rank 0,
so deriving the axes from it would reduce nothing.
"""
_pool_avg_global(x::AbstractArray{Float32,N}) where {N} =
    dropdims(sum(x; dims=ntuple(i -> i + 1, Val(N - 2))); dims=ntuple(i -> i + 1, Val(N - 2))) ./
    Float32(prod(ntuple(d -> size(x, d + 1), Val(N - 2))))

"""
    _conv_im2col(x, W, stride, pad) -> (B, out_spatial…, C_out)

One in-edge's convolution contribution: CROSS-CORRELATION (no kernel flip), channels-last,
computed as an im2col gather + a single GEMM. `x` is `(B, in_spatial…, C_in)`; `W` is
`(window…, C_in, C_out)` — upstream's HWIO/LIO/DHWIO layout, used VERBATIM (no permutation).

🔴 THE PATCH AXIS AND `reshape(W, …)` MUST UNROLL IDENTICALLY — this is the contract the whole
function exists to honour. Julia's `reshape` is COLUMN-MAJOR, so `reshape(W, K*C_in, C_out)`'s
leading index unrolls `(window…, C_in)` with `window[1]` fastest: flat index `k + (c-1)*K`,
where `k` is the column-major linear index over the window. The patch axis below is built to
match exactly — `_im2col_plan` enumerates window cells column-major (that is WHY it does; see
the plan block note above), and the trailing `reshape` of `(K, C_in)` yields the same
`k + (c-1)*K`. Get the pairing wrong and the GEMM contracts mismatched pairs, producing a
correctly-SHAPED, entirely wrong tensor: no error, no shape mismatch, just wrong numbers. Only
the non-square-kernel / asymmetric-weight fixtures catch it — which is why they exist.
(`_pool_plan` is ROW-major for an unrelated reason: tie-breaking. The two orders disagree on
purpose. Do not "unify" them.)

Allocation-functional throughout (gather → permutedims → reshape → GEMM; no mutation), which is
what lets Zygote/Enzyme differentiate it directly. The only mutation is inside the Int-valued
`_im2col_plan`, which is `@nograd` and carries no differentiable value.
"""
function _conv_im2col(
    x::AbstractArray{Float32,N}, W::AbstractArray{Float32,M},
    stride::NTuple{R,Int}, pad::PadSpec
) where {N,M,R}
    B = size(x, 1)
    # Ranks come from the ARRAYS (type params), never from `node.shape::Tuple` (abstract).
    in_sp = ntuple(i -> size(x, i + 1), Val(R))
    win = ntuple(i -> size(W, i), Val(R))
    Cin, Cout = size(W, R + 1), size(W, R + 2)
    pads = _pads(in_sp, win, stride, pad)
    out_sp = compute_windowed_output_shape(in_sp, win, stride, pad)

    J = _im2col_plan(in_sp, out_sp, win, stride, pads)      # (K, O) — COLUMN-MAJOR window order
    K, O = size(J, 1), size(J, 2)
    G = _gather(x, J, Cin)                                  # (B, K, O, C_in)
    # (B,K,O,Cin) → (B,O,K,Cin) → (B,O,K*Cin): the trailing reshape unrolls (K,Cin) as
    # k + (c-1)*K — identical to reshape(W, K*Cin, Cout). THAT identity is the contract above.
    P = reshape(permutedims(G, (1, 3, 2, 4)), B, O, K * Cin)
    # (B*O, K*Cin) * (K*Cin, Cout). Column-major flattening puts b fastest in the row index, so
    # the result reshapes back with b fastest and o unrolling column-major over out_sp — matching
    # how `_window_plan` built J's columns (`_unravel_col(oi, out_sp)`).
    out2 = reshape(P, B * O, K * Cin) * reshape(W, K * Cin, Cout)
    return reshape(out2, B, out_sp..., Cout)
end
