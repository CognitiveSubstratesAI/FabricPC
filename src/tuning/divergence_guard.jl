# Trial pruning for hyperparameter search. Port of the FRAMEWORK-INDEPENDENT contribution of
# fabricpc/tuning/bayesian_tuner.py — the scale-free divergence guard — decoupled from optuna.
#
# DECISION (C-10, decisions.md §NN): BayesianTuner-proper is a 418-line optuna wrapper (TPESampler +
# HyperbandPruner). optuna's VALUE is TPE Bayesian sampling; the faithful Julia sampler is
# TreeParzen.jl (same TPE algorithm), NOT PyCall→optuna (a Python shim, not a port) and NOT
# BayesianOptimization.jl (Gaussian-process BO, a different algorithm). The full sampler wrapper
# waits for a live consumer. But the divergence guard below is optuna-INDEPENDENT and IS the novel
# logic — so it is ported now, consuming the `epoch_callback` surface (train_autoregressive).

"""Raised by a [`divergence_guard`](@ref) epoch callback to stop a diverging/unstable training
trial early — the Julia analog of optuna's `TrialPruned`. A tuner loop catches it and records the
trial as pruned rather than a failure."""
struct TrialPruned <: Exception
    reason::String
end

"""
    divergence_guard(; rel_tol=0.1, verbose=false) -> epoch_callback

Build an `epoch_callback(epoch, params, structure, config, rng; energy, ce_loss)` (the
`train_autoregressive` signature) that throws [`TrialPruned`](@ref) when a trial's per-epoch mean
energy shows SCALE-FREE INSTABILITY — i.e. instability that is comparable across architectures,
unlike raw energy magnitude. Two triggers, matching `bayesian_tuner.py:129-145` exactly:

  1. non-finite energy (NaN/Inf) at any epoch, or
  2. energy risen above its best epoch so far by more than `rel_tol`
     (`energy > minimum(seen) * (1 + rel_tol)`).

Stops a blowing-up trial mid-training instead of wasting the full budget. Framework-independent:
pair it with any sampler (TreeParzen.jl for a faithful TPE search) via `train_autoregressive`'s
`epoch_callback`. Returns the epoch's `energy` so `epoch_results` still carries the trajectory.
"""
function divergence_guard(; rel_tol::Real=0.1, verbose::Bool=false)
    seen = Float32[]
    return function (epoch, params, structure, config, rng; energy, ce_loss)
        if !isfinite(energy)
            throw(TrialPruned("Non-finite energy at epoch $epoch"))
        end
        push!(seen, Float32(energy))
        baseline = minimum(seen)
        if energy > baseline * (1 + rel_tol)
            throw(TrialPruned(
                "Energy diverged at epoch $epoch: rose to " *
                "$(round(energy; digits=4)) from a low of $(round(baseline; digits=4))"
            ))
        end
        verbose && @info "trial epoch" epoch energy=round(energy; digits=4) baseline=round(baseline; digits=4)
        return energy
    end
end
