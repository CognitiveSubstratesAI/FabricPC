# FabricPC benchmark results

Committed reference numbers. Re-run with `julia --project=. benchmark/throughput.jl`.

## Eager train_step throughput

Net `784 → 128 → 10` (Tanh hidden, Gaussian output), `infer_steps = 30`, full PC
`train_step` (inference relaxation + local weight gradients + SGD). Min-of-20,
GC disabled, 2-core laptop. **Eager baseline — no Reactant/JIT yet.**

| batch | train (ms/step) | throughput (samp/s) |
|------:|----------------:|--------------------:|
| 16    |  63.4           | 252.4 |
| 64    | 158.2           | 404.5 |
| 256   | 637.8           | 401.4 |

Throughput plateaus near ~400 samples/s; batch 64+ amortizes the per-step
overhead. A Reactant/XLA path (deferred) is the obvious next lever.

## MNIST classifier accuracy

`examples/mnist_pc.jl`, 5000 train / 2000 test, 6 epochs, lr 0.002,
eta_infer 0.1, infer_steps 30 (≈1m45s):

| epoch | mean energy | test accuracy |
|------:|------------:|--------------:|
| 0 (random) | — | 0.0595 |
| 1 | 0.2335 | 0.7465 |
| 3 | 0.1613 | 0.7705 |
| 6 | 0.1162 | **0.8165** |

Pure autodiff-free predictive-coding MLP: 5.95% → 81.65% on the MNIST subset.

**muPC variant** (`FPC_MUPC=1`, hidden-only, eta_infer 0.02, lr 0.02, 40 infer
steps): 5.70% → 77.45%. muPC-on trains but does not beat the plain config on this
shallow task (lower lr ceiling); the variance-control property is verified
separately in `test_mupc.jl`. See `docs/decisions.md` §9.

## muPC FC-ResNet (full recipe) — `examples/mupc_resnet.jl`

Deep FC-ResNet with the COMPLETE muPC recipe (MuPCInitializer unit-variance init +
per-edge scaling + AdamW + exact Softmax/CE gradient). 8 residual blocks, hidden
64, 8000 train / 2000 test, 5 epochs, lr 0.002 (~5m):

| epoch | mean energy | test accuracy |
|------:|------------:|--------------:|
| 0 (random) | — | 0.0585 |
| 1 | 0.7964 | 0.5415 |
| 3 | 0.1931 | 0.7620 |
| 5 | 0.0867 | **0.8160** |

The energy descends monotonically and a **deep (8-layer) PC ResNet trains** —
this is the muPC win (decisions.md §10). The earlier §9 failure (energy ascending)
was a unit-variance-init + exact-softmax-gradient gap, now closed.

## Reactant/XLA JIT — feasibility + payoff (`reactant_jit.jl`)

FabricPC's eager path uses Dict-keyed `GraphState`, which XLA cannot trace. This
benchmark reformulates the SAME PC inference math (the hot `infer_steps` loop)
over a fixed tuple of per-node arrays — the representation a full Reactant
integration would use — and compiles it with `Reactant.@compile`. MNIST-shaped
MLP (784→128→10, Gaussian), batch 256, 20 inference steps:

| path | time (ms) | speedup |
|------|----------:|--------:|
| eager (naive Julia) | 50.7 | 1× |
| Reactant/XLA JIT (synced) | 5.8 | **8.8×** |

JIT matches eager to `max|Δ| = 2.9e-6`. (Timing forces `Array(...)` materialization
— without it, async dispatch shows a misleading ~1000×.) The eager baseline here
is allocation-naive; FabricPC's Dict-based eager is slower still, so the
in-package payoff would be ≥ this. Confirms the JIT path is worth the GraphState
refactor — see `docs/decisions.md` §11.

## Reactant/XLA JIT — in-package (FabricPCReactantExt, `examples/jit_inference.jl`)

`compile_inference` (the opt-in extension) JIT-compiles the inference loop. Vs the
real eager **Dict** path (not the naive-array microbenchmark above), MNIST-shaped
MLP (784→128→10), batch 256, 20 steps:

| path | time (ms) | speedup |
|------|----------:|--------:|
| eager (Dict GraphState) | 280.0 | 1× |
| Reactant/XLA JIT | 8.7 | **32×** |

JIT == eager to `max|Δ| = 0` (bit-exact). Reactant is a weakdep — core stays
Reactant-free. See `docs/decisions.md` §11.
