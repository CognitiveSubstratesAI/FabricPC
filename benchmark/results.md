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
task (lower lr ceiling); the variance-control property is verified separately in
`test_mupc.jl`. See `docs/decisions.md` §9.
