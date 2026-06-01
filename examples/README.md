# FabricPC examples

## `mnist_pc.jl` — MNIST predictive-coding classifier

A self-contained exhibit of the full FabricPC stack on **real MNIST**, trained
with local predictive-coding learning (inference relaxation + local weight
updates) — **no autodiff**.

```
x(784) → h(128, Tanh) → y(10, Gaussian / MSE-to-one-hot)
```

The standard PC-classification setup: the output is a Gaussian (MSE) energy on a
one-hot target (exact local gradients, clean energy descent), which sidesteps the
diagonal-Softmax approximation. Prediction is `argmax` over the 10 output units.

Dependency-light: the MNIST IDX files are fetched with the `Downloads` stdlib +
`gunzip` and parsed directly — no MLDatasets / CodecZlib. Data is cached under
`~/.cache/fabricpc_mnist/`.

### Run

```bash
julia --project=. examples/mnist_pc.jl
```

Tunables via env vars: `FPC_NTRAIN` (5000), `FPC_NTEST` (2000), `FPC_EPOCHS` (6),
`FPC_LR` (0.002), `FPC_BATCH` (64).

### Reference result (committed)

5000 train / 2000 test, 6 epochs, lr 0.002, eta_infer 0.1, infer_steps 30
(≈1m45s on a 2-core laptop, eager):

| epoch | mean energy | test accuracy |
|------:|------------:|--------------:|
| 0 (random init) | — | 0.0595 |
| 1 | 0.2335 | 0.7465 |
| 2 | 0.1628 | 0.7475 |
| 3 | 0.1613 | 0.7705 |
| 4 | 0.1369 | 0.7410 |
| 5 | 0.1622 | 0.7945 |
| 6 | 0.1162 | **0.8165** |

A pure autodiff-free PC MLP reaches ~82% on the MNIST subset; more data/epochs
push it higher. (Not tuned for SOTA — the point is that the local PC learning
pipeline trains a real classifier end-to-end.)

### Notes

- **Stability**: with 784 fan-in, `lr ≳ 0.005` or too large an `eta_infer`
  diverges to NaN. This is the regime muPC targets — but muPC's PC inference
  needs different relaxation settings than these eager defaults, so the exhibit
  ships the plain, stable config. muPC-on MNIST is left for future tuning.
