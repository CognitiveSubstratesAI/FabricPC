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

### muPC variant

`FPC_MUPC=1 julia --project=. examples/mnist_pc.jl` enables **hidden-only muPC**
(scales the hidden edges to keep activations O(1); the MSE one-hot output is left
unscaled). It switches to the stable muPC inference settings (`eta_infer=0.02`,
`infer_steps=40`, `lr=0.02`) and trains to **~77%** (5.70% → 77.45%, 5k/2k, 6ep).

Findings (see `docs/decisions.md` §9):

- `MuPCConfig(include_output=true)` is unsuitable for an MSE one-hot classifier —
  it scales the output by ≈1/N, capping `z_mu` near 0 so it cannot reach a
  one-hot target. Use `include_output=false` (hidden-only) for MSE classification.
- This shallow `FPC_MUPC` variant does not beat the plain config. The **full**
  muPC recipe (unit-variance init + AdamW + deep ResNet + exact softmax/CE) DOES
  train deep nets — see `mupc_resnet.jl` below and `docs/decisions.md` §10.

## `mupc_resnet.jl` — deep muPC FC-ResNet (the full recipe)

`julia --project=. examples/mupc_resnet.jl` trains a deep fully-connected
residual PC network with the **complete muPC recipe**: `MuPCInitializer`
(unit-variance) + per-edge muPC scaling + `AdamW` + the exact Softmax/CE gradient
(`dE/dpre = s − y`). Autodiff-free.

```
input(784) → stem(64) → [8 × LinearResidual(64, Tanh)] → output(10, Softmax/CE)
```

Reference result (8 blocks, 8000 train / 2000 test, 5 epochs, lr 0.002, ~5m):

| epoch | mean energy | test accuracy |
|------:|------------:|--------------:|
| 0 (random) | — | 0.0585 |
| 1 | 0.7964 | 0.5415 |
| 5 | 0.0867 | **0.8160** |

A deep (8-layer) PC ResNet trains, energy descending monotonically — the muPC
win. Env: `FPC_BLOCKS` (8), `FPC_HID` (64), `FPC_NTRAIN`/`FPC_NTEST`/`FPC_EPOCHS`.

### Notes

- **Stability**: with 784 fan-in, plain `lr ≳ 0.005` or too large an `eta_infer`
  diverges to NaN — hence the small default lr.
