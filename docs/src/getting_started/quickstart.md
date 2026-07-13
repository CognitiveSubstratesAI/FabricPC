# Quickstart

A minimal predictive-coding classifier: `x(4) → h(8, Tanh) → y(3, Softmax/CrossEntropy)`,
trained with **local PC learning** (no backprop).

```julia
using FabricPC
using Random

rng = MersenneTwister(0)

# Synthetic separable data: 3 classes in 4-d space.
B = 32
labels = rand(rng, 1:3, B)
x = Float32.(randn(rng, 4, B)') .+ Float32.(labels' .* 2)   # (B, 4), class-shifted
y = Float32[labels[b] == c ? 1 : 0 for b in 1:B, c in 1:3]  # one-hot, (B, 3)

# Build the graph.
xn = Linear((4,), "x")
hn = Linear((8,), "h"; activation=TanhActivation())
yn = Linear((3,), "y"; activation=SoftmaxActivation(), energy=CrossEntropyEnergy())
structure = graph(
    [xn, hn, yn], [Edge(xn, hn), Edge(hn, yn)],
    TaskMap(; x=xn, y=yn),
    InferenceSGD(; eta_infer=0.1, infer_steps=20)
)
params = initialize_params(structure, MersenneTwister(1))

# Train: each step clamps (x, y), relaxes the graph by local inference, then
# applies a local (Hebbian-like) weight update — no backprop anywhere.
batch = Dict("x" => x, "y" => y)
for step in 1:100
    global params
    params, energy, _ = train_step(params, batch, structure, 0.05, rng)
    step % 20 == 0 && @info "train_step" step energy
end

# Predict on a fresh batch (only "x" clamped; "y" relaxes freely to the model's
# best guess).
pred = predict(params, structure, Dict("x" => x), rng)
accuracy = sum(argmax(pred[b, :]) == labels[b] for b in 1:B) / B
@info "accuracy" accuracy
```

## What just happened

- **Inference** (inner loop): with `x`/`y` clamped, `InferenceSGD` relaxes every
  *other* node's latent (`h`, here) by local gradient descent on its own
  prediction error — `infer_steps` times.
- **Learning** (outer loop): [`train_step`](@ref) takes the relaxed state and
  applies a **local** weight update per edge — each edge only needs its own
  source activation and its target's local error, never a gradient that's been
  backpropagated through the rest of the network.
- **Prediction**: [`predict`](@ref) clamps only the input and lets the output
  node's latent relax to the network's own prediction.

## Next steps

- Real data: see `examples/mnist_pc.jl` in the repo (IDX-format MNIST, no
  external dataset dependency) or `examples/char_lm_pc.jl` (character-level
  language modeling on Tiny Shakespeare).
- muPC scaling for deep/wide graphs: pass `scaling=MuPCConfig(...)` to
  [`graph`](@ref) — see [Architecture](@ref).
- Transformers: [`TransformerBlock`](@ref) (monolithic) or the decomposed
  [`MhaResidualNode`](@ref)/[`LnMlp1Node`](@ref)/[`Mlp2ResidualNode`](@ref)
  family — both trained by the same local PC rule, no backprop.
- Compiling inference to XLA: [JIT with Reactant](@ref).
