# API Reference

```@meta
CurrentModule = FabricPC
```

Auto-generated from inline docstrings, grouped to match `src/FabricPC.jl`'s own
export sections.

## Activations

```@docs
AbstractActivation
IdentityActivation
SigmoidActivation
TanhActivation
ReLUActivation
LeakyReLUActivation
SoftplusActivation
GeluActivation
HardTanhActivation
SoftmaxActivation
variance_gain
jacobian_gain
jacobian
```

## Energy functionals

```@docs
AbstractEnergy
GaussianEnergy
BernoulliEnergy
CrossEntropyEnergy
LaplacianEnergy
HuberEnergy
KLDivergenceEnergy
energy
grad_latent
grad_mu
```

## Initializers

```@docs
AbstractInitializer
ZerosInitializer
NormalInitializer
MuPCInitializer
XavierInitializer
KaimingInitializer
OnesInitializer
UniformInitializer
initialize
```

## Topology

```@docs
SlotSpec
SlotRef
Edge
slot
```

## Nodes

```@docs
AbstractNode
Linear
IdentityNode
SkipConnection
LinearResidual
get_slots
```

## Phase-D autodiff seam

A node implementing only `compute_mu` gets its local PC gradients for free via
reverse-mode AD (Zygote or Enzyme — exactly one backend per session, see
[decisions.md §19/§23](https://github.com/CognitiveSubstratesAI/FabricPC/blob/main/docs/decisions.md)).

```@docs
compute_mu
energy_kernel
```

## Transformer nodes

```@docs
TransformerBlock
MhaResidualNode
LnMlp1Node
Mlp2ResidualNode
EmbeddingNode
VocabProjectionNode
transformer_lm
```

## Hopfield

```@docs
StorkeyHopfield
```

## Core types

```@docs
NodeParams
GraphParams
NodeState
GraphState
NodeInfo
EdgeInfo
SlotInfo
GraphStructure
```

## muPC scaling

```@docs
MuPCConfig
MuPCScalingFactors
compute_mupc_scalings
```

## Inference

```@docs
InferenceSGD
InferenceSGDNormClip
run_inference
inference_step
compute_local_weight_gradients
```

## Graph assembly + initialization

```@docs
graph
TaskMap
initialize_params
FeedforwardStateInit
initialize_graph_state
```

## Training

```@docs
get_graph_param_gradient
train_step
train_step!
sgd_update
predict
AdamW
step!
train_pcn
evaluate_pcn
NaturalGradientDiag
NaturalGradientLayerwise
precondition!
```

## Autoregressive / sequence training

```@docs
create_causal_mask
compute_loss
train_step_autoregressive
train_autoregressive
generate_autoregressive
evaluate_autoregressive
```

## JIT (Dict-free flat lane — foundation for Reactant)

```@docs
CompiledPlan
to_flat_params
to_flat_state
flat_run_inference
flatten_param_arrays
repack_params
state_from_latents
jit_inference_runner
compile_inference
```
