# FabricPC — Julia port of Matthew Behrend's FabricPC (predictive-coding graph
# training framework). Layer 2 of the NGC Julia stack.
#
# See docs/DESIGN.md for the architecture + phased plan, and docs/decisions.md
# for cross-cutting decisions. Standalone package — no NGCSimLib/NGCLearn dep
# (FabricPC's Node/Edge/slot graph is a parallel abstraction; composition is a
# deferred Layer 3).
#
# Phase B (this file's current scope): the minimal autodiff-FREE trainable core.
# Predictive coding's outer (weight) loop does NOT backprop through the inner
# (inference) loop — both gradients are local/analytic. We port upstream's
# explicit closed-form Gaussian gradient path (LinearExplicitGrad), so the whole
# minimal graph trains with no autodiff. Enzyme is deferred to Phase D.

module FabricPC

using Random
using Accessors

const FABRICPC_VERSION = v"0.1.0"

# Every node descriptor subtypes this. Methods (forward, initialize_params,
# forward_and_latent_grads, …) dispatch on the concrete node type — the Julia
# analogue of upstream's `node_class` static-method dispatch.
abstract type AbstractNode end

node_name(n::AbstractNode) = n.name
node_shape(n::AbstractNode) = n.shape

include("core/activations.jl")
include("core/energy.jl")
include("core/initializers.jl")
include("core/topology.jl")
include("core/types.jl")
include("core/state_ops.jl")
include("nodes/base.jl")
include("nodes/linear.jl")
include("nodes/identity.jl")
include("nodes/skip_connection.jl")
include("nodes/linear_residual.jl")
include("core/mupc.jl")
include("core/scaling.jl")
include("core/inference.jl")
include("core/learning.jl")
include("graph_assembly/graph_construction.jl")
include("graph_initialization/params_initializer.jl")
include("graph_initialization/state_initializer.jl")
include("training/train.jl")
include("training/adam.jl")
include("jit_flat.jl")

export FABRICPC_VERSION
# Activations / energy / initializers
export AbstractActivation,
    IdentityActivation,
    SigmoidActivation,
    TanhActivation,
    ReLUActivation,
    LeakyReLUActivation,
    GeluActivation,
    HardTanhActivation,
    SoftmaxActivation,
    variance_gain,
    jacobian_gain
export AbstractEnergy,
    GaussianEnergy,
    BernoulliEnergy,
    CrossEntropyEnergy,
    LaplacianEnergy,
    HuberEnergy,
    KLDivergenceEnergy,
    energy,
    grad_latent,
    grad_mu
export AbstractInitializer,
    ZerosInitializer,
    NormalInitializer,
    MuPCInitializer,
    XavierInitializer,
    initialize
# Topology
export SlotSpec, SlotRef, Edge, slot
# Nodes
export AbstractNode, Linear, IdentityNode, SkipConnection, LinearResidual, get_slots
# Core types
export NodeParams,
    GraphParams,
    NodeState,
    GraphState,
    NodeInfo,
    EdgeInfo,
    SlotInfo,
    GraphStructure
# muPC (Phase C)
export MuPCConfig, MuPCScalingFactors, compute_mupc_scalings
# Inference / learning
export InferenceSGD, run_inference, inference_step, compute_local_weight_gradients
# Assembly + init
export graph,
    TaskMap,
    initialize_params,
    FeedforwardStateInit,
    initialize_graph_state
# Training
export get_graph_param_gradient, train_step, sgd_update, predict, AdamW, step!, train_step!
# JIT (Dict-free traceable inference path — foundation for Reactant)
export CompiledPlan, to_flat_params, to_flat_state, flat_run_inference

end # module FabricPC
