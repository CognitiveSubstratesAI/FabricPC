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

"""
    AbstractNode

Every node descriptor subtypes this ([`Linear`](@ref), [`IdentityNode`](@ref),
[`SkipConnection`](@ref), [`LinearResidual`](@ref), [`TransformerBlock`](@ref),
[`MhaResidualNode`](@ref), [`LnMlp1Node`](@ref), [`Mlp2ResidualNode`](@ref),
[`EmbeddingNode`](@ref), [`VocabProjectionNode`](@ref), [`StorkeyHopfield`](@ref)). Methods
(`forward`, `initialize_params`, `forward_and_latent_grads`, `forward_and_weight_grads`, …)
dispatch on the concrete node type — the Julia analogue of upstream's `node_class`
static-method dispatch.
"""
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
include("nodes/autodiff.jl")
include("nodes/transformer.jl")
include("nodes/transformer_decomposed.jl")
include("nodes/storkey_hopfield.jl")
include("core/mupc.jl")
include("core/scaling.jl")
include("core/inference.jl")
include("core/learning.jl")
include("graph_assembly/graph_construction.jl")
include("graph_initialization/params_initializer.jl")
include("graph_initialization/state_initializer.jl")
include("training/train.jl")
include("training/adam.jl")
include("training/natural_gradients.jl")
include("training/train_autoregressive.jl")
export create_causal_mask,
    compute_loss,
    train_step_autoregressive,
    train_autoregressive,
    generate_autoregressive,
    evaluate_autoregressive
include("models/transformer_lm.jl")
export transformer_lm
include("jit_flat.jl")

export FABRICPC_VERSION
# Activations / energy / initializers
export AbstractActivation,
    IdentityActivation,
    SigmoidActivation,
    TanhActivation,
    ReLUActivation,
    LeakyReLUActivation,
    SoftplusActivation,
    GeluActivation,
    HardTanhActivation,
    SoftmaxActivation,
    variance_gain,
    jacobian_gain,
    jacobian
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
    KaimingInitializer,
    OnesInitializer,
    UniformInitializer,
    initialize
# Topology
export SlotSpec, SlotRef, Edge, slot
# Nodes
export AbstractNode, Linear, IdentityNode, SkipConnection, LinearResidual, get_slots
# Phase-D autodiff seam (generic local-PC gradients; activate with `using Enzyme`)
export compute_mu, energy_kernel
# AD backend selection (F-04, docs/AUDIT_REGISTER.md section 1): ADBackend/NoBackend/
# ZygoteBackend/EnzymeBackend are marker types dispatched on by the Phase-D seam;
# set_ad_backend! is the deliberate-switch escape hatch (bypasses the load-order guard).
export ADBackend, NoBackend, ZygoteBackend, EnzymeBackend, set_ad_backend!
# PC-transformer block (local-PC gradients via the Enzyme seam)
export TransformerBlock
# Decomposed (fully-PC) transformer stages
export MhaResidualNode, LnMlp1Node, Mlp2ResidualNode, EmbeddingNode, VocabProjectionNode
# Hopfield associative-memory node (composite PC + attractor energy)
export StorkeyHopfield
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
export InferenceSGD,
    InferenceSGDNormClip, InferenceSGDMomentum, run_inference, inference_step,
    momentum_inference_step, compute_local_weight_gradients
# Assembly + init
export graph,
    TaskMap,
    initialize_params,
    FeedforwardStateInit,
    initialize_graph_state
# Training
export get_graph_param_gradient, train_step, sgd_update, predict, AdamW, step!, train_step!
export train_pcn, evaluate_pcn
export NaturalGradientDiag, NaturalGradientLayerwise, precondition!
# JIT (Dict-free traceable inference path — foundation for Reactant)
export CompiledPlan,
    to_flat_params,
    to_flat_state,
    flat_run_inference,
    flatten_param_arrays,
    repack_params,
    state_from_latents,
    jit_inference_runner,
    compile_inference

end # module FabricPC
