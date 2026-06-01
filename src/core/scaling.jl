# muPC scaling. Port of fabricpc/core/scaling.py.
#
# In v0 every node's `scaling_config` is `nothing` (muPC off), so all four
# scaling functions are the identity. Phase C adds `MuPCScalingFactors` plus the
# real per-edge methods (forward_scale / topdown_grad_scale / self_grad_scale /
# weight_grad_scale). Keeping the call-sites wired now means turning muPC on is
# purely additive — the inference + learning loops already invoke these.

scale_inputs(inputs::AbstractDict, ::Nothing) = inputs
scale_input_grads(input_grads::AbstractDict, ::Nothing) = input_grads
scale_self_grad(self_grad, ::Nothing) = self_grad
scale_weight_grads(params_grad::NodeParams, ::Nothing) = params_grad
