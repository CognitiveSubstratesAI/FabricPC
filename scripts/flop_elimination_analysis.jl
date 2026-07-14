# Static FLOP-elimination-fraction analyzer for the clamped-node hoist+prune optimizations.
# RUNNABLE, not hand-derived: walks a real CompiledPlan, computes per-node forward FLOPs and
# per-edge input-gradient FLOPs from the ACTUAL node types + shapes, classifies each as
# clamped-incident (hoistable forward: all in-sources clamped; prunable input-grad: the edge's
# source node is clamped), and reports (clamped-incident per-step FLOPs)/(total per-step FLOPs).
# Validated against the MNIST-MLP HLO ground truth (2081M JAX total / 77.6M Julia residual,
# established by operand-level @code_xla vs jax.jit .as_text() inspection this session).
#
# FLOP model (2*M*K*N per GEMM; elementwise ops counted 0 -- negligible vs GEMMs, and the point is
# the GEMM-FLOP ratio XLA's DCE/hoist act on):
#   Linear.forward           = 2*rows*Kin*Nout       (src/nodes/linear.jl:96, pre = x*W)
#   Linear.input_grad/edge   = 2*rows*Nout*Kin       (linear.jl:160, dpre*transpose(W))
#   EmbeddingNode.forward    = 0 (gather, _embed_lookup, transformer_decomposed.jl:294)
#   EmbeddingNode.input_grad = 0 (discrete: input_grads = zero(inputs[k]), transformer_decomposed.jl:335)
#   VocabProjection.forward  = 2*rows*E*V ; input_grad/edge = 2*rows*V*E
#   Identity/Skip            = 0 GEMM (elementwise sum/scale)
#   TransformerBlock.forward = attention+FFN GEMMs (standard formula, for DENOMINATOR context only)
#   TransformerBlock.igrad   = ~2x forward (standard backward; DENOMINATOR only -- see note)
# rows = B * prod(out_shape[1:end-1]).

using FabricPC
using FabricPC: CompiledPlan
using Random

rows(out_shape, B) = B * prod(out_shape[1:end-1])

# forward GEMM FLOPs for one node given its output shape and its in-edge source shapes
function fwd_flops(node, out_shape, src_shapes, B)
    r = rows(out_shape, B)
    Nout = out_shape[end]
    if node isa FabricPC.Linear
        return sum(2 * r * s[end] * Nout for s in src_shapes)          # sum over in-edges
    elseif node isa FabricPC.VocabProjectionNode
        return sum(2 * r * s[end] * Nout for s in src_shapes)          # E -> V GEMM
    elseif node isa FabricPC.EmbeddingNode
        return 0                                                        # gather
    elseif node isa FabricPC.TransformerBlock
        S, E = out_shape
        H = node.num_heads; F = node.ff_dim
        qkv = 3 * 2 * B * S * E * E
        scores = 2 * B * S * S * E
        av = 2 * B * S * S * E
        oproj = 2 * B * S * E * E
        ffn = 2 * B * S * E * F + 2 * B * S * F * E
        return qkv + scores + av + oproj + ffn
    else                                                                # Identity/Skip: elementwise
        return 0
    end
end

# input-gradient GEMM FLOPs for ONE in-edge (gradient flowing back to that edge's source)
function igrad_flops(node, out_shape, src_shape, B)
    r = rows(out_shape, B)
    Nout = out_shape[end]
    if node isa FabricPC.Linear
        return 2 * r * Nout * src_shape[end]
    elseif node isa FabricPC.VocabProjectionNode
        return 2 * r * Nout * src_shape[end]
    elseif node isa FabricPC.EmbeddingNode
        return 0                                                        # discrete: zero(inputs)
    elseif node isa FabricPC.TransformerBlock
        # eager block backward ~2x forward; DENOMINATOR context only (never clamped-incident here)
        return 2 * fwd_flops(node, out_shape, [src_shape], B)
    else
        return 0                                                        # Identity/Skip elementwise
    end
end

function analyze(structure, clamps, B; label="")
    plan = CompiledPlan(structure)
    names = plan.names
    clamped = Bool[n in keys(clamps) for n in names]
    out_shape(i) = (B, plan.infos[i].shape...)[2:end]   # drop batch to get feature shape tuple
    shape_of(i) = Tuple(plan.infos[i].shape)

    total = 0.0
    hoistable = 0.0    # forward of nodes whose ALL in-sources are clamped
    prunable  = 0.0    # input-grad of edges whose SOURCE node (gradient target) is clamped

    for i in eachindex(names)
        info = plan.infos[i]
        node = plan.nodes[i]
        srcs = plan.in_src[i]
        oshape = shape_of(i)
        src_shapes = [shape_of(s) for s in srcs]

        if info.in_degree == 0
            continue                                      # pure source: 0 FLOPs (linear.jl:127-138)
        end
        # eval-output branch (out_degree==0 && !clamped): forward only. Our topologies clamp outputs.
        eval_output = (info.out_degree == 0 && !clamped[i])

        f = fwd_flops(node, oshape, src_shapes, B)
        total += f
        if all(clamped[s] for s in srcs)                  # hoistable: z_mu constant across steps
            hoistable += f
        end

        if !eval_output
            for (k, s) in enumerate(srcs)
                g = igrad_flops(node, oshape, src_shapes[k], B)
                total += g
                if clamped[s]                             # prunable: gradient into a clamped node
                    prunable += g
                end
            end
        end
    end

    ci = hoistable + prunable
    frac = ci / total
    # implied speedup ceiling if XLA hoists (saves (T-1)/T of hoistable) + prunes (all): for a
    # T-step relaxation, JAX pays total*T; optimized pays total*T - hoistable*(T-1) - prunable*T.
    T = structure.config.inference.infer_steps
    opt = total * T - hoistable * (T - 1) - prunable * T
    ceiling = (total * T) / opt
    println(rpad(label, 26),
        "  total/step=", round(total/1e6, digits=2), "M",
        "  hoistable=", round(hoistable/1e6, digits=2), "M",
        "  prunable=", round(prunable/1e6, digits=2), "M",
        "  clamped-incident=", round(100*frac, digits=2), "%",
        "  T=", T, "  FLOP-ceiling=", round(ceiling, digits=2), "x")
    return (; total, hoistable, prunable, frac, T, ceiling, opt, jax=total*T)
end

function run_flop_analysis()
    # ---- MNIST-MLP validation against ground truth (2081M JAX / 77.6M Julia) ----
    println("="^140)
    println("VALIDATION: MNIST-MLP against operand-level HLO ground truth")
    println("="^140)
    B = 256
    x = Linear((784,), "x"; use_bias=false)
    h = Linear((128,), "h"; activation=TanhActivation())
    y = Linear((10,), "y"; energy=GaussianEnergy())
    mnist = graph([x, h, y], [Edge(x, h), Edge(h, y)], TaskMap(; x=x, y=y),
                  InferenceSGD(eta_infer=0.1, infer_steps=20, latent_decay=0.0))
    r = analyze(mnist, Dict("x"=>1, "y"=>1), B; label="MNIST-MLP (784->128->10)")
    println("  JAX-side total (total*20) = ", round(r.jax/1e6, digits=1), "M  [ground truth 2081M]")
    println("  Julia-side optimized      = ", round(r.opt/1e6, digits=1), "M  [ground truth 77.6M]")

    # ---- Deep chains (uniform width) ----
    println()
    println("="^140)
    println("DEEP CHAINS (uniform width W=128): clamped-incident fraction vs depth L")
    println("="^140)
    function chain(L; W=128, B=256, T=20)
        nodes = FabricPC.AbstractNode[Linear((W,), "n0"; use_bias=false)]  # clamped input source
        for i in 1:L
            push!(nodes, Linear((W,), "n$(i)"; activation=TanhActivation()))
        end
        push!(nodes, Linear((W,), "out"; energy=GaussianEnergy()))
        edges = Edge[]
        for i in 1:(length(nodes)-1)
            push!(edges, Edge(nodes[i], nodes[i+1]))
        end
        structure = graph(nodes, edges, TaskMap(; x=nodes[1], y=nodes[end]),
                          InferenceSGD(eta_infer=0.1, infer_steps=T, latent_decay=0.0))
        return structure
    end
    for L in (2, 4, 8, 16, 32)
        s = chain(L)
        analyze(s, Dict("n0"=>1, "out"=>1), 256; label="chain L=$(L) (128-wide)")
    end

    # ---- Transformer-LM (the flagship model) ----
    println()
    println("="^140)
    println("TRANSFORMER-LM (clamped input feeds EmbeddingNode=gather; clamped output from unclamped src)")
    println("="^140)
    for (S, V, E, Hh, nb) in [(8, 10, 8, 2, 1), (128, 256, 64, 4, 2), (256, 1000, 256, 8, 6)]
        st = transformer_lm(; seq_len=S, vocab_size=V, embed_dim=E, num_heads=Hh, num_blocks=nb,
                            eta_infer=0.01)
        lbl = "tf S=$(S) E=$(E) H=$(Hh) blk=$(nb)"
        analyze(st, Dict("input"=>1, "output"=>1), 32; label=lbl)
    end
end
run_flop_analysis()
