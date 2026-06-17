using FabricPC
using Test

@testset "FabricPC.jl" begin
    @testset "scaffold" begin
        @test FabricPC.FABRICPC_VERSION == v"0.1.0"
    end
    include("test_phase_b.jl")
    include("test_residual.jl")
    include("test_mupc.jl")
    include("test_activations.jl")
    include("test_energies.jl")
    include("test_classifier.jl")
    include("test_optim.jl")
    include("test_mupc_resnet.jl")
    include("test_jit_flat.jl")
    include("test_initializers_natgrad.jl")
    include("test_train_pcn.jl")
    include("test_autodiff_seam.jl")   # Phase D: Enzyme node-local autodiff seam
    include("test_transformer.jl")     # Phase D: PC-transformer block (Enzyme seam)
    include("test_transformer_decomposed.jl")  # fully-PC decomposed stages
    include("test_storkey_hopfield.jl")         # Hopfield associative memory (composite energy)
    include("test_sequence_training.jl")        # autoregressive/sequence trainer (port phases)
end
