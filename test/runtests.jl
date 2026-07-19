using LatticeHamiltonians
using Test
using LinearAlgebra
using Logging
using Random
using SparseArrays

include("wavefunction_helpers.jl")

@testset "LatticeHamiltonians.jl" verbose = true begin
    @testset "Unit tests" begin
        include("exprs.jl")
        include("frontend_types.jl")
        include("frontend_lowering.jl")
    end

    @testset "Integration Tests" begin
        include("comparison.jl")
        include("site_dependent.jl")
        include("3d_hamiltonian_build.jl")
        include("fused_apply.jl")
        include("hop_folding.jl")
        include("parameter_mutability.jl")
        include("linear_algebra.jl")
        include("frontend_models.jl")
        include("frontend_local.jl")
        include("frontend_geometry.jl")
    end
end
