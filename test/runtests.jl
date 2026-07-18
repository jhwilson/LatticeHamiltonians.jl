using LatticeHamiltonians
using Test
using LinearAlgebra
using SparseArrays

include("wavefunction_helpers.jl")

@testset "LatticeHamiltonians.jl" verbose = true begin
    @testset "Unit tests" begin
        include("exprs.jl")
    end

    @testset "Integration Tests" begin
        include("comparison.jl")
        include("3d_hamiltonian_build.jl")
        include("parameter_mutability.jl")
    end
end
