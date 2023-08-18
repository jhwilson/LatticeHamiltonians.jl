using LatticeHamiltonians
using Test
using LinearAlgebra

include("wavefunction_helpers.jl")

@testset "LatticeHamiltonians.jl" begin
    include("comparison.jl")
    include("3d_hamiltonian_build.jl")
end
