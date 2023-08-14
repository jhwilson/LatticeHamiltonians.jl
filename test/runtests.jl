using LatticeHamiltonians
using Test
using LinearAlgebra

include("wavefunction_helpers.jl")

@testset "LatticeHamiltonians.jl" begin
    include("comparison.jl")
end
