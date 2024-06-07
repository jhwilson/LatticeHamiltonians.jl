"""
Functions and structures for constructing and applying Hamiltonians on lattice systems.
"""
module LatticeHamiltonians

using StaticArrays, LinearAlgebra, SparseArrays
using MacroTools
import SparseArrays: sparse
import Base: *, size, length, eltype, adjoint

export LatticeHamiltonian, mul!, *, sparse, size, length, eltype, adjoint
export @lattice_hamiltonian

include("lattice_hamiltonian.jl")
include("build_lattice_operator.jl")
include("dsl.jl")

end
