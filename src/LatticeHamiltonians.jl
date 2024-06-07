"""
Functions and structures for constructing and applying Hamiltonians on lattice systems.
"""
module LatticeHamiltonians

import StaticArrays: SVector, SMatrix, MVector
import LinearAlgebra: I
import MacroTools
import MacroTools: isexpr
import SparseArrays: sparse, dropzeros!
import Base: *, size, length, eltype, adjoint

export LatticeHamiltonian, mul!, *, sparse, size, length, eltype, adjoint
export @lattice_hamiltonian

include("lattice_hamiltonian.jl")
include("build_lattice_operator.jl")
include("dsl.jl")

end
