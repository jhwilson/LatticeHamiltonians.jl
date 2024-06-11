"""
Functions and structures for constructing and applying Hamiltonians on lattice systems.
"""
module LatticeHamiltonians

import StaticArrays: SVector, SMatrix, MVector
import SparseArrays: sparse, dropzeros!
import Base: *, size, length, eltype, adjoint

export LatticeHamiltonian, @lattice_hamiltonian

include("lattice_hamiltonian.jl")
include("linear_algebra.jl")
include("build_lattice_operator.jl")
include("dsl.jl")

end
