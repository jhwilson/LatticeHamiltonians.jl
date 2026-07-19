"""
Functions and structures for constructing and applying Hamiltonians on lattice systems.
"""
module LatticeHamiltonians

import StaticArrays: SVector, SMatrix, MVector
import SparseArrays: sparse, dropzeros!
import Base: *, size, length, eltype, adjoint

export LatticeHamiltonian, @lattice_hamiltonian
export Lattice,
    hamiltonian,
    onsite!,
    hopping!,
    build,
    Periodic,
    Open,
    Twisted,
    uniform,
    FieldSpec,
    Source,
    Target,
    set_parameter!,
    set_field!,
    refresh_caches!,
    parameters,
    fields,
    terms,
    describe,
    validate,
    displacement,
    midpoint

include("lattice_hamiltonian.jl")
include("dsl.jl")
include("builder.jl")
include("build_lattice_operator.jl")
include("linear_algebra.jl")
include("lattice.jl")
include("parameters.jl")
include("coefficients.jl")
include("contexts.jl")
include("model.jl")
include("lower.jl")
include("realization.jl")

end
