import LinearAlgebra: I

"""
    LatticeHamiltonian{real_dim,lattice_dim,F,S}

Stores information about the Hamiltonian of the system, so that matrix-vector multiplication
can be performed efficiently.

This object should not be initialized by the user.
Instead, for construction of the Hamiltonian, see the `@lattice_hamiltonian` macro.

# Fields

  - `A::SMatrix{real_dim,lattice_dim,Float64}`: Matrix whose columns are the primitive lattice vectors
  - `B::SMatrix{real_dim,lattice_dim,Float64}`: Matrix whose columns are the reciprocal lattice vectors
  - `d::Int`: Number of orbitals per site
  - `L::MVector{lattice_dim,Int}`: System size
  - `params::Dict{Symbol,ComplexF64}`: Parameters for potential and hopping functions
  - `r::Vector{SVector{real_dim,Float64}}`: Real-space coordinates of orbitals within a unit cell
  - `apply!::F`: Function that applies the Hamiltonian to an input wavefunction
  - `sparse::S`: Function that constructs the sparse matrix representation of the Hamiltonian
"""
struct LatticeHamiltonian{real_dim,lattice_dim,F,S}
    A::SMatrix{real_dim,lattice_dim,Float64}
    B::SMatrix{real_dim,lattice_dim,Float64}
    d::Int
    L::MVector{lattice_dim,Int}
    params::Dict{Symbol,ComplexF64}
    r::Vector{SVector{real_dim,Float64}}
    apply!::F
    sparse::S
end

"""
    lattice_hamiltonian(input)

Define a Hamiltonian using a convenient mini domain specific language (described below).
The user can provide an expression that defines the parameters for the potential and hopping terms (params, V, T, L).
The macro generates code that constructs a `LatticeHamiltonian` object and a function to apply the Hamiltonian.

# Domain Specific Language

`lattice_hamiltonian` uses a simple domain specific language to define the Hamiltonian.
It expects the following types of expressions, supplied in any order.

  - **REQUIRED** `L = [L1, L2, ...]`: A vector of integers, of length `lattice_dim` that define the size of the system.
    The length of this vector defines the dimensionality of the system.
    And each entry is the the number of unit cells in that direction.
  - **REQUIRED** `V = [V1, V2, ...]`: A vector of numbers of length `d`, that define the on-site potential
    for each orbital within a unit cell.
  - Expressions of the orm `(δ1, δ2, ...) -> T`, with `δi` integers and `T` a \\(d\\times d\\) matrix:
    The hopping matrix for to hop by `(δ1, δ2, ...)` unit cells.

# Examples

The following creates a Su-Schrieffer-Heeger (SSH) Hamiltonian on a 1D lattice,
with 10 unit cells.

    H_ssh = @lattice_hamiltonian begin
        L = [10]
        V = [0.0, 0.0]
        (0) -> [0 t1
                t1 0]
        (1) -> [0 t2
                0 0]
        (-1) -> [0 0
                t2 0]
        t1 = 1.0
        t2 = 2.0
    end
"""
macro lattice_hamiltonian(input)
    builder = parse_lattice_dsl(input)
    build(builder)
end

function sitenumber(r, H::LatticeHamiltonian)
    sitenum = r[H.dim]

    for j = 1:(H.dim-1)
        sitenum = r[H.dim-j] + sitenum * H.L[H.dim-j]
    end

    1 + sitenum
end
