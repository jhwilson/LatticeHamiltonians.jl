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
  - `fields::Dict{Symbol,Any}`: Site-dependent matrix-element data and functions
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
    fields::Dict{Symbol,Any}
    r::Vector{SVector{real_dim,Float64}}
    apply!::F
    sparse::S
end

"""
    lattice_hamiltonian(input)

Define a Hamiltonian using a convenient mini domain specific language (described below).
The macro generates code that constructs a `LatticeHamiltonian` object and a function to apply the Hamiltonian.

# Domain Specific Language

`lattice_hamiltonian` uses a simple domain specific language to define the Hamiltonian.
It expects the following types of expressions, supplied in any order.

  - **REQUIRED** `L = [L1, L2, ...]`: A vector of integers, of length `lattice_dim` that define the size of the system.
    The length of this vector defines the dimensionality of the system.
    And each entry is the the number of unit cells in that direction.
  - **REQUIRED** Hopping expressions of the form `(δ1, δ2, ...) -> M`, with `δi` integers and `M` a \\(d\\times d\\) matrix
    (a matrix literal, a scalar, or a `(rows, cols, vals)` tuple):
    The hopping matrix for a hop by `(δ1, δ2, ...)` unit cells.
    The onsite expression `(0, 0, ...) -> M` doubles as the potential:
    the diagonal of `M` is the per-orbital on-site potential, and off-diagonal
    entries become an on-site orbital hopping.
  - Assignments `name = value`, evaluated in the calling module at macro expansion time.
    A `Number` value declares an adjustable scalar parameter, stored in `H.params`
    (mutating `H.params[:t]` changes the Hamiltonian without recompiling).
    Any other value (an array, a function, ...) declares a *field*, stored in `H.fields`;
    matrix elements may reference fields using the site coordinates `n1, n2, ...`
    of the destination unit cell — e.g. `W[n1]` or `f(n1, n2)` — to build
    site-dependent (e.g. disordered) Hamiltonians. See the tutorial for details
    and conventions.

# Examples

The following creates a Su-Schrieffer-Heeger (SSH) Hamiltonian on a 1D lattice,
with 10 unit cells.

    H_ssh = @lattice_hamiltonian begin
        L = [10]
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
    builder = parse_lattice_dsl(input, __module__)
    build(builder)
end

function sitenumber(r, H::LatticeHamiltonian)
    sitenum = r[H.dim]

    for j = 1:(H.dim-1)
        sitenum = r[H.dim-j] + sitenum * H.L[H.dim-j]
    end

    1 + sitenum
end
