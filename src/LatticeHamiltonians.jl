"""
Functions and structures for constructing and applying Hamiltonians on lattice systems.
"""
module LatticeHamiltonians

using StaticArrays, LinearAlgebra, SparseArrays
using MacroTools
import SparseArrays: sparse
import LinearAlgebra: mul!
import Base: *, size, length, eltype, adjoint

export LatticeHamiltonian, mul!, *, sparse, size, length, eltype, adjoint
export @lattice_hamiltonian

"""
    LiteralOrSymbolic

Possible types of parameters for the potential and hopping functions.
"""
LiteralOrSymbolic = Union{Symbol,Expr,ComplexF64}

"""
    SparseEntry{T}

A tuple of the form `(i, j, val)` that describes a non-zero element of a `Matrix{T}`.
"""
SparseEntry{T} = Tuple{Vector{Int64},Vector{Int64},Vector{T}}

include("build_lattice_operator.jl")

"""
    LatticeHamiltonian{real_dim,lattice_dim,F}

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
"""
struct LatticeHamiltonian{real_dim,lattice_dim,F}
    A::SMatrix{real_dim,lattice_dim,Float64}
    B::SMatrix{real_dim,lattice_dim,Float64}
    d::Int
    L::MVector{lattice_dim,Int}
    params::Dict{Symbol,ComplexF64}
    r::Vector{SVector{real_dim,Float64}}
    apply!::F
end

# should we go ahead and implement AbstractMatrix?
function size(H::LatticeHamiltonian)
    l = H.d * prod(H.L)
    return l, l
end

function length(H::LatticeHamiltonian)
    return (H.d * prod(H.L))^2
end

function eltype(::LatticeHamiltonian) #Hardcoded as ComplexF64 for the moment.
    return ComplexF64
end

function adjoint(H::LatticeHamiltonian) #Hardcoding that the matrix is Hermitian!!
    return H
end

"""
    bound_nonzero(vol, d, T)

Provide an upper bound on the number of non-zero elements in the Hamiltonian matrix,
based on the provided hoppings `T`, the number of orbitals `d`, and the volume of the system `vol`.
"""
function bound_nonzero(vol, d::Int, T)
    non_zero = d + sum(t -> length(t[2][1]), T)
    return non_zero * vol
end

function sitenumber(r, H::LatticeHamiltonian)
    dim = length(H.L)
    sitenum = r[dim]
    for j = 1:(dim-1)
        sitenum = r[dim-j] + sitenum * H.L[dim-j]
    end
    return 1 + sitenum
end

"""
    mul!(ψout::AbstractVector, H::LatticeHamiltonian, ψin::AbstractVector)

Efficiently perform the matrix multiplaction `H*ψin`, writing the result to `ψout`.
"""
function mul!(ψout::AbstractVector, H::LatticeHamiltonian, ψin::AbstractVector)
    H.apply!(ψout, ψin, H.d, length(H.L), H.L, H.params)
end

"""
    *(H::LatticeHamiltonian, ψ::AbstractVector)

Overloaded matrix multiplication operator for `LatticeHamiltonian`s
using the `mul!` function.
"""
function *(H::LatticeHamiltonian, ψ::AbstractVector)
    v = Array{eltype(ψ)}(undef, length(ψ))
    mul!(v, H, ψ)
    return v
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
    exprL = :()
    exprV = :()
    exprd = :()
    hops = Vector{Expr}()  # Store all hopping expressions in a vector
    params = Dict{Symbol,ComplexF64}() #Initialize dictionary

    for ex in input.args # loops over exprL/O/hops/params
        # skip any expressions that don't have an args list
        # as they're irrelevant to our DSL
        if !hasproperty(ex, :args)
            continue
        end
        if ex.args[1] == :L
            exprL = ex
        elseif ex.head == :(->)
            # seperate out the onsite hopping matrix
            # as it requires special handling
            if isonsite(ex)

              exprV = ex
            else
              push!(hops, ex)  # Add hopping expression to the vector

            end
        elseif ex.head == :(=)
            param_key = ex.args[1]
            param_val = eval(ex.args[2])
            params[param_key] = param_val #adds it to the dictionary
        end
    end

    if exprL.args[1] != :L
        error("Invalid input. Format should be :L = [nums].")

        end
    else
        throw(ArgumentError("Expected a matrix or vector"))
    end
    return (rows = rows, cols = cols, vals = vals)
end

"""
    foreach_element(f :: Function, expr::Expr)

Generic iteration over elements of a matrix or vector expression.
`f` should be a function of the form `f(ri, ci, elem)`
where `ri` and `ci` are the row and column indices of the element,
and `elem` is the value.
"""
function foreach_element(f :: Function, expr::Expr)
    if isexpr(expr, :vcat)
    for (ri, row) in enumerate(expr.args)
        for (ci, elem) in enumerate(row.args)
            f(ri, ci, elem)
        end
    end
  elseif isexpr(expr, :vect)
    for (i, elem) in enumerate(expr.args)
            f(i, i, elem)
    end
  else
    throw(ArgumentError("Expected a matrix or vector"))
  end
end

end
