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

include("build_lattice_operator.jl")

"""
    LatticeHamiltonian{real_dim,lattice_dim,F}

Stores information about the Hamiltonian of the system, so that matrix-vector multiplication
can be performed efficiently.

This object should not be initialized by the user.
Instead, for construction of the Hamiltonian, see the `@lattice_hamiltonian` macro.

# Fields

  - `A::SMatrix{real_dim,lattice_dim,Float64}`: Real-space basis
  - `B::SMatrix{real_dim,lattice_dim,Float64}`: Real-space basis
  - `d::Int`: Number of orbitals per site
  - `L::MVector{lattice_dim,Int}`: System size
  - `params::Dict{Symbol,ComplexF64}`: Parameters for potential and hopping functions
  - `r::Vector{SVector{real_dim,Float64}}`: Real-space coordinates of orbitals within a unit cell
  - `apply!::F`: Function that applies the Hamiltonian to an input wavefunction
"""
struct LatticeHamiltonian{real_dim,lattice_dim,F}
    # are these A and B sublattices for a bipartite lattice?
    A::SMatrix{real_dim,lattice_dim,Float64}
    B::SMatrix{real_dim,lattice_dim,Float64}
    d::Int
    # why is this mutable?
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
Define a Hamiltonian using a convenient mini domain specific language (described below).
The user can provide an expression that defines the parameters for the potential and hopping terms (params, V, T, L).
The macro generates code that constructs a `LatticeHamiltonian` object and a function to apply the Hamiltonian.
"""
macro lattice_hamiltonian(input)
    exprL = :()
    exprV = :()
    hops = Vector()  # Store all hopping expressions in a vector
    params = Dict{Symbol,ComplexF64}() #Initialize dictionary

    for ex in input.args # loops over exprL/O/hops/params
        try
            if ex.args[1] == :L
                exprL = ex
            elseif ex.args[1] == :V
                exprV = ex
            elseif ex.head == :(->)
                push!(hops, ex)  # Add hopping expression to the vector
            elseif ex.head == :(=)
                param_key = ex.args[1]
                param_val = eval(ex.args[2])
                params[param_key] = param_val #adds it to the dictionary
            end
        catch e
        end
    end

    L = eval(exprL.args[2])
    dim = length(L)
    if exprL.args[1] != :L
        error("Invalid input. Format should be :L = [nums].")
    end
    if !(isa(L, Vector) && all(isinteger, L))
        error("Invalid input. Expected a vector of integers.")
    end
    if ~(typeof(exprV.args[2].args) <: Vector)
        error("Invalid input. Expected a vector for V.")
    end
    L = MVector{length(L)}(L)
    if isempty(hops)
        error("Invalid input. Expected at least one hopping expression.")
    end

    d = length(exprV.args[2].args)
    V = Vector{LiteralOrSymbolic}(undef, d)
    for i in eachindex(exprV.args[2].args)
        try
            V[i] = ComplexF64(eval(exprV.args[2].args[i]))
        catch e
            ex = exprV.args[2].args[i]
            for key in keys(params)
                ex = MacroTools.postwalk(
                    x -> x == key ? :(params[$(QuoteNode(key))]) : x,
                    ex,
                )
            end
            V[i] = ex
        end
    end
    T = Dict{Vector{Int64},Tuple{Vector{Int64},Vector{Int64},Vector{LiteralOrSymbolic}}}()
    for hop in hops
        Base.remove_linenums!(hop)

        # extract non-zero elements from the hopping matrix
        m = hop.args[2].args[1]
        rows, cols, values = nonzero_elements(m)

        # replace parameter symbols in the hoppings with their values
        map!(
            function (value)
                for key in keys(params)
                    value = MacroTools.postwalk(
                        x -> x == key ? :(params[$(QuoteNode(key))]) : x,
                        value,
                    )
                end
                value
            end,
            values,
            values,
        )

        args1 = vec(eval(hop.args[1]) |> collect)
        T[args1] = (rows, cols, values)
    end
    println(T)

    A = SMatrix{dim,dim,Float64}(I)
    B = SMatrix{dim,dim,Float64}(I * 2 * pi)
    r = fill(SVector{dim}(zeros(Float64, dim)), d)
    apply = eval(make_apply(params, V, T, dim))
    H = LatticeHamiltonian(A, B, d, L, params, r, apply)

    build_sparse(H, V, T)

    quote
        $H
    end
end

function build_sparse(H::LatticeHamiltonian, V, T)
    nz = bound_nonzero(prod(H.L), H.d, T)
    dim = length(H.L)
    eval(
        quote
            function sparse(H::$(typeof(H)))
                ivals = Array{Int64}(undef, $nz)
                jvals = Array{Int64}(undef, $nz)
                hvals = Array{ComplexF64}(undef, $nz)
                idx = 1
                params = H.params
                d = H.d
                N = H.L
                dim = length(N)
                $(ham_expr(V, T, dim; sparse = true))
                idx -= 1
                return dropzeros!(
                    sparse(ivals[1:idx], jvals[1:idx], hvals[1:idx], size(H)...),
                )
            end
        end,
    )
end

"""
    nonzero_elements(matrix)

Takes an expression describing a matrix and returns a named tuple of vectors (rows, cols, vals)
where each vector respectively contains the row, column, and value of a nonzero element in the matrix.
All numbers are converted to `ComplexF64` numbers.
"""
function nonzero_elements(matrix::Expr)
    rows = Int[]
    cols = Int[]
    vals = LiteralOrSymbolic[]

    for (ri, row) in enumerate(matrix.args)
        for (ci, elem) in enumerate(row.args)
            if !(elem isa Number && iszero(elem))
                push!(rows, ri)
                push!(cols, ci)
                if elem isa Number
                    push!(vals, ComplexF64(elem))
                else
                    push!(vals, elem)
                end
            end
        end
    end
    return (rows = rows, cols = cols, vals = vals)
end

end
