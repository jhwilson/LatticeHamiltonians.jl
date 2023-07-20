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

    L = Vector(exprL.args[2].args)
    dim = length(L)

    if !(isa(L, Vector) && all(isinteger, L))
        error("Invalid input. Expected a vector of integers.")
    end

    L = MVector{length(L),Int}(L)
    if isempty(hops)
        error("Invalid input. Expected at least one hopping expression.")
    end


    # Hamiltonian matrix elements
    T = Dict{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
        parse_hopping(hop, params) for hop in hops
    )
    # compute the number of orbitals as the maximum index of the hopping matrix
    d = maximum([max(maximum(t.second[1]), maximum(t.second[2])) for t in T])

    # Extract the onsite potential and hopping
    V, onsite = extract_potential(exprV, dim, d, params)
    if onsite !== nothing
      T[onsite.first] = onsite.second
    end

    # Real space structure
    A = SMatrix{dim,dim,Float64}(I) #TODO
    B = SMatrix{dim,dim,Float64}(I * 2 * pi) #TODO
    r = fill(SVector{dim}(zeros(Float64, dim)), d) #TODO

    apply = eval(make_apply(params, V, T, dim))
    H = LatticeHamiltonian(A, B, d, L, params, r, apply)

    build_sparse(H, V, T)

    quote
        $H
    end
end

"""
    isonsite(expr::Expr)

Given an expression of the form `δ -> [t1 t2 ...; t3 t4 ...; ...]`,
checks if the hopping is on-site, i.e. if `δ` is zero.

On site values for δ are

    0 -> ...
    (0) -> ...
    (0, 0, ...) -> ...
"""
function isonsite(expr::Expr)
    lhs = expr.args[1]
    (lhs isa Number && iszero(lhs)) || (isexpr(lhs, :tuple) && all(iszero, lhs.args))
end

"""
    extract_potential(exprV, dim, d, params::Dict{Symbol,ComplexF64})

Given an onsite hopping seperate the potential from the site-local hopping matrix.
Allowed forms for the right hand side are the same as `parse_hopping`.

Besides the expression, also takes as input the lattice dimension `dim`,
the number of orbitals per site `d`, and the parameter dictionary `params`.

Return as tuple of the form `(V, hop)`.

The potential is returned as a vector of `LiteralOrSymbolic` values.
If there is no non-zero onsite hopping, `hop=nothing`, otherwise `hop` is a `Pair` of the form

    δ => (rows, cols, values)

```@example
extract_potential(:([Δ t; t -Δ], 1, 2, Dict{Symbol,ComplexF64}(:Δ => 1.0))
```
"""
function extract_potential(exprV::Expr, dim::Int, d::Int, params::Dict{Symbol,ComplexF64})
  # on site hoppings
  orows = Int[]
  ocols = Int[]
  ovalues = LiteralOrSymbolic[]


  # on site potential
  V = Vector{LiteralOrSymbolic}(undef, d)
  fill!(V, zero(ComplexF64))

  # implicitly set V to zero if not provided
  if exprV == :()
    return V, nothing
  end

  pair = parse_hopping(exprV, params)
  (rows, cols, values) = pair[2]


  # seperate the diagonal and off-diagonal elements
  for (r, c, v) in zip(rows, cols, values)
    if r == c
      V[r] = v
    else
      push!(orows, r)
      push!(ocols, c)
      push!(ovalues, v)
    end
  end

  hop = isempty(orows) ? nothing : zeros(Int, dim) => (orows, ocols, ovalues)

  V, hop
end

"""
    parse_hopping(hop::Expr, params::Dict{Symbol,ComplexF64})

Extract the key and value from a hopping expression of the form

    (δ1, δ2, ...) -> [t1 t2 ...; t3 t4 ...; ...]

and return a pair of the form

    key => (rows, cols, values)

where `key` is a vector of integers that describe the hopping vector,
and `rows`, `cols`, and `values` are vectors that describe the non-zero elements of the hopping matrix.

The hopping matrix itself may be a singleton, a matrix literal, or a tuple of vectors `(rows, cols, values)`.
"""
function parse_hopping(hop::Expr, params::Dict{Symbol,ComplexF64})
    Base.remove_linenums!(hop)

    lhs = hop.args[1]

    # extract non-zero elements from the hopping matrix
    rhs = hop.args[2].args[1]

    rows, cols, values = if isexpr(rhs, :vcat, :vect)
        nonzero_elements(rhs)
    elseif isexpr(rhs, :tuple)
        Tuple([item.args for item in rhs.args])
    else
        [1], [1], [rhs]
    end

    # replace parameter symbols in the hoppings with
    # with lookups in the the parameter dictionary
    values = LiteralOrSymbolic[symbols_to_lookups(v, params) for v in values]

    key = if lhs isa Integer
        [lhs]
    else
        Vector{Int}(hop.args[1].args)
    end

    key => (rows, cols, values)
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
                L = H.L
                dim = length(L)
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
    symbols_to_lookups(expr, params::Dict{Symbol,ComplexF64})

Attempts to convert terms to a canonical form:

  - All literal numbers are converted to `ComplexF64`
  - All symbols that are keys in `params` are converted to lookups, i.e. `t1` becomes `params[:t1]`.
  - Everything else is left as is.
"""
function symbols_to_lookups(expr, params::Dict{Symbol,ComplexF64})
    MacroTools.postwalk(function (x)
        if isexpr(x, Number)
            ComplexF64(x)
        elseif haskey(params, x)
            :(params[$(QuoteNode(x))])
        else
            x
        end
    end, expr)
end



"""
    nonzero_elements(matrix)

Takes an expression describing a matrix and returns a named tuple of vectors (rows, cols, vals)
where each vector respectively contains the row, column, and value of a nonzero element in the matrix.
All numbers are converted to `ComplexF64` numbers.

If passed a vector, returns the nonzero elements of the corresponding diagonal matrix.
"""
function nonzero_elements(matrix::Expr)
    rows = Int[]
    cols = Int[]
    vals = LiteralOrSymbolic[]
    foreach_element(matrix) do ri, ci, elem
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

function fnzi(matrix)
    rows = 0
    cols = 0
    i = Int[]
    j = Int[]
    k = Vector{Union{ComplexF64, Symbol, Expr}}(undef, 0)

    for (ri, row) in enumerate(matrix.args)
        cols = max(cols, length(row.args))
        for (ci, elem) in enumerate(row.args)
            if elem != 0
                push!(i, ri)
                push!(j, ci)
                if elem isa Number #7_6 added ifelse 
                    push!(k, ComplexF64(elem))
                end
            end
        end
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
