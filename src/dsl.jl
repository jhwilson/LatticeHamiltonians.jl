import MacroTools
import MacroTools: isexpr

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

"""
    LatticeVector{lattice_dim}

A vector of integers that describes a lattice vector in a `lattice_dim` dimensional lattice.
"""
LatticeVector{lattice_dim} = SVector{lattice_dim,Int64}
onsite(lattice_dim) = LatticeVector{lattice_dim}(zeros(Int, lattice_dim))

function parse_lattice_dsl(input)
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

    lattice_dim = length(L)
    L = MVector{lattice_dim,Int64}(L)
    if isempty(hops)
        error("Invalid input. Expected at least one hopping expression.")
    end


    # Hamiltonian matrix elements
    T = Dict{LatticeVector{lattice_dim},SparseEntry{LiteralOrSymbolic}}(
        parse_hopping(hop, params) for hop in hops
    )
    d = orbital_dim(T)

    # Extract the onsite potential and hopping
    V, onsite_hops = extract_potential(exprV, dim, d, params)
    if onsite_hops !== nothing
      T[onsite(lattice_dim)] = onsite_hops
    end

    (params=params, L=L, V=V, T=T, dim=dim, d=d)
end

"""
    orbital_dim(T)

compute the number of orbitals as the maximum index of the orbital hopping matrices
"""
function orbital_dim(T)
    map(values(T)) do (rows, cols, _)
        max(maximum(rows), maximum(cols))
    end |> maximum
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
  (rows, cols, values) = pair.second

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

  hop = isempty(orows) ? nothing : (orows, ocols, ovalues)

  V, hop
end

"""
    parse_hopping(hop::Expr, params::Dict{Symbol,ComplexF64})

Extract the key and value from a hopping expression of the form

    (δ1, δ2, ...) -> [t1 t2 ...; t3 t4 ...; ...]

and return a pair of the form

    key => (rows, cols, values)

where `key` is a vector of integers that describe the hopping vector,
and `rows`, `cols`, and `values` are vectors that describe the non-zero elements of the orbital hopping matrix.

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
    bound_nonzero(vol, d, T)

Provide an upper bound on the number of non-zero elements in the Hamiltonian matrix,
based on the provided hoppings `T`, the number of orbitals `d`, and the volume of the system `vol`.
"""
function bound_nonzero(vol, d::Int, T)
    non_zero = d + sum(t -> length(t[2][1]), T)
    non_zero * vol
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

   (rows = rows, cols = cols, vals = vals)
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
