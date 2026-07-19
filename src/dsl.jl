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

parse_lattice_dsl(input) = parse_lattice_dsl(input, LatticeHamiltonians)

function parse_lattice_dsl(input, mod::Module)
    exprL = :()
    exprV = :()
    exprd = :()
    hops = Vector{Expr}()  # Store all hopping expressions in a vector
    params = Dict{Symbol,ComplexF64}() #Initialize dictionary
    fields = Dict{Symbol,Any}()

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
            name = ex.args[1]
            name isa Symbol || error("Invalid assignment name $name in lattice DSL.")
            value = Core.eval(mod, ex.args[2])
            (haskey(params, name) || haskey(fields, name)) && error(
                "Name $name is assigned twice in the lattice DSL; " *
                "assign each parameter or field once.",
            )
            if value isa Number
                params[name] = ComplexF64(value)
            else
                fields[name] = value
            end
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
    validate_field_names(keys(fields), lattice_dim)
    if isempty(hops)
        error("Invalid input. Expected at least one hopping expression.")
    end

    fieldnames = Set{Symbol}(keys(fields))

    # Hamiltonian matrix elements
    T = Dict{LatticeVector{lattice_dim},SparseEntry{LiteralOrSymbolic}}()
    for hop in hops
        key, entry = parse_hopping(hop, params, fieldnames)
        merge_hopping!(T, fold_hopping(key, L, lattice_dim), entry)
    end
    d = orbital_dim(T)

    # Extract the onsite potential and hopping
    V, onsite_hops = extract_potential(exprV, dim, d, params, fieldnames)
    if onsite_hops !== nothing
      merge_hopping!(T, onsite(lattice_dim), onsite_hops)
    end

    HamiltonianBuilder{lattice_dim,lattice_dim}(;
        params = params,
        fields = fields,
        L = L,
        V = V,
        T = T,
        d = d,
    )
end

function validate_field_names(fieldnames, dim)
    reserved = Set{Symbol}(
        [
            :L,
            :d,
            :params,
            :fields,
            :ψin,
            :ψout,
            :i,
            :idx,
            :i_in,
            :i_out,
            :ivals,
            :jvals,
            :hvals,
            :im,
            :nz,
            :matrix_size,
            # module prefixes of qualified calls in generated code
            :Base,
            :SparseArrays,
            [site_loop_var(j) for j = 1:dim]...,
            [dim_span_var(j) for j = 1:dim]...,
        ],
    )
    prefixes = ("_h", "_acc", "_lo", "_hi", "_hoff", "_iin", "_m")
    for name in fieldnames
        if name in reserved || any(prefix -> startswith(String(name), prefix), prefixes)
            error("Field name $name is reserved by generated lattice Hamiltonian code.")
        end
    end
end

"""
    fold_hopping(key, L, lattice_dim)

Reduce each component of a hopping displacement into `-(L-1):(L-1)` for its
periodic lattice direction. In-range components (including negative ones) are
untouched; longer displacements are physically equivalent to their remainder,
so they are folded rather than rejected, with an informational message since
the wrap-around is usually unintended.
"""
function fold_hopping(key, L, lattice_dim)
    folded = rem.(key, L)
    if folded != key
        @info "Hopping displacement $(Tuple(key)) reaches around the periodic " *
              "lattice of extent L = $(Tuple(L)); treating it as the " *
              "equivalent displacement $(Tuple(folded))."
    end
    LatticeVector{lattice_dim}(folded)
end

"""
    merge_hopping!(T, key, (rows, cols, values))

Insert a hopping entry, concatenating with any entry already stored at `key`
(as happens when two displacements fold to the same lattice vector). Repeated
(row, col) pairs are summed by both the sparse constructor and the fused
accumulators.
"""
function merge_hopping!(T, key, entry)
    if haskey(T, key)
        (rows, cols, values) = T[key]
        append!(rows, entry[1])
        append!(cols, entry[2])
        append!(values, entry[3])
    else
        T[key] = entry
    end
    T
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
If there is no non-zero onsite hopping, `hop=nothing`, otherwise `hop` is a `Tuple` of the form

    (rows, cols, values)

```@example
extract_potential(:([Δ t; t -Δ], 1, 2, Dict{Symbol,ComplexF64}(:Δ => 1.0))
```
"""
function extract_potential(
    exprV::Expr,
    dim::Int,
    d::Int,
    params::Dict{Symbol,ComplexF64},
    fieldnames::Set{Symbol} = Set{Symbol}(),
)
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

  pair = parse_hopping(exprV, params, fieldnames)
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
function parse_hopping(
    hop::Expr,
    params::Dict{Symbol,ComplexF64},
    fieldnames::Set{Symbol} = Set{Symbol}(),
)
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
    for value in values
        value isa Symbol && value in fieldnames && error(
            "Field $value cannot be used as a bare matrix element; " *
            "index it, for example $value[n1].",
        )
    end
    values = LiteralOrSymbolic[
        symbols_to_lookups(value, params, fieldnames) for value in values
    ]

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
symbols_to_lookups(expr, params::Dict{Symbol,ComplexF64}) =
    symbols_to_lookups(expr, params, Set{Symbol}())

function symbols_to_lookups(
    expr,
    params::Dict{Symbol,ComplexF64},
    fieldnames::Set{Symbol},
)
    if expr isa Number
        return ComplexF64(expr)
    elseif expr isa Symbol
        return haskey(params, expr) ? :(params[$(QuoteNode(expr))]) : expr
    elseif !(expr isa Expr)
        return expr
    elseif expr.head == :ref
        array = symbols_to_lookups(expr.args[1], params, fieldnames)
        indices = [rewrite_params(i, params; index = true) for i in expr.args[2:end]]
        return Expr(:ref, array, indices...)
    elseif expr.head == :call && expr.args[1] isa Symbol && expr.args[1] in fieldnames
        arguments = [rewrite_params(a, params; index = false) for a in expr.args[2:end]]
        return Expr(:call, expr.args[1], arguments...)
    end

    converted_args = [symbols_to_lookups(arg, params, fieldnames) for arg in expr.args]
    Expr(expr.head, converted_args...)
end

"""
    rewrite_params(expr, params; index)

Rewrite parameter symbols inside a field index or field-call argument as
`params` lookups, leaving literal numbers untouched (unlike the matrix-element
path, which folds them to `ComplexF64`). With `index = true` the lookup is
converted with `Int(real(...))`, since parameters are stored as `ComplexF64`
but array indices must be integers; non-integer values throw `InexactError`
at application. Indices of any nested array reference are always rewritten in
index mode.
"""
function rewrite_params(expr, params::Dict{Symbol,ComplexF64}; index::Bool)
    if expr isa Symbol
        haskey(params, expr) || return expr
        lookup = :(params[$(QuoteNode(expr))])
        return index ? :(Base.Int(Base.real($lookup))) : lookup
    elseif expr isa Expr && expr.head == :ref
        array = rewrite_params(expr.args[1], params; index = false)
        indices = [rewrite_params(i, params; index = true) for i in expr.args[2:end]]
        return Expr(:ref, array, indices...)
    elseif expr isa Expr
        return Expr(
            expr.head,
            (rewrite_params(arg, params; index = index) for arg in expr.args)...,
        )
    end
    expr
end

"""
    bound_nonzero(vol, d, T)

Provide an upper bound on the number of non-zero elements in the Hamiltonian matrix,
based on the provided hoppings `T`, the number of orbitals `d`, and the volume of the system `vol`.
"""
function bound_nonzero(vol, d::Int, T)
    non_zero = d + (isempty(T) ? 0 : sum(t -> length(t[2][1]), T))
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
        # simplify literal expressions
        if isliteral(elem)
            elem = ComplexF64(elem isa Number ? elem : eval(elem))
            # skip zero elements
            if iszero(elem)
                return
            end
        end
        push!(rows, ri)
        push!(cols, ci)
        push!(vals, elem)
    end

   (rows = rows, cols = cols, vals = vals)
end

"""
    isliteral(expr)

Check if an expression is a literal number or a literal complex number, using a simple heuristic.
In general, any expression that is a number or a simple arithmetic operation on numbers is considered literal.

!!! note
    Currently only supports numbers, `im`, and the arithmetic operations `+`, `-`, `*`, `/`, and `^`. 
    Function calls like `sqrt(2)` are not considered literal.
"""
function isliteral(expr)
    MacroTools.postwalk(expr) do x
        if isexpr(x, Number) || x in [:im, :*, :+, :/, :-, :^]
            true
        elseif x isa Expr
            all(x.args)
        else
            false
        end
    end
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
