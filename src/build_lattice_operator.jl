# VARIABLE NAMES
# ----
"""
    site_loop_var(j)

Generate a symbol for the loop variable in the generated code.
`j` indexes the lattice dimension.
For example, in a 2D lattice, we might generate `n1`, `n2` for the loop variables.

See also: `site_vector_var`
"""
site_loop_var(j) = Symbol("n$j")

"""
    site_vector_var(dim)

Generate a vector of symbols for the lattice site in the generated code.
`dim` is the total lattice dimension.

For example, for a 2D lattice,

```jldoctest
julia> using LatticeHamiltonians: site_var_vector

julia> site_var_vector(2)
2-element Vector{Symbol}:
 :n1
 :n2
```
"""
site_var_vector(dim) = [site_loop_var(j) for j = 1:dim]

"""
    dim_span_var(j)

Generate a symbol for the span of the lattice dimension in the generated code.
This describes how much the linear index changes when traversing all sites in this dimension.
For example, in a 2D lattice, we might generate `Lprod1`, `Lprod2` for the span of the lattice dimensions.
Suppose that there are `d=3` orbitals on-site, and the lattice dimensions are L1=2 and L2=3.
Then the linear index changes by `d * L1` when traversing all sites in the first dimension,
and by `d * L1 * L2` when traversing all sites in the second dimension, so the spans are `6` and `18` respectively.
"""
dim_span_var(j) = Symbol("Lprod$j")
# ----

"""
    matrix_element(h::Function; site)
    matrix_element(h::ComplexF64; site)
    matrix_element(h; site)

Generate an expression for the orbital matrix element `h` at a given lattice site.
If `h` is a function, it is evaluated at the given site, i.e., `h(n1, n2, n3...)`.
If `h` is a number, it is returned as a constant.
If `h` is a symbol, it is looked up from the `params` dictionary.
"""
matrix_element(h::Function; site) = :($(h$(site...)))
matrix_element(h::ComplexF64; site) = h == zero(ComplexF64) ? :() : h
matrix_element(h; site) = h

"""
    make_diag_expr(Vs; s = :n)

Constructs the diagonal part of the Hamiltonian matrix
in an efficient way by using metaprogramming features of Julia.
It takes the on-site potentials `Vs` and generates a corresponding `Expr` block,
which, when evaluated, would perform the operations of the diagonal part of the Hamiltonian.

If `sparse` is set to `true`, the function generates an `Expr` block that
constructs the sparse matrix representation of the diagonal part of the Hamiltonian.
"""
function make_diag_expr(Vs, dim; sparse = false)
    is = eachindex(Vs)
    expr_array = if sparse
        @. sparse_diag_expr(Vs, is)
    else
        @. diag_expr(Vs, is; site = site_var_vector(dim))
    end

    push!(expr_array, :(i_out += d))
    Expr(:block, expr_array...)
end

sparse_diag_expr(V, i::Int) = sparse_hop_expr(V, :(i_out + $i), :(i_out + $i))

function diag_expr(V, i::Int; site)
    m_el = matrix_element(V; site = site)
    if m_el == :()
        :(ψout[i_out+$i] = zero(ComplexF64))
    else
        :(ψout[i_out+$i] = $m_el * ψin[i_out+$i])
    end
end

"""
    make_hop_expr(is, js, hs, dim; s = :n, sparse=false)

Constructs the off-diagonal part of the Hamiltonian matrix
by generating an `Expr` block, which performs the operations of the off-diagonal part of the Hamiltonian.

If `sparse` is set to `true`, the function generates an `Expr` block that
constructs the sparse matrix representation of the off-diagonal part of the Hamiltonian.
"""
function make_hop_expr(is, js, hs, dim; sparse = false)
    expr_array = if sparse
        @. sparse_hop_expr(hs, is, js)
    else
        @. hop_expr(hs, is, js; site = site_var_vector(dim))
    end

    push!(expr_array, :(i_out += d; i_in += d))
    Expr(:block, expr_array...)
end

sparse_hop_expr(V, i::Int, j::Int) = sparse_hop_expr(V, :(i_in + $i), :(i_out + $j))
function sparse_hop_expr(V, in_idx::Expr, out_idx::Expr)
    quote
        ivals[idx] = $in_idx
        jvals[idx] = $out_idx
        hvals[idx] = $V
        idx += 1
    end
end

function hop_expr(V, i::Int, j::Int; site)
    mel = matrix_element(V; site = site)
    if mel == :()
        mel
    else
        :(ψout[i_out+$j] += $mel * ψin[i_in+$i])
    end
end

"""
    site_expr(hops[, j])

Recursively generate an expression for calculating the site
`site_expr` is overloaded for different numbers of arguments.
"""
site_expr(hops) = site_expr(hops, 1)

function site_expr(hops, j)
    if j == length(hops)
        return hops[j] ≥ 0 ? :($(hops[j])) : :(L[$j] - $(-hops[j]))
    end
    if hops[j] ≥ 0
        :($(hops[j]) + $(site_expr(hops, j + 1)) * L[$j])
    else
        :(L[$j] * (1 + $(site_expr(hops, j + 1))) - $(-hops[j]))
    end
end

"""
    loop_sites(dim, ex)

Recursively generate an `Expr` block which loops over lattice sites
and inserts the expressions `ex` at each site.
"""
function loop_sites(dim, ex)
    quote
        i_out = 0
        $(loop_site_pairs(zeros(Int, dim), ex, dim))
    end
end

"""
    loop_site_pairs(hops, ex)

Recursively generate an `Expr` block which loops over all pairs of lattice sites
separated by the lattice vector `hops` and insert the expression `ex` for each pair.

```jldoctest
julia> using LatticeHamiltonians: loop_periodic

julia> using MacroTools: prettify

julia> prettify(loop_periodic([1 0 1], :(expr)))

quote
    i_out = 0
    i_in = d * (1 + (0 + 1 * L[2]) * L[1])
    for n3 = 1:L[3] - 1
        for n2 = 1:L[2]
            for n1 = 1:L[1] - 1
                expr
            end
            i_in -= Lprod1
            for n1 = L[1] - 0:L[1]
                expr
            end
            i_in += Lprod1
        end
    end
    i_in -= Lprod3
    for n3 = L[3] - 0:L[3]
        for n2 = 1:L[2]
            for n1 = 1:L[1] - 1
                expr
            end
            i_in -= Lprod1
            for n1 = L[1] - 0:L[1]
                expr
            end
            i_in += Lprod1
        end
    end
    i_in += Lprod3
end
```
"""
function loop_site_pairs(hops, ex; open = falses(length(hops)))
    quote
        # initialize the indices of the output and input vectors 
        i_out = 0
        i_in = d * $(site_expr(hops))
        $(loop_site_pairs(hops, ex, length(hops); open = open))
    end
end

"""
    loop_site_pairs(hops, ex, axis)

The main method of `loop_site_pairs` works by recursively calling in to this method.
Each call to this method generates a block of for loops for a single lattice dimension,
and then calls the next dimension recursively, decreasing `axis` by 1.
When we have reached the last dimension, we insert the expression `ex` at each site.
"""
function loop_site_pairs(hops, ex, axis; open = falses(axis))
    # axis == 1 is the slowest changing lattice dimension
    # axis == 0 corresponds orbital hoppings
    # which should already be handled in the supplied expression
    # so we simply return it
    if axis == 0
        return ex
    end

    # the distance to hop in this dimension
    hop_range::Int = hops[axis]

    loop_var = site_loop_var(axis)
    if iszero(hop_range)
        # this is an on-site hopping in this dimension
        # just loop over all sites
        return quote
            for $loop_var = 1:L[$axis]
                $(loop_site_pairs(hops, ex, axis - 1; open = open))
            end
        end
    end

    index_before_wrap = hop_range > 0 ? :(L[$axis] - $hop_range) : -hop_range
    index_after_wrap = hop_range > 0 ? :(L[$axis] - $(hop_range - 1)) : (-hop_range + 1)
    dim_span = dim_span_var(axis)

    # per-cell advance of i_out/i_in along this axis = span of all inner dimensions
    stride = axis == 1 ? :d : dim_span_var(axis - 1)

    if open[axis]
        # Open boundary: the hop crosses the edge for exactly the cells the periodic
        # code sends through the wrap loop, so we drop that loop. Which loop is the
        # wrap depends on the sign of the hop: for hop_range > 0 the wrap is the
        # trailing run, for hop_range < 0 it is the leading run. In both cases we
        # replicate the dropped loop's i_out/i_in advance so the NET advance across
        # this axis stays `dim_span` (keeps multi-dimensional bookkeeping aligned).
        if hop_range > 0
            quote
                for $loop_var = 1:$index_before_wrap
                    $(loop_site_pairs(hops, ex, axis - 1; open = open))
                end
                i_out += $hop_range * $stride
                i_in += $hop_range * $stride
            end
        else
            quote
                i_out += $(-hop_range) * $stride
                i_in += $(-hop_range) * $stride
                i_in -= $dim_span
                for $loop_var = $index_after_wrap:L[$axis]
                    $(loop_site_pairs(hops, ex, axis - 1; open = open))
                end
                i_in += $dim_span
            end
        end
    else
        # periodic boundary: sites past the edge wrap to the other end of the axis.
        quote
            for $loop_var = 1:$index_before_wrap
                $(loop_site_pairs(hops, ex, axis - 1; open = open))
            end
            i_in -= $dim_span
            for $loop_var = $index_after_wrap:L[$axis]
                $(loop_site_pairs(hops, ex, axis - 1; open = open))
            end
            i_in += $dim_span
        end
    end
end

"""
    ham_expr(V, T, dim; sparse=false)

Combine the expressions for the diagonal and hopping terms to generate
a block of expressions that applies the full Hamiltonian.

The behavior of the function can be controlled by the `sparse` keyword argument:

  - when sparse is `false` (default), the function explicitly performs a matrix-vector multiplication
    on a vector ψin (which is assumed to be defined in the enclosing scope) and stores the results in ψout.

  - when sparse is `true`, the function generates a sparse matrix representation of the Hamiltonian
    which can be used to instantiate a sparse matrix.

When implementing matrix-vector multiplication directly, the incoming and outgoing vectors are flattened into 1D arrays.
This effectively corresponds to a Kroneker product of the lattice dimensions.
Because Julia uses column-major ordering, the indices of the multi-dimensional array are ordered fastest to slowest changing: [o, i_1, i_2, ..., i_n] where o is the oribital index and i_1, i_2, ..., i_n are the lattice indices in dimensions 1, 2, ..., n.
"""
function ham_expr(V, T, dim; sparse = false, open = falses(dim))
    # since we are flattening the multi-dimensional arrays into 1D arrays,
    # we need to keep track of the stride of the lattice dimensions
    # for each lattice dimension

    # the orbital index changes the fastest
    # then the lattice indices
    # here we track the span of each lattice dimension
    # that is how much the linear index when traverse all sites in this dimensions
    # this is used to implement periodic boundary conditions
    # where we need to go back to the start of the dimension when we reach the end
    expr_spans = Expr(
        :block,
        [
            [:($(dim_span_var(1)) = d * L[1])]
            [:($(dim_span_var(i)) = $(dim_span_var(i - 1)) * L[$i]) for i = 2:dim]
        ]...,
    )
    # evaluate the diagonal elements of the Hamiltonian
    # NOTE: it is very important that the diagonal code is evaluated first
    # as it is responsible for zeroing out the output vector
    expr_V = make_diag_expr(V, dim; sparse = sparse)
    expr_diag = loop_sites(dim, expr_V)

    # evaluate the off-diagonal elements
    expr_hops = Vector{Expr}(undef, length(T))
    idx = 1
    for (hops, (is, js, hs)) in T
        expr_T = make_hop_expr(is, js, hs, dim; sparse = sparse)
        expr_hops[idx] = loop_site_pairs(hops, expr_T; open = open)
        idx += 1
    end

    Expr(:block, expr_spans, expr_diag, expr_hops...)
end

"""
    make_apply(V, T, dim)

Generate an `Expr` that defines a function to apply the Hamiltonian
given parameters for the potential and hopping terms (V and T).
"""
function make_apply(V, T, dim; open = falses(dim))
    quote
        function (
            ψout::AbstractArray,
            ψin::AbstractArray,
            d::Int,
            L::MVector{$dim,Int64},
            params::Dict{Symbol,ComplexF64},
        )
            # The arguments of the function are used implicitly
            # in the generated expressions
            #
            # see e.g., diag_expr, hop_expr
            $(ham_expr(V, T, dim; open = open))
        end
    end
end

function make_apply(
    builder::HamiltonianBuilder{real_dim,lattice_dim},
) where {real_dim,lattice_dim}
    make_apply(builder.V, builder.T, lattice_dim; open = builder.open)
end

"""
    make_sparse(V, T, dim)

Generate an `Expr` that defines a function to construct the sparse matrix representation
of the Hamiltonian given parameters for the potential and hopping terms (V and T).
"""
function make_sparse(V, T, dim; open = falses(dim))
    quote
        function (d::Int, L::MVector{$dim,Int64}, params::Dict{Symbol,ComplexF64})
            nz = bound_nonzero(prod(L), d, $T)
            matrix_size = d * prod(L)

            ivals = Array{Int64}(undef, nz)
            jvals = Array{Int64}(undef, nz)
            hvals = Array{ComplexF64}(undef, nz)

            idx = 1
            $(ham_expr(V, T, dim; sparse = true, open = open))
            idx -= 1

            dropzeros!(
                SparseArrays.sparse(
                    ivals[1:idx],
                    jvals[1:idx],
                    hvals[1:idx],
                    matrix_size,
                    matrix_size,
                ),
            )
        end
    end
end

function make_sparse(
    builder::HamiltonianBuilder{real_dim,lattice_dim},
) where {real_dim,lattice_dim}
    make_sparse(builder.V, builder.T, lattice_dim; open = builder.open)
end