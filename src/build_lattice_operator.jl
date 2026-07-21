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
    hoist_matrix_elements(V, T, dim)

Replace site-independent symbolic matrix elements in `V` and `T` with local binding
symbols. Identical values share a binding, while literals and values that reference
generated loop or workspace variables remain unchanged.

Returns `(bindings, V2, T2)`, where `bindings` is an expression block that evaluates
each hoisted value exactly once as a `ComplexF64`.
"""
function hoist_matrix_elements(V, T, dim)
    disallowed = Set{Symbol}(
        [
            :L,
            :d,
            :i_in,
            :i_out,
            :idx,
            :ψin,
            :ψout,
            :ivals,
            :jvals,
            :hvals,
            [site_loop_var(j) for j = 1:dim]...,
            [dim_span_var(j) for j = 1:dim]...,
        ],
    )
    contains_disallowed(value) = if value isa Symbol
        value in disallowed
    elseif value isa Expr
        any(contains_disallowed, value.args)
    else
        false
    end

    source_symbols = Set{Symbol}()
    collect_symbols!(value) = if value isa Symbol
        push!(source_symbols, value)
    elseif value isa Expr
        foreach(collect_symbols!, value.args)
    end
    foreach(collect_symbols!, V)
    for (_, (_, _, values)) in T
        foreach(collect_symbols!, values)
    end

    bindings = Expr[]
    binding_symbols = Dict{LiteralOrSymbolic,Symbol}()
    function hoist(value::LiteralOrSymbolic)
        if value isa ComplexF64 || contains_disallowed(value)
            return value
        end

        get!(binding_symbols, value) do
            binding_index = length(bindings) + 1
            binding = Symbol("_h$binding_index")
            while binding in source_symbols
                binding_index += 1
                binding = Symbol("_h$binding_index")
            end
            push!(source_symbols, binding)
            push!(bindings, :($binding::ComplexF64 = $value))
            binding
        end
    end

    V2 = LiteralOrSymbolic[hoist(value) for value in V]
    T2 = empty(T)
    for (hops, (rows, cols, values)) in T
        T2[hops] = (rows, cols, LiteralOrSymbolic[hoist(value) for value in values])
    end

    Expr(:block, bindings...), V2, T2
end

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
    expr_diag = if !sparse && all(V -> V isa ComplexF64 && iszero(V), V)
        :(fill!(ψout, zero(ComplexF64)))
    else
        loop_sites(dim, expr_V)
    end

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
    hop_linear_offset(hops)

Generate the constant linear offset for an unwrapped hop in the fused interior
region. The first lattice coordinate has stride `d`; subsequent coordinates use
the previously generated `Lprod` spans.
"""
function hop_linear_offset(hops)
    terms = Any[]
    for (axis, hop) in enumerate(hops)
        iszero(hop) && continue
        stride = axis == 1 ? :d : dim_span_var(axis - 1)
        push!(terms, :($hop * $stride))
    end
    isempty(terms) ? 0 : Expr(:call, :+, terms...)
end

"""
    site_linear_index(coordinates)

Generate the zero-based orbital-vector index of a lattice site. This is used once
per dimension-1 segment in the interior and once per site in the boundary slabs;
it never introduces integer division or remainder operations.
"""
function site_linear_index(coordinates)
    terms = Any[:(d * ($(coordinates[1]) - 1))]
    for axis = 2:length(coordinates)
        push!(terms, :($(dim_span_var(axis - 1)) * ($(coordinates[axis]) - 1)))
    end
    Expr(:call, :+, terms...)
end

"""
    interior_margins(T, dim)

Compute the code-generation-time lower and upper margins of the common no-wrap
interior. The resulting box is `1 + lower[j]:L[j] - upper[j]` in dimension `j`;
its complement is emitted as ordered, disjoint boundary slabs.
"""
function interior_margins(T, dim)
    lower = zeros(Int, dim)
    upper = zeros(Int, dim)
    for (hops, _) in T, axis = 1:dim
        lower[axis] = max(lower[axis], -hops[axis])
        upper[axis] = max(upper[axis], hops[axis])
    end
    lower, upper
end

"""
    loop_region(ranges, ex[, axis])

Generate a column-major lattice loop nest for the supplied coordinate `ranges`.
A `nothing` range omits that dimension, which lets the fused interior provide its
own incrementally indexed innermost loop while reusing this region generator.
"""
function loop_region(ranges, ex, axis = length(ranges))
    axis == 0 && return ex
    inner = loop_region(ranges, ex, axis - 1)
    ranges[axis] === nothing && return inner
    loop_var = site_loop_var(axis)
    quote
        for $loop_var in $(ranges[axis])
            $inner
        end
    end
end

"""
    fused_site_expr(V, T, dim; boundary, real_values)

Generate one fused site update. Each output orbital is accumulated in a register:
the diagonal contribution comes first, followed by hopping contributions in `T`
iteration order and entry-vector order. Interior sites use precomputed hop offsets;
boundary sites first construct an explicitly wrapped input index for every hop.
"""
function fused_site_expr(V, T, dim; boundary, real_values = false)
    site = site_var_vector(dim)
    statements = Expr[]

    if boundary
        for (hop_index, (hops, _)) in enumerate(T)
            wrapped_coordinates = Any[]
            for axis = 1:dim
                coordinate = Symbol("_m$(hop_index)_$axis")
                loop_var = site_loop_var(axis)
                hop = hops[axis]
                push!(statements, :($coordinate = $loop_var + $hop))
                push!(
                    statements,
                    :(
                        $coordinate = if $coordinate > L[$axis]
                            $coordinate - L[$axis]
                        else
                            ($coordinate < 1 ? $coordinate + L[$axis] : $coordinate)
                        end
                    ),
                )
                push!(wrapped_coordinates, coordinate)
            end
            input_index = Symbol("_iin$hop_index")
            push!(statements, :($input_index = $(site_linear_index(wrapped_coordinates))))
        end
    end

    for orbital in eachindex(V)
        accumulator = Symbol("_acc$orbital")
        diagonal = matrix_element(V[orbital]; site = site)
        if diagonal == :()
            push!(statements, :($accumulator = zero(ComplexF64)))
        else
            product = if real_values
                :(real($diagonal) * ψin[i+$orbital])
            else
                :($diagonal * ψin[i+$orbital])
            end
            push!(statements, :($accumulator = $product))
        end

        for (hop_index, (_, (rows, cols, values))) in enumerate(T)
            input_index = if boundary
                Symbol("_iin$hop_index")
            else
                :(i + $(Symbol("_hoff$hop_index")))
            end
            for (row, col, value) in zip(rows, cols, values)
                col == orbital || continue
                matrix_value = matrix_element(value; site = site)
                matrix_value == :() && continue
                product = if real_values
                    :(real($matrix_value) * ψin[$input_index+$row])
                else
                    :($matrix_value * ψin[$input_index+$row])
                end
                push!(statements, :($accumulator += $product))
            end
        end
        push!(statements, :(ψout[i+$orbital] = $accumulator))
    end

    Expr(:block, statements...)
end

"""
    loop_fused_interior(V, T, dim; real_values)

Generate the single loop nest over the common no-wrap interior box. The base index
is computed once at the start of each dimension-1 segment and then advanced by `d`,
so all hop inputs use constant linear offsets and every output is written once.
"""
function loop_fused_interior(V, T, dim; real_values = false)
    lower_vars = [Symbol("_lo$axis") for axis = 1:dim]
    upper_vars = [Symbol("_hi$axis") for axis = 1:dim]
    coordinates = Any[lower_vars[1], [site_loop_var(axis) for axis = 2:dim]...]
    site_expr = fused_site_expr(V, T, dim; boundary = false, real_values = real_values)
    site_loop = if real_values
        quote
            @simd ivdep for $(site_loop_var(1)) = ($(lower_vars[1])):($(upper_vars[1]))
                $site_expr
                i += d
            end
        end
    else
        quote
            for $(site_loop_var(1)) = ($(lower_vars[1])):($(upper_vars[1]))
                $site_expr
                i += d
            end
        end
    end
    inner = quote
        i = $(site_linear_index(coordinates))
        $site_loop
    end
    ranges =
        Any[nothing, [:(($(lower_vars[axis])):($(upper_vars[axis]))) for axis = 2:dim]...]
    loop_region(ranges, inner)
end

"""
    loop_fused_boundary_slab(V, T, dim, slab_axis)

Generate one slab of the complement of the fused interior. Earlier coordinates are
restricted to the interior, `slab_axis` is split into disjoint low/high ranges, and
later coordinates span the lattice. Using `max(hi + 1, lo)` for the high range also
partitions the full axis exactly once when that dimension's interior is empty.
"""
function loop_fused_boundary_slab(V, T, dim, slab_axis)
    lower_vars = [Symbol("_lo$axis") for axis = 1:dim]
    upper_vars = [Symbol("_hi$axis") for axis = 1:dim]
    body = quote
        i = $(site_linear_index(site_var_vector(dim)))
        $(fused_site_expr(V, T, dim; boundary = true))
    end

    ranges = Any[
        if axis < slab_axis
            :(($(lower_vars[axis])):($(upper_vars[axis])))
        elseif axis > slab_axis
            :(1:L[$axis])
        else
            nothing
        end for axis = 1:dim
    ]
    ranges[slab_axis] = :(1:($(lower_vars[slab_axis])-1))
    low_slab = loop_region(ranges, body)
    ranges[slab_axis] =
        :(max($(upper_vars[slab_axis])+1, $(lower_vars[slab_axis])):L[$slab_axis])
    high_slab = loop_region(ranges, body)
    Expr(:block, low_slab, high_slab)
end

"""
    all_real_matrix_elements(V, T)

Generate a prelude condition for the fused real-coefficient specialization. Hoisted
symbols are checked once per apply; real literals need no check, while complex
literals and site-dependent expressions conservatively select the exact complex
interior. Boundary slabs always use the exact complex expressions.

The specialization multiplies by `real(t)` instead of `t + 0im`. For finite inputs
this is exact, but when ψin contains `Inf` or `NaN` components it skips the
`0 * Inf = NaN` cross terms that full complex multiplication would produce, so
non-finite inputs can yield different (finite-imaginary) results than `sparse(H)`
or the boundary sites. This trade is intentional: it is what enables the `@simd
ivdep` interior sweep.
"""
function all_real_matrix_elements(V, T)
    checks = Expr[]
    matrix_values = Any[V...]
    for (_, (_, _, values)) in T
        append!(matrix_values, values)
    end
    for value in matrix_values
        if value isa ComplexF64
            isreal(value) || return false
        elseif value isa Symbol
            push!(checks, :(isreal($value)))
        else
            return false
        end
    end
    isempty(checks) ? true : foldl((left, right) -> :($left && $right), checks)
end

"""
    fused_apply_expr(V, T, dim)

Generate the matrix-free Hamiltonian body as one interior sweep plus ordered,
pairwise-disjoint boundary slabs. The prelude retains the lattice spans and adds
one constant linear hop offset and the common interior bounds.
"""
function fused_apply_expr(V, T, dim)
    # A `T[hops]` entry `(row, col, value)` means H[(n + hops, row), (n, col)] = value,
    # matching the sparse constructor: `(1) -> t` puts `t` on ⟨n+1|H|n⟩. The fused
    # kernels gather into the output site, so re-key the table by the gather
    # displacement: at output site n the entry contributes value * ψin[n - hops, col]
    # to the accumulator for orbital `row`. Sorting keeps the generated
    # accumulation order independent of Dict internals.
    T = sort!(
        [-hops => (cols, rows, values) for (hops, (rows, cols, values)) in T];
        by = Tuple ∘ first,
    )
    spans = Expr[
        :($(dim_span_var(1)) = d * L[1]),
        [:($(dim_span_var(axis))=$(dim_span_var(axis-1))*L[$axis]) for axis = 2:dim]...,
    ]
    offsets = Expr[
        :($(Symbol("_hoff$hop_index")) = $(hop_linear_offset(hops))) for
        (hop_index, (hops, _)) in enumerate(T)
    ]
    lower, upper = interior_margins(T, dim)
    bounds = Expr[]
    for axis = 1:dim
        push!(bounds, :($(Symbol("_lo$axis")) = $(1 + lower[axis])))
        push!(bounds, :($(Symbol("_hi$axis")) = L[$axis] - $(upper[axis])))
    end
    real_condition = all_real_matrix_elements(V, T)
    interior = if real_condition === true
        loop_fused_interior(V, T, dim; real_values = true)
    elseif real_condition === false
        loop_fused_interior(V, T, dim)
    else
        quote
            if $real_condition
                $(loop_fused_interior(V, T, dim; real_values = true))
            else
                $(loop_fused_interior(V, T, dim))
            end
        end
    end
    slabs = [loop_fused_boundary_slab(V, T, dim, axis) for axis = 1:dim]
    Expr(:block, spans..., offsets..., bounds..., interior, slabs...)
end

"""
    hop_extent_guard(T, dim)

Generate call-time checks that every compiled hopping displacement still fits
the lattice extents. The DSL folds long displacements at parse time, but `L` is
mutable on the built Hamiltonian; the wrap logic corrects by at most one
lattice period, so a displacement with `abs(hop) > L[axis]` would index out of
bounds inside the `@inbounds` kernels.
"""
function hop_extent_guard(T, dim)
    checks = Expr[]
    seen = Set{NTuple{2,Int}}()
    for (hops, _) in T, axis = 1:dim
        hop = abs(hops[axis])
        iszero(hop) && continue
        (hop, axis) in seen && continue
        push!(seen, (hop, axis))
        message = "hopping displacement of magnitude $hop exceeds the lattice extent along dimension $axis; rebuild the Hamiltonian for this lattice size"
        push!(checks, :($hop <= L[$axis] || throw(ArgumentError($message))))
    end
    Expr(:block, checks...)
end

"""
    make_apply(V, T, dim)

Generate an `Expr` that defines a function to apply the Hamiltonian
given parameters for the potential and hopping terms (V and T).
"""
function make_apply(V, T, dim; open = falses(dim))
    bindings, V2, T2 = hoist_matrix_elements(V, T, dim)
    quote
        function (
            ψout::AbstractArray,
            ψin::AbstractArray,
            d::Int,
            L::MVector{$dim,Int64},
            params::Dict{Symbol,ComplexF64},
        )
            length(ψout) == d * prod(L) && length(ψin) == d * prod(L) ||
                throw(DimensionMismatch("input and output vectors must both have length d * prod(L)"))
            $(hop_extent_guard(T2, dim))
            # The arguments of the function are used implicitly
            # in the generated expressions
            #
            # see e.g., diag_expr, hop_expr
            @inbounds begin
                $bindings
                $(fused_apply_expr(V2, T2, dim; open = open))
            end
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
    bindings, V2, T2 = hoist_matrix_elements(V, T, dim)
    per_cell_count = bound_nonzero(1, length(V), T)
    quote
        function (d::Int, L::MVector{$dim,Int64}, params::Dict{Symbol,ComplexF64})
            $(hop_extent_guard(T2, dim))
            $bindings
            nz = $per_cell_count * prod(L)
            matrix_size = d * prod(L)

            ivals = Array{Int64}(undef, nz)
            jvals = Array{Int64}(undef, nz)
            hvals = Array{ComplexF64}(undef, nz)

            idx = 1
            @inbounds begin
                $(ham_expr(V2, T2, dim; sparse = true, open = open))
            end
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
