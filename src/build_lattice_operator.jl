"""
    make_diag_expr(Vs; s = :n)

Constructs the diagonal part of the Hamiltonian matrix
in an efficient way by using metaprogramming features of Julia.
It takes the on-site potentials `Vs` and generates a corresponding `Expr` block,
which, when evaluated, would perform the operations of the diagonal part of the Hamiltonian.

If `sparse` is set to `true`, the function generates an `Expr` block that
constructs the sparse matrix representation of the diagonal part of the Hamiltonian.
"""
function make_diag_expr(Vs; s = :n, sparse = false)
    expr_array = Vector{Expr}(undef, length(Vs) + 1)
    for i in eachindex(Vs)
        V = Vs[i]
        if sparse
            expr_array[i] = quote
                ivals[idx] = ind1 + $i
                jvals[idx] = ind1 + $i
                hvals[idx] = $V
                idx += 1
            end
        else
            if typeof(V) <: ComplexF64
                expr_array[i] = if (V != zero(ComplexF64))
                    :(ψout[ind1+$i] = $V * ψin[ind1+$i])
                else
                    :(ψout[ind1+$i] = zero(ComplexF64))
                end
            elseif typeof(V) <: Symbol
                expr_array[i] = :(ψout[ind1+$i] = $V * ψin[ind1+$i])
            elseif typeof(V) <: Function
                ss = [Symbol("$(s)$m") for m = 1:length(Vs)]
                expr_array[i] = :(ψout[ind1+$i] = $V($(ss...)) * ψin[ind1+$i])
            elseif typeof(V) <: Expr
                expr_array[i] = :(ψout[ind1+$i] = $(V) * ψin[ind1+$i])
            end
        end
    end
    expr_array[end] = :(ind1 += d)
    return Expr(:block, expr_array...)
end

"""
    make_hop_expr(is, js, Vs, dim; s = :n)

Constructs the off-diagonal part of the Hamiltonian matrix
by generating an `Expr` block, which performs the operations of the off-diagonal part of the Hamiltonian.

If `sparse` is set to `true`, the function generates an `Expr` block that
constructs the sparse matrix representation of the off-diagonal part of the Hamiltonian.
"""
function make_hop_expr(is, js, Vs, dim; s = :n, sparse = false)
    expr_array = Vector{Expr}(undef, length(is) + 1)
    for idx in eachindex(is)
        i = is[idx]
        j = js[idx]
        V = Vs[idx]
        if sparse
            expr_array[idx] = quote
                ivals[idx] = ind2 + $i
                jvals[idx] = ind1 + $j
                hvals[idx] = $V
                idx += 1
            end
        else
            if typeof(V) <: ComplexF64
                expr_array[idx] = if (V != zero(ComplexF64))
                    :(ψout[ind1+$j] += $V * ψin[ind2+$i])
                else
                    :(zero(ComplexF64))
                end
            elseif typeof(V) <: Symbol
                expr_array[idx] = :(ψout[ind1+$j] += $V * ψin[ind2+$i])
            elseif typeof(V) <: Function
                ss = [Symbol("$(s)$m") for m = 1:dim]
                expr_array[idx] = :(ψout[ind1+$j] += $V($(ss...)) * ψin[ind2+$i])
            elseif typeof(V) <: Expr
                expr_array[idx] = :(ψout[ind1+$j] += $(V) * ψin[ind2+$i])
            end
        end
    end
    expr_array[end] = :(ind1 += d; ind2 += d)
    return Expr(:block, expr_array...)
end

"""
    site_expr(hops[, j])

Recursively generate an expression for calculating the site
`site_expr` is overloaded for different numbers of arguments.
"""
site_expr(hops) = site_expr(hops, 1)

function site_expr(hops, j)
    if j == length(hops)
        t = hops[j] ≥ 0 ? :($(hops[j])) : :(L[$j] - $(-hops[j]))
        return t
    end
    t = if hops[j] ≥ 0
        :($(hops[j]) + $(site_expr(hops, j + 1)) * L[$j])
    else
        :(L[$j] * (1 + $(site_expr(hops, j + 1))) - $(-hops[j]))
    end
    return t
end

"""
    loop_periodic(hops, ex; s = :n)
    loop_periodic_diag(dim, d, ex; s = :n)
    loop_periodic(s, hop, ex, j)

Generate an `Expr` block for loops iterating over lattice sites
in a periodic system.
"""
function loop_periodic(hops, ex; s = :n)
    quote
        ind1 = 0
        ind2 = d * $(site_expr(hops))
        $(loop_periodic(s, hops, ex, length(hops)))
    end
end

function loop_periodic_diag(dim, d, ex; s = :n)
    quote
        ind1 = 0
        $(loop_periodic(s, zeros(Int, dim), ex, dim))
    end
end

function loop_periodic(s, hop, ex, j)
    if j == 0
        return ex
    end
    m::Int = hop[j]
    sj = Symbol("$(s)$j")
    Lprodj = Symbol("Lprod$j")
    if m < 0
        expr = quote
            for $sj = 1:$(-m)
                $(loop_periodic(s, hop, ex, j - 1))
            end
            ind2 -= $Lprodj
            for $sj = $(-m + 1):L[$j]
                $(loop_periodic(s, hop, ex, j - 1))
            end
            ind2 += $Lprodj
        end
    elseif m > 0
        expr = quote
            for $sj = 1:(L[$j]-$m)
                $(loop_periodic(s, hop, ex, j - 1))
            end
            ind2 -= $Lprodj
            for $sj = (L[$j]-$(m - 1)):L[$j]
                $(loop_periodic(s, hop, ex, j - 1))
            end
            ind2 += $Lprodj
        end
    else
        expr = quote
            for $sj = 1:L[$j]
                $(loop_periodic(s, hop, ex, j - 1))
            end
        end
    end
end

"""
    ham_expr(V, T, dim)

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
function ham_expr(V, T, dim; sparse = false)
    # since we are flattening the multi-dimensional arrays into 1D arrays,
    # we need to keep track of the stride of the lattice dimensions
    # for each lattice dimension

    # the orbital index changes the fastest
    # then the lattice indices
    # The variables Lprod1, Lprod2, ..., Lproddim store the stride of the lattice dimensions
    # so that increasing the index by Lprod$(i-1) corresponds to moving to the next lattice site in the $i-th dimension
    # Lprod$i is the total span traversed by the $i-th dimension when holding all other indices fixed.
    expr_Lprods = Expr(
        :block,
        [
            [:(Lprod1 = d * L[1])]
            [:($(Symbol("Lprod$i")) = $(Symbol("Lprod$(i-1)")) * L[$i]) for i = 2:dim]
        ]...,
    )
    # evaluate the diagonal elements of the Hamiltonian
    expr_V = make_diag_expr(V; sparse = sparse)
    expr_diag = loop_periodic_diag(dim, length(V), expr_V)

    # evaluate the off-diagonal elements
    expr_hops = Vector{Expr}(undef, length(T))
    idx = 1
    for (hops, (is, js, hs)) in T
        expr_T = make_hop_expr(is, js, hs, dim; sparse = sparse)
        expr_hops[idx] = loop_periodic(hops, expr_T)
        idx += 1
    end

    return Expr(:block, expr_Lprods, expr_diag, expr_hops...)
end

"""
    make_apply(params, V, T, dim)

Generate an `Expr` that defines a function to apply the Hamiltonian
given parameters for the potential and hopping terms (V and T).
"""
function make_apply(params, V, T, dim)
    # ps = [:($k::typeof($v)) for (k, v) in params]
    # println(ps)
    return quote
        function (
            ψout::AbstractArray,
            ψin::AbstractArray,
            d::Int,
            dim::Int,
            L::MVector{$dim,Int64},
            params::Dict{Symbol,ComplexF64},
        )
            # The arguments of the function are used implicitly
            # in the generated expressions
            #
            # see make_diag_expr
            $(ham_expr(V, T, dim))
        end
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