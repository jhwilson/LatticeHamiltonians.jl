"""
    make_diag_expr(Vs; s = :n)

Constructs the diagonal part of the Hamiltonian matrix
in an efficient way by using metaprogramming features of Julia.
It takes the on-site potentials `Vs` and generates a corresponding `Expr` block,
which, when evaluated, would perform the operations of the diagonal part of the Hamiltonian.
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

    loop_periodic(s, hop, ex, j, BCs = "true/1, true/1, false/0"), ex. array for periodic in two directions and open in one


Generate an `Expr` block for loops iterating over lattice sites
in a periodic system.
"""

function loop_periodic(hops, ex; s=:n)
    quote
        ind1 = 0
        ind2 = d * $(site_expr(hops))
        $(loop_periodic(s, hops, ex, length(hops); BCs="Periodic"))
    end
end

function loop_periodic_diag(dim, d, ex; s=:n)
    quote
        ind1 = 0
        $(loop_periodic(s, zeros(Int, dim), ex, dim; BCs="Periodic"))
    end
end

function loop_periodic(s, hop, ex, j; BCs="Periodic")
    if j == 0
        return ex
    end
    if BCs == "Periodic"
        BCs = ones(Bool, j)
    elseif BCs == "Open"
        BCs = zeros(Bool, j)
    else
        if eltype(BCs) <: Bool
            if length(BCs) != j
                if length(BCs) > j
                    if length(BCs) == j + 1 #truncate BCs to current j, guaranteed to be BoundsError if over
                        BCs = deleteat!(deepcopy(BCs), j + 1)
                    else
                        error("BCs if mixed or not should be array with length matching no. of dims")
                    end
                elseif length(BCs) < j
                    error("BCs if mixed or not should be array with length matching no. of dims")
                end
            end
        else
            error("BCs should be Bools")
        end

    end
    m::Int = hop[j]
    sj = Symbol("$(s)$j")
    Lprodj = Symbol("Lprod$j")
    if m < 0 && !BCs[j]
        expr = quote
            for $sj = 1:$(-m)
                $(loop_periodic(s, hop, ex, j - 1; BCs=BCs))
            end
            ind2 -= $Lprodj
            for $sj = $(-m + 1):L[$j]
                $(loop_periodic(s, hop, ex, j - 1; BCs=BCs))
            end
            ind2 += $Lprodj
        end
    elseif m < 0 && BCs[j] #change to comport with function header
        expr = quote
            for $sj = 1:$(-m)
                $(loop_periodic(s, hop, ex, j - 1; BCs=BCs))
            end
            ind1 += m * $Lprodj - 1
            ind2 += m * $Lprodj - 1
        end
    elseif m > 0 && !BCs[j] #change to comport with function header
        expr = quote
            for $sj = 1:(L[$j]-$m)
                $(loop_periodic(s, hop, ex, j - 1; BCs=BCs))
            end
            ind2 -= $Lprodj
            for $sj = (L[$j]-$(m - 1)):L[$j]
                $(loop_periodic(s, hop, ex, j - 1; BCs=BCs))
            end
            ind2 += $Lprodj
        end
    elseif m > 0 && BCs[j]
        expr = quote
            ind1 += m * $Lprodj - 1
            for $sj = (L[$j]-$(m - 1)):L[$j]
                $(loop_periodic(s, hop, ex, j - 1; BCs=BCs))
            end
            ind2 += m * $Lprodj - 1
        end
    else
        expr = quote
            for $sj = 1:L[$j]
                $(loop_periodic(s, hop, ex, j - 1; BCs=BCs))

            end
        end
    end
end

"""
    ham_expr(V, T, dim)

Combine the expressions for the diagonal and hopping terms to generate
a block of expressions that applies the full Hamiltonian.
"""

function ham_expr(V, T, dim; sparse=false)
    expr_Lprods = Expr(
        :block,
        [
            [:(Lprod1 = d * L[1])]
            [:($(Symbol("Lprod$i")) = $(Symbol("Lprod$(i-1)")) * L[$i]) for i = 2:dim]
        ]...,
    )
    expr_V = make_diag_expr(V; sparse=sparse)
    expr_diag = loop_periodic_diag(dim, length(V), expr_V)
    expr_hops = Vector{Expr}(undef, length(T))
    idx = 1
    for (hops, (is, js, hs)) in T

        expr_T = make_hop_expr(is, js, hs, dim; sparse=sparse)
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
            $(ham_expr(V, T, dim))
        end
    end
end
