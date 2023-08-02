"""
Functions and structures for constructing and applying Hamiltonians on lattice systems.
"""
module LatticeHamiltonians

using StaticArrays, LinearAlgebra, SparseArrays
import SparseArrays: sparse
import LinearAlgebra: mul!
import Base: *, size, length, eltype, adjoint

export LatticeHamiltonian, mul!, *, sparse, size, length, eltype, adjoint
export @lattice_hamiltonian

"""
    make_diag_expr(Vs; s = :n)

Constructs the diagonal part of the Hamiltonian matrix
in an efficient way by using metaprogramming features of Julia.
It takes the on-site potentials `Vs` and generates a corresponding `Expr` block,
which, when evaluated, would perform the operations of the diagonal part of the Hamiltonian.
"""
function make_diag_expr(Vs; s = :n)
    expr_array = Vector{Expr}(undef, length(Vs) + 1)
    for i in eachindex(Vs)
        V = Vs[i]
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
    expr_array[end] = :(ind1 += d)
    println(expr_array)
    return Expr(:block, expr_array...)
end

"""
    make_hop_expr(is, js, Vs, dim; s = :n)

Constructs the off-diagonal part of the Hamiltonian matrix
by generating an `Expr` block, which performs the operations of the off-diagonal part of the Hamiltonian.
"""
function make_hop_expr(is, js, Vs, dim; s = :n)
    expr_array = Vector{Expr}(undef, length(is) + 1)
    for idx in eachindex(is)
        i = is[idx]
        j = js[idx]
        V = Vs[idx]
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
        t = hops[j] ≥ 0 ? :($(hops[j])) : :(N[$j] - $(-hops[j]))
        return t
    end
    t = if hops[j] ≥ 0
        :($(hops[j]) + $(site_expr(hops, j + 1)) * N[$j])
    else
        :(N[$j] * (1 + $(site_expr(hops, j + 1))) - $(-hops[j]))
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
            for $sj = $(-m + 1):N[$j]
                $(loop_periodic(s, hop, ex, j - 1))
            end
            ind2 += $Lprodj
        end
    elseif m > 0
        expr = quote
            for $sj = 1:(N[$j]-$m)
                $(loop_periodic(s, hop, ex, j - 1))
            end
            ind2 -= $Lprodj
            for $sj = (N[$j]-$(m - 1)):N[$j]
                $(loop_periodic(s, hop, ex, j - 1))
            end
            ind2 += $Lprodj
        end
    else
        expr = quote
            for $sj = 1:N[$j]
                $(loop_periodic(s, hop, ex, j - 1))
            end
        end
    end
end

"""
    ham_expr(V, T, dim)

Combine the expressions for the diagonal and hopping terms to generate
a block of expressions that applies the full Hamiltonian.
"""
function ham_expr(V, T, dim)
    expr_Lprods = Expr(
        :block,
        [
            [:(Lprod1 = d * N[1])]
            [:($(Symbol("Lprod$i")) = $(Symbol("Lprod$(i-1)")) * N[$i]) for i = 2:dim]
        ]...,
    )
    expr_V = make_diag_expr(V)
    expr_diag = loop_periodic_diag(dim, length(V), expr_V)
    expr_hops = Vector{Expr}(undef, length(T))
    idx = 1
    for (hops, (is, js, hs)) in T
        expr_T = make_hop_expr(is, js, hs, dim)
        expr_hops[idx] = loop_periodic(hops, expr_T)
        idx += 1
    end
    return Expr(:block, expr_Lprods, expr_diag, expr_hops...)
end

"""
This structure is used to store information about the Hamiltonian of the system.
It includes information about the real-space basis (A, B), number of orbitals (d), system size (L),
parameters for potential and hopping functions (params), real space coordinates of orbitals within a unit cell (r),
and a function that applies the Hamiltonian to an input wavefunction (apply!).
"""
struct LatticeHamiltonian{real_dim,lattice_dim,F}
    A::SMatrix{real_dim,lattice_dim,Float64}
    B::SMatrix{real_dim,lattice_dim,Float64}
    d::Int
    L::MVector{lattice_dim,Int}
    params::Dict{Symbol,ComplexF64}
    T::Dict{
        Vector{Int64},
        Tuple{Vector{Int64},Vector{Int64},Vector{Union{Expr,Symbol,ComplexF64}}},
    }
    V::Vector{Union{Expr,Symbol,ComplexF64}}
    r::Vector{SVector{real_dim,Float64}}
    apply!::F
end

function size(H::LatticeHamiltonian)
    l = H.d * prod(H.L)
    return l, l
end

function length(H::LatticeHamiltonian)
    return (H.d * prod(H.L))^2
end

function eltype(H::LatticeHamiltonian) #Hardcoded as ComplexF64 for the moment.
    return ComplexF64
end

function adjoint(H::LatticeHamiltonian) #Hardcoding that the matrix is Hermitian!!
    return H
end

function countnz(H::LatticeHamiltonian)
    vol = prod(H.L)
    nz = H.d
    for t in H.T
        nz += length(t[2][1])
    end
    nz *= vol
    return nz
end

function sitenumber(r, H::LatticeHamiltonian)
    dim = length(H.L)
    sitenum = r[dim]
    for j = 1:(dim-1)
        sitenum = r[dim-j] + sitenum * H.L[dim-j]
    end
    return 1 + sitenum
end

function sparse(H::LatticeHamiltonian)
    nz = countnz(H)
    ivals = Array{Int64}(undef, nz)
    jvals = Array{Int64}(undef, nz)
    hvals = Array{ComplexF64}(undef, nz)
    ii = 1
    R = CartesianIndices(Tuple(0:(H.L[j]-1) for j in eachindex(H.L)))
    nhop = Vector{Int64}(undef, length(H.L))
    for p in H.params
        eval(:($(p[1]) = $(p[2])))
    end
    for n in R
        sitenum = sitenumber(n, H)
        for σ = 1:H.d
            r = H.d * (sitenum - 1) + σ
            ivals[ii] = r
            jvals[ii] = r
            hvals[ii] = eval(H.V[σ])
            ii += 1
        end
        for t in H.T
            for i in eachindex(nhop)
                nhop[i] = mod(n[i] + t[1][i], H.L[i])
            end
            sitenum_hop = sitenumber(nhop, H)
            for σ in eachindex(t[2][1])
                ivals[ii] = H.d * (sitenum_hop - 1) + t[2][1][σ]
                jvals[ii] = H.d * (sitenum - 1) + t[2][2][σ]
                hvals[ii] = eval(t[2][3][σ])
                ii += 1
            end
        end
    end
    return dropzeros(sparse(ivals, jvals, hvals))
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
            N,
            $([:($k::$(typeof(v))) for (k, v) in params]...),
        )
            $(ham_expr(V, T, dim))
        end
    end
end

# BELOW IS DEPRACATED BUT NOT DELETED YET
"""
Generate an `Expr` that defines a function to apply the Hamiltonian
to an input wavefunction using the method defined by `make_apply`.
"""
function make_multiply(H)
    params_expr_array = Vector{Expr}(undef, length(H.params))
    idx = 1
    for (s, val) in H.params
        params_expr_array[idx] = :($(Symbol(s))::$(typeof(val)) = H.params[$(QuoteNode(s))])
        idx += 1
    end
    return quote
        function mult!(ψout::AbstractArray, H::$(typeof(H)), ψin::AbstractArray)
            d::Int = H.d
            dim::Int = length(H.L)
            N = H.L
            $(Expr(:block, params_expr_array...))
            H.apply!(ψout, ψin, d, dim, N, $(keys(H.params)...))
        end
    end
end

function mul!(ψout::AbstractArray, H::LatticeHamiltonian, ψin::AbstractArray)
    H.apply!(ψout, ψin, H.d, length(H.L), H.L, values(H.params)...)
end

function *(H::LatticeHamiltonian, ψ::AbstractVector)
    v = copy(ψ)
    mul!(v, H, ψ)
    return v
end

# `lattice_hamiltonian` is a macro that provides a Convenient interface for defining a Hamiltonian.

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

    # T::Dict{Vector{Int64},Tuple{Vector{Int64},Vector{Int64},Vector{Union{Expr, Symbol, ComplexF64}}}}
    # V::Vector{Union{Expr, Symbol, ComplexF64}}
    # V = eval(exprV.args[2])
    # d = length(V)
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
    V = Vector{Union{Expr,Symbol,ComplexF64}}(undef, d)
    for i in eachindex(exprV.args[2].args)
        try
            V[i] = ComplexF64(eval(exprV.args[2].args[i]))
        catch e
            V[i] = exprV.args[2].args[i]
        end
    end
    T = Dict{
        Vector{Int64},
        Tuple{Vector{Int64},Vector{Int64},Vector{Union{Expr,Symbol,ComplexF64}}},
    }()
    for i in eachindex(hops)
        hop = hops[i]
        Base.remove_linenums!(hop)
        args1 = vec(eval(hop.args[1]) |> collect)
        m = hop.args[2].args[1]
        y, j, h = fnzi(m)
        T[args1] = (y, j, h)
    end

    A = SMatrix{dim,dim,Float64}(I)
    B = SMatrix{dim,dim,Float64}(I * 2 * pi)
    r = [SVector{dim}(zeros(Float64, dim)) for i = 1:d]
    apply = eval(make_apply(params, V, T, dim))
    H = LatticeHamiltonian(A, B, d, L, params, T, V, r, apply)
    quote
        $H
    end
end

function fnzi(matrix)
    cols = 0
    i = Int[]
    j = Int[]
    k = Vector{Union{ComplexF64,Symbol,Expr}}(undef, 0)

    for (ri, row) in enumerate(matrix.args)
        cols = max(cols, length(row.args))
        for (ci, elem) in enumerate(row.args)
            if elem != 0
                push!(i, ri)
                push!(j, ci)
                if elem isa Number #7_6 added ifelse
                    push!(k, ComplexF64(elem))
                else
                    push!(k, elem)
                end
            end
        end
    end
    return i, j, k
end

end
