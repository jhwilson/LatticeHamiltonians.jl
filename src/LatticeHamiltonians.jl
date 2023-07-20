# The `LatticeHamiltonians` module contains functions and structures
# for constructing and applying Hamiltonians on lattice systems.
module LatticeHamiltonians

using StaticArrays, LinearAlgebra

export LatticeHamiltonian
export @lattice_hamiltonian

# This function constructs the diagonal part of the Hamiltonian matrix
# in an efficient way by using metaprogramming features of Julia. 
# It takes the on-site potentials `Vs` and generates a corresponding `Expr` block,
# which, when evaluated, would perform the operations of the diagonal part of the Hamiltonian.
function make_diag_expr(Vs; s = :n)
    expr_array = Vector{Expr}(undef, length(Vs)+1)
    for i = eachindex(Vs)
        V = Vs[i]
        if typeof(V) <: ComplexF64
            expr_array[i] = (V != zero(ComplexF64)) ? :(ψout[ind1 + $i] = $V * ψin[ind1 + $i]) : :(ψout[ind1 + $i] = zero(ComplexF64))
        elseif typeof(V) <: Function
            ss = [Symbol("$(s)$m") for m = 1:length(Vs)]
            expr_array[i] = :(ψout[ind1 + $i] = $V($(ss...)) * ψin[ind1 + $i])
        elseif typeof(V) <: Expr
            expr_array[i] = :(ψout[ind1 + $i] = $(V) * ψin[ind1 + $j])
        end
    end
    expr_array[end] = :(ind1 += d)
    return Expr(:block, expr_array...)
end

# This function constructs the off-diagonal part of the Hamiltonian matrix 
# by generating an `Expr` block, which performs the operations of the off-diagonal part of the Hamiltonian.
function make_hop_expr(is, js, Vs, dim; s = :n)
    expr_array = Vector{Expr}(undef, length(is)+1)
    for idx = eachindex(is)
        i = is[idx]
        j = js[idx]
        V = Vs[idx]
        if typeof(V) <: ComplexF64  
            expr_array[idx] = (V!=zero(ComplexF64)) ? :(ψout[ind1 + $i] += $V * ψin[ind2 + $j]) : :(zero(ComplexF64))
        elseif typeof(V) <: Function
            ss = [Symbol("$(s)$m") for m = 1:dim]
            expr_array[idx] = :(ψout[ind1 + $i] += $V($(ss...)) * ψin[ind2 + $j])
        elseif typeof(V) <: Expr
            expr_array[idx] = :(ψout[ind1 + $i] += $(V) * ψin[ind2 + $j])
        end 
    end
    expr_array[end] = :(ind1 += d; ind2 += d)
    return Expr(:block, expr_array...)
end

# These functions recursively generate an expression for calculating the site index 
# in the 1D unwrapped lattice from the coordinates of the site in the original lattice.
# `site_expr` is overloaded for different numbers of arguments. 
site_expr(hops) = site_expr(hops, 1)

function site_expr(hops, j)
    if j == length(hops) 
        t = hops[j] ≥ 0 ? :($(hops[j])) : :(N[$j] - $(-hops[j]))
        return t
    end
    t = hops[j] ≥ 0 ? :($(hops[j]) + $(site_expr(hops, j + 1)) * N[$j] ) : :(N[$j] * (1 + $(site_expr(hops, j + 1))) - $(-hops[j]))
    return t
end

# These functions generate an `Expr` block for loops iterating over lattice sites 
# in a periodic system.
loop_periodic(hops, ex; s = :n) = :(ind1 = 0; 
    ind2 = d * $(site_expr(hops));
    $(loop_periodic(s, hops, ex, length(hops))))

loop_periodic_diag(dim, d, ex; s = :n) = :(ind1 = 0;  
    $(loop_periodic(s, zeros(Int, dim), ex, dim)))

    
function loop_periodic(s, hop, ex, j, flag)
    if j == 0
        return ex
    end
    m::Int = hop[j]
    if m < 0
        sj = Symbol("$(s)$j")
        if (flag)
            expr = :(for $sj = 1:$(-m)
                $(loop_periodic(s, hop, ex, j - 1))
            end)
            expr = :(
        end
        ind2 -= d * cumprod(N[1:j], 1); #NB: cumprod and prod should both work left to right, also use Arrayview here? subarray?
        for $(Symbol("$(s)$j")) = $(-m + 1):N[$j]
            $(loop_periodic(s, hop, ex, j - 1))
        end;
        ind2 += d * cumprod(N[1:j], 1))
    elseif m > 0
        if (flag)
            expr = :(for $(Symbol("$(s)$j")) = 1:(N[$j]-$m)
                $(loop_periodic(s, hop, ex, j - 1))
            end;)
            expr = :(
        end
        ind2 -= d * cumprod(N[1:j], 1);
        for $(Symbol("$(s)$j")) = (N[$j]-$(m - 1)):N[$j]
            $(loop_periodic(s, hop, ex, j - 1))
        end;
        ind2 += d * cumprod(N[1:j], 1))
    else
        expr = :(for $(Symbol("$(s)$j")) = 1:N[$j]
            $(loop_periodic(s, hop, ex, j - 1))
        end)
    end
end

# `ham_expr` is a function that combines the expressions for the diagonal and hopping terms to generate 
# a block of expressions that applies the full Hamiltonian.
function ham_expr(V, T, dim)
    expr_V = make_diag_expr(V)
    expr_diag = loop_periodic_diag(dim, length(V), expr_V)
    expr_hops = Vector{Expr}(undef, length(T))
    idx = 1
    for (hops, (is, js, hs)) in T
        expr_T = make_hop_expr(is, js, hs, dim)
        expr_hops[idx] = loop_periodic(hops, expr_T)
        idx += 1
    end
    return Expr(:block, expr_diag, expr_hops...)
end

# This structure is used to store information about the Hamiltonian of the system.
# It includes information about the real-space basis (A, B), number of orbitals (d), system size (L),
# parameters for potential and hopping functions (params), real space coordinates of orbitals within a unit cell (r),
# and a function that applies the Hamiltonian to an input wavefunction (apply!).
struct LatticeHamiltonian{real_dim, lattice_dim, F}
    A::SMatrix{real_dim, lattice_dim, Float64}
    B::SMatrix{real_dim, lattice_dim, Float64}
    d::Int
    L::MVector{lattice_dim, Int}
    params::Dict{Symbol, ComplexF64}
    r::Vector{SVector{real_dim,Float64}}
    apply!::F
end

# `make_apply` function generates an `Expr` that defines a function to apply the Hamiltonian
# given parameters for the potential and hopping terms (V and T).
function make_apply(params, V, T, dim)
    # ps = [:($k::typeof($v)) for (k, v) in params]
    # println(ps)
    return quote
        function (ψout::AbstractArray, ψin::AbstractArray, d::Int, dim::Int, N, $([:($k::$(typeof(v))) for (k, v) in params]...))
            $(ham_expr(V, T, dim))
        end
    end
end

# `make_multiply` function generates an `Expr` that defines a function to apply the Hamiltonian
# to an input wavefunction using the method defined by `make_apply`.
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

# `lattice_hamiltonian` is a macro that provides a convenient interface for defining a Hamiltonian. 
# The user can provide an expression that defines the parameters for the potential and hopping terms (params, V, T, Ls).
# The macro generates code that constructs a `LatticeHamiltonian` object and a function to apply the Hamiltonian.
macro lattice_hamiltonian(ex)
    # V = ComplexF64[0.0,0.0]
    # T = Dict([0] => ([1,2],[2,1],ComplexF64[1.0, 1.0]),
    # [1] => ([2], [1], [:(1.0 * t)]),
    # [-1] => ([1], [2], [:(1.0 * t)]))
    # params = Dict{Symbol,ComplexF64}(:t => 1.0)
    # L = [10]
    # params = ... 
    # V = ... 
    # T = ...
    # L = ...
    # eval(ex)
    # dim = maximum(length(k) for k in keys(T))
    # d = length(V)
    # A = SMatrix{dim, dim, Float64}(I)
    # B = SMatrix{dim, dim, Float64}(I)
    # L = MVector{dim, Int}(Ls)
    # r = [SVector{dim}(zeros(Float64, dim)) for i = 1:d]
    # apply = make_apply(params, V, T, dim)
    # H = LatticeHamiltonian(A, B, d, L, params, r, apply)
    # mult_expr = make_multiply(H)

    # From Kimberly's macro put all of the front matter here
    
    return esc(quote
        $(ex) #This will be gone
        dim = maximum(length(k) for k in keys(T)) #moved out of quote block
        d = length(V) #moved out quote block
        A = SMatrix{dim, dim, Float64}(I) #moved out of quote block
        B = SMatrix{dim, dim, Float64}(I) #moved out of quote block
        L = MVector{dim, Int}(Ls) #moved out of quote block
        r = [SVector{dim}(zeros(Float64, dim)) for i = 1:d] #moved out quote block
        apply = eval(LatticeHamiltonians.make_apply(params, V, T, dim)) #moved out of quote block
        H = LatticeHamiltonian(A, B, d, L, params, r, apply) #moved out of quote block (??)
        mult_expr = LatticeHamiltonians.make_multiply(H) # moved out of quote block
        eval(mult_expr) # replaced with $mult_expr
        # end with $H
    end)
end

end
