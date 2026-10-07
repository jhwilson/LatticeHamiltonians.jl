"""
    HamiltonianBuilder{real_dim,lattice_dim}

Contains the information needed to build a Hamiltonian for a lattice system.

# Fields

The fields here are subset of those on `LatticeHamiltonian` and are used to construct the Hamiltonian.

See also: [`@builder`](@ref), [`build`](@ref)
"""
Base.@kwdef struct HamiltonianBuilder{real_dim,lattice_dim}
    # A::SMatrix{real_dim,lattice_dim,Float64}
    # B::SMatrix{real_dim,lattice_dim,Float64}
    # r::Vector{SVector{real_dim,Float64}}
    V::Vector{LiteralOrSymbolic}
    T::Dict{LatticeVector{lattice_dim},SparseEntry{LiteralOrSymbolic}}
    d::Int
    L::MVector{lattice_dim,Int}
    open::MVector{lattice_dim,Bool}
    params::Dict{Symbol,ComplexF64}
end

"""
    @builder

Construct as `HamiltonianBuilder` using the same DSL as
[`@lattice_hamiltonian`](@ref).
"""
macro builder(input)
    parse_lattice_dsl(input)
end

"""
    build(builder::HamiltonianBuilder{real_dim,lattice_dim})

Create a `LatticeHamiltonian` from a `HamiltonianBuilder`.
"""
function build(
    builder::HamiltonianBuilder{real_dim,lattice_dim},
) where {real_dim,lattice_dim}
    # Real space structure
    A = SMatrix{lattice_dim,lattice_dim,Float64}(I) #TODO
    B = SMatrix{lattice_dim,lattice_dim,Float64}(I * 2 * pi) #TODO
    r = fill(SVector{lattice_dim}(zeros(Float64, lattice_dim)), builder.d) #TODO

    LatticeHamiltonian(
        A,
        B,
        builder.d,
        builder.L,
        builder.params,
        r,
        eval(make_apply(builder)),
        eval(make_sparse(builder)),
    )
end