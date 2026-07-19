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
    params::Dict{Symbol,ComplexF64}
    fields::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

"""
    @builder

Construct as `HamiltonianBuilder` using the same DSL as
[`@lattice_hamiltonian`](@ref).
"""
macro builder(input)
    parse_lattice_dsl(input, __module__)
end

"""
    build(builder::HamiltonianBuilder{real_dim,lattice_dim})

Create a `LatticeHamiltonian` from a `HamiltonianBuilder`.
"""
function build(
    builder::HamiltonianBuilder{real_dim,lattice_dim},
) where {real_dim,lattice_dim}
    # Placeholder geometry for the macro path, which has no Lattice input.
    A = SMatrix{lattice_dim,lattice_dim,Float64}(I)
    B = SMatrix{lattice_dim,lattice_dim,Float64}(I * 2 * pi)
    r = fill(SVector{lattice_dim}(zeros(Float64, lattice_dim)), builder.d)

    build(builder, A, B, r)
end

"""
    build(builder::HamiltonianBuilder, A, B, r; meta = nothing)

Compile a `HamiltonianBuilder` with explicit geometry: `A`/`B` the primitive
and reciprocal vector matrices, `r` the per-orbital positions.
"""
function build(
    builder::HamiltonianBuilder,
    A::SMatrix,
    B::SMatrix,
    r::Vector{<:SVector};
    meta = nothing,
)
    LatticeHamiltonian(
        A,
        B,
        builder.d,
        builder.L,
        builder.params,
        builder.fields,
        r,
        eval(make_apply(builder)),
        eval(make_sparse(builder)),
        meta,
    )
end
