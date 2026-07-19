"""
    Lattice{RD,LD}

Geometry of an infinite lattice: primitive vectors, named orbitals, and the
position of each orbital inside the unit cell.

`RD` is the real-space dimension and `LD` the lattice dimension (number of
primitive vectors). Logical unit cells are addressed by zero-based integer
coordinates `n`, and the physical position of orbital `β` in cell `n` is
`r(n, β) = A * n + positions[β]`.

# Fields

  - `A::SMatrix{RD,LD,Float64}`: matrix whose columns are the primitive vectors
  - `orbitals::Vector{Symbol}`: ordered, unique orbital names
  - `positions::Vector{SVector{RD,Float64}}`: one position per orbital
"""
struct Lattice{RD,LD}
    A::SMatrix{RD,LD,Float64}
    orbitals::Vector{Symbol}
    positions::Vector{SVector{RD,Float64}}

    function Lattice{RD,LD}(
        A::SMatrix{RD,LD,Float64},
        orbitals::Vector{Symbol},
        positions::Vector{SVector{RD,Float64}},
    ) where {RD,LD}
        LD >= 1 || error("lattice dimension must be at least 1")
        isempty(orbitals) && error("a Lattice needs at least one orbital")
        allunique(orbitals) || error("orbital names must be unique, got $(Tuple(orbitals))")
        length(positions) == length(orbitals) || error(
            "got $(length(positions)) positions for $(length(orbitals)) orbitals; " *
            "declare exactly one position per orbital",
        )
        all(isfinite, A) || error("primitive vectors must have finite entries")
        for (name, τ) in zip(orbitals, positions)
            all(isfinite, τ) || error("position of orbital $name must be finite")
        end
        new{RD,LD}(A, orbitals, positions)
    end
end

"""
    Lattice(A::AbstractMatrix; orbitals = (:orb,), positions = nothing)

Construct a [`Lattice`](@ref) from a `real_dim × lattice_dim` matrix whose
columns are the primitive vectors. `orbitals` is a tuple or vector of unique
`Symbol` names; `positions` is a matching collection of real-space positions
(each a vector of length `real_dim`), defaulting to all orbitals at the origin.

# Examples

```julia
chain = Lattice(reshape([1.0], 1, 1))
graphene = Lattice(
    [1/2 -1/2; sqrt(3)/2 sqrt(3)/2];
    orbitals = (:A, :B),
    positions = ([0.0, 0.0], [0.0, 1 / sqrt(3)]),
)
```
"""
function Lattice(A::AbstractMatrix; orbitals = (:orb,), positions = nothing)
    RD, LD = size(A)
    names = Symbol[Symbol(name) for name in orbitals]
    τs = if positions === nothing
        [zero(SVector{RD,Float64}) for _ in names]
    else
        positions isa Union{Tuple,AbstractVector} ||
            error("positions must be a tuple or vector of per-orbital positions")
        map(collect(positions)) do τ
            length(τ) == RD || error(
                "each orbital position must have length $RD " *
                "(the real-space dimension of A); got $τ",
            )
            SVector{RD,Float64}(τ)
        end
    end
    Lattice{RD,LD}(SMatrix{RD,LD,Float64}(A), names, τs)
end

"""
    norbitals(lattice::Lattice)

Number of orbitals per unit cell.
"""
norbitals(lattice::Lattice) = length(lattice.orbitals)

"""
    orbital_index(lattice::Lattice, name::Symbol)

Index of the orbital `name` in the lattice's declared orbital order.
"""
function orbital_index(lattice::Lattice, name::Symbol)
    index = findfirst(isequal(name), lattice.orbitals)
    index === nothing &&
        error("unknown orbital $name; declared orbitals are $(Tuple(lattice.orbitals))")
    index
end

orbital_position(lattice::Lattice, name::Symbol) =
    lattice.positions[orbital_index(lattice, name)]

function Base.show(io::IO, lattice::Lattice{RD,LD}) where {RD,LD}
    print(
        io,
        "Lattice{$RD,$LD} with $(norbitals(lattice)) orbital",
        norbitals(lattice) == 1 ? "" : "s",
        " ",
        Tuple(lattice.orbitals),
    )
end
