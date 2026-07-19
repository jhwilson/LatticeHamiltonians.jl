"""
    Source()

Endpoint marker for the source (annihilation) end of a bond in geometry
helpers, e.g. `position(bond, Source(), :A)`.
"""
struct Source end

"""
    Target()

Endpoint marker for the target (creation) end of a bond in geometry helpers,
e.g. `position(bond, Target(), :A)`.
"""
struct Target end

"""
    SiteContext

Passed to `onsite!` do-block callbacks. Properties:

  - `cell_index::NTuple{LD,Int}`: 1-based finite-array index of the cell
    (use directly for field lookups, `env.fields.W[site.cell_index...]`);
  - `cell::NTuple{LD,Int}`: zero-based logical cell coordinate
    (`r = A * cell + τ`);
  - `lattice`, `L`.
"""
struct SiteContext{LD,RD}
    cell_index::NTuple{LD,Int}
    lattice::Lattice{RD,LD}
    L::NTuple{LD,Int}
end

function Base.getproperty(site::SiteContext, name::Symbol)
    if name === :cell
        return getfield(site, :cell_index) .- 1
    end
    getfield(site, name)
end

Base.propertynames(::SiteContext) = (:cell_index, :cell, :lattice, :L)

"""
    BondContext

Passed to `hopping!` do-block callbacks, describing the *forward* directed bond
`c†[n + a] ⋯ c[n]`. Properties:

  - `displacement::NTuple{LD,Int}`: the hop `a` in lattice coordinates;
  - `source_cell_index::NTuple{LD,Int}`: 1-based finite index of the source
    cell `n` (the canonical representative in `1:L`);
  - `source_cell::NTuple{LD,Int}`: zero-based logical coordinate of `n`;
  - `target_cell::NTuple{LD,Int}`: `source_cell .+ displacement`, *not*
    wrapped — relative bond geometry is always physically correct;
  - `target_cell_index::NTuple{LD,Int}`: boundary-wrapped 1-based index of the
    target cell;
  - `crossed_boundary::NTuple{LD,Int}`: signed per-axis boundary winding
    counts (zero in the bulk);
  - `lattice`, `L`.
"""
struct BondContext{LD,RD}
    displacement::NTuple{LD,Int}
    source_cell_index::NTuple{LD,Int}
    lattice::Lattice{RD,LD}
    L::NTuple{LD,Int}
end

function Base.getproperty(bond::BondContext, name::Symbol)
    if name === :source_cell
        return getfield(bond, :source_cell_index) .- 1
    elseif name === :target_cell
        return getfield(bond, :source_cell_index) .- 1 .+ getfield(bond, :displacement)
    elseif name === :target_cell_index
        return mod1.(
            getfield(bond, :source_cell_index) .+ getfield(bond, :displacement),
            getfield(bond, :L),
        )
    elseif name === :crossed_boundary
        return fld.(
            getfield(bond, :source_cell_index) .- 1 .+ getfield(bond, :displacement),
            getfield(bond, :L),
        )
    end
    getfield(bond, name)
end

Base.propertynames(::BondContext) = (
    :displacement,
    :source_cell_index,
    :source_cell,
    :target_cell,
    :target_cell_index,
    :crossed_boundary,
    :lattice,
    :L,
)

_cell_position(lattice::Lattice, cell, orbital::Symbol) =
    lattice.A * SVector(cell) + orbital_position(lattice, orbital)

"""
    position(site::SiteContext, orbital::Symbol)
    position(bond::BondContext, Source(), orbital::Symbol)
    position(bond::BondContext, Target(), orbital::Symbol)

Real-space position `A * cell + τ_orbital` of an orbital at a context's cell.
The target endpoint uses the unwrapped `target_cell`, so bond geometry stays
correct across periodic boundaries.
"""
Base.position(site::SiteContext, orbital::Symbol) =
    _cell_position(site.lattice, site.cell, orbital)
Base.position(bond::BondContext, ::Source, orbital::Symbol) =
    _cell_position(bond.lattice, bond.source_cell, orbital)
Base.position(bond::BondContext, ::Target, orbital::Symbol) =
    _cell_position(bond.lattice, bond.target_cell, orbital)

"""
    displacement(bond::BondContext, target_orbital::Symbol, source_orbital::Symbol)

Real-space displacement from the source orbital to the target orbital of the
bond: `position(bond, Target(), target_orbital) - position(bond, Source(), source_orbital)`.
"""
displacement(bond::BondContext, target_orbital::Symbol, source_orbital::Symbol) =
    position(bond, Target(), target_orbital) - position(bond, Source(), source_orbital)

"""
    midpoint(bond::BondContext, target_orbital::Symbol, source_orbital::Symbol)

Real-space midpoint of the bond between the given orbitals.
"""
midpoint(bond::BondContext, target_orbital::Symbol, source_orbital::Symbol) =
    (position(bond, Target(), target_orbital) + position(bond, Source(), source_orbital)) /
    2

"""
    ParameterView

Checked read-only view of the parameter dictionary, exposed to callbacks as
`env.parameters`. Property access returns the current `ComplexF64` value.
"""
struct ParameterView
    store::Dict{Symbol,ComplexF64}
    names::Set{Symbol}
end

function Base.getproperty(view::ParameterView, name::Symbol)
    names = getfield(view, :names)
    name in names || error(
        "unknown parameter $name; declared parameters are $(Tuple(sort!(collect(names))))",
    )
    getfield(view, :store)[name]
end

Base.propertynames(view::ParameterView) = Tuple(sort!(collect(getfield(view, :names))))

"""
    FieldView

Checked read-only view of the declared fields, exposed to callbacks as
`env.fields`. Property access returns the current field value.
"""
struct FieldView
    store::Dict{Symbol,Any}
    names::Set{Symbol}
end

function Base.getproperty(view::FieldView, name::Symbol)
    names = getfield(view, :names)
    name in names ||
        error("unknown field $name; declared fields are $(Tuple(sort!(collect(names))))")
    getfield(view, :store)[name]
end

Base.propertynames(view::FieldView) = Tuple(sort!(collect(getfield(view, :names))))

"""
    Environment

The `env` argument of coefficient callbacks, with separate `parameters` and
`fields` namespaces. Both views read the live dictionaries stored on the built
Hamiltonian, so `set_parameter!`/`set_field!` updates are visible without a
rebuild.
"""
struct Environment
    parameters::ParameterView
    fields::FieldView
end
