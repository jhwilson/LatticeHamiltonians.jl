"""
    Directed()
    PlusHermitianConjugate()

Adjoint policy of a term: `Directed` terms contribute exactly the written
directed sum; `PlusHermitianConjugate` terms (from `plus_hc = true`) also add
the adjoint of that exact directed term.
"""
struct Directed end
struct PlusHermitianConjugate end
const AdjointPolicy = Union{Directed,PlusHermitianConjugate}

"""
    TermSpec{LD}

One normalized term of a [`HamiltonianModel`](@ref): the directed bond sum
`Σ_n c†[n + displacement, α] M[α, β] c[n, β]` (rows of the coefficient are
target/creation orbitals, columns are source/annihilation orbitals), plus its
adjoint when the policy is `PlusHermitianConjugate`. `kind` is `:onsite` for
terms recorded by [`onsite!`](@ref) (complete zero-displacement matrices) and
`:hopping` otherwise.
"""
struct TermSpec{LD}
    displacement::NTuple{LD,Int}
    coefficient::Coefficient
    adjoint_policy::AdjointPolicy
    kind::Symbol
    label::Union{Nothing,Symbol}
end

struct ParameterSchema
    names::Vector{Symbol}
    defaults::Dict{Symbol,ComplexF64}
end

"""
    HamiltonianModel{RD,LD}

The infinite-lattice physics: a [`Lattice`](@ref), the parameter and field
schemas, and a vector of [`TermSpec`](@ref)s. Created by [`hamiltonian`](@ref)
and realized at a finite size by [`build`](@ref). Inspect with
[`terms`](@ref), [`describe`](@ref), and [`validate`](@ref).
"""
struct HamiltonianModel{RD,LD}
    lattice::Lattice{RD,LD}
    parameter_schema::ParameterSchema
    field_schema::Vector{FieldSpec}
    terms::Vector{TermSpec{LD}}
    hermitian::Bool
end

"""
    ModelBuilder

The `h` argument of the [`hamiltonian`](@ref) do-block; [`onsite!`](@ref) and
[`hopping!`](@ref) record terms on it.
"""
struct ModelBuilder{RD,LD}
    lattice::Lattice{RD,LD}
    parameter_schema::ParameterSchema
    field_schema::Vector{FieldSpec}
    terms::Vector{TermSpec{LD}}
end

"""
    hamiltonian(f, lattice::Lattice; parameters = (;), fields = (), hermitian = true)

Build a [`HamiltonianModel`](@ref) by calling `f(h, p)` where `h` is a
[`ModelBuilder`](@ref) and `p` a checked parameter namespace (`p.t` references
the declared parameter `t`). Typically used with do-block syntax:

```julia
model = hamiltonian(chain; parameters = (t = 1.0, μ = 0.0)) do h, p
    onsite!(h, -p.μ)
    hopping!(h, (1,), -p.t; plus_hc = true)
end
```

`parameters` declares mutable scalar parameters with their default values.
`fields` declares site-dependent data supplied later at [`build`](@ref) time,
as a tuple of names or [`FieldSpec`](@ref)s. `hermitian = true` (the default)
declares the model Hermitian and validates it; pass `false` for directed,
non-Hermitian models (note `adjoint(H)` currently assumes Hermiticity).
"""
function hamiltonian(
    f::Function,
    lattice::Lattice{RD,LD};
    parameters::NamedTuple = NamedTuple(),
    fields = (),
    hermitian::Bool = true,
) where {RD,LD}
    defaults = Dict{Symbol,ComplexF64}()
    names = Symbol[]
    for name in keys(parameters)
        value = parameters[name]
        value isa Number ||
            error("parameter $name must default to a number, got $(typeof(value))")
        isfinite(ComplexF64(value)) ||
            error("parameter $name has a non-finite default $value")
        defaults[name] = ComplexF64(value)
        push!(names, name)
    end

    field_specs = FieldSpec[fieldspec(spec) for spec in fields]
    field_names = [spec.name for spec in field_specs]
    allunique(field_names) || error("field names must be unique, got $(Tuple(field_names))")
    for name in field_names
        haskey(defaults, name) && error("field name $name collides with a parameter name")
        startswith(String(name), "__lh_") &&
            error("field names starting with __lh_ are reserved")
    end
    validate_field_names(field_names, LD)

    builder = ModelBuilder{RD,LD}(
        lattice,
        ParameterSchema(names, defaults),
        field_specs,
        TermSpec{LD}[],
    )
    f(builder, ParameterNamespace(Set(names)))

    model = HamiltonianModel{RD,LD}(
        lattice,
        builder.parameter_schema,
        builder.field_schema,
        builder.terms,
        hermitian,
    )
    validate(model)
    model
end

function normalize_displacement(a, ::Val{LD}) where {LD}
    if a isa Integer
        LD == 1 || error(
            "a bare integer displacement is only valid on a 1-dimensional " *
            "lattice; this lattice has dimension $LD",
        )
        return (Int(a),)
    end
    a isa Union{Tuple,AbstractVector} ||
        error("displacement must be a tuple or vector of integers, got $(typeof(a))")
    length(a) == LD || error(
        "displacement $(Tuple(a)) has length $(length(a)); " *
        "the lattice dimension is $LD",
    )
    all(x -> x isa Integer, a) ||
        error("displacement components must be integers, got $(Tuple(a))")
    NTuple{LD,Int}(a)
end

function check_depends_on(builder::ModelBuilder, depends_on)
    depends_on === nothing && return nothing
    declared = union(
        Set(builder.parameter_schema.names),
        Set(spec.name for spec in builder.field_schema),
    )
    for name in depends_on
        name in declared || error(
            "depends_on names must be declared parameters or fields; " * "$name is neither",
        )
    end
    nothing
end

function record_term!(
    builder::ModelBuilder{RD,LD},
    displacement::NTuple{LD,Int},
    coefficient::Coefficient,
    policy::AdjointPolicy,
    kind::Symbol,
    label,
) where {RD,LD}
    label === nothing || label isa Symbol || error("term labels must be Symbols")
    if policy isa PlusHermitianConjugate && all(iszero, displacement)
        d = norbitals(builder.lattice)
        if structural_diagonal(coefficient, d)
            hint = if coefficient isa Union{UniformCoefficient,LocalCoefficient}
                "the diagonal of a callback coefficient cannot be checked, so " *
                "zero-displacement plus_hc callback terms are not allowed; use " *
                "onsite! with the complete matrix instead"
            else
                "on the diagonal the bond is its own reverse, so plus_hc would " *
                "double it; use onsite! for the complete onsite matrix, or pass " *
                "a matrix with a structurally zero diagonal"
            end
            error("invalid zero-displacement plus_hc term: " * hint)
        end
    end
    push!(builder.terms, TermSpec{LD}(displacement, coefficient, policy, kind, label))
    builder
end

"""
    onsite!(h, coeff; label = nothing)
    onsite!(h; materialize = false, depends_on = nothing, label = nothing) do site, env

Add the complete zero-displacement matrix `Σ_n c†[n, α] M[α, β] c[n, β]`.
`M` is used as written — no Hermitian conjugate is added (its Hermiticity is
validated when the model is declared Hermitian). `coeff` may be a scalar
(single-orbital lattices only), `x * I`, a `d × d` matrix of numbers and
parameter expressions, or [`uniform`](@ref)`(f)`. The do-block form receives a
[`SiteContext`](@ref) and [`Environment`](@ref) per site; `materialize = true`
precomputes its values into an array refreshed via [`set_parameter!`](@ref) /
[`set_field!`](@ref) / [`refresh_caches!`](@ref).
"""
function onsite!(builder::ModelBuilder{RD,LD}, coeff; label = nothing) where {RD,LD}
    coefficient = normalize_coefficient(coeff, norbitals(builder.lattice))
    zero_displacement = ntuple(_ -> 0, Val(LD))
    record_term!(builder, zero_displacement, coefficient, Directed(), :onsite, label)
end

function onsite!(
    f::Function,
    builder::ModelBuilder{RD,LD};
    materialize::Bool = false,
    depends_on = nothing,
    label = nothing,
) where {RD,LD}
    depends = normalize_depends_on(depends_on)
    check_depends_on(builder, depends)
    returns = norbitals(builder.lattice) == 1 ? :scalar : :matrix
    coefficient = LocalCoefficient(f, materialize, depends, returns)
    zero_displacement = ntuple(_ -> 0, Val(LD))
    record_term!(builder, zero_displacement, coefficient, Directed(), :onsite, label)
end

"""
    hopping!(h, a, coeff; plus_hc = false, label = nothing)
    hopping!(h, a; plus_hc = false, materialize = false,
             depends_on = nothing, label = nothing) do bond, env

Add the directed bond term `Σ_n c†[n + a, α] M[α, β] c[n, β]`: `n` is the
source cell, `n + a` the target, row `α` of `M` the target/creation orbital
and column `β` the source/annihilation orbital. With `plus_hc = true` the
adjoint of this exact directed term is added as well.

`coeff` may be a scalar (single-orbital lattices only), `x * I`, a `d × d`
matrix of numbers and parameter expressions, or [`uniform`](@ref)`(f)`. The
do-block form receives a [`BondContext`](@ref) (describing the forward bond)
and an [`Environment`](@ref); with `plus_hc = true` the reverse bond reuses
the forward callback via `M₋ₐ(m) = Mₐ(m - a)†`, so no shifted field arrays
are needed. `materialize = true` precomputes the callback into arrays
refreshed on dependency updates.
"""
function hopping!(
    builder::ModelBuilder{RD,LD},
    a,
    coeff;
    plus_hc::Bool = false,
    label = nothing,
) where {RD,LD}
    displacement = normalize_displacement(a, Val(LD))
    coefficient = normalize_coefficient(coeff, norbitals(builder.lattice))
    policy = plus_hc ? PlusHermitianConjugate() : Directed()
    record_term!(builder, displacement, coefficient, policy, :hopping, label)
end

function hopping!(
    f::Function,
    builder::ModelBuilder{RD,LD},
    a;
    plus_hc::Bool = false,
    materialize::Bool = false,
    depends_on = nothing,
    label = nothing,
) where {RD,LD}
    displacement = normalize_displacement(a, Val(LD))
    depends = normalize_depends_on(depends_on)
    check_depends_on(builder, depends)
    returns = norbitals(builder.lattice) == 1 ? :scalar : :matrix
    coefficient = LocalCoefficient(f, materialize, depends, returns)
    policy = plus_hc ? PlusHermitianConjugate() : Directed()
    record_term!(builder, displacement, coefficient, policy, :hopping, label)
end

"""
    terms(model::HamiltonianModel)

The vector of normalized [`TermSpec`](@ref)s of the model.
"""
terms(model::HamiltonianModel) = model.terms

_entry_value(entry::ComplexF64, defaults::Dict{Symbol,ComplexF64}) = entry
_entry_value(entry::ParamLike, defaults::Dict{Symbol,ComplexF64}) = _peval(entry)(defaults)

function numeric_matrix(coeff::StaticCoefficient, d::Int, defaults::Dict{Symbol,ComplexF64})
    ComplexF64[_entry_value(coeff.values[i, j], defaults) for i = 1:d, j = 1:d]
end

function numeric_matrix(coeff::ScaledIdentity, d::Int, defaults::Dict{Symbol,ComplexF64})
    scalar = _entry_value(
        coeff.scalar isa Number ? ComplexF64(coeff.scalar) : coeff.scalar,
        defaults,
    )
    ComplexF64[i == j ? scalar : zero(ComplexF64) for i = 1:d, j = 1:d]
end

term_name(term::TermSpec) = if term.label === nothing
    "at displacement $(term.displacement)"
else
    ":$(term.label) at displacement $(term.displacement)"
end

"""
    validate(model::HamiltonianModel)

Run cross-term model checks, throwing on failure. When the model is declared
Hermitian, static terms are checked numerically at the default parameter
values: directed contributions must satisfy `M(-a) = M(a)†` and the combined
zero-displacement matrix must be Hermitian. Callback coefficients cannot be
checked structurally and produce an advisory warning instead (verify with
`ishermitian(Matrix(sparse(H)))` on a small system).
"""
validate(model::HamiltonianModel) =
    validate_model(model, model.parameter_schema.defaults, true)

function validate_model(
    model::HamiltonianModel{RD,LD},
    params::Dict{Symbol,ComplexF64},
    warn_unverifiable::Bool,
) where {RD,LD}
    labels = [term.label for term in model.terms if term.label !== nothing]
    if warn_unverifiable
        allunique(labels) || @warn "duplicate term labels" labels = Tuple(labels)
    end

    model.hermitian || return nothing

    d = norbitals(model.lattice)
    defaults = params
    accumulated = Dict{NTuple{LD,Int},Matrix{ComplexF64}}()
    unverifiable = String[]
    for term in model.terms
        coeff = term.coefficient
        if coeff isa StaticCoefficient || coeff isa ScaledIdentity
            # plus_hc static terms are Hermitian by construction; only the
            # directed parts constrain the check.
            term.adjoint_policy isa Directed || continue
            matrix = numeric_matrix(coeff, d, defaults)
            key = term.displacement
            accumulated[key] = haskey(accumulated, key) ? accumulated[key] + matrix : matrix
        elseif term.adjoint_policy isa Directed
            push!(unverifiable, term_name(term))
        end
    end
    if warn_unverifiable && !isempty(unverifiable)
        @warn "cannot verify Hermiticity of directed callback terms; " *
              "verify with ishermitian(Matrix(sparse(H))) on a small system" terms =
            Tuple(unverifiable)
    end

    zero_displacement = ntuple(_ -> 0, Val(LD))
    for (key, matrix) in accumulated
        reverse_key = map(-, key)
        reverse_matrix = get(accumulated, reverse_key) do
            zeros(ComplexF64, d, d)
        end
        deviation = maximum(abs.(reverse_matrix - matrix'))
        scale = 1 + maximum(abs.(matrix))
        if deviation > 1e-12 * scale
            description = if key == zero_displacement
                "the combined zero-displacement matrix is not Hermitian"
            else
                "directed terms at displacement $key do not satisfy M(-a) = M(a)†"
            end
            error(
                "model is declared hermitian = true but $description at the " *
                "supplied parameter values (deviation $deviation); add the " *
                "reverse term, use plus_hc = true, or declare hermitian = false",
            )
        end
    end
    nothing
end

describe_coefficient(coeff::StaticCoefficient) = if isempty(coeff.deps)
    "matrix of constants"
else
    "matrix, parameters $(Tuple(sort!(collect(coeff.deps))))"
end
describe_coefficient(coeff::ScaledIdentity) = if isempty(coeff.deps)
    "($(coeff.scalar)) * I"
else
    "(parameter expression, parameters $(Tuple(sort!(collect(coeff.deps))))) * I"
end
function describe_coefficient(coeff::UniformCoefficient)
    suffix = coeff.materialize ? ", materialized" : ""
    deps = coeff.depends_on === nothing ? "" : ", depends on $(coeff.depends_on)"
    "uniform callback" * suffix * deps
end
function describe_coefficient(coeff::LocalCoefficient)
    suffix = coeff.materialize ? ", materialized" : ""
    deps = coeff.depends_on === nothing ? "" : ", depends on $(coeff.depends_on)"
    "local callback" * suffix * deps
end

"""
    describe([io::IO], model::HamiltonianModel)

Print the mathematical interpretation of each term: the directed sum
`Σ_n c†[n+a, α] M[α, β] c[n, β]`, whether `+ h.c.` is added, and the kind and
parameter dependencies of its coefficient.
"""
describe(model::HamiltonianModel) = describe(stdout, model)

function describe(io::IO, model::HamiltonianModel{RD,LD}) where {RD,LD}
    lattice = model.lattice
    println(
        io,
        "HamiltonianModel: $(LD)D lattice, $(norbitals(lattice)) orbital" *
        (norbitals(lattice) == 1 ? "" : "s") *
        " $(Tuple(lattice.orbitals)), " *
        (model.hermitian ? "hermitian" : "not declared hermitian"),
    )
    schema = model.parameter_schema
    if !isempty(schema.names)
        pairs = join(("$name = $(schema.defaults[name])" for name in schema.names), ", ")
        println(io, "  parameters: $pairs")
    end
    if !isempty(model.field_schema)
        entries = join(
            (
                begin
                    extras = String[]
                    spec.rank === nothing || push!(extras, "rank $(spec.rank)")
                    spec.eltype === nothing || push!(extras, "eltype $(spec.eltype)")
                    if isempty(extras)
                        "$(spec.name)"
                    else
                        "$(spec.name) ($(join(extras, ", ")))"
                    end
                end for spec in model.field_schema
            ),
            ", ",
        )
        println(io, "  fields:     $entries")
    end
    println(io, "  terms:")
    for (index, term) in enumerate(model.terms)
        label = term.label === nothing ? "(unlabeled)" : ":$(term.label)"
        cell = all(iszero, term.displacement) ? "n" : "n+$(term.displacement)"
        hc = term.adjoint_policy isa PlusHermitianConjugate ? " + h.c." : ""
        println(io, "    [$index] $label  Σ_n c†[$cell,α] M[α,β] c[n,β]$hc")
        println(io, "        coefficient: $(describe_coefficient(term.coefficient))")
    end
    println(
        io,
        "  convention: M[α,β] multiplies c†[target,α] c[source,β]; " *
        "rows are target orbitals, columns are source orbitals",
    )
    nothing
end

function Base.show(io::IO, model::HamiltonianModel{RD,LD}) where {RD,LD}
    print(
        io,
        "HamiltonianModel{$RD,$LD}: $(norbitals(model.lattice)) orbital(s), " *
        "$(length(model.terms)) term(s); use describe(model) for details",
    )
end
