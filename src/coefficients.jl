import LinearAlgebra: UniformScaling

"""
    Coefficient

Internal representation of a term coefficient. Concrete kinds:

  - [`StaticCoefficient`](@ref): literal numbers and parameter expressions,
    normalized to a `d × d` matrix of entries;
  - [`ScaledIdentity`](@ref): a scalar (literal or parameter expression) times
    the identity, from `x * I`;
  - [`UniformCoefficient`](@ref): `uniform(f)`, position-independent but
    computed by arbitrary code;
  - [`LocalCoefficient`](@ref): a site- or bond-dependent do-block callback.
"""
abstract type Coefficient end

"""
    StaticCoefficient

A `d × d` coefficient whose entries are `ComplexF64` literals or parameter
expressions ([`ParameterRef`](@ref)/[`ParameterExpression`](@ref)). `deps` is
the union of parameter names the entries depend on.
"""
struct StaticCoefficient <: Coefficient
    values::Matrix{Any}
    deps::Set{Symbol}
end

"""
    ScaledIdentity

A scalar multiple of the orbital identity, produced by `x * I` with `x` a
number or parameter expression.
"""
struct ScaledIdentity <: Coefficient
    scalar::Any
    deps::Set{Symbol}
end

ScaledIdentity(scalar::Number) = ScaledIdentity(ComplexF64(scalar), Set{Symbol}())
ScaledIdentity(scalar::ParamLike) = ScaledIdentity(scalar, _pdeps(scalar))

Base.:*(p::ParamLike, J::UniformScaling) = ScaledIdentity(J.λ == 1 ? p : p * J.λ)
Base.:*(J::UniformScaling, p::ParamLike) = p * J

"""
    UniformCoefficient

See [`uniform`](@ref).
"""
struct UniformCoefficient <: Coefficient
    f::Function
    materialize::Bool
    depends_on::Union{Nothing,Tuple{Vararg{Symbol}}}
end

"""
    uniform(f; materialize = false, depends_on = nothing)

Wrap a position-independent callback `f(env)` as a term coefficient. `env` has
`env.parameters` and `env.fields` namespaces. The callback must return a scalar
on a single-orbital lattice and a `d × d` matrix otherwise.

By default the value is re-evaluated once per operator application (`mul!`).
With `materialize = true` it is instead precomputed at [`build`](@ref) time and
refreshed when a dependency is mutated through [`set_parameter!`](@ref) /
[`set_field!`](@ref); `depends_on` narrows the refresh dependencies from the
conservative default (all parameters and declared fields).
"""
uniform(f::Function; materialize::Bool = false, depends_on = nothing) =
    UniformCoefficient(f, materialize, normalize_depends_on(depends_on))

"""
    LocalCoefficient

A site- or bond-dependent callback coefficient, recorded from the do-block
forms of [`onsite!`](@ref) and [`hopping!`](@ref). `returns` is `:scalar` on
single-orbital lattices and `:matrix` otherwise.
"""
struct LocalCoefficient <: Coefficient
    f::Function
    materialize::Bool
    depends_on::Union{Nothing,Tuple{Vararg{Symbol}}}
    returns::Symbol
end

normalize_depends_on(::Nothing) = nothing
normalize_depends_on(name::Symbol) = (name,)
normalize_depends_on(names::Union{Tuple,AbstractVector}) =
    Tuple(Symbol(name) for name in names)

coefficient_deps(c::StaticCoefficient) = c.deps
coefficient_deps(c::ScaledIdentity) = c.deps
coefficient_deps(::Coefficient) = Set{Symbol}()

"""
    normalize_coefficient(coeff, d) -> Coefficient

Normalize a user-supplied coefficient for a `d`-orbital lattice, validating its
shape eagerly. Accepts numbers and parameter expressions (single-orbital only),
`UniformScaling` (`x * I`), `d × d` matrices mixing literals and parameter
expressions, and already-constructed [`Coefficient`](@ref)s.
"""
normalize_coefficient(coeff::Coefficient, d::Int) = coeff

function normalize_coefficient(coeff::Union{Number,ParamLike}, d::Int)
    if d > 1
        error(
            "a bare scalar coefficient is ambiguous on a $d-orbital lattice; " *
            "pass a full $d×$d matrix, or `coeff * I` " *
            "(LinearAlgebra.UniformScaling) for a multiple of the identity",
        )
    end
    entry = coeff isa Number ? ComplexF64(coeff) : coeff
    check_finite_entry(entry)
    values = Matrix{Any}(undef, 1, 1)
    values[1, 1] = entry
    StaticCoefficient(values, _entry_deps(entry))
end

normalize_coefficient(coeff::UniformScaling, d::Int) = ScaledIdentity(coeff.λ)

function normalize_coefficient(coeff::AbstractMatrix, d::Int)
    size(coeff) == (d, d) || error(
        "coefficient matrix must be $d×$d (rows are target/creation orbitals, " *
        "columns are source/annihilation orbitals), got size $(size(coeff))",
    )
    values = Matrix{Any}(undef, d, d)
    deps = Set{Symbol}()
    for j = 1:d, i = 1:d
        entry = coeff[i, j]
        if entry isa Number
            entry = ComplexF64(entry)
        elseif !(entry isa ParamLike)
            error(
                "unsupported coefficient entry $(entry) of type $(typeof(entry)) " *
                "at [$i, $j]; entries must be numbers or parameter expressions",
            )
        end
        check_finite_entry(entry)
        values[i, j] = entry
        union!(deps, _entry_deps(entry))
    end
    StaticCoefficient(values, deps)
end

normalize_coefficient(coeff::Function, d::Int) = error(
    "a bare function is ambiguous as a coefficient; use the do-block form of " *
    "onsite!/hopping! for a site- or bond-dependent callback, or wrap it with " *
    "uniform(f) for a position-independent one",
)

normalize_coefficient(coeff, d::Int) =
    error("unsupported coefficient of type $(typeof(coeff))")

_entry_deps(::ComplexF64) = Set{Symbol}()
_entry_deps(entry::ParamLike) = _pdeps(entry)

check_finite_entry(entry::ComplexF64) =
    isfinite(entry) || error("coefficient entries must be finite, got $entry")
check_finite_entry(::ParamLike) = nothing

"""
    structural_diagonal(coeff, d)

Whether the coefficient can have a nonzero diagonal entry, used to reject
zero-displacement `plus_hc` terms that would double their own diagonal.
Callback coefficients are conservatively assumed to have one.
"""
function structural_diagonal(coeff::StaticCoefficient, d::Int)
    any(1:d) do i
        entry = coeff.values[i, i]
        !(entry isa ComplexF64 && iszero(entry))
    end
end
structural_diagonal(coeff::ScaledIdentity, d::Int) =
    !(coeff.scalar isa ComplexF64 && iszero(coeff.scalar))
structural_diagonal(::Coefficient, d::Int) = true
