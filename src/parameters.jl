"""
    ParameterRef

A typed reference to a declared model parameter, produced by property access on
the parameter namespace passed to the [`hamiltonian`](@ref) do-block (`p.t`).
Arithmetic on references builds a [`ParameterExpression`](@ref).
"""
struct ParameterRef
    name::Symbol
end

"""
    ParameterExpression

A small symbolic expression over parameter references, built by operator
overloading (`-p.t`, `p.t * conj(p.u)`, ...). Carries three synchronized
representations: the backend expression (`params[:t]`-style lookups), an
evaluator closure used for validation and cache refresh, and the set of
parameter names it depends on.
"""
struct ParameterExpression
    expr::Union{Expr,ComplexF64}
    evalf::Function
    deps::Set{Symbol}
end

const ParamLike = Union{ParameterRef,ParameterExpression}

_pexpr(p::ParameterRef) = :(params[$(QuoteNode(p.name))])
_pexpr(p::ParameterExpression) = p.expr
_pexpr(x::Number) = ComplexF64(x)

_peval(p::ParameterRef) = params -> params[p.name]
_peval(p::ParameterExpression) = p.evalf
function _peval(x::Number)
    value = ComplexF64(x)
    _ -> value
end

_pdeps(p::ParameterRef) = Set{Symbol}((p.name,))
_pdeps(p::ParameterExpression) = p.deps
_pdeps(::Number) = Set{Symbol}()

function _pcombine(f, opname::Symbol, args...)
    exprs = map(_pexpr, args)
    evals = map(_peval, args)
    deps = mapreduce(_pdeps, union!, args; init = Set{Symbol}())
    evalf = params -> ComplexF64(f(map(g -> g(params), evals)...))
    ParameterExpression(Expr(:call, opname, exprs...), evalf, deps)
end

for op in (:+, :-, :*, :/, :^)
    @eval begin
        Base.$op(a::ParamLike, b::ParamLike) = _pcombine($op, $(QuoteNode(op)), a, b)
        Base.$op(a::ParamLike, b::Number) = _pcombine($op, $(QuoteNode(op)), a, b)
        Base.$op(a::Number, b::ParamLike) = _pcombine($op, $(QuoteNode(op)), a, b)
    end
end

# The documented set of scalar functions usable in parameter expressions.
for fn in (:-, :conj, :exp, :sqrt, :cos, :sin, :abs)
    @eval Base.$fn(a::ParamLike) = _pcombine($fn, $(QuoteNode(fn)), a)
end

"""
    ParameterNamespace

The `p` argument of the [`hamiltonian`](@ref) do-block. Property access
(`p.t`) returns a checked [`ParameterRef`](@ref); unknown names throw.
"""
struct ParameterNamespace
    names::Set{Symbol}
end

function Base.getproperty(p::ParameterNamespace, name::Symbol)
    names = getfield(p, :names)
    name in names || error(
        "unknown parameter $name; declared parameters are $(Tuple(sort!(collect(names))))",
    )
    ParameterRef(name)
end

Base.propertynames(p::ParameterNamespace) = Tuple(sort!(collect(getfield(p, :names))))

"""
    FieldSpec(name::Symbol; rank = nothing, eltype = nothing)

Schema entry for a declared field. `rank` and `eltype`, when given, are
validated against the concrete field supplied at [`build`](@ref) time (and on
[`set_field!`](@ref)). A bare `Symbol` in the `fields` declaration is
equivalent to a name-only `FieldSpec`.
"""
struct FieldSpec
    name::Symbol
    rank::Union{Nothing,Int}
    eltype::Union{Nothing,Type}
end

FieldSpec(name::Symbol; rank = nothing, eltype = nothing) = FieldSpec(name, rank, eltype)

fieldspec(spec::FieldSpec) = spec
fieldspec(name::Symbol) = FieldSpec(name)
fieldspec(other) =
    error("field declarations must be Symbols or FieldSpecs, got $(typeof(other))")

"""
    check_field_value(spec::FieldSpec, value)

Validate a concrete field value against its schema entry. Array fields check
`rank` against `ndims` and `eltype` against the array element type; function
fields skip both checks unless `rank`/`eltype` are declared, in which case a
non-array value is rejected.
"""
function check_field_value(spec::FieldSpec, value)
    spec.rank === nothing && spec.eltype === nothing && return nothing
    if value isa AbstractArray
        spec.rank === nothing ||
            ndims(value) == spec.rank ||
            error(
                "field $(spec.name) must have rank $(spec.rank), " *
                "got a $(ndims(value))-dimensional array",
            )
        spec.eltype === nothing ||
            Base.eltype(value) <: spec.eltype ||
            error(
                "field $(spec.name) must have element type $(spec.eltype), " *
                "got $(Base.eltype(value))",
            )
    else
        error(
            "field $(spec.name) declares rank/eltype and therefore must be " *
            "an array, got $(typeof(value))",
        )
    end
    nothing
end
