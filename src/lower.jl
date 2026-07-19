import LinearAlgebra: pinv

"""
    Periodic()

Periodic boundary conditions on every lattice axis (the only boundary policy
supported by the compiled kernels in this version).
"""
struct Periodic end

"""
    Open()

Open boundaries: bonds whose target lies outside the sample are dropped.
Not yet supported by the compiled kernels; [`build`](@ref) throws.
"""
struct Open end

"""
    Twisted(θ...)

Twisted boundaries: wrapped bonds pick up `exp(i θ ⋅ w)` with `w` the signed
crossing counts. Not yet supported by the compiled kernels; [`build`](@ref)
throws.
"""
struct Twisted{N}
    θ::NTuple{N,Float64}
end
Twisted(θ::Real...) = Twisted(map(Float64, θ))

"""
    CacheRecipe

Refresh recipe for the materialized caches of one term: the generated cache
field names, a `refill!` closure that re-evaluates the coefficient into the
same arrays in place, and the parameter/field names whose mutation triggers a
refresh.
"""
struct CacheRecipe
    names::Vector{Symbol}
    refill!::Function
    parameter_deps::Set{Symbol}
    field_deps::Set{Symbol}
end

"""
    Realization

Build-time metadata stored on `H.meta` by the model path of [`build`](@ref):
the source model, boundary policy, and cache refresh recipes. Enables
[`set_parameter!`](@ref), [`set_field!`](@ref), and
[`refresh_caches!`](@ref).
"""
struct Realization
    model::HamiltonianModel
    boundary::Any
    recipes::Vector{CacheRecipe}
end

# --- lowering: new directed-term IR → legacy HamiltonianBuilder -------------
#
# A legacy `T[δ]` entry `(row, col, value)` is the matrix element
# ⟨n+δ, row| H |n, col⟩, with site-dependent values written in terms of the
# source cell n. A directed term Σ_n c†[n+a, α] M[α, β] c[n, β] therefore
# lowers directly: key δ = +a with the element at (row = α, col = β) and the
# callback anchored at its own cell n. The plus_hc reverse piece lands at key
# -a with (row = β, col = α) and the conjugated value; its source cell is the
# forward bond's target m = n + a, so its callback is read at m - a (wrapped),
# realizing the anchor rule M₋ₐ(m) = Mₐ(m - a)†.

combine_values(a::ComplexF64, b::ComplexF64) = a + b
combine_values(a, b) = Expr(:call, :+, a, b)

iszero_value(value) = value isa ComplexF64 && iszero(value)

function add_entry!(acc, key, row::Int, col::Int, value)
    iszero_value(value) && return nothing
    entries = get!(acc, key) do
        Dict{Tuple{Int,Int},Any}()
    end
    entries[(row, col)] = if haskey(entries, (row, col))
        combine_values(entries[(row, col)], value)
    else
        value
    end
    nothing
end

function add_diagonal!(Vacc::Vector{Any}, orbital::Int, value)
    iszero_value(value) && return nothing
    Vacc[orbital] = if iszero_value(Vacc[orbital])
        value
    else
        combine_values(Vacc[orbital], value)
    end
    nothing
end

lower_value(entry::ComplexF64) = entry
lower_value(entry::ParamLike) = _pexpr(entry)

conj_value(value::ComplexF64) = conj(value)
conj_value(value) = Expr(:call, :conj, value)

"""
    static_elements(coeff, d)

Nonzero `(α, β, entry)` elements of a static coefficient, `α` the target and
`β` the source orbital. Literal zeros are structural and dropped, matching the
legacy parser.
"""
function static_elements(coeff::StaticCoefficient, d::Int)
    elements = Tuple{Int,Int,Any}[]
    for β = 1:d, α = 1:d
        entry = coeff.values[α, β]
        iszero_value(entry isa Number ? ComplexF64(entry) : entry) && continue
        push!(elements, (α, β, entry))
    end
    elements
end

function static_elements(coeff::ScaledIdentity, d::Int)
    scalar = coeff.scalar isa Number ? ComplexF64(coeff.scalar) : coeff.scalar
    iszero_value(scalar) && return Tuple{Int,Int,Any}[]
    Tuple{Int,Int,Any}[(α, α, scalar) for α = 1:d]
end

# Kernel expressions for the wrapped anchor cell of a reverse bond with
# forward displacement `a`, evaluated at the reverse bond's source cell
# `n1, ..., nd` (the forward bond's target): its callback reads at n - a.
function shifted_site_args(a::NTuple{LD,Int}) where {LD}
    map(1:LD) do j
        hop = a[j]
        if hop == 0
            site_loop_var(j)
        else
            :(mod1($(site_loop_var(j)) - $hop, L[$j]))
        end
    end
end

plain_site_args(LD::Int) = Any[site_loop_var(j) for j = 1:LD]

# Wrappers stored in the fields dict; the kernel binds them to type-asserted
# locals, so calls dispatch statically on the concrete closure type.
function make_site_wrapper(
    f::Function,
    env::Environment,
    lattice::Lattice{RD,LD},
    L::NTuple{LD,Int},
    ::Val{D},
) where {RD,LD,D}
    if D == 1
        (args::Vararg{Int,LD}) -> ComplexF64(f(SiteContext{LD,RD}(args, lattice, L), env))
    else
        (args::Vararg{Int,LD}) ->
            SMatrix{D,D,ComplexF64}(f(SiteContext{LD,RD}(args, lattice, L), env))
    end
end

function make_bond_wrapper(
    f::Function,
    env::Environment,
    lattice::Lattice{RD,LD},
    L::NTuple{LD,Int},
    a::NTuple{LD,Int},
    ::Val{D},
) where {RD,LD,D}
    if D == 1
        (args::Vararg{Int,LD}) ->
            ComplexF64(f(BondContext{LD,RD}(a, args, lattice, L), env))
    else
        (args::Vararg{Int,LD}) ->
            SMatrix{D,D,ComplexF64}(f(BondContext{LD,RD}(a, args, lattice, L), env))
    end
end

# Evaluate a local callback at every source cell, as a d × d × L... array.
function evaluate_local(
    f::Function,
    env::Environment,
    lattice::Lattice{RD,LD},
    L::NTuple{LD,Int},
    a::Union{Nothing,NTuple{LD,Int}},
    d::Int,
) where {RD,LD}
    raw = Array{ComplexF64}(undef, d, d, L...)
    for cell in CartesianIndices(L)
        source = Tuple(cell)
        context = if a === nothing
            SiteContext{LD,RD}(source, lattice, L)
        else
            BondContext{LD,RD}(a, source, lattice, L)
        end
        value = f(context, env)
        if d == 1
            raw[1, 1, source...] = ComplexF64(value)
        else
            matrix = SMatrix{d,d,ComplexF64}(value)
            for β = 1:d, α = 1:d
                raw[α, β, source...] = matrix[α, β]
            end
        end
    end
    raw
end

function fill_local_caches!(
    forward::Dict{Tuple{Int,Int},<:AbstractArray},
    reverse::Dict{Tuple{Int,Int},<:AbstractArray},
    raw::Array{ComplexF64},
    a::NTuple{LD,Int},
    L::NTuple{LD,Int},
    term_description::String,
) where {LD}
    for cell in CartesianIndices(L)
        source = Tuple(cell)
        target = mod1.(source .+ a, L)
        for ((α, β), cache) in forward
            store_cache_value!(cache, source, raw[α, β, source...], term_description)
        end
        for ((α, β), cache) in reverse
            store_cache_value!(cache, target, conj(raw[α, β, source...]), term_description)
        end
    end
    nothing
end

function store_cache_value!(
    cache::AbstractArray{Float64},
    index,
    value::ComplexF64,
    term_description::String,
)
    if !iszero(imag(value))
        error(
            "coefficient for term $term_description became complex after a " *
            "parameter or field update but was materialized with real storage; " *
            "rebuild the Hamiltonian",
        )
    end
    cache[index...] = real(value)
    nothing
end

function store_cache_value!(
    cache::AbstractArray{ComplexF64},
    index,
    value::ComplexF64,
    ::String,
)
    cache[index...] = value
    nothing
end

function recipe_deps(depends_on, schema::ParameterSchema, field_names::Set{Symbol})
    if depends_on === nothing
        Set(schema.names), copy(field_names)
    else
        Set(name for name in depends_on if name in Set(schema.names)),
        Set(name for name in depends_on if name in field_names)
    end
end

"""
    lower(model, L, params, all_fields)

Lower a [`HamiltonianModel`](@ref) into the legacy `HamiltonianBuilder` IR.
Generated callback wrappers and materialized cache arrays are inserted into
`all_fields` (the dict that becomes `H.fields`) under `__lh_`-prefixed names.
Returns `(builder, recipes)`.
"""
function lower(
    model::HamiltonianModel{RD,LD},
    L::MVector{LD,Int},
    params::Dict{Symbol,ComplexF64},
    all_fields::Dict{Symbol,Any},
) where {RD,LD}
    lattice = model.lattice
    d = norbitals(lattice)
    Ltuple = NTuple{LD,Int}(L)
    schema = model.parameter_schema
    user_field_names = Set(spec.name for spec in model.field_schema)
    env = Environment(
        ParameterView(params, Set(schema.names)),
        FieldView(all_fields, user_field_names),
    )

    acc = Dict{LatticeVector{LD},Dict{Tuple{Int,Int},Any}}()
    Vacc = Any[zero(ComplexF64) for _ = 1:d]
    recipes = CacheRecipe[]

    for (index, term) in enumerate(model.terms)
        coeff = term.coefficient
        a = term.displacement
        key_forward = LatticeVector{LD}(a)
        key_reverse = -LatticeVector{LD}(a)
        plus_hc = term.adjoint_policy isa PlusHermitianConjugate
        is_onsite = term.kind === :onsite

        emit! = function (α, β, forward_value, reverse_value)
            if is_onsite && α == β
                add_diagonal!(Vacc, α, forward_value)
            else
                add_entry!(acc, key_forward, α, β, forward_value)
            end
            if plus_hc
                add_entry!(acc, key_reverse, β, α, reverse_value)
            end
            nothing
        end

        if coeff isa StaticCoefficient || coeff isa ScaledIdentity
            for (α, β, entry) in static_elements(coeff, d)
                value = lower_value(entry)
                emit!(α, β, value, conj_value(value))
            end
        elseif coeff isa UniformCoefficient
            if coeff.materialize
                lower_materialized_uniform!(
                    emit!,
                    recipes,
                    all_fields,
                    coeff,
                    env,
                    d,
                    index,
                    schema,
                    user_field_names,
                    term_name(term),
                    plus_hc,
                )
            else
                name = Symbol("__lh_u$index")
                all_fields[name] = if d == 1
                    () -> ComplexF64(coeff.f(env))
                else
                    () -> SMatrix{d,d,ComplexF64}(coeff.f(env))
                end
                for β = 1:d, α = 1:d
                    forward = d == 1 ? :($name()) : :(($name())[$α, $β])
                    emit!(α, β, forward, Expr(:call, :conj, forward))
                end
            end
        elseif coeff isa LocalCoefficient
            if coeff.materialize
                lower_materialized_local!(
                    emit!,
                    recipes,
                    all_fields,
                    coeff,
                    env,
                    lattice,
                    Ltuple,
                    is_onsite ? nothing : a,
                    a,
                    d,
                    index,
                    schema,
                    user_field_names,
                    term_name(term),
                    plus_hc,
                )
            else
                name = Symbol("__lh_loc$index")
                all_fields[name] = if is_onsite
                    make_site_wrapper(coeff.f, env, lattice, Ltuple, Val(d))
                else
                    make_bond_wrapper(coeff.f, env, lattice, Ltuple, a, Val(d))
                end
                shifted = shifted_site_args(a)
                plain = plain_site_args(LD)
                for β = 1:d, α = 1:d
                    forward = if d == 1
                        :($name($(plain...)))
                    else
                        :(($name($(plain...)))[$α, $β])
                    end
                    reverse = if d == 1
                        :(conj($name($(shifted...))))
                    else
                        :(conj(($name($(shifted...)))[$α, $β]))
                    end
                    emit!(α, β, forward, reverse)
                end
            end
        else
            error("unsupported coefficient of type $(typeof(coeff))")
        end
    end

    T = Dict{LatticeVector{LD},SparseEntry{LiteralOrSymbolic}}()
    for (key, entries) in acc
        pairs = sort!(collect(entries); by = first)
        rows = Int[pair.first[1] for pair in pairs]
        cols = Int[pair.first[2] for pair in pairs]
        values = LiteralOrSymbolic[pair.second for pair in pairs]
        T[key] = (rows, cols, values)
    end
    V = LiteralOrSymbolic[value for value in Vacc]

    builder = HamiltonianBuilder{LD,LD}(;
        V = V,
        T = T,
        d = d,
        L = L,
        params = params,
        fields = all_fields,
    )
    builder, recipes
end

function lower_materialized_uniform!(
    emit!,
    recipes::Vector{CacheRecipe},
    all_fields::Dict{Symbol,Any},
    coeff::UniformCoefficient,
    env::Environment,
    d::Int,
    index::Int,
    schema::ParameterSchema,
    user_field_names::Set{Symbol},
    term_description::String,
    plus_hc::Bool,
)
    evaluate = if d == 1
        () -> SMatrix{1,1,ComplexF64}(ComplexF64(coeff.f(env)))
    else
        () -> SMatrix{d,d,ComplexF64}(coeff.f(env))
    end
    initial = evaluate()
    scalar_type = all(value -> iszero(imag(value)), initial) ? Float64 : ComplexF64

    forward = Dict{Tuple{Int,Int},Array{scalar_type,0}}()
    reverse = Dict{Tuple{Int,Int},Array{scalar_type,0}}()
    names = Symbol[]
    for β = 1:d, α = 1:d
        fname = Symbol("__lh_c$(index)f_$(α)_$(β)")
        forward[(α, β)] = Array{scalar_type,0}(undef)
        all_fields[fname] = forward[(α, β)]
        push!(names, fname)
        reverse_expr = nothing
        if plus_hc
            rname = Symbol("__lh_c$(index)r_$(α)_$(β)")
            reverse[(α, β)] = Array{scalar_type,0}(undef)
            all_fields[rname] = reverse[(α, β)]
            push!(names, rname)
            reverse_expr = :($rname[])
        end
        emit!(α, β, :($fname[]), reverse_expr)
    end

    refill! = function ()
        matrix = evaluate()
        for β = 1:d, α = 1:d
            store_cache_value!(forward[(α, β)], (), matrix[α, β], term_description)
            if plus_hc
                store_cache_value!(reverse[(α, β)], (), conj(matrix[α, β]), term_description)
            end
        end
        nothing
    end
    refill!()

    parameter_deps, field_deps = recipe_deps(coeff.depends_on, schema, user_field_names)
    push!(recipes, CacheRecipe(names, refill!, parameter_deps, field_deps))
    nothing
end

function lower_materialized_local!(
    emit!,
    recipes::Vector{CacheRecipe},
    all_fields::Dict{Symbol,Any},
    coeff::LocalCoefficient,
    env::Environment,
    lattice::Lattice{RD,LD},
    Ltuple::NTuple{LD,Int},
    context_displacement::Union{Nothing,NTuple{LD,Int}},
    a::NTuple{LD,Int},
    d::Int,
    index::Int,
    schema::ParameterSchema,
    user_field_names::Set{Symbol},
    term_description::String,
    plus_hc::Bool,
) where {RD,LD}
    raw = evaluate_local(coeff.f, env, lattice, Ltuple, context_displacement, d)
    scalar_type = all(value -> iszero(imag(value)), raw) ? Float64 : ComplexF64

    forward = Dict{Tuple{Int,Int},Array{scalar_type,LD}}()
    reverse = Dict{Tuple{Int,Int},Array{scalar_type,LD}}()
    names = Symbol[]
    site = plain_site_args(LD)
    for β = 1:d, α = 1:d
        fname = Symbol("__lh_c$(index)f_$(α)_$(β)")
        forward[(α, β)] = Array{scalar_type,LD}(undef, Ltuple...)
        all_fields[fname] = forward[(α, β)]
        push!(names, fname)
        reverse_ref = nothing
        if plus_hc
            rname = Symbol("__lh_c$(index)r_$(α)_$(β)")
            reverse[(α, β)] = Array{scalar_type,LD}(undef, Ltuple...)
            all_fields[rname] = reverse[(α, β)]
            push!(names, rname)
            reverse_ref = Expr(:ref, rname, site...)
        end
        emit!(α, β, Expr(:ref, fname, site...), reverse_ref)
    end

    refill! = function ()
        updated = evaluate_local(coeff.f, env, lattice, Ltuple, context_displacement, d)
        fill_local_caches!(forward, reverse, updated, a, Ltuple, term_description)
        nothing
    end
    fill_local_caches!(forward, reverse, raw, a, Ltuple, term_description)

    parameter_deps, field_deps = recipe_deps(coeff.depends_on, schema, user_field_names)
    push!(recipes, CacheRecipe(names, refill!, parameter_deps, field_deps))
    nothing
end

function reciprocal_matrix(A::SMatrix{RD,LD,Float64}) where {RD,LD}
    if RD == LD
        SMatrix{RD,LD,Float64}(2π * transpose(inv(A)))
    else
        SMatrix{RD,LD,Float64}(2π * transpose(pinv(Matrix(A))))
    end
end

"""
    build(model::HamiltonianModel, size; boundary = Periodic(),
          parameters = (;), fields = (;))

Realize a model at finite size and compile it into a [`LatticeHamiltonian`](@ref).
`size` is a tuple (or splatted integers) with one extent per lattice
dimension. `parameters` overrides declared defaults; `fields` supplies a
concrete value for every declared field. Only `Periodic()` boundaries are
supported in this version.

The compiled kernels are runtime-generated functions
(RuntimeGeneratedFunctions.jl), so the returned Hamiltonian is immediately
usable — including inside the function that called `build`, with no world-age
restrictions.

!!! note "Precompiled downstream packages"

    A package that precompiles should create Hamiltonians at runtime (inside
    functions, or in `__init__`) rather than storing a *built* Hamiltonian in
    a top-level `const` that gets baked into its precompile image; the kernel
    body cache does not survive that round trip. Storing the *model* as a
    `const` and building from it at runtime is fully supported.
"""
function build(
    model::HamiltonianModel{RD,LD},
    dimensions::Tuple{Vararg{Integer}};
    boundary = Periodic(),
    parameters::NamedTuple = NamedTuple(),
    fields::NamedTuple = NamedTuple(),
) where {RD,LD}
    boundary isa Periodic || throw(
        ArgumentError(
            "$(typeof(boundary)) boundaries require backend support; " *
            "only Periodic() is available in this version",
        ),
    )
    length(dimensions) == LD || error(
        "size $(dimensions) has length $(length(dimensions)); " *
        "the lattice dimension is $LD",
    )
    all(extent -> extent >= 1, dimensions) ||
        error("system size must be positive, got $(dimensions)")
    L = MVector{LD,Int}(dimensions)

    params = copy(model.parameter_schema.defaults)
    for name in keys(parameters)
        haskey(params, name) || error(
            "unknown parameter $name in build override; declared parameters " *
            "are $(Tuple(model.parameter_schema.names))",
        )
        value = parameters[name]
        value isa Number || error("parameter $name must be a number")
        isfinite(ComplexF64(value)) || error("parameter $name must be finite")
        params[name] = ComplexF64(value)
    end

    declared_names = [spec.name for spec in model.field_schema]
    supplied_names = collect(keys(fields))
    for name in declared_names
        name in supplied_names || error("declared field $name was not supplied to build")
    end
    for name in supplied_names
        name in declared_names || error(
            "field $name was not declared in the model; declared fields are " *
            "$(Tuple(declared_names))",
        )
    end
    all_fields = Dict{Symbol,Any}()
    for spec in model.field_schema
        value = fields[spec.name]
        check_field_value(spec, value)
        if value isa AbstractArray && spec.rank == LD
            for axis = 1:LD
                size(value, axis) >= L[axis] || error(
                    "field $(spec.name) is too small along dimension $axis: " *
                    "size $(size(value)) for system size $(Tuple(L))",
                )
            end
        end
        all_fields[spec.name] = value
    end

    for term in model.terms
        a = term.displacement
        for axis = 1:LD
            abs(a[axis]) <= L[axis] || error(
                "displacement $(a) of term $(term_name(term)) exceeds the " *
                "system size $(Tuple(L)) along axis $axis; the compiled " *
                "kernels assume at most single wrapping (|a| ≤ L)",
            )
        end
        if !all(iszero, a) && any(axis -> abs(a[axis]) == L[axis], 1:LD)
            @warn "displacement spans the full system size and wraps onto " *
                  "itself; matrix elements alias" term = term_name(term) L = Tuple(L)
        elseif term.adjoint_policy isa PlusHermitianConjugate &&
               !all(iszero, a) &&
               all(axis -> mod(2 * a[axis], L[axis]) == 0, 1:LD)
            @warn "displacement satisfies a ≡ -a modulo the system size, so " *
                  "forward and Hermitian-conjugate bonds coincide; this " *
                  "often explains unexpected factors of two" term = term_name(term) L =
                Tuple(L)
        end
    end

    model.hermitian ||
        @warn "model is not declared hermitian, but adjoint(H) and eigensolver " *
              "wrappers currently assume Hermiticity"

    # Re-check Hermiticity with the effective (possibly overridden) parameter
    # values; the callback warning already fired at model construction.
    validate_model(model, params, false)

    builder, recipes = lower(model, L, params, all_fields)
    A = model.lattice.A
    build(
        builder,
        A,
        reciprocal_matrix(A),
        copy(model.lattice.positions);
        meta = Realization(model, boundary, recipes),
    )
end

build(model::HamiltonianModel, dimensions::Integer...; kwargs...) =
    build(model, dimensions; kwargs...)
build(model::HamiltonianModel, dimensions::AbstractVector{<:Integer}; kwargs...) =
    build(model, Tuple(dimensions); kwargs...)
