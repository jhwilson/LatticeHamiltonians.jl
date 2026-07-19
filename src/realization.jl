function model_realization(H::LatticeHamiltonian, operation::String)
    meta = H.meta
    meta isa Realization || error(
        "$operation requires a Hamiltonian built from a HamiltonianModel; " *
        "for macro-built Hamiltonians mutate H.params / H.fields directly",
    )
    meta
end

function refresh_recipes!(H::LatticeHamiltonian, mutated::Set{Symbol})
    realization = H.meta
    realization isa Realization || return nothing
    for recipe in realization.recipes
        if !isempty(intersect(recipe.parameter_deps, mutated)) ||
           !isempty(intersect(recipe.field_deps, mutated))
            recipe.refill!()
        end
    end
    nothing
end

"""
    set_parameter!(H::LatticeHamiltonian; name = value, ...)

Checked parameter update for a Hamiltonian built from a
[`HamiltonianModel`](@ref): validates the names against the declared schema,
converts the values to `ComplexF64`, and refreshes any materialized caches
that depend on the mutated parameters. Returns `H`.
"""
function set_parameter!(H::LatticeHamiltonian; kwargs...)
    realization = model_realization(H, "set_parameter!")
    schema = realization.model.parameter_schema
    for (name, value) in kwargs
        name in schema.names ||
            error("unknown parameter $name; declared parameters are $(Tuple(schema.names))")
        value isa Number ||
            error("parameter $name must be set to a number, got $(typeof(value))")
        isfinite(ComplexF64(value)) || error("parameter $name must be finite, got $value")
        H.params[name] = ComplexF64(value)
    end
    refresh_recipes!(H, Set{Symbol}(keys(kwargs)))
    H
end

"""
    set_field!(H::LatticeHamiltonian; name = value, ...)

Checked field replacement for a Hamiltonian built from a
[`HamiltonianModel`](@ref). The replacement must have exactly the type the
kernel was compiled against (the type of the current value); a different type
requires a rebuild. Refreshes materialized caches that depend on the mutated
fields. Returns `H`.

For in-place mutation of a field array (e.g. `fill!` or elementwise writes),
no call is needed for per-apply coefficients, but materialized caches must be
refreshed explicitly with [`refresh_caches!`](@ref).
"""
function set_field!(H::LatticeHamiltonian; kwargs...)
    realization = model_realization(H, "set_field!")
    specs = Dict(spec.name => spec for spec in realization.model.field_schema)
    for (name, value) in kwargs
        haskey(specs, name) ||
            error("unknown field $name; declared fields are $(Tuple(keys(specs)))")
        current = H.fields[name]
        if typeof(value) !== typeof(current)
            error(
                "replacing field $name with a $(typeof(value)) requires a " *
                "rebuild; the compiled kernel expects $(typeof(current))",
            )
        end
        check_field_value(specs[name], value)
        if value isa AbstractArray
            for axis = 1:length(H.L)
                ndims(value) >= axis || break
                size(value, axis) >= H.L[axis] || error(
                    "field $name is too small along dimension $axis: " *
                    "size $(size(value)) for system size $(Tuple(H.L))",
                )
            end
        end
        H.fields[name] = value
    end
    refresh_recipes!(H, Set{Symbol}(keys(kwargs)))
    H
end

"""
    refresh_caches!(H::LatticeHamiltonian)
    refresh_caches!(H::LatticeHamiltonian, names::Symbol...)

Re-evaluate materialized coefficient caches in place. With no `names`, every
recipe is refreshed — use this after mutating a field array in place, which
the dependency tracking of [`set_field!`](@ref) cannot observe. With `names`,
only recipes depending on the given parameter or field names are refreshed.
Returns `H`.
"""
function refresh_caches!(H::LatticeHamiltonian)
    realization = model_realization(H, "refresh_caches!")
    for recipe in realization.recipes
        recipe.refill!()
    end
    H
end

function refresh_caches!(H::LatticeHamiltonian, names::Symbol...)
    model_realization(H, "refresh_caches!")
    refresh_recipes!(H, Set{Symbol}(names))
    H
end

"""
    parameters(H::LatticeHamiltonian)

Current parameter values as a `NamedTuple` in declaration order.
"""
function parameters(H::LatticeHamiltonian)
    realization = model_realization(H, "parameters")
    names = realization.model.parameter_schema.names
    NamedTuple{Tuple(names)}(Tuple(H.params[name] for name in names))
end

"""
    fields(H::LatticeHamiltonian)

Current values of the *declared* fields as a `NamedTuple` (generated callback
wrappers and materialized caches are not included).
"""
function fields(H::LatticeHamiltonian)
    realization = model_realization(H, "fields")
    names = [spec.name for spec in realization.model.field_schema]
    NamedTuple{Tuple(names)}(Tuple(H.fields[name] for name in names))
end
