using LatticeHamiltonians:
    ParameterRef,
    ParameterExpression,
    ParameterNamespace,
    StaticCoefficient,
    ScaledIdentity,
    UniformCoefficient,
    LocalCoefficient,
    normalize_coefficient,
    normalize_displacement,
    fieldspec,
    check_field_value,
    orbital_index,
    norbitals

@testset "Frontend types" begin
    @testset "Lattice" begin
        chain = Lattice(reshape([1.0], 1, 1))
        @test norbitals(chain) == 1
        @test chain.orbitals == [:orb]
        @test chain.positions[1] == [0.0]

        graphene = Lattice(
            [1/2 -1/2; sqrt(3)/2 sqrt(3)/2];
            orbitals = (:A, :B),
            positions = ([0.0, 0.0], [0.0, 1/sqrt(3)]),
        )
        @test norbitals(graphene) == 2
        @test orbital_index(graphene, :B) == 2
        @test graphene.positions[2] == [0.0, 1/sqrt(3)]
        @test_throws ErrorException orbital_index(graphene, :C)

        @test_throws ErrorException Lattice(reshape([1.0], 1, 1); orbitals = (:A, :A))
        @test_throws ErrorException Lattice(
            reshape([1.0], 1, 1);
            orbitals = (:A, :B),
            positions = ([0.0],),
        )
        @test_throws ErrorException Lattice(
            reshape([1.0], 1, 1);
            orbitals = (:A,),
            positions = ([0.0, 0.0],),
        )
        @test_throws ErrorException Lattice(reshape([NaN], 1, 1))
    end

    @testset "Parameter references" begin
        p = ParameterNamespace(Set([:t, :u]))
        @test p.t isa ParameterRef
        @test p.t.name === :t
        @test_throws ErrorException p.wrong

        expression = p.t * 2 - conj(p.u)
        @test expression isa ParameterExpression
        @test expression.expr == Expr(
            :call,
            :-,
            Expr(:call, :*, :(params[:t]), ComplexF64(2)),
            Expr(:call, :conj, :(params[:u])),
        )
        @test expression.deps == Set([:t, :u])
        values = Dict{Symbol,ComplexF64}(:t => 2.0 + 0.0im, :u => 1.0 + 2.0im)
        @test expression.evalf(values) == 4.0 - (1.0 - 2.0im)

        negated = -p.t
        @test negated.expr == Expr(:call, :-, :(params[:t]))
        @test negated.evalf(values) == -2.0 + 0.0im

        powered = p.t^2 + exp(p.u)
        @test powered.evalf(values) == (2.0 + 0.0im)^2 + exp(1.0 + 2.0im)
    end

    @testset "Coefficient normalization" begin
        p = ParameterNamespace(Set([:t, :μ]))

        scalar = normalize_coefficient(1.5, 1)
        @test scalar isa StaticCoefficient
        @test scalar.values[1, 1] == ComplexF64(1.5)
        @test isempty(scalar.deps)

        ref = normalize_coefficient(-p.μ, 1)
        @test ref isa StaticCoefficient
        @test ref.deps == Set([:μ])

        err = try
            normalize_coefficient(1.5, 2)
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        @test occursin("2×2 matrix", err.msg)
        @test occursin("UniformScaling", err.msg)

        scaled = normalize_coefficient(2.0I, 2)
        @test scaled isa ScaledIdentity
        @test scaled.scalar == ComplexF64(2.0)

        pscaled = p.μ * I
        @test pscaled isa ScaledIdentity
        @test pscaled.deps == Set([:μ])
        pscaled2 = (-p.μ * 3) * I
        @test pscaled2 isa ScaledIdentity

        matrix = normalize_coefficient([0 p.t; 0 0], 2)
        @test matrix isa StaticCoefficient
        @test matrix.values[1, 2] isa ParameterRef
        @test matrix.values[2, 1] == zero(ComplexF64)
        @test matrix.deps == Set([:t])

        @test_throws ErrorException normalize_coefficient([0 1; 1 0], 3)
        @test_throws ErrorException normalize_coefficient([0 "x"; 0 0], 2)
        @test_throws ErrorException normalize_coefficient(NaN, 1)
        @test_throws ErrorException normalize_coefficient(sqrt, 1)

        wrapped = uniform(env -> 1.0)
        @test wrapped isa UniformCoefficient
        @test !wrapped.materialize
        @test wrapped.depends_on === nothing
        narrowed = uniform(env -> 1.0; materialize = true, depends_on = (:t,))
        @test narrowed.materialize
        @test narrowed.depends_on == (:t,)
    end

    @testset "Displacements" begin
        @test normalize_displacement(1, Val(1)) == (1,)
        @test normalize_displacement((1, -2), Val(2)) == (1, -2)
        @test normalize_displacement([0, 3], Val(2)) == (0, 3)
        @test_throws ErrorException normalize_displacement(1, Val(2))
        @test_throws ErrorException normalize_displacement((1,), Val(2))
        @test_throws ErrorException normalize_displacement((1.5,), Val(1))
    end

    @testset "Field specs" begin
        spec = fieldspec(:W)
        @test spec.name === :W
        @test spec.rank === nothing
        @test spec.eltype === nothing

        rich = FieldSpec(:J; rank = 1, eltype = ComplexF64)
        @test fieldspec(rich) === rich
        @test check_field_value(rich, zeros(ComplexF64, 4)) === nothing
        @test_throws ErrorException check_field_value(rich, zeros(ComplexF64, 2, 2))
        @test_throws ErrorException check_field_value(rich, zeros(Float64, 4))
        @test_throws ErrorException check_field_value(rich, sin)
        @test check_field_value(fieldspec(:W), sin) === nothing
        @test_throws ErrorException fieldspec("W")
    end

    @testset "Model construction errors" begin
        chain = Lattice(reshape([1.0], 1, 1))
        two_band = Lattice(reshape([1.0], 1, 1); orbitals = (:A, :B))

        # reserved and colliding field names
        @test_throws ErrorException hamiltonian(chain; fields = (:ψin,)) do h, p
            nothing
        end
        @test_throws ErrorException hamiltonian(chain; fields = (:__lh_x,)) do h, p
            nothing
        end
        @test_throws ErrorException hamiltonian(chain; fields = (:_hoff3,)) do h, p
            nothing
        end
        @test_throws ErrorException hamiltonian(
            chain;
            parameters = (W = 1.0,),
            fields = (:W,),
        ) do h, p
            nothing
        end
        @test_throws ErrorException hamiltonian(chain; parameters = (t = NaN,)) do h, p
            nothing
        end

        # zero-displacement plus_hc with a structural diagonal
        @test_throws ErrorException hamiltonian(two_band; parameters = (t = 1.0,)) do h, p
            hopping!(h, (0,), [p.t 0; 0 0]; plus_hc = true)
        end
        @test_throws ErrorException hamiltonian(chain) do h, p
            hopping!(h, (0,), 1.0; plus_hc = true)
        end
        @test_throws ErrorException hamiltonian(two_band) do h, p
            hopping!(h, (0,); plus_hc = true) do bond, env
                zeros(2, 2)
            end
        end
        # strictly triangular zero-displacement plus_hc is fine
        model = hamiltonian(two_band; parameters = (t = 1.0,)) do h, p
            hopping!(h, (0,), [0 p.t; 0 0]; plus_hc = true)
        end
        @test length(terms(model)) == 1

        # unknown depends_on name
        @test_throws ErrorException hamiltonian(chain; fields = (:W,)) do h, p
            onsite!(h; depends_on = (:V,)) do site, env
                0.0
            end
        end

        # Hermiticity validation of static terms
        @test_throws ErrorException hamiltonian(chain; parameters = (t = 1.0,)) do h, p
            hopping!(h, (1,), p.t)   # no reverse term, declared hermitian
        end
        @test_throws ErrorException hamiltonian(two_band) do h, p
            onsite!(h, [0 1.0im; 1.0im 0])   # not Hermitian
        end
        paired = hamiltonian(chain; parameters = (t = 1.0,)) do h, p
            hopping!(h, (1,), 2.0 + 1.0im)
            hopping!(h, (-1,), 2.0 - 1.0im)
        end
        @test length(terms(paired)) == 2
        directed = hamiltonian(chain; parameters = (t = 1.0,), hermitian = false) do h, p
            hopping!(h, (1,), p.t)
        end
        @test length(terms(directed)) == 1

        # callback Hermiticity is advisory
        @test_logs (:warn, r"cannot verify Hermiticity") hamiltonian(
            chain;
            fields = (:W,),
        ) do h, p
            onsite!(h) do site, env
                env.fields.W[site.cell_index...]
            end
        end
    end
end
