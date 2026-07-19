using LatticeHamiltonians: lower, LatticeVector, MVector, HamiltonianBuilder, CacheRecipe

@testset "Frontend lowering" begin
    no_params() = Dict{Symbol,ComplexF64}()
    no_fields() = Dict{Symbol,Any}()
    chain = Lattice(reshape([1.0], 1, 1))
    two_band = Lattice(reshape([1.0], 1, 1); orbitals = (:A, :B))
    key(a...) = LatticeVector{length(a)}(a)

    @testset "Directed hop: negation and transpose" begin
        c = 0.4 + 0.7im
        model = hamiltonian(two_band; hermitian = false) do h, p
            hopping!(h, (1,), [0 c; 0 0])
        end
        builder, recipes = lower(model, MVector{1,Int}(6), no_params(), no_fields())
        @test isempty(recipes)
        @test builder.d == 2
        # c†[n+1, α=1] c c[n, β=2] → legacy key -1, row = β = 2, col = α = 1
        @test collect(keys(builder.T)) == [key(-1)]
        @test builder.T[key(-1)] == ([2], [1], [ComplexF64(c)])
        @test builder.V == [zero(ComplexF64), zero(ComplexF64)]
    end

    @testset "plus_hc expansion" begin
        c = 0.4 + 0.7im
        model = hamiltonian(two_band) do h, p
            hopping!(h, (1,), [0 c; 0 0]; plus_hc = true)
        end
        builder, _ = lower(model, MVector{1,Int}(6), no_params(), no_fields())
        @test Set(keys(builder.T)) == Set([key(-1), key(1)])
        @test builder.T[key(-1)] == ([2], [1], [ComplexF64(c)])
        @test builder.T[key(1)] == ([1], [2], [conj(ComplexF64(c))])
    end

    @testset "Same-displacement accumulation" begin
        model = hamiltonian(chain; hermitian = false) do h, p
            hopping!(h, (1,), 1.0 + 2.0im)
            hopping!(h, (1,), 0.5)
        end
        builder, _ = lower(model, MVector{1,Int}(6), no_params(), no_fields())
        @test builder.T[key(-1)] == ([1], [1], [ComplexF64(1.5 + 2.0im)])

        mixed = hamiltonian(chain; parameters = (t = 0.3,), hermitian = false) do h, p
            hopping!(h, (1,), 0.5)
            hopping!(h, (1,), p.t)
        end
        builder2, _ =
            lower(mixed, MVector{1,Int}(6), Dict{Symbol,ComplexF64}(:t => 0.3), no_fields())
        @test builder2.T[key(-1)] ==
              ([1], [1], [Expr(:call, :+, ComplexF64(0.5), :(params[:t]))])
    end

    @testset "Onsite routing: diagonal to V, off-diagonal to T[0]" begin
        model = hamiltonian(two_band; parameters = (Δ = 0.3, t1 = 0.8)) do h, p
            onsite!(h, [p.Δ p.t1; conj(p.t1) -p.Δ])
        end
        params = Dict{Symbol,ComplexF64}(:Δ => 0.3, :t1 => 0.8)
        builder, _ = lower(model, MVector{1,Int}(6), params, no_fields())
        @test builder.V == [:(params[:Δ]), Expr(:call, :-, :(params[:Δ]))]
        # (α=1, β=2) → (row 2, col 1); (α=2, β=1) → (row 1, col 2); sorted by (row, col)
        @test builder.T[key(0)] ==
              ([1, 2], [2, 1], [Expr(:call, :conj, :(params[:t1])), :(params[:t1])])
    end

    @testset "Scaled identity" begin
        model = hamiltonian(two_band) do h, p
            onsite!(h, 2.5I)
        end
        builder, _ = lower(model, MVector{1,Int}(4), no_params(), no_fields())
        @test builder.V == [ComplexF64(2.5), ComplexF64(2.5)]
        @test isempty(builder.T)
    end

    @testset "Local per-apply expressions and anchor rule" begin
        model = hamiltonian(chain; fields = (:J,)) do h, p
            hopping!(h, (1,); plus_hc = true) do bond, env
                env.fields.J[bond.source_cell_index...]
            end
        end
        all_fields = Dict{Symbol,Any}(:J => randn(6))
        builder, recipes = lower(model, MVector{1,Int}(6), no_params(), all_fields)
        @test isempty(recipes)
        wrapper = Symbol("__lh_loc1")
        @test haskey(builder.fields, wrapper)
        @test builder.fields[wrapper] isa Function
        # forward piece reads the callback at the bond source m - a (wrapped)
        @test builder.T[key(-1)] == ([1], [1], [:($wrapper(mod1(n1 - 1, L[1])))])
        # reverse piece reads it unshifted and conjugated (anchor rule)
        @test builder.T[key(1)] == ([1], [1], [:(conj($wrapper(n1)))])
    end

    @testset "Materialized local caches and recipes" begin
        J = randn(ComplexF64, 6)
        model = hamiltonian(chain; parameters = (t = 1.0,), fields = (:J,)) do h, p
            hopping!(
                h,
                (1,);
                plus_hc = true,
                materialize = true,
                depends_on = (:J,),
            ) do bond, env
                env.fields.J[bond.source_cell_index...]
            end
        end
        all_fields = Dict{Symbol,Any}(:J => J)
        params = Dict{Symbol,ComplexF64}(:t => 1.0)
        builder, recipes = lower(model, MVector{1,Int}(6), params, all_fields)
        forward_name = Symbol("__lh_c1f_1_1")
        reverse_name = Symbol("__lh_c1r_1_1")
        @test builder.T[key(-1)] == ([1], [1], [Expr(:ref, forward_name, :n1)])
        @test builder.T[key(1)] == ([1], [1], [Expr(:ref, reverse_name, :n1)])
        forward_cache = builder.fields[forward_name]
        reverse_cache = builder.fields[reverse_name]
        @test forward_cache isa Vector{ComplexF64}
        # forward cache is pre-shifted to the output-cell anchor
        @test forward_cache == [J[mod1(m - 1, 6)] for m = 1:6]
        @test reverse_cache == conj.(J)

        @test length(recipes) == 1
        @test recipes[1].parameter_deps == Set{Symbol}()
        @test recipes[1].field_deps == Set([:J])
        @test Set(recipes[1].names) == Set([forward_name, reverse_name])

        # conservative dependency default: everything declared
        conservative = hamiltonian(chain; parameters = (t = 1.0,), fields = (:J,)) do h, p
            hopping!(h, (1,); plus_hc = true, materialize = true) do bond, env
                env.fields.J[bond.source_cell_index...]
            end
        end
        _, conservative_recipes = lower(
            conservative,
            MVector{1,Int}(6),
            Dict{Symbol,ComplexF64}(:t => 1.0),
            Dict{Symbol,Any}(:J => copy(J)),
        )
        @test conservative_recipes[1].parameter_deps == Set([:t])
        @test conservative_recipes[1].field_deps == Set([:J])
    end

    @testset "Real materialized caches use Float64 storage" begin
        W = rand(5)
        model = @test_logs (:warn, r"cannot verify Hermiticity") hamiltonian(
            chain;
            fields = (:W,),
        ) do h, p
            onsite!(h; materialize = true) do site, env
                env.fields.W[site.cell_index...]
            end
        end
        builder, _ = lower(model, MVector{1,Int}(5), no_params(), Dict{Symbol,Any}(:W => W))
        cache = builder.fields[Symbol("__lh_c1f_1_1")]
        @test cache isa Vector{Float64}
        @test cache == W[1:5]
        @test builder.V == [Expr(:ref, Symbol("__lh_c1f_1_1"), :n1)]
    end
end
