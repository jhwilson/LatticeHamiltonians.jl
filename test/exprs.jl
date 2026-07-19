using LatticeHamiltonians:
    nonzero_elements,
    symbols_to_lookups,
    LiteralOrSymbolic,
    SparseEntry,
    parse_hopping,
    extract_potential,
    isonsite,
    parse_lattice_dsl

EXPR_TEST_FIELD = [1.0, 2.0]

@testset "Expression manipulation" begin
    @testset "nonzero_elements" begin
        @test nonzero_elements(:([1 0; 0 1])) ==
              (rows = [1, 2], cols = [1, 2], vals = [1, 1])
        @test nonzero_elements(:([x 0; 0 y])) ==
              (rows = [1, 2], cols = [1, 2], vals = [:x, :y])
        @test nonzero_elements(:([1, 0, 2, 3, ψ, 0, 0])) ==
              (rows = [1, 3, 4, 5], cols = [1, 3, 4, 5], vals = [1, 2, 3, :ψ])
        @test nonzero_elements(:([
            1 0 2
            3 4.0 0
            ψ 0.0 ϕ
        ])) == (
            rows = [1, 1, 2, 2, 3, 3],
            cols = [1, 3, 1, 2, 1, 3],
            vals = [1, 2, 3, 4, :ψ, :ϕ],
        )
        # Test expressions with that clearly evaluate to zero
        @test nonzero_elements(:([0im 0+0; 0-0 0*0])) == (rows = [], cols = [], vals = [])
        # Test literal expressions or literal complex numbers
        @test nonzero_elements(:([4im 0; 1+2im 2*3])) ==
              (rows = [1, 2, 2], cols = [1, 1, 2], vals = [4im, 1 + 2im, 6])
    end

    @testset "symbols_to_lookup" begin
        # Expected literals are folded to ComplexF64, matching what
        # symbols_to_lookups emits (nightly's Expr `==` compares them strictly)
        @test symbols_to_lookups(:(1 + 2), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :($(ComplexF64(1)) + $(ComplexF64(2)))
        @test symbols_to_lookups(:(x + y), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :(params[:x] + params[:y])
        @test symbols_to_lookups(:(1 + x + z), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :($(ComplexF64(1)) + params[:x] + z)
        fieldnames = Set([:W, :f])
        @test symbols_to_lookups(
            :(W[n1 - 1]),
            Dict{Symbol,ComplexF64}(),
            fieldnames,
        ) == :(W[n1 - 1])
        @test symbols_to_lookups(
            :(t * W[n1]),
            Dict{Symbol,ComplexF64}(:t => 1),
            fieldnames,
        ) == :(params[:t] * W[n1])
        @test symbols_to_lookups(
            :(f(2)),
            Dict{Symbol,ComplexF64}(),
            fieldnames,
        ) == :(f(2))
        # parameters inside field indices become integer-converted lookups;
        # inside field-call arguments they stay ComplexF64 lookups
        @test symbols_to_lookups(
            :(W[mod1(n1 + shift, L[1])]),
            Dict{Symbol,ComplexF64}(:shift => 1),
            fieldnames,
        ) == :(W[mod1(n1 + Base.Int(Base.real(params[:shift])), L[1])])
        @test symbols_to_lookups(
            :(f(n1, scale)),
            Dict{Symbol,ComplexF64}(:scale => 2),
            fieldnames,
        ) == :(f(n1, params[:scale]))
    end

    @testset "field validation" begin
        @test_throws ErrorException parse_lattice_dsl(
            quote
                L = [2]
                (0) -> collision_field[n1]
                (1) -> 1
                collision_field = EXPR_TEST_FIELD
                collision_field = 1
            end,
            @__MODULE__,
        )
        @test_throws ErrorException parse_lattice_dsl(
            quote
                L = [2]
                (0) -> bare_field
                (1) -> 1
                bare_field = EXPR_TEST_FIELD
            end,
            @__MODULE__,
        )
    end

    @testset "isonsite" begin
        @test isonsite(:(0 -> [1 0; 0 1])) == true
        @test isonsite(:((0) -> [1 0; 0 1])) == true
        @test isonsite(:((0, 0, 0) -> [1 0; 0 1])) == true
        @test isonsite(:((0, 1, 0) -> [1 0; 0 1])) == false
        @test isonsite(:((-1) -> [1 0; 0 1])) == false
        @test isonsite(:(1 -> [1 0; 0 1])) == false
    end

    @testset "extract_potential" begin
        @test extract_potential(:(), 1, 3, Dict{Symbol,ComplexF64}(:z => 3)) ==
              (LiteralOrSymbolic[0.0+0.0im, 0.0+0.0im, 0.0+0.0im], nothing)
        @test extract_potential(
            :(0 -> [1, ψ, z]),
            1,
            3,
            Dict{Symbol,ComplexF64}(:z => 3),
        ) == (LiteralOrSymbolic[1.0+0.0im, :ψ, :(params[:z])], nothing)
        @test extract_potential(
            :(0 -> [1 0 0; 0 ψ 0; 0 0 z]),
            1,
            3,
            Dict{Symbol,ComplexF64}(:z => 3),
        ) == (LiteralOrSymbolic[1.0+0.0im, :ψ, :(params[:z])], nothing)
        @test extract_potential(
            :(0 -> [1 t 0; 0 ψ 0; j 0 z]),
            1,
            3,
            Dict{Symbol,ComplexF64}(:z => 3),
        ) == (LiteralOrSymbolic[1.0+0.0im, :ψ, :(params[:z])], ([1, 3], [2, 1], [:t, :j]))
    end

    @testset "parse_hopping" begin
        @test parse_hopping(:((1) -> [1 0; 0 1]), Dict{Symbol,ComplexF64}()) ==
              Pair{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
            [1],
            ([1, 2], [1, 2], [1.0 + 0.0im, 1.0 + 0.0im]),
        )
        @test parse_hopping(
            :((1, 0, -1) -> [1 0; y z]),
            Dict{Symbol,ComplexF64}(:y => 2.0),
        ) == Pair{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
            [1, 0, -1],
            ([1, 2, 2], [1, 1, 2], [1.0 + 0.0im, :(params[:y]), :z]),
        )
        # allows singletons
        @test parse_hopping(:((1, 0, -1) -> 1), Dict{Symbol,ComplexF64}(:y => 2.0)) ==
              Pair{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
            [1, 0, -1],
            ([1], [1], [1.0 + 0.0im]),
        )
        # allows tuples
        @test parse_hopping(
            :((1, 0, -1) -> ([1, 2, 2], [1, 1, 2], [1, y, z])),
            Dict{Symbol,ComplexF64}(:y => 2.0),
        ) == Pair{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
            [1, 0, -1],
            ([1, 2, 2], [1, 1, 2], [1.0 + 0.0im, :(params[:y]), :z]),
        )
        # allow vectors
        @test parse_hopping(
            :((1, 0, -1) -> [1, y, z]),
            Dict{Symbol,ComplexF64}(:y => 2.0),
        ) == Pair{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
            [1, 0, -1],
            ([1, 2, 3], [1, 2, 3], [1.0 + 0.0im, :(params[:y]), :z]),
        )
    end
end
