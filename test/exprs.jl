using LatticeHamiltonians:
    nonzero_elements,
    symbols_to_lookups,
    extract_matrix_elements,
    LiteralOrSymbolic,
    SparseEntry,
    parse_hopping

@testset "Expression manipulation" begin
    @testset "nonzero_elements" begin
        @test nonzero_elements(:([1 0; 0 1])) ==
              (rows = [1, 2], cols = [1, 2], vals = [1, 1])
        @test nonzero_elements(:([x 0; 0 y])) ==
              (rows = [1, 2], cols = [1, 2], vals = [:x, :y])
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
        @test_broken nonzero_elements(:([0im 0 + 0 0 - 0 0 * 0])) ==
                     (row = [], cols = [], vals = [])
        # Test literal expressions or literal complex numbers
        @test_broken nonzero_elements(:([4im, 0, 1 + 2im, 2 * 3])) ==
                     (rows = [1, 2, 2], cols = [1, 1, 2], vals = [4im, 1 + 2im, 6])
    end

    @testset "symbols_to_lookup" begin
        @test symbols_to_lookups(:(1 + 2), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :(1 + 2)
        @test symbols_to_lookups(:(x + y), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :(params[:x] + params[:y])
        @test symbols_to_lookups(:(1 + x + z), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :(1 + params[:x] + z)
    end

    @testset "extract_matrix_elements" begin
        @test extract_matrix_elements(
            [:((0, 1) -> [1 0; 0 1]), :((1, 1) -> [1 0; y z])],
            :(V = []),
            Dict{Symbol,ComplexF64}(:y => 2.0),
        ) == (
            Dict{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
                [0, 1] => ([1, 2], [1, 2], [1.0 + 0.0im, 1.0 + 0.0im]),
                [1, 1] => ([1, 2, 2], [1, 1, 2], [1.0 + 0.0im, :(params[:y]), :z]),
            ),
            LiteralOrSymbolic[],
        )
        @test extract_matrix_elements(
            Expr[],
            :(V = [1, ψ, z]),
            Dict{Symbol,ComplexF64}(:z => 3),
        ) == (
            Dict{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(),
            LiteralOrSymbolic[1.0+0.0im, :ψ, :(params[:z])],
        )
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
        @test parse_hopping(
            :((1, 0, -1) -> 1),
            Dict{Symbol,ComplexF64}(:y => 2.0),
        ) == Pair{Vector{Int64},SparseEntry{LiteralOrSymbolic}}(
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
    end
end
