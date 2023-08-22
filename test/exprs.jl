using LatticeHamiltonians: nonzero_elements, symbols_to_lookups

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
        @test nonzero_elements(:([0im 0 + 0 0 - 0 0 * 0])) ==
              (row = [], cols = [], vals = []) broken = true
        # Test literal expressions or literal complex numbers
        @test nonzero_elements(:([4im, 0, 1 + 2im, 2 * 3])) ==
              (rows = [1, 2, 2], cols = [1, 1, 2], vals = [4im, 1 + 2im, 6]) broken = true
    end

    @testset "symbols_to_lookup" begin
        @test symbols_to_lookups(:(1 + 2), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :(1 + 2)
        @test symbols_to_lookups(:(x + y), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :(params[:x] + params[:y])
        @test symbols_to_lookups(:(1 + x + z), Dict{Symbol,ComplexF64}(:x => 1, :y => 2)) ==
              :(1 + params[:x] + z)
    end
end
