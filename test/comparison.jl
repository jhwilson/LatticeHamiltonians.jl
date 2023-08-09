
@testset "Polyacetylene" begin
    H = @lattice_hamiltonian begin
        L = [10]
        V = [0.0, 0.0]
        (0) -> [0 t1; t1 0]
        (1) -> [0 t2; 0 0]
        (-1) -> [0 0; t2 0]
        t1 = 1.0
        t2 = 2.0
    end

    Hmat = diagm(
        1 => [complex(1.0 + iseven(j)) for j = 1:19],
        -1 => [complex(1.0 + iseven(j)) for j = 1:19],
    )
    Hmat[1, 20] = 2
    Hmat[20, 1] = 2

    @testset "Compare with manually written Hamiltonian" begin
        for i = 1:100
            ψ = randn(ComplexF64, 20)
            @assert iszero((H * ψ) - (Hmat * ψ))
        end
    end
end
