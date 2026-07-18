@testset "Linear algebra" begin
    H = @lattice_hamiltonian begin
        L = [4]
        (0) -> [0 t; t 0]
        (1) -> [0 -t; -t 0]
        (-1) -> [0 -t; -t 0]
        t = 1
    end

    ψ = randn(ComplexF64, size(H)[1])
    @test_throws DimensionMismatch mul!(similar(ψ, length(ψ) - 1), H, ψ)
    @test_throws DimensionMismatch mul!(similar(ψ), H, ψ[1:(end - 1)])
    @test_throws DimensionMismatch mul!(similar(ψ, length(ψ) + 1), H, ψ)
    @test_throws DimensionMismatch mul!(similar(ψ), H, [ψ; zero(ComplexF64)])
end
