@testset "Symbolic parameter mutability" begin
    H = @lattice_hamiltonian begin
        L = [4]
        (0) -> [Δ+3t t; t Δ-t]
        (1) -> [0 -t; -t 0]
        (-1) -> [0 -t; -t 0]
        t = 1
        Δ = 0.5
    end

    ψ = ComplexF64.(1:size(H)[1])
    Hsp_before = sparse(H)
    result_before = H * ψ
    @test result_before ≈ Hsp_before * ψ

    H.params[:t] = 2
    Hsp_after = sparse(H)
    result_after = H * ψ
    @test result_after ≈ Hsp_after * ψ
    @test result_after ≠ result_before
    @test Hsp_after ≠ Hsp_before
end
