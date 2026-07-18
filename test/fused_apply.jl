function test_fused_against_sparse(H)
    ψin = randn(ComplexF64, size(H)[1])
    ψout = similar(ψin)
    Hsparse = sparse(H)

    mul!(ψout, H, ψin)
    @test ψout ≈ Hsparse * ψin
    @test (@allocated mul!(ψout, H, ψin)) == 0
end

@testset "Fused apply boundary regions" begin
    @testset "1D nearest neighbor, L=3" begin
        H = @lattice_hamiltonian begin
            L = [3]
            (0) -> [Δ t0; t0 -Δ]
            (1) -> [0 t1; 0 0]
            (-1) -> [0 0; t1 0]
            Δ = 0.4
            t0 = 0.7
            t1 = -1.2
        end
        test_fused_against_sparse(H)
    end

    @testset "1D nearest and next-nearest neighbors, L=6" begin
        H = @lattice_hamiltonian begin
            L = [6]
            (0) -> μ
            (1) -> t1
            (-1) -> t1
            (2) -> t2
            (-2) -> t2
            μ = 0.25
            t1 = -1.0
            t2 = 0.3
        end
        test_fused_against_sparse(H)
    end

    @testset "2D anisotropic and diagonal hops" begin
        H = @lattice_hamiltonian begin
            L = [3, 4]
            (0, 0) -> μ
            (1, 0) -> tx
            (-1, 0) -> tx
            (0, 1) -> ty
            (0, -1) -> ty
            (1, 1) -> td
            (-1, -1) -> td
            μ = 0.2
            tx = -1.0
            ty = -0.6
            td = 0.15
        end
        test_fused_against_sparse(H)
    end

    @testset "3D two-band boundary stress" begin
        H = @lattice_hamiltonian begin
            L = [4, 3, 5]
            (0, 0, 0) -> [0 Δ+3t; Δ+3t 0]
            (1, 0, 0) -> [0 -t; -t 0]
            (0, 1, 0) -> [0 -t; -t 0]
            (-1, 0, 0) -> [0 -t; -t 0]
            (0, -1, 0) -> [0 -t; -t 0]
            (0, 0, 1) -> [0 -t-t2; -t+t2 0]
            (0, 0, -1) -> [0 -t+t2; -t-t2 0]
            t = 1
            Δ = 1
            t2 = 0.3
        end
        test_fused_against_sparse(H)
    end

    @testset "Empty interior, L=3 with range-two hops" begin
        H = @lattice_hamiltonian begin
            L = [3]
            (0) -> μ
            (2) -> t2
            (-2) -> t2
            μ = -0.1
            t2 = 0.8
        end
        test_fused_against_sparse(H)
    end

    @testset "Complex coefficients retain exact accumulation" begin
        H = @lattice_hamiltonian begin
            L = [32]
            (0) -> μ
            (1) -> a
            (-1) -> aconj
            μ = 0.3
            a = 0.7 + 0.2im
            aconj = 0.7 - 0.2im
        end
        ψin = randn(ComplexF64, size(H)[1])
        expected = similar(ψin)
        for n = 1:32
            accumulator = ComplexF64(0.3) * ψin[n]
            accumulator += ComplexF64(0.7 + 0.2im) * ψin[n == 32 ? 1 : n + 1]
            accumulator += ComplexF64(0.7 - 0.2im) * ψin[n == 1 ? 32 : n - 1]
            expected[n] = accumulator
        end
        @test iszero(H * ψin - expected)
    end
end
