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

    # Open-BC mirror of the diagonal test above. A (1,1) bond crosses the boundary in
    # TWO axes at once, so the fused apply must exclude it from the @simd ivdep interior
    # (interior_margins must shrink in both axes) and drop/keep it per-axis in the
    # boundary slab. sparse(H) carries the open-BC wrap-drop (cross-validated against
    # open_reference in comparison.jl), so fused-vs-sparse here closes fused-vs-oracle.
    @testset "2D diagonal hops, open [true, true]" begin
        H = @lattice_hamiltonian begin
            L = [3, 4]
            open = [true, true]
            (0, 0) -> μ
            (1, 0) -> tx
            (-1, 0) -> tx
            (0, 1) -> ty
            (0, -1) -> ty
            (1, 1) -> td
            (-1, -1) -> td
            (1, -1) -> ta
            (-1, 1) -> ta
            μ = 0.2; tx = -1.0; ty = -0.6; td = 0.15; ta = 0.1
        end
        test_fused_against_sparse(H)
    end

    @testset "2D diagonal hops, mixed open [true, false]" begin
        H = @lattice_hamiltonian begin
            L = [3, 4]
            open = [true, false]
            (0, 0) -> μ
            (1, 0) -> tx
            (-1, 0) -> tx
            (0, 1) -> ty
            (0, -1) -> ty
            (1, 1) -> td
            (-1, -1) -> td
            (1, -1) -> ta
            (-1, 1) -> ta
            μ = 0.2; tx = -1.0; ty = -0.6; td = 0.15; ta = 0.1
        end
        test_fused_against_sparse(H)
    end

    @testset "2D diagonal hops, mixed open [false, true]" begin
        H = @lattice_hamiltonian begin
            L = [3, 4]
            open = [false, true]
            (0, 0) -> μ
            (1, 0) -> tx
            (-1, 0) -> tx
            (0, 1) -> ty
            (0, -1) -> ty
            (1, 1) -> td
            (-1, -1) -> td
            (1, -1) -> ta
            (-1, 1) -> ta
            μ = 0.2; tx = -1.0; ty = -0.6; td = 0.15; ta = 0.1
        end
        test_fused_against_sparse(H)
    end

    @testset "2D diagonal hops, larger L=[8,5] open [true,false]" begin
        # Non-trivial @simd ivdep interior (interior x-range 2:7) with diagonal
        # boundary slabs around it -- the L=[3,4] cases have a length-1 interior loop,
        # so this exercises the vectorized interior alongside the diagonal wrap-drop.
        H = @lattice_hamiltonian begin
            L = [8, 5]
            open = [true, false]
            (0, 0) -> μ
            (1, 0) -> tx
            (-1, 0) -> tx
            (0, 1) -> ty
            (0, -1) -> ty
            (1, 1) -> td
            (-1, -1) -> td
            (1, -1) -> ta
            (-1, 1) -> ta
            μ = 0.2; tx = -1.0; ty = -0.6; td = 0.15; ta = 0.1
        end
        test_fused_against_sparse(H)
    end

    # 3D body-diagonal (1,1,1) crosses THREE axes at once -- the strongest test of the
    # per-axis wrap-drop and of interior_margins shrinking in all three dims. Includes a
    # fully-open and a mixed (open x,z / periodic y) case.
    @testset "3D body-diagonal hops, open [true, true, true]" begin
        H = @lattice_hamiltonian begin
            L = [4, 3, 3]
            open = [true, true, true]
            (0, 0, 0) -> μ
            (1, 0, 0) -> tx
            (-1, 0, 0) -> tx
            (0, 1, 0) -> ty
            (0, -1, 0) -> ty
            (0, 0, 1) -> tz
            (0, 0, -1) -> tz
            (1, 1, 1) -> tb
            (-1, -1, -1) -> tb
            μ = 0.2; tx = -1.0; ty = -0.6; tz = -0.4; tb = 0.12
        end
        test_fused_against_sparse(H)
    end

    @testset "3D body-diagonal hops, mixed open [true, false, true]" begin
        H = @lattice_hamiltonian begin
            L = [4, 3, 3]
            open = [true, false, true]
            (0, 0, 0) -> μ
            (1, 0, 0) -> tx
            (-1, 0, 0) -> tx
            (0, 1, 0) -> ty
            (0, -1, 0) -> ty
            (0, 0, 1) -> tz
            (0, 0, -1) -> tz
            (1, 1, 1) -> tb
            (-1, -1, -1) -> tb
            μ = 0.2; tx = -1.0; ty = -0.6; tz = -0.4; tb = 0.12
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
        # (δ) -> t sets ⟨n+δ|H|n⟩ = t, so the output at n gathers t * ψin[n-δ]
        ψin = randn(ComplexF64, size(H)[1])
        expected = similar(ψin)
        for n = 1:32
            accumulator = ComplexF64(0.3) * ψin[n]
            accumulator += ComplexF64(0.7 + 0.2im) * ψin[n == 1 ? 32 : n - 1]
            accumulator += ComplexF64(0.7 - 0.2im) * ψin[n == 32 ? 1 : n + 1]
            expected[n] = accumulator
        end
        @test iszero(H * ψin - expected)
        test_fused_against_sparse(H)
    end

    @testset "Hopping convention: (δ) -> t is ⟨n+δ|H|n⟩" begin
        H = @lattice_hamiltonian begin
            L = [4]
            (1) -> t
            (-1) -> tconj
            t = 0.7 + 0.2im
            tconj = 0.7 - 0.2im
        end
        M = Matrix(sparse(H))
        @test M[2, 1] == 0.7 + 0.2im
        @test M[1, 2] == 0.7 - 0.2im
        # the matrix-free apply realizes the same matrix, column by column
        for n = 1:4
            e = zeros(ComplexF64, 4)
            e[n] = 1
            @test H * e ≈ M[:, n]
        end
    end

    @testset "Complex Hermitian hops match sparse in 2D" begin
        H = @lattice_hamiltonian begin
            L = [4, 3]
            (0, 0) -> [μ 0; 0 -μ]
            (1, 0) -> [0 a; b 0]
            (-1, 0) -> [0 conj(b); conj(a) 0]
            (0, 1) -> c
            (0, -1) -> cconj
            μ = 0.25
            a = 0.4 + 0.9im
            b = -0.3 + 0.1im
            c = 0.2 - 0.6im
            cconj = 0.2 + 0.6im
        end
        test_fused_against_sparse(H)
    end
end
