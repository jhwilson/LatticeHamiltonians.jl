function compare_over_wavefunctions(H, Hmat)
    @testset "Random wavefunctions" begin
        for _ = 1:100
            ψ = randn(ComplexF64, 20)
            @test iszero(H * ψ - Hmat * ψ)
        end
    end
    @testset "Plane waves" begin
        for n = 1:10
            ψ = plane_wave([n], [10], 2)
            @test iszero(H * ψ - Hmat * ψ)
        end
    end
    @testset "Gaussian Localized" begin
        for _ = 1:100
            σ = 2 * rand()
            for n = 1:20
                ψ = gaussian(n, 20, σ)
                @test iszero(H * ψ - Hmat * ψ)
            end
        end
    end
    @testset "Exp Localized" begin
        for _ = 1:100
            ξ = 2 * rand()
            for n = 1:20
                ψ = exponential(n, 20, ξ)
                @test iszero(H * ψ - Hmat * ψ)
            end
        end
    end
end

function system_test_suite(H, Hmat)
    @testset "Compare with manually written Hamiltonian" begin
        compare_over_wavefunctions(H, Hmat)
    end

    @testset "Right inverse" begin
        Hinv = inv(Hmat)
        @test hcat([H * Hinv[i, :] for i = 1:20]...) ≈ Matrix(I, 20, 20)
    end
end

@testset "Polyacetylene" begin
    H = @lattice_hamiltonian begin
        L = [10]
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

    system_test_suite(H, Hmat)
end
