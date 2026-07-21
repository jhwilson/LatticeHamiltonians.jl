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

@testset "Polyacetylene (open boundary conditions)" begin
    # Same SSH chain as the periodic Polyacetylene test, but the single lattice
    # axis is open (open = [true]) so the last unit cell must NOT hop back around
    # to the first. The reference matrix is therefore identical EXCEPT it omits
    # the periodic wraparound bond Hmat[1, 20] = Hmat[20, 1] = 2.
    H = @lattice_hamiltonian begin
        L = [10]
        open = [true]
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
    # NOTE: the periodic version additionally sets Hmat[1, 20] = Hmat[20, 1] = 2;
    # deliberately omitted here — an open chain has no wraparound coupling.

    @testset "Compare with manually written open-chain Hamiltonian" begin
        compare_over_wavefunctions(H, Hmat)
    end

    @testset "No wraparound coupling" begin
        # Directly assert the boundary bond open BCs must drop: exciting the last
        # site produces no amplitude on the first (2.0 under periodic BCs).
        ψ = zeros(ComplexF64, 20)
        ψ[20] = 1.0
        @test iszero((H * ψ)[1])
    end
end
