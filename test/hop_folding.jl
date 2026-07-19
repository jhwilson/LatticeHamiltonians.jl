# The DSL parses inside macro expansion, so runtime `eval` is needed for
# @test_logs to observe the folding message.
build_at_runtime(expr) = @eval @lattice_hamiltonian $expr

@testset "Hopping displacements beyond the lattice extent" begin
    @testset "1D long hops fold with an informational message" begin
        H = @test_logs (:info, r"reaches around the periodic lattice") match_mode = :any build_at_runtime(
            :(begin
                L = [3]
                (0) -> μ
                (4) -> t
                (-4) -> t
                (7) -> u
                (-7) -> u
                μ = 0.5
                t = -1.0 + 0.3im
                u = 0.2
            end),
        )
        Href = @lattice_hamiltonian begin
            L = [3]
            (0) -> μ
            (1) -> t + u
            (-1) -> t + u
            μ = 0.5
            t = -1.0 + 0.3im
            u = 0.2
        end
        @test sparse(H) == sparse(Href)
        ψ = randn(ComplexF64, 3)
        @test H * ψ ≈ Href * ψ
        @test H * ψ ≈ sparse(H) * ψ
    end

    @testset "hop of exactly L folds onto the diagonal" begin
        H = @test_logs (:info, r"reaches around the periodic lattice") match_mode = :any build_at_runtime(
            :(begin
                L = [3]
                (0) -> μ
                (3) -> t
                (1) -> s
                (-1) -> s
                μ = 0.5
                t = 0.25
                s = -1.0
            end),
        )
        M = Matrix(sparse(H))
        @test all(M[n, n] == 0.5 + 0.25 for n = 1:3)
        ψ = randn(ComplexF64, 3)
        @test H * ψ ≈ sparse(H) * ψ
    end

    @testset "2D mixed in- and out-of-range components" begin
        H = @test_logs (:info, r"reaches around the periodic lattice") match_mode = :any build_at_runtime(
            :(begin
                L = [3, 4]
                (0, 0) -> μ
                (4, -5) -> t
                (-4, 5) -> tconj
                μ = 0.2
                t = 0.4 - 0.1im
                tconj = 0.4 + 0.1im
            end),
        )
        Href = @lattice_hamiltonian begin
            L = [3, 4]
            (0, 0) -> μ
            (1, -1) -> t
            (-1, 1) -> tconj
            μ = 0.2
            t = 0.4 - 0.1im
            tconj = 0.4 + 0.1im
        end
        @test sparse(H) == sparse(Href)
        ψ = randn(ComplexF64, size(H)[1])
        @test H * ψ ≈ Href * ψ
        @test H * ψ ≈ sparse(H) * ψ
    end

    @testset "in-range hops build silently" begin
        @test_logs min_level = Logging.Info build_at_runtime(
            :(begin
                L = [4]
                (0) -> μ
                (1) -> t
                (-1) -> t
                (3) -> u
                (-3) -> u
                μ = 0.1
                t = -1.0
                u = 0.05
            end),
        )
    end

    @testset "shrinking L below a compiled hop throws instead of reading out of bounds" begin
        H = @lattice_hamiltonian begin
            L = [4]
            (0) -> μ
            (2) -> t
            (-2) -> t
            μ = 0.3
            t = -1.0
        end
        H.L[1] = 1
        @test_throws ArgumentError H * ones(ComplexF64, 1)
        @test_throws ArgumentError sparse(H)
    end
end
