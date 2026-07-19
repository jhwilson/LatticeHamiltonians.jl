using LatticeHamiltonians: parse_lattice_dsl

Random.seed!(48329)
SITE_W_1D = randn(10)
SITE_W_SSH = randn(10)
SITE_W_2D = randn(6, 5)
SITE_J_BOND = randn(ComplexF64, 10)
SITE_FUNCTION = n -> exp(-0.1 * n)
SITE_W_MUTABLE = randn(12)
SITE_W_ALLOC = randn(64)
W_SHORT = randn(2, 1)
W_NARROW = randn(3, 1)
W_SHIFTED = randn(6)
REAL_FIELD = randn(4)
SIZE_FIELD = randn(4)
scaled_site(n, s) = 0.05 * real(s) * n

# has size (4,) but axes 0:3 — must be rejected by the field extent guards
struct OffsetAxesVector <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(v::OffsetAxesVector) = size(v.data)
Base.axes(v::OffsetAxesVector) = (0:(length(v.data)-1),)
Base.getindex(v::OffsetAxesVector, i::Int) = v.data[i+1]
W_OFFSET = OffsetAxesVector(randn(4))

flat_site_index(orbital, site, d, L) =
    orbital + d * sum((site[axis] - 1) * prod(L[1:(axis-1)]) for axis = 1:length(L))

function add_dense_hop!(
    matrix,
    value,
    output_site,
    input_site,
    output_orbital,
    input_orbital,
    d,
    L,
)
    output_index = flat_site_index(output_orbital, output_site, d, L)
    input_index = flat_site_index(input_orbital, input_site, d, L)
    matrix[output_index, input_index] += value
end

function compare_site_dependent_wavefunctions(H, matrix, L, d)
    Random.seed!(9217)
    matrix_size = size(H)[1]
    wavefunctions = Any[
        randn(ComplexF64, matrix_size),
        randn(ComplexF64, matrix_size),
        plane_wave(ones(Int, length(L)), L, d),
        ComplexF64.(gaussian(cld(matrix_size, 2), matrix_size, 1.7)),
        ComplexF64.(exponential(cld(matrix_size, 2), matrix_size, 2.3)),
    ]
    for wavefunction in wavefunctions
        @test H * wavefunction ≈ matrix * wavefunction
    end
end

@testset "Site-dependent matrix elements" begin
    @testset "1D Anderson disorder" begin
        H = @lattice_hamiltonian begin
            L = [10]
            (0) -> W[n1]
            (1) -> t
            (-1) -> t
            W = SITE_W_1D
            t = -0.8
        end

        L = [10]
        matrix = zeros(ComplexF64, 10, 10)
        for n = 1:10
            matrix[n, n] = SITE_W_1D[n]
            matrix[n, mod1(n + 1, 10)] = -0.8
            matrix[n, mod1(n - 1, 10)] = -0.8
        end
        compare_site_dependent_wavefunctions(H, matrix, L, 1)
    end

    @testset "Disordered SSH onsite potential" begin
        H = @lattice_hamiltonian begin
            L = [10]
            (0) -> [W[n1] t1; t1 -W[n1]]
            (1) -> [0 t2; 0 0]
            (-1) -> [0 0; t2 0]
            W = SITE_W_SSH
            t1 = 0.6
            t2 = -1.1
        end

        L = [10]
        d = 2
        matrix = zeros(ComplexF64, 20, 20)
        for n = 1:10
            add_dense_hop!(matrix, SITE_W_SSH[n], (n,), (n,), 1, 1, d, L)
            add_dense_hop!(matrix, -SITE_W_SSH[n], (n,), (n,), 2, 2, d, L)
            add_dense_hop!(matrix, 0.6, (n,), (n,), 2, 1, d, L)
            add_dense_hop!(matrix, 0.6, (n,), (n,), 1, 2, d, L)
            add_dense_hop!(matrix, -1.1, (n,), (mod1(n + 1, 10),), 2, 1, d, L)
            add_dense_hop!(matrix, -1.1, (n,), (mod1(n - 1, 10),), 1, 2, d, L)
        end
        compare_site_dependent_wavefunctions(H, matrix, L, d)
    end

    @testset "2D onsite disorder" begin
        H = @lattice_hamiltonian begin
            L = [6, 5]
            (0, 0) -> W[n1, n2]
            (1, 0) -> tx
            (-1, 0) -> tx
            (0, 1) -> ty
            (0, -1) -> ty
            W = SITE_W_2D
            tx = -0.7
            ty = 0.25
        end

        L = [6, 5]
        matrix = zeros(ComplexF64, 30, 30)
        for n2 = 1:5, n1 = 1:6
            output = (n1, n2)
            add_dense_hop!(matrix, SITE_W_2D[n1, n2], output, output, 1, 1, 1, L)
            add_dense_hop!(matrix, -0.7, output, (mod1(n1 + 1, 6), n2), 1, 1, 1, L)
            add_dense_hop!(matrix, -0.7, output, (mod1(n1 - 1, 6), n2), 1, 1, 1, L)
            add_dense_hop!(matrix, 0.25, output, (n1, mod1(n2 + 1, 5)), 1, 1, 1, L)
            add_dense_hop!(matrix, 0.25, output, (n1, mod1(n2 - 1, 5)), 1, 1, 1, L)
        end
        compare_site_dependent_wavefunctions(H, matrix, L, 1)
    end

    @testset "Bond disorder uses source-site coordinates" begin
        H = @lattice_hamiltonian begin
            L = [10]
            (0) -> 0
            (1) -> J[n1]
            (-1) -> conj(J[mod1(n1 - 1, L[1])])
            J = SITE_J_BOND
        end

        # (1) -> J[n1] sets ⟨n+1|H|n⟩ = J[n]: the bond (n, n+1) is labeled by
        # its source site n, and the conjugate hop reads the same bond value.
        matrix = zeros(ComplexF64, 10, 10)
        for n = 1:10
            matrix[mod1(n + 1, 10), n] = SITE_J_BOND[n]
            matrix[mod1(n - 1, 10), n] = conj(SITE_J_BOND[mod1(n - 1, 10)])
        end
        compare_site_dependent_wavefunctions(H, matrix, [10], 1)
        @test Matrix(sparse(H)) ≈ matrix
    end

    @testset "Sparse orientation for a nonsymmetric model" begin
        H = @lattice_hamiltonian begin
            L = [5]
            (0) -> 0
            (1) -> t
            t = 0.4 + 0.7im
        end
        matrix = zeros(ComplexF64, 5, 5)
        for n = 1:5
            matrix[mod1(n + 1, 5), n] = 0.4 + 0.7im
        end
        for column = 1:5
            basis = zeros(ComplexF64, 5)
            basis[column] = 1
            @test iszero(H * basis - matrix * basis)
        end
        @test Matrix(sparse(H)) == matrix
    end

    @testset "Function field" begin
        H = @lattice_hamiltonian begin
            L = [10]
            (0) -> f(n1)
            (1) -> t
            (-1) -> t
            f = n -> exp(-0.1 * n)
            t = 0.2
        end
        matrix = zeros(ComplexF64, 10, 10)
        for n = 1:10
            matrix[n, n] = SITE_FUNCTION(n)
            matrix[n, mod1(n + 1, 10)] = 0.2
            matrix[n, mod1(n - 1, 10)] = 0.2
        end
        compare_site_dependent_wavefunctions(H, matrix, [10], 1)
    end

    @testset "Field mutability and concrete type assertion" begin
        H = @lattice_hamiltonian begin
            L = [12]
            (0) -> W[n1]
            (1) -> t
            (-1) -> t
            W = SITE_W_MUTABLE
            t = -0.3
        end
        wavefunction = randn(ComplexF64, 12)
        result_before = H * wavefunction
        sparse_before = sparse(H)

        fill!(H.fields[:W], 1.25)
        result_after_refill = H * wavefunction
        sparse_after_refill = sparse(H)
        @test result_after_refill ≈ sparse_after_refill * wavefunction
        @test result_after_refill != result_before
        @test sparse_after_refill != sparse_before

        replacement = collect(range(-1.0, 1.0; length = 12))
        H.fields[:W] = replacement
        result_after_rebind = H * wavefunction
        @test result_after_rebind ≈ sparse(H) * wavefunction
        @test result_after_rebind != result_after_refill

        H.fields[:W] = collect(1:12)
        output = similar(wavefunction)
        @test_throws TypeError mul!(output, H, wavefunction)
    end

    @testset "Disordered mul! allocation" begin
        H = @lattice_hamiltonian begin
            L = [64]
            (0) -> W[n1]
            (1) -> t
            (-1) -> t
            W = SITE_W_ALLOC
            t = -1.0
        end
        input = randn(ComplexF64, 64)
        output = similar(input)
        mul!(output, H, input)
        @test (@allocated mul!(output, H, input)) == 0
    end

    @testset "Field extent guards" begin
        # a bare site index in one position is guarded even when another
        # position holds a literal
        H = @lattice_hamiltonian begin
            L = [3, 1]
            (0, 0) -> W[n1, 1]
            (1, 0) -> t
            (-1, 0) -> t
            W = W_SHORT
            t = -1.0
        end
        ψ = ones(ComplexF64, 3)
        @test_throws DimensionMismatch H * ψ
        @test_throws DimensionMismatch sparse(H)

        # a literal index must exist in that dimension's axes
        H = @lattice_hamiltonian begin
            L = [3, 1]
            (0, 0) -> W[n1, 2]
            (1, 0) -> t
            (-1, 0) -> t
            W = W_NARROW
            t = -1.0
        end
        @test_throws DimensionMismatch H * ψ

        # offset axes are rejected: covering size but not the 1:L index range
        H = @lattice_hamiltonian begin
            L = [4]
            (0) -> W[n1]
            (1) -> t
            (-1) -> t
            W = W_OFFSET
            t = -1.0
        end
        @test_throws DimensionMismatch H * ones(ComplexF64, 4)
    end

    @testset "Parameters inside field indices and call arguments" begin
        H = @lattice_hamiltonian begin
            L = [6]
            (0) -> W[mod1(n1 + shift, L[1])]
            (1) -> t
            (-1) -> t
            W = W_SHIFTED
            shift = 1
            t = -0.4
        end
        Hf = @lattice_hamiltonian begin
            L = [6]
            (0) -> f(n1, scale)
            (1) -> t
            (-1) -> t
            f = scaled_site
            scale = 2
            t = -0.4
        end
        for shift in (1, 2)
            H.params[:shift] = shift
            matrix = zeros(ComplexF64, 6, 6)
            for n = 1:6
                matrix[n, n] = W_SHIFTED[mod1(n + shift, 6)]
                matrix[mod1(n + 1, 6), n] = -0.4
                matrix[mod1(n - 1, 6), n] = -0.4
            end
            compare_site_dependent_wavefunctions(H, matrix, [6], 1)
        end
        for scale in (2, 5)
            Hf.params[:scale] = scale
            matrix = zeros(ComplexF64, 6, 6)
            for n = 1:6
                matrix[n, n] = 0.05 * scale * n
                matrix[mod1(n + 1, 6), n] = -0.4
                matrix[mod1(n - 1, 6), n] = -0.4
            end
            compare_site_dependent_wavefunctions(Hf, matrix, [6], 1)
        end
    end

    @testset "Fields may shadow Base helper names" begin
        H = @lattice_hamiltonian begin
            L = [4]
            (0) -> real[n1] + size[n1]
            (1) -> t
            (-1) -> t
            real = REAL_FIELD
            size = SIZE_FIELD
            t = -1.0
        end
        matrix = zeros(ComplexF64, 4, 4)
        for n = 1:4
            matrix[n, n] = REAL_FIELD[n] + SIZE_FIELD[n]
            matrix[mod1(n + 1, 4), n] = -1.0
            matrix[mod1(n - 1, 4), n] = -1.0
        end
        compare_site_dependent_wavefunctions(H, matrix, [4], 1)
        @test Matrix(sparse(H)) ≈ matrix
    end

    @testset "Duplicate assignments are rejected" begin
        @test_throws ErrorException parse_lattice_dsl(
            :(begin
                L = [3]
                (0) -> W[n1]
                (1) -> t
                (-1) -> t
                W = [1.0, 2.0, 3.0]
                W = [9.0, 8.0, 7.0]
                t = -1.0
            end),
            @__MODULE__
        )
        @test_throws ErrorException parse_lattice_dsl(
            :(begin
                L = [3]
                (0) -> 0
                (1) -> t
                (-1) -> t
                t = -1.0
                t = -2.0
            end),
            @__MODULE__
        )
    end
end
