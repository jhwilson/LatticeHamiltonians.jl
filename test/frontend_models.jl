# Static-coefficient models through the builder frontend, compared against
# hand-written dense matrices and against the legacy macro.
#
# `build(model, ...)` compiles kernels with `eval` at call time, so every build
# lives at file top level; testsets only apply the resulting operators.

using LatticeHamiltonians: Realization

Random.seed!(0x4f5a11)

fm_flat(orbital, cell, d) = orbital + d * (cell - 1)

function fm_compare(H, reference)
    @test Matrix(SparseArrays.sparse(H)) ≈ reference
    for _ = 1:5
        ψ = randn(ComplexF64, size(reference, 1))
        @test H * ψ ≈ reference * ψ
    end
end

# --- periodic chain: onsite potential + nearest-neighbor hopping ------------
FM_CHAIN = Lattice(reshape([1.0], 1, 1); orbitals = (:c,))
FM_CHAIN_MODEL = hamiltonian(FM_CHAIN; parameters = (t = 1.0, μ = 0.25)) do h, p
    onsite!(h, -p.μ)
    hopping!(h, (1,), -p.t; plus_hc = true, label = :nearest_neighbor)
end
FM_CHAIN_H = build(FM_CHAIN_MODEL, (10,))
FM_CHAIN_OVERRIDE_H = build(FM_CHAIN_MODEL, (10,); parameters = (t = 2.0,))

function fm_chain_dense(L, t, μ)
    reference = zeros(ComplexF64, L, L)
    for n = 1:L
        reference[n, n] = -μ
        target = mod1(n + 1, L)
        reference[target, n] += -t
        reference[n, target] += -conj(t)
    end
    reference
end

# --- SSH with a nonsymmetric complex intercell hopping ----------------------
FM_T2 = 2.0 + 0.5im
FM_SSH_LATTICE =
    Lattice(reshape([1.0], 1, 1); orbitals = (:A, :B), positions = ([0.0], [0.5]))
FM_SSH_MODEL = hamiltonian(FM_SSH_LATTICE; parameters = (t1 = 1.0,)) do h, p
    hopping!(h, (0,), [0 p.t1; 0 0]; plus_hc = true, label = :intracell)
    hopping!(h, (1,), [0 FM_T2; 0 0]; plus_hc = true, label = :intercell)
end
FM_SSH_H = build(FM_SSH_MODEL, (10,))
FM_SSH_LEGACY = @lattice_hamiltonian begin
    L = [10]
    (0) -> [0 t1; t1 0]
    (1) -> [0 t2; 0 0]
    (-1) -> [0 0; t2c 0]
    t1 = 1.0
    t2 = 2.0 + 0.5im
    t2c = 2.0 - 0.5im
end

function fm_ssh_dense(L, t1, t2)
    reference = zeros(ComplexF64, 2L, 2L)
    for n = 1:L
        # intracell: c†[n, A] t1 c[n, B] + h.c.
        reference[fm_flat(1, n, 2), fm_flat(2, n, 2)] += t1
        reference[fm_flat(2, n, 2), fm_flat(1, n, 2)] += conj(t1)
        # intercell: c†[n+1, A] t2 c[n, B] + h.c.
        target = mod1(n + 1, L)
        reference[fm_flat(1, target, 2), fm_flat(2, n, 2)] += t2
        reference[fm_flat(2, n, 2), fm_flat(1, target, 2)] += conj(t2)
    end
    reference
end

# --- orientation golden test: a single directed hop -------------------------
FM_GOLD = 0.4 + 0.7im
FM_DIRECTED_MODEL = hamiltonian(FM_SSH_LATTICE; hermitian = false) do h, p
    hopping!(h, (1,), [0 FM_GOLD; 0 0])
end
FM_DIRECTED_H = @test_logs (:warn, r"assume Hermiticity") build(FM_DIRECTED_MODEL, (5,))

# --- 2D two-band model ------------------------------------------------------
FM_2D_T = 0.7 + 0.2im
FM_2D_LATTICE = Lattice([1.0 0.0; 0.0 1.0]; orbitals = (:A, :B))
FM_2D_MODEL = hamiltonian(FM_2D_LATTICE; parameters = (Δ = 0.3,)) do h, p
    onsite!(h, [p.Δ 0; 0 -p.Δ])
    hopping!(h, (1, 0), [0 FM_2D_T; 0 0]; plus_hc = true)
    hopping!(h, (0, 1), FM_2D_T * I; plus_hc = true)
end
FM_2D_L = (4, 3)
FM_2D_H = build(FM_2D_MODEL, FM_2D_L)

function fm_2d_dense(L1, L2, Δ, t)
    cell(n1, n2) = n1 + L1 * (n2 - 1)
    N = 2 * L1 * L2
    reference = zeros(ComplexF64, N, N)
    for n2 = 1:L2, n1 = 1:L1
        n = cell(n1, n2)
        reference[fm_flat(1, n, 2), fm_flat(1, n, 2)] = Δ
        reference[fm_flat(2, n, 2), fm_flat(2, n, 2)] = -Δ
        target1 = cell(mod1(n1 + 1, L1), n2)
        reference[fm_flat(1, target1, 2), fm_flat(2, n, 2)] += t
        reference[fm_flat(2, n, 2), fm_flat(1, target1, 2)] += conj(t)
        target2 = cell(n1, mod1(n2 + 1, L2))
        for orbital = 1:2
            reference[fm_flat(orbital, target2, 2), fm_flat(orbital, n, 2)] += t
            reference[fm_flat(orbital, n, 2), fm_flat(orbital, target2, 2)] += conj(t)
        end
    end
    reference
end

# --- onsite-only model (empty hopping set) ----------------------------------
FM_ONSITE_MODEL = hamiltonian(FM_CHAIN; parameters = (μ = 0.75,)) do h, p
    onsite!(h, -p.μ)
end
FM_ONSITE_H = build(FM_ONSITE_MODEL, (6,))

# --- aliasing diagnostics ---------------------------------------------------
FM_ALIAS_H = @test_logs (:warn, r"a ≡ -a") build(FM_CHAIN_MODEL, (2,))
FM_SPAN_MODEL = hamiltonian(FM_CHAIN; parameters = (t = 0.5,)) do h, p
    hopping!(h, (3,), p.t; plus_hc = true)
end
FM_SPAN_H = @test_logs (:warn, r"spans the full system") build(FM_SPAN_MODEL, (3,))

@testset "Frontend models" begin
    @testset "Periodic chain vs dense" begin
        fm_compare(FM_CHAIN_H, fm_chain_dense(10, 1.0, 0.25))
        @test ishermitian(Matrix(SparseArrays.sparse(FM_CHAIN_H)))
    end

    @testset "Build-time parameter overrides" begin
        fm_compare(FM_CHAIN_OVERRIDE_H, fm_chain_dense(10, 2.0, 0.25))
        @test parameters(FM_CHAIN_OVERRIDE_H) == (t = 2.0 + 0.0im, μ = 0.25 + 0.0im)
    end

    @testset "set_parameter!" begin
        set_parameter!(FM_CHAIN_H; t = 1.7, μ = -0.1)
        fm_compare(FM_CHAIN_H, fm_chain_dense(10, 1.7, -0.1))
        @test parameters(FM_CHAIN_H) == (t = 1.7 + 0.0im, μ = -0.1 + 0.0im)
        set_parameter!(FM_CHAIN_H; t = 1.0, μ = 0.25)
        @test_throws ErrorException set_parameter!(FM_CHAIN_H; q = 1.0)
        @test_throws ErrorException set_parameter!(FM_CHAIN_H; t = NaN)
    end

    @testset "SSH vs dense and legacy macro" begin
        reference = fm_ssh_dense(10, 1.0, FM_T2)
        fm_compare(FM_SSH_H, reference)
        @test Matrix(SparseArrays.sparse(FM_SSH_H)) ≈
              Matrix(SparseArrays.sparse(FM_SSH_LEGACY))
        for _ = 1:5
            ψ = randn(ComplexF64, 20)
            @test FM_SSH_H * ψ ≈ FM_SSH_LEGACY * ψ
        end
        @test ishermitian(Matrix(SparseArrays.sparse(FM_SSH_H)))
    end

    @testset "Orientation golden test" begin
        matrix = Matrix(SparseArrays.sparse(FM_DIRECTED_H))
        expected = zeros(ComplexF64, 10, 10)
        for n = 1:5
            # c†[n+1, α=1] c c[n, β=2]: row is the target orbital :A at cell n+1
            expected[fm_flat(1, mod1(n + 1, 5), 2), fm_flat(2, n, 2)] = FM_GOLD
        end
        @test matrix == expected
    end

    @testset "2D two-band model vs dense" begin
        fm_compare(FM_2D_H, fm_2d_dense(FM_2D_L..., 0.3, FM_2D_T))
    end

    @testset "Onsite-only model" begin
        fm_compare(FM_ONSITE_H, Matrix((-0.75 + 0.0im) * I, 6, 6))
    end

    @testset "Aliasing diagnostics still accumulate correctly" begin
        # L = 2 with a = (1,) plus_hc: forward and reverse bonds coincide.
        reference = fm_chain_dense(2, 1.0, 0.25)
        fm_compare(FM_ALIAS_H, reference)
        # a = (3,) on L = 3 wraps onto the diagonal.
        span_reference = zeros(ComplexF64, 3, 3)
        for n = 1:3
            span_reference[n, n] = 0.5 + conj(0.5)
        end
        fm_compare(FM_SPAN_H, span_reference)
    end

    @testset "build validation errors" begin
        @test_throws ErrorException build(FM_CHAIN_MODEL, (10, 10))
        @test_throws ErrorException build(FM_CHAIN_MODEL, (0,))
        @test_throws ErrorException build(FM_CHAIN_MODEL, (10,); parameters = (q = 1.0,))
        @test_throws ErrorException build(FM_SPAN_MODEL, (2,))  # |a| > L
        @test_throws ArgumentError build(FM_CHAIN_MODEL, (10,); boundary = Open())
        @test_throws ArgumentError build(FM_CHAIN_MODEL, (10,); boundary = Twisted(0.1))
    end

    @testset "Realization metadata" begin
        @test FM_CHAIN_H.meta isa Realization
        @test FM_SSH_LEGACY.meta === nothing
        @test_throws ErrorException set_parameter!(FM_SSH_LEGACY; t1 = 2.0)
        @test_throws ErrorException parameters(FM_SSH_LEGACY)
    end
end
