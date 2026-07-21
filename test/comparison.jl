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

# ── multi-dimensional open BCs ────────────────────────────────────────────────
# Independent oracle: start from the trusted PERIODIC operator and remove exactly
# the nearest-neighbour bonds that wrap across an open axis. For |δ| ≤ 1 hops and
# L[axis] ≥ 3, a bond whose cell-coordinate difference along an axis has magnitude
# L[axis] - 1 can only arise by wrapping, so it is the boundary bond to drop. This
# depends solely on the documented orbital-fastest, column-major flattening — not on
# any codegen internal — so it is a genuine cross-check of the multi-dim open path.
function cell_coords(idx, L, d)
    lin = div(idx - 1, d)                       # 0-based cell (orbital index dropped)
    coords = similar(L)
    for a in eachindex(L)
        coords[a] = mod(lin, L[a]) + 1          # column-major: dim 1 fastest
        lin = div(lin, L[a])
    end
    coords
end

function open_reference(Hp, L, d, open)
    Href = copy(Hp)
    rows, cols, _ = findnz(Hp)
    for (r, c) in zip(rows, cols)
        cr = cell_coords(r, L, d)
        cc = cell_coords(c, L, d)
        for a in eachindex(L)
            if open[a] && abs(cc[a] - cr[a]) == L[a] - 1
                Href[r, c] = 0                  # a NN bond wrapping this open axis
                break
            end
        end
    end
    dropzeros!(Href)
end

@testset "2D open boundary conditions (L = [3, 4])" begin
    Hp = sparse(@lattice_hamiltonian begin
        L = [3, 4]
        (0, 0) -> [0 1; 1 0]
        (1, 0) -> [0.5 0; 0 0.5]
        (-1, 0) -> [0.5 0; 0 0.5]
        (0, 1) -> [0.3 0; 0 0.3]
        (0, -1) -> [0.3 0; 0 0.3]
    end)
    @test open_reference(Hp, [3, 4], 2, [false, false]) == Hp     # sanity: nothing removed

    H_tt = sparse(@lattice_hamiltonian begin
        L = [3, 4]
        open = [true, true]
        (0, 0) -> [0 1; 1 0]
        (1, 0) -> [0.5 0; 0 0.5]
        (-1, 0) -> [0.5 0; 0 0.5]
        (0, 1) -> [0.3 0; 0 0.3]
        (0, -1) -> [0.3 0; 0 0.3]
    end)
    @test H_tt == open_reference(Hp, [3, 4], 2, [true, true])

    H_tf = sparse(@lattice_hamiltonian begin
        L = [3, 4]
        open = [true, false]
        (0, 0) -> [0 1; 1 0]
        (1, 0) -> [0.5 0; 0 0.5]
        (-1, 0) -> [0.5 0; 0 0.5]
        (0, 1) -> [0.3 0; 0 0.3]
        (0, -1) -> [0.3 0; 0 0.3]
    end)
    @test H_tf == open_reference(Hp, [3, 4], 2, [true, false])
end

@testset "3D open boundary conditions (L = [3, 3, 3])" begin
    Hp = sparse(@lattice_hamiltonian begin
        L = [3, 3, 3]
        (0, 0, 0) -> [0 1; 1 0]
        (1, 0, 0) -> [0.5 0; 0 0.5]
        (-1, 0, 0) -> [0.5 0; 0 0.5]
        (0, 1, 0) -> [0.3 0; 0 0.3]
        (0, -1, 0) -> [0.3 0; 0 0.3]
        (0, 0, 1) -> [0.2 0; 0 0.2]
        (0, 0, -1) -> [0.2 0; 0 0.2]
    end)

    H_ttt = sparse(@lattice_hamiltonian begin
        L = [3, 3, 3]
        open = [true, true, true]
        (0, 0, 0) -> [0 1; 1 0]
        (1, 0, 0) -> [0.5 0; 0 0.5]
        (-1, 0, 0) -> [0.5 0; 0 0.5]
        (0, 1, 0) -> [0.3 0; 0 0.3]
        (0, -1, 0) -> [0.3 0; 0 0.3]
        (0, 0, 1) -> [0.2 0; 0 0.2]
        (0, 0, -1) -> [0.2 0; 0 0.2]
    end)
    @test H_ttt == open_reference(Hp, [3, 3, 3], 2, [true, true, true])

    H_mid = sparse(@lattice_hamiltonian begin
        L = [3, 3, 3]
        open = [false, true, false]
        (0, 0, 0) -> [0 1; 1 0]
        (1, 0, 0) -> [0.5 0; 0 0.5]
        (-1, 0, 0) -> [0.5 0; 0 0.5]
        (0, 1, 0) -> [0.3 0; 0 0.3]
        (0, -1, 0) -> [0.3 0; 0 0.3]
        (0, 0, 1) -> [0.2 0; 0 0.2]
        (0, 0, -1) -> [0.2 0; 0 0.2]
    end)
    @test H_mid == open_reference(Hp, [3, 3, 3], 2, [false, true, false])
end
