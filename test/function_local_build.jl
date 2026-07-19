# Build, apply, sparsify, and iteratively solve entirely inside one function
# call — the kernels are runtime-generated functions, so none of this needs
# top-level scope or Base.invokelatest.

using LatticeHamiltonians: FieldSpec

# Dominant eigenvalue by power iteration on |H| through mul!; H is only
# reached via the LinearAlgebra interface, exactly as an iterative solver
# would use it.
function flb_power_iteration(H; iterations = 300)
    n = size(H)[1]
    v = normalize!(randn(ComplexF64, n))
    w = similar(v)
    for _ = 1:iterations
        mul!(w, H, v)
        normalize!(w)
        v, w = w, v
    end
    real(v' * (H * v))
end

function flb_macro_workflow()
    H = @lattice_hamiltonian begin
        L = [8]
        (0) -> [0 t1; t1 0]
        (1) -> [0 t2; 0 0]
        (-1) -> [0 0; t2 0]
        t1 = 1.0
        t2 = 2.0
    end
    ψ = randn(ComplexF64, size(H)[1])
    S = SparseArrays.sparse(H)
    (H = H, matvec = H * ψ, reference = S * ψ, sparse = S)
end

# Referenced by the @builder block below at macro-expansion time, so it must
# be defined before the enclosing function.
FLB_W = rand(6, 6)

function flb_builder_workflow()
    b = LatticeHamiltonians.@builder begin
        L = [6, 6]
        (0, 0) -> W[n1, n2]
        (1, 0) -> t
        (-1, 0) -> t
        (0, 1) -> t
        (0, -1) -> t
        t = 1.0
        W = FLB_W
    end
    H = build(b)
    ψ = randn(ComplexF64, size(H)[1])
    S = SparseArrays.sparse(H)
    (matvec = H * ψ, reference = S * ψ)
end

function flb_model_workflow(L, W)
    chain = Lattice(reshape([1.0], 1, 1); orbitals = (:c,))
    model = hamiltonian(
        chain;
        parameters = (t = 1.0,),
        fields = (FieldSpec(:W; rank = 1, eltype = Float64),),
    ) do h, p
        hopping!(h, (1,), -p.t; plus_hc = true)
        onsite!(h; materialize = true, depends_on = (:W,)) do site, env
            env.fields.W[site.cell_index...]
        end
    end
    H = build(model, (L,); fields = (W = W,))
    S = SparseArrays.sparse(H)
    λ = flb_power_iteration(H)
    dense = Hermitian(Matrix(S))
    extremes = extrema(eigvals(dense))
    λref = abs(extremes[1]) > abs(extremes[2]) ? extremes[1] : extremes[2]
    ψ = randn(ComplexF64, L)
    (matvec = H * ψ, reference = S * ψ, λ = λ, λref = λref)
end

@testset "Function-local build" begin
    @testset "macro path inside a function" begin
        result = flb_macro_workflow()
        @test result.matvec ≈ result.reference
        # each evaluation builds a fresh Hamiltonian with its own params
        other = flb_macro_workflow()
        result.H.params[:t1] = 7.0
        @test other.H.params[:t1] == 1.0
        @test flb_macro_workflow().H.params[:t1] == 1.0
    end

    @testset "@builder + build inside a function (2D, array field)" begin
        result = flb_builder_workflow()
        @test result.matvec ≈ result.reference
    end

    @testset "model frontend inside a function (disorder + solver)" begin
        W = rand(12) .- 0.5
        result = flb_model_workflow(12, W)
        @test result.matvec ≈ result.reference
        @test result.λ ≈ result.λref rtol = 1e-6
    end
end
