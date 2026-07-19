using LatticeHamiltonians
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Arpack

# Small 1D SSH chain, where a dense reference is still affordable
H = @lattice_hamiltonian begin
    L = [100]
    (0) -> [0 t1; t1 0]
    (1) -> [0 t2; 0 0]
    (-1) -> [0 0; t2 0]
    t1 = 1.0
    t2 = 2.0
end

# Define the Hamiltonian to benchmark
BENCHMARK_SIZE = H.d * H.L[1]

Hsparse = sparse(H)

Hmat = diagm(
    1 => [complex(1.0 + iseven(j)) for j = 1:(BENCHMARK_SIZE-1)],
    -1 => [complex(1.0 + iseven(j)) for j = 1:(BENCHMARK_SIZE-1)],
)
Hmat[1, BENCHMARK_SIZE] = 2
Hmat[BENCHMARK_SIZE, 1] = 2

# Larger 1D chain, sized for meaningful matrix-vector timings
H1d = @lattice_hamiltonian begin
    L = [10000]
    (0) -> [0 t1; t1 0]
    (1) -> [0 t2; 0 0]
    (-1) -> [0 0; t2 0]
    t1 = 1.0
    t2 = 2.0
end
H1d_sparse = sparse(H1d)

# 3D two-band model (same model as test/3d_hamiltonian_build.jl)
H3d = @lattice_hamiltonian begin
    L = [20, 20, 20]
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
H3d_sparse = sparse(H3d)

# Define a benchmark suite

SUITE = BenchmarkGroup()

Eigs = BenchmarkGroup()

Eigs["Manual"] = @benchmarkable eigs($(Hmat))
Eigs["Lattice Hamiltonians"] = @benchmarkable eigs($(H))
Eigs["Sparse"] = @benchmarkable eigs($(Hsparse))

SUITE["eigs"] = Eigs

# Matrix-vector products: the matrix-free apply! must stay comparable to
# (or beat) the sparse representation — that comparison is the point of
# this package, so benchmark mul! directly.

Mul = BenchmarkGroup()

for (case, Hmf, Hsp) in [("1D", H1d, H1d_sparse), ("3D", H3d, H3d_sparse)]
    N = size(Hmf)[1]
    group = BenchmarkGroup()
    group["Lattice Hamiltonians"] =
        @benchmarkable mul!(ψout, $Hmf, ψin) setup = (
            ψout = Vector{ComplexF64}(undef, $N);
            ψin = randn(ComplexF64, $N)
        )
    group["Sparse"] = @benchmarkable mul!(ψout, $Hsp, ψin) setup = (
        ψout = Vector{ComplexF64}(undef, $N);
        ψin = randn(ComplexF64, $N)
    )
    Mul[case] = group
end

SUITE["mul!"] = Mul

tune!(SUITE)

results = run(SUITE; verbose = true)

println(results)
