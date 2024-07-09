using LatticeHamiltonians
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Arpack

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

# Define a benchmark suite

SUITE = BenchmarkGroup()

Eigs = BenchmarkGroup()

Eigs["Manual"] = @benchmarkable eigs($(Hmat))
Eigs["Lattice Hamiltonians"] = @benchmarkable eigs($(H))
Eigs["Sparse"] = @benchmarkable eigs($(Hsparse))

SUITE["eigs"] = Eigs

tune!(SUITE)

results = run(SUITE; verbose = true)

println(results)
