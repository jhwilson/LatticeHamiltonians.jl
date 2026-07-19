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

# Disordered variants: site-dependent matrix elements are the point of the
# fields feature, so benchmark the flagship Anderson cases and a function
# entry against their sparse representations.
W1d = randn(10000)
H1d_dis = @lattice_hamiltonian begin
    L = [10000]
    (0) -> [W[n1] t1; t1 W[n1]]
    (1) -> [0 t2; 0 0]
    (-1) -> [0 0; t2 0]
    t1 = 1.0
    t2 = 2.0
    W = W1d
end
H1d_dis_sparse = sparse(H1d_dis)

W3d = randn(20, 20, 20)
H3d_dis = @lattice_hamiltonian begin
    L = [20, 20, 20]
    (0, 0, 0) -> [W[n1, n2, n3] Δ+3t; Δ+3t W[n1, n2, n3]]
    (1, 0, 0) -> [0 -t; -t 0]
    (0, 1, 0) -> [0 -t; -t 0]
    (-1, 0, 0) -> [0 -t; -t 0]
    (0, -1, 0) -> [0 -t; -t 0]
    (0, 0, 1) -> [0 -t-t2; -t+t2 0]
    (0, 0, -1) -> [0 -t+t2; -t-t2 0]
    t = 1
    Δ = 1
    t2 = 0.3
    W = W3d
end
H3d_dis_sparse = sparse(H3d_dis)

potential_fn = n -> 0.1 * sin(0.37 * n)
H1d_fn = @lattice_hamiltonian begin
    L = [10000]
    (0) -> f(n1)
    (1) -> t
    (-1) -> t
    t = 1.0
    f = potential_fn
end
H1d_fn_sparse = sparse(H1d_fn)

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

for (case, Hmf, Hsp) in [
    ("1D", H1d, H1d_sparse),
    ("3D", H3d, H3d_sparse),
    ("1D Anderson", H1d_dis, H1d_dis_sparse),
    ("3D Anderson", H3d_dis, H3d_dis_sparse),
    ("1D function", H1d_fn, H1d_fn_sparse),
]
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
