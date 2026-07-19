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

# Builder-frontend variants of the same models: parity with the macro path is
# an acceptance gate for the frontend (its lowering must not cost speed).
fe_ssh_lattice =
    Lattice(reshape([1.0], 1, 1); orbitals = (:A, :B), positions = ([0.0], [0.5]))
fe_ssh_model = hamiltonian(fe_ssh_lattice; parameters = (t1 = 1.0, t2 = 2.0)) do h, p
    hopping!(h, (0,), [0 p.t1; 0 0]; plus_hc = true)
    hopping!(h, (1,), [0 p.t2; 0 0]; plus_hc = true)
end
H1d_fe = build(fe_ssh_model, (10000,))
H1d_fe_sparse = sparse(H1d_fe)

fe_chain = Lattice(reshape([1.0], 1, 1))
fe_anderson(materialize) = hamiltonian(
    fe_chain;
    parameters = (t = 1.0,),
    fields = (FieldSpec(:W; rank = 1, eltype = Float64),),
) do h, p
    hopping!(h, (1,), -p.t; plus_hc = true)
    onsite!(h; materialize = materialize, depends_on = (:W,)) do site, env
        env.fields.W[site.cell_index...]
    end
end
Wfe = randn(10000)
H1d_fe_mat = build(fe_anderson(true), (10000,); fields = (W = Wfe,))
H1d_fe_pa = build(fe_anderson(false), (10000,); fields = (W = Wfe,))
H1d_fe_mat_sparse = sparse(H1d_fe_mat)

fe_3d = Lattice(Matrix(1.0I, 3, 3); orbitals = (:A, :B))
fe_3d_model = hamiltonian(fe_3d; parameters = (t = 1.0, Δ = 1.0, t2 = 0.3)) do h, p
    onsite!(h, [0 p.Δ+3p.t; p.Δ+3p.t 0])
    hopping!(h, (1, 0, 0), [0 -p.t; -p.t 0]; plus_hc = true)
    hopping!(h, (0, 1, 0), [0 -p.t; -p.t 0]; plus_hc = true)
    hopping!(h, (0, 0, 1), [0 -p.t-p.t2; -p.t+p.t2 0]; plus_hc = true)
end
H3d_fe = build(fe_3d_model, (20, 20, 20))
H3d_fe_sparse = sparse(H3d_fe)

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
    ("1D (frontend)", H1d_fe, H1d_fe_sparse),
    ("1D Anderson (frontend, materialized)", H1d_fe_mat, H1d_fe_mat_sparse),
    ("1D Anderson (frontend, per-apply)", H1d_fe_pa, H1d_fe_mat_sparse),
    ("3D (frontend)", H3d_fe, H3d_fe_sparse),
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
