# Site- and bond-dependent callbacks: per-apply and materialized modes must
# agree with each other and with hand-written dense references, and
# materialized caches must refresh under checked mutation.
#
# Builds live at file top level (see frontend_models.jl).

Random.seed!(0x10c41)

fl_flat(orbital, cell, d) = orbital + d * (cell - 1)

function fl_compare(H, reference)
    @test Matrix(SparseArrays.sparse(H)) ≈ reference
    for _ = 1:5
        ψ = randn(ComplexF64, size(reference, 1))
        @test H * ψ ≈ reference * ψ
    end
end

FL_CHAIN = Lattice(reshape([1.0], 1, 1); orbitals = (:c,))
FL_L = 12

# --- Anderson disorder: per-apply vs materialized ---------------------------
FL_W = rand(FL_L)
fl_anderson(materialize) = @test_logs (:warn, r"cannot verify Hermiticity") hamiltonian(
    FL_CHAIN;
    parameters = (t = 1.0,),
    fields = (FieldSpec(:W; rank = 1, eltype = Float64),),
) do h, p
    hopping!(h, (1,), -p.t; plus_hc = true)
    onsite!(h; materialize = materialize, depends_on = (:W,)) do site, env
        env.fields.W[site.cell_index...]
    end
end
FL_ANDERSON_PA = build(fl_anderson(false), (FL_L,); fields = (W = FL_W,))
FL_ANDERSON_MAT = build(fl_anderson(true), (FL_L,); fields = (W = FL_W,))

function fl_anderson_dense(W, t)
    L = length(W)
    reference = zeros(ComplexF64, L, L)
    for n = 1:L
        reference[n, n] = W[n]
        target = mod1(n + 1, L)
        reference[target, n] += -t
        reference[n, target] += -conj(t)
    end
    reference
end

# --- complex bond disorder with plus_hc (anchor rule) -----------------------
FL_J = randn(ComplexF64, FL_L)
fl_bond(materialize) = hamiltonian(FL_CHAIN; fields = (:J,)) do h, p
    hopping!(h, (1,); plus_hc = true, materialize = materialize) do bond, env
        env.fields.J[bond.source_cell_index...]
    end
end
FL_BOND_PA = build(fl_bond(false), (FL_L,); fields = (J = FL_J,))
FL_BOND_MAT = build(fl_bond(true), (FL_L,); fields = (J = FL_J,))

function fl_bond_dense(J)
    L = length(J)
    reference = zeros(ComplexF64, L, L)
    for n = 1:L
        # forward bond anchored at its source n, including the boundary bond
        target = mod1(n + 1, L)
        reference[target, n] += J[n]
        reference[n, target] += conj(J[n])
    end
    reference
end

# --- matrix-valued nonsymmetric local callback ------------------------------
FL_TWO_BAND = Lattice(reshape([1.0], 1, 1); orbitals = (:A, :B))
FL_M = randn(ComplexF64, FL_L)
fl_matrix(materialize) = hamiltonian(FL_TWO_BAND; fields = (:J,)) do h, p
    hopping!(h, (1,); plus_hc = true, materialize = materialize) do bond, env
        j = env.fields.J[bond.source_cell_index...]
        [0.1j 0.2j; 0.3j 0.4j]
    end
end
FL_MATRIX_PA = build(fl_matrix(false), (FL_L,); fields = (J = FL_M,))
FL_MATRIX_MAT = build(fl_matrix(true), (FL_L,); fields = (J = FL_M,))

function fl_matrix_dense(J)
    L = length(J)
    reference = zeros(ComplexF64, 2L, 2L)
    for n = 1:L
        target = mod1(n + 1, L)
        M = [0.1J[n] 0.2J[n]; 0.3J[n] 0.4J[n]]
        for β = 1:2, α = 1:2
            reference[fl_flat(α, target, 2), fl_flat(β, n, 2)] += M[α, β]
            reference[fl_flat(β, n, 2), fl_flat(α, target, 2)] += conj(M[α, β])
        end
    end
    reference
end

# --- uniform coefficients ---------------------------------------------------
FL_PHASE = exp(0.3im)
FL_UNIFORM_MODEL = hamiltonian(FL_CHAIN; parameters = (t = 1.0,)) do h, p
    hopping!(h, (1,), uniform(env -> -env.parameters.t * FL_PHASE); plus_hc = true)
end
FL_UNIFORM_H = build(FL_UNIFORM_MODEL, (FL_L,))

function fl_uniform_dense(L, value)
    reference = zeros(ComplexF64, L, L)
    for n = 1:L
        target = mod1(n + 1, L)
        reference[target, n] += value
        reference[n, target] += conj(value)
    end
    reference
end

# uniform materialized with narrowed dependencies: value reads t and g but
# only declares t, so mutating g leaves the cache intentionally stale.
FL_NARROW_MODEL = hamiltonian(FL_CHAIN; parameters = (t = 1.0, g = 0.5)) do h, p
    hopping!(
        h,
        (1,),
        uniform(
            env -> env.parameters.t + env.parameters.g;
            materialize = true,
            depends_on = (:t,),
        );
        plus_hc = true,
    )
end
FL_NARROW_H = build(FL_NARROW_MODEL, (6,))

# --- materialized-real cache that turns complex on refresh ------------------
FL_REALCACHE_MODEL = @test_logs (:warn, r"cannot verify Hermiticity") hamiltonian(
    FL_CHAIN;
    parameters = (μ = 0.5,),
) do h, p
    onsite!(h, uniform(env -> -env.parameters.μ; materialize = true))
end
FL_REALCACHE_H = build(FL_REALCACHE_MODEL, (4,))

@testset "Frontend local coefficients" begin
    @testset "Anderson disorder" begin
        reference = fl_anderson_dense(FL_W, 1.0)
        fl_compare(FL_ANDERSON_PA, reference)
        fl_compare(FL_ANDERSON_MAT, reference)
        ψ = randn(ComplexF64, FL_L)
        @test FL_ANDERSON_PA * ψ ≈ FL_ANDERSON_MAT * ψ
    end

    @testset "Complex bond disorder with plus_hc" begin
        reference = fl_bond_dense(FL_J)
        fl_compare(FL_BOND_PA, reference)
        fl_compare(FL_BOND_MAT, reference)
        @test ishermitian(Matrix(SparseArrays.sparse(FL_BOND_MAT)))
    end

    @testset "Matrix-valued local callback" begin
        reference = fl_matrix_dense(FL_M)
        fl_compare(FL_MATRIX_PA, reference)
        fl_compare(FL_MATRIX_MAT, reference)
    end

    @testset "Uniform coefficient" begin
        fl_compare(FL_UNIFORM_H, fl_uniform_dense(FL_L, -FL_PHASE))
        # per-apply uniform values track parameter updates immediately
        set_parameter!(FL_UNIFORM_H; t = 2.5)
        fl_compare(FL_UNIFORM_H, fl_uniform_dense(FL_L, -2.5 * FL_PHASE))
        set_parameter!(FL_UNIFORM_H; t = 1.0)
    end

    @testset "Materialized uniform: narrowed dependency refresh" begin
        fl_compare(FL_NARROW_H, fl_uniform_dense(6, 1.5))
        # g is not a declared dependency: the cache stays stale by design
        set_parameter!(FL_NARROW_H; g = 5.0)
        fl_compare(FL_NARROW_H, fl_uniform_dense(6, 1.5))
        # explicit refresh picks up the new g
        refresh_caches!(FL_NARROW_H)
        fl_compare(FL_NARROW_H, fl_uniform_dense(6, 6.0))
        # a declared dependency triggers the refresh automatically
        set_parameter!(FL_NARROW_H; t = 2.0)
        fl_compare(FL_NARROW_H, fl_uniform_dense(6, 7.0))
        set_parameter!(FL_NARROW_H; t = 1.0, g = 0.5)
    end

    @testset "set_field! and refresh_caches!" begin
        new_W = rand(FL_L)
        set_field!(FL_ANDERSON_MAT; W = new_W)
        fl_compare(FL_ANDERSON_MAT, fl_anderson_dense(new_W, 1.0))

        # in-place mutation needs an explicit refresh for materialized caches
        fill!(fields(FL_ANDERSON_MAT).W, 0.25)
        refresh_caches!(FL_ANDERSON_MAT)
        fl_compare(FL_ANDERSON_MAT, fl_anderson_dense(fill(0.25, FL_L), 1.0))

        # per-apply coefficients see in-place mutation immediately
        fill!(fields(FL_ANDERSON_PA).W, 0.75)
        fl_compare(FL_ANDERSON_PA, fl_anderson_dense(fill(0.75, FL_L), 1.0))
        set_field!(FL_ANDERSON_PA; W = copy(FL_W))
        set_field!(FL_ANDERSON_MAT; W = copy(FL_W))

        @test_throws ErrorException set_field!(FL_ANDERSON_MAT; W = zeros(Int, FL_L))
        @test_throws ErrorException set_field!(FL_ANDERSON_MAT; V = FL_W)
    end

    @testset "Zero-allocation mul! for materialized models" begin
        ψin = randn(ComplexF64, FL_L)
        ψout = similar(ψin)
        mul!(ψout, FL_ANDERSON_MAT, ψin)
        @test (@allocated mul!(ψout, FL_ANDERSON_MAT, ψin)) == 0
        mul!(ψout, FL_BOND_MAT, ψin)
        @test (@allocated mul!(ψout, FL_BOND_MAT, ψin)) == 0
    end

    @testset "Real materialized cache rejects complex refresh" begin
        fl_compare(FL_REALCACHE_H, Matrix((-0.5 + 0.0im) * I, 4, 4))
        @test_throws ErrorException set_parameter!(FL_REALCACHE_H; μ = 1.0 + 2.0im)
    end
end
