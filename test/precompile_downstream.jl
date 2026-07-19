# A downstream package must be able to precompile while using
# LatticeHamiltonians, and — after loading from the precompile cache in a
# fresh session — build and apply Hamiltonians at runtime. The supported
# pattern: store models/builders as top-level consts, build at runtime.
# On the old eval backend this test failed twice over: the in-function
# @lattice_hamiltonian expansion aborted downstream precompilation with
# "evaluation into the closed module", and the model path, though it
# precompiled, hit a world-age MethodError at first runtime use.

@testset "Downstream package precompilation" begin
    mktempdir() do dir
        pkgdir = joinpath(dir, "RGFDownstream")
        mkpath(joinpath(pkgdir, "src"))

        write(
            joinpath(pkgdir, "Project.toml"),
            """
            name = "RGFDownstream"
            uuid = "8f7a1c2e-1111-4a5b-9c3d-0d9e8f7a6b5c"
            version = "0.1.0"

            [deps]
            LatticeHamiltonians = "9e9b75d1-dfe4-4175-a81f-d9649fafcdca"
            LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
            SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
            """,
        )

        write(
            joinpath(pkgdir, "src", "RGFDownstream.jl"),
            """
            module RGFDownstream

            using LatticeHamiltonians
            using LinearAlgebra, SparseArrays

            const CHAIN = Lattice(reshape([1.0], 1, 1); orbitals = (:c,))
            const MODEL = hamiltonian(CHAIN; parameters = (t = 1.0, μ = 0.25)) do h, p
                onsite!(h, -p.μ)
                hopping!(h, (1,), -p.t; plus_hc = true)
            end

            function check(L = 10)
                H = build(MODEL, (L,))
                ψ = randn(ComplexF64, L)
                matrix_free = H * ψ
                against_sparse = SparseArrays.sparse(H) * ψ

                H2 = @lattice_hamiltonian begin
                    L = [10]
                    (0) -> [0 t; t 0]
                    (1) -> [0 t; 0 0]
                    (-1) -> [0 0; t 0]
                    t = 1.0
                end
                ψ2 = randn(ComplexF64, size(H2)[1])
                macro_free = H2 * ψ2
                macro_sparse = SparseArrays.sparse(H2) * ψ2

                matrix_free ≈ against_sparse && macro_free ≈ macro_sparse
            end

            end
            """,
        )

        envdir = joinpath(dir, "env")
        package_root = dirname(@__DIR__)
        setup = """
        using Pkg
        Pkg.develop(path = $(repr(package_root)); io = devnull)
        Pkg.develop(path = $(repr(pkgdir)); io = devnull)
        Pkg.precompile("RGFDownstream")
        """
        usage = """
        using RGFDownstream
        RGFDownstream.check() || error("downstream check failed after cache load")
        """

        # Pkg.test narrows JULIA_LOAD_PATH to the test sandbox; the
        # subprocesses need the default load path to see stdlibs and the
        # temp environment.
        env = copy(ENV)
        env["JULIA_LOAD_PATH"] = join(["@", "@stdlib"], Sys.iswindows() ? ";" : ":")
        delete!(env, "JULIA_PROJECT")
        cmd(code) = setenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$envdir -e $code`,
            env;
            dir = dir,
        )
        # Two sessions: the first precompiles the downstream package, the
        # second loads it from the cache and builds/applies at runtime.
        @test success(pipeline(cmd(setup); stderr = stderr))
        @test success(pipeline(cmd(usage); stderr = stderr))
    end
end
