@testset "3D build test" begin
    H = @lattice_hamiltonian begin
        L = [10, 10, 10]
        V = [0, 0]
        (0, 0, 0) -> [0 Δ + 3t ; Δ + 3t 0]
        (1, 0, 0) -> [0 -t ; -t 0] 
        (0, 1, 0) -> [0 -t ; -t 0]
        (-1, 0, 0) -> [0 -t; -t 0]
        (0, -1, 0) -> [0 -t; -t 0]
        (0, 0, 1) -> [0 -t - t2 ; -t + t2 0]
        (0, 0, -1) -> [0 -t + t2 ; -t - t2 0]
        t = 1
        Δ = 1
        t2 = 0.3
    end

    Hsp = sparse(H)
    ψ = randn(ComplexF64, size(H)[1])
    H*ψ
    @test isapprox(maximum(abs.(Hsp * ψ - H * ψ)), 0, atol = 1e-10)
end