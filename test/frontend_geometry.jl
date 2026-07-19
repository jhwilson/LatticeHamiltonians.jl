# Geometry wiring: lattice data flows onto the built Hamiltonian, and context
# helpers report physically correct positions, including across boundaries.

using LatticeHamiltonians: SiteContext, BondContext

FG_SSH_LATTICE =
    Lattice(reshape([1.0], 1, 1); orbitals = (:A, :B), positions = ([0.0], [0.5]))
FG_SSH_MODEL = hamiltonian(FG_SSH_LATTICE; parameters = (t1 = 1.0,)) do h, p
    hopping!(h, (0,), [0 p.t1; 0 0]; plus_hc = true)
end
FG_SSH_H = build(FG_SSH_MODEL, (10,))

FG_GRAPHENE = Lattice(
    [1/2 -1/2; sqrt(3)/2 sqrt(3)/2];
    orbitals = (:A, :B),
    positions = ([0.0, 0.0], [0.0, 1/sqrt(3)]),
)
FG_GRAPHENE_MODEL = hamiltonian(FG_GRAPHENE; parameters = (t = 1.0,)) do h, p
    hopping!(h, (0, 0), [0 p.t; 0 0]; plus_hc = true)
end
FG_GRAPHENE_H = build(FG_GRAPHENE_MODEL, (4, 3))

FG_LEGACY = @lattice_hamiltonian begin
    L = [4]
    (1) -> t
    (-1) -> t
    t = 1.0
end

@testset "Frontend geometry" begin
    @testset "Lattice data on the built Hamiltonian" begin
        @test FG_SSH_H.A == reshape([1.0], 1, 1)
        @test FG_SSH_H.B ≈ reshape([2π], 1, 1)
        @test FG_SSH_H.r == [[0.0], [0.5]]

        A = FG_GRAPHENE.A
        @test FG_GRAPHENE_H.A == A
        @test FG_GRAPHENE_H.B ≈ 2π * transpose(inv(A))
        @test FG_GRAPHENE_H.r == [[0.0, 0.0], [0.0, 1/sqrt(3)]]
    end

    @testset "Legacy macro keeps placeholder geometry" begin
        @test FG_LEGACY.A == [1.0][:, :]
        @test FG_LEGACY.r == [[0.0]]
        @test FG_LEGACY.meta === nothing
    end

    @testset "Site context" begin
        site = SiteContext{1,1}((3,), FG_SSH_LATTICE, (10,))
        @test site.cell_index == (3,)
        @test site.cell == (2,)
        @test position(site, :A) == [2.0]
        @test position(site, :B) == [2.5]
    end

    @testset "Bond context in the bulk" begin
        bond = BondContext{1,1}((1,), (4,), FG_SSH_LATTICE, (10,))
        @test bond.source_cell == (3,)
        @test bond.target_cell == (4,)
        @test bond.target_cell_index == (5,)
        @test bond.crossed_boundary == (0,)
        @test position(bond, Source(), :B) == [3.5]
        @test position(bond, Target(), :A) == [4.0]
        @test displacement(bond, :A, :B) == [0.5]
        @test midpoint(bond, :A, :A) == [3.5]
    end

    @testset "Bond context across the boundary" begin
        bond = BondContext{1,1}((1,), (10,), FG_SSH_LATTICE, (10,))
        @test bond.source_cell == (9,)
        @test bond.target_cell == (10,)      # unwrapped
        @test bond.target_cell_index == (1,) # wrapped
        @test bond.crossed_boundary == (1,)
        # relative geometry stays that of a nearest-neighbor bond
        @test position(bond, Target(), :A) == [10.0]
        @test displacement(bond, :A, :A) == [1.0]
        @test midpoint(bond, :A, :A) == [9.5]

        backward = BondContext{1,1}((-1,), (1,), FG_SSH_LATTICE, (10,))
        @test backward.target_cell == (-1,)
        @test backward.target_cell_index == (10,)
        @test backward.crossed_boundary == (-1,)
    end

    @testset "Bond context on graphene" begin
        A = FG_GRAPHENE.A
        bond = BondContext{2,2}((1, 0), (4, 1), FG_GRAPHENE, (4, 3))
        @test bond.source_cell == (3, 0)
        @test bond.target_cell == (4, 0)
        @test bond.crossed_boundary == (1, 0)
        @test displacement(bond, :A, :A) ≈ A * [1, 0]
        @test displacement(bond, :B, :A) ≈ A * [1, 0] + [0.0, 1/sqrt(3)]
        onsite_bond = BondContext{2,2}((0, 0), (2, 2), FG_GRAPHENE, (4, 3))
        @test displacement(onsite_bond, :B, :A) ≈ [0.0, 1/sqrt(3)]
        @test onsite_bond.crossed_boundary == (0, 0)
    end
end
