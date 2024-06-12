
import LinearAlgebra
import LinearAlgebra: mul!
import SparseArrays

# should we go ahead and implement AbstractMatrix?
function Base.size(H::LatticeHamiltonian)
    l = H.d * prod(H.L)
    l, l
end

function Base.length(H::LatticeHamiltonian)
    (H.d * prod(H.L))^2
end

function Base.eltype(::LatticeHamiltonian) #Hardcoded as ComplexF64 for the moment.
    ComplexF64
end

function Base.adjoint(H::LatticeHamiltonian) #Hardcoding that the matrix is Hermitian!!
    H
end

"""
    mul!(ψout::AbstractVector, H::LatticeHamiltonian, ψin::AbstractVector)

Efficiently perform the matrix multiplaction `H*ψin`, writing the result to `ψout`.
"""
function LinearAlgebra.mul!(
    ψout::AbstractVector,
    H::LatticeHamiltonian,
    ψin::AbstractVector,
)
    H.apply!(ψout, ψin, H.d, H.L, H.params)
end

"""
    *(H::LatticeHamiltonian, ψ::AbstractVector)

Overloaded matrix multiplication operator for `LatticeHamiltonian`s
using the `mul!` function.
"""
function *(H::LatticeHamiltonian, ψ::AbstractVector)
    v = Array{eltype(ψ)}(undef, length(ψ))
    mul!(v, H, ψ)
    v
end

"""
    sparse(H::LatticeHamiltonian)

Return the sparse representation of the Hamiltonian matrix.
"""
function SparseArrays.sparse(H::LatticeHamiltonian)
    H.sparse(H.d, H.L, H.params)
end