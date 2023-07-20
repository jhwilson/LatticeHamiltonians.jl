"""
    plane_wave(n, L)

A plane-wave state on a periodic lattice, with wavevector 2π`n`/L.
The `d` dimensional orbital degree of freedom is ignored
"""
function plane_wave(n::Vector{T}, L::Vector{T}, d::T) where {T<:Integer}
    ψ = Array{ComplexF64}(undef, d, L...)
    for i in eachindex(IndexCartesian(), ψ)
        # i.I gives the tuple of indices along each dimension
        # The inner most dimension will be the on site orbital dof which we ignore
        # Next come the spatial dimensions in order
        # We subtract 1 since julia indices are 1 based
        # And then comute the total phase via the dot product
        ψ[i] = exp.(im * 2π * dot(n ./ L, i.I[2:end] .- 1))
    end
    reshape(ψ, d * prod(L))
end

"""
    exponential(n, L, ξ)

An exponentially localized wavefunction at site `n`, with decay length `ξ` on a periodic 1D lattice of length `L`.
"""
function exponential(n, L, ξ)
    exp.(-(abs.(collect(1:L) .- n)) / (2ξ))
end

"""
    gaussian(n, L, σ)

A gaussian localized wavefunction at site `n`, with width `σ` on a periodic 1D lattice of length `L`.
"""
function gaussian(n, L, σ)
    exp.(-((collect(1:L) .- n) .^ 2) / (2σ^2))
end
