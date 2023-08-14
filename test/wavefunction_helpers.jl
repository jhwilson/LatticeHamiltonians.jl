"""
    plane_wave(n, L)

A one-dimensional plane-wave state on a peridoic lattice, with wavevector 2π`n`/L.
"""
function plane_wave(n::T, L::T) where {T<:Integer}
    exp.(im * 2π * n * collect(1:L) ./ L)
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
