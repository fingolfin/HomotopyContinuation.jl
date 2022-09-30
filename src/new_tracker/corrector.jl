
Base.@kwdef mutable struct CorrectorPrecState{T}
    Δx̄::Vector{T}
    u::Vector{T}
end

Base.@kwdef struct CorrectorState
    cache_ComplexF64::CorrectorPrecState{ComplexF64}
    cache_ComplexDF64::CorrectorPrecState{ComplexDF64}
end
function CorrectorPrecState{T}(H::AbstractHomotopy) where {T}
    m, n = size(H)
    CorrectorPrecState(Δx̄ = zeros(T, n), u = zeros(T, m))
end
CorrectorState(H::AbstractHomotopy) = CorrectorState(
    cache_ComplexF64 = CorrectorPrecState{ComplexF64}(H),
    cache_ComplexDF64 = CorrectorPrecState{ComplexDF64}(H),
)
function correct!(
    x̄,
    H,
    x̂::Vector{T},
    t,
    corrector_state::CorrectorPrecState{T},
    tracker_state::AdaptiveTrackerPrecisionState{T},
) where {T}
    @unpack u, Δx̄ = corrector_state
    x̄ .= x̂
    for iter = 1:3
        evaluate_and_jacobian!(u, tracker_state.M, H, x̄, t)
        HomotopyContinuation.updated!(tracker_state.M)
        LA.ldiv!(Δx̄, tracker_state.M, u)
        x̄ .= x̄ .- Δx̄
        if norm(Δx̄) < 1e-6 * norm(x̄)
            return (x̄, :success)
        end
    end
    return (x̄, :failure)
end