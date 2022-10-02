
Base.@kwdef mutable struct CorrectorPrecState{T}
    x̄::Vector{T}
    Δx̄::Vector{T}
    u::Vector{T}
end

Base.@kwdef struct CorrectorState
    cache_ComplexF64::CorrectorPrecState{ComplexF64}
    cache_ComplexDF64::CorrectorPrecState{ComplexDF64}
end
function CorrectorPrecState{T}(H::AbstractHomotopy) where {T}
    m, n = size(H)
    CorrectorPrecState(x̄ = zeros(T, n), Δx̄ = zeros(T, n), u = zeros(T, m))
end
CorrectorState(H::AbstractHomotopy) = CorrectorState(
    cache_ComplexF64 = CorrectorPrecState{ComplexF64}(H),
    cache_ComplexDF64 = CorrectorPrecState{ComplexDF64}(H),
)
function correct!(
    x̄_out,
    H,
    x̂::AbstractVector,
    t,
    corrector_state::CorrectorPrecState{T},
    tracker_state::AdaptiveTrackerPrecisionState{T},
) where {T}
    @unpack u, Δx̄, x̄ = corrector_state
    x̄ .= x̂
    norm = WeightedNorm(InfNorm(), x̄)
    HomotopyContinuation.init!(norm, x̄)
    norm_Δx₀ = 0.0
    ω = NaN
    for iter = 0:3
        evaluate_and_jacobian!(u, tracker_state.M, H, x̄, t)
        HomotopyContinuation.updated!(tracker_state.M)
        LA.ldiv!(Δx̄, tracker_state.M, u)
        # δ = HomotopyContinuation.mixed_precision_iterative_refinement!(
        #     Δx̄,
        #     tracker_state.M,
        #     u,
        #     norm,
        # )
        # @show δ
        # δ = HomotopyContinuation.mixed_precision_iterative_refinement!(
        #     Δx̄,
        #     tracker_state.M,
        #     u,
        #     norm,
        # )
        @show norm(x̄)
        (iter == 0) && (norm_Δx₀ = norm(Δx̄))
        (iter == 1) && (ω = 2 * norm(Δx̄) / norm_Δx₀^2)
        x̄ .= x̄ .- Δx̄

        if norm(Δx̄) < 1e-12 * norm(x̄)
            x̄_out .= x̄
            return (x̄_out, :success, ω, norm_Δx₀)
        end
    end
    x̄_out .= x̄
    return (x̄_out, :failure, ω, norm_Δx₀)
end