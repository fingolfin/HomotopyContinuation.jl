
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
    norm,
    corrector_state::CorrectorPrecState{T},
    tracker_state::AdaptiveTrackerPrecisionState{T},
) where {T}
    @unpack u, Δx̄, x̄ = corrector_state
    x̄ .= x̂
    norm_Δxᵢ = 0.0
    norm_Δx₀ = 0.0
    ω = NaN
    ε = 1e-12
    for iter = 0:2
        evaluate_and_jacobian!(u, tracker_state.M, H, x̄, t)
        # evaluate!(u, H, ComplexDF64.(x̄), t)
        HomotopyContinuation.updated!(tracker_state.M)
        LA.ldiv!(Δx̄, tracker_state.M, u)
        # HomotopyContinuation.fixed_precision_iterative_refinement!(Δx̄, tracker_state.M, u)
        # HomotopyContinuation.mixed_precision_iterative_refinement!(Δx̄, tracker_state.M, u)
        norm_Δxᵢ = norm(Δx̄)
        # @show δ
        # δ = HomotopyContinuation.mixed_precision_iterative_refinement!(
        #     Δx̄,
        #     tracker_state.M,
        #     u,
        #     norm,
        # )
        (iter == 0) && (norm_Δx₀ = norm_Δxᵢ)
        (iter == 1) && (ω = 2 * norm(Δx̄) / norm_Δx₀^2)
        x̄ .= x̄ .- Δx̄


        if iter >= 1 && ω * norm_Δxᵢ^2 ≤ ε
            x̄_out .= x̄
            return (x̄_out, :success, ω, norm_Δx₀)
        end

    end

    x̄_out .= x̄
    return (x̄_out, :failure, ω, norm_Δx₀)
end