import HomotopyContinuation: MatrixWorkspace
import Parameters: @unpack
const LA = LinearAlgebra

struct AdaptiveTrackerPrecisionState{T,M}
    x::Vector{T} # current x
    x̂::Vector{T} # last prediction
    x̄::Vector{T} # candidate for new x
    M::MatrixWorkspace{T,M}
end

mutable struct AdaptiveTrackerState{M1,M2}
    prec_ComplexF64::AdaptiveTrackerPrecisionState{ComplexF64,M1}
    prec_ComplexDF64::AdaptiveTrackerPrecisionState{ComplexDF64,M2}
    # internal step size
    # segment_stepper::SegmentStepper
    # Δs_prev::Float64 # previous step size
    # # path tracking algorithm
    # accuracy::Float64 # norm(x - x(t))
    # code::TrackerCode.codes
end

function AdaptiveTrackerPrecisionState{T}(H::AbstractHomotopy) where {T}
    m, n = size(H)
    AdaptiveTrackerPrecisionState(
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        MatrixWorkspace{T}(m, n),
    )
end

function AdaptiveTrackerState(H::AbstractHomotopy)
    AdaptiveTrackerState(
        AdaptiveTrackerPrecisionState{ComplexF64}(H),
        AdaptiveTrackerPrecisionState{ComplexDF64}(H),
    )
end


Base.@kwdef mutable struct PredictorPrecState{T}
    tx⁰::TaylorVector{1,T}
    tx¹::TaylorVector{2,T}
    tx²::TaylorVector{3,T}
    tx³::TaylorVector{4,T}
    tx⁴::TaylorVector{5,T}
    t::ComplexF64 = complex(NaN)

    u::Vector{T}
    x::Vector{T}
end

Base.@kwdef struct PredictorState
    cache_ComplexF64::PredictorPrecState{ComplexF64}
    cache_ComplexDF64::PredictorPrecState{ComplexDF64}
end

function PredictorPrecState{T}(H::AbstractHomotopy) where {T}
    m, n = size(H)
    tx⁴ = TaylorVector{5}(T, n)
    PredictorPrecState(
        tx⁰ = TaylorVector{1}(tx⁴),
        tx¹ = TaylorVector{2}(tx⁴),
        tx² = TaylorVector{3}(tx⁴),
        tx³ = TaylorVector{4}(tx⁴),
        tx⁴ = tx⁴,
        x = zeros(T, n),
        u = zeros(T, m),
    )
end

PredictorState(H) = PredictorState(
    cache_ComplexF64 = PredictorPrecState{ComplexF64}(H),
    cache_ComplexDF64 = PredictorPrecState{ComplexDF64}(H),
)

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


function predict!(
    x̂,
    H,
    x::Vector{T},
    t,
    Δt,
    pred_state::PredictorPrecState{T},
    tracker_state::AdaptiveTrackerPrecisionState{T},
) where {T}
    @unpack tx⁰, tx¹, tx², tx³, tx⁴, u = pred_state
    x⁰, x¹, x², x³, x⁴ = vectors(tx⁴)


    # Assume that we have an up to date + factorized Jacobian
    x⁰ .= x

    taylor!(u, Val(1), H, x, t)
    u .= .-u
    LA.ldiv!(pred_state.x, tracker_state.M, u)
    x¹ .= pred_state.x

    taylor!(u, Val(2), H, tx¹, t)
    u .= .-u
    LA.ldiv!(pred_state.x, tracker_state.M, u)
    x² .= pred_state.x

    taylor!(u, Val(3), H, tx², t)
    u .= .-u
    LA.ldiv!(pred_state.x, tracker_state.M, u)
    x³ .= pred_state.x

    # taylor!(u, Val(4), H, tx³, t)
    # u .= .-u
    # LA.ldiv!(pred_state.x, tracker_state.M, u)
    # x⁴ .= pred_state.x

    for (i, (xᵢ, xᵢ¹, xᵢ², xᵢ³, xᵢ⁴)) in enumerate(tx⁴)
        # δᵢ = 1 - Δt * xᵢ⁴ / xᵢ³
        # x̂[i] = xᵢ + Δt * (xᵢ¹ + Δt * (xᵢ² + Δt * xᵢ³ / δᵢ))
        δᵢ = 1 - Δt * xᵢ³ / xᵢ²
        x̂[i] = xᵢ + Δt * (xᵢ¹ + Δt * xᵢ² / δᵢ)
    end

    x̂
end

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
        if norm(Δx̄) < 1e-11 * norm(x̄)
            return (x̄, :success)
        end
    end
    return (x̄, :failure)
end

struct AllTrackerState2{M1,M2}
    tracker_state::AdaptiveTrackerState{M1,M2}
    predictor_state::PredictorState
    corrector_state::CorrectorState
end
function AllTrackerState2(H::AbstractHomotopy)
    tracker_state = AdaptiveTrackerState(H)
    predictor = PredictorState(H)
    corrector = CorrectorState(H)
    AllTrackerState2(tracker_state, predictor, corrector)
end


function adaptive_track(H::AbstractHomotopy, x₀, state::AllTrackerState2, max_iters = 5000)
    @unpack tracker_state, predictor_state, corrector_state = state
    @unpack x, x̂, x̄ = tracker_state.prec_ComplexF64
    x .= x₀


    # Check if really zero
    # evaluate!(corrector.cache_ComplexF64.u, H, xs[1], 1.0)
    # if norm(u) > 1e-12
    #     @warn ("Norm of given value: ", norm(u))
    # end

    t = 1.0
    successes = 0
    Δt = -1e-2

    iter = 0
    last_iter_success = false
    while abs(Δt) > 16 * eps()
        iter += 1

        if !last_iter_success
            evaluate_and_jacobian!(
                corrector_state.cache_ComplexF64.u,
                tracker_state.prec_ComplexF64.M.A,
                H,
                x,
                t,
            )
            HomotopyContinuation.updated!(tracker_state.prec_ComplexF64.M)
        end

        predict!(
            x̂,
            H,
            x,
            t,
            Δt,
            predictor_state.cache_ComplexF64,
            tracker_state.prec_ComplexF64,
        )
        t̂ = t + Δt

        (x̄, code) = correct!(
            x̄,
            H,
            x̂,
            t̂,
            corrector_state.cache_ComplexF64,
            tracker_state.prec_ComplexF64,
        )
        if code == :success
            x .= x̄
            t = t̂
            successes += 1
            if (successes >= 3)
                Δt *= 2
                successes = 0
            end
            Δt = -min(t, abs(Δt))
            last_iter_success = true
        else
            last_iter_success = false
            successes = 0
            Δt *= 0.5
        end

        (iter > max_iters) && break
    end


    return x, t, iter
end