import HomotopyContinuation: MatrixWorkspace
import Parameters: @unpack
const LA = LinearAlgebra

include("./tracker_state.jl")
include("./predictor.jl")
include("./corrector.jl")

Base.@kwdef struct AdaptivePathTracker{H<:AbstractHomotopy,M1,M2}
    homotopy::H
    tracker_state::AdaptiveTrackerState{M1,M2}
    predictor::Predictor
    corrector_state::CorrectorState
end
function AdaptivePathTracker(H::AbstractHomotopy; predictor_order = 4)
    tracker_state = AdaptiveTrackerState(H)
    predictor = Predictor(H; order = predictor_order)
    corrector = CorrectorState(H)
    AdaptivePathTracker(H, tracker_state, predictor, corrector)
end

Base.@kwdef struct AdaptiveTrackerResult
    x::Vector{ComplexF64}
    t::Float64
    code::Symbol
    iter::Int
end

function Base.show(io::IO, z::AdaptiveTrackerResult)
    print(io, "AdaptiveTrackerResult(", z.code, ",", z.iter, ")")
end

# Possible precision combinations
# All Complex64
# Eval in ComplexDF64
# Eval + Jac in ComplexDF64
# Eval + Jac + Taylor in ComplexDF64
# Eval + Jac + Taylor + LinearAlgebra in ComplexDF64
function adaptive_track(
    tracker::AdaptivePathTracker,
    xs::AbstractVector{T},
    max_iters = 5000,
) where {T<:AbstractVector}
    map(x -> adaptive_track(tracker, x, max_iters), xs)
end

function adaptive_track(
    tracker::AdaptivePathTracker,
    x₀::AbstractVector{T},
    max_iters = 5000,
) where {T<:Number}
    @unpack homotopy, tracker_state, predictor, corrector_state = tracker
    @unpack x, x̂, x̄ = tracker_state.prec_ComplexF64
    x .= x₀


    # Check if really zero
    # evaluate!(corrector.cache_ComplexDF64.u, H, xs[1], 1.0)
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
                homotopy,
                x,
                t,
            )
            HomotopyContinuation.updated!(tracker_state.prec_ComplexF64.M)
        end

        predict!(
            x̂,
            homotopy,
            x,
            t,
            Δt,
            predictor.config,
            predictor.state.cache_ComplexF64,
            tracker_state.prec_ComplexF64,
        )
        t̂ = t + Δt

        (x̄, code) = correct!(
            x̄,
            homotopy,
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


    return AdaptiveTrackerResult(x, t, iszero(t) ? :success : :failed, iter)
end