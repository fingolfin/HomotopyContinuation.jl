import HomotopyContinuation: MatrixWorkspace
import Parameters: @unpack
import LinearAlgebra
import PrettyTables
const LA = LinearAlgebra

include("./tracker_state.jl")
include("./predictor.jl")
include("./corrector.jl")

Base.@kwdef struct AdaptivePathTracker{H<:AbstractHomotopy,M1,M2}
    homotopy::H
    state::AdaptiveTrackerState{M1,M2}
    predictor::Predictor
    corrector_state::CorrectorState
end
function AdaptivePathTracker(H::AbstractHomotopy; predictor_order = 4)
    state = AdaptiveTrackerState(H)
    predictor = Predictor(H; order = predictor_order)
    corrector = CorrectorState(H)
    AdaptivePathTracker(H, state, predictor, corrector)
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


function init!(tracker::AdaptivePathTracker, x₀::AbstractVector)
    @unpack state = tracker
    @unpack x, x̂, x̄ = state.prec_ComplexF64
    x .= x₀
    state.t = 1.0
    state.successes = 0
    state.Δt = -1e-2
    state.iter = 0
    state.last_step_failed = true
    state.code = :tracking
end

function step!(tracker::AdaptivePathTracker)
    @unpack homotopy, state, predictor, corrector_state = tracker
    @unpack x, x̂, x̄ = state.prec_ComplexF64

    if abs(state.Δt) < 16 * eps()
        state.code = iszero(state.t) ? :success : :failed_min_step_size
        return
    end

    state.iter += 1

    # if state.last_step_failed
    evaluate_and_jacobian!(
        corrector_state.cache_ComplexF64.u,
        state.prec_ComplexF64.M.A,
        homotopy,
        x,
        state.t,
    )
    # state.prec_ComplexDF64.M.A .= state.prec_ComplexF64.M.A
    HomotopyContinuation.updated!(state.prec_ComplexF64.M)
    # end

    predict!(
        x̂,
        homotopy,
        x,
        state.t,
        state.Δt,
        predictor.config,
        predictor.state.cache_ComplexF64,
        state.prec_ComplexF64,
    )
    t̂ = state.t + state.Δt
    if t̂ < 16 * eps()
        t̂ = 0.0
    end

    (x̄, code) = correct!(
        x̄,
        homotopy,
        x̂,
        t̂,
        corrector_state.cache_ComplexDF64,
        state.prec_ComplexDF64,
    )
    if code == :success
        x .= x̄
        state.t = t̂
        state.successes += 1
        if (state.successes >= 3)
            state.Δt *= 2
            state.successes = 0
        end
        state.Δt = -min(state.t, abs(state.Δt))
        state.last_step_failed = false
    else
        state.last_step_failed = true
        state.successes = 0
        state.Δt *= 0.5
    end

    if (state.iter > 5000)
        state.code = :failed_max_iters
    end

    return
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
    @unpack homotopy, state, predictor, corrector_state = tracker
    @unpack x, x̂, x̄ = state.prec_ComplexF64

    init!(tracker, x₀)
    while state.code == :tracking
        step!(tracker)

    end


    return AdaptiveTrackerResult(copy(x), state.t, state.code, state.iter)
end

include("./tracker_path_info.jl")