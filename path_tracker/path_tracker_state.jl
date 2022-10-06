import HomotopyContinuation: MatrixWorkspace

struct AdaptiveTrackerPrecisionState{T,M}
    x::Vector{T} # current x
    x̂::Vector{T} # last prediction
    x̄::Vector{T} # candidate for new x
    M::MatrixWorkspace{T,M}
end

Base.@kwdef mutable struct AdaptiveTrackerState{M1,M2}
    prec_ComplexF64::AdaptiveTrackerPrecisionState{ComplexF64,M1}
    prec_ComplexDF64::AdaptiveTrackerPrecisionState{ComplexDF64,M2}

    t::Float64 = 1.0
    Δt::Float64 = -1e-2
    successes::Int = 0
    iter::Int = 0
    last_step_failed::Bool = true
    code::Symbol = :tracking
    η::Float64
    ω::Float64
    τ::Float64 = 0.0
    norm_Δx₀::Float64 = 0.0
    norm::WeightedNorm{InfNorm}
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
        prec_ComplexF64 = AdaptiveTrackerPrecisionState{ComplexF64}(H),
        prec_ComplexDF64 = AdaptiveTrackerPrecisionState{ComplexDF64}(H),
        η = 1.0,
        ω = 1.0,
        norm = WeightedNorm(ones(size(H, 2)), InfNorm()),
    )
end
