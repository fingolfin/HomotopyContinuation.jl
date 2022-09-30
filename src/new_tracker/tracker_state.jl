import HomotopyContinuation: MatrixWorkspace

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
