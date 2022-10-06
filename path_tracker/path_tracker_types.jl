
###
### Options and Parameters
###

"""
    PathTrackerParameters

Parameters that control the performance and robustness characteristics of the path tracking
algorithm.
"""
Base.@kwdef mutable struct PathTrackerParameters
    ε::Float64 = 1e-14
    N::Int = 3
    β::Float64 = 0.8
end
Base.show(io::IO, TP::PathTrackerParameters) = print_fieldnames(io, TP)

"The default [`PathTrackerParameters`](@ref) which have a good balance between robustness and efficiency."
const DEFAULT_TRACKER_PARAMETERS = PathTrackerParameters()


"""
    PathTrackerOptions(; options...)

The set of options for a [`PathTracker`](@ref).

## Options

* `automatic_differentiation = 1`: The value `automatic_differentiation` determines
  up to which order the derivative is computed using automatic differentiation.
  Otherwise numerical differentiation is used. The automatic differentiation results
  in additional compilation time, however for numerically challenging paths it is strongly
  recommended to use `automatic_differentiation = 3`.
* `max_steps = 10_000`: The maximal number of steps a Pathtracker attempts
* `max_step_size = Inf`: The maximal size of a step
* `max_initial_step_size = Inf`: The maximal size of the first step
* `min_step_size = 1e-48`: The minimal step size. If a smaller step size would
  be necessary, then the tracking gets terminated.
* `extended_precision = true`: Whether to allow for the use of extended precision,
  if necessary, in some computations. This can greatly improve the ability to track
  numerically difficult paths.
* `terminate_cond = 1e13`: If the relative component-wise condition number
  `cond(H_x, ẋ)` is larger than `terminate_cond` then the path is terminated as too
  ill-conditioned.
* `parameters::Union{Symbol,TrackerParameters} = :default` Set the
  [`PathTrackerParameters`](@ref) to control the performance of the path tracking algorithm.
  The values `:default`, `:conservative` and `:fast` are shorthands for using
  [`DEFAULT_TRACKER_PARAMETERS`](@ref), [`CONSERVATIVE_TRACKER_PARAMETERS`](@ref) resp.
  [`FAST_TRACKER_PARAMETERS`](@ref).
"""
Base.@kwdef mutable struct PathTrackerOptions
    max_steps::Int = 10_000
    max_step_size::Float64 = Inf
    max_initial_step_size::Float64 = Inf
    extended_precision::Bool = true
    min_step_size::Float64 = 1e-48
    min_rel_step_size::Float64 = 0.0
    terminate_cond::Float64 = 1e13
    parameters::PathTrackerParameters = DEFAULT_TRACKER_PARAMETERS
end
Base.show(io::IO, opts::PathTrackerOptions) = print_fieldnames(io, opts)
Base.show(io::IO, ::MIME"application/prs.juno.inline", opts::PathTrackerOptions) = opts


###
### PathTrackerCode
###

@doc """
    PathTrackerCode

The possible states a `CoreTracker` can have are of type `PathTrackerCode.codes` and can be

* `PathTrackerCode.success`: Indicates a successfull tracking.
* `PathTrackerCode.tracking`: The tracking is still in progress.
* `PathTrackerCode.terminated_accuracy_limit`: Tracking terminaed since the accuracy was insufficient.
* `PathTrackerCode.terminated_invalid_startvalue`: Tracking terminated since the provided start value was invalid.
* `PathTrackerCode.terminated_ill_conditioned`: Tracking terminated since the path was too ill-conditioned.
* `PathTrackerCode.terminated_max_steps`: Tracking terminated since maximal number of steps is reached.
* `PathTrackerCode.terminated_step_size_too_small`: Trackint terminated since the step size was too small.
* `PathTrackerCode.terminated_unknown`: An unintended error occured. Please consider reporting an issue.
"""
module PathTrackerCode

@enum codes begin
    tracking
    success
    terminated_max_steps
    terminated_accuracy_limit
    terminated_ill_conditioned
    terminated_invalid_startvalue
    terminated_invalid_startvalue_singular_jacobian
    terminated_step_size_too_small
    terminated_unknown
end

end

"""
    is_success(code::PathTrackerCode.codes)

Returns `true` if `code` indicates a success in the path tracking.
"""
is_success(S::PathTrackerCode.codes) = S == PathTrackerCode.success

"""
    is_terminated(code::PathTrackerCode.codes)

Returns `true` if `code` indicates that the path tracking got terminated.
"""
is_terminated(S::PathTrackerCode.codes) =
    S ≠ PathTrackerCode.tracking && S ≠ PathTrackerCode.success

"""
    is_invalid_startvalue(code::PathTrackerCode.codes)

Returns `true` if `code` indicates that the path tracking got terminated since the start
value was not a regular zero.
"""
function is_invalid_startvalue(S::PathTrackerCode.codes)
    S == PathTrackerCode.terminated_invalid_startvalue ||
        S == PathTrackerCode.terminated_invalid_startvalue_singular_jacobian
end

"""
    is_tracking(code::PathTrackerCode.codes)

Returns `true` if `code` indicates that the path tracking is not yet finished.
"""
is_tracking(S::PathTrackerCode.codes) = S == PathTrackerCode.tracking

###
### RESULT
###

"""
    PathTrackerResult

Containing the result of tracking a path with a [`PathTracker`](@ref).

## Fields

* `return_code::Symbol`: A code indicating whether the tracking was successfull (`:success`).
  See [`PathTrackerCode`](@ref) for all possible values.
* `solution::V`: The solution when the tracking stopped.
* `t::ComplexF64`: The value of `t` when the tracking stopped.
* `accuracy::Float64`: Estimate of the relative accuracy of the `solution`.
* `accepted_steps::Int`: Number of steps that got accepted.
* `rejected_steps::Int`: Number of steps that got rejected.
* `extended_precision::Bool`: Indicate whether extended precision is necessary to achieve
  the accuracy of the `solution`.
* `extended_precision_used::Bool`: This is `true` if during the tracking at any point
  extended precision was used.
"""
struct PathTrackerResult
    return_code::Symbol
    solution::Vector{ComplexF64}
    t::ComplexF64
    accuracy::Float64
    accepted_steps::Int
    rejected_steps::Int
    extended_precision::Bool
    extended_precision_used::Bool
    ω::Float64
    μ::Float64
    τ::Float64
end

Base.show(io::IO, result::PathTrackerResult) = print_fieldnames(io, result)
Base.show(io::IO, ::MIME"application/prs.juno.inline", result::PathTrackerResult) = result

"""
    is_success(result::PathTrackerResult)

Returns `true` if the path tracking was successfull.
"""
is_success(R::PathTrackerResult) = R.return_code == :success

"""
    is_invalid_startvalue(result::PathTrackerResult)

Returns `true` if the path tracking failed since the start value was invalid.
You can inspect `result.return_code` to get the exact return code. Possible values
if `is_invalid_startvalue(result) == true` are
* `:terminated_invalid_startvalue_singular_jacobian` indicates that the Jacobian of the homotopy at
  the provided start value is singular, i.e., if it has not full-column rank.
* `:terminated_invalid_startvalue` indicates that the the provided start value is not sufficiently
  close to a solution of the homotopy.
"""
function is_invalid_startvalue(R::PathTrackerResult)
    R.return_code == :terminated_invalid_startvalue ||
        R.return_code == :terminated_invalid_startvalue_singular_jacobian
end

"""
    solution(result::PathTrackerResult)

Returns the solutions obtained by the `PathTracker`.
"""
solution(result::PathTrackerResult) = result.solution

"""
    steps(result::PathTrackerResult)

Returns the number of steps done.
"""
steps(result::PathTrackerResult) = accepted_steps(result) + rejected_steps(result)

"""
    accepted_steps(result::PathTrackerResult)

Returns the number of accepted steps.
"""
accepted_steps(result::PathTrackerResult) = result.accepted_steps

"""
    rejected_steps(result::PathTrackerResult)

Returns the number of rejected_steps steps.
"""
rejected_steps(result::PathTrackerResult) = result.rejected_steps

###
### STATE
###

mutable struct PathTrackerState{M<:AbstractMatrix{ComplexF64}}
    x::Vector{ComplexF64} # current x
    x̂::Vector{ComplexF64} # last prediction
    x̄::Vector{ComplexF64} # candidate for new x
    # internal step size
    segment_stepper::SegmentStepper
    Δs_prev::Float64 # previous step size
    # path tracking algorithm
    accuracy::Float64 # norm(x - x(t))
    ω::Float64 # liptschitz constant estimate, see arxiv:1902.02968
    ω_prev::Float64
    μ::Float64 # limit accuracy
    τ::Float64 # trust region size
    norm_Δx₀::Float64 # debug info only
    extended_prec::Bool
    used_extended_prec::Bool
    refined_extended_prec::Bool
    keep_extended_prec::Bool
    norm::WeightedNorm{InfNorm}

    jacobian::Jacobian{M}
    cond_J_ẋ::Float64 # estimate of cond(H(x(t),t), ẋ(t))
    code::PathTrackerCode.codes

    # statistics
    accepted_steps::Int
    rejected_steps::Int
    last_steps_failed::Int
    ext_accepted_steps::Int
    ext_rejected_steps::Int
end

function PathTrackerState(H, x₁::AbstractVector, norm::WeightedNorm{InfNorm})
    x = convert(Vector{ComplexF64}, x₁)
    x̂ = zero(x)
    x̄ = zero(x)

    segment_stepper = SegmentStepper(1.0, 0.0)
    Δs_prev = 0.0
    accuracy = 0.0
    μ = eps()
    ω = 1.0
    ω_prev = 1.0
    τ = Inf
    norm_Δx₀ = NaN
    used_extended_prec = extended_prec = keep_extended_prec = refined_extended_prec = false
    jacobian = Jacobian(zeros(ComplexF64, size(H)))
    cond_J_ẋ = NaN
    code = PathTrackerCode.tracking
    accepted_steps = rejected_steps = ext_accepted_steps = ext_rejected_steps = 0
    last_steps_failed = 0

    PathTrackerState(
        x,
        x̂,
        x̄,
        segment_stepper,
        Δs_prev,
        accuracy,
        ω,
        ω_prev,
        μ,
        τ,
        norm_Δx₀,
        extended_prec,
        used_extended_prec,
        refined_extended_prec,
        keep_extended_prec,
        norm,
        jacobian,
        cond_J_ẋ,
        code,
        accepted_steps,
        rejected_steps,
        last_steps_failed,
        ext_accepted_steps,
        ext_rejected_steps,
    )
end

Base.show(io::IO, state::PathTrackerState) = print_fieldnames(io, state)
Base.show(io::IO, ::MIME"application/prs.juno.inline", state::PathTrackerState) = state
function Base.getproperty(state::PathTrackerState, sym::Symbol)
    if sym === :t
        return getfield(state, :segment_stepper).t
    elseif sym == :Δt
        return getfield(state, :segment_stepper).Δt
    elseif sym == :t′
        return getfield(state, :segment_stepper).t′
    elseif sym == :last_step_failed
        return getfield(state, :last_steps_failed) > 0
    else # fallback to getfield
        return getfield(state, sym)
    end
end

steps(S::PathTrackerState) = S.accepted_steps + S.rejected_steps
ext_steps(S::PathTrackerState) = S.ext_accepted_steps + S.ext_rejected_steps
