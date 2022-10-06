export AbstractPathTracker,
    PathTracker,
    PathTrackerResult,
    PathTrackerOptions,
    PathTrackerParameters,
    PathTrackerCode,
    DEFAULT_TRACKER_PARAMETERS,
    FAST_TRACKER_PARAMETERS,
    CONSERVATIVE_TRACKER_PARAMETERS,
    track,
    init!,
    track!,
    step!,
    start_parameters!,
    target_parameters!,
    status,
    state,
    is_success,
    is_terminated,
    is_invalid_startvalue,
    is_tracking,
    solution,
    steps,
    accepted_steps,
    rejected_steps,
    iterator

include("path_tracker_corrector.jl")
include("path_tracker_predictor.jl")
include("path_tracker_types.jl")

###
### PathTRACKER
###

"""
    PathTracker(H::AbstractHomotopy;
            options = PathTrackerOptions(),
            weighted_norm_options = WeightedNormOptions())

Construct a Pathtracker for the given homotopy `H`. The algorithm computes along the path ``x(t)``
the local derivatives up to order 4.
For `options` see also [`PathTrackerOptions`](@ref).
The algorithm uses as a weighted infinity norm to measure distances.
See also [`WeightedNorm`](@ref).

[^Tim20]: Timme, S. "Mixed Precision Path Tracking for Polynomial Homotopy Continuation". arXiv:1902.02968 (2020)


## Example

We want to solve the system
```julia
@var x y t
F = System([x^2 + y^2 - 3, 2x^2 + 0.5x*y + 3y^2 - 2])
```
using a total degree homotopy and `PathTracker`.
```julia
# construct start system and homotopy
G = System(im * [x^2 - 1, y^2 - 1])
H = StraightLineHomotopy(G, F)
start_solutions = [[1,1], [-1,1], [1,-1], [-1,-1]]
# construct Pathtracker
tracker = PathTracker(H)
# track each start solution separetely
results = track.(tracker, start_solutions)
println("# successfull: ", count(is_success, results))
```
We see that we tracked all 4 paths successfully.
```
# successfull: 4
```
"""
struct PathTracker{H<:AbstractHomotopy,M<:AbstractMatrix{ComplexF64}}
    homotopy::H
    predictor::PathTrackerPredictor
    corrector::PathTrackerCorrector
    # these are mutable
    state::PathTrackerState{M}
    options::PathTrackerOptions
end

Tracker(H::ModelKit.Homotopy; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[], kwargs...) =
    PathTracker(fixed(H; compile = compile); kwargs...)
function PathTracker(
    H::AbstractHomotopy,
    x::AbstractVector = zeros(size(H, 2));
    weighted_norm_options::WeightedNormOptions = WeightedNormOptions(),
    options = PathTrackerOptions(),
)
    if !isa(options, PathTrackerOptions)
        options = PathTrackerOptions(; options...)
    else
        options = deepcopy(options)
    end
    norm = WeightedNorm(ones(size(H, 2)), InfNorm(), weighted_norm_options)
    state = PathTrackerState(H, x, norm)
    predictor = PathTrackerPredictor(H)
    corrector = PathTrackerCorrector(options.parameters.a, state.x, size(H, 1))
    PathTracker(H, predictor, corrector, state, options)
end

Base.show(io::IO, C::PathTracker) = print(io, typeof(C), "()")
Base.show(::IO, ::MIME"application/prs.juno.inline", x::PathTracker) = x
Base.broadcastable(C::PathTracker) = Ref(C)

"""
    state(tracker::PathTracker)

Return the state of the tracker.
"""
state(tracker::PathTracker) = tracker.state

"""
    status(tracker::PathTracker)

Get the current [`PathTrackerCode`](@ref) of `Pathtracker`.
"""

"""
    solution(tracker::PathTracker)

Get the current solution.
"""
solution(T::PathTracker) = get_solution(T.homotopy, T.state.x, T.state.t)

status(tracker::PathTracker) = tracker.state.code


LA.cond(tracker::PathTracker) = tracker.state.cond_J_ẋ

function LA.cond(tracker::PathTracker, x, t, d_l = nothing, d_r = nothing)
    J = tracker.state.jacobian
    evaluate_and_jacobian!(tracker.corrector.r, matrix(J), tracker.homotopy, x, t)
    updated!(J)
    LA.cond(J, d_l, d_r)
end


###############
## Step Size ##
###############


# _h(a) = 2a * (√(4 * a^2 + 1) - 2a)

# intial step size
function initial_step_size(
    state::PathTrackerState,
    predictor::PathTrackerPredictor,
    options::PathTrackerOptions,
)
    # a = options.parameters.β_a * options.parameters.a
    # p = order(predictor)
    # ω = state.ω
    # e = local_error(predictor)
    # if isinf(e)
    #     # don't have anything we can use, so just use a conservative number
    #     # (in all well-behaved cases)
    #     e = 1e5
    # end
    # τ = trust_region(predictor)
    # Δs₁ = nthroot((√(1 + 2 * _h(a)) - 1) / (ω * e), p) / options.parameters.β_ω_p
    # Δs₂ = options.parameters.β_τ * τ
    # Δs = nanmin(Δs₁, Δs₂)
    # min(Δs, options.max_step_size, options.max_initial_step_size)
    error("Not implemented")
end

function update_stepsize!(
    state::PathTrackerState,
    result::PathTrackerCorrectorResult,
    options::PathTrackerOptions,
    predictor::PathTrackerPredictor;
    ad_for_error_estimate::Bool = true,
)
    error("Not implemented")
    # a = options.parameters.β_a * options.parameters.a
    # p = order(predictor)
    # ω = clamp(state.ω + 2(state.ω - state.ω_prev), state.ω, 8state.ω)
    # τ = state.τ #trust_region(predictor)
    # if is_converged(result)
    #     e = local_error(predictor)
    #     Δs₁ = nthroot((√(1 + 2 * _h(a)) - 1) / (ω * e), p) / options.parameters.β_ω_p
    #     Δs₂ = options.parameters.β_τ * τ
    #     if state.use_strict_β_τ || dist_to_target(state.segment_stepper) < Δs₂
    #         Δs₂ = options.parameters.strict_β_τ * τ
    #     end
    #     Δs = min(nanmin(Δs₁, Δs₂), options.max_step_size)
    #     if state.use_strict_β_τ && dist_to_target(state.segment_stepper) < Δs
    #         Δs *= options.parameters.strict_β_τ
    #     end
    #     # increase step size by a factor of at most 10 in one step
    #     Δs = min(Δs, 10 * state.Δs_prev)
    #     if state.last_step_failed
    #         Δs = min(Δs, state.Δs_prev)
    #     end
    # else
    #     j = result.iters - 2
    #     Θ_j = nthroot(result.θ, 1 << j)
    #     h_Θ_j = _h(Θ_j)
    #     h_a = _h(0.5a)
    #     if isnan(Θ_j) ||
    #        result.return_code == NEWT_SINGULARITY ||
    #        isnan(result.accuracy) ||
    #        result.iters == 1 ||
    #        h_Θ_j < h_a

    #         Δs = 0.25 * state.segment_stepper.Δs
    #     else
    #         Δs =
    #             nthroot((√(1 + 2 * _h(0.5a)) - 1) / (√(1 + 2 * _h(Θ_j)) - 1), p) *
    #             state.segment_stepper.Δs
    #     end
    # end
    # propose_step!(state.segment_stepper, Δs)
    # nothing
end


function check_terminated!(state::PathTrackerState, options::PathTrackerOptions)
    if state.extended_prec || !options.extended_precision
        error("not implemented")
        # @unpack a, min_newton_iters = options.parameters
        # tol_acc = a^(2^min_newton_iters - 1) * _h(a)
    else
        tol_acc = Inf
    end


    t′ = state.segment_stepper.t′
    t = state.segment_stepper.t
    if is_done(state.segment_stepper)
        state.code = PathTrackerCode.success
    elseif steps(state) ≥ options.max_steps
        state.code = PathTrackerCode.terminated_max_steps
    elseif state.ω * state.μ > tol_acc
        state.code = PathTrackerCode.terminated_accuracy_limit
    elseif state.segment_stepper.Δs < options.min_step_size
        state.code = PathTrackerCode.terminated_step_size_too_small
    elseif fast_abs(t′ - t) ≤ 2eps(fast_abs(t))
        state.code = PathTrackerCode.terminated_step_size_too_small
    elseif options.min_rel_step_size > 0 &&
           t′ != state.segment_stepper.target &&
           fast_abs(t′ - t) < fast_abs(t) * options.min_rel_step_size
        state.code = PathTrackerCode.terminated_step_size_too_small
    end
    nothing
end

function update_predictor!(tracker::PathTracker, x̂ = nothing, Δs = nothing, t_prev = NaN)
    @unpack predictor, homotopy, state = tracker
    update!(predictor, homotopy, state.x, state.t, state.jacobian, state.norm, x̂)
end

"""
    init!(tracker::PathTracker, x₁, t₁, t₀)

Setup `Pathtracker` to track `x₁` from `t₁` to `t₀`.

    init!(tracker::PathTracker, t₀)

Setup `Pathtracker` to continue tracking the current solution to `t₀`.
This keeps the current state.
"""
init!(tracker::PathTracker, r::PathTrackerResult, t₁::Number = 1.0, t₀::Number = 0.0) =
    init!(tracker, solution(r), t₁, t₀; ω = r.ω, μ = r.μ)

function init!(
    Pathtracker::PathTracker,
    x₁::AbstractVector,
    t₁::Number = 1.0,
    t₀::Number = 0.0;
    ω::Float64 = NaN,
    μ::Float64 = NaN,
    τ::Float64 = Inf,
    max_initial_step_size::Float64 = Inf,
    keep_steps::Bool = false,
    extended_precision::Bool = false,
)
    @unpack state, predictor, corrector, homotopy, options = Pathtracker
    @unpack x, x̄, norm, jacobian = state

    # intialize state
    set_solution!(x, homotopy, x₁, t₁)
    init!(norm, x)
    init!(jacobian)

    init!(state.segment_stepper, t₁, t₀)
    state.Δs_prev = 0.0
    state.accuracy = eps()
    state.ω = 1.0
    state.keep_extended_prec = false
    state.code = PathTrackerCode.tracking
    if !keep_steps
        state.accepted_steps = state.rejected_steps = 0
        state.ext_accepted_steps = state.ext_rejected_steps = 0
    end
    state.last_steps_failed = 0


    # compute ω and limit accuracy μ for the start value
    t = state.t
    if isnan(ω) || isnan(μ)
        error("not implemented")
        # a = options.parameters.a
        # valid, ω, μ = init_newton!(
        #     x̄,
        #     corrector,
        #     homotopy,
        #     x,
        #     t,
        #     jacobian,
        #     norm;
        #     a = a,
        #     extended_precision = extended_precision,
        # )
        # if !valid && !extended_precision
        #     extended_precision = true
        #     valid, ω, μ = init_newton!(
        #         x̄,
        #         corrector,
        #         homotopy,
        #         x,
        #         t,
        #         jacobian,
        #         norm;
        #         a = a,
        #         extended_precision = true,
        #     )
        # end
    else
        valid = true
    end
    state.used_extended_prec = state.extended_prec = extended_precision

    if !isnan(ω)
        state.ω = ω
    end
    if valid
        state.accuracy = μ
        state.μ = max(μ, eps())
    else
        # Let's do some more computations to figure out why
        # the solution is bad (could be non-zero or a singular Solution)
        J = matrix(workspace(jacobian))
        evaluate_and_jacobian!(corrector.r, J, homotopy, x, t)
        if J isa StructArrays.StructArray
            corank = size(J, 2) - LA.rank(Matrix(J), rtol = 1e-14)
        else
            corank = size(J, 2) - LA.rank(J, rtol = 1e-14)
        end

        state.code = if corank > 0
            PathTrackerCode.terminated_invalid_startvalue_singular_jacobian
        else
            PathTrackerCode.terminated_invalid_startvalue
        end

        return false
    end


    # initialize the predictor
    state.τ = τ
    evaluate_and_jacobian!(corrector.r, workspace(jacobian), homotopy, state.x, t)
    updated!(jacobian)

    init!(tracker.predictor)
    update_predictor!(tracker)
    state.τ = trust_region(predictor)
    # compute initial step size
    Δs = initial_step_size(state, predictor, tracker.options)
    Δs = max(min(Δs, max_initial_step_size), tracker.options.min_step_size)
    propose_step!(state.segment_stepper, Δs)
    state.ω_prev = state.ω

    is_tracking(state.code)
end

function init!(tracker::PathTracker, t₀::Number; max_initial_step_size::Float64 = Inf)
    @unpack state, predictor, options = Pathtracker
    state.code = PathTrackerCode.tracking
    init!(state.segment_stepper, state.t, t₀)
    Δs = initial_step_size(state, predictor, tracker.options)
    Δs = min(Δs, max_initial_step_size)
    propose_step!(state.segment_stepper, Δs)
    state.Δs_prev = 0.0

    tracker
end

function update_precision!(tracker::PathTracker, μ_low)
    @unpack homotopy, corrector, state, options = Pathtracker
    @unpack μ, ω, x, t, jacobian, norm = state

    options.extended_precision || return false

    # if state.extended_prec && !state.keep_extended_prec && !isnan(μ_low) && μ_low > μ
    #     # check if we can go low again
    #     if μ_low * ω < a^7 * _h(a)
    #         state.extended_prec = false
    #         state.μ = μ_low
    #     end
    # elseif μ * ω > a^5 * _h(a)
    #     use_extended_precision!(tracker)
    # end

    error("not implemented")
    state.extended_prec
end

function use_extended_precision!(tracker::PathTracker)
    @unpack homotopy, corrector, state, options = tracker
    @unpack μ, ω, x, t, jacobian, norm = state
    @unpack a = options.parameters

    options.extended_precision || return state.μ
    !state.extended_prec || return state.μ

    state.extended_prec = true
    state.used_extended_prec = true
    # do two refinement steps
    for i = 1:2
        μ = extended_prec_refinement_step!(
            x,
            corrector,
            homotopy,
            x,
            t,
            jacobian,
            norm;
            simple_newton_step = false,
        )
    end
    state.μ = max(μ, eps())
    state.μ
end

function refine_current_solution!(tracker; min_tol::Float64 = 4 * eps())
    @unpack homotopy, corrector, state, options = Pathtracker
    @unpack x, x̄, t, jacobian, norm = state

    μ = state.accuracy
    μ̄ = extended_prec_refinement_step!(
        x̄,
        corrector,
        homotopy,
        x,
        t,
        jacobian,
        norm;
        simple_newton_step = false,
    )
    if μ̄ < μ
        x .= x̄
        μ = μ̄
    end
    k = 1
    while (μ > min_tol && k ≤ 3)
        μ̄ = extended_prec_refinement_step!(x̄, corrector, homotopy, x, t, jacobian, norm)
        if μ̄ < μ
            x .= x̄
            μ = μ̄
        end
        k += 1
    end
    μ
end

"""
    step!(tracker::PathTracker, debug::Bool = false)

Perform a single tracking step. Returns `true` if the step was accepted.
"""
function step!(tracker::PathTracker)
    @unpack homotopy, corrector, predictor, state, options = Pathtracker
    @unpack t, Δt, t′, x, x̂, x̄, jacobian, norm = state

    # Use the current approximation of x(t) to obtain estimate
    # x̂ ≈ x(t + Δt) using the predictor
    predict!(x̂, predictor, homotopy, x, t, Δt)

    # Correct the predicted value x̂ to obtain x̄.
    # If the correction is successfull we have x̄ ≈ x(t+Δt).
    result = newton!(
        x̄,
        corrector,
        homotopy,
        x̂,
        t′, # = t + Δt
        jacobian,
        norm;
        ω = state.ω,
        μ = state.μ,
        extended_precision = state.extended_prec,
        first_correction = state.accepted_steps == 0,
    )

    if is_converged(result)
        # move forward
        x .= x̄
        state.Δs_prev = state.segment_stepper.Δs
        step_success!(state.segment_stepper)

        state.accuracy = result.accuracy
        state.μ = max(result.accuracy, eps())
        ω = max(result.ω, 1)
        state.ω_prev = state.ω
        state.ω = ω
        # state.ω = max(result.ω, 0.5 * state.ω, 0.1)
        update_precision!(tracker, result.μ_low)

        if is_done(state.segment_stepper) &&
           options.extended_precision &&
           state.accuracy > 1e-14
            state.accuracy = refine_current_solution!(tracker; min_tol = 1e-14)
            state.refined_extended_prec = true
        end
        update_predictor!(tracker, x̂, state.Δs_prev, t)
        state.τ = trust_region(predictor)

        state.accepted_steps += 1
        state.ext_accepted_steps += state.extended_prec
        state.last_steps_failed = 0

        update!(state.norm, x)
    else
        # Step failed, so we have to try with a new (smaller) step size
        state.rejected_steps += 1
        state.ext_rejected_steps += state.extended_prec
        state.last_steps_failed += 1
    end
    state.norm_Δx₀ = result.norm_Δx₀
    update_stepsize!(state, result, options, predictor; ad_for_error_estimate = false)

    check_terminated!(state, options)

    !state.last_step_failed
end

"""
    track!(tracker::PathTracker, x, t₁ = 1.0, t₀ = 0.0; debug::Bool = false)

The same as [`track`](@ref) but only returns the final [`PathTrackerCode`](@ref).

    track!(tracker::PathTracker, t₀; debug::Bool = false)

Track with `Pathtracker` the current solution to `t₀`. This keeps the current state.
"""
function track!(
    tracker::PathTracker,
    x::AbstractVector,
    t₁ = 1.0,
    t₀ = 0.0;
    ω::Float64 = NaN,
    μ::Float64 = NaN,
    extended_precision::Bool = false,
    τ::Float64 = Inf,
    keep_steps::Bool = false,
    max_initial_step_size::Float64 = Inf,
    debug::Bool = false,
)
    init!(
        tracker,
        x,
        t₁,
        t₀;
        ω = ω,
        μ = μ,
        extended_precision = extended_precision,
        τ = τ,
        keep_steps = keep_steps,
        max_initial_step_size = max_initial_step_size,
    )

    while is_tracking(tracker.state.code)
        step!(tracker, debug)
    end

    tracker.state.code
end

function track!(
    tracker::PathTracker,
    r::PathTrackerResult,
    t₁ = 1.0,
    t₀ = 0.0;
    debug::Bool = false,
)
    track!(
        tracker,
        solution(r),
        t₁,
        t₀;
        debug = debug,
        ω = r.ω,
        μ = r.μ,
        τ = r.τ,
        extended_precision = r.extended_precision,
    )
end
function track!(
    tracker::PathTracker,
    t₀;
    debug::Bool = false,
    max_initial_step_size::Float64 = Inf,
)
    init!(tracker, t₀; max_initial_step_size = max_initial_step_size)

    while is_tracking(tracker.state.code)
        step!(tracker, debug)
    end

    tracker.state.code
end

function PathTrackerResult(H::AbstractHomotopy, state::PathTrackerState)
    PathTrackerResult(
        Symbol(state.code),
        get_solution(H, state.x, state.t),
        state.t,
        state.accuracy,
        state.accepted_steps,
        state.rejected_steps,
        state.extended_prec || state.refined_extended_prec,
        state.used_extended_prec || state.refined_extended_prec,
        state.ω,
        state.μ,
        state.τ,
    )
end

"""
     track(tracker::PathTracker, x::AbstractVector, t₁ = 1.0, t₀ = 0.0; debug::Bool = false)

Track the given solution `x` at `t₁` using `Pathtracker` to a solution at `t₀`.

    track(tracker::PathTracker, r::PathTrackerResult, t₁ = 1.0, t₀ = 0.0; debug::Bool = false)

Track the solution of the result `r` from `t₁` to `t₀`.
"""
@inline function track(tracker::PathTracker, x, t₁ = 1.0, t₀ = 0.0; kwargs...)
    track!(tracker, x, t₁, t₀; kwargs...)
    PathTrackerResult(tracker.homotopy, tracker.state)
end

"""
    start_parameters!(tracker::PathTracker, p)

Set the start parameters of the homotopy of the tracker.
"""
start_parameters!(T::PathTracker, p) = (start_parameters!(T.homotopy, p); T)

"""
    target_parameters!(tracker::PathTracker, p)

Set the target parameters of the homotopy of the tracker.
"""
target_parameters!(T::PathTracker, p) = (target_parameters!(T.homotopy, p); T)

parameters!(T::PathTracker, p, q) = (parameters!(T.homotopy, p, q); T)

# PathIterator #
struct PathIterator{T<:Tracker}
    Pathtracker::T
    t_real::Bool
end
Base.IteratorSize(::Type{<:PathIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:PathIterator}) = Base.HasEltype()

"""
    iterator(tracker::PathTracker, x₁, t₁=1.0, t₀=0.0)

Prepare a Pathtracker to make it usable as a (stateful) iterator. Use this if you want to inspect a specific
path. In each iteration the tuple `(x,t)` is returned.

## Example

Assume you have `PathTracker` `Pathtracker` and you wan to track `x₁` from 1.0 to 0.25:
```julia
for (x,t) in iterator(tracker, x₁, 1.0, 0.25)
    println("x at t=\$t:")
    println(x)
end
```

Note that this is a stateful iterator. You can still introspect the state of the tracker.
For example to check whether the Pathtracker was successfull
(and did not terminate early due to some problem) you can do
```julia
println("Success: ", is_success(status(tracker)))
```
"""
function iterator(tracker::PathTracker, x₁, t₁ = 1.0, t₀ = 0.0; kwargs...)
    init!(tracker, x₁, t₁, t₀; kwargs...)
    PathIterator(tracker, typeof(t₁ - t₀) <: Real)
end

function current_x_t(iter::PathIterator)
    @unpack x, t = iter.tracker.state
    (copy(x), iter.t_real ? real(t) : t)
end

function Base.iterate(iter::PathIterator, state = nothing)
    state === nothing && return current_x_t(iter), 1
    iter.tracker.state.code != PathTrackerCode.tracking && return nothing

    while is_tracking(iter.tracker.state.code)
        step_failed = !step!(iter.tracker)
        step_failed || break
    end
    current_x_t(iter), state + 1
end
