struct SolverOptions
    endgame_start::Float64
    abstol::Float64
    refinement_maxiters::Int
    verbose::Bool
end

struct Solver{
    H<:AbstractHomotopy,
    P<:Pathtracker,
    E<:Endgamer,
    StartIter}
    homotopy::H
    pathtracker::P
    endgamer::E

    startvalues::StartIter

    options::SolverOptions
end

# This currently relies on the fact that we can keep all solutions in memory. This could
# not be viable for big systems...
# The main problem for a pure streaming / iterator solution is the pathcrossing check
# We need the @inline here for the typeinference
function solve(solver::Solver)
    @unpack options, pathtracker, endgamer, startvalues = solver
    @unpack endgame_start = options

    # TODO: pmap. Will this preserve the order of the arguments? Otherwise we have to
    # return a tuple or something like that
    endgame_start_results = map(startvalues) do startvalue
        run!(pathtracker, startvalue, 1.0, endgame_start)
        # do we need informations about  condition_jacobian?
        PathtrackerResult(pathtracker, false)
    end


    # TODO: Rerun failed paths with higher precision.

    # TODO: We can add a first pathcrossing check here:
    # Check whether two endgame_start_results solutions are "close".
    # Since the paths should all be unique this should not happen
    # If we find two of them we should rerun them with tighter bounds

    # TODO: pmap. Will this preserve the order of the arguments? Otherwise we have to
    # return a tuple or something like that
    if endgame_start > 0.0
        endgame_results = map(endgame_start_results) do result
            if result.retcode == :success
                run!(endgamer, result.solution, endgame_start)
                EndgamerResult(endgamer)
            else
                # We just carry over the result from the failed path.
                # This should not happen since we catch things above, but you never know...
                EndgamerResult(endgamer, result)
            end
        end
    else
        # we just carry over the results to make the rest of the code clearer
        endgame_results = map(r -> EndgamerResult(endgamer, r), endgame_start_results)
    end

    # TODO: We can do a second pathcrossing check here:
    # The cauchy endgame will give us the winding number. This gives the multiplicity.
    # So we should have a match between the winding number and the number of solutions
    # at a given point. Otherwise some pathcrossing happend.

    # Refine solution pass

    results = map(startvalues, endgame_start_results, endgame_results) do s, esr, er
        refine_and_pathresult(s, esr, er, pathtracker, options.abstol, options.refinement_maxiters)
    end

    # Return solution
    Result(results)
end

function refine_and_pathresult(
    startvalue,
    endgame_start_result::PathtrackerResult{T},
    endgamer_result::EndgamerResult,
    pathtracker,
    abstol,
    refinement_maxiters) where T
    @unpack retcode, solution, windingnumber = endgamer_result

    # we refine the solution if possible
    if retcode == :success
        solution = refinesolution(solution, pathtracker, windingnumber, abstol, refinement_maxiters)
    end

    residual, newton_residual, condition_jacobian = residual_estimates(solution, pathtracker)

    # check whether startvalue was affine and our solution is projective
    N = length(startvalue)
    if length(solution) == N + 1
        # make affine
        # This is a more memory efficient variant from:
        # solution = solution[2:end] / solution[1]
        homog_var = solution[1]
        for i=2:N+1
            solution[i - 1] = solution[i] / homog_var
        end
        resize!(solution, N)

        homogenous_coordinate_magnitude = norm(homog_var)
    else
        homogenous_coordinate_magnitude = 1.0
    end

    PathResult{T}(
        retcode,
        solution,
        residual,
        newton_residual,
        condition_jacobian,
        windingnumber,
        homogenous_coordinate_magnitude,
        copy(startvalue),
        endgame_start_result.iterations,
        endgamer_result.iterations,
        endgamer_result.npredictions
        )
end


function refinesolution(solution, tracker::Pathtracker, windingnumber, abstol, maxiters)
    @unpack H, J_H!, cache = tracker.low
    # TODO: we should switch to a higher precision if necessary
    # Since we have the winding number available
    # See the cauchy endgame test, the refinement is nearly useless...

    sol = copy(solution)
    correct!(sol, 0.0, H, J_H!, cache, abstol, maxiters)
    sol
end

function residual_estimates(solution, tracker::Pathtracker{Low}) where Low
    @unpack H, J_H! = tracker.low
    jacobian = J_H!(zeros(Complex{Low}, length(H), nvariables(H)), solution, 0.0)
    res = evaluate(H, solution, 0.0)
    residual = norm(res)
    newton_residual::Float64 = norm(jacobian \ res)
    condition_jacobian::Float64 = cond(jacobian)

    residual, newton_residual, condition_jacobian
end
