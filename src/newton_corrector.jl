# This file contains the Newton corrector used for the path tracking in tracker.jl
# For a derivation see:
# Mixed Precision Path Tracking for Polynomial Homotopy Continuation,
# Sascha Timme (2020), arXiv:1902.02968

@doc """
    NewtonCorrectorReturnCode

The possible return codes of Newton's method

* `NEWT_CONVERGED`
* `NEWT_TERMINATED`
* `NEWT_MAX_ITERS`
""" @enum NewtonCorrectorCodes begin
    NEWT_CONVERGED
    NEWT_TERMINATED
    NEWT_MAX_ITERS
    NEWT_SINGULARITY
end

struct NewtonCorrectorResult
    return_code::NewtonCorrectorCodes
    accuracy::Float64
    iters::Int
    ω::Float64
    θ::Float64
    μ_low::Float64
    norm_Δx₀::Float64
end

Base.show(io::IO, ::MIME"application/prs.juno.inline", r::NewtonCorrectorResult) = r
Base.show(io::IO, result::NewtonCorrectorResult) = print_fieldnames(io, result)
is_converged(R::NewtonCorrectorResult) = R.return_code == NEWT_CONVERGED

struct NewtonCorrector
    N::Int
    ε::Float64
    Δx::Vector{ComplexF64}
    r::Vector{ComplexF64}
    r̄::Vector{ComplexF64}
    x_extended::Vector{ComplexDF64}
end

function NewtonCorrector(N::Int, ε::Float64, x::AbstractVector{ComplexF64}, m::Int)
    n = length(x)
    Δx = zeros(ComplexF64, n)
    r = zeros(ComplexF64, m)
    r̄ = zero(r)
    x_extended = ComplexDF64.(x)
    NewtonCorrector(N, ε, Δx, r, r̄, x_extended)
end

function extended_prec_refinement_step!(
    x̄::AbstractVector,
    NC::NewtonCorrector,
    H::AbstractHomotopy,
    x::AbstractVector,
    t::Number,
    J::Jacobian,
    norm::AbstractNorm;
    simple_newton_step::Bool = true,
)
    @unpack Δx, r, x_extended = NC
    evaluate_and_jacobian!(r, matrix(J), H, x, t)
    x_extended .= x
    evaluate!(r, H, x_extended, t)
    LA.ldiv!(Δx, updated!(J), r, norm)
    iterative_refinement!(Δx, J, r, norm; tol = 1e-8, max_iters = 3)
    x̄ .= x .- Δx
    if simple_newton_step
        x_extended .= x̄
        evaluate!(r, H, x_extended, t)
        LA.ldiv!(Δx, J, r, norm)
    end
    norm(Δx)
end

function newton!(
    x̄::AbstractVector,
    NC::NewtonCorrector,
    H::AbstractHomotopy,
    x₀::AbstractVector,
    t::Number,
    J::Jacobian,
    norm::AbstractNorm;
    μ::Float64 = throw(UndefKeywordError(:μ)),
    ω::Float64 = throw(UndefKeywordError(:ω)),
    extended_precision::Bool = false,
)
    @unpack Δx, r, r̄, x_extended, N, ε = NC
    x̄ .= x₀
    xᵢ₊₁ = xᵢ = x̄
    Δxᵢ₊₁ = Δxᵢ = Δx
    μ_low = θ = norm_Δxᵢ = norm_Δxᵢ₋₁ = norm_Δx₀ = NaN

    mixed_precision_refinement = extended_precision
    needs_refinement = extended_precision

    for iter = 0:(N-1)
        evaluate_and_jacobian!(r, matrix(J), H, xᵢ, t)
        if extended_precision
            x_extended .= xᵢ
            evaluate!(r, H, x_extended, t)
        end
        LA.ldiv!(Δxᵢ, updated!(J), r, norm)
        if (iter === 0 || needs_refinement)
            (δ, refinment_iters) = iterative_refinement!(
                Δxᵢ,
                J,
                r,
                norm;
                tol = 1e-12,
                max_iters = 4,
                mixed_precision = needs_refinement,
            )
            # if (iter === 0)
            #     if refinment_iters > 1
            #         needs_refinement = true
            #     end
            #     # If not sufficiently precise redo the refinement with mixed precision
            #     if δ > 1e-8 && !mixed_precision_refinement
            #         mixed_precision_refinement = true
            #         iterative_refinement!(
            #             Δxᵢ,
            #             J,
            #             r,
            #             norm;
            #             tol = 1e-8,
            #             max_iters = 3,
            #             mixed_precision = mixed_precision_refinement,
            #         )
            #     end
            # end
        end

        norm_Δxᵢ = norm(Δxᵢ)

        if isnan(norm_Δxᵢ)
            return NewtonCorrectorResult(
                NEWT_SINGULARITY,
                norm_Δxᵢ,
                iter + 1,
                ω,
                θ,
                μ_low,
                norm_Δx₀,
            )
        end

        xᵢ₊₁ .= xᵢ .- Δxᵢ

        iter == 0 && (norm_Δx₀ = norm_Δxᵢ)
        iter == 1 && (ω = 2 * norm_Δxᵢ / norm_Δxᵢ₋₁^2)
        iter >= 1 && (θ = norm_Δxᵢ / norm_Δxᵢ₋₁)
        # TODO: What if first iteration is super exact already?
        if (ω * norm_Δxᵢ^2 ≤ ε)
            evaluate!(r, H, xᵢ, t)
            LA.ldiv!(Δxᵢ₊₁, J, r)
            μ_low = norm(Δxᵢ₊₁)
            if extended_precision
                x_extended .= xᵢ
                evaluate!(r, H, x_extended, t)
                LA.ldiv!(Δxᵢ₊₁, J, r)
            end
            μ = norm(Δxᵢ₊₁)
            xᵢ₊₁ .= xᵢ .- Δxᵢ₊₁

            return NewtonCorrectorResult(NEWT_CONVERGED, μ, iter, ω, θ, μ_low, norm_Δx₀)
        end
        norm_Δxᵢ₋₁ = norm_Δxᵢ
    end

    return NewtonCorrectorResult(NEWT_MAX_ITERS, μ, N + 1, ω, θ, μ_low, norm_Δx₀)
end

function init_newton!(
    x̄::AbstractVector,
    NC::NewtonCorrector,
    H::AbstractHomotopy,
    x₀::AbstractVector,
    t::Number,
    J::Jacobian,
    norm::WeightedNorm;
    extended_precision::Bool = true,
)
    # x₂ = x₁ = x̄ # alias to make logic easier
    @unpack Δx, r, x_extended, ε = NC

    evaluate_and_jacobian!(r, matrix(J), H, x₀, t)
    LA.ldiv!(Δx, updated!(J), r, norm)
    norm_Δx = norm(Δx)
    norm_Δx < sqrt(ε) && return true, norm_Δx
    if extended_precision
        x_extended .= x₀
        evaluate!(r, H, x_extended, t)
        LA.ldiv!(Δx, J, r, norm)
        norm_Δx = norm(Δx)
        norm_Δx < sqrt(ε) && return true, norm_Δx
    end
    return false, norm_Δx


    # end
    # v = norm(Δx) + eps()
    # valid = false
    # ω = μ = NaN
    # ε = sqrt(v)
    # for k = 1:3
    #     x̄ .= x₀ .+ ε .* weights(norm)

    #     evaluate_and_jacobian!(r, matrix(J), H, x̄, t)
    #     if extended_precision
    #         x_extended .= x̄
    #         evaluate!(r, H, x_extended, t)
    #     end
    #     LA.ldiv!(Δx, updated!(J), r, norm)

    #     x₁ .= x̄ .- Δx
    #     norm_Δx₀ = norm(Δx)
    #     if extended_precision
    #         x_extended .= x₁
    #         evaluate!(r, H, x_extended, t)
    #     else
    #         evaluate!(r, H, x₁, t)
    #     end
    #     LA.ldiv!(Δx, J, r, norm)
    #     x₂ .= x₁ .- Δx
    #     norm_Δx₁ = norm(Δx) + eps()
    #     if norm_Δx₁ < a * norm_Δx₀
    #         ω = 2 * norm_Δx₁ / norm_Δx₀^2
    #         μ = norm_Δx₁
    #         if ω * μ > a^7
    #             refined_res = newton!(
    #                 x̄,
    #                 NC,
    #                 H,
    #                 x̄,
    #                 t,
    #                 J,
    #                 norm;
    #                 ω = ω,
    #                 μ = a^7 / ω,
    #                 accurate_μ = true,
    #                 extended_precision = extended_precision,
    #             )
    #             if is_converged(refined_res)
    #                 valid = true
    #                 ω = refined_res.ω
    #                 μ = refined_res.accuracy
    #             else
    #                 valid = false
    #             end
    #         else
    #             valid = true
    #             break
    #         end 
    #     else
    #         ε *= sqrt(ε)
    #     end
    # end

    # (valid = valid, ω = ω, μ = μ)
end
