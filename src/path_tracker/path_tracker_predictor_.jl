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

Base.@kwdef struct PredictorConfig
    order::Int = 4
end

Base.@kwdef struct Predictor
    config::PredictorConfig
    state::PredictorState
end
Predictor(H; order = 4) =
    Predictor(config = PredictorConfig(order = order), state = PredictorState(H))


function predict!(
    x̂,
    H,
    x::AbstractVector,
    t,
    Δt,
    config::PredictorConfig,
    norm,
    pred_state::PredictorPrecState,
    tracker_state::AdaptiveTrackerPrecisionState,
)
    @unpack tx⁰, tx¹, tx², tx³, tx⁴, u = pred_state
    x⁰, x¹, x², x³, x⁴ = vectors(tx⁴)


    ldiv_and_refine =
        () -> begin
            LA.ldiv!(pred_state.x, tracker_state.M, u)

            HomotopyContinuation.fixed_precision_iterative_refinement!(
                pred_state.x,
                tracker_state.M,
                u,
                norm,
            )

        end
    # Assume that we have an up to date + factorized Jacobian
    x⁰ .= x

    taylor!(u, Val(1), H, ComplexDF64.(x), t)
    u .= .-u
    ldiv_and_refine()
    x¹ .= pred_state.x

    if (config.order == 1)
        x̂ .= x .+ Δt .* x¹
        τ = norm(x¹)
        return τ
    end

    taylor!(u, Val(2), H, tx¹, t)
    u .= .-u
    ldiv_and_refine()
    x² .= pred_state.x

    if (config.order == 2)
        for (i, (xᵢ, xᵢ¹, xᵢ²)) in enumerate(tx²)
            δᵢ = 1 - Δt * xᵢ² / xᵢ¹
            x̂[i] = xᵢ + Δt * xᵢ¹ / δᵢ
        end
        τ = norm(x¹) / norm(x²)
        return τ
    end

    taylor!(u, Val(3), H, tx², t)
    u .= .-u
    ldiv_and_refine()
    x³ .= pred_state.x

    if (config.order == 3)
        for (i, (xᵢ, xᵢ¹, xᵢ², xᵢ³)) in enumerate(tx³)
            δᵢ = 1 - Δt * xᵢ³ / xᵢ²
            x̂[i] = xᵢ + Δt * (xᵢ¹ + Δt * xᵢ² / δᵢ)
        end
        τ = norm(x²) / norm(x³)
        return τ
    end

    taylor!(u, Val(4), H, tx³, t)
    u .= .-u
    ldiv_and_refine()
    x⁴ .= pred_state.x

    for (i, (xᵢ, xᵢ¹, xᵢ², xᵢ³, xᵢ⁴)) in enumerate(tx⁴)
        δᵢ = 1 - Δt * xᵢ⁴ / xᵢ³
        x̂[i] = xᵢ + Δt * (xᵢ¹ + Δt * (xᵢ² + Δt * xᵢ³ / δᵢ))
    end

    τ = norm(x³) / norm(x⁴)
    return τ
end