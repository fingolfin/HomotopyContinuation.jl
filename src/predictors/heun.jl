export Heun


"""
    Heun()

The [Heun](https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods#Heun's_method)
predictor of order 2.
"""
struct Heun <: AbstractStatelessPredictor end
struct HeunCache{T} <: AbstractStatelessPredictorCache
    A::Matrix{T}
    mk₁::Vector{T}
    mk₂::Vector{T}
end

function cache(::Heun, H, x, t)
    mk₁ = dt(H, x, t)
    HeunCache(jacobian(H, x, t), mk₁, copy(mk₁))
end
#
function predict!(xnext, ::Heun, cache::HeunCache, H::HomotopyWithCache, x, t, Δt, ẋ)
    mk₁, mk₂ = cache.mk₁, cache.mk₂
    n = length(xnext)
    @inbounds for i=1:n
        mk₁[i] = -ẋ[i]
        xnext[i] = x[i] + Δt * ẋ[i]
    end

    minus_ẋ!(mk₂, H, xnext, t + Δt, cache.A)

    @inbounds for i=1:n
        xnext[i] = x[i] - 0.5 * Δt * (mk₁[i] + mk₂[i])
    end

    nothing
end

order(::Heun) = 3
