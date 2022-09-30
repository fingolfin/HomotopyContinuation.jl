using HomotopyContinuation
using HomotopyContinuation: ComplexDF64
import Arblib
import Arblib: Acb, Arb, AcbMatrix, AcbVector
using LinearAlgebra
include("test/test_systems.jl")

using BenchmarkTools
include("src/new_tracker/adaptive_tracker.jl")

F = four_bar()

q = read_parameters("four_bar_params_start.txt");
p = read_parameters("four_bar_params_target.txt");
xs = read_solutions("four_bar_sols.txt");
bad_paths = [
    135,
    779,
    845,
    861,
    1250,
    1450,
    1950,
    1979,
    2138,
    2365,
    2698,
    3050,
    3194,
    3566,
    3589,
    3709,
    3970,
    4643,
    4828,
    4997,
    5275,
    5567,
    5619,
    6139,
    6535,
    6910,
    7100,
    7480,
    7937,
    8196,
]

H = ParameterHomotopy(F, q, p; compile = false);
x₀ = xs[1];
tracker = AdaptivePathTracker(H; predictor_order = 3);
@time S = adaptive_track(tracker, xs[1:200]);

findall(s -> s.code !== :success, S)


@time r = adaptive_track(tracker, xs[1])
@benchmark adaptive_track($tracker, $(xs[1:100]))



@benchmark adaptive_track($H, $x₀, $S, 5000)
t = 1.0


T = Tracker(H)
@time rs = map(x -> track(T, x), xs);
count(r -> is_success(r), rs)

@benchmark map(x -> track($T, x), $(xs[1:100]))
@benchmark track($(Tracker(H)), $(xs[1]))


u = zeros(ComplexF64, size(H, 1))
tracker_state = AdaptiveTrackerState{ComplexF64}(H)
predictor = PredictorState(H)

x = copy(x₀)
# predicted
x̂ = similar(x)
# corrected
x̄ = similar(x)

# Check if really zero
evaluate!(u, H, xs[1], 1.0)
if norm(u) > 1e-12
    @warn ("Norm of given value: ", norm(u))
end

t = 1.0
successes = 0
Δt = -1e-2

iter = 0
max_iters = 1000
while abs(Δt) > 16 * eps()
    iter += 1

    evaluate_and_jacobian!(u, tracker_state.prec_ComplexF64.M.A, H, x, t)
    HomotopyContinuation.updated!(tracker_state.prec_ComplexF64.M)

    x̂ = predict(H, x, t, Δt, predictor.cache_ComplexF64, tracker_state.prec_ComplexF64)
    t̂ = t + Δt

    (x̄, code) = correct(H, x̂, t̂)
    if code == :success
        x = x̄
        t = t̂
        successes += 1
        if (successes >= 3)
            Δt *= 2
            successes = 0
        end
        Δt = -min(t, abs(Δt))
    else
        successes = 0
        Δt *= 0.5
    end

    (iter > max_iters) && break
end


return x, t, iter






Arblib.Acb(x::ComplexDF64; prec = 256) = Acb(
    Arb(real(x).hi, prec = prec) + Arb(real(x).lo, prec = prec),
    Arb(imag(x).hi, prec = prec) + Arb(imag(x).lo, prec = prec),
)

F = four_bar()
# S = monodromy_solve(F)
# write_solutions("four_bar_sols.txt", solutions(S))
# write_parameters("four_bar_params_start.txt", parameters(S))
# write_parameters("four_bar_params_target.txt", randn(ComplexF64, length(parameters(S))))

q = read_parameters("four_bar_params_start.txt");
p = read_parameters("four_bar_params_target.txt");
xs = read_solutions("four_bar_sols.txt");

H = ParameterHomotopy(F, q, p; compile = false);
x = ComplexDF64.(xs[1])
t = 1.0
t_start = 1.0
t_target = 0.0
u = zeros(ComplexDF64, size(H, 1))

evaluate!(u, H, xs[1], 1.0)

Base.@kwdef struct PredictorPrecisionState
    cache_ComplexF64::PredictorCache{ComplexF64}
end

Base.@kwdef mutable struct PredictorCache{T}
    tx⁰::TaylorVector{1,T}
    tx¹::TaylorVector{2,T}
    tx²::TaylorVector{3,T}
    tx³::TaylorVector{4,T}
    t::ComplexF64 = complex(NaN)

    u::Vector{T}
    x::Vector{T}
end

function PredictorCache{T}(H::AbstractHomotopy) where {T}
    m, n = size(H)
    tx³ = TaylorVector{4}(T, n)
    PredictorCache(
        tx⁰ = TaylorVector{1}(tx³),
        tx¹ = TaylorVector{2}(tx³),
        tx² = TaylorVector{3}(tx³),
        tx³ = tx³,
        x = zeros(ComplexF64, n),
        u = zeros(ComplexF64, m),
    )
end


function predict(H, x, t, Δt)
    m, n = size(H)
    u = similar(x, n)
    x̂ = similar(x)
    tx³ = TaylorVector{4}(eltype(x), n)
    tx² = TaylorVector{3}(tx³)
    tx¹ = TaylorVector{2}(tx³)
    tx⁰ = TaylorVector{1}(tx³)
    x⁰, x¹, x², x³ = vectors(tx³)

    H_x = similar(x, m, n)
    evaluate_and_jacobian!(u, H_x, H, x, t)

    x⁰ .= x

    taylor!(u, Val(1), H, x, t)
    x¹ .= -(H_x \ u)

    taylor!(u, Val(2), H, tx¹, t)
    x² .= -(H_x \ u)

    taylor!(u, Val(3), H, tx², t)
    x³ .= -(H_x \ u)

    for (i, (xi, xi¹, xi², xi³)) in enumerate(tx³)
        δᵢ = 1 - Δt * xi³ / xi²
        x̂[i] = xi + Δt * (xi¹ + Δt * xi² / δᵢ)
    end


    x̂
end

function correct(H, x̂, t̂)
    m, n = size(H)
    u = similar(x̂, m)
    H_x = similar(x̂, m, n)

    Δx̄ = similar(x̂, m)
    x̄ = copy(x̂)


    for iter = 1:3
        evaluate_and_jacobian!(u, H_x, H, x̄, t̂)
        Δx̄ = H_x \ u
        x̄ = x̄ - Δx̄
        if norm(ComplexF64.(Δx̄)) < 1e-11 * norm(ComplexF64.(x̄))
            return (x̄, :success)
        end
    end
    return (x̄, :failure)
end

function adaptive_track(H::AbstractHomotopy, x₀; max_iters = 5000)
    u = zeros(ComplexF64, size(H, 1))
    predictor = HomotopyContinuation.Predictor(H)
    init!(predictor)
    x = copy(x₀)
    # predicted
    x̂ = similar(x)
    # corrected
    x̄ = similar(x)

    # Check if really zero
    evaluate!(u, H, xs[1], 1.0)
    if norm(u) > 1e-12
        @warn ("Norm of given value: ", norm(u))
    end

    t = 1.0
    successes = 0
    Δt = -1e-2

    iter = 0
    while abs(Δt) > 16 * eps()
        iter += 1

        x̂ = predict(H, x, t, Δt)
        t̂ = t + Δt

        (x̄, code) = correct(H, x̂, t̂)
        if code == :success
            x = x̄
            t = t̂
            successes += 1
            if (successes >= 3)
                Δt *= 2
                successes = 0
            end
            Δt = -min(t, abs(Δt))
        else
            successes = 0
            Δt *= 0.5
        end

        (iter > max_iters) && break
    end


    return x, t, iter
end

PredictorCache{ComplexF64}(H)

@time (y, s, iter) = adaptive_track(H, ComplexDF64.(xs[4]))



x̂ = Arblib.AcbVector(xs[4], prec = 512)
m, n = size(H)
u = similar(x̂, m)
H_x = similar(x̂, m, n)

x̄ = similar(x̂)
x̄ .= x̂

ComplexF64.(x̄)

Arblib.solve!(Arblib.AcbMatrix(x̄), H_x, Arblib.AcbMatrix(u))



@time (y, s, iter) = adaptive_track(H, xs[4])

# STUCK VALUE 
y = ComplexF64[
    -33.213067473353185+196.16004540833595im,
    -4.372213609293898-1.2382436314255036im,
    0.00016494687429948436-0.23993422730066152im,
    -0.013018800881789907-0.23788388709977465im,
    0.06973265575634985-0.043518328548475406im,
    -0.002595700251195619+0.09674979810227978im,
    0.01500781437870132-0.04354957242962884im,
    1.6656964130969656-1.6424036458209432im,
    -0.9782460633689788-0.007355169388765582im,
    -0.17328989701659664+1.2945066183445961im,
    -0.9799614017516131-0.021594169302720706im,
    -0.9681656336296728-0.009581728449883626im,
    -0.9786799692941063-0.010187900339149333im,
    -1.627541258474613-0.18837586094939im,
    -0.9820291843620675-0.0030835896423094334im,
    -0.978383348836015-0.008242659028210843im,
    40.252807070850196+13.947883958413712im,
    -0.6495798739009003-0.548706458043055im,
    22.089837958491632+24.882272894921403im,
    27.8032192067137+8.669392747155422im,
    37.18487730909027+18.24686511263379im,
    -2.4618000603555408+0.4388043673410284im,
    53.054273471980366+9.275104767584017im,
    39.38826902197336+15.400476593822487im,
]
s = 0.6059543885663016

m, n = size(H)
u = similar(y, m)
H_x = similar(y, m, n)
evaluate_and_jacobian!(u, H_x, H, y, s)

m, n = size(H)
Du = ComplexDF64.(similar(y, m))
DH_x = ComplexDF64.(similar(y, m, n))
evaluate_and_jacobian!(Du, DH_x, H, ComplexDF64.(y), s)
evaluate!(Du, H, ComplexDF64.(y), s)



prec = 53
ACBu = AcbMatrix(similar(y, m, 1); prec = prec)p
ACBH_x = AcbMatrix(similar(y, m, n); prec = prec)
ACBy = AcbVector(y; prec = prec)
evaluate_and_jacobian!(ACBu, ACBH_x, H, ACBy, s)
ACBu


Du
DH_x - H_x




@time evaluate!(u, H, y, 0.0)


track(Tracker(H), xs[4])



A0 = (randn(ComplexF64, 8, 8))
b0 = (randn(ComplexF64, 8))

A = A0 .^ 2
b = b0 .^ 2
A1 = ComplexDF64.(A0) .^ 2
b1 = ComplexDF64.(b0) .^ 2
A2 = Arblib.AcbMatrix(Arblib.AcbMatrix(A0) .^ 2)
b2 = Arblib.AcbMatrix(Arblib.AcbMatrix(b0) .^ 2)



c = A \ b
c1 = A1 \ b1

c2 = Arblib.AcbMatrix(8, 1)
Arblib.solve!(c2, A2, b2)


Arblib.Acb(x::ComplexDF64; prec = 256) = Acb(
    Arb(real(x).hi, prec = prec) + Arb(real(x).lo, prec = prec),
    Arb(imag(x).hi, prec = prec) + Arb(imag(x).lo, prec = prec),
)

c2 - c1



c - c0



function gen_taylor_code(M, T)
    taylor_field = Symbol(:taylor_, T)
    order_field = Symbol(:order_, M)
    quote
        I = F.$(taylor_field).$(order_field)
        if isnothing(I)
            I′ = interpreter(TruncatedTaylorSeries{$(M + 1),$T}, F.$(Symbol(:eval_, T)))
            F.$(taylor_field).$(order_field) = I′
            execute_taylor!(
                u,
                Order,
                I′,
                x,
                p;
                assign_highest_order_only = assign_highest_order_only,
            )
        else
            execute_taylor!(
                u,
                Order,
                I,
                x,
                p;
                assign_highest_order_only = assign_highest_order_only,
            )
        end
        u
    end
end

gen_taylor_code()