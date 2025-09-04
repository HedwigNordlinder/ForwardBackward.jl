using Distributions, Random

# --- Process ---------------------------------------------------------------

struct SwitchingBM{T} <: ContinuousProcess
    κ::T            # kept for compatibility; set to 0.0 here (Brownian, no drift)
    σ::T            # diffusion scale
    λ::Function     # state-dependent jump rate λ(x)
end

# minimal fix: default to κ = 0 for the Brownian case
SwitchingBM(σ::T, λ::Function) where {T} = SwitchingBM(zero(T), σ, λ)

# --- Helpers ---------------------------------------------------------------

αsλ0(λ0, s) = 0.5*(1 + exp(-2*λ0*s))    # P[even # flips] over span s at frozen rate λ0
βsλ0(λ0, s) = 1 - αsλ0(λ0, s)

kroeneckerdelta(i,j) = (i == j) ? 1 : 0

# One-step, frozen-rate bridge kernel as a 4-component Gaussian mixture
function transition_mixture(process::SwitchingBM, x, a, Δt, t, λ0)
    τ  = 1 - t                          # remaining horizon to T=1
    @assert 0.0 < Δt ≤ τ "Δt must satisfy 0 < Δt ≤ 1 - t"

    # common variance for all mixture components (bridge for Brownian)
    s2 = process.σ^2 * Δt * (τ - Δt) / τ
    s  = sqrt(s2)

    # component means μ_{ij} with i,j ∈ {−1,+1}
    μs = [((τ - Δt)*i*x + Δt*j*a)/τ for i in (-1, +1), j in (-1, +1)]
    μs = vec(μs)  # flatten 2×2 -> 4

    # unnormalized weights w_{ij}
    #   first-leg parity uses i, second-leg parity uses j  (BUGFIX)
    #   normalizer uses ϕ(i x ; j a, σ^2 τ)
    ws = Float64[]
    for i in (-1, +1), j in (-1, +1)
        w_parity1 = αsλ0(λ0, Δt)*kroeneckerdelta(i, +1) + βsλ0(λ0, Δt)*kroeneckerdelta(i, -1)
        w_parity2 = αsλ0(λ0, τ-Δt)*kroeneckerdelta(j, +1) + βsλ0(λ0, τ-Δt)*kroeneckerdelta(j, -1)
        # (BUGFIX) Normal takes std, not variance:
        w_norm    = pdf(Normal(j*a, process.σ*sqrt(τ)), i*x)
        push!(ws, w_parity1 * w_parity2 * w_norm)
    end
    ws_normalised = ws ./ sum(ws)

    comps = [Normal(μ, s) for μ in μs]
    return MixtureModel(comps, ws_normalised)
end

# --- Single ε-step of the bridge (left-point freezing of λ) ----------------

function bridge_step(process::SwitchingBM, Xt::ContinuousState, X1::ContinuousState; Δt = 1e-4, t = 0.0)
    x = Xt.x[1]         # assumes scalar state stored as length-1 vector
    a = X1.x[1]
    λ0 = process.λ(x)   # freeze rate at the left endpoint for this tiny step
    mix = transition_mixture(process, x, a, Δt, t, λ0)
    x′  = rand(mix)
    return ContinuousState([x′])
end

# --- Public: draw X_t | (X_0, X_1 = a) ------------------------------------

"""
    endpoint_conditioned_sample(X0, X1, process::SwitchingBM, t; Δt=1e-3)

Sample `X_t` for the mirror-jump Brownian with state-dependent rate λ(x),
using small frozen-rate bridge steps of size `Δt` from 0 → t.  
Assumes scalar states stored as `ContinuousState([x])`.  
T is fixed to 1.0 (so `t ∈ [0,1)`).

Also provides a vector-of-times method: `endpoint_conditioned_sample(X0, X1, process, t_vec; Δt=...)`
which returns a `ContinuousState` holding samples at each time in `t_vec`.
"""
function endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState,
                                    process::SwitchingBM, t::Real; Δt=1e-3)
    @assert 0.0 ≤ t < 1.0 "t must be in [0,1)"
    xstate = X0
    s = 0.0
    while s < t - 1e-12
        δ = min(Δt, t - s)                    # last step may be shorter
        xstate = bridge_step(process, xstate, X1; Δt=δ, t=s)
        s += δ
    end
    return xstate
end

function endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState,
                                    process::SwitchingBM, tvec::AbstractVector{<:Real}; Δt=1e-3)
    @assert all(0.0 .≤ tvec .< 1.0) "all times must be in [0,1)"
    perm   = sortperm(tvec)
    tsort  = sort(tvec)
    out    = similar(tsort, Float64)

    xstate = X0
    s = 0.0
    k = 1
    for target in tsort
        while s < target - 1e-12
            δ = min(Δt, target - s)
            xstate = bridge_step(process, xstate, X1; Δt=δ, t=s)
            s += δ
        end
        out[k] = xstate.x[1]
        k += 1
    end
    # return in original order as a ContinuousState of scalar samples
    invperm = invperm(perm)
    return ContinuousState(out[invperm])
end
