# --- Process ---------------------------------------------------------------

struct SwitchingBM{T} <: ContinuousProcess
    κ::T            # kept for compatibility; set to 0.0 here (Brownian, no drift)
    σ::T            # diffusion scale
    λ::Function     # state-dependent jump rate λ(x)
end

# minimal fix: default to κ = 0 for the Brownian case
SwitchingBM(σ::T, λ::Function) where {T} = SwitchingBM(zero(T), σ, λ)
