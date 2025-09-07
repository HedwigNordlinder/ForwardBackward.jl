
# Parametric functions for type stability
struct SDEProcess{Fμ,Fσ} <: ContinuousProcess
    μ::Fμ   # μ(x,t) -> AbstractVector (length D)
    σ::Fσ   # σ(x,t) -> Number | AbstractVector (D) | AbstractMatrix (D×M)
end
SDEProcess(μ, σ) = SDEProcess{typeof(μ),typeof(σ)}(μ, σ)

# --- helpers ---------------------------------------------------------------

# Generate a D-vector noise with covariance fac^2 * (σ σᵀ):
@inline function _bridge_noise(Σ, fac::T, D::Int) where {T<:Real}
    if Σ isa Number
        return fac .* (Σ .* randn(T, D))
    elseif Σ isa AbstractVector
        @assert length(Σ) == D "σ(x,t) as vector must have length D"
        return fac .* (Σ .* randn(T, D))
    elseif Σ isa AbstractMatrix
        M = size(Σ, 2)
        return fac .* (Σ * randn(T, M))
    else
        throw(ArgumentError("σ(x,t) must return Number, AbstractVector, or AbstractMatrix; got $(typeof(Σ))"))
    end
end

# One tiny guided Brownian-bridge step: x -> x⁺ at time τ+h toward endpoint a at time 1
@inline function _bb_step!(x::AbstractVector{T}, a::AbstractVector{T},
                           τ::T, h::T, μ::Function, σ::Function) where {T<:Real}
    R  = one(T) - τ                    # remaining time to the endpoint
    ρ  = h / R                         # bridge fraction in this step
    fac = sqrt(max(zero(T), h * (one(T) - ρ)))  # variance reduction factor
    μxt = μ(x, τ)
    Σxt = σ(x, τ)
    noise = _bridge_noise(Σxt, fac, length(x))
    @. x = x + μxt*h + ρ*(a - x) + noise
    return nothing
end

# --- API: endpoint-conditioned sampling for SDEProcess ---------------------

# Vector-of-times t[1:N] with X0/X1 tensors D×N
function endpoint_conditioned_sample(X0::ContinuousState{T},
                                     X1::ContinuousState{T},
                                     P::SDEProcess,
                                     t::AbstractVector{T};
                                     Δt::Real = 1e-3) where {T<:Real}
    x0 = tensor(X0)           # D×N
    x1 = tensor(X1)           # D×N
    D, N = size(x0)
    @assert size(x1) == (D, N)
    @assert length(t) == N

    out = similar(x0)
    x   = zeros(T, D)         # work buffer

    @inbounds for n in 1:N
        t_target = t[n]
        if t_target ≤ zero(T)
            @views out[:, n] .= @view x0[:, n]
            continue
        elseif t_target ≥ one(T)
            @views out[:, n] .= @view x1[:, n]
            continue
        end

        @views x .= x0[:, n]
        @views a  =  x1[:, n]

        τ = zero(T)
        # march to t[n] using steps of size ≤ Δt
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            _bb_step!(x, a, τ, h, P.μ, P.σ)
            τ += h
        end
        @views out[:, n] .= x
    end
    return ContinuousState(out)
end

# Two-vector (tF, tB) → per-column t = tF/(tF+tB)
function endpoint_conditioned_sample(X0::ContinuousState{T},
                                     X1::ContinuousState{T},
                                     P::SDEProcess,
                                     tF::AbstractVector{T},
                                     tB::AbstractVector{T};
                                     Δt::Real = 1e-3) where {T<:Real}
    @assert length(tF) == length(tB)
    t = tF ./ (tF .+ tB)
    return endpoint_conditioned_sample(X0, X1, P, t; Δt=Δt)
end

# Scalar t replicated across columns
function endpoint_conditioned_sample(X0::ContinuousState{T},
                                     X1::ContinuousState{T},
                                     P::SDEProcess,
                                     t::T; Δt::Real = 1e-3) where {T<:Real}
    N = size(tensor(X0), 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(t, N); Δt=Δt)
end

# Scalar (tF, tB) replicated across columns
function endpoint_conditioned_sample(X0::ContinuousState{T},
                                     X1::ContinuousState{T},
                                     P::SDEProcess,
                                     tF::T, tB::T; Δt::Real = 1e-3) where {T<:Real}
    t = tF / (tF + tB)
    N = size(tensor(X0), 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(t, N); Δt=Δt)
end
