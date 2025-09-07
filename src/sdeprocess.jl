# Parametric process wrapper for specialization
struct SDEProcess{Fμ,Fσ} <: ContinuousProcess
    μ::Fμ   # (x,t) -> drift vector in ℝ^D
    σ::Fσ   # (x,t) -> Number | Vector (D) | Matrix (D×M) factor
end
SDEProcess(μ, σ) = SDEProcess{typeof(μ),typeof(σ)}(μ, σ)

# --- helpers ---------------------------------------------------------------

# Expand a time scalar/vector to match array dimensionality (like your manifold utils)
expand(t::Real, x) = t
function expand(t::AbstractArray, d::Int)
    ndt = ndims(t)
    d - ndt < 0 && error("Cannot expand array of size $(size(t)) to $d dimensions.")
    reshape(t, ntuple(_ -> 1, d - ndt)..., size(t)...)
end

# Generate a D-vector ξ with covariance fac^2 * (Σ Σᵀ)
@inline function _bridge_noise(Σ, fac::T, D::Int) where {T<:Real}
    if Σ isa Number
        return fac .* (Σ .* randn(T, D))                 # cov = fac^2 σ^2 I
    elseif Σ isa AbstractVector
        @assert length(Σ) == D "σ(x,t) as Vector must have length D"
        return fac .* (Σ .* randn(T, D))                 # cov = fac^2 diagm(Σ.^2)
    elseif Σ isa AbstractMatrix
        @assert size(Σ, 1) == D "σ(x,t) as Matrix must have size (D, M)"
        M = size(Σ, 2)
        return fac .* (Σ * randn(T, M))                  # cov = fac^2 ΣΣᵀ
    else
        throw(ArgumentError("σ(x,t) must return Number, Vector, or Matrix; got $(typeof(Σ))"))
    end
end

# One tiny guided bridge step over [τ, τ+h] toward endpoint a at terminal time T_end.
# This recomputes the "local bridge" every step (uses current x and remaining time R = T_end - τ).
@inline function step_toward!(x::AbstractVector{T}, a::AbstractVector{T},
    τ::T, h::T, μ, σ; T_end::T=one(T)) where {T<:Real}
    R = T_end - τ
    @assert R > zero(T) "Remaining time must be positive; got R=$(R)"
    # Bridge fraction and variance shrinkage (Brownian-bridge exact; guided for general σ)
    ρ = min(one(T), h / R)                     # clamp for safety
    fac = sqrt(max(zero(T), h * (one(T) - ρ)))   # sqrt(h*(R-h)/R)
    μxt = μ(x, τ)
    Σxt = σ(x, τ)
    noise = _bridge_noise(Σxt, fac, length(x))
    @. x = x + μxt * h + ρ * (a - x) + noise
    return nothing
end

# Convenience non-mutating wrapper mirrors manifold style
@inline function step_toward(x::AbstractVector{T}, a::AbstractVector{T},
    τ::T, h::T, μ, σ; T_end::T=one(T)) where {T<:Real}
    y = similar(x)
    copyto!(y, x)
    step_toward!(y, a, τ, h, μ, σ; T_end=T_end)
    return y
end

# --- API: endpoint-conditioned sampling -----------------------------------

# Vector-of-times t[1:N] in [0,1] with X0/X1 of shape D×N (columns are independent bridges).
function endpoint_conditioned_sample(X0::ContinuousState{T},
    X1::ContinuousState{T},
    P::SDEProcess,
    t::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}
    x0 = tensor(X0)                 # D×N
    x1 = tensor(X1)                 # D×N
    D, N = size(x0)
    @assert size(x1) == (D, N)
    @assert length(t) == N

    out = similar(x0)
    x = similar(x0, D)           # work buffer

    @inbounds for n in 1:N
        t_target = t[n]
        if t_target ≤ zero(T)
            @views out[:, n] .= x0[:, n]
            continue
        end
        if t_target ≥ one(T)
            @views out[:, n] .= x1[:, n]
            continue
        end

        @views x .= x0[:, n]
        @views a = x1[:, n]

        τ = zero(T)                # absolute time in [0,1]
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            step_toward!(x, a, τ, h, P.μ, P.σ; T_end=one(T))
            τ += h
        end
        @views out[:, n] .= x
    end
    return ContinuousState(out)
end

# Two-vectors (tF, tB) of the same length → per-column horizon T_n = tF[n]+tB[n], target tF[n]
function endpoint_conditioned_sample(X0::ContinuousState{T},
    X1::ContinuousState{T},
    P::SDEProcess,
    tF::AbstractVector{T},
    tB::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}
    @assert length(tF) == length(tB)
    x0 = tensor(X0)
    x1 = tensor(X1)
    D, N = size(x0)
    @assert size(x1) == (D, N)
    @assert N == length(tF)

    out = similar(x0)
    x = similar(x0, D)

    @inbounds for n in 1:N
        T_end = tF[n] + tB[n]
        t_target = tF[n]
        if t_target ≤ zero(T)
            @views out[:, n] .= x0[:, n]
            continue
        end
        if t_target ≥ T_end
            @views out[:, n] .= x1[:, n]
            continue
        end

        @views x .= x0[:, n]
        @views a = x1[:, n]
        τ = zero(T)   # absolute time in [0, T_end]
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            step_toward!(x, a, τ, h, P.μ, P.σ; T_end=T_end)
            τ += h
        end
        @views out[:, n] .= x
    end
    return ContinuousState(out)
end

# Scalar t replicated across columns
function endpoint_conditioned_sample(X0::ContinuousState{T},
    X1::ContinuousState{T},
    P::SDEProcess,
    t::T; Δt::Real=1e-3) where {T<:Real}
    N = size(tensor(X0), 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(t, N); Δt=Δt)
end

# Scalar (tF, tB) replicated across columns
function endpoint_conditioned_sample(X0::ContinuousState{T},
    X1::ContinuousState{T},
    P::SDEProcess,
    tF::T, tB::T; Δt::Real=1e-3) where {T<:Real}
    N = size(tensor(X0), 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(tF, N), fill(tB, N); Δt=Δt)
end
# Euler–Maruyama noise with covariance fac^2 * (Σ Σᵀ) where fac = √h
@inline function _em_noise(Σ, fac::T, D::Int) where {T<:Real}
    if Σ isa Number
        return fac .* (Σ .* randn(T, D))                 # cov = h σ^2 I
    elseif Σ isa AbstractVector
        @assert length(Σ) == D "σ(x,t) as Vector must have length D"
        return fac .* (Σ .* randn(T, D))                 # cov = h diagm(Σ.^2)
    elseif Σ isa AbstractMatrix
        @assert size(Σ, 1) == D
        M = size(Σ, 2)
        return fac .* (Σ * randn(T, M))                  # cov = h ΣΣᵀ
    else
        throw(ArgumentError("σ(x,t) must return Number, Vector, or Matrix; got $(typeof(Σ))"))
    end
end

# One Euler–Maruyama step: x ← x + μ(x,τ)h + σ(x,τ) √h ξ
@inline function _em_step!(x::AbstractVector{T}, τ::T, h::T, μ, σ) where {T<:Real}
    μxt = μ(x, τ)
    Σxt = σ(x, τ)
    noise = _em_noise(Σxt, sqrt(h), length(x))
    @. x = x + μxt * h + noise
    return nothing
end
# Vector-of-times t[1:N] (each column can have a different target time)
function forward!(Xdest::ContinuousState{T},
    X0::ContinuousState{T},
    P::SDEProcess,
    t::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}

    x0 = tensor(X0)                 # D×N
    D, N = size(x0)
    @assert length(t) == N
    xd = tensor(Xdest)
    @assert size(xd) == (D, N)

    x = similar(x0, D)              # work buffer

    @inbounds for n in 1:N
        t_target = t[n]
        if t_target ≤ zero(T)
            @views xd[:, n] .= x0[:, n]
            continue
        end

        @views x .= x0[:, n]
        τ = zero(T)
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            _em_step!(x, τ, h, P.μ, P.σ)
            τ += h
        end
        @views xd[:, n] .= x
    end
    return Xdest
end

# Out-of-place convenience
function forward(X0::ContinuousState{T},
    P::SDEProcess,
    t::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}
    Xdest = ContinuousState(similar(tensor(X0)))
    return forward!(Xdest, X0, P, t; Δt=Δt)
end

# Scalar time t replicated across columns
function forward!(Xdest::ContinuousState{T},
    X0::ContinuousState{T},
    P::SDEProcess,
    t::T; Δt::Real=1e-3) where {T<:Real}
    N = size(tensor(X0), 2)
    return forward!(Xdest, X0, P, fill(t, N); Δt=Δt)
end

function forward(X0::ContinuousState{T},
    P::SDEProcess,
    t::T; Δt::Real=1e-3) where {T<:Real}
    N = size(tensor(X0), 2)
    return forward(X0, P, fill(t, N); Δt=Δt)
end

# Optional: two-vector interface (tF, tB) -> simulate forward to tF (ignores tB)
function forward(X0::ContinuousState{T},
    P::SDEProcess,
    tF::AbstractVector{T},
    tB::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}
    @assert length(tF) == length(tB)
    return forward(X0, P, tF; Δt=Δt)
end
