# A conditional bridge process that switches between two Brownian-bridge anchors:
# the true endpoint a, and an alternate anchor b near a. The rate of switching to
# the true endpoint explodes as t -> T_end to ensure arrival at a.
struct ConditionalBridgeProcess{T} <: ContinuousProcess
    σ::T       # diffusion scale (Number); used in Brownian bridge noise
    κ::T       # strength of time-singular rate toward true endpoint
    λ_b::T     # optional baseline rate to switch away from true endpoint (often 0)
    ε::T       # fraction of ||a-x0|| used to place the alternate anchor b beyond a
end

ConditionalBridgeProcess(σ::T; κ::T=one(T), λ_b::T=zero(T), ε::T=T(0.1)) where {T<:Real} =
    ConditionalBridgeProcess{T}(σ, κ, λ_b, ε)

@inline _μ_zero(x, t) = zero.(x)
@inline _σ_const(σ) = (x, t) -> σ

# Sample alternate anchor b uniformly in a D-ball of fixed radius r around a
@inline function _alt_anchor(a::AbstractVector{T}, r::T) where {T<:Real}
    if r == zero(T)
        return copy(a)
    end
    Dlen = length(a)
    z = randn(T, Dlen)
    zn = sqrt(sum(abs2, z))
    if zn == zero(T)
        z[1] = one(T); zn = one(T)
    end
    z_unit = @. z / zn
    u = rand(T)
    r_ball = r * (u^(one(T)/T(Dlen)))
    return @. a + r_ball * z_unit
end

# Hazard rate to switch to true endpoint: κ / (T_end - τ)
@inline _λ_to_a(κ::T, τ::T, T_end::T) where {T<:Real} = κ / max(T(1e-12), T_end - τ)

# One step of the switching-bridge toward current anchor (either a or b), then possibly switch anchors
@inline function _conditional_bridge_step!(x::AbstractVector{T}, current_to_a::Base.RefValue{Bool},
    a::AbstractVector{T}, b::AbstractVector{T}, τ::T, h::T, P::ConditionalBridgeProcess{T}, T_end::T) where {T<:Real}
    anchor = current_to_a[] ? a : b
    step_toward!(x, anchor, τ, h, _μ_zero, _σ_const(P.σ); T_end=T_end)
    # Switching after stepping
    if current_to_a[]
        # option to switch away from a (typically 0)
        pb = one(T) - exp(-P.λ_b * h)
        if pb > zero(T) && rand() < pb
            current_to_a[] = false
        end
    else
        pa = one(T) - exp(-_λ_to_a(P.κ, τ, T_end) * h)
        if pa > zero(T) && rand() < pa
            current_to_a[] = true
        end
    end
    return nothing
end

# --- endpoint-conditioned sampling for ConditionalBridgeProcess --------------

# New: endpoint-conditioned sampling on ConditionalBridgeState (position + 2-state CTMC)

# Vector-of-times t[1:N] in [0,1] (T_end = 1)
function endpoint_conditioned_sample(X0::ConditionalBridgeState{T},
    X1::ConditionalBridgeState{T},
    P::ConditionalBridgeProcess{T},
    t::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}

    x0 = X0.continuous_state        # D×N
    x1 = X1.continuous_state        # D×N
    s0 = X0.anchor_state            # N (1 => a, 2 => b)
    s1 = X1.anchor_state            # N (used if t ≥ 1)
    D, N = size(x0)
    @assert size(x1) == (D, N)
    @assert length(s0) == N
    @assert length(s1) == N
    @assert length(t)  == N

    outc = similar(x0)
    outs = similar(s0)
    x = similar(x0, D)

    @inbounds for n in 1:N
        t_target = t[n]
        if t_target ≤ zero(T)
            @views outc[:, n] .= x0[:, n]
            outs[n] = s0[n]
            continue
        end
        if t_target ≥ one(T)
            @views outc[:, n] .= x1[:, n]
            outs[n] = s1[n]
            continue
        end

        @views x .= x0[:, n]
        @views a = x1[:, n]
        b = _alt_anchor(a, P.ε * P.σ)
        to_a = Ref(s0[n] == 1)
        τ = zero(T)
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            _conditional_bridge_step!(x, to_a, a, b, τ, h, P, one(T))
            τ += h
        end
        @views outc[:, n] .= x
        outs[n] = to_a[] ? 1 : 2
    end
    return ConditionalBridgeState(outc, outs)
end

# Two-vectors (tF, tB) → per-column horizon T_end = tF[n] + tB[n], target tF[n]
function endpoint_conditioned_sample(X0::ConditionalBridgeState{T},
    X1::ConditionalBridgeState{T},
    P::ConditionalBridgeProcess{T},
    tF::AbstractVector{T},
    tB::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}

    @assert length(tF) == length(tB)
    x0 = X0.continuous_state
    x1 = X1.continuous_state
    s0 = X0.anchor_state
    s1 = X1.anchor_state
    D, N = size(x0)
    @assert size(x1) == (D, N)
    @assert length(s0) == N
    @assert length(s1) == N
    @assert N == length(tF)

    outc = similar(x0)
    outs = similar(s0)
    x = similar(x0, D)

    @inbounds for n in 1:N
        T_end = tF[n] + tB[n]
        t_target = tF[n]
        if t_target ≤ zero(T)
            @views outc[:, n] .= x0[:, n]
            outs[n] = s0[n]
            continue
        end
        if t_target ≥ T_end
            @views outc[:, n] .= x1[:, n]
            outs[n] = s1[n]
            continue
        end
        @views x .= x0[:, n]
        @views a = x1[:, n]
        b = _alt_anchor(a, P.ε * P.σ)
        to_a = Ref(s0[n] == 1)
        τ = zero(T)
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            _conditional_bridge_step!(x, to_a, a, b, τ, h, P, T_end)
            τ += h
        end
        @views outc[:, n] .= x
        outs[n] = to_a[] ? 1 : 2
    end
    return ConditionalBridgeState(outc, outs)
end

# Scalar wrappers
function endpoint_conditioned_sample(X0::ConditionalBridgeState{T},
    X1::ConditionalBridgeState{T},
    P::ConditionalBridgeProcess{T},
    t::T; Δt::Real=1e-3) where {T<:Real}
    N = size(X0.continuous_state, 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(t, N); Δt=Δt)
end

function endpoint_conditioned_sample(X0::ConditionalBridgeState{T},
    X1::ConditionalBridgeState{T},
    P::ConditionalBridgeProcess{T},
    tF::T, tB::T; Δt::Real=1e-3) where {T<:Real}
    N = size(X0.continuous_state, 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(tF, N), fill(tB, N); Δt=Δt)
end

######## Enforce new API: disallow ContinuousState for ConditionalBridgeProcess ########
function endpoint_conditioned_sample(X0::ContinuousState{T}, X1::ContinuousState{T}, P::ConditionalBridgeProcess{T}, t; Δt::Real=1e-3) where {T<:Real}
    throw(ArgumentError("ConditionalBridgeProcess now requires ConditionalBridgeState (position + anchor state). Construct ConditionalBridgeState(X.continuous, anchors) and retry."))
end
function endpoint_conditioned_sample(X0::ContinuousState{T}, X1::ContinuousState{T}, P::ConditionalBridgeProcess{T}, tF, tB; Δt::Real=1e-3) where {T<:Real}
    throw(ArgumentError("ConditionalBridgeProcess now requires ConditionalBridgeState (position + anchor state). Construct ConditionalBridgeState(X.continuous, anchors) and retry."))
end

######## Enforce new API: disallow SwitchingSDEState for ConditionalBridgeProcess ########
function endpoint_conditioned_sample(X0::SwitchingSDEState{T}, X1::SwitchingSDEState{T}, P::ConditionalBridgeProcess{T}, t; Δt::Real=1e-3) where {T<:Real}
    throw(ArgumentError("ConditionalBridgeProcess now requires ConditionalBridgeState (position + anchor state). Construct ConditionalBridgeState and retry."))
end
function endpoint_conditioned_sample(X0::SwitchingSDEState{T}, X1::SwitchingSDEState{T}, P::ConditionalBridgeProcess{T}, tF, tB; Δt::Real=1e-3) where {T<:Real}
    throw(ArgumentError("ConditionalBridgeProcess now requires ConditionalBridgeState (position + anchor state). Construct ConditionalBridgeState and retry."))
end
