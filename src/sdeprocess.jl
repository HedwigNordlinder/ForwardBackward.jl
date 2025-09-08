# Parametric process wrapper for specialization
struct SDEProcess{Fμ,Fσ} <: ContinuousProcess
    μ::Fμ   # (x,t) -> drift vector in ℝ^D
    σ::Fσ   # (x,t) -> Number | Vector (D) | Matrix (D×M) factor
end
SDEProcess(μ, σ) = SDEProcess{typeof(μ),typeof(σ)}(μ, σ)

"""
    SwitchingSDEProcess(μ_array, σ_array, Q)

A switching SDE process where drift and diffusion coefficients are selected from arrays based on a CTMC state.

# Parameters
- `μ_array`: Array/Vector of drift functions μ_k(x,t), one for each discrete regime k
- `σ_array`: Array/Vector of diffusion functions σ_k(x,t), one for each discrete regime k  
- `Q`: Either a static K×K transition rate matrix, or a callable returning such a matrix as `Q(x)` or `Q(x, t)`

# Examples
```julia
# Static-Q example (2 regimes)
μ_funcs = [(x,t) -> -0.5*x, (x,t) -> 0.1*x]
σ_funcs = [(x,t) -> 1.0, (x,t) -> 2.0]
Q = [-1.0 1.0; 2.0 -2.0]
process = SwitchingSDEProcess(μ_funcs, σ_funcs, Q)

# State-dependent-Q example (2 regimes)
Qfun(x, t) = [-1.0 - 0.1*norm(x) 1.0 + 0.1*norm(x);
               2.0                 -2.0]
process = SwitchingSDEProcess(μ_funcs, σ_funcs, Qfun)
```
"""
struct SwitchingSDEProcess{Fμ,Fσ,TQ} <: ContinuousProcess
    μ_array::Fμ     # Array of drift functions: μ_k(x,t) for regime k
    σ_array::Fσ     # Array of diffusion functions: σ_k(x,t) for regime k  
    Q::TQ           # CTMC transition rate matrix (K×K) or callable Q(x[,t])
end

# Constructor for static matrix Q
function SwitchingSDEProcess(μ_array, σ_array, Q::AbstractMatrix)
    @assert length(μ_array) == length(σ_array) "μ_array and σ_array must have same length"
    @assert size(Q, 1) == size(Q, 2) == length(μ_array) "Q must be square with size matching μ_array length"
    @assert all(diag(Q) .<= 0) "Diagonal elements of Q must be non-positive"
    @assert all(Q[i,j] >= 0 for i in 1:size(Q,1), j in 1:size(Q,2) if i != j) "Off-diagonal elements of Q must be non-negative"
    SwitchingSDEProcess{typeof(μ_array),typeof(σ_array),typeof(Q)}(μ_array, σ_array, Q)
end

# Constructor for callable Q(x[, t])
function SwitchingSDEProcess(μ_array, σ_array, Qfun)
    @assert length(μ_array) == length(σ_array) "μ_array and σ_array must have same length"
    # Can't validate sizes/rates at construction; will validate per-call
    return SwitchingSDEProcess{typeof(μ_array),typeof(σ_array),typeof(Qfun)}(μ_array, σ_array, Qfun)
end

# Helper: current regime's drift/diffusion
@inline function _get_regime_functions(P::SwitchingSDEProcess, regime::Integer)
    return P.μ_array[regime], P.σ_array[regime]
end

# Helper: obtain Q matrix for current state/time
@inline function _Q_at(P::SwitchingSDEProcess, x, τ)
    Qsrc = P.Q
    if Qsrc isa AbstractMatrix
        return Qsrc
    else
        # Try Q(x, τ); fallback to Q(x)
        return try
            Qsrc(x, τ)
        catch
            Qsrc(x)
        end
    end
end

# Simulate CTMC transitions for a single step
@inline function _ctmc_step!(discrete_state::Integer, Q::AbstractMatrix, h::T) where {T<:Real}
    K = size(Q, 1)
    current_state = discrete_state
    
    # Total exit rate from current state
    λ = -Q[current_state, current_state]
    
    if λ <= zero(T)
        return current_state  # No transitions possible
    end
    
    # Time to next jump follows exponential distribution
    # Probability of no jump in time h is exp(-λh)
    if rand() > exp(-λ * h)
        # A jump occurs - choose destination state
        # Transition probabilities are proportional to off-diagonal rates
        rates = [Q[current_state, j] for j in 1:K]
        rates[current_state] = zero(T)  # Can't transition to self
        
        # Sample new state proportional to transition rates
        total_rate = sum(rates)
        if total_rate > zero(T)
            cumulative = zero(T)
            r = rand() * total_rate
            for j in 1:K
                if j != current_state
                    cumulative += rates[j]
                    if r <= cumulative
                        return j
                    end
                end
            end
        end
    end
    
    return current_state  # No transition occurred
end

# Combined SDE + CTMC step for switching process
@inline function _switching_sde_step!(x::AbstractVector{T}, discrete_state_ref::Ref{<:Integer}, 
    τ::T, h::T, P::SwitchingSDEProcess) where {T<:Real}
    
    # Step 1: Update discrete state via CTMC with frozen Q(x, τ)
    Qmat = _Q_at(P, x, τ)
    discrete_state_ref[] = _ctmc_step!(discrete_state_ref[], Qmat, h)
    
    # Step 2: Get current regime's drift and diffusion
    μ_current, σ_current = _get_regime_functions(P, discrete_state_ref[])
    
    # Step 3: Take SDE step with current regime's parameters
    μxt = μ_current(x, τ)
    Σxt = σ_current(x, τ)
    noise = _em_noise(Σxt, sqrt(h), length(x))
    @. x = x + μxt * h + noise
    
    return nothing
end

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

# --- Forward simulation for SwitchingSDEProcess ---

# Vector-of-times t[1:N] (each column can have a different target time)
function forward!(Xdest::SwitchingSDEState{T},
    X0::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    t::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}
    
    x0_cont = X0.continuous_state
    x0_disc = X0.discrete_state
    D, N = size(x0_cont)
    @assert length(t) == N
    @assert size(x0_disc) == (N,)
    @assert X0.K == size(P.Q, 1) || true
    
    xd_cont = Xdest.continuous_state
    xd_disc = Xdest.discrete_state
    @assert size(xd_cont) == (D, N)
    @assert size(xd_disc) == (N,)
    
    x = similar(x0_cont, D)  # work buffer for continuous state
    
    @inbounds for n in 1:N
        t_target = t[n]
        if t_target ≤ zero(T)
            @views xd_cont[:, n] .= x0_cont[:, n]
            xd_disc[n] = x0_disc[n]
            continue
        end
        
        @views x .= x0_cont[:, n]
        discrete_state_ref = Ref(x0_disc[n])
        
        τ = zero(T)
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            _switching_sde_step!(x, discrete_state_ref, τ, h, P)
            τ += h
        end
        
        @views xd_cont[:, n] .= x
        xd_disc[n] = discrete_state_ref[]
    end
    return Xdest
end

# Out-of-place convenience
function forward(X0::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    t::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}
    cont_dest = similar(X0.continuous_state)
    disc_dest = similar(X0.discrete_state)
    Xdest = SwitchingSDEState(cont_dest, disc_dest, X0.K; validate=false)
    return forward!(Xdest, X0, P, t; Δt=Δt)
end

# Scalar time t replicated across columns
function forward!(Xdest::SwitchingSDEState{T},
    X0::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    t::T; Δt::Real=1e-3) where {T<:Real}
    N = size(X0.continuous_state, 2)
    return forward!(Xdest, X0, P, fill(t, N); Δt=Δt)
end

function forward(X0::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    t::T; Δt::Real=1e-3) where {T<:Real}
    N = size(X0.continuous_state, 2)
    return forward(X0, P, fill(t, N); Δt=Δt)
end

# Two-vector interface (tF, tB) -> simulate forward to tF (ignores tB for now)
function forward(X0::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    tF::AbstractVector{T},
    tB::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}
    @assert length(tF) == length(tB)
    return forward(X0, P, tF; Δt=Δt)
end

# --- endpoint-conditioned sampling for SwitchingSDEProcess --------------------

# Vector-of-times t[1:N] in [0,1] (total horizon T_end = 1)
function endpoint_conditioned_sample(X0::SwitchingSDEState{T},
    X1::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    t::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}

    x0c = X0.continuous_state               # D×N
    x1c = X1.continuous_state               # D×N
    r0  = X0.discrete_state                 # N
    r1  = X1.discrete_state                 # N (used only if t ≥ 1)
    D, N = size(x0c)
    @assert size(x1c) == (D, N)
    @assert length(r0) == N
    @assert length(r1) == N
    @assert length(t)  == N
    @assert X0.K == size(P.Q, 1) || true

    outc = similar(x0c)
    outr = similar(r0)
    x    = similar(x0c, D)                  # work buffer

    @inbounds for n in 1:N
        t_target = t[n]
        if t_target ≤ zero(T)
            @views outc[:, n] .= x0c[:, n]
            outr[n] = r0[n]
            continue
        end
        if t_target ≥ one(T)
            @views outc[:, n] .= x1c[:, n]
            outr[n] = r1[n]
            continue
        end

        @views x .= x0c[:, n]
        @views a = x1c[:, n]
        r = r0[n]

        τ = zero(T)                # absolute time in [0,1]
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            μk, σk = _get_regime_functions(P, r)   # freeze regime over [τ, τ+h]
            step_toward!(x, a, τ, h, μk, σk; T_end=one(T))
            Qmat = _Q_at(P, x, τ)
            r = _ctmc_step!(r, Qmat, h)            # allow jump after the local bridge step
            τ += h
        end
        @views outc[:, n] .= x
        outr[n] = r
    end
    return SwitchingSDEState(outc, outr, X0.K; validate=false)
end

# Two-vectors (tF, tB) of the same length → per-column horizon T_n = tF[n]+tB[n], target tF[n]
function endpoint_conditioned_sample(X0::SwitchingSDEState{T},
    X1::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    tF::AbstractVector{T},
    tB::AbstractVector{T};
    Δt::Real=1e-3) where {T<:Real}

    @assert length(tF) == length(tB)

    x0c = X0.continuous_state
    x1c = X1.continuous_state
    r0  = X0.discrete_state
    r1  = X1.discrete_state
    D, N = size(x0c)
    @assert size(x1c) == (D, N)
    @assert length(r0) == N
    @assert length(r1) == N
    @assert N == length(tF)
    @assert X0.K == size(P.Q, 1) || true

    outc = similar(x0c)
    outr = similar(r0)
    x    = similar(x0c, D)

    @inbounds for n in 1:N
        T_end    = tF[n] + tB[n]
        t_target = tF[n]
        if t_target ≤ zero(T)
            @views outc[:, n] .= x0c[:, n]
            outr[n] = r0[n]
            continue
        end
        if t_target ≥ T_end
            @views outc[:, n] .= x1c[:, n]
            outr[n] = r1[n]
            continue
        end

        @views x .= x0c[:, n]
        @views a = x1c[:, n]
        r = r0[n]
        τ = zero(T)   # absolute time in [0, T_end]
        while τ < t_target
            h = min(T(Δt), t_target - τ)
            μk, σk = _get_regime_functions(P, r)   # freeze regime over [τ, τ+h]
            step_toward!(x, a, τ, h, μk, σk; T_end=T_end)
            Qmat = _Q_at(P, x, τ)
            r = _ctmc_step!(r, Qmat, h)
            τ += h
        end
        @views outc[:, n] .= x
        outr[n] = r
    end
    return SwitchingSDEState(outc, outr, X0.K; validate=false)
end

# Scalar t replicated across columns (total horizon 1)
function endpoint_conditioned_sample(X0::SwitchingSDEState{T},
    X1::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    t::T; Δt::Real=1e-3) where {T<:Real}
    N = size(X0.continuous_state, 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(t, N); Δt=Δt)
end

# Scalar (tF, tB) replicated across columns
function endpoint_conditioned_sample(X0::SwitchingSDEState{T},
    X1::SwitchingSDEState{T},
    P::SwitchingSDEProcess,
    tF::T, tB::T; Δt::Real=1e-3) where {T<:Real}
    N = size(X0.continuous_state, 2)
    return endpoint_conditioned_sample(X0, X1, P, fill(tF, N), fill(tB, N); Δt=Δt)
end
