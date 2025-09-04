expand(t::Real, x) = t
function expand(t::AbstractArray, d::Int)
    ndt = ndims(t)
    d - ndt < 0 && error("Cannot expand array of size $(size(t)) to $d dimensions.")
    reshape(t, ntuple(Returns(1), d - ndt)..., size(t)...)
end

"""
    forward!(Xdest::StateLikelihood, Xt::State, process::Process, t)
    forward(Xt::StateLikelihood, process::Process, t)
    forward(Xt::State, process::Process, t)

Propagate a state or likelihood forward in time according to the process dynamics.

# Parameters
- `Xdest`: Destination for in-place operation
- `Xt`: Initial state or likelihood
- `process`: The stochastic process
- `t`: Time to propagate forward

# Returns
The forward-propagated state or likelihood
"""
forward!(Xdest::StateLikelihood, Xt::State, process::Process, t) = forward!(Xdest, stochastic(eltype(t), Xt), process, t)
forward(Xt::StateLikelihood, process::Process, t) = forward!(copy(Xt), Xt, process, t)
forward(Xt::State, process::Process, t) = forward!(stochastic(eltype(t), Xt), Xt, process, t)

"""
    backward!(Xdest::StateLikelihood, Xt::State, process::Process, t)
    backward(Xt::StateLikelihood, process::Process, t)
    backward(Xt::State, process::Process, t)

Propagate a state or likelihood backward in time according to the process dynamics.

# Parameters
- `Xdest`: Destination for in-place operation
- `Xt`: Final state or likelihood
- `process`: The stochastic process
- `t`: Time to propagate backward

# Returns
The backward-propagated state or likelihood
"""
backward!(Xdest::StateLikelihood, Xt::State, process::Process, t) = backward!(Xdest, stochastic(eltype(t), Xt), process, t)
backward(Xt::StateLikelihood, process::Process, t) = backward!(copy(Xt), Xt, process, t)
backward(Xt::State, process::Process, t) = backward!(stochastic(eltype(t), Xt), Xt, process, t)

"""
    interpolate(X0::ContinuousState, X1::ContinuousState, tF, tB)

Linearly interpolate between two continuous states.

# Parameters
- `X0`: Initial state
- `X1`: Final state
- `tF`: Forward time
- `tB`: Backward time

# Returns
The interpolated state
"""
function interpolate(X0::ContinuousState, X1::ContinuousState, tF, tB)
    t0 = @. tF/(tF + tB)
    t1 = @. 1 - t0
    return ContinuousState(X0.state .* expand(t1, ndims(X0.state)) .+ X1.state .* expand(t0, ndims(X1.state)))
end

"""
    endpoint_conditioned_sample(X0, X1, p, tF, tB)
    endpoint_conditioned_sample(X0, X1, p, t)
    endpoint_conditioned_sample(X0, X1, p::Deterministic, tF, tB)

Generate a sample from the endpoint-conditioned process.

# Parameters
- `X0`: Initial state
- `X1`: Final state
- `p`: The stochastic process
- `t`, `tF`: Forward time
- `tB`: Backward time (defaults to 1-t for single time parameter)

# Returns
A sample from the endpoint-conditioned distribution

# Notes
For continuous processes, uses the forward-backward algorithm.
For deterministic processes, uses linear interpolation.
"""
endpoint_conditioned_sample(X0, X1, p, tF, tB) = rand(forward(X0, p, tF) ⊙ backward(X1, p, tB))
endpoint_conditioned_sample(X0, X1, p, t) = endpoint_conditioned_sample(X0, X1, p, t, clamp.(1 .- t, 0, 1))
endpoint_conditioned_sample(X0, X1, p::Deterministic, tF, tB) = interpolate(X0, X1, tF, tB)

function forward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::OrnsteinUhlenbeck, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    μ, v, θ = process.μ, process.v, process.θ
    @. x_dest.mu = μ + exp(-θ * t) * (Xt.mu - μ)
    @. x_dest.var = exp(-2θ * t) * Xt.var + (v / (2θ)) * (1 - exp(-2θ * t))
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function backward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::OrnsteinUhlenbeck, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    μ, v, θ = process.μ, process.v, process.θ
    @. x_dest.mu = μ + exp(θ * t) * (Xt.mu - μ)
    @. x_dest.var = exp(2θ * t) * (Xt.var + (v / (2θ)) * (1 - exp(-2θ * t)))
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function forward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::BrownianMotion, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    x_dest.mu .= @. Xt.mu + process.δ * t
    x_dest.var .= @. process.v * t + Xt.var
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function backward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::BrownianMotion, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    x_dest.mu .= @. Xt.mu - process.δ * t
    x_dest.var .= @. process.v * t + Xt.var
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::PiQ, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    scals = sum(source.dist, dims = 1)
    pow = @. exp(-process.β * process.r * t)
    c1 = @. (1 - pow) * process.π
    c2 = @. pow + (1 - pow) * process.π
    dest.dist .= @. (scals - source.dist) * c1 + source.dist * c2
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::PiQ, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    pow = @. exp(-process.β * process.r * t)
    c1 = @. (1 - pow) * process.π
    vsum = sum(source.dist .* c1, dims=1)
    dest.dist .= pow .* source.dist .+ vsum
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformDiscrete, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    scals = sum(source.dist, dims = 1)
    r = process.μ * 1/(1-1/K)   
    p = (1/K)
    pow = @. exp(-r * t)
    c1 = @. (1 - pow) * p
    c2 = @. pow + (1 - pow) * p
    dest.dist .= @. (scals - source.dist) * c1 + source.dist * c2
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformDiscrete, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    r = process.μ * 1/(1-1/K)   
    p = (1/K)
    pow = @. exp(-r * t)
    c1 = @. (1 - pow) * p
    vsum = sum(source.dist .* c1, dims=1)
    dest.dist .= pow .* source.dist .+ vsum
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformUnmasking, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    mask_volume = selectdim(source.dist, 1, K:K)
    event_p = @. 1 - exp(-process.μ * t)
    #Distribute lost mask volume among all other states equally, and decay it from the mask:
    selectdim(dest.dist, 1, 1:(K-1)) .= selectdim(source.dist, 1, 1:(K-1)) .+ mask_volume .* (1/(K-1)) .* event_p
    selectdim(dest.dist, 1, K:K) .= mask_volume .* (1 .- event_p)
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformUnmasking, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    event_p = @. 1 - exp(-process.μ * t)
    #Nonmask states pass through unchanged.
    selectdim(dest.dist, 1, 1:(K-1)) .= selectdim(source.dist, 1, 1:(K-1))
    #Mask state's message gathers contributions from nonmask states.
    vsum = sum(selectdim(source.dist, 1, 1:(K-1)) .* (event_p/(K-1)), dims=1)
    selectdim(dest.dist, 1, K:K) .= (1 .- event_p) .* selectdim(source.dist, 1, K:K) .+ vsum
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::GeneralDiscrete, t::Real)
    P = exp(process.Q .* t)
    clamp!(P, 0, 1)
    reshape(dest.dist, size(source.dist,1), :) .= (reshape(source.dist, size(source.dist,1), :)' * P)'
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::GeneralDiscrete, t::Real)
    P = exp(process.Q .* t)
    clamp!(P, 0, 1)
    mul!(reshape(dest.dist, size(source.dist,1), :), P, reshape(source.dist, size(source.dist,1), :))
    dest.log_norm_const .= source.log_norm_const
    return dest
end

#To add: DiagonalizadCTMC, HQtPi
# =============== SwitchingBM endpoint-conditioned sampler ====================

# Evaluate the state-dependent rate at a scalar; if states are vectors, use ‖x‖.
# (Change this if you prefer a different reduction.)
λ_at(P::SwitchingBM, x) = P.λ(x isa Number ? x : norm(x))

# One-shot frozen-rate mirror bridge at time t ∈ (0,1).
# Exact for constant λ; first-order accurate if λ depends on state.
function _switchingbm_bridge!(y::AbstractVector{T}, x::AbstractVector{T}, a::AbstractVector{T},
                              t::T, σ::T, λ0::T) where {T<:Real}
    if t ≤ zero(T)
        y .= x; return y
    elseif t ≥ one(T)
        y .= a; return y
    end
    τ  = one(T) - t
    s  = σ*sqrt(t*τ)                 # Brownian-bridge std at time t
    αt = 0.5*(1 + exp(-2*λ0*t))      # even-parity on [0,t]
    ατ = 0.5*(1 + exp(-2*λ0*τ))      # even-parity on [t,1]

    # log-weights for (i,j) ∈ {+1,-1}×{+1,-1}
    lp = (i,j)-> log(i==+1 ? αt : 1-αt) + log(j==+1 ? ατ : 1-ατ)
    # Gaussian normalizer from product φ(i x; j a, σ²) (constants drop in softmax)
    lg = (i,j)-> - (sum(abs2, @. (i*x) - (j*a))) / (2*σ^2)

    Lpp = lp(+1,+1) + lg(+1,+1)
    Lpm = lp(+1,-1) + lg(+1,-1)
    Lmp = lp(-1,+1) + lg(-1,+1)
    Lmm = lp(-1,-1) + lg(-1,-1)
    m   = max(max(Lpp,Lpm), max(Lmp,Lmm))
    wpp = exp(Lpp - m); wpm = exp(Lpm - m); wmp = exp(Lmp - m); wmm = exp(Lmm - m)
    Z   = wpp + wpm + wmp + wmm
    wpp /= Z; wpm /= Z; wmp /= Z; wmm /= Z

    # sample (i,j)
    u = rand()
    i = +1; j = +1
    if u > wpp
        u -= wpp
        if u ≤ wpm
            i=+1; j=-1
        elseif (u -= wpm) ≤ wmp
            i=-1; j=+1
        else
            i=-1; j=-1
        end
    end

    # mean μ_{ij} = (1−t) i x + t j a ; variance s^2 I
    @. y = (τ * (i*x)) + (t * (j*a)) + s * randn(T)
    return y
end

# Vector-of-times API that Flowfusion calls: endpoint_conditioned_sample(X0, X1, P, t::Vector)
function endpoint_conditioned_sample(X0::ContinuousState{T},
                                     X1::ContinuousState{T},
                                     P::SwitchingBM{T},
                                     t::AbstractVector{T}) where {T<:Real}
    x0 = tensor(X0)         # D×N
    a1 = tensor(X1)         # D×N
    D, N = size(x0)
    @assert size(a1) == (D, N)
    @assert length(t) == N

    out = similar(x0)
    y   = zeros(T, D)
    @inbounds for n in 1:N
        λ0 = λ_at(P, view(x0, :, n))   # freeze λ at the left endpoint for this sample
        _switchingbm_bridge!(y, view(x0, :, n), view(a1, :, n), t[n], P.σ, λ0)
        @views out[:, n] .= y
    end
    return ContinuousState(out)
end

# Two-vector API used internally by ForwardBackward/Flowfusion:
# interpret tF, tB as proportional parts of total 1.0 and map to t = tF/(tF+tB).
function endpoint_conditioned_sample(X0::ContinuousState{T},
                                     X1::ContinuousState{T},
                                     P::SwitchingBM{T},
                                     tF::AbstractVector{T},
                                     tB::AbstractVector{T}) where {T<:Real}
    @assert length(tF) == length(tB)
    t = tF ./ (tF .+ tB)
    return endpoint_conditioned_sample(X0, X1, P, t)
end
