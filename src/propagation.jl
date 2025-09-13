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
# =============== Switching bridge endpoint-conditioned sampler ===============

"""
    endpoint_conditioned_sample(X0::SwitchBridgeState, X1::SwitchBridgeState, p::SwitchBridgeProcess, tF, tB; Δt = 0.05, Xalt = X0.continuous_state)

Step-toward approximation for a process that switches between two Brownian bridges:
one toward the provided endpoint `X1.continuous_state` ("original"), and one toward
an alternative decoy endpoint `Xalt` (defaults to the start location).

- Switching is a 2-state CTMC with constant rates: from original→alternative at `λ_alt`
  and alternative→original at `λ_orig`.
- Within a small step `Δt`, we optionally switch regime, add Brownian noise with
  diffusion scale `σ`, then take a geodesic-like step toward the current target with
  fraction `Δt / remaining_time` (Euclidean version of the manifold step-toward).

This mirrors the manifold stepping scheme but in Euclidean coordinates.
"""
function endpoint_conditioned_sample(
    X0::SwitchBridgeState,
    X1::SwitchBridgeState,
    p::SwitchBridgeProcess,
    tF,
    tB;
    Δt = 0.05,
    Xalt::ContinuousState = X0.continuous_state,
)
    T = eltype(flatview(X0.continuous_state.state))

    # Shapes
    Xt = copy(X0.continuous_state.state)             # D × N (typical)
    D = size(Xt, 1)
    N = size(Xt)[end]

    # Per-sample forward/backward horizons
    tFv = (length(tF) == 1) ? fill(T(tF), N) : T.(vec(tF))
    tBv = (length(tB) == 1) ? fill(T(tB), N) : T.(vec(tB))
    (length(tFv) == N && length(tBv) == N) || throw(ArgumentError("tF/tB must have length 1 or match number of samples ($N)"))

    # Regime flags per sample
    is_alt = copy(X0.is_alternative)
    length(is_alt) == N || throw(ArgumentError("is_alternative length $(length(is_alt)) must match number of samples $N"))

    # Work per sample so the discrete regime is shared across all dimensions
    for j in 1:N
        t = T(0)
        tot = tFv[j] + tBv[j]
        target = tFv[j]

        # View into current sample vector
        x = view(Xt, :, j)
        xalt = view(Xalt.state, :, j)
        x1 = view(X1.continuous_state.state, :, j)

        while t < target
            inc = min(T(Δt), target - t)

            # Diffuse all dimensions jointly
            if p.σ > 0
                @. x = x + sqrt(p.σ * inc) * rand(Normal(0, 1))
            end

            # Possibly switch within this interval (rate may depend on state)
            λ = is_alt[j] ? p.λ_orig(x) : p.λ_alt(x)
            if λ > 0 && rand() < (1 - exp(-λ * inc))
                is_alt[j] = !is_alt[j]
            end

            # Current target; force original endpoint at the terminal step only
            last_step = (t + inc) >= target
            q = last_step ? x1 : (is_alt[j] ? xalt : x1)

            # Geodesic-like step-to-target
            remaining = max(tot - t, T(1e-12))
            frac = inc / remaining
            @. x = x + (q - x) * frac

            t += inc
        end
    end

    return SwitchBridgeState(ContinuousState(Xt), is_alt)
end
