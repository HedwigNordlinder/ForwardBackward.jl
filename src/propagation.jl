#Note: time-inhomogeneous processes must not implement the single-time forward/backward methods, and the two-time endpoint-conditioned sampling method.

expand(t::Real, x) = t
function expand(t::AbstractArray, d::Int)
    ndt = ndims(t)
    d - ndt < 0 && error("Cannot expand array of size $(size(t)) to $d dimensions.")
    reshape(t, ntuple(Returns(1), d - ndt)..., size(t)...)
end

"""
    forward!(Xdest::StateLikelihood, Xt::State, process::Process, t)
    forward!(Xdest, Xt, process::Process, t1, t2) = forward!(Xdest, Xt, process::Process, t2-t1) #For time-homogeneous processes
    forward(Xt::StateLikelihood, process::Process, t)
    forward(Xt::State, process::Process, t)
    forward(Xt, process::Process, t1, t2) = forward!(Xt, process, t2 - t1) #For time-homogeneous processes
    forward(Xt::StateLikelihood, process::Process, t1, t2)
    forward(Xt::State, process::Process, t1, t2)

Propagate a state or likelihood forward in time according to the process dynamics.

# Parameters
- `Xdest`: Destination for in-place operation
- `Xt`: Initial state or likelihood
- `process`: The stochastic process
- `t`: Time to propagate forward, for the single-time call
- `t1`, `t2`: Start and end times, for the two-time call


# Returns
The forward-propagated state or likelihood. 
"""
forward!(Xdest::StateLikelihood, Xt::State, process::Process, t) = forward!(Xdest, stochastic(eltype(t), Xt), process, t)
forward(Xt::StateLikelihood, process::Process, t) = forward!(copy(Xt), Xt, process, t)
forward(Xt::State, process::Process, t) = forward!(stochastic(eltype(t), Xt), Xt, process, t)

forward!(Xdest, Xt, process::Process, t1, t2) = error() #forward!(Xdest, Xt, process, t2 - t1) #Overload for time-homogeneous processes
forward!(Xdest::StateLikelihood, Xt::State, process::Process, t1, t2) = forward!(Xdest, stochastic(eltype(t1), Xt), process, t1, t2)
forward(Xt::State, process::Process, t1, t2) = forward!(stochastic(eltype(t1), Xt), Xt, process, t1, t2)
forward(Xt::StateLikelihood, process::Process, t1, t2) = forward!(copy(Xt), Xt, process, t1, t2)

"""
    backward!(Xdest::StateLikelihood, Xt::State, process::Process, t)
    backward!(Xdest, Xt, process::Process, t1, t2) = backward!(Xdest, Xt, process, t2 - t1) #For time-homogeneous processes
    backward(Xt::StateLikelihood, process::Process, t)
    backward(Xt::State, process::Process, t)
    backward(Xt, process::Process, t1, t2) = backward!(Xt, process, t2 - t1) #For time-homogeneous processes
    backward(Xt::StateLikelihood, process::Process, t1, t2)
    backward(Xt::State, process::Process, t1, t2)

Propagate a state or likelihood backward in time according to the process dynamics.

# Parameters
- `Xdest`: Destination for in-place operation
- `Xt`: Final state or likelihood
- `process`: The stochastic process
- `t`: Time to propagate backward, for the single-time call
- `t1`, `t2`: Start and end times, for the two-time call

# Returns
The backward-propagated state or likelihood
"""
backward!(Xdest::StateLikelihood, Xt::State, process::Process, t) = backward!(Xdest, stochastic(eltype(t), Xt), process, t)
backward(Xt::StateLikelihood, process::Process, t) = backward!(copy(Xt), Xt, process, t)
backward(Xt::State, process::Process, t) = backward!(stochastic(eltype(t), Xt), Xt, process, t)

backward!(Xdest, Xt, process::Process, t1, t2) = error() #backward!(Xdest, Xt, process, t2 - t1) #Overload for time-homogeneous processes
backward!(Xdest::StateLikelihood, Xt::State, process::Process, t1, t2) = backward!(Xdest, stochastic(eltype(t1), Xt), process, t1, t2)
backward(Xt::State, process::Process, t1, t2) = backward!(stochastic(eltype(t1), Xt), Xt, process, t1, t2)
backward(Xt::StateLikelihood, process::Process, t1, t2) = backward!(copy(Xt), Xt, process, t1, t2)

"""
    interpolate(X0::ContinuousState, X1::ContinuousState, tF, tB)
    interpolate(X_a::ContinuousState, X_c::ContinuousState, t_a, t_b, t_c) = interpolate(X_a, X_c, t_b .- t_a, t_c .- t_b)

Linearly interpolate between two continuous states.

# Parameters
- `X0`: Initial state
- `X1`: Final state
- `tF`: Forward time
- `tB`: Backward time
- `t_a`, `t_b`, `t_c`: If 3-time call, this assumes `t_b-t_a` is the forward time and `t_c-t_b` is the backward time.

# Returns
The interpolated state
"""
function interpolate(X0::ContinuousState, X1::ContinuousState, tF, tB)
    t0 = @. tF/(tF + tB)
    t1 = @. 1 - t0
    return ContinuousState(X0.state .* expand(t1, ndims(X0.state)) .+ X1.state .* expand(t0, ndims(X1.state)))
end
interpolate(X_a::ContinuousState, X_c::ContinuousState, t_a, t_b, t_c) = interpolate(X_a, X_c, t_b .- t_a, t_c .- t_b)

"""
    endpoint_conditioned_sample(Xa, Xc, P::Process, t)
    endpoint_conditioned_sample(Xa, Xc, P::Process, tF, tB)
    endpoint_conditioned_sample(Xa, Xc, P::Process, t_a, t_b, t_c)
    endpoint_conditioned_sample(Xa, Xc, P::Deterministic, tF, tB)
    

Generate a sample from the endpoint-conditioned process.

# Parameters
- `Xa`: Initial state
- `Xc`: Final state
- `P`: The stochastic process

# Time argumenrs
- `t`: For single-time call, this samples at time=t assuming endpoints at time=0 and time=1.
- `tF`, `tB`: For two-time call, this assumes `tF` is the forward time and `tB` is the backward time (allowed for time-homogeneous processes)
- `t_a`, `t_b`, `t_c`: If 3-time call, this samples at time=t_b assuming endpoints at time=t_a and time=t_c.

# Returns
A sample from the endpoint-conditioned distribution

# Notes
For continuous processes, uses the forward-backward algorithm.
For deterministic processes, uses linear interpolation.
"""
endpoint_conditioned_sample(X0, X1, p, tF, tB) = rand(forward(X0, p, tF) ⊙ backward(X1, p, tB))
endpoint_conditioned_sample(X0, X1, p, t) = endpoint_conditioned_sample(X0, X1, p, t, clamp.(1 .- t, 0, 1))
endpoint_conditioned_sample(X0, X1, p::Deterministic, tF, tB) = interpolate(X0, X1, tF, tB)
endpoint_conditioned_sample(Xa, Xc, p::Process, t_a, t_b, t_c) = rand(forward(Xa, p, t_a, t_b) ⊙ backward(Xc, p, t_b, t_c))

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

function forward!(x_dest::GaussianLikelihood, Xs::GaussianLikelihood, P::OrnsteinUhlenbeckExpVar, t1, t2)
    μ, θ = P.μ, P.θ
    t1e = expand(t1, ndims(Xs.mu))
    t2e = expand(t2, ndims(Xs.mu))
    Δ   = t2e .- t1e
    Q   = _ou_noise_Q(t1e, t2e, θ, P.a0, P.w, P.β)
    @. x_dest.mu  = μ + exp(-θ * Δ) * (Xs.mu - μ)
    @. x_dest.var = exp(-2θ * Δ) * Xs.var + Q
    x_dest.log_norm_const .= Xs.log_norm_const
    return x_dest
end

function backward!(x_dest::GaussianLikelihood, Xu::GaussianLikelihood, P::OrnsteinUhlenbeckExpVar, t1, t2)
    μ, θ = P.μ, P.θ
    t1e = expand(t1, ndims(Xu.mu))
    t2e = expand(t2, ndims(Xu.mu))
    Δ   = t2e .- t1e
    Q   = _ou_noise_Q(t1e, t2e, θ, P.a0, P.w, P.β)
    @. x_dest.mu  = μ + exp( θ * Δ) * (Xu.mu - μ)
    @. x_dest.var = exp( 2θ * Δ) * (Xu.var + Q)
    x_dest.log_norm_const .= Xu.log_norm_const
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

function endpoint_conditioned_sample(X0::AuxillaryState, X1::AuxillaryState, P::AuxillaryProcess, tF, tB)
    error("Not allowed for this type of process, only single time argument may be passed.")
end

function endpoint_conditioned_sample(X0::AuxillaryState, X1::AuxillaryState, P::AuxillaryProcess, t; ϵ = 1e-6)
    
    drift_state = copy(X0.ctmc_state)
    cont_state = copy(X0.cont_state)
    curr_time = 0
    while curr_time < t
        timestep = min(ϵ, t - curr_time)
        drift_state = endpoint_conditioned_sample(drift_state, X1.ctmc_state, P.dproc, t,t+timestep, 1)
        next_bridge_point = next_drift_state.state == 1 ? X1.cont_state : X0.cont_state
        cont_state = rand(endpoint_conditioned_sample(X0.cont_state, next_bridge_point, P.cproc, timestep, 1))
        curr_time += timestep
        # I think this is the correct way to step forward
    end
    return AuxillaryState(drift_state, cont_state)
end



function endpoint_conditioned_sample(X0::AuxillaryState, X1::AuxillaryState, P::AuxillaryProcess, t::AbstractArray; kwargs...)
    Xt = AuxillaryState(copy(X0.ctmc_state), copy(X0.cont_state))
    axes(X0.ctmc_state.state) == axes(t) || error("Discrete state axes must match time grid axes.")
    axes(X1.ctmc_state.state) == axes(t) || error("Discrete state axes must match time grid axes.")
    prefix_len = ndims(X0.cont_state.state) - ndims(t)
    prefix_len < 0 && error("Cannot broadcast AuxillaryState over time array with higher rank.")
    prefix = ntuple(_ -> Colon(), prefix_len)
    for I in CartesianIndices(t)
        idx = Tuple(I)
        x0 = AuxillaryState(
            DiscreteState(X0.ctmc_state.K, view(X0.ctmc_state.state, idx...)),
            ContinuousState(view(X0.cont_state.state, prefix..., idx...))
        )
        x1 = AuxillaryState(
            DiscreteState(X1.ctmc_state.K, view(X1.ctmc_state.state, idx...)),
            ContinuousState(view(X1.cont_state.state, prefix..., idx...))
        )

        sample = endpoint_conditioned_sample(x0, x1, P, t[I])
        copyto!(view(Xt.ctmc_state.state, idx...), sample.ctmc_state.state)
        copyto!(view(Xt.cont_state.state, prefix..., idx...), sample.cont_state.state)
    end
    return Xt
end

function bridge_generator(Q::AbstractMatrix, t1::Real, x1::Integer, t2::Real, x2::Integer; eps::Real=1e-12)
    n = size(Q,1)
    size(Q,1) == size(Q,2) || throw(ArgumentError("Q must be square"))
    t2 > t1 || throw(ArgumentError("Require t2 > t1 (got t1=$t1, t2=$t2)"))
    1 ≤ x1 ≤ n || throw(ArgumentError("x1 out of range 1:$n"))
    1 ≤ x2 ≤ n || throw(ArgumentError("x2 out of range 1:$n"))

    e_x2 = zeros(eltype(Q), n); e_x2[x2] = one(eltype(Q))

    # Return a closure Q'(t)
    function Qprime(t::Real)
        (t1 ≤ t < t2) || throw(ArgumentError("t must lie in [t1, t2): got t=$t, interval=[$t1, $t2)"))
        # h(t) = P(X_{t2}=x2 | X_t = ·) as a column vector
        H = exp((t2 - t) * Q) * e_x2
        if eps > 0
            # Clamp to avoid division by 0 when extremely close to t2
            H = max.(H, eps)
        end
        # Off-diagonals: q'_{ij}(t) = q_{ij} * h_j / h_i
        Qoff = similar(Q)
        @inbounds for i in 1:n, j in 1:n
            if i == j
                Qoff[i,j] = zero(eltype(Q))
            else
                Qoff[i,j] = Q[i,j] * (H[j] / H[i])
            end
        end
        # Diagonals: make rows sum to zero
        Qdiag = -sum(Qoff, dims=2)
        Qp = copy(Qoff)
        @inbounds for i in 1:n
            Qp[i,i] = Qdiag[i]
        end
        return Qp
    end

    return Qprime
end


function endpoint_conditioned_sample(X0::DiscreteState, X1::Discretetate, P::GeneralDiscrete, ta, tb, tc; ϵ = 1e-6)

    elapsed_time = 0
    state = copy(X0.state)    
    while elapsed_time < tb - ta
        timestep = min(ϵ, (tb - ta) - elapsed_time)
        Qprime = bridge_generator(P.Q, ta, X0.state, tc, X1.state)(ta + elapsed_time)
        state = rand(forward(stochastic(state),GeneralDiscrete(Qprime),timestep))
        elapsed_time += timestep
    end
    return DiscreteState(X0.K, state)

end