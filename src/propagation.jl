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

forward!(Xdest, Xt, process::Process, t1, t2) = forward!(Xdest, Xt, process, t2 - t1) #Overload for time-homogeneous processes
#forward!(Xdest, Xt, process::Process, t1, t2) = error() #forward!(Xdest, Xt, process, t2 - t1) #Overload for time-homogeneous processes
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

#backward!(Xdest, Xt, process::Process, t1, t2) = error() #backward!(Xdest, Xt, process, t2 - t1) #Overload for time-homogeneous processes
backward!(Xdest, Xt, process::Process, t1, t2) = backward!(Xdest, Xt, process, t2 - t1) #Overload for time-homogeneous processes
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
endpoint_conditioned_sample(Xa, Xc, p::Deterministic, t_a, t_b, t_c) = interpolate(Xa, Xc, t_a, t_b, t_c)
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

function forward(Xt::ContinuousState, process::BrownianMotion, t)
    return ContinuousState(Xt.state .+ process.δ * t + process.v .* sqrt(t) .* randn(size(Xt.state)))
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



function endpoint_conditioned_sample(X0::SwitchState, X1::SwitchState, process::SwitchBridgeProcess, t::Real; ϵ = 1e-2)
    xt = copy(X0)
    current_time = eltype(t)(0.0)
    while current_time < t
        δ = eltype(t)(min(t - current_time, ϵ))
        next_switching_state = endpoint_conditioned_sample(xt.switching_state, X1.switching_state, process.switching_process, current_time, current_time+δ,eltype(t)(1))
        target_endpoint = next_switching_state.state[1] == 1 ? X1.main_state : X0.main_state
        next_main_state = endpoint_conditioned_sample(xt.main_state, target_endpoint, process.main_process, current_time, current_time+δ,eltype(t)(1))
        xt = SwitchState(next_main_state, next_switching_state)
        current_time += δ
    end
    return xt
end
# Helper to expand a batch mask to the leading continuous dimension
_expand_to_cont(mask, nd_cont) = expand(mask, nd_cont)

# Discrete step — multiple dispatch controls vectorisation capability
function _discrete_step!(disc, X1_disc, p::PiQ, t_a, t_b, t_c, apply_mask, K)
    Snext = endpoint_conditioned_sample(DiscreteState(K, disc), DiscreteState(K, X1_disc), p, t_a, t_b, t_c)
    @inbounds disc .= ifelse.(apply_mask, Snext.state, disc)
    return disc
end

function _discrete_step!(disc, X1_disc, p::UniformDiscrete, t_a, t_b, t_c, apply_mask, K)
    Snext = endpoint_conditioned_sample(DiscreteState(K, disc), DiscreteState(K, X1_disc), p, t_a, t_b, t_c)
    @inbounds disc .= ifelse.(apply_mask, Snext.state, disc)
    return disc
end

function _discrete_step!(disc, X1_disc, p::GeneralDiscrete, t_a, t_b, t_c, apply_mask, K)
    @inbounds for ind in CartesianIndices(t_a)
        if apply_mask[ind]
            s = endpoint_conditioned_sample(
                DiscreteState(K, disc[ind]),
                DiscreteState(K, X1_disc[ind]),
                p, t_a[ind], t_b[ind], one(eltype(t_a))
            )
            disc[ind] = s.state
        end
    end
    return disc
end

# Batched endpoint-conditioned sample for SwitchBridgeProcess
function endpoint_conditioned_sample(
    X0::SwitchState, X1::SwitchState, process::SwitchBridgeProcess, t::AbstractArray;
    ϵ = 1e-2, tracker::Function = Returns(nothing)
)
    cont = copy(X0.main_state.state)
    disc = copy(X0.switching_state.state)

    cont_nd = ndims(cont)
    disc_nd = ndims(disc)
    cont_nd == disc_nd + 1 || throw(DimensionMismatch("main_state must have one extra leading dim; got $(size(cont)) vs $(size(disc))."))

    t_full = expand(t, disc_nd)
    τ = zero.(t_full)
    t_c = one(eltype(t_full)) .+ zero.(t_full)
    K = X0.switching_state.K

    while true
        active = τ .< t_full
        any(active) || break
        δ = min(ϵ, minimum((t_full .- τ)[active]))
        t_a = τ
        t_b = min.(t_full, τ .+ δ)
        apply_mask = t_a .< t_b

        # 1) Discrete update (via dispatch on switching process)
        disc = _discrete_step!(disc, X1.switching_state.state, process.switching_process, t_a, t_b, t_c, apply_mask, K)

        # 2) Continuous update for all sites in one shot
        choose_X1 = disc .== 1
        target_main = ifelse.(_expand_to_cont(choose_X1, cont_nd), X1.main_state.state, X0.main_state.state)
        Xmain_next = endpoint_conditioned_sample(
            ContinuousState(cont), ContinuousState(target_main), process.main_process, t_a, t_b, t_c
        )
        @inbounds cont .= ifelse.(_expand_to_cont(apply_mask, cont_nd), Xmain_next.state, cont)

        τ = t_b
    end

    return SwitchState(ContinuousState(cont), DiscreteState(K, disc))
end

# Batched endpoint-conditioned sample for XDependentSwitchBridgeProcess
function endpoint_conditioned_sample(
    X0::SwitchState, X1::SwitchState, process::XDependentSwitchBridgeProcess, t::AbstractArray;
    ϵ = 1e-2, tracker::Function = Returns(nothing)
)
    cont = copy(X0.main_state.state)
    disc = copy(X0.switching_state.state)

    cont_nd = ndims(cont)
    disc_nd = ndims(disc)
    cont_nd == disc_nd + 1 || throw(DimensionMismatch("main_state must have one extra leading dim; got $(size(cont)) vs $(size(disc))."))

    t_full = expand(t, disc_nd)
    τ = zero.(t_full)
    t_c = one(eltype(t_full)) .+ zero.(t_full)
    K = X0.switching_state.K

    while true
        active = τ .< t_full
        any(active) || break
        δ = min(ϵ, minimum((t_full .- τ)[active]))
        t_a = τ
        t_b = min.(t_full, τ .+ δ)
        apply_mask = t_a .< t_b

        # 1) Discrete update depends on current continuous state ⇒ per-site
        @inbounds for ind in CartesianIndices(t_full)
            if apply_mask[ind]
                x_cont = ContinuousState(@view cont[:, ntuple(i -> i <= disc_nd ? ind.I[i] : Colon(), cont_nd)...])
                Q = process.Q_function(x_cont)
                s = endpoint_conditioned_sample(
                    DiscreteState(K, disc[ind]),
                    DiscreteState(K, X1.switching_state.state[ind]),
                    GeneralDiscrete(Q), t_a[ind], t_b[ind], one(eltype(t_a))
                )
                disc[ind] = s.state
            end
        end

        # 2) Continuous update for all sites
        choose_X1 = disc .== 1
        target_main = ifelse.(expand(choose_X1, cont_nd), X1.main_state.state, X0.main_state.state)
        Xmain_next = endpoint_conditioned_sample(
            ContinuousState(cont), ContinuousState(target_main), process.main_process, t_a, t_b, t_c
        )
        @inbounds cont .= ifelse.(expand(apply_mask, cont_nd), Xmain_next.state, cont)

        τ = t_b
    end

    return SwitchState(ContinuousState(cont), DiscreteState(K, disc))
end
function endpoint_conditioned_sample(X0::SwitchState, X1::SwitchState, process::XDependentSwitchBridgeProcess, t::Real; ϵ = 1e-2, tracker::Function=Returns(nothing))

    xt = copy(X0)
    current_time = eltype(t)(0.0)
    while current_time < t
        δ = eltype(t)(min(t - current_time, ϵ))
        switching_process = GeneralDiscrete(process.Q_function(xt.main_state))
        next_switching_state = endpoint_conditioned_sample(xt.switching_state, X1.switching_state, switching_process, current_time, current_time+δ,eltype(t)(1))
        target_endpoint = next_switching_state.state[1] == 1 ? X1.main_state : X0.main_state
        next_main_state = endpoint_conditioned_sample(xt.main_state, target_endpoint, process.main_process, current_time, current_time+δ,eltype(t)(1))
        xt = SwitchState(next_main_state, next_switching_state)
        tracker(current_time, xt)
        current_time += δ
    end
    return xt

end

# We will step with Brownian bridges
function endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState, process::DriftDiffusionProcess, t; ϵ = 1e-2)
    xt = copy(X0)
    current_time = eltype(t)(0.0)
    while current_time < t
        δ = eltype(t)(min(t - current_time, ϵ))
        local_drift = process.μ(current_time, xt)
        local_diffusion = process.σ(current_time, xt)
        local_process = BrownianMotion(local_drift, local_diffusion)
        xt = endpoint_conditioned_sample(xt, X1, local_process, current_time, current_time+δ,eltype(t)(1))
    end
    return xt
end

function endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState, process::DriftDiffusionProcess, t::AbstractArray; ϵ = 1e-2)
    cont_state = similar(X0.state)
    @inbounds for ind in CartesianIndices(t)
        cont_state[:,ind] = X0.state[:,ind]
        xt = endpoint_conditioned_sample(xt, X1, process, t[ind]; ϵ = ϵ)
        cont_state[:,ind] = xt.state
    end
    return ContinuousState(cont_state)
end

function step(Xt::ContinuousState, process::DriftDiffusionProcess, t0::Real, t1::Real; ϵ = 1e-2)
    xt = copy(Xt)
    current_time = t0
    while current_time < t1
        δ = eltype(t)(min(t1 - current_time, ϵ))
        local_drift = process.μ(current_time, xt)
        local_diffusion = process.σ(current_time, xt)
        local_process = BrownianMotion(local_drift, local_diffusion)
        xt = forward(xt, local_process, δ)
        current_time += δ
    end
    return xt
end
function step(Xt::ContinuousState, process::DriftDiffusionProcess, t::Real; ϵ = 1e-2)
    return step(Xt, process, eltype(t)(0), t; ϵ = ϵ)
end

function step(Xt::ContinuousState, process::DriftDiffusionProcess, t::AbstractArray; ϵ = 1e-2)
    cont_state = similar(Xt.state)
    @inbounds for ind in CartesianIndices(t)
        cont_state[:,ind] = Xt.state[:,ind]
        xt = step(Xt, process, t[ind]; ϵ = ϵ)
        cont_state[:,ind] = xt.state
    end
    return ContinuousState(cont_state)
end

function step(Xt::ContinuousState, process::DriftDiffusionProcess, t0::AbstractArray, t1::AbstractArray; ϵ = 1e-2)
    cont_state = similar(Xt.state)
    if size(t0) != size(t1) || size(t0) != size(Xt.state)
        throw(DimensionMismatch("Time arrays must be the same size as the state array. Got size(t0)=$(size(t0)), size(t1)=$(size(t1)), size(Xt.state)=$(size(Xt.state))"))
    end
    @inbounds for ind in CartesianIndices(t0)
        cont_state[:,ind] = Xt.state[:,ind]
        xt = step(Xt, process, t0[ind], t1[ind]; ϵ = ϵ)
        cont_state[:,ind] = xt.state
    end
    return ContinuousState(cont_state)
end
