

struct SDEState{T<:Real} <: State
    state::AbstractArray{T}
    time::Float64
end
struct SDEProcess{T} <: ContinuousProcess
    μ::Function # takes in two arguments, time and current value
    σ::Function # same as above
    μ0::SDEState{T}
    dimension::Int64
end

function SDEProcess(I::Function, γ::Function, x0::T, x1::T) where T
    dIdt = t -> Zygote.jacobian(s -> I(s, x0, x1), t)[1]
    dγdt = t -> Zygote.gradient(s -> γ(s), t)[1]
    μ(t, x) =
        if det(γ(t)) == 0
            zero(x)
        else
            dIdt(t) + (dγdt(t) / γ(t)) * (x - I(t, x0, x1))
        end
    σ(t, x) =
        if γ(t) == 0
            zeros(length(x), length(x))  # Return zero matrix
        else
            √(2 * γ(t) * dγdt(t)) * LinearAlgebra.I(length(x))  # Scalar times identity matrix
        end

    μ0 = SDEState(x0, 0.0)
    dimension = length(x0)
    return SDEProcess(μ, σ, μ0, dimension)
end

function step_forward(Xt::SDEState{T}, process::SDEProcess{T}; Δt=1e-4) where {T}
    new_state = Xt.state .+ Δt .* process.μ(Xt.time, Xt.state) .+ process.σ(Xt.time, Xt.state) * rand(Normal(0, √Δt), process.dimension)
    new_time = Xt.time + Δt
    return SDEState{T}(new_state, new_time)
end

function realise_path(Xt::SDEState, process::SDEProcess, t; Δt=1e-4)
    current_state = Xt
    time_steps = Int(floor(t / Δt))
    points = []
    for _ in 1:time_steps
        current_state = step_forward(current_state, process; Δt=Δt)
        push!(points, current_state)
    end
    current_state = step_forward(current_state, process; Δt=t - time_steps * Δt)
    push!(points, current_state)
    return points
end
function forward(Xt::SDEState, process::SDEProcess, t; Δt=1e-4)
    current_state = Xt
    time_steps = Int(floor(t / Δt))

    for _ in 1:time_steps
        current_state = step_forward(current_state, process; Δt=Δt)
    end
    current_state = step_forward(current_state, process; Δt=t - time_steps * Δt)

    return current_state
end


