module ForwardBackward

using Distributions, LinearAlgebra, ArraysOfArrays, Zygote

include("maths.jl")
include("processes.jl")
include("states.jl")
include("propagation.jl")
include("manifolds.jl")
include("sdeprocess.jl")
include("conditional_bridge.jl")

export
    #Abstract Types
    Process,
    DiscreteProcess,
    State,
    StateLikelihood,
    ContinuousProcess,
    SDEProcess,
    #Processes
    Deterministic,    
    BrownianMotion,
    OrnsteinUhlenbeck,
    SwitchingBM,
    SwitchingSDEProcess,
    ConditionalBridgeProcess,
    ConditionalBridgeState,
    UniformDiscrete,
    UniformUnmasking,
    GeneralDiscrete,
    PiQ,
    #Likelihoods & States
    CategoricalLikelihood,
    GaussianLikelihood,
    SwitchingSDELikelihood,
    DiscreteState,
    ContinuousState,
    SwitchingSDEState,
    #Functions
    endpoint_conditioned_sample,
    interpolate,
    ⊙,
    forward,
    backward,
    forward!,
    backward!,
    tensor,
    sumnorm,
    stochastic,
    #Manifolds
    ManifoldProcess,
    ManifoldState,
    perturb!,
    perturb,
    expand,
    SDEState,
    SDEProcess,
    realise_path
end
