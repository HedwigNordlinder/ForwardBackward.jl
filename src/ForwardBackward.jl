module ForwardBackward

using Distributions, LinearAlgebra, ArraysOfArrays, Zygote

include("maths.jl")
include("processes.jl")
include("states.jl")
include("propagation.jl")
include("manifolds.jl")

export
    # Abstract Types
    Process,
    DiscreteProcess,
    State,
    StateLikelihood,
    ContinuousProcess,
    # Processes
    Deterministic,
    BrownianMotion,
    OrnsteinUhlenbeck,
    UniformDiscrete,
    UniformUnmasking,
    GeneralDiscrete,
    PiQ,
    SwitchBridgeProcess,
    # Likelihoods & States
    CategoricalLikelihood,
    GaussianLikelihood,
    DiscreteState,
    ContinuousState,
    SwitchBridgeState,
    # Functions
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
    # Manifolds
    ManifoldProcess,
    ManifoldState,
    perturb!,
    perturb,
    expand

end
