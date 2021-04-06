module antfarm

# imports
using Revise
using Debugger
using Colors
using ProgressMeter: @showprogress
using Random
using Plots
using LinearAlgebra
using Flux
using Zygote
using Zygote: Params, Grads, @showgrad, dropgrad
using Flux.Optimise: update!, train!
using StatsBase
using Agents
using InteractiveDynamics
using DataFrames
using BSON
import CairoMakie


# include simulation stuff
#include("simulation/model_init.jl")
include("simulation/simulation_wrapper.jl")
include("simulation/rl_model_init.jl")
include("simulation/simulation_utils.jl")

# include RL stuff
include("rl/MDP_formulation.jl")
include("rl/A3C.jl")
include("rl/DQN.jl")

# include analysis stuff
include("analysis/analysis_tools.jl")

# include fmp stuff
include("fmp/fmp.jl")


end # module
