module antfarm

# imports
using Revise
using Debugger
using CSV
using Colors
using Random
using Plots
using LinearAlgebra
using ProgressMeter
using Flux
using Zygote: Params, Grads, @showgrad, dropgrad
using Flux.Optimise: update!
using StatsBase
using Agents
using InteractiveDynamics
using DataFrames
using BSON
import CairoMakie


# include simulation stuff
include("simulation/model_init.jl")
include("simulation/simulation_wrapper.jl")
include("simulation/rl_model_init.jl")
include("simulation/simulation_utils.jl")

# include RL stuff
include("rl/rl_wrapper.jl")
include("rl/MDP_formulation.jl")
#include("rl/POMDP_formulation.jl")
include("rl/A3C.jl")

# include analysis stuff
include("analysis/analysis_tools.jl")

# include fmp stuff
include("fmp/fmp.jl")


end # module
