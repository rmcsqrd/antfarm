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
using Flux.Optimise: update!
using StatsBase
using Agents
using InteractiveDynamics
using DataFrames
using BSON
import CairoMakie


# include simulation stuff
include("multiagent/simulation_init.jl")
include("multiagent/plot_utils.jl")
include("multiagent/simulation.jl")
include("multiagent/rl_simulation_init.jl")

# include RL stuff
include("rl/MDP_formulation.jl")
#include("rl/transition_functions.jl")
include("rl/A3C.jl")

# include analysis stuff
include("analysis/analysis_tools.jl")

# include util stuff
include("utils/utils.jl")

end # module
