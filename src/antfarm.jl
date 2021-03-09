module antfarm

# imports
using Revise
using Debugger
using Colors
using Random
using Plots
using LinearAlgebra
using ProgressMeter
using Flux
using StatsBase
using Agents
using InteractiveDynamics
using DataFrames
import CairoMakie


# include simulation stuff
include("multiagent/simulation_init.jl")
include("multiagent/simulation_utils.jl")
include("multiagent/simulation.jl")
include("multiagent/rl_simulation_init.jl")

# include RL stuff
include("rl/MDP_formulation.jl")
#include("rl/transition_functions.jl")
include("rl/A3C.jl")

# include analysis stuff
include("analysis/analysis_tools.jl")

end # module
