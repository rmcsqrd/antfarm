module antfarm

# imports
using Revise
using Debugger
using Colors
using Random
using Plots
using LinearAlgebra
using ProgressMeter
using Agents
using InteractiveDynamics
import GLMakie

# include simulation stuff
include("multiagent/simulation.jl")
include("multiagent/simulation_init.jl")
include("multiagent/simulation_utils.jl")

# include RL stuff
include("rl/rl_formulation.jl")

end # module
