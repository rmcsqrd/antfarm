module antfarm
using Revise
using Debugger

## internal

include("multiagent/simulation.jl")
include("multiagent/simulation_init.jl")
include("multiagent/simulation_utils.jl")
include("fmp/fmp.jl")
include("analysis/gridlock_analysis.jl")
include("analysis/dispersal_simulation.jl")
include("analysis/simulation_wrapper.jl")



end # module
