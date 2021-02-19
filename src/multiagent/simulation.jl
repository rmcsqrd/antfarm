using Agents, Random, AgentsPlots, Plots, ProgressMeter

mutable struct FMP_Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    tau::NTuple{2, Float64}
    color::String
    type::Symbol
    radius::Float64
    SSdims::NTuple{2, Float64}  # include this for plotting
    # NOTE: if you change anything in this you need to restart the REPL
    # (I think it is the precompilation step)
end

"""
Initialization function for FMP simulation. Contains all model parameters.
"""
function FMP_Model(simtype;
                   rho = 7.5e6,
                   rho_obstacle = 7.5e6,
                   step_inc = 2,
                   dt = 0.01,
                   num_agents = 10,
                   SS_dims = (1, 1),  # x,y should be equal for proper plot scaling
                   num_steps = 300,
                   terminal_max_dis = 0.01,
                   c1 = 10,
                   c2 = 10,
                   vmax = 0.1,
                   d = 0.02, # distance from centroid to centroid
                   r = (3*vmax^2/(2*rho))^(1/3)+d,
                   obstacle_list = [],
                  )

    # define AgentBasedModel (ABM)
    properties = Dict(:rho=>rho,
                      :rho_obstacle=>rho_obstacle,
                      :step_inc=>step_inc,
                      :r=>r,
                      :d=>d,
                      :dt=>dt,
                      :num_agents=>num_agents,
                      :num_steps=>num_steps,
                      :terminal_max_dis=>terminal_max_dis,
                      :c1=>c1,
                      :c2=>c2,
                      :vmax=>vmax,
                      :obstacle_list=>obstacle_list,
                     )
    
    space2d = ContinuousSpace(SS_dims; periodic=true)
    model = ABM(FMP_Agent, space2d, properties=properties)
    AgentPositionInit(model, num_agents; type=simtype)

    # append obstacles into obstacle_list
    for agent in allagents(model)
        if agent.type == :O
            append!(model.obstacle_list, agent.id)
        end
    end
    
    return model

end

"""
Simulation wrapper for FMP simulations. 

Initializes model based on "type" parameter which dictates type of simulation to perform. Different simulation descriptions can be found in `/multiagent/simulation_init.jl`.

Possible inputs include:
- `circle` have agents move/swap places around perimeter of a circle
- `circle_object` have agents move/swap places around perimeter of a circle with an object in the middle
- `line` have agents move left to right in vertical line
- `centered_line_object` have agents remain stationary in vertical line and an object move through them
- `moving_line` have agents move left to right in vertical line past an object
- `random` have agents start in random positions with random velocities

Next it loops through the number of simulation steps (specified in model params) and create simulation display using `plotabm()`.

Finished by saving at the location specified by `outputpath` variable. 
"""
function FMP_Simulation(simtype::String; outputpath = "output/simresult.gif")
    gr()
    cd(@__DIR__)
    
    # init model
    model = FMP_Model(simtype)
    agent_step!(agent, model) = move_agent!(agent, model, model.dt)
    
    # init state space
    e = model.space.extent
    step_range = 1:model.step_inc:model.num_steps

    mean_norms = Array{Float64}(undef,1,)

    # setup progress meter counter
    p = Progress(round(Int,model.num_steps/model.step_inc))
    anim = @animate for i in step_range
        
        # step model including plot stuff
        FMP(model)
        p1 = AgentsPlots.plotabm(
            model,
            as = PlotABM_RadiusUtil,
            ac = PlotABM_ColorUtil,
            am = PlotABM_ShapeUtil,
            #showaxis = false,
            grid = false,
            xlims = (0, e[1]),
            ylims = (0, e[2]),
            aspect_ratio=:equal,
            scheduler = PlotABM_Scheduler,
        )
        title!(p1, "FMP Simulation (step $(i))")
        
        # step model and progress counter
        step!(model, agent_step!, model.step_inc)
        next!(p)
    end
    gif(anim, outputpath, fps = 100)

end



