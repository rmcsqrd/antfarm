using Agents, Random, AgentsPlots, Plots, ProgressMeter

mutable struct FMP_Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    tau::NTuple{2, Float64}
    color::String
    type::Symbol
    radius::Float64
    # NOTE: if you change anything in this you need to restart the REPL
    # (I think it is the precompilation step)
end

"""
Initialization function for FMP simulation. Contains all model parameters.
"""
function FMP_Model(simtype;
                   rho = 7.5e6,
                   rho_obstacle = 7.5e6,
                   dt = 0.01,
                   num_agents = 50,
                   SS_dims = (1,1),
                   num_steps = 200,
                   terminal_max_dis = 0.01,
                   c1 = 10,
                   c2 = 10,
                   vmax = 0.1,
                   d = 0.01,
                   r = (3*vmax^2/(2*rho))^(1/3)+d,
                   obstacle_list = [],
                  )

    # define AgentBasedModel (ABM)
    properties = Dict(:rho=>rho,
                      :rho_obstacle=>rho_obstacle,
                      :r=>r,
                      :dt=>dt,
                      :num_agents=>num_agents,
                      :num_steps=>num_steps,
                      :terminal_max_dis=>terminal_max_dis,
                      :c1=>c1,
                      :c2=>c2,
                      :vmax=>vmax,
                      :obstacle_list=>obstacle_list,
                     )
    
    space2d = ContinuousSpace(2; periodic=true, extend=SS_dims)
    model = ABM(FMP_Agent, space2d, properties=properties)
    AgentPositionInit(model, num_agents; type=simtype)

    # append obstacles into obstacle_list
    for agent in allagents(model)
        if agent.type == :O
            append!(model.obstacle_list, agent.id)
        end
    end
    
    index!(model)
    return model

end

"""
Simulation wrapper for FMP simulations. 

Initializes model based on "type" parameter which dictates type of simulation to perform. Different simulation descriptions can be found in `/multiagent/simulation_init.jl`.

Next it loops through the number of simulation steps (specified in model params) and create simulation display using `plotabm()`.

Finished by saving at the location specified by `outputpath` variable. 
"""
function FMP_Simulation(simtype::String; outputpath = "output/simresult.gif")
    gr()
    cd(@__DIR__)
    
    model = FMP_Model(simtype)
    agent_step!(agent, model) = move_agent!(agent, model, model.dt)

    e = model.space.extend
    num_steps = model.num_steps

    p = Progress(round(Int,num_steps/2))
    anim = @animate for i in 1:2:num_steps

        FMP(model)
        p1 = plotabm(
            model,
            as = PlotABM_RadiusUtil,
            ac = PlotABM_ColorUtil,
            am = PlotABM_ShapeUtil,
            #showaxis = false,
            grid = false,
            xlims = (0, e[1]),
            ylims = (0, e[2]),
        )

        title!(p1, "FMP Simulation (step $(i))")
        step!(model, agent_step!, 2)
        next!(p)
    end
    gif(anim, outputpath, fps = 25)

end



