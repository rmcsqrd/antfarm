using Agents, Random, AgentsPlots, Plots, ProgressMeter

mutable struct FMP_Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    tau::NTuple{2, Float64}
    # NOTE: if you change anything in this you need to restart the REPL
    # (I think it is the precompilation step)
end

# initialize model
function FMP_Model(; 
                   rho = 7.5e6,
                   r = 1,
                   dt = 0.01,
                   num_agents = 50,
                   SS_dims = (1,1),
                   num_steps = 100,
                   terminal_max_dis = 0.01,
                   c1 = 1,
                   c2 = 1,
                   vmax = 0.2,
                  )

    # define AgentBasedModel (ABM)
    properties = Dict(:rho=>rho,
                      :r=>r,
                      :dt=>dt,
                      :num_agents=>num_agents,
                      :num_steps=>num_steps,
                      :terminal_max_dis=>terminal_max_dis,
                      :c1=>c1,
                      :c2=>c2,
                      :vmax=>vmax,
                     )
    
    space2d = ContinuousSpace(2; periodic=true, extend=SS_dims)
    model = ABM(FMP_Agent, space2d, properties=properties)
    InitAgentPositions(model, num_agents)
    
    index!(model)
    return model

end

function FMP_Simulation()
    gr()
    cd(@__DIR__)
    
    model = FMP_Model()
    agent_step!(agent, model) = move_agent!(agent, model, model.dt)

    e = model.space.extend  # this gives dimensions of model space I think
    num_steps = model.num_steps

    p = Progress(round(Int,num_steps/2))
    anim = @animate for i in 1:2:num_steps

        FMP(model)

        p1 = plotabm(
            model,
            as = 4,
            #showaxis = false,
            #grid = false,
            xlims = (0, e[1]),
            ylims = (0, e[1]),
        )

        title!(p1, "step $(i)")
        step!(model, agent_step!, 2)
        next!(p)
    end
    gif(anim, "output/socialdist1.gif", fps = 25)

end

function InitAgentPositions(model, num_agents)

    Random.seed!(42)
    for ind in 1:num_agents
        pos = Tuple(rand(2))
        vel = Tuple(rand(2))
        tau = Tuple(rand(2))
        add_agent!(pos, model, vel, tau)
    end

    return model

end


