using Agents, Random, AgentsPlots, Plots, ProgressMeter

function DispersalModel(;
                   rho = 7.5e6,
                   rho_obstacle = 7.5e6,
                   step_inc = 2,
                   dt = 0.01,
                   num_agents = 100,
                   SS_dims = (1,1),  # x,y should be equal for proper plot scaling
                   num_steps = 400,
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
    
    space2d = ContinuousSpace(2; periodic=true, extend=SS_dims)
    model = ABM(FMP_Agent, space2d, properties=properties)
    AgentPositionInit(model, num_agents; type="circle")

    # append obstacles into obstacle_list
    for agent in allagents(model)
        if agent.type == :O
            append!(model.obstacle_list, agent.id)
        end
    end
    
    index!(model)
    return model

end

function DispersalSimulation(model; outputpath = "output/simresult.gif")
    gr()
    cd(@__DIR__)
    
    # init model
    agent_step!(agent, model) = move_agent!(agent, model, model.dt)
    
    # init state space
    e = model.space.extend
    step_range = 1:model.step_inc:model.num_steps


    mean_norms = Array{Float64}(undef,length(step_range))

    # loop through steps and compute mean distance
    for (plotcnt, i) in enumerate(step_range)
        FMP(model)
        adata = [:pos, :type]
        agent_df = init_agent_dataframe(model, adata)
        step!(model, agent_step!, model.step_inc)

        collect_agent_data!(agent_df, model, adata)
        mean_norms[plotcnt] = GridlockDispersalAnalyze(agent_df, model)
    end
    return mean_norms

end
