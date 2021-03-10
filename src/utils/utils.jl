function StateSpaceHashing(model)

    # create associations between agent ID's and RL matrices
    #   Note that Agents.jl stores goal positions and objects as agents with
    #   unique IDs but this gives issues when trying to index into the RL state
    #   space arrays. We solve this by hashing
    
    # start by getting list of agent ids that have symbol :A
    agent_list = [hash(agent_id) for agent_id in keys(model.agents) if model.agents[agent_id].type == :A]
    model.AgentHash = Dict(zip(agent_list, 1:length(agent_list)))
    
    # next get list of ids that have symbol :T
    # create ways to "hash into" state space array from ContinuousSpace model, and "hash out of" state
    # space array into ContinuousSpace model so we can update agent target
    # positions
    goal_list = [goal_id for goal_id in keys(model.agents) if model.agents[goal_id].type == :T]
    for (idx, goal_id) in enumerate(goal_list)

        # Agents.jl id -> RL id is hashed
        model.GoalHash[hash(goal_id)] = idx

        # RL id -> Agents.jl id is not hashed (to avoid collisions)
        model.GoalHash[idx] = goal_id 
    end

end

"""
Run model, return data
"""
function RunModelCollect(model, agent_step!, model_step!)
    adata = [ :type, :State, :Action, :Reward]
    agent_data, _ = run!(model, agent_step!, model_step!, model.num_steps; adata)
    return agent_data
end

"""
Run model only
"""
function RunModel(model, agent_step!, model_step!)
    
    for i in 1:model.num_steps
        step!(model, agent_step!, model_step!)
        next!(p)
    end
end

"""
Run model and create output plot
"""
function RunModelPlot(model, agent_step!, model_step!)
    # delete original file
    filepath = "/Users/riomcmahon/Desktop/circle_swap.mp4"
    try
        rm(filepath)
    catch
        println("file doesn't exist, moving on")
    end

    # plot stuff
    InteractiveDynamics.abm_video(
        filepath,
        model,
        agent_step!,
        model_step!,
        title = "FMP Simulation",
        frames = model.num_steps,
        framerate = 100,
        resolution = (600, 600),
        as = PlotABM_RadiusUtil,
        ac = PlotABM_ColorUtil,
        am = PlotABM_ShapeUtil,
        equalaspect=true,
        scheduler = PlotABM_Scheduler,
       )

end
