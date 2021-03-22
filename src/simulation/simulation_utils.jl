"""
Run model, return data
"""
function RunModelCollect(model, agent_step!, model_step!)
    adata = [ :type, :State, :Action, :PiAction, :Value, :Reward]
    agent_data, _ = run!(model, agent_step!, model_step!, model.num_steps; adata)
    return agent_data
end

"""
Run model only
"""
function RunModel(model, agent_step!, model_step!)
    
    for i in 1:model.num_steps
        step!(model, agent_step!, model_step!)
    end
end

"""
Run model and create output plot
"""
function RunModelPlot(model, agent_step!, model_step!, filepath)
    # delete original file
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
        as = as_f(a) = 1200*1/minimum(a.SSdims)*a.radius,  ## this was defined empirically
        ac = ac_f(a) = a.type in (:A, :O) ? a.color : "#ffffff",
        am = am_f(a) = a.type in (:A, :O, :T) ? :circle : :circle,
        equalaspect=true,
        scheduler = PlotABM_Scheduler,
       )

end


"""
This function is a scheduler to determine draw order of agents. Draw order (left to right) is :T, :O, :A
"""
function PlotABM_Scheduler(model::ABM)

    # init blank lists
    agent_list = []
    object_list = []
    target_list = []
    for agent in values(model.agents)
        if agent.type == :A
            append!(agent_list, agent.id)
        elseif agent.type == :T
            append!(target_list, agent.id)
        elseif agent.type == :O
            append!(object_list, agent.id)
        end
    end

    # make composite list [targets, objects, agents]
    draw_order = []
    append!(draw_order, target_list)
    append!(draw_order, object_list)
    append!(draw_order, agent_list)

    return draw_order
end
