
"""
This function is a utility function that takes an agent and overwrites its color if it is of type :T (target) and gives it a grey color for display purposes.
"""
function PlotABM_ColorUtil(a::AbstractAgent)
    if a.type == :A || a.type == :O
        return a.color
    else
        return "#A9A9A9"
    end
end

"""
This function is a utility function for assigning agent display type based on agent type.
"""
function PlotABM_ShapeUtil(a::AbstractAgent)
    # potential options here:
    # https://gr-framework.org/julia-gr.html (search for "marker type")
    if a.type == :A
        return :circle
    elseif a.type == :O
        return :circle
    elseif a.type == :T
        return :circle
    else
        return :circle
    end
end

"""
This function is a utility function for setting agent plot size - note that this is for display purposes only and does not impact calculations involving agent radius. 
"""
function PlotABM_RadiusUtil(a::AbstractAgent)
    # this is for display purposes only and does not impact FMP algorithm results
    # the scaling values are empirically selected
    # the object scale is based on the agent scaling
    
    # magic number appears to be scaling factor for plotting
    #   ex, an agent/object with radius=1 place in center of SS
    #   would take up the entire state space. This was empirically
    #   tested. The 1/a.SSdims is also tested and works well.

    #SS_scale = 380*1/minimum(a.SSdims)  # technically a.SSdims[1] and [2] should be equal but just in case.
    SS_scale = 1000*1/minimum(a.SSdims)
    
    if a.type == :O
        return a.radius*SS_scale
    else
        return a.radius*SS_scale
    end
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

"""
This function is a utility function for coloring agents.
"""
function AgentInitColor(i, num_agents)
    color_range = range(HSV(0,1,1), stop=HSV(-360,1,1), length=num_agents)
    agent_color = color_range[i]
    return string("#", hex(agent_color))

end

