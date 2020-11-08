using Colors

function PlotABM_ColorUtil(a)
    if a.type == :A || a.type == :O
        return a.color
    else
        return "#ffffff"
    end
end

function PlotABM_ShapeUtil(a)
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

function PlotABM_RadiusUtil(a)
    return a.radius * 200  # this is for display purposes only
end

function AgentInitColor(i, num_agents)
    color_range = range(HSV(0,1,1), stop=HSV(-360,1,1), length=num_agents)
    agent_color = color_range[i]
    return string("#", hex(agent_color))

end
