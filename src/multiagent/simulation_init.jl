"""
Wrapper function for generating agent positions. Different position functions share common inputs:

type = type of agent
    :A = agent (has velocity, subject to repulsive forces)
    :O = object (has no velocity, cannot move, agents must go around)
    :T = target (no velocity, just a projection of agent target zone)
radius = radius of agent multiplier (1 is normal)
"""
function AgentPositionInit(model, num_agents; type="random")

    if type == "circle"
        return CirclePositions(model, num_agents)
    elseif type == "line"
        return LinePositions(model, num_agents)
    elseif type == "random"
        return RandomPositions(model, num_agents)
    else
        return RandomPositions(model, num_agents)  # return random anyways
    end
end

function LinePositions(model, num_agents)

    x, y = model.space.extend
    for i in 1:num_agents
        xi = 0.1*x
        yi = y - (0.1*y+0.9*y/(num_agents)*(i-1))

        pos = (xi, yi)
        vel = (0,0)
        tau = pos .+ (0.8*x, 0)
        type = :O
        radius = 1
        color = AgentInitColor(i, num_agents)
        add_agent!(pos, model, vel, tau, color, type, radius)

    end
    return model

end

function CirclePositions(model, num_agents)

    # determine circle params
    x, y = model.space.extend
    r = 0.9*(min(x,y)/2)

    for i in 1:num_agents

        # compute position around circle
        theta_i = (2*π/num_agents)*i
        xi = r*cos(theta_i)+x/2
        yi = r*sin(theta_i)-y/2

        # set agent params
        pos = (xi, yi)
        vel = (0,0)
        tau = -1 .* pos  # goal is on opposite side of circle
        #tau = (x/2,y/2)
        type = :A
        radius = 0.02
        color = AgentInitColor(i, num_agents)
        add_agent!(pos, model, vel, tau, color, type, radius)
    end
    return model

end

function RandomPositions(model, num_agents)
    Random.seed!(42)
    for i in 1:num_agents
        pos = Tuple(rand(2))
        vel = Tuple(rand(2))
        tau = Tuple(rand(2))
        type = :A
        radius = 0.02
        color = AgentInitColor(i, num_agents)
        add_agent!(pos, model, vel, tau, color, type, radius)
    end

    return model

end
