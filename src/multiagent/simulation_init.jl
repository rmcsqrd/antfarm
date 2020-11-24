"""
Wrapper function for generating agent and object positions/targets. Different position functions share common inputs:

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
    elseif type == "centered_line_object"
        return CenteredLineObject(model, num_agents)
    elseif type == "moving_line"
        return CenteredObjectMovingLine(model, num_agents)
    elseif type == "random"
        return RandomPositions(model, num_agents)
    else
        return RandomPositions(model, num_agents)  # return random anyways
    end
end

"""
Simulation where agents start on one side of the state space and move in a vertical line from left to right.
"""
function LinePositions(model, num_agents)

    x, y = model.space.extend
    for i in 1:num_agents
        xi = 0.1*x
        yi = y - (0.1*y+0.9*y/(num_agents)*(i-1))

        pos = (xi, yi)
        vel = (0,0)
        tau = pos .+ (0.8*x, 0)
        radius = 1
        color = AgentInitColor(i, num_agents)
        add_agent!(pos, model, vel, tau, color, :A, radius)  # add agents
        add_agent!(tau, model, vel, tau, color, :T, radius)  # add targets

    end
    return model

end

"""
Simulation with unmoving vertical line of agents in middle of state space. A moving object is moving from left to right through line of agents. Agents must move around object and attempt to reorient themselves in the vertical line.
"""
function CenteredLineObject(model, num_agents)
    
    x, y = model.space.extend
    for i in 1:num_agents
        xi = 0.5*x
        yi = y - (0.1*y+0.9*y/(num_agents)*(i-1))

        pos = (xi, yi)
        vel = (0,0)
        tau = pos 
        radius = 0.05
        color = AgentInitColor(i, num_agents)
        add_agent!(pos, model, vel, tau, color, :A, radius)  # add agents
        
    end

    xio = 0.1*x
    yio = 0.5*y
    object_pos = (xio, yio)
    object_vel = (0,0)
    object_tau = (x-0.1*x, 0.5*y)
    object_radius = 0.2
    color = "#ff0000"
    add_agent!(object_pos, model, object_vel, object_tau, color, :O, object_radius)  # add object
    add_agent!(object_tau, model, object_vel, object_tau, color, :T, object_radius)  # add object target

    return model

end

"""
Simulation similar to "Line Positions" with object in middle of state space that agents must navigate around.
"""
function CenteredObjectMovingLine(model, num_agents)
    
    x, y = model.space.extend
    for i in 1:num_agents
        xi = 0.1*x
        yi = y - (0.1*y+0.9*y/(num_agents)*(i-1))

        pos = (xi, yi)
        vel = (0,0)
        tau = (0.8*x,0) .+ pos 
        radius = 0.02
        color = AgentInitColor(i, num_agents)
        add_agent!(tau, model, vel, tau, color, :T, radius)  # add object target
        add_agent!(pos, model, vel, tau, color, :A, radius)  # add agents
        
    end

    xio = 0.5*x
    yio = 0.5*y
    object_pos = (xio, yio)
    object_vel = (0,0)
    object_tau = object_pos
    object_radius = 0.2
    color = "#ff0000"
    add_agent!(object_pos, model, object_vel, object_tau, color, :O, object_radius)  # add object

    return model

end

"""
Agents start around the perimeter of a circle and attempt to move to a position on the opposite side of the circle - all agents end up driving towards the center.
"""
function CirclePositions(model, num_agents)

    # determine circle params
    x, y = model.space.extend
    r = 0.9*(min(x,y)/2)

    for i in 1:num_agents

        # compute position around circle
        theta_i = (2*π/num_agents)*i
        xi = r*cos(theta_i)+x/2
        yi = r*sin(theta_i)-y/2
        
        xitau = r*cos(theta_i+π)+x/2
        yitau = r*sin(theta_i+π)+y/2

        # set agent params
        pos = (xi, yi)
        vel = (0,0)
        tau = (xitau, yitau)  # goal is on opposite side of circle
        #tau = (x/2,y/2)
        type = :A
        radius = 0.02
        color = AgentInitColor(i, num_agents)
        add_agent!(pos, model, vel, tau, color, :A, radius)
        add_agent!(tau, model, vel, tau, color, :T, radius)
    end

    #add_agent!((x/2,y/2), model, (0,0), (x/2,y/2), "#ff0000", :O, 0.1)  # add object in middle
    return model

end

"""
Agents start in random positions with random velocities and seek a random target position.
"""
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
