"""
Wrapper function for generating agent and object positions/targets. Different position functions share common inputs:

type = type of agent
    :A = agent (has velocity, subject to repulsive forces)
    :O = object (has no velocity, cannot move, agents must go around)
    :T = target (no velocity, just a projection of agent target zone)
radius = model.d/2 because model.d describes the minimum distance from agent centroid to agent centroid. Assuming that agents are circular and the same size, this means that model.d is also the diameter of the agent. Thus we use a radius of model.FMP_params.d/2
"""
function AgentPositionInit(model; type="random")

    if type == "circle"
        return CirclePositions(model)
    elseif type == "circle_object"
        return CirclePositionsObject(model)
    elseif type == "line"
        return LinePositions(model)
    elseif type == "centered_line_object"
        return CenteredLineObject(model)
    elseif type == "moving_line"
        return CenteredObjectMovingLine(model)
    elseif type == "random"
        return RandomPositions(model)
    else
        @warn "Invalid simulation type; simulating random"
        return RandomPositions(model)  # return random anyways
    end
end

"""
Simulation where agents start on one side of the state space and move in a vertical line from left to right.
"""
function LinePositions(model)

    x, y = model.space.extent
    for i in 1:model.num_agents
        xi = 0.1*x
        yi = y - (0.1*y+0.9*y/(model.num_agents)*(i-1))

        pos = (xi, yi)
        vel = (0,0)
        tau = pos .+ (0.8*x, 0)
        radius = model.FMP_params.d/2
        color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        add_agent!(pos, model, vel, tau, color, :A, radius, model.space.extent, [],[])  # add agents
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [],[])  # add targets

    end
    return model

end

"""
Simulation with unmoving vertical line of agents in middle of state space. A moving object is moving from left to right through line of agents. Agents must move around object and attempt to reorient themselves in the vertical line.
"""
function CenteredLineObject(model)
    
    x, y = model.space.extent
    for i in 1:model.num_agents
        xi = 0.5*x
        yi = y - (0.1*y+0.9*y/(model.num_agents)*(i-1))

        pos = (xi, yi)
        vel = (0,0)
        tau = pos 
        radius = model.FMP_params.d/2
        color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        add_agent!(pos, model, vel, tau, color, :A, radius, model.space.extent, [],[])  # add agents
        
    end

    xio = 0.1*x
    yio = 0.5*y
    object_pos = (xio, yio)
    object_vel = (0,0)
    object_tau = (x-0.1*x, 0.5*y)
    object_radius = 0.1
    color = "#ff0000"
    add_agent!(object_pos, model, object_vel, object_tau, color, :O, object_radius, model.space.extent, [],[])  # add object
    add_agent!(object_tau, model, object_vel, object_tau, color, :T, object_radius, model.space.extent, [],[])  # add object target

    return model

end

"""
Simulation similar to "Line Positions" with object in middle of state space that agents must navigate around.
"""
function CenteredObjectMovingLine(model)
    
    x, y = model.space.extent
    for i in 1:model.num_agents
        xi = 0.1*x
        yi = y - (0.1*y+0.9*y/(model.num_agents)*(i-1))

        pos = (xi, yi)
        vel = (0,0)
        tau = (0.8*x,0) .+ pos 
        radius = model.FMP_params.d/2
        color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [],[])  # add object target
        add_agent!(pos, model, vel, tau, color, :A, radius, model.space.extent, [],[])  # add agents
        
    end

    xio = 0.3*x
    yio = 0.5*y
    object_pos = (xio, yio)
    object_vel = (0,0)
    object_tau = object_pos
    object_radius = 0.2
    color = "#ff0000"
    add_agent!(object_pos, model, object_vel, object_tau, color, :O, object_radius, model.space.extent, [],[])  # add object

    return model

end

"""
Agents start around the perimeter of a circle and attempt to move to a position on the opposite side of the circle - all agents end up driving towards the center.
"""
function CirclePositions(model)

    # determine circle params
    x, y = model.space.extent
    r = 0.9*(min(x,y)/2)

    for i in 1:model.num_agents

        # compute position around circle
        theta_i = (2*π/model.num_agents)*i
        xi = r*cos(theta_i)+x/2
        yi = r*sin(theta_i)+y/2
        
        xitau = r*cos(theta_i+π)+x/2
        yitau = r*sin(theta_i+π)+y/2

        # set agent params
        pos = (xi, yi)
        vel = (0,0)
        tau = (xitau, yitau)  # goal is on opposite side of circle
        #tau = (x/2,y/2)
        type = :A
        radius = model.FMP_params.d/2
        color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        add_agent!(pos, model, vel, tau, color, :A, radius, model.space.extent, [], [])
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [], [])
    end

    return model

end


"""
Agents start around the perimeter of a circle and attempt to move to a position on the opposite side of the circle - all agents end up driving towards the center. There is also an object in the middle.
"""
function CirclePositionsObject(model)

    # determine circle params
    x, y = model.space.extent
    r = 0.9*(min(x,y)/2)

    for i in 1:model.num_agents

        # compute position around circle
        theta_i = (2*π/model.num_agents)*i
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
        radius = model.FMP_params.d/2
        color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        add_agent!(pos, model, vel, tau, color, :A, radius, model.space.extent, [],[])
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [],[])
    end

    object_radius = 0.1
    add_agent!((x/2,y/2), model, (0,0), (x/2,y/2), "#ff0000", :O, object_radius, model.space.extent, [],[])  # add object in middle
    return model

end

"""
Agents start in random positions with random velocities and seek a random target position.
"""
function RandomPositions(model)
    Random.seed!(42)
    for i in 1:model.num_agents
        pos = Tuple(rand(2))
        vel = Tuple(rand(2))
        tau = Tuple(rand(2))
        type = :A
        radius = model.FMP_params.d/2
        color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        add_agent!(pos, model, vel, tau, color, type, radius, model.space.extent, [],[])
    end

    return model

end
