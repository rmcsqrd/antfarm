using Agents, LinearAlgebra

function FMP(model)

    # get list of interacting_pairs within some radius
    agent_iter = interacting_pairs(model, model.r, :all)

    # construct interaction_array which is (num_agents x num_agents)
    #   array where interaction_array[i,j] = 1 implies that
    #   agent_i and agent_j are within the specified interaction radius
    interaction_array = falses( nagents(model), nagents(model))
    agents = agent_iter.agents
    for pair in agent_iter.pairs

        i, j = pair
        if agents[i].type == :A && agents[j].type == :A
            interaction_array[i, j] = true
            interaction_array[j, i] = true
        end

    end

    # determine object ids


    # loop through agents and update velocities
    for i in keys(agents)
        Ni = findall(x->x==1, interaction_array[i, :])
        
        # move_this_agent_to_new_position(i) in FMP paper
        UpdateVelocity(model, i, Ni, agents)
    end
end

function UpdateVelocity(model, i, Ni, agents)

    # compute forces and resultant velocities
    fiR = RepulsiveForce(model, agents, i, Ni)
    fiGamma = NavigationalFeedback(model, agents, i)
    fiObject = ObstactleFeedback(model, agents, i)
    ui = fiR .+ fiGamma .+ fiObject
    vi = agents[i].vel .+ ui .* model.dt
    vi = CapVelocity(model.vmax, vi)

    # update agent velocities
    agents[i].vel = vi

end

function RepulsiveForce(model, agents, i, Ni)
    # compute repulsive force for each agent
    # note the "." before most math operations, required for component wise tuple math
    f = ntuple(i->0, length(agents[i].vel))
    for j in Ni
        dist = norm(agents[j].pos .- agents[i].pos)
        if dist < model.r
            force = -model.rho * (dist - model.r)^2
            distnorm = (agents[j].pos .- agents[i].pos) ./dist
            f = f .+ (force .* distnorm)
        end
    end

    # targets/objects do not experience repulsive feedback
    if agents[i].type == :O || agents[i].type == :T
        return  ntuple(i->0, length(agents[i].vel))
    else
        return f
    end
end

function NavigationalFeedback(model, agents, i)
    # compute navigational force for each agent
    # note the "." before most math operations, required for component wise tuple math
    f = (-model.c1 .* (agents[i].pos .- agents[i].tau)) .+ (- model.c2 .* agents[i].vel)
    if agents[i].type == :T
        return  ntuple(i->0, length(agents[i].vel))  # targets to not experience navigational feedback
    else
        return f
    end
end

function ObstactleFeedback(model, agents, i)
    # determine obstacle avoidance feedback term
    # note the "." before most math operations, required for component wise tuple math
    f = ntuple(i->0, length(agents[i].vel))

    for id in model.obstacle_list
        dist = norm(agents[id].pos  .- agents[i].pos)
        if dist < agents[id].radius/2
            force = -model.rho_obstacle * (dist - agents[id].radius)^2
            distnorm = (agents[id].pos .- agents[i].pos) ./ norm(agents[id].pos .- agents[i].pos)
            f = f .+ (force .* distnorm)
        end
    end
    if agents[i].type == :O || agents[i].type == :T
        return ntuple(i->0, length(agents[i].vel))
    else
        return f
    end
    

end

function CapVelocity(vmax, vel)
    # bound velocity by vmax
    # note the "." before most math operations, required for component wise tuple math
    if norm(vel) > vmax
        vi = (vel ./ norm(vel)) .* vmax
        return vi
    else
        return vel
    end
end
