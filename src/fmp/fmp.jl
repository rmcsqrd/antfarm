using Agents, LinearAlgebra

function FMP(model)

    # get list of interacting_pairs within some radius
    agent_iter = interacting_pairs(model, model.r, :all)

    # construct interaction_array which is (num_agents x num_agents)
    #   boolean array where interaction_array[i,j] = true implies that
    #   agent_i and agent_j are within the specified interaction radius
    #   (this array will be symmetric about the main diagonal)
    interaction_array = falses(model.num_agents, model.num_agents)
    agents = agent_iter.agents
    for pair in agent_iter.pairs
        i, j = pair
        interaction_array[i, j] = true
        interaction_array[j, i] = true
    end

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
    ui = fiR .+ fiGamma
    vi = agents[i].vel .+ ui .* model.dt
    vi = CapVelocity(model.vmax, vi)

    # update agent velocities
    agents[i].vel = vi

end

function RepulsiveForce(model, agents, i, Ni)
    f = ntuple(i->0, length(agents[i].vel))
    for j in Ni
        dist = norm(agents[j].pos .- agents[i].pos)
        if dist < model.r
            force = -model.rho * (dist - model.r)^2
            distnorm = (agents[j].pos .- agents[i].pos) ./dist
            f = f .+ (force .* distnorm)
        end
    end
    return f
end

function NavigationalFeedback(model, agents, i)
    f = -model.c1 .* (agents[i].pos .- agents[i].tau) .- model.c2 .* agents[i].vel
    return f
end

function CapVelocity(vmax, vel)
    if norm(vel) > vmax
        vi = (vel ./ norm(vel)) .* vmax
        return vi
    else
        return vel
    end
end
