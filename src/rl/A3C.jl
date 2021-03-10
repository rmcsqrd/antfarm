function A3C_Epoch_Init()

end

function A3C_Episode_Init(model, a3c_global_params)

    # get starting substate, all agents are same so just choose agent 1 for
    # seed dims
    flattened_state = GetSubstate(model, 1)

    # build network
    layer1 = LSTM(length(flattened_state), 128)
    layer2 = LSTM(128, model.num_goals+1)
    theta = Chain(layer1, layer2) # theta is policy
    theta_v = theta

    # seed each agent with networks
    goal_idx = 1
    for agent_id in keys(model.agents)

        # first, assign policy to agents
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            model.agents[agent_id].Pip = theta
            model.agents[agent_id].Vp = theta_v

        # next, create dict of goals. key = RL index (1:num_goals; NOT
        # Agents.jl agent.id), value = Agents.jl agent.pos
        elseif model.agents[agent_id].type == :T
            model.Goals[goal_idx] = model.agents[agent_id].pos
            goal_idx += 1
        end
    end

    # include global network params
    model.A3C[:Global_Theta] = theta
    model.A3C[:Global_Theta_v] = theta_v

end

function GetSubstate(model, i)

        # get agent substate and flatten
        GAi = model.SS.GA[i, :]
        GOi = model.SS.GO[i, :]
        GIi = model.SS.GI[i, :]
        AIi = model.SS.AI[i, :]

        flattened_state = state_flatten(GAi, GOi, GIi, AIi)
        return reshape(flattened_state, (length(flattened_state),))
    
end

function state_flatten(GAi, GOi, GIi, AIi)
    input_reshape(x) = reshape(x, (prod(size(x)), 1))

    flatten_state = [
                        input_reshape(GAi);
                        input_reshape(GOi);
                        input_reshape(GIi);
                        input_reshape(AIi)
                    ]
    return flatten_state
end

function PolicyEvaluate(model, agent_id)
    i = model.AgentHash[hash(agent_id)]
    state = GetSubstate(model, i)
    action_dist = model.agents[agent_id].Pip(state)
    action_dist = reshape(action_dist, (length(action_dist),))
    
    actions = []
    [push!(actions, x) for x in 1:model.num_goals]
    push!(actions, model.num_goals+1) # this is :random
    action = sample(actions, ProbabilityWeights(action_dist))
    return action  # returns an integer corresponding to goal or number larger than goals for random
end


