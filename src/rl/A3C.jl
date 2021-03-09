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
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]

            # pack the A3C struct with agent info
            state_hist = zeros(Bool, length(flattened_state), model.num_steps)
            action_hist = Array{Tuple, 2}(undef, 1, model.num_steps)
            reward_hist = zeros(Float64, 1, model.num_steps)
            agent_i_A3C = A3C_Agent(theta, theta_v, state_hist, action_hist, reward_hist)
            model.A3C[i] = agent_i_A3C
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
        return flattened_state
    
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

function PolicyEvaluate(model, i)
    state = GetSubstate(model, i)
    action_dist = model.A3C[i].Pi_prime(state)
    action_dist = reshape(action_dist, (length(action_dist),))
    
    actions = []
    [push!(actions, x) for x in 1:model.num_goals]
    push!(actions, :random)
    action = sample(actions, ProbabilityWeights(action_dist))
    return action  # returns an integer corresponding to goal or :random symbol
end


