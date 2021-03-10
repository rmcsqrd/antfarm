mutable struct A3C_Global
    num_agents::Int64
    num_goals::Int64
    num_steps::Int64
    num_episodes::Int64
    episode_number::Int64
    Pi
    V
end

function A3C_Policy_Init(state_dim, num_goals)
    
    # build network
    layer1 = LSTM(state_dim, 128)
    layer2 = LSTM(128, num_goals+1)
    theta = Chain(layer1, layer2) # theta is policy
    return theta
end

function A3C_Value_Init()
    theta_v = 1  # BONE this is dummy
    return theta_v
end

function A3C_Episode_Init(model, A3C_params)

    # get seed policy/value
    theta = A3C_params.Pi
    theta_v = A3C_params.V

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
   
    # generate action list
    actions = []
    [push!(actions, x) for x in 1:model.num_goals]
    push!(actions, model.num_goals+1) # this is :random

    # get probabilities
    probs = ProbabilityWeights(softmax(action_dist))

    # select action
    action = sample(actions, ProbabilityWeights(action_dist))
    return action  # returns an integer corresponding to goal or number larger than goals for random
end


