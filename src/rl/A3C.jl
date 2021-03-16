
mutable struct A3C_Global
    num_agents::Int64
    num_goals::Int64
    num_steps::Int64
    num_episodes::Int64
    episode_number::Int64
    Pi # note that we share params
    gamma::Float64
end

function A3C_Policy_Init(state_dim, num_goals)
    
    # build network
    theta = Chain(
                  Dense(state_dim, 128, relu),
                  LSTM(128, num_goals+2) # num goals, random action, V(si)
                 )
    return theta
end


function A3C_Episode_Init(model, A3C_params)

    # get seed policy/value
    # note that we share the same network
    theta = A3C_params.Pi

    # seed each agent with networks
    goal_idx = 1
    for agent_id in keys(model.agents)

        # first, assign policy to agents
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            model.agents[agent_id].Pip = theta

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
    policy_output = model.agents[agent_id].Pip(state)
    pi_sa = policy_output[1:model.num_agents+1]  #pi(s,a)
    vi_s = policy_output[length(policy_output)]  # v(s)
    pi_sa = reshape(pi_sa, (length(pi_sa),))
   
    # generate action list
    actions = []
    [push!(actions, x) for x in 1:model.num_goals]
    push!(actions, model.num_goals+1) # this is :random

    # get probabilities
    probs = ProbabilityWeights(softmax(pi_sa))

    # select action
    action = sample(actions, probs)
    return action, probs[action], vi_s  # returns an integer corresponding to goal or number larger than goals for random
end

function PolicyTrain(agent_data, A3C_params)

    agent_ids = agent_data[1:A3C_params.num_agents, :].id
    opt = ADAM()
    global_reward = 0
    policy = A3C_params.Pi
    temp_policy = A3C_params.Pi
    for id in agent_ids
        agent_df = agent_data[ [x==id for x in agent_data.id], :]
        R = last(agent_df).Value
        for t in reverse(1:A3C_params.num_steps-1)
            step_data = agent_df[ [step == t for step in agent_df.step], :]

            # note all this [1] nonsense is because step_data is a 1x9 df
            ri = step_data.Reward[1]
            vi = step_data.Value[1]
            pi_sa = step_data.PiAction[1]
            R += ri + A3C_params.gamma*R
            ps = params(policy)
            A_sa = R-vi  # Advantage(s,a)

            # update actor gradients based on loss from critic
            actor_gradients = gradient(() -> log(pi_sa)*A_sa, ps)
            update!(opt, ps, actor_gradients)

            # update critic gradients
            critic_gradients = gradient(() -> A_sa^2, ps)
            update!(opt, ps, actor_gradients)
        end
        global_reward += R
        
    end
    println("Global Reward for Epoch = $global_reward")
end
