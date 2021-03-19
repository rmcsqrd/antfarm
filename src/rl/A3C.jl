
mutable struct A3C_Global
    num_agents::Int64
    num_goals::Int64
    num_steps::Int64
    num_episodes::Int64
    episode_number::Int64
    model # this is the model architecture
    θ     # this is the set of parameters. We share params so θ = θ_v
    gamma::Float64
end

function A3C_Episode_Init(model, A3C_params)

    # seed each agent with networks
    goal_idx = 1
    for agent_id in keys(model.agents)

        # first, assign policy to agents
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            model.agents[agent_id].Model = A3C_params.model

        # create dict of goals. key = RL index (1:num_goals; NOT
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

        input_reshape(x) = reshape(x, (prod(size(x)), 1))
        flattened_state = [
                        input_reshape(GAi);
                        input_reshape(GOi);
                        input_reshape(GIi);
                        input_reshape(AIi)
                        ]
        return reshape(flattened_state, (length(flattened_state),))
    
end

function PolicyEvaluate(model, agent_id)
    i = model.AgentHash[hash(agent_id)]
    state = GetSubstate(model, i)
    policy_output = model.agents[agent_id].Model(state)
    pi_sa = policy_output[1:model.num_agents+length(model.Actions)]  #pi(s,a)
    vi_s = policy_output[length(policy_output)]  # v(s)
    pi_sa = reshape(pi_sa, (length(pi_sa),))
   
    # generate action list
    actions = []
    [push!(actions, x) for x in 1:model.num_goals]
    [push!(actions, x) for x in model.num_goals+1:model.num_goals+length(model.Actions)] 

    # get probabilities
    probs = ProbabilityWeights(softmax(pi_sa))

    # select action
    action = sample(actions, probs)
    return action, probs[action], vi_s  # returns an integer corresponding to goal or number larger than goals for random
end

function PolicyTrain(agent_data, A3C_params)

    agent_ids = agent_data[1:A3C_params.num_agents, :].id
    opt = RMSProp()
    global_reward = 0
    for id in agent_ids

        # get individual agent dataframe
        agent_df = agent_data[ [x==id for x in agent_data.id], :]
        R = last(agent_df).Value

        # get training data
        tmax = size(agent_df)[1]
        data = Array{Tuple{Float64, Float64}}(undef, tmax)
        rewards = agent_df.Reward[1:tmax-1]
        values = agent_df.Value[1:tmax-1]
        pi_action = agent_df.PiAction[1:tmax-1]
        advantages = zeros(tmax-1)

        # build training data
        for i in reverse(1:tmax-1)

            # compute advantage
            R = rewards[i] + A3C_params.gamma*R
            advantages[i] = R - values[i]  # advantages for reverse order
            data[i] = (pi_action[i], advantages[i])
        end
        global_reward += sum(rewards)
        
        # create loss functions
        actor_loss_function(π_sa, A_sa) = log(π_sa)*A_sa
        critic_loss_function(π_sa, A_sa) = A_sa^2
        local actor_loss, critic_loss

        # train model
        for d in data
            
            # start with actor gradients
            dθ = gradient(A3C_params.θ) do
                actor_loss = actor_loss_function(d...)
                return actor_loss
            end
            update!(opt, A3C_params.θ, dθ)

            # next do critic gradients
            dθ_v = gradient(A3C_params.θ) do
                critic_loss = critic_loss_function(d...)
                return critic_loss
            end
            update!(opt, A3C_params.θ, dθ_v)
        end
    end
    return global_reward
end
