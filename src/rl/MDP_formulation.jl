mutable struct StateSpace
    # Goal awareness vector for agent
    # GA(i, g) = agent i is interacting with goal g
    GA::Array{Bool,2} 

    GO::Array{Bool,2}
end

mutable struct RL_Wrapper
    params  # container for specific RL architecture params
    policy_train!  # function for training the RL architecture
    policy_evaluate  # function for evaluating the model state, returns an action
    episode_init!  # function for initializing RL struct at beginning of episode
    γ::Float64  # discount factor
end

function RL_Update!(model)
    # update global state (goal awareness/goal occupation)
    GlobalStateTransition(model)
    
    # compute rewards
    rewards = GlobalReward(model)

    # next, do individual agent actions
    goal_loc_array = sort(collect(values(model.Goals)))
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.Agents2RL[agent_id]

            # first, determine agent i's knowledge of goal positions
            GAi = model.SS.GA[i, :]

            # next, compare GA with goal locations. Return true location if
            # agent i is aware of goal location, (Inf, Inf) if not
            goal_pos_i = [xi == 1 ? yi : (-1,-1) for xi in GAi, yi in goal_loc_array[1,:]]  
            
            # vectorize to create state. Need to use iterators because
            # vec(Tuple) doesn't work
            s_t = [collect(Iterators.flatten(model.agents[agent_id].pos));
                   collect(Iterators.flatten(goal_pos_i));
                   vec(GAi)
                  ]
            s_t = round.(s_t, digits = 3)  # round to limit state space
            
            # select action according to RL policy
            t = model.ModelStep
            r_t = rewards[i]
            a_t = model.RL.policy_evaluate(i, t, s_t, r_t, model)

            # update model
            model.agents[agent_id].tau = model.agents[agent_id].pos .+ model.action_dict[a_t]

        end
    end

end

function GlobalStateTransition(model)
    model.SS.GO = zeros(Bool, model.num_agents, model.num_goals)

    # update goal awareness based on agent/target interaction
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.Agents2RL[agent_id]
            for goal_id in model.agents[agent_id].Gi
                g = model.Agents2RL[goal_id]
                model.SS.GA[i,g] = 1
                model.SS.GO[i,g] = 1
            end
        end
    end
    
    # update goal awareness based on agent/agent interaction
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.Agents2RL[agent_id]

            for neighbor_id in model.agents[agent_id].Ni

                # first update agent interactions
                j = model.Agents2RL[neighbor_id]

                # next update goal awareness using bitwise or
                # to simulate information exchange
                model.SS.GA[i, :] = model.SS.GA[i, :] .| model.SS.GA[j, :]  # period makes it elementwise
                model.SS.GA[j, :] = model.SS.GA[j, :] .| model.SS.GA[i, :]  # period makes it elementwise
            end
        end
    end
end

function GlobalReward(model)
    # compute team performance scaling factors
    team_performance = sum(model.SS.GO, dims=1)
    opt_performance = [model.num_agents/model.num_goals for x in 1:model.num_goals]
    team_performance = reshape(team_performance, length(team_performance))
    opt_performance = reshape(opt_performance, length(opt_performance))
    alpha = 1/max(1,norm(team_performance-opt_performance))

    rewards = zeros(Float64, model.num_agents)
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.Agents2RL[agent_id]

            # give reward for communication
            for neighbor_id in model.agents[agent_id].Ni  
                j = model.Agents2RL[neighbor_id]
                info_exchange = xor.(model.SS.GA[i, :], model.SS.GA[j, :])
                beta = sum(info_exchange)/model.num_goals
                rewards[i] += 10*beta
            end

            # get reward for goal occupation
            rewards[i] += sum(model.SS.GO[i,:]) 
            rewards[i] += sum(model.SS.GO[i,:])*alpha

            # agents pay penalty for goals they don't know location of
            rewards[i] += -0.1*sum(1 .- model.SS.GA[i, :])

        end
    end
    return rewards
end

