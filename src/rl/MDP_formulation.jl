mutable struct StateSpace
    # Goal awareness vector for agent
    # GA(i, g) = agent i is interacting with goal g
    GA::Array{Bool,2} 

    GO::Array{Bool,2}
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

            # "invert reward" because gradient descent wants to minimize loss.
            # We want to maximize reward so inversion make large reward as
            # small as possible.
            #rewards[i] = rewards[i]*-1 


        end
    end
    return rewards
end

