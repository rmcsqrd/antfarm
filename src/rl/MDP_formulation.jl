mutable struct StateSpace
    # Goal awareness vector for agent
    # GA(i, g) = agent i is interacting with goal g
    GA::Array{Bool,2} 

    GO::Array{Bool,2}
end

function RL_Update!(model)
    # update global state (goal awareness/goal occupation)
    GlobalStateTransition!(model)
    
    # compute rewards
    rewards = GlobalReward(model)

    # next, do individual agent actions
    goal_loc_array = sort(collect(values(model.Goals)))
    for agent_id in keys(model.agents)

        if model.agents[agent_id].type == :A
            i = model.Agents2RL[agent_id]

            # get model state from agent and  push transition into replay buffer
            a_t1 = model.agents[agent_id].a_t1
            s_t1, s_t = get_state(model, agent_id, i)
            r_t = rewards[i]

            DQN_buffer_update!(s_t1, a_t1, r_t, s_t, model)

            # select action according to RL policy
            a_t = DQN_policy_eval!(s_t, model)
            model.agents[agent_id].a_t1 = a_t

            # update agent with action
            model.agents[agent_id].tau = model.agents[agent_id].pos .+ model.action_dict[a_t1]
            
            # update episode reward
            model.DQN_params.ep_rew += r_t
        end
    end

end

function get_state(model, agent_id, i)
   
    # store previous position
    s_t1 = model.agents[agent_id].s_t

    # compute relative distances
    s_t = []
    push!(s_t, model.agents[agent_id].pos)

    for j in keys(sort(collect(pairs(model.Goals))))
        # figure out relative distances to goals
        push!(s_t, model.Goals[j] .- model.agents[agent_id].pos)
    end

    # NOTE: just keep pushing into s_t for state


    # finally, flatten into a vector and return the state
    s_t = collect(Iterators.flatten(s_t))

    # save to FMP agent
    model.agents[agent_id].s_t = s_t
    model.agents[agent_id].s_t1 = s_t1

    # return 
    return s_t1, s_t
end

function GlobalStateTransition!(model)
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

            if length(model.agents[agent_id].Gi) > 0
                model.agents[agent_id].color = "#3CB371"
            else
                model.agents[agent_id].color = "#FF0000"
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
                #rewards[i] += 1*beta
            end

            # get reward for goal occupation
            rewards[i] += sum(model.SS.GO[i,:])*1
            #rewards[i] += sum(model.SS.GO[i,:])*1*alpha

            # agents pay penalty for goals they don't know location of
            #rewards[i] += -0.001*sum(1 .- model.SS.GA[i, :])  #BONE
            rewards[i] += -0.01*sum(1 .- model.SS.GO[i, :])

        end
    end
    return rewards
end

