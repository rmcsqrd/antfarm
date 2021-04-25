function RL_Update!(model)

    # next, do individual agent actions
    for agent_id in keys(model.agents)

        if model.agents[agent_id].type == :A
            i = model.Agents2RL[agent_id]

            # get model state from agent and  push transition into replay buffer
            a_t1 = model.agents[agent_id].a_t1
            s_t1, s_t = get_state(model, agent_id, i)
            r_t = get_reward(model, agent_id, i, s_t)

            DQN_buffer_update!(s_t1, a_t1, r_t, s_t, model)

            # select action according to RL policy
            a_t = DQN_policy_eval!(s_t, model, agent_id)
            model.agents[agent_id].a_t1 = a_t

            # update agent with action
            model.agents[agent_id].tau = model.agents[agent_id].pos .+ model.action_dict[a_t]
            
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

    # give agent position of assigned goal
    goals = values(sort(collect(pairs(model.Goals))))
    push!(s_t, model.Goals[i] .- model.agents[agent_id].pos)

    # store obstacle relative distance
    agent_keys = keys(model.agents)
    sorted_keys = sort(collect(agent_keys))
    for other_agent_id in sorted_keys
        if model.agents[other_agent_id].type == :O
            push!(s_t, model.agents[other_agent_id].pos .- model.agents[agent_id].pos)
        end
    end

    # finally, flatten into a vector and return the state
    s_t = collect(Iterators.flatten(s_t))

    # save to FMP agent
    model.agents[agent_id].s_t = s_t
    model.agents[agent_id].s_t1 = s_t1

    # return 
    return s_t1, s_t
end

function get_reward(model, agent_id, i, s_t)

    rewards = 0

    # agent-goal interaction
    model.agents[agent_id].color = "#FF0000"

    # give proportional reward for goal proximity
    # first two state positions are relative distances to goal
    x, y = model.space.extent
    max_dist = √(x^2+y^2)
    ag_offset_x = s_t[1]
    ag_offset_y = s_t[2]
    agent_goal_dist = √(ag_offset_x^2+ag_offset_y^2)
    offset_ratio = agent_goal_dist/max_dist
    rewards += 0.01*(1-offset_ratio)
    

    # give large reward for goal occupation
    if !isempty(model.agents[agent_id].Gi)
        for goal_id in model.agents[agent_id].Gi
            rewards += 1
        end
        model.agents[agent_id].color = "#3CB371"
    end

    # agent/obstacle bumps
    for obstacle_id in model.agents[agent_id].Oi
        model.agents[agent_id].color = "#0000FF"
        rewards -= 0.01
    end
    return rewards
end

