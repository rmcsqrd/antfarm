function StateSpaceHashing(model)

    # create associations between agent ID's and RL matrices
    #   Note that Agents.jl stores goal positions and objects as agents with
    #   unique IDs but this gives issues when trying to index into the RL state
    #   space arrays. We solve this by hashing
    
    # start by getting list of agent ids that have symbol :A
    agent_list = [hash(agent_id) for agent_id in keys(model.agents) if model.agents[agent_id].type == :A]
    model.AgentHash = Dict(zip(agent_list, 1:length(agent_list)))
    
    # next get list of ids that have symbol :T
    # create ways to "hash into" state space array from ContinuousSpace model, and "hash out of" state
    # space array into ContinuousSpace model so we can update agent target
    # positions
    goal_list = [goal_id for goal_id in keys(model.agents) if model.agents[goal_id].type == :T]
    for (idx, goal_id) in enumerate(goal_list)

        # Agents.jl id -> RL id is hashed
        model.GoalHash[hash(goal_id)] = idx

        # RL id -> Agents.jl id is not hashed (to avoid collisions)
        model.GoalHash[idx] = goal_id 
    end

end
