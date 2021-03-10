function RewardPlot(df)
    agent_info_df = df[ [x==:A for x in df.type], :]
    num_steps = last(df).step
   
    cumulative_reward = zeros(num_steps)

    for step_i in 1:num_steps
        cumulative_reward[step_i] = sum(df[ [x==step_i for x in df.step], :].Reward)
    end
    plot(1:num_steps, cumulative_reward)
end

function ProcessData(df, model)

    # isolate out agents by type
    agent_df = df[ [x==:A for x in df.type], :]

    # sort agents by agent.id into a dictionary of dataframes
    agent_df_dict = Dict{Int64, DataFrame}()
    agent_ids = agent_df.id[1:model.num_agents]
    for i in agent_ids
        agent_df_dict[i] = agent_df[ [x==i for x in agent_df.id], :]
    end
    return agent_df_dict
end
