function RewardPlot(model)
    cumulative_reward = zeros(Float64, 1, model.num_steps)
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            cumulative_reward += model.A3C[i].RewardHist
        end
    end
    plot(1:model.num_steps, reshape(cumulative_reward, (length(cumulative_reward),)))
end
