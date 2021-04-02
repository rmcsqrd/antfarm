function RewardPlot(df)
    agent_info_df = df[ [x==:A for x in df.type], :]
    num_steps = last(df).step
   
    cumulative_reward = zeros(num_steps)

    for step_i in 1:num_steps
        cumulative_reward[step_i] = sum(df[ [x==step_i for x in df.step], :].Reward)
    end
    plot(1:num_steps, cumulative_reward)
end

function PlotCurrentReward(;step=0)
    r = BSON.load(string(homedir(),"/Programming/antfarm/src/data_output/_reward_hist.bson"))
    rew = r[:Rewards]
    if step != 0
        rew_parse = rew[1:step]
    else
        rew_parse = [x for x in rew if x != 0.0]
    end
    plt = Plots.scatter(1:length(rew_parse), rew_parse)
    ylabel!("Reward")
    title!("Epoch Rewards")
    xlabel!("Epoch Number")
    display(plt)
end

function ContinuousPlotCurrentReward(sim_params)
    r = BSON.load(string(homedir(), "/Programming/antfarm/src/data_output/_reward_hist.bson"))
    rew = r[:Rewards]
    rew_parse = rew[1:sim_params.episode_number]
    plt = Plots.scatter(1:length(rew_parse), rew_parse)
    ylabel!("Reward")
    title!("Epoch Rewards")
    xlabel!("Epoch Number")
    savefig(string(homedir(), "/Programming/antfarm/src/data_output/_reward.png"))
end

function PlotRewardWindow(n=100)
    r = BSON.load(string(homedir(),"/Programming/antfarm/src/data_output/_reward_hist.bson"))
    rew = r[:Rewards]
    rew = rew[rew .!= 0.0]  # maybe a little fast and loose but chance of exactly 0.0 reward is low

    μ = zeros(length(rew))  # this makes indexing easier, we'll lop off zero values later
    σ = zeros(length(μ))
    for k in n+1:length(rew)
        μ[k] = sum(rew[k-n:k])/n
        σ[k] = sum((rew[k-n:k] .- μ[k]) .^ 2)/n
    end

    μ = μ[n+1:length(μ)]
    σ = σ[n+1:length(σ)]
    σ_scale = 0.001
    σ = σ_scale .* σ

    plt = Plots.scatter(1:length(rew), rew,
                        label="Loss @ Epoch",
                        legend=:bottomleft,
                        markerstrokewidth=0,
                        markersize=1.5,
                        markercolor="#60d9cb"
                       )
    Plots.plot!(n+1:length(rew), μ,
                label="\"Sliding Window\" Training Loss (n = $n)",
                color="#d9cb60"
               )
    Plots.plot!(n+1:length(rew), μ+ 2 .* σ,
                label="+/- $σ_scale*2σ Variance",
                color="#cb60d9")
    Plots.plot!(n+1:length(rew), μ- 2 .* σ,
                label="",
                color="#cb60d9")
    xlabel!("Epoch")
    ylabel!("Training Loss")
    title!("Training Loss Over Time")
    savefig(string(homedir(), "/Programming/antfarm/src/data_output/_reward.png"))

end
