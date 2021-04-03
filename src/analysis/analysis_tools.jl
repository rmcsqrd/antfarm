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
    loss = r[:Loss]
    loss = loss[loss .!= 0.0]

    μ = zeros(length(rew))  # this makes indexing easier, we'll lop off zero values later
    σ = zeros(length(μ))
    μ_loss = zeros(length(rew))
    for k in n+1:length(rew)
        μ[k] = sum(rew[k-n:k])/(n+1)
        σ[k] = sum((rew[k-n:k] .- μ[k]) .^ 2)/(n+1)
        μ_loss[k] = sum(loss[k-n:k])/(n+1)
    end

    μ = μ[n+1:length(μ)]
    σ = σ[n+1:length(σ)]
    μ_loss = μ_loss[n+1:length(μ_loss)]
    #σ_scale = 0.001
    #σ = σ_scale .* σ

    # plot reward stuff
    plt = Plots.scatter(1:length(rew), rew,
                        label="Reward @ Epoch",
                        legend=:topleft,
                        markerstrokewidth=0,
                        markersize=3,
                        markercolor="#60d9cb"
                       )
    Plots.scatter!([1],[rew[1]],
                label="Loss @ Epoch",
                markerstrokewidth=0,
                markersize=3,
                markercolor="#d98e60"
               )
    Plots.plot!(n+1:length(rew), μ,
                label="\"Sliding Window\" Average Reward (n = $n)",
                color="#d9cb60"
               )
    # plot dummy data to get same legend
    Plots.plot!([1],[NaN*1],
                label="\"Sliding Window\" Average Loss (n = $n)",
                color="#cb60d9",
               )
    xlabel!("Epoch")
    ylabel!("Reward (multiplied by -1)")
    p = twinx()
    Plots.scatter!(p, 1:length(loss), loss,
                        label="Loss @ Epoch",
                        legend=false,
                        markerstrokewidth=0,
                        markersize=3,
                        markercolor="#d98e60",
                        right_margin = 3*Plots.mm,
                        ylabel="Cumulative Training Loss",
                   )
    Plots.plot!(p, n+1:length(loss), μ_loss,
                    color="#cb60d9"
               )
    #Plots.plot!(n+1:length(rew), μ+ 2 .* σ,
    #            label="+/- $σ_scale*2σ Variance",
    #            color="#cb60d9")
    #Plots.plot!(n+1:length(rew), μ- 2 .* σ,
    #            label="",
    #            color="#cb60d9")
    title!("Training Loss Over Time")
    savefig(string(homedir(), "/Programming/antfarm/src/data_output/_reward.png"))

end
