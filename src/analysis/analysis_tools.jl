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
    if isodd(n)
        n -= 1  # window midpoints don't like odd numbers
    end
    r = BSON.load(string(homedir(),"/Programming/antfarm/src/data_output/_reward_hist.bson"))
    rew = r[:Rewards]
    rew = rew[rew .!= 0.0]  # maybe a little fast and loose but chance of exactly 0.0 reward is low
    loss = r[:Loss]
    loss = loss[loss .!= 0.0]

    μ = zeros(length(rew))  # this makes indexing easier, we'll lop off zero values later
    σ = zeros(length(μ))
    μ_loss = zeros(length(loss))
    σ_loss = zeros(length(μ_loss))
    for k in n+1:length(rew)
        μ[k] = sum(rew[k-n:k])/length(k-n:k)
        σ[k] = sum((rew[k-n:k] .- μ[k]) .^ 2)/length(k-n:k)
        μ_loss[k] = sum(loss[k-n:k])/length(k-n:k)
        σ_loss[k] = sum((loss[k-n:k] .- μ_loss[k]) .^ 2)/length(k-n:k)
    end

    mid_point = round(Int64, (n+1)/2)
    μ = μ[n+1:length(μ)]
    σ = σ[n+1:length(σ)]
    μ_loss = μ_loss[n+1:length(μ_loss)]
    σ_loss = σ_loss[n+1:length(σ_loss)]
    σ_scale = 0.001
    σ_loss = σ_scale .* σ_loss
    σ = σ_scale .* σ

    # plot reward stuff
    p1 = Plots.scatter(1:length(rew), rew,
                        label="Reward @ Epoch",
                        legend=:topleft,
                        markerstrokewidth=0,
                        markersize=3,
                        markercolor="#60d9cb",
                        ylabel="-1*Reward",
                       )
    Plots.plot!(p1, mid_point+1:length(rew)-mid_point, μ,
                label="\"Sliding Window\" Average Reward (n = $n)",
                color="#d99960",
                linewidth=3,
               )
    Plots.plot!(p1, mid_point+1:length(rew)-mid_point, μ+ 2 .* σ,
                label="+/- $σ_scale*2σ Variance",
                color="#cb60d9")
    Plots.plot!(p1, mid_point+1:length(rew)-mid_point, μ+ 2 .* σ,
                label="",
                color="#cb60d9")

    # plot loss stuff
    title!("Reward Over Time")
    p2 = Plots.scatter(1:length(loss), loss,
                        label="Loss @ Epoch",
                        legend=:topleft,
                        markerstrokewidth=0,
                        markersize=3,
                        markercolor="#60d9cb",
                        right_margin = 3*Plots.mm,
                        ylabel="Training Loss",
                   )
    Plots.plot!(p2, mid_point+1:length(loss)-mid_point, μ_loss,
                    color="#d99960",
                    linewidth=3,
                    label="\"Sliding Window\" Average Loss (n = $n)",
               )
    Plots.plot!(p2, mid_point+1:length(loss)-mid_point, μ_loss+ 2 .* σ_loss,
                label="+/- $σ_scale*2σ Variance",
                color="#cb60d9")
    Plots.plot!(p2,mid_point+1:length(loss)-mid_point, μ_loss- 2 .* σ_loss,
                label="",
                color="#cb60d9")
    title!("Training Loss Over Time")
    xlabel!("Epoch")
    plot(p1,p2, layout=(2,1), size=(1000,1000))
    savefig(string(homedir(), "/Programming/antfarm/src/data_output/_reward.png"))

end
