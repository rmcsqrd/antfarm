function RewardPlot(df)
    agent_info_df = df[ [x==:A for x in df.type], :]
    num_steps = last(df).step
   
    cumulative_reward = zeros(num_steps)

    for step_i in 1:num_steps
        cumulative_reward[step_i] = sum(df[ [x==step_i for x in df.step], :].Reward)
    end
    plot(1:num_steps, cumulative_reward)
end

function PlotCurrentReward()
                  r = BSON.load("/Users/riomcmahon/Programming/antfarm/src/data_output/reward_hist.bson")
                  rew = r[:Rewards]
                  rew_parse = [x for x in rew if x != 0.0]
                  plt = Plots.scatter(1:length(rew_parse), rew_parse)
                  ylabel!("Reward")
                  title!("Epoch Rewards")
                  xlabel!("Epoch Number")
                  display(plt)
             end

function PlotCurrentRewardWindow(window)
                  r = BSON.load("/Users/riomcmahon/Programming/antfarm/src/data_output/reward_hist.bson")
                  rew = r[:Rewards]
                  rew_parse = [x for x in rew if x != 0.0]
                  plt = Plots.scatter(1:length(rew_parse), rew_parse)
                  ylabel!("Reward")
                  title!("Epoch Rewards")
                  xlabel!("Epoch Number")
                  mean_stuff = zeros(length(rew_parse))
                  for i in window+1:length(rew_parse)
                      mean = sum(rew_parse[i-window])/window
                      mean_stuff[i] = mean
                  end
                  display(plt)
                 plot!(twinx(), 1:length(rew_parse), mean_stuff, ylabel="Rolling Average Reward, window=$window")
             end

