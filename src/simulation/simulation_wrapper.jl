export model_run

mutable struct SimulationParams
    num_agents::Int64
    num_goals::Int64
    num_steps::Int64
    num_episodes::Int64
    sim_vid_interval::Int64
    sim_type::String
    rl_type::String
    episode_number::Int64
end

function model_run(;num_agents=20,
                    num_goals = 20,
                    num_steps = 5000,
                    num_episodes = 10000,
                    sim_vid_interval = 100,
                    sim_type = "lost_hiker",
                    rl_type = "A3C",
                  )

    # setup simulation parameters
    sim_params = SimulationParams(num_agents,
                                  num_goals,
                                  num_steps,
                                  num_episodes,
                                  sim_vid_interval,
                                  sim_type,
                                  rl_type,
                                  1
                                 )

    # setup misc params
    #BLAS.set_num_threads(1)

    # specify global MDP formulation (assume 2D)
    #   State = agent position tuple, goal position tuples, GoalAwareness
    #   Actions = {up, down, left, right, none}
    if rl_type == "A3C"
        rl_arch = a3c_struct_init(sim_params)
    else
        @error "RL type unknown"
    end

    # setup prelim stuff for data recording
    csv_write_path = "/Users/riomcmahon/Programming/antfarm/src/data_output/run_history.csv"
    model_write_path = "/Users/riomcmahon/Programming/antfarm/src/data_output/model_weights/"
    reward_write_path = "/Users/riomcmahon/Programming/antfarm/src/data_output/reward_hist.bson"
    reward_hist = zeros(sim_params.num_episodes)
    time_hist = zeros(sim_params.num_episodes)
    
    # train model
    for episode in 1:num_episodes
        println("\nEpoch #$episode of $num_episodes")

        if episode % sim_vid_interval == 0
            reward_hist[episode] = episode_run(rl_arch, sim_params, plot_sim=true)

        else

            start_time = time()
            reward_hist[episode] = episode_run(rl_arch, sim_params)
            # BONE - need to figure out how to save policy

            end_time = time()
            time_hist[episode] = end_time-start_time
            bson(reward_write_path, Dict(:Rewards=>reward_hist,
                                         :TimeHist=>time_hist))
            println("Global Reward for Epoch = $(reward_hist[episode])")
            println("Time Elapsed for Epoch = ", end_time-start_time)


            # save weights
            bson(string(model_write_path, "theta_episode$episode.bson"), Dict(:Policy => rl_arch.params))
        end
    end
end

# Now that we've defined the plot utilities, lets re-run our simulation with
# some additional options. We do this by redefining the model, re-adding the
# agents but this time with a color parameter that is actually used. 
function episode_run(rl_arch, sim_params; plot_sim=false)

    # define model
    model = fmp_model_init(rl_arch, sim_params; num_agents=sim_params.num_agents, 
                               num_goals=sim_params.num_goals, 
                               num_steps=sim_params.num_steps)
    
    # run simulation
    if plot_sim == true
        @info "plotting simulation"
        filepath = "/Users/riomcmahon/Desktop/episode_$(sim_params.episode_number).mp4"
        RunModelPlot(model, agent_step!, model_step!, filepath)  # plot sim_vid
        ContinuousPlotCurrentReward(sim_params)  # plot losses
    else

        # run simulation, update model, and update global policy
        RunModelCollect(model, agent_step!, model_step!)
        model.RL.policy_train(model)
        rl_arch.params.θ = model.RL.params.θ

    end

    # update sim params
    sim_params.episode_number += 1

    return sum(model.RL.params.r_sa)
end 

