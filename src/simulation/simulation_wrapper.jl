export model_run

mutable struct SimulationParams
    num_agents::Int64          # number of agents in simulation
    num_goals::Int64           # number of goals
    num_steps::Int64           # number of steps per episode
    num_episodes::Int64        # number of episodes per simulation
    sim_vid_interval::Int64    # interval step to create output mp4
    sim_type::String           # simulation type for agent IC
    rl_type::String            # RL algorithm to train with 
    episode_number::Int64      # current episode/epoch number
    prev_run::String           # previous run indicator string (load in previous?)
    num_dimensions::String     # 1D vs 2D
end

function model_run(;num_agents=20,
                    num_goals = 20,
                    num_steps = 5000,     # number of steps per episode (epoch)
                    num_episodes = 10000, # number of episodes (epochs)
                    sim_vid_interval = 100,
                    sim_type = "lost_hiker",
                    rl_type = "DQN",
                    prev_run="none",
                    num_dims="1D",
                  )

    # setup simulation parameters
    if prev_run != "none"
        @info "Loading previous model..."
        prev_model = BSON.load(prev_run, @__MODULE__)
        sim_params = prev_model[:sim_params]
        sim_params.prev_run = prev_run
        reward_hist = prev_model[:reward_hist]
        loss_hist = prev_model[:loss_hist]
    else
        @info "No previous model specified, starting from scratch..."
        sim_params = SimulationParams(num_agents,
                                  num_goals,
                                  num_steps,
                                  num_episodes,
                                  sim_vid_interval,
                                  sim_type,
                                  rl_type,
                                  1,
                                  prev_run,
                                  num_dims,
                                 )
        reward_hist = zeros(sim_params.num_episodes)
        loss_hist = zeros(sim_params.num_episodes)
    end

    # specify global MDP formulation
    if rl_type == "A3C"
        @info "A3C Selected"
        rl_arch = a3c_struct_init(sim_params)
    elseif rl_type == "DQN"
        @info "DQN Selected"
        rl_arch = dqn_struct_init(sim_params)
    else
        @error "RL type unknown"
    end

    # setup prelim stuff for data recording
    model_write_path = string(homedir(),"/Programming/antfarm/src/data_output/_model_weights/")
    if !isdir(model_write_path)
        mkdir(model_write_path)  # it gets deleted when I archive the data
    end
    
    # define model once to preserve RL stuff between episodes
    model = fmp_model_init(rl_arch, sim_params)

    # train model
    for episode in 1:sim_params.num_episodes  # BONE, find clever solution to this

        # run episode
        run_time = @elapsed reward_hist[episode], loss_hist[episode] = episode_run!(model)

        # write output to console
        println("\nEpoch #$episode of $num_episodes")
        println("Global Reward for Epoch = $(reward_hist[episode])")
        println("Time Elapsed for Epoch = $run_time")

        # save weights
        bson(string(model_write_path, "_theta_episode$episode.bson"), 
             Dict(:Policy => model.RL.params,
                  :sim_params => sim_params,
                  :reward_hist => reward_hist,
                  :loss_hist => loss_hist,
                 )
            )
        # reset agent based model
        fmp_model_reset!(model)

        # reset RL parameters for next episode
        model.RL.episode_init!(model)

        # update episode number
        model.sim_params.episode_number = episode
    end
end

function episode_run!(model)

    # record simulation or just run normally
    if model.sim_params.episode_number % model.sim_params.sim_vid_interval == 0
        @info "plotting simulation"
        plot_reward_window(model.sim_params)  # plot losses/rewards
        run_model_plot!(model, agent_step!, model_step!, model.sim_params)  # plot sim_vid

    else
        run_model!(model, agent_step!, model_step!)
    end

    # train model
    training_loss = model.RL.policy_train!(model)

    # return losses
    if model.sim_params.rl_type == "A3C"
        return sum(model.RL.params.r_t), training_loss
    elseif model.sim_params.rl_type == "DQN"
        return sum(model.RL.params.r_t), training_loss
    end

end 

