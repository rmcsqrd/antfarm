export model_run

mutable struct SimulationParams
    num_agents::Int64          # number of agents in simulation
    num_goals::Int64           # number of goals
    num_steps::Int64           # number of steps per episode
    num_vid_steps::Int64
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
                    num_steps = 10000,     # number of steps per episode (epoch)
                    num_vid_steps = 1000,
                    num_episodes = 10000, # number of episodes (epochs)
                    sim_vid_interval = 100,
                    sim_type = "lost_hiker",
                    rl_type = "DQN",
                    prev_run="none",
                    num_dims="2D",
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
                                  num_vid_steps,
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
    dqn_params, dqn_network, buffer = DQN_init(sim_params)

    # setup prelim stuff for data recording
    model_write_path = string(homedir(),"/Programming/antfarm/src/data_output/_model_weights/")
    if !isdir(model_write_path)
        mkdir(model_write_path)  # it gets deleted when I archive the data
    end
    
    # define model once to preserve RL stuff between episodes
    model = fmp_model_init(dqn_params, dqn_network, buffer, sim_params)

    # train model
    for episode in 1:sim_params.num_episodes  # BONE, find clever solution to this

        # run episode
        run_time = @elapsed reward_hist[episode], loss_hist[episode] = episode_run!(model)

        # write output to console
        pr(x) = round(x, digits=5)
        println("\nEpisode #$episode of $num_episodes")
        println("Global Reward for Episode = $(pr(reward_hist[episode]))")
        println("Training Loss for Epoch   = $(pr(loss_hist[episode]))")
        println("Time Elapsed for Episode  = $(pr(run_time))")

        # save weights
        bson(string(model_write_path, "_theta_episode$episode.bson"), 
             Dict(:Policy => model.DQN,
                  #:model_params=>model.DQN_params,  # saving the replay buffer
                  #was taking a long time
                  :sim_params => sim_params,
                  :reward_hist => reward_hist,
                  :loss_hist => loss_hist,
                 )
            )
        # reset agent based model
        fmp_model_reset!(model)

        # update episode number and clear reward/losses
        model.sim_params.episode_number = episode
        model.DQN_params.ep_rew = 0
        model.DQN_params.ep_loss = 0
    end
end

function episode_run!(model)

    # record simulation or just run normally
    if model.sim_params.episode_number % model.sim_params.sim_vid_interval == 0
        @info "plotting simulation"
        plot_reward_window(model)  # plot losses/rewards
        run_model_plot!(model, agent_step!, model_step!)  # plot sim_vid
    else
        run_model!(model, agent_step!, model_step!)
    end
    return model.DQN_params.ep_rew, model.DQN_params.ep_loss
end 

