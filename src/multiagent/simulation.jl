export FMP_Epoch

mutable struct FMP_Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    tau::NTuple{2, Float64}
    color::String
    type::Symbol
    radius::Float64
    SSdims::NTuple{2, Float64}  ## include this for plotting
    Ni::Vector{Int64} ## array of neighboring agent IDs
    Gi::Vector{Int64} ## array of neighboring goal IDs
    Action::Int64
    PiAction::Float64
    Value::Float64
    Reward::Float64
    State::Vector{Bool}
    Model  # this is the model used to evaluate actions
end


## define AgentBasedModel (ABM)

function FMP_Model(; num_agents=20, num_goals=num_agents, num_steps=1500)
    properties = Dict(:FMP_params=>FMP_Parameter_Init(),
                      :dt => 0.05,
                      :num_agents=>num_agents,
                      :num_goals=>num_goals,
                      :num_steps=>num_steps,
                      :step_inc=>2,
                      :SS=>StateSpace(zeros(Bool, num_agents, num_goals),  # GA
                             zeros(Bool, num_agents, num_goals),  # GO
                             zeros(Bool, num_agents, num_goals),  # GI
                             zeros(Bool, num_agents, num_agents), # AI
                            ),
                      :AgentHash=>Dict{Int128, Int128}(),
                      :GoalHash=>Dict{Int128, Int128}(),
                      :Goals=>Dict{Int64, Tuple{Float64, Float64}}(),
                      :ModelStep=>1,
                     )

    space2d = ContinuousSpace((1,1); periodic = true)
    model = ABM(FMP_Agent, space2d, properties=properties)
    return model
end

function FMP_Epoch()

    # set global hyperparams
    num_agents = 20
    num_goals = 20
    num_steps = 10000
    num_episodes = 10000
    discount_factor = 0.95

    # initialize stuff
    state_dim = 3*num_goals+num_agents
    model = Chain(
                  Dense(state_dim, 128, tanh),
                  LSTM(128, num_goals+2) # num goals, random action, V(si)
                 )
    θ = params(model)
    A3C_params = A3C_Global(num_agents, 
                            num_goals, 
                            num_steps, 
                            num_episodes,
                            1,
                            model,
                            θ,
                            discount_factor,
                           )
    # train model
    run_history = DataFrame(episode_num = Int64[],
                            step = Int64[],
                            id = Int64[],
                            type = Symbol[],
                            State = Array[],
                            Action = Int64[],
                            PiAction = Float64[],
                            Value = Float64[],
                            Reward = Float64[],
                           )
    csv_write_path = "/Users/riomcmahon/Programming/antfarm/src/data_output/run_history.csv"
    model_write_path = "/Users/riomcmahon/Programming/antfarm/src/data_output/model_weights/"
    reward_write_path = "/Users/riomcmahon/Programming/antfarm/src/data_output/reward_hist.bson"
    try
        rm(csv_write_path)
        println("run_history.csv overwritten")
    catch
        println("run_history.csv doesn't exist, moving on")
    end
    reward_hist = zeros(num_episodes)
    for episode in 1:num_episodes
        println("\nEpoch #$episode of $num_episodes")

        if episode % 100 == 0
            FMP_Episode(A3C_params, plot_sim=true)

        else

            agent_data = FMP_Episode(A3C_params)
            if episode == 1
                CSV.write(csv_write_path, agent_data)  # remove append option to preserve header
            elseif episode % 20 == 0 # only record every hundred steps
                CSV.write(csv_write_path, agent_data, append=true)
            end

            # train policy, collect reward, save
            epoch_reward = PolicyTrain(agent_data, A3C_params)
            reward_hist[episode] = epoch_reward
            bson(reward_write_path, Dict(:Rewards=>reward_hist))
            println("Global Reward for Epoch = $epoch_reward")


            # save weights
            epnum = A3C_params.episode_number
            bson(string(model_write_path, "weights_epnum$epnum.bson"), Dict(:Policy => A3C_params.model))
        end
        # step model forward
        A3C_params.episode_number += 1
    end
end

# Now that we've defined the plot utilities, lets re-run our simulation with
# some additional options. We do this by redefining the model, re-adding the
# agents but this time with a color parameter that is actually used. 
function FMP_Episode(A3C_params; plot_sim=false)

    # define model
    model = FMP_Model(; num_agents=A3C_params.num_agents, 
                        num_goals=A3C_params.num_goals, 
                        num_steps=A3C_params.num_steps)
    
    # initialize model by adding in agents
    LostHiker(model)

    # create agent/goal hashes for RL stuff
    StateSpaceHashing(model)
    
    # initialize the A3C struct
    A3C_Episode_Init(model, A3C_params)

    # define agent/model step stuff
    function agent_step!(agent, model)
        move_agent!(agent, model, model.dt)
    end

    function model_step!(model)
        # do RL stuff 
        StateTransition(model)
        Reward(model)
        Action(model)  # if you comment this out it behaves as vanilla FMP

        # do FMP stuff - figure out interacting pairs and update velocities
        # accordingly
        FMP_Update_Interacting_Pairs(model)
        for agent_id in keys(model.agents)
            FMP_Update_Vel(model.agents[agent_id], model)
        end
        
        # step model
        model.ModelStep += 1

    end

    if plot_sim == true
        @info "plotting simulation"
        ep_num = A3C_params.episode_number
        filepath = "/Users/riomcmahon/Desktop/episode_$ep_num.mp4"
        RunModelPlot(model, agent_step!, model_step!, filepath)
    else
        raw_data = RunModelCollect(model, agent_step!, model_step!)
        agent_data = raw_data[ [x==:A for x in raw_data.type], :]
        insertcols!(agent_data, 1, :episode_num=>[A3C_params.episode_number for x in 1:nrow(agent_data)])

        return agent_data
    end
end 

