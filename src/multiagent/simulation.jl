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
    Pip  # share actor/critic params in network
end


## define AgentBasedModel (ABM)

function FMP_Model(; num_agents=20, num_goals=num_agents, num_steps=1500)
    properties = Dict(:FMP_params=>FMP_Parameter_Init(),
                      :dt => 0.01,
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
    num_steps = 1500
    num_episodes = 10

    # initialize stuff
    state_dim = 3*num_goals+num_agents
    A3C_params = A3C_Global(num_agents, 
                            num_goals, 
                            num_steps, 
                            num_episodes,
                            1,
                            A3C_Policy_Init(state_dim, num_goals),
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
    write_path = "/Users/riomcmahon/Programming/antfarm/src/data_output/run_history.csv"
    try
        rm(write_path)
        println("run_history.csv overwritten")
    catch
        println("run_history.csv doesn't exist, moving on")
    end
    @showprogress for episode in 1:num_episodes
        agent_data = FMP_Episode(A3C_params)
        if episode == 1
            CSV.write(write_path, agent_data)  # remove append option to preserve header
        else
            CSV.write(write_path, agent_data, append=true)
        end
        append!(run_history, agent_data)
        #display(agent_data)
        # train policy
        # train value function
        A3C_params.episode_number += 1
    end
end

# Now that we've defined the plot utilities, lets re-run our simulation with
# some additional options. We do this by redefining the model, re-adding the
# agents but this time with a color parameter that is actually used. 
function FMP_Episode(A3C_params)

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
        # do FMP stuff - figure out interacting pairs and update velocities
        # accordingly
        FMP_Update_Interacting_Pairs(model)
        for agent_id in keys(model.agents)
            FMP_Update_Vel(model.agents[agent_id], model)
        end

        # do RL stuff 
        StateTransition(model)
        Reward(model)
        Action(model)  # if you comment this out it behaves as vanilla FMP
        model.ModelStep += 1

    end
    raw_data = RunModelCollect(model, agent_step!, model_step!)
    agent_data = raw_data[ [x==:A for x in raw_data.type], :]
    insertcols!(agent_data, 1, :episode_num=>[A3C_params.episode_number for x in 1:nrow(agent_data)])

    if A3C_params.episode_number == A3C_params.num_episodes
        RunModelPlot(model, agent_step!, model_step!)
    end

    return agent_data
end 

