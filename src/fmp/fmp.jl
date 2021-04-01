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
end

function fmp_model_init(rl_arch, sim_params)

    # first define model properties/space/etc for ABM
    extents = (1,1)
    properties = Dict(:FMP_params=>fmp_parameter_init(),
                      :dt => 0.05,
                      :num_agents=>sim_params.num_agents,
                      :num_goals=>sim_params.num_goals,
                      :num_steps=>sim_params.num_steps,
                      :step_inc=>2,
                      :SS=>StateSpace(  #GA, GO
                           zeros(Bool, sim_params.num_agents, sim_params.num_goals),
                           zeros(Bool, sim_params.num_agents, sim_params.num_goals),
                                     ),
                      :action_dict=>Dict(1=>(0,1),  # up
                                         2=>(0,-1), # down
                                         3=>(-1,0), # left
                                         4=>(1,0),  # right
                                         5=>(0,0)   # no action
                                        ),
                      :Agents2RL=>Dict{Int64, Int64}(),  # dict to map Agents.jl agent_ids to RL formulation id values
                      :Goals=>Dict{Int64, Tuple{Float64, Float64}}(),  # dict to map RL formulation goal id's to position of Agents.jl goal (agent.type == :T)
                      :ModelStep=>1,
                      :RL=>rl_arch,
                     )

    space2d = ContinuousSpace(extents; periodic = false)
    model = ABM(FMP_Agent, space2d, properties=properties)

    # next, initialize RL for the episode
    model.RL.episode_init(model)
    
    # next, initialize model by adding in agents
    if sim_params.sim_type == "lost_hiker"
        LostHiker(model)
    elseif sim_params.sim_type == "simple_test"
        SimpleTest(model)
    else
        @error "Simulation type not defined"
    end

    #  finally, form a relationship from the Agents.jl agent_id
    #  to the RL agent_id in the form of a dictionary
    #  note that goals and agents have distinct id's in Agents.jl
    #  but not in the RL simulation (the keys of the dict are distinct)

    goal_idx = 1
    agent_idx = 1
    for agent_id in keys(model.agents)

        # first, assign policy to agents
        if model.agents[agent_id].type == :A
            model.Agents2RL[agent_id] = agent_idx
            agent_idx += 1

        # create dict of goals. key = RL index (1:num_goals; NOT
        # Agents.jl agent.id), value = Agents.jl agent.pos
        elseif model.agents[agent_id].type == :T
            model.Goals[goal_idx] = model.agents[agent_id].pos
            model.Agents2RL[agent_id] = goal_idx
            goal_idx += 1
        end
    end

    return model
end

# define agent/model step stuff
function agent_step!(agent, model)
    # check model extents to respect boundary
    px, py = agent.pos .+ model.dt .* agent.vel
    ex, ey = model.space.extent
    if !(0 ≤ px ≤ ex && 0 ≤ py ≤ ey)
        agent.vel = (0.0, 0.0)
    end
    move_agent!(agent, model, model.dt)
end

function model_step!(model)

    # do FMP stuff - figure out interacting pairs and update velocities
    # accordingly
    fmp_update_interacting_pairs(model)
    for agent_id in keys(model.agents)
        fmp_update_vel(model.agents[agent_id], model)
    end

    # do RL stuff 
    RL_Update(model)
    
    # step model
    model.ModelStep += 1

end
