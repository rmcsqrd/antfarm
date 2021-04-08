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
    s_t1
    a_t1
    s_t
end

function fmp_model_init(dqn_params, dqn_network, sim_params)

    # first define model properties/space/etc for ABM
    extents = (1,1)
    action_dict = Dict()
    if sim_params.num_dimensions == "1D"
        @info "1D Selected"
        action_dict[1] = (-0.1,0)  # left
        action_dict[2] = (0.1,0)   # right
        action_dict[3] = (0,0)   # no action
    elseif sim_params.num_dimensions == "2D"
        @info "2D Selected"
        action_dict[1] = (-0.1,0)  # left
        action_dict[2] = (0.1,0)   # right
        action_dict[3] = (0,0)   # no action
        action_dict[4] = (0,0.1)   # up
        action_dict[5] = (0,-0.1)  # down
    else
        @error "Incorrect number of Dimensions"
    end
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
                      :action_dict=>action_dict,
                      :Agents2RL=>Dict{Int64, Int64}(),  # dict to map Agents.jl agent_ids to RL formulation id values
                      :Goals=>Dict{Int64, Tuple{Float64, Float64}}(),  # dict to map RL formulation goal id's to position of Agents.jl goal (agent.type == :T)
                      :t=>1,  # current step
                      :sim_params=>sim_params,
                      :DQN=>dqn_network,
                      :DQN_params=>dqn_params
                     )

    space2d = ContinuousSpace(extents; periodic = false)
    model = ABM(FMP_Agent, space2d, properties=properties)

    # add in agents
    fmp_model_add_agents!(model)
    
    return model
end

function fmp_model_add_agents!(model)

    # initialize model by adding in agents
    if model.sim_params.sim_type == "lost_hiker"
        LostHiker(model)
    elseif model.sim_params.sim_type == "simple_test"
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

        # first, assign MDP id to agent and initial stat
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
end

function fmp_model_reset!(model)

    # remove all agents, reset step counter, clear out goal/agent dicts
    genocide!(model)
    model.Agents2RL = Dict{Int64, Int64}()
    model.Goals = Dict{Int64, Tuple{Float64, Float64}}()
    model.t = 1
    model.SS.GO = zeros(Bool, model.num_agents, model.num_goals)
    model.SS.GA = zeros(Bool, model.num_agents, model.num_goals)

    # finally, re-add in agents
    fmp_model_add_agents!(model)

end

# define agent/model step stuff
function agent_step!(agent, model)

    # check model extents to respect boundary
    px, py = agent.pos .+ model.step_inc*model.dt .* agent.vel
    ex, ey = model.space.extent
    if !(0 ≤ px ≤ ex && 0 ≤ py ≤ ey)
        agent.vel = (0.0, 0.0)
    end
    move_agent!(agent, model, model.dt)
end

function model_step!(model)
    # do FMP stuff - figure out interacting pairs and update velocities
    # accordingly

    # do RL stuff 
    RL_Update!(model)

    fmp_update_interacting_pairs(model)
    for agent_id in keys(model.agents)
        fmp_update_vel(model.agents[agent_id], model)

        # update agent states
        model.agents[agent_id].s_t1 = model.agents[agent_id].s_t
        model.agents[agent_id].s_t = model.agents[agent_id].pos

    end

    # train model if required
    if model.DQN_params.K % model.t == 0
        DQN_train!(model)
    end

    # step forward
    model.t += 1
    
end
