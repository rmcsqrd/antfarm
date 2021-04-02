#######################################################################################
# GLOBAL RL STUFF
#######################################################################################

mutable struct RL_Wrapper
    params  # container for specific RL architecture params
    policy_train  # function for training the RL architecture
    policy_evaluate  # function for evaluating the model state, returns an action
    episode_init  # function for initializing RL struct at beginning of episode
    γ::Float64  # discount factor
end

function RL_Update(model)
    # update global state (goal awareness/goal occupation)
    GlobalStateTransition(model)
    
    # compute rewards
    rewards = GlobalReward(model)

    # next, do individual agent actions
    goal_loc_array = sort(collect(values(model.Goals)))
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.Agents2RL[agent_id]

            # first, determine agent i's knowledge of goal positions
            GAi = model.SS.GA[i, :]

            # next, compare GA with goal locations. Return true location if
            # agent i is aware of goal location, (Inf, Inf) if not
            goal_pos_i = [xi == 1 ? yi : (-1,-1) for xi in GAi, yi in goal_loc_array[1,:]]  
            

            # vectorize to create state. Need to use iterators because
            # vec(Tuple) doesn't work
            s_t = [collect(Iterators.flatten(model.agents[agent_id].pos));
                   collect(Iterators.flatten(goal_pos_i));
                   vec(GAi)
                  ]
            
            # select action according to RL policy
            t = model.ModelStep
            r_t = rewards[i]
            a_t = model.RL.policy_evaluate(i, t, s_t, r_t, model)

            # update model
            model.agents[agent_id].tau = model.agents[agent_id].pos .+ model.action_dict[a_t]

        end
    end

end

#######################################################################################
# A3C STUFF
#######################################################################################

function a3c_struct_init(sim_params)
    
    # define dimensions
    # state_dim = agent_position (x,y): this is two coord
    #             goal_1 (x,y): this is two coords
    #                .
    #                .
    #                .
    #             goal_g (x,y)
    #             GA(i,1): this is a scalar
    #                .
    #                .
    #                .
    #             GA(i,g)
    state_dim = 2+sim_params.num_goals*2 + sim_params.num_goals
    action_dim = 5

    if sim_params.prev_run == "none"
        model = Chain(
                      Dense(state_dim, 128, relu),
                      Dense(128, action_dim+1)
                     )
        θ = params(model)
    else
        # load in previous model
        prev_model = BSON.load(sim_params.prev_run, @__MODULE__)
        model = prev_model[:Policy].model
        θ = prev_model[:Policy].θ
    end

    γ = 0.99
    η = 0.00005
    β = 0.05
    r_matrix = zeros(Float32, sim_params.num_agents, sim_params.num_steps)
    s_matrix = zeros(Float32, sim_params.num_agents, state_dim, sim_params.num_steps)
    action_matrix = zeros(Float32, sim_params.num_agents, action_dim, sim_params.num_steps)
    A3C_params = A3C_Global(model, θ, η, β, r_matrix, s_matrix, action_matrix)

    return RL_Wrapper(A3C_params, A3C_policy_train, A3C_policy_eval, A3C_episode_init, γ)

end

