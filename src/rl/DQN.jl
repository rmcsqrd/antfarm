mutable struct DQN_Global
    Q     # action-value function
    θ     # action-value params
    Q̂     # target action-value function
    θ⁻    # target action-value params
    opt   # optimiser
    ϵ     # ϵ-greedy parameter
    replay_size::Int64
    C
    r_t::Array{Float32, 2}   # r_t(i,t) = reward for agent i at step t
    s_t::Array{Float32, 3}   # s_t(i, :, t) = state for agent i at step t
    a_t::Array{Int64, 2}     # a_t(i, t) = action  for agent i at step t 
end

function DQN_policy_train!(model)
    
    training_loss = 0
    for i in 1:model.num_agents

        e = []
        for j in 1:model.num_steps-1 
            s_j = model.RL.params.s_t[i, :, j]
            s_j1 = model.RL.params.s_t[i, :, j+1]
            r_j = model.RL.params.r_t[i, j]
            a_j = model.RL.params.a_t[i, j]
            push!(e, (s_j, a_j, r_j, s_j1))
        end

        data = rand(e, model.RL.params.replay_size)

        function DQN_loss(st, at, rt, st1)
            y = rt + model.RL.γ*maximum(model.RL.params.Q̂(st1))
            return (y - model.RL.params.Q(st)[at])^2
        end

        for d in data
            dθ = gradient(model.RL.params.θ) do
                loss = DQN_loss(d...)
                training_loss += loss
                return loss
            end
            update!(model.RL.params.opt, model.RL.params.θ, dθ)
        end
        #display(model.RL.params.θ) 
    end

    # update target network with current if it is better performing
    if model.sim_params.episode_number % model.RL.params.C == 0
        @info "Setting Q̂ = Q"
        model.RL.params.Q̂ = deepcopy(model.RL.params.Q)
        model.RL.params.θ⁻ = deepcopy(model.RL.params.θ)
    end

    println("Training Loss for Epoch = $training_loss")
    #display(dθ.params)
    #display(dθ.grads)
    
    return training_loss
end

function DQN_policy_eval!(i, t, s_t, r_t, model)
    # select action via ϵ-greedy
    if rand() < model.RL.params.ϵ(model.sim_params.episode_number)
        action = rand(1:length(keys(model.action_dict)))
    else
        action = argmax(model.RL.params.Q(s_t))
    end

    # update history for training
    model.RL.params.r_t[i, t] = r_t
    model.RL.params.s_t[i, :, t] = s_t
    model.RL.params.a_t[i, t] = action

    return action
end

function DQN_episode_init!(model)
    state_dim = 2#+model.num_goals*2 + model.num_goals  #BONE
    action_dim = length(keys(model.action_dict))

    model.RL.params.r_t = zeros(Float32, model.num_agents, model.num_steps)
    model.RL.params.s_t = zeros(Float32, model.num_agents, state_dim, model.num_steps)
    model.RL.params.a_t = zeros(Float32, model.num_agents, model.num_steps)

end

function dqn_struct_init(sim_params)
    state_dim = 2#+sim_params.num_goals*2 + sim_params.num_goals  #BONE
    action_dim = 0
    if sim_params.num_dimensions == "1D"
        action_dim = 3
    elseif sim_params.num_dimensions == "2D"
        action_dim = 5
    else
        @error "Wrong number of dimensions"
    end
    if sim_params.prev_run == "none"
        model = Chain(
                      Dense(state_dim, 32, relu),
                      Dense(32, action_dim)
                     )
    else
        # load in previous model
        prev_model = BSON.load(sim_params.prev_run, @__MODULE__)
        model = prev_model[:Policy].model
    end
    θ = params(model)
    θ⁻ = deepcopy(θ)
    γ = 0.99
    η = 0.005
    ϵ_factor = 1000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    replay_size = 1000
    C = 20
    opt = ADAM(η)
    r_matrix = zeros(Float32, sim_params.num_agents, sim_params.num_steps)
    s_matrix = zeros(Float32, sim_params.num_agents, state_dim, sim_params.num_steps)
    action_matrix = zeros(Float32, sim_params.num_agents, sim_params.num_steps)
    DQN_params = DQN_Global(model, θ, deepcopy(model), θ⁻, opt, ϵ, replay_size, C, r_matrix, s_matrix, action_matrix)

    return RL_Wrapper(DQN_params, DQN_policy_train!, DQN_policy_eval!, DQN_episode_init!, γ)

end
