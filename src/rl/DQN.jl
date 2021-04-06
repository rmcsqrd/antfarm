mutable struct DQN_Global
    Q     # action-value function
    Q̂     # target action-value function
    Q̂_rew # reward for Q̂
    opt   # optimiser
    ϵ     # ϵ-greedy parameter
    replay_size::Int64  # how many data samples are trained on
    minibatch_size::Int64
    C
    r_t::Array{Float32, 2}   # r_t(i,t) = reward for agent i at step t
    s_t::Array{Float32, 3}   # s_t(i, :, t) = state for agent i at step t
    a_t::Array{Int64, 2}     # a_t(i, t) = action  for agent i at step t 
end

function DQN_policy_train!(model)

    # TRAINING OVERVIEW
    # 1. initialize empty grads
    # 2. accumulate gradients for each agent
    # 3. update model params
    
    training_loss = 0
    state_dim = 2+model.num_goals*2 + model.num_goals
    action_dim = length(keys(model.action_dict))
    for i in 1:model.num_agents

        # generate replay data
        data_idx = rand(1:model.num_steps-1, model.RL.params.replay_size-1)  # -1 because 
        s_j = model.RL.params.s_t[i, :, data_idx]
        s_j1 = model.RL.params.s_t[i, :, data_idx .+ 1]
        r_j = model.RL.params.r_t[i, data_idx]
        Q̂_j1 = model.RL.params.Q̂(s_j1)
        y_j = r_j + model.RL.γ .* vec(Q̂_j1[argmax(Q̂_j1, dims=1)])
       
        a_j = zeros(action_dim, model.RL.params.replay_size-1)
        for (j, k) in enumerate(data_idx)  # i is agent, j is index, k is time step
            a_j[model.RL.params.a_t[i, k], j] = 1
        end

        # define loss function
        function loss_function(st, st1, a, r)
            out = model.RL.params.Q(st)
            Qsa = diag(out'a)
            y = r + model.RL.γ .* vec(maximum(model.RL.params.Q̂(st1), dims=1))
            #clipped_grad = min.(max.(y .- Qsa, -1), 1)
            #return sum(clipped_grad .^ 2)
            return sum((y .- Qsa) .^2)
        end

        # compute batch bounds
        mbs = model.RL.params.minibatch_size
        rps = model.RL.params.replay_size-1
        low = [i for i in 1:mbs:rps]
        high = [i for i in mbs:mbs:rps]
        batch_bounds = map((i,j)->(i,j), low, high)

        for (low, high) in batch_bounds

            s_batch = s_j[:, low:high]
            st1_batch = s_j1[:, low:high]
            a_batch = a_j[:, low:high]
            r_batch = r_j[low:high]
            
            dθ = gradient(params(model.RL.params.Q)) do 
                loss = loss_function(s_batch, st1_batch, a_batch, r_batch)
                training_loss += loss
                return loss
            end

            update!(model.RL.params.opt, params(model.RL.params.Q), dθ)
            
        end

    end

    # update target network with current if it is better performing
    if model.sim_params.episode_number % model.RL.params.C == 0
    #if sum(model.RL.params.r_t) > model.RL.params.Q̂_rew
        @info "Setting Q̂ = Q"
        model.RL.params.Q̂ = deepcopy(model.RL.params.Q)
        model.RL.params.Q̂_rew = sum(model.RL.params.r_t)
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
end

function DQN_episode_init!(model)
    state_dim = 2+model.num_goals*2 + model.num_goals
    action_dim = length(keys(model.action_dict))

    model.RL.params.r_t = zeros(Float32, model.num_agents, model.num_steps)
    model.RL.params.s_t = zeros(Float32, model.num_agents, state_dim, model.num_steps)
    model.RL.params.a_t = zeros(Float32, model.num_agents, model.num_steps)

end

function dqn_struct_init(sim_params)
    state_dim = 2+sim_params.num_goals*2 + sim_params.num_goals
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
    γ = 0.99
    η = 0.005
    ϵ_factor = 2000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    replay_size = 5000
    minibatch_len = 1
    C = 100
    Q̂_rew = -Inf
    opt = ADAM(η)
    r_matrix = zeros(Float32, sim_params.num_agents, sim_params.num_steps)
    s_matrix = zeros(Float32, sim_params.num_agents, state_dim, sim_params.num_steps)
    action_matrix = zeros(Float32, sim_params.num_agents, sim_params.num_steps)
    DQN_params = DQN_Global(model, deepcopy(model), Q̂_rew, opt, ϵ, replay_size, minibatch_len, C, r_matrix, s_matrix, action_matrix)

    return RL_Wrapper(DQN_params, DQN_policy_train!, DQN_policy_eval!, DQN_episode_init!, γ)

end
