mutable struct DQN_Global
    Q     # action-value function
    Q̂     # target action-value function
    Q̂_rew # reward for Q̂
    opt   # optimiser
    ϵ     # ϵ-greedy parameter
    minibatch_length::Int64  # how many data samples are trained on
    r_t::Array{Float32, 2}   # r_t(i,t) = reward for agent i at step t
    s_t::Array{Float32, 3}   # s_t(i, :, t) = state for agent i at step t
    a_t::Array{Int64, 2}     # a_t(i, t) = action  for agent i at step t 
end

function DQN_policy_train!(model)

    # TRAINING OVERVIEW
    # 1. initialize empty grads
    # 2. accumulate gradients for each agent
    # 3. update model params
    
    dθ = Grads(IdDict(ps => nothing for ps in params(model.RL.params.Q)), params(model.RL.params.Q))
    training_loss = 0
    state_dim = 2+model.num_goals*2 + model.num_goals
    action_dim = length(keys(model.action_dict))
    for i in 1:model.num_agents

        # generate minibatch data
        data_idx = rand(1:model.num_steps-1, model.RL.params.minibatch_length)  # -1 because 
        y = zeros(model.RL.params.minibatch_length)
        s = zeros(state_dim, model.RL.params.minibatch_length)
        a = zeros(action_dim, model.RL.params.minibatch_length)
        for (j, k) in enumerate(data_idx)  # i is agent, j is index, k is time step
            s_t = model.RL.params.s_t[i, :, k]
            s_t1 = model.RL.params.s_t[i, :, k+1]
            s[:, j] = s_t
            a[model.RL.params.a_t[i, k], j] = 1
            if k+1 == model.num_steps
                y[j] = model.RL.params.r_t[k]
            else
                y[j] = model.RL.params.r_t[k] + model.RL.γ*maximum(model.RL.params.Q̂(s_t1))
            end
        end
        
        # define loss function
        function loss_function(s, a, y)
            out = model.RL.params.Q(s)
            Qsa = diag(out'a)
            #H = -model.RL.params.ϵ*sum(softmax(out) .* log.(softmax(out)), dims=1)  # entropy
            #return sum((y .- Qsa) .^ 2 + vec(H))
            return sum((y .- Qsa) .^ 2)
        end

        # do gradient descent and update model
        dθ .+= gradient(params(model.RL.params.Q)) do
            loss = loss_function(s, a, y)
            training_loss += loss
            return loss
        end
    end
    update!(model.RL.params.opt, params(model.RL.params.Q), dθ)

    # update target network with current if it is better performing
    current_reward = sum(model.RL.params.r_t .* [model.RL.γ^(t-1) for t in 1:model.num_steps])
    println("Q reward = $current_reward, Q̂ reward = $(model.RL.params.Q̂_rew)")
    if current_reward > model.RL.params.Q̂_rew
        @info "Replacing Q̂"
        model.RL.params.Q̂ = deepcopy(model.RL.params.Q)
        model.RL.params.Q̂_rew = current_reward
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
                      Dense(state_dim, 128, relu),
                      Dense(128, action_dim)
                     )
    else
        # load in previous model
        prev_model = BSON.load(sim_params.prev_run, @__MODULE__)
        model = prev_model[:Policy].model
    end
    γ = 0.99
    η = 0.001
    ϵ_factor = 5000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    minibatch_len = 5_000
    Q̂_rew = -Inf
    opt = ADAM(η)
    r_matrix = zeros(Float32, sim_params.num_agents, sim_params.num_steps)
    s_matrix = zeros(Float32, sim_params.num_agents, state_dim, sim_params.num_steps)
    action_matrix = zeros(Float32, sim_params.num_agents, sim_params.num_steps)
    DQN_params = DQN_Global(model, deepcopy(model), Q̂_rew, opt, ϵ, minibatch_len, r_matrix, s_matrix, action_matrix)

    return RL_Wrapper(DQN_params, DQN_policy_train!, DQN_policy_eval!, DQN_episode_init!, γ)

end
