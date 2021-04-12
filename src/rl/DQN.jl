mutable struct DQN_HyperParams
    α   # prioritization factor
    β   # weight factor
    K   # replay period
    k   # minibatch size
    N   # replay buffer max size
    τ   # soft update weight
    ϵ   # ϵ-greedy parameter
    γ   # MDP discount factor
    ep_rew  # episode reward
    ep_loss  # episode loss
end

mutable struct ReplayBuffer
    H   # replay buffer
    p   # probability samples
    max_w  # max computed weight so far
    max_p  # max computed probability so far
end

mutable struct DQN_Network
    Q     # action-value function
    θ     # action-value network params
    Q̂     # target action-value function
    θ⁻    # target action-value network params
    opt   # optimiser
end


function DQN_train!(model)
    
    training_loss = 0

    # define loss function
    function DQN_loss(st, at, rt, st1)
        return model.DQN.Q(st)[at]
    end

    function TD(st, at, rt, st1)
        y = rt + model.DQN_params.γ*maximum(model.DQN.Q̂(st1))
        δj = y - model.DQN.Q(st)[at]
        return δj
    end

    # define PER helper function
    function importance_sampling_weight(Pj, N)
        # anneal beta and compute weight
        βp = model.DQN_params.β(model.sim_params.episode_number)
        wj = (1/N * 1/Pj)^βp
        
        # scale
        if model.buffer.max_w != 0
            wj = wj/model.buffer.max_w
        end

        # overwrite weight max if required
        if wj > model.buffer.max_w
            model.buffer.max_w = wj
        end
        return wj
    end

    for _ in 1:model.num_agents

        # get minibatch data of size k from buffer (H) and update grads
        dθ = Zygote.Grads(IdDict(ps => nothing for ps in model.DQN.θ), model.DQN.θ)
        probs = ProbabilityWeights((model.buffer.p .^ model.DQN_params.α) /sum(model.buffer.p .^ model.DQN_params.α))

#        display(model.buffer.p)
#        display(model.buffer.H)
#        @show model.buffer.max_w
#        @show model.buffer.max_p
        for _ in 1:model.DQN_params.k

            # sample according to bias and compute weight
            N = length(model.buffer.H)

            j = sample(1:N, probs)

            # compute weight and scale
            wj = importance_sampling_weight(probs[j], N)

            # compute gradient
            loss_j = 0
            Hj = model.buffer.H[j]
            dθj = gradient(model.DQN.θ) do
                loss_j = DQN_loss(Hj...)
                return loss_j
            end
            #display(dθj.grads)
            # modify gradient by weight and TD
            δj = TD(model.buffer.H[j]...)
#            @show loss_j
#            @show δj
#            @show wj
#            display( dθ.grads)
#            println()
            model.buffer.p[j] = abs(δj)
            training_loss += δj*wj*loss_j
            #dθ .+= dθj .*(δj*wj)
#            dθj = dθj .*(δj*wj)
#            display(dθj.grads)
#            println()
#            dθ .+= dθj .*(δj*wj)

            # overwrite probability max if required
            if model.buffer.p[j] > model.buffer.max_p
                model.buffer.max_p = model.buffer.p[j]
            end
        end
        update!(model.DQN.opt, model.DQN.θ, dθ)

    
       # do soft update of target network

        x = [0.2;0.5]
#        for i in 1:length(model.DQN.θ⁻)
#            model.DQN.θ⁻[i] .= model.DQN.θ[i]*model.DQN_params.τ + model.DQN.θ⁻[i]*(1-model.DQN_params.τ)
#        end
        if model.sim_params.episode_number % 50 == 0
            model.DQN.Q̂ = model.DQN.Q
        end
    end

    model.DQN_params.ep_loss += training_loss
end

function DQN_policy_eval!(s_t, model)
    # select action via ϵ-greedy
    if rand() < model.DQN_params.ϵ(model.sim_params.episode_number)
         action = rand(1:length(keys(model.action_dict)))
    else
        action = argmax(model.DQN.Q(s_t))
    end
    return action
end

function DQN_init(sim_params)
    state_dim = 2#+sim_params.num_goals*2 + sim_params.num_goals
    action_dim = 0
    if sim_params.num_dimensions == "1D"
        action_dim = 3
    elseif sim_params.num_dimensions == "2D"
        action_dim = 5
    else
        @error "Wrong number of dimensions"
    end
    if sim_params.prev_run == "none"
        Q = Chain(
                      Dense(state_dim, 16, relu),
                      Dense(16, action_dim)
                     )
    else
        # load in previous model
        prev_model = BSON.load(sim_params.prev_run, @__MODULE__)
        Q = prev_model[:Policy].model
    end
    θ = params(Q)
    Q̂ = deepcopy(Q)
    θ⁻ = params(Q̂)
    η = 0.00025 
    # note, 0.00025 and hidden layer dim = 16 work for RMSProp
    #η = 0.00025
    opt = Flux.Optimise.Optimiser(ClipValue(1), ADAM(η))
    #opt = RMSProp(η)

## HYPER PARAMS
#    α   # prioritization factor
#    β   # weight factor
#    H   # replay buffer
#    K   # replay period
#    k   # minibatch size
#    N   # replay buffer max size
#    τ   # soft update weight
#    ϵ   # ϵ-greedy parameter
#    γ   # MDP discount factor
#    ep_rew  # episode reward
#    ep_loss  # episode loss
#
    α = 0.6
    β_factor = 1000
    β_min = 0.4
    β(i) = minimum((maximum((β_min, i/β_factor)),1))
    K = 1000
    k = 32
    N = 50_000
    γ = 0.99
    ϵ_factor = 1000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    τ = 0.0001
    γ = 0.99

    DQN_params = DQN_HyperParams(α, β, K, k, N, τ, ϵ, γ, 0, 0)
    DQN_network = DQN_Network(Q, θ, Q̂, θ⁻, opt)
    replay_buffer = ReplayBuffer([], [], 0, 1)

    return DQN_params, DQN_network, replay_buffer

end
