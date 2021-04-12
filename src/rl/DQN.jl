mutable struct DQN_HyperParams
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
        y = rt + model.DQN_params.γ*maximum(model.DQN.Q̂(st1))
        return (y - model.DQN.Q(st)[at])^2
    end

    for _ in 1:model.num_agents

        # get minibatch data of size k from buffer (H) and update grads
        dθ = Zygote.Grads(IdDict(ps => nothing for ps in model.DQN.θ), model.DQN.θ)

        data = rand(model.buffer.H, model.DQN_params.k)
        for d in data

            # compute gradient
            loss_j = 0
            dθ .+= gradient(model.DQN.θ) do
                loss_j = DQN_loss(d...)
                return loss_j
            end
            training_loss += loss_j
        end
        update!(model.DQN.opt, model.DQN.θ, dθ)

    
       # do soft update of target network

        x = [0.2;0.5]
#        for i in 1:length(model.DQN.θ⁻)
#            model.DQN.θ⁻[i] .= model.DQN.θ[i]*model.DQN_params.τ + model.DQN.θ⁻[i]*(1-model.DQN_params.τ)
#        end
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

    Q = Chain(
                  Dense(state_dim, 16, relu),
                  Dense(16, action_dim)
                 )
    θ = params(Q)
    Q̂ = deepcopy(Q)
    θ⁻ = params(Q̂)
    η = 0.00025 
    # note, 0.00025 and hidden layer dim = 16 work for RMSProp
    #η = 0.00025
    #opt = Flux.Optimise.Optimiser(ClipValue(1), ADAM(η))
    opt = Flux.Optimise.Optimiser(ClipValue(1), RMSProp(η))
    #opt = RMSProp(η)

## HYPER PARAMS
#    K   # replay period
#    k   # minibatch size
#    N   # replay buffer max size
#    τ   # soft update weight
#    ϵ   # ϵ-greedy parameter
#    γ   # MDP discount factor
#    ep_rew  # episode reward
#    ep_loss  # episode loss

    K = 1000
    k = 32
    N = 50_000
    γ = 0.99
    ϵ_factor = 1000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    τ = 0.0001
    γ = 0.99

    DQN_params = DQN_HyperParams(K, k, N, τ, ϵ, γ, 0, 0)
    DQN_network = DQN_Network(Q, θ, Q̂, θ⁻, opt)
    replay_buffer = ReplayBuffer([])

    return DQN_params, DQN_network, replay_buffer

end
