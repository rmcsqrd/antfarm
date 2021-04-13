mutable struct DQN_HyperParams
    K   # replay period
    k   # minibatch size
    N   # replay buffer max size
    τ   # soft update weight
    ϵ   # ϵ-greedy parameter
    γ   # MDP discount factor
    C   # DQN model freeze interval
    ep_rew  # episode reward
    ep_loss  # episode loss
end

mutable struct ReplayBuffer
    H   # replay buffer
end

mutable struct DQN_Network
    Q     # action-value function
    Q̂     # target action-value function
    opt   # optimiser
end


function DQN_train!(model)
    
    training_loss = 0

    # define loss function
    function DQN_loss(st, at, rt, st1)
        y = rt + model.DQN_params.γ*maximum(model.DQN.Q̂(st1))
        return (y - model.DQN.Q(st)[at])^2
    end

    # get minibatch data of size k from buffer (H) and update grads
    dθ = Zygote.Grads(IdDict(ps => nothing for ps in params(model.DQN.Q)), params(model.DQN.Q))

    data = rand(model.buffer.H, model.DQN_params.k)
    for d in data

        # compute gradient
        loss_j = 0
        dθ .+= gradient(params(model.DQN.Q)) do
            loss_j = DQN_loss(d...)
            return loss_j
        end
        training_loss += loss_j
    end

    update!(model.DQN.opt, params(model.DQN.Q), dθ)
    model.DQN_params.ep_loss += training_loss
end

function DQN_update_check!(model)

    # save model
    if model.sim_params.total_steps % model.DQN_params.C == 0
        model.DQN.Q̂ = model.DQN.Q
    end
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

function DQN_buffer_update!(s_t1, a_t1, r_t, s_t, model)
    # first step in model is nothing and we don't want to push that
    if !isnothing(s_t1)
        if length(model.buffer.H) == model.DQN_params.N
            popfirst!(model.buffer.H)
#                popfirst!(model.buffer.p)
        end
        push!(model.buffer.H, (s_t1, a_t1, r_t, s_t))
#            push!(model.buffer.p, model.buffer.max_p)
    end
end

function DQN_init(sim_params)
    Q = Chain(
                  Dense(sim_params.state_dim, 64, relu),
                  Dense(64, sim_params.action_dim)
                 )
    Q̂ = deepcopy(Q)
    η = 0.00025 
    # note, 0.00025 and hidden layer dim = 16 work for RMSProp
    #η = 0.00025
    #opt = Flux.Optimise.Optimiser(ClipValue(1), RMSProp(η))
    opt = Flux.Optimise.Optimiser(ClipValue(10), RMSProp(η))
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

    K = 10_000
    k = 32
    N = 500_000
    γ = 0.99
    ϵ_factor = 1000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    τ = 0.0001
    γ = 0.99
    C = 10_000

    DQN_params = DQN_HyperParams(K, k, N, τ, ϵ, γ, C, 0, 0)
    DQN_network = DQN_Network(Q, Q̂, opt)
    replay_buffer = ReplayBuffer([])

    return DQN_params, DQN_network, replay_buffer

end
