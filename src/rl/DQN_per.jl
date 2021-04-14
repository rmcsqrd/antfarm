mutable struct DQN_HyperParams
    α   # prioritization factor
    β   # weight factor
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
    p   # probability buffer
    max_p
    max_w
end

mutable struct DQN_Network
    Q     # action-value function
    Q̂     # target action-value function
    opt   # optimiser
end


function DQN_train!(model)
    
    training_loss = 0

    # get minibatch data of size k from buffer (H) and update grads
    #dθ = Zygote.Grads(IdDict(ps => nothing for ps in params(model.DQN.Q)), params(model.DQN.Q))

    N = length(model.buffer.H)
    pj_alpha = model.buffer.p .^ model.DQN_params.α
    probs = ProbabilityWeights(pj_alpha ./ sum(pj_alpha))
    j_list = sample(1:N, probs, model.DQN_params.k)

    data = []
    p = []

    for j in j_list
        push!(data, model.buffer.H[j])
    end

    function DQN_PER_loss(s_t1, a_t1, r_t, s_t)
        y = r_t + model.DQN_params.γ*maximum(model.DQN.Q̂(s_t))
        δj = Flux.Losses.huber_loss(y, model.DQN.Q(s_t1)[a_t1])
        return δj
    end

    for (cnt, d) in enumerate(data)
        pj = 0
        s_t1, a_t1, r_t, s_t = d
        dθ = gradient(params(model.DQN.Q)) do
            δj = DQN_PER_loss(d...)
            pj = abs(δj)
            return δj*model.DQN.Q(s_t1)[a_t1]
        end
        push!(p, pj)
        update!(model.DQN.opt, params(model.DQN.Q), dθ)
    end
    #Flux.train!(DQN_PER_loss, params(model.DQN.Q), data, model.DQN.opt)

        # update selection probability
    for (cnt, j) in enumerate(j_list)
        model.buffer.p[j] = p[cnt] + 0.001  # want non-zero probability of selection
        if model.buffer.p[j] > model.buffer.max_p
            model.buffer.max_p = model.buffer.p[j]
        end
        #display(dθ.grads)
    end

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
            popfirst!(model.buffer.p)
        end
        push!(model.buffer.H, (s_t1, a_t1, r_t, s_t))
        push!(model.buffer.p, model.buffer.max_p)
    end
end

function DQN_init(sim_params)
    Q = Chain(
                  Dense(sim_params.state_dim, 16, relu),
                  Dense(16, sim_params.action_dim)
                 )
    Q̂ = deepcopy(Q)
    η = 0.001
    # note, 0.00025 and hidden layer dim = 16 work for RMSProp
    #η = 0.00025
    #opt = Flux.Optimise.Optimiser(ClipValue(1), RMSProp(η))
    #opt = Flux.Optimise.Optimiser(ClipValue(10), RMSProp(η))
    opt = Flux.Optimise.Optimiser(ClipValue(1), ADAM(η))

## HYPER PARAMS
#    α   # prioritization factor
#    β   # weight factor
#    K   # replay period
#    k   # minibatch size
#    N   # replay buffer max size
#    τ   # soft update weight
#    ϵ   # ϵ-greedy parameter
#    γ   # MDP discount factor
#    ep_rew  # episode reward
#    ep_loss  # episode loss

    α = 0.6
    β_factor = 1000
    β_min = 0.4
    β(i) = minimum((maximum((β_min, i/β_factor)),1))
    K = 100
    k = 32
    N = 250_000
    γ = 0.99
    ϵ_factor = 1000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    τ = 0.0001
    γ = 0.99
    C = 10_000

    DQN_params = DQN_HyperParams(α, β, K, k, N, τ, ϵ, γ, C, 0, 0)
    DQN_network = DQN_Network(Q, Q̂, opt)
    replay_buffer = ReplayBuffer([], [], 1, 0)

    return DQN_params, DQN_network, replay_buffer

end
