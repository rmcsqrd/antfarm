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
    function DQN_loss(st_1, at_1, rt, st)
        y = rt + model.DQN_params.γ*maximum(model.DQN.Q̂(st))
        δ_mse = (y - model.DQN.Q(st_1)[at_1])^2 
        training_loss += δ_mse
        return δ_mse
    end

    data = rand(model.buffer.H, model.DQN_params.k)
    #push!(data, model.buffer.H[length(model.buffer.H)])  # BONE, push last state for combined experience replay
    Flux.train!(DQN_loss, params(model.DQN.Q), data, model.DQN.opt)
    model.DQN_params.ep_loss += training_loss

end

function DQN_update_check!(model)

    # save model
    if model.sim_params.total_steps % model.DQN_params.C == 0
        model.DQN.Q̂ = deepcopy(model.DQN.Q)
    end
end

function DQN_policy_eval!(s_t, model, agent_id)
    # select action via ϵ-greedy. Also select by inertia term τ
    if rand() < model.DQN_params.ϵ(model.sim_params.episode_number)
        if rand() < model.DQN_params.τ(model.sim_params.episode_number)
            action = model.agents[agent_id].a_t1
        else
            action = rand(1:length(keys(model.action_dict)))
        end
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
        end
        push!(model.buffer.H, (s_t1, a_t1, r_t, s_t))
    end
end

function DQN_init(sim_params)
    if sim_params.prev_run != "none"
        prev_model = BSON.load(sim_params.prev_run, @__MODULE__)
        Q = prev_model[:Policy].Q
    else
        Q = Chain(
                      Dense(sim_params.state_dim, 64, relu),
                      Dense(64, sim_params.action_dim)
                     )
    end
    Q̂ = deepcopy(Q)
    η = 0.000025
    # note, 0.00025 and hidden layer dim = 16 work for RMSProp
    #η = 0.00025
    opt = Flux.Optimise.Optimiser(ClipValue(1), RMSProp(η))
    #opt = Flux.Optimise.Optimiser(ClipValue(10), RMSProp(η))
    #opt = ADAM(η)

## HYPER PARAMS
#    K   # replay period
#    k   # minibatch size
#    N   # replay buffer max size
#    τ   # soft update weight
#    ϵ   # ϵ-greedy parameter
#    γ   # MDP discount factor
#    ep_rew  # episode reward
#    ep_loss  # episode loss

    K = 100
    k = 32
    N = 1_000_000
    γ = 0.99
    ϵ_factor = 1000
    ϵ(i) = maximum((0.1, (ϵ_factor-i)/ϵ_factor))
    τ_factor = 1000
    τ(i) = maximum((0.0, (τ_factor-i)/τ_factor))*0.85
    γ = 0.99
    C = 100_000

    DQN_params = DQN_HyperParams(K, k, N, τ, ϵ, γ, C, 0, 0)
    DQN_network = DQN_Network(Q, Q̂, opt)
    replay_buffer = ReplayBuffer([])

    return DQN_params, DQN_network, replay_buffer

end
