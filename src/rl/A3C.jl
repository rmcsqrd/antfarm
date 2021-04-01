mutable struct A3C_Global
    model # this is the NN model used for evaluation
    θ     # this is the set of parameters. We share params so θ = θ_v
    η     # this is learning rate
    β     # this is entropy parameter
    r_sa::Array{Float32, 2}  # r_sa(i,t) = reward for agent i at step t
    s_t::Array{Float32, 3}   # s_t(i, :, t) = state for agent i at step t
    a_t::Array{Int64, 3}     # a_t(i, : ,t) = action index for agent i at step t (all zeros except at index of action. Bottom row is all zeros
end

function A3C_episode_init(model)
    state_dim = 2+model.num_goals*2 + model.num_goals
    action_dim = length(keys(model.action_dict))
    r_matrix = zeros(Float32, model.num_agents, model.num_steps)
    s_matrix = zeros(Float32, model.num_agents, state_dim, model.num_steps)
    action_matrix = zeros(Float32, model.num_agents, action_dim, model.num_steps)

    model.RL.params.r_sa = r_matrix
    model.RL.params.s_t = s_matrix
    model.RL.params.a_t = action_matrix

end

function A3C_policy_eval(i, t, s_t, r_t, model)
    y = model.RL.params.model(s_t)
    π_sa = y[1:length(keys(model.action_dict))]
   
    # generate action list
    actions = [x for x in 1:length(keys(model.action_dict))]

    # get probabilities
    probs = ProbabilityWeights(softmax(π_sa))

    # select action
    action = sample(actions, probs)

    # update history for training
    model.RL.params.r_sa[i, t] = r_t
    model.RL.params.s_t[i,:, t] = s_t
    model.RL.params.a_t[i, action, t] = 1

    return action
end

function A3C_policy_train(model)

    # create loss functions
    function actor_loss_function(R, s_t, a_t)
        y = model.RL.params.model(s_t)
        π_s = softmax(y[1:size(y)[1]-1, :]) # probabilities of all actions
        v_s = y[size(y)[1], :]              # value function
        π_sa = diag(π_s'a_t)                # probability of selected action
        H = -model.RL.params.β*sum(π_s .* log.(π_s), dims=1)  # entropy
        return sum((log.(π_sa) .* (R-v_s))+vec(H))
        #return sum((log.(π_sa) .* (R-v_s)))
    end

    function critic_loss_function(R, s_t)
        y = model.RL.params.model(s_t)
        v_s = y[size(y)[1], :]
        return sum((R-v_s).^2)

    end
    # TRAINING OVERVIEW
    # 1. initialize empty grads
    # 2. accumulate gradients for each agent
    # 3. update model params
    
    opt = ADAM(model.RL.params.η)
    global_reward = 0
    dθ = Grads(IdDict(ps => nothing for ps in model.RL.params.θ), model.RL.params.θ)
    dθ_v = Grads(IdDict(ps => nothing for ps in model.RL.params.θ), model.RL.params.θ)
    for i in 1:model.num_agents

        # initialize stuff and calculate rewards
        tmax = model.ModelStep
        R = zeros(tmax)
        R[tmax]= model.RL.params.model(model.RL.params.s_t[i, :, tmax-1])[length(keys(model.action_dict))+1]
        for t in reverse(1:tmax-1)
            R[t] = model.RL.params.r_sa[i, t] + model.RL.γ*R[t+1]
        end
        R = R[1:tmax-1]

        # get state in proper shape, compute gradients, update
        s_t = model.RL.params.s_t[i, :, :]
        a_t = model.RL.params.a_t[i, :, :]
        dθ .+= gradient(()->actor_loss_function(R, s_t, a_t), model.RL.params.θ)
        dθ_v .+= gradient(()->critic_loss_function(R, s_t), model.RL.params.θ)
        
        # update model with accumulated gradients
        global_reward += sum(model.RL.params.r_sa[i, :])
    end
    update!(opt, model.RL.params.θ, dθ)
    update!(opt, model.RL.params.θ, dθ_v)
    display(model.RL.params.θ)
    #display(dθ.grads)
    #display(dθ_v.grads)
    return global_reward
end
