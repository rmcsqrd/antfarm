mutable struct A3C_Global
    model # this is the NN model used for evaluation
    θ     # this is the set of parameters. We share params so θ = θ_v
    r_sa::Array{Float32, 2}  # r_sa(i,t) = reward for agent i at step t
    s_t::Array{Float32, 3}   # s_t(i, :, t) = state for agent i at step t
    a_t::Array{Int64, 3}     # a_t(i, : ,t) = action index for agent i at step t (all zeros except at index of action. Bottom row is all zeros
end

function A3C_episode_init(model)
    state_dim = 2+model.num_goals*2 + model.num_goals
    action_dim = length(keys(model.action_dict))
    r_matrix = zeros(Float32, model.num_agents, model.num_steps)
    s_matrix = zeros(Float32, model.num_agents, state_dim, model.num_steps)
    action_matrix = zeros(Float32, model.num_agents, action_dim+1, model.num_steps)

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
        π_sa = diag(y'a_t)
        v_s = y[size(y)[1], :]
        return sum(log.(softmax(π_sa)) .* (R-v_s))
    end

    function critic_loss_function(R, s_t)
        y = model.RL.params.model(s_t)
        v_s = y[size(y)[1], :]
        return sum((R-v_s).^2)

    end
    
    opt = ADAM(0.001)
    global_reward = 0
    for i in 1:model.num_agents

        # initialize stuff and calculate rewards
        tmax = model.ModelStep
        R = zeros(tmax-1)
        R[tmax-1]= model.RL.params.model(model.RL.params.s_t[i, :, tmax-1])[length(keys(model.action_dict))+1]
        for t in reverse(1:tmax-2)
            R[t] = model.RL.params.r_sa[i, t] + model.RL.γ*R[t+1]
        end

        # get state in proper shape, compute gradients, update
        s_t = model.RL.params.s_t[i, :, :]
        a_t = model.RL.params.a_t[i, :, :]
        dθ = gradient(model.RL.params.θ) do
            actor_loss = actor_loss_function(R, s_t, a_t)

            # was having issues with NaN so this returns an "empty" gradient so
            # you don't poison the well
            if isnan(actor_loss) || isinf(actor_loss)
                return dropgrad(actor_loss)
            else
                return actor_loss
            end
        end
        dθ_v = gradient(model.RL.params.θ) do
            critic_loss = critic_loss_function(R, s_t)

            # was having issues with NaN so this returns an "empty" gradient so
            # you don't poison the well
            if isnan(critic_loss) || isinf(critic_loss)
                return dropgrad(critic_loss)
            else
                return critic_loss
            end
        end
        display(dθ.grads)
        display(dθ_v.grads)
        update!(opt, model.RL.params.θ, dθ)
        update!(opt, model.RL.params.θ, dθ_v)

        # update model with accumulated gradients
        global_reward += sum(model.RL.params.r_sa[i, :])
    end
    return global_reward
end
