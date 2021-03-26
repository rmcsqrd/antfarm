mutable struct A3C_Global
    model # this is the NN model used for evaluation
    θ     # this is the set of parameters. We share params so θ = θ_v
    π_sa::Array{Float64, 2}  # π_sa(i,t) = probability of action for agent i at step t
    v_s::Array{Float64, 2}   # v_s(i, t) = value of state for agent i at step t
    r_sa::Array{Float64, 2}  # r_sa(i,t) = reward for agent i at step t
end

function A3C_policy_eval(i, t, s_t, r_t, model)
    output = model.RL.params.model(s_t)
    πi_sa = vec(output[1:4])  # 4 actions
    vi_s = output[5]
   
    # generate action list
    actions = [x for x in 1:4]

    # get probabilities
    probs = ProbabilityWeights(softmax(πi_sa))

    # select action
    action = sample(actions, probs)

    # update history for training
    model.RL.params.π_sa[i, t]= probs[action]
    model.RL.params.v_s[i, t] = vi_s
    model.RL.params.r_sa[i, t] = r_t

    return action
end

function A3C_policy_train(model)

    
    opt = ADAM()
    global_reward = 0
    for i in 1:model.num_agents

        # generate history for agent
        tmax = model.ModelStep
        data = Array{Tuple{Float64, Float64}}(undef, tmax)
        [data[i] = (0.0,0.0) for i in 1:length(data)]
        R = model.RL.params.v_s[i, tmax-1]

        for t in reverse(1:tmax-1)
            R = model.RL.params.r_sa[i, t] + model.RL.γ*R
            A_sa = R - model.RL.params.v_s[i, t]
            data[t] = (model.RL.params.π_sa[i, t], A_sa)
        end
        global_reward += sum(model.RL.params.r_sa[i, :])

        # create loss functions
        actor_loss_function(π_sa, A_sa) = log(π_sa)*A_sa
        critic_loss_function(π_sa, A_sa) = A_sa^2
        local actor_loss, critic_loss

        # train model
        for d in data
            
            # start with actor gradients
            dθ = gradient(model.RL.params.θ) do
                actor_loss = actor_loss_function(d[1], d[2])
                return actor_loss
            end
            update!(opt, model.RL.params.θ, dθ)

            # next do critic gradients
            dθ_v = gradient(model.RL.params.θ) do
                critic_loss = critic_loss_function(d[1], d[2])
                return critic_loss
            end
            update!(opt, model.RL.params.θ, dθ_v)
        end
    end
    return global_reward
end
