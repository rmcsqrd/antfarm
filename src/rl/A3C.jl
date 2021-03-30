mutable struct A3C_Global
    model # this is the NN model used for evaluation
    θ     # this is the set of parameters. We share params so θ = θ_v
    r_sa::Array{Float32, 2}  # r_sa(i,t) = reward for agent i at step t
    s_t::Array{Float32, 3}   # s_t(i, :, t) = state for agent i at step t
    a_t::Array{Int64, 2}     # a_t(i,t) = action index for agent i at step t
end

# custom Flux layer
struct A3C_Output
    W
end

# define custom layer for output
A3C_Output(in::Int64, out::Int64) = 
    A3C_Output(randn(Float32, out, in))

(m::A3C_Output)(x) = (softmax((m.W*x)[1:size(m.W)[1]-1]),  # π_sa
                      (m.W*x)[size(m.W)[1]])               # v_s

Flux.@functor A3C_Output


function A3C_policy_eval(i, t, s_t, r_t, model)
    πi_sa, _ = model.RL.params.model(s_t)
   
    # generate action list
    actions = [x for x in 1:length(keys(model.action_dict))]

    # get probabilities
    probs = ProbabilityWeights(πi_sa)

    # select action
    action = sample(actions, probs)

    # update history for training
    model.RL.params.r_sa[i, t] = r_t
    model.RL.params.s_t[i,:, t] = s_t
    model.RL.params.a_t[i, t] = action
    return action
end

function A3C_policy_train(model)
    # create loss functions
    function actor_loss_function(R, s_t, a_t)
        model_output = model.RL.params.model(s_t)
        π_sa = model_output[1][a_t]
        v_s = model_output[2]
        return log(π_sa)*(R-v_s)
    end

    function critic_loss_function(R, s_t)
        model_output = model.RL.params.model(s_t)
        v_s = model_output[2]
        return (R-v_s)^2

    end
    
    opt = ADAM(0.01)
    global_reward = 0
    dθ = Grads(IdDict(ps => nothing for ps in model.RL.params.θ), model.RL.params.θ)
    dθ_v = Grads(IdDict(ps => nothing for ps in model.RL.params.θ), model.RL.params.θ)
    for i in 1:model.num_agents

        # get initial rewards and initialize gradients
        tmax = model.ModelStep
        _ , R = model.RL.params.model(model.RL.params.s_t[i, :, tmax-1])

        # accumulate gradients
        for t in reverse(1:tmax-1)
            R = model.RL.params.r_sa[i, t] + model.RL.γ*R
            s_t = model.RL.params.s_t[i, :, t]
            a_t = model.RL.params.a_t[i, t]

            dθ .+= gradient(()->actor_loss_function(R, s_t, a_t), model.RL.params.θ)
            dθ_v .+= gradient(()->critic_loss_function(R, s_t), model.RL.params.θ)
        end

        # update model with accumulated gradients
        global_reward += sum(model.RL.params.r_sa[i, :])
    end
    display(dθ.grads)
    update!(opt, model.RL.params.θ, dθ)
    update!(opt, model.RL.params.θ, dθ_v)
    return global_reward
end
