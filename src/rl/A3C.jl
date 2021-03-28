mutable struct A3C_Global
    model # this is the NN model used for evaluation
    θ     # this is the set of parameters. We share params so θ = θ_v
    π_sa::Array{Float64, 2}  # π_sa(i,t) = probability of action for agent i at step t
    v_s::Array{Float64, 2}   # v_s(i, t) = value of state for agent i at step t
    r_sa::Array{Float64, 2}  # r_sa(i,t) = reward for agent i at step t
end

# custom Flux layer
mutable struct A3C_Output
    W
end

# define custom layer for output
A3C_Output(in::Int64, out::Int64) = 
    A3C_Output(randn(out, in))

(m::A3C_Output)(x) = (softmax((m.W*x)[1:size(m.W)[1]-1]),  # π_sa
                            (m.W*x)[size(m.W)[1]])               # v_s

Flux.@functor A3C_Output


function A3C_policy_eval(i, t, s_t, r_t, model)
    πi_sa, vi_s = model.RL.params.model(s_t)
   
    # generate action list
    actions = [x for x in 1:4]

    # get probabilities
    probs = ProbabilityWeights(πi_sa)

    # select action
    action = sample(actions, probs)
   # println("s_t = $s_t")
   # println("action = $action")
   # println("probs = $probs\n")

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
        data = Array{Tuple{Float64, Float64}}(undef, tmax-1)
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
        #local actor_loss_function, critic_loss_function

        # BEGIN BONE
#        println("pre-training: ")
#        display(model.RL.params.θ)
#
#        # this works
#        loss(x, y) = Flux.Losses.mse(model.RL.params.model(x), y)
#        data = (rand(5), rand(5))
#        println(loss(data[1], data[2]))
#        dθ = gradient(() -> loss(data[1], data[2]), model.RL.params.θ)
#        update!(opt, model.RL.params.θ, dθ)

        #END BONE
        # BONE: this does not
        # train model
        for d in data
            # start with actor gradients
            dθ = gradient(() -> actor_loss_function(d[1], d[2]), model.RL.params.θ)
            update!(opt, model.RL.params.θ, dθ)

#            # next do critic gradients
#            dθ_v = gradient(() -> critic_loss_function(d[1], d[2]), model.RL.params.θ)
#            update!(opt, model.RL.params.θ, dθ_v)
        end
        #for d in data
            #println(d)
            #println(actor_loss_function(d))
            #Flux.train!(actor_loss_function, model.RL.params.θ, data, opt)
            #println(d[1], "  ", d[2])
        #end
    end
        display(model.RL.params.θ)
        println("\n")
        sleep(1)
    #display(model.RL.params.π_sa)  # BONE, don't delete. this is a clue, between iterations it appears to be copying the probability of selection for agents? is this expected
    return global_reward
end
