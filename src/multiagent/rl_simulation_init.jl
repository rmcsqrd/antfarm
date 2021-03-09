function LostHiker(model)

    # determine circle params
    x, y = model.space.extent
    starting_pos = (rand(0:0.01:x), rand(0:0.01:y)).*0.9

    for i in 1:model.num_agents
        # add some noise to positions because I was getting error for having agents in same position
        eps = rand(0:0.001:0.1) 

        # set agent params
        pos = starting_pos .+ eps
        vel = (0,0)
        #tau = (rand(0:0.01:x), rand(0:0.01:y)).*0.9  # also add scale factor so it doesn't end up outside of space
        tau = starting_pos .+ (eps*2)
        radius = model.FMP_params.d/2
        color = AgentInitColor(i, model.num_agents)
        add_agent!(pos, model, vel, tau, color, :A, radius, model.space.extent, [], [])
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [], [])
    end
end
