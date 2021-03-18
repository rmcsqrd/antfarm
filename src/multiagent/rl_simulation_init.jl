function LostHiker(model)

    # determine circle params
    absx, absy = model.space.extent
    x, y = (absx, absy) .* 0.9 # also add scale factor so it doesn't end up outside of space
    starting_pos = (rand(0:0.1:x), rand(0:0.1:y))

    # add some noise to positions because I was getting error for having agents in same position

    for i in 1:model.num_agents
        # set agent params
        pos = starting_pos .+ (rand(0:0.00001:0.0001), rand(0:0.00001:0.0001))
        vel = (0,0)
        tau = (rand(0:0.01:x), rand(0:0.01:y))
        radius = model.FMP_params.d/2
        color = AgentInitColor(i, model.num_agents)

        # seed agents with random positions initially
        add_agent!(pos, model, vel, pos, color, :A, radius, model.space.extent, [], [], 0, 0.0, 0.0, 0.0, [], [])

        # add targets normally
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [], [], 0, 0.0, 0.0, 0.0, [],  [])
    end
end
