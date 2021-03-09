function LostHiker(model)

    # determine circle params
    absx, absy = model.space.extent
    x, y = (absx, absy) .* 0.9 # also add scale factor so it doesn't end up outside of space
    starting_pos = (rand(0:0.01:x), rand(0:0.01:y))

    # add some noise to positions because I was getting error for having agents in same position
    agent_place(start) = start .+ (rand(0:0.0001:0.01), rand(0:0.0001:0.01))
    goal_place() = (rand(0:0.01:x), rand(0:0.01:y))

    for i in 1:model.num_agents
        # set agent params
        pos = agent_place(starting_pos)
        vel = (0,0)
        tau = goal_place()
        radius = model.FMP_params.d/2
        color = AgentInitColor(i, model.num_agents)
        add_agent!(pos, model, vel, tau, color, :A, radius, model.space.extent, [], [])
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [], [])
    end
end
