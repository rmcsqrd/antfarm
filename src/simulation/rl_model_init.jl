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

        if model.num_agents > 1
            # if only one agent this returns an error
            color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        else
            color = "#FF0000"
        end


        # seed agents with random positions initially
        add_agent!(pos, model, vel, pos, color, :A, radius, model.space.extent, [], [])

        # add targets normally
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [], [])
    end
end

function SimpleTest(model)
    x,y = model.space.extent
    starting_pos = (0.8*x, 0.5*y)
    for i in 1:model.num_agents
        pos = starting_pos .+ (rand(0:0.00001:0.0001), rand(0:0.00001:0.0001))
        vel = (0,0)
        tau = (0.2*x, 0.5*y)
        radius = model.FMP_params.d/2
        if model.num_agents > 1
            # if only one agent this returns an error
            color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        else
            color = "#FF0000"
        end
        # seed agents with random positions initially
        add_agent!(pos, model, vel, pos, color, :A, radius, model.space.extent, [], [])

    end

    for i in 1:model.num_goals
        vel = (0,0)
        tau = (0.2*x, 0.5*y)
        radius = model.FMP_params.d/2
        if model.num_agents > 1
            # if only one agent this returns an error
            color = string("#", hex(range(HSV(0,1,1), stop=HSV(-360,1,1), length=model.num_agents)[i]))
        else
            color = "#FF0000"
        end

        # add targets normally
        add_agent!(tau, model, vel, tau, color, :T, radius, model.space.extent, [], [])
    end
end
