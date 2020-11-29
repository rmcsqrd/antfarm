using DataFrames, JSONTables, JSON3, JSON

function DispersalSimulationWrapper()
    # simulation params
    max_num_agents = 10
    num_steps = 100

    # run simulation
    agents = 2:max_num_agents
    df = DataFrame(RunID = Any[], RunResults = Any[])
    for agent in agents
        model = DispersalModel(num_agents=agent,
                               num_steps=num_steps)
        norm_time_hist = DispersalSimulation(model)
        push!(df, (agent, norm_time_hist))

    end

    # write the data
    write("simulation_$max_num_agents.json", objecttable(df))

    # read the data
    df = DataFrame(jsontable(open("simulation_10.json", "r")))
    println(df.RunID)
    println(df.RunResults)
    println(df[1, :]) #access the first row of data


end
