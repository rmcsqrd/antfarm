using DataFrames, JSONTables, ProgressMeter, Plots

function DispersalPlot(fileloc)
    cd(@__DIR__)
    df = DataFrame(jsontable(read(fileloc)))
    p = plot()
    for run_df in eachrow(df)
        num_agents = run_df.NumAgents
        x = 1:run_df.StepInc:run_df.NumSteps
        y = run_df.RunResults
        plot!(x,y, label="# agents = $num_agents")
    end
    xlabel!("# Steps")
    ylabel!("Average Distance from SS Center")
    title!("Dispersal Simulation: Time vs Distance from Centroid")
    savefig("distplot.png")
    
    p = plot(legend=false)
    for run_df in eachrow(df)
        num_agents = run_df.NumAgents
        x = num_agents
        y = run_df.RunTime
        scatter!((x,y))
    end
    xlabel!("# Agents")
    ylabel!("Run Time (s)")
    title!("Dispersal Simulation: Run Time")
    savefig("runtimePlot.png")
        
end

function DispersalSimulationWrapper()
    # simulation params
    min_num_agents = 10
    max_num_agents = 1000
    agent_range_interval = 10

    num_steps = 3500
    step_inc = 2
    SS_dims = (8, 8)

    # run simulation
    agents_range = min_num_agents:agent_range_interval:max_num_agents
    df = DataFrame(NumAgents = Any[],
                   NumSteps = Any[],
                   StepInc = Any[],
                   RunTime = Any[],
                   RunResults = Any[])

    @showprogress for num_agents in agents_range
        model = DispersalModel(num_agents=num_agents,
                               step_inc=step_inc,
                               num_steps=num_steps,
                               SS_dims=SS_dims)
        # time function: result[1] = function result
        #                result[2] = time elapsed
        norm_time_hist = @timed DispersalSimulation(model)
        push!(df, (num_agents, 
                   num_steps,
                   step_inc,
                   norm_time_hist[2],
                   norm_time_hist[1]))

    end

    # write the data
    write(string("simres/simulation_min$min_num_agents",
                 "_int$agent_range_interval",
                 "_max$max_num_agents",
                 "_n$num_steps",
                 "_stepinc$step_inc",
                 "_SSdims$SS_dims",
                 ".json"), objecttable(df))


end


function DispersalPlotRadius(fileloc)
    cd(@__DIR__)
    df = DataFrame(jsontable(read(fileloc)))
    p = plot(legend=:bottomright)
    k=1

    for run_df in eachrow(df)
        num_agents = run_df.NumAgents
        x = 1:run_df.StepInc:run_df.NumSteps
        y = run_df.RunResults
        plot!(x,y, label="r=ra*$k")
        k+=0.1
        k = round(k;digits=2)
    end
    xlabel!("# Steps")
    ylabel!("Average Distance from SS Center")
    title!("Dispersal Simulation: Time vs Distance from Centroid")
    savefig("distplot.png")
    
    p = plot(legend=false)
    k = 1
    for run_df in eachrow(df)
        num_agents = run_df.NumAgents
        x = k
        y = run_df.RunTime
        scatter!((x,y))
        k+=0.1
        k = round(k;digits=2)
    end
    xlabel!("# Agents")
    ylabel!("Run Time (s)")
    title!("Dispersal Simulation: Run Time")
    savefig("runtimePlot.png")
        
end
function DispersalSimulationWrapperRadius()
    # simulation params
    min_num_agents = 100
    max_num_agents = 100
    agent_range_interval = 100

    num_steps = 3500
    step_inc = 2
    SS_dims = (1, 1)
    radmultrange = 1:0.25:3

    # run simulation
    agents_range = min_num_agents:agent_range_interval:max_num_agents
    df = DataFrame(NumAgents = Any[],
                   NumSteps = Any[],
                   StepInc = Any[],
                   RunTime = Any[],
                   RunResults = Any[])

    @showprogress for radmult in radmultrange
        model = DispersalModel(num_agents=100,
                               step_inc=step_inc,
                               num_steps=num_steps,
                               r=0.02*radmult,
                               SS_dims=SS_dims)
        # time function: result[1] = function result
        #                result[2] = time elapsed
        norm_time_hist = @timed DispersalSimulation(model)
        push!(df, (100, 
                   num_steps,
                   step_inc,
                   norm_time_hist[2],
                   norm_time_hist[1]))

    end

    # write the data
    write(string("simres/simulation_min$min_num_agents",
                 "_int$agent_range_interval",
                 "_max$max_num_agents",
                 "_n$num_steps",
                 "_stepinc$step_inc",
                 "_SSdims$SS_dims",
                 "_radmult$radmultrange",
                 ".json"), objecttable(df))


end
