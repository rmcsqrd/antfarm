"""
Run model only
"""
function run_model!(model, agent_step!, model_step!)
    run!(model, agent_step!, model_step!, model.num_steps)
end

"""
Run model and create output video
"""
function run_model_plot!(model, agent_step!, model_step!, sim_params)
    # delete original file
    try
        rm(filepath)
    catch
    end

    # plot stuff
    filepath = string(homedir(),"/Programming/antfarm/src/data_output/episode_$(sim_params.episode_number).mp4")
    InteractiveDynamics.abm_video(
        filepath,
        model,
        agent_step!,
        model_step!,
        title = "FMP Simulation, Epoch #$(sim_params.episode_number)",
        frames = model.num_steps,
        framerate = 100,
        resolution = (600, 600),
        as = as_f(a) = 1200*1/minimum(a.SSdims)*a.radius,  ## this was defined empirically
        ac = ac_f(a) = a.type in (:A, :O) ? a.color : "#A9A9A9",
        am = am_f(a) = a.type in (:A, :O, :T) ? :circle : :circle,
        equalaspect=true,
        scheduler = PlotABM_Scheduler,
       )

end


"""
This function is a scheduler to determine draw order of agents. Draw order (left to right) is :T, :O, :A
"""
function PlotABM_Scheduler(model::ABM)

    # init blank lists
    agent_list = []
    object_list = []
    target_list = []
    for agent in values(model.agents)
        if agent.type == :A
            append!(agent_list, agent.id)
        elseif agent.type == :T
            append!(target_list, agent.id)
        elseif agent.type == :O
            append!(object_list, agent.id)
        end
    end

    # make composite list [targets, objects, agents]
    draw_order = []
    append!(draw_order, target_list)
    append!(draw_order, object_list)
    append!(draw_order, agent_list)

    return draw_order
end

function DebugScreenshotPlot(model; filepath="/Users/riomcmahon/Desktop/plot.png")
        plotabm(model,
            as = as_f(a) = 380*1/minimum(a.SSdims)*a.radius,  ## this was defined empirically
            ac = ac_f(a) = a.type in (:A, :O) ? a.color : "#A9A9A9",
            am = am_f(a) = a.type in (:A, :O, :T) ? :circle : :circle,
            #showaxis = false,
            grid = false,
            xlims = (0, model.space.extent[1]),
            ylims = (0, model.space.extent[2]),
            aspect_ratio=:equal,
            scheduler = PlotABM_Scheduler
               )
        savefig(filepath)
end
