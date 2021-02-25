cd(@__DIR__)
mutable struct FMP_Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    tau::NTuple{2, Float64}
    color::String
    type::Symbol
    radius::Float64
    SSdims::NTuple{2, Float64}  ## include this for plotting
    Ni::Array{Int64} ## array of tuples with agent ids and agent positions
end

## define AgentBasedModel (ABM)

function FMP_Model()
    properties = Dict(:FMP_params=>FMP_Parameter_Init(),
                      :dt => 0.01,
                      :num_agents=>20,
                      :num_steps=>1500,
                      :step_inc=>2,
                     )

    #space2d = ContinuousSpace((1,1); periodic = true, update_vel! = FMP_Update_Vel)  #BONE commented this out
    space2d = ContinuousSpace((1,1); periodic = true)
    model = ABM(FMP_Agent, space2d, properties=properties)
    return model
end

# Now that we've defined the plot utilities, lets re-run our simulation with
# some additional options. We do this by redefining the model, re-adding the
# agents but this time with a color parameter that is actually used. 
function FMP_Simulation()
    model = FMP_Model()

    x, y = model.space.extent
    r = 0.9*(min(x,y)/2)

    for i in 1:model.num_agents

        ## compute position around circle
        theta_i = (2*π/model.num_agents)*i
        xi = r*cos(theta_i)+x/2
        yi = r*sin(theta_i)+y/2
        
        xitau = r*cos(theta_i+π)+x/2
        yitau = r*sin(theta_i+π)+y/2

        ## set agent params
        pos = (xi, yi)
        vel = (0,0)
        tau = (xitau, yitau)  ## goal is on opposite side of circle
        radius = model.FMP_params.d/2
        agent_color = AgentInitColor(i, model.num_agents)  ## This is new
        add_agent!(pos, model, vel, tau, agent_color, :A, radius, model.space.extent, [])
        add_agent!(tau, model, vel, tau, agent_color, :T, radius, model.space.extent, [])
    end

    agent_step!(agent, model) = move_agent!(agent, model, model.dt)

    function model_step!(model)
        FMP_Update_Interacting_Pairs(model)
        for agent_id in keys(model.agents)
            FMP_Update_Vel(model.agents[agent_id], model)
        end
    end

    e = model.space.extent
    step_range = 1:model.step_inc:model.num_steps

    InteractiveDynamics.abm_video(
        "circle_swap.mp4",
        model,
        agent_step!,
        model_step!,
        title = "FMP Simulation",
        frames = model.num_steps,
        framerate = 100,
        resolution = (600, 600),
        as = PlotABM_RadiusUtil,
        ac = PlotABM_ColorUtil,
        am = PlotABM_ShapeUtil,
        equalaspect=true,
        scheduler = PlotABM_Scheduler,
       )
end 
