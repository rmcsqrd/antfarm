export FMP_Epoch

mutable struct FMP_Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    tau::NTuple{2, Float64}
    color::String
    type::Symbol
    radius::Float64
    SSdims::NTuple{2, Float64}  ## include this for plotting
    Ni::Vector{Int64} ## array of neighboring agent IDs
    Gi::Vector{Int64} ## array of neighboring goal IDs
end

mutable struct A3C_Agent
    Pi_prime
    V_prime
    StateHist
    ActionHist
    RewardHist
end

mutable struct A3C_Global
    Pi
    V
end

## define AgentBasedModel (ABM)

function FMP_Model(; num_agents=20, num_goals=num_agents, num_steps=1500)
    properties = Dict(:FMP_params=>FMP_Parameter_Init(),
                      :dt => 0.01,
                      :num_agents=>num_agents,
                      :num_goals=>num_goals,
                      :num_steps=>num_steps,
                      :step_inc=>2,
                      :SS=>StateSpace(zeros(Bool, num_agents, num_goals),  # GA
                             zeros(Bool, num_agents, num_goals),  # GO
                             zeros(Bool, num_agents, num_goals),  # GI
                             zeros(Bool, num_agents, num_agents), # AI
                            ),
                      :AgentHash=>Dict{Int128, Int128}(),
                      :GoalHash=>Dict{Int128, Int128}(),
                      :A3C=>Dict(),
                      :ModelStep=>1,
                     )

    space2d = ContinuousSpace((1,1); periodic = true)
    model = ABM(FMP_Agent, space2d, properties=properties)
    return model
end

function FMP_Epoch()
    params = []
    FMP_Episode(params)
end

# Now that we've defined the plot utilities, lets re-run our simulation with
# some additional options. We do this by redefining the model, re-adding the
# agents but this time with a color parameter that is actually used. 
function FMP_Episode(a3c_global_params)

    # define FMP params
    num_agents = 20
    num_goals = 20
    num_steps = 1500

    # define model
    model = FMP_Model(; num_agents=num_agents, num_goals=num_goals, num_steps=num_steps)
    
    # initialize model by adding in agents
    LostHiker(model)
    #AgentPositionInit(model; type="circle")

    # create agent/goal hashes for RL stuff
    StateSpaceHashing(model)
    
    # initialize the A3C struct
    A3C_Episode_Init(model, a3c_global_params)

    # define agent/model step stuff
    function agent_step!(agent, model)
        move_agent!(agent, model, model.dt)
    end
    # define progress meter (optional)
    p = Progress(model.num_steps)


    function model_step!(model)
        # do FMP stuff - figure out interacting pairs and update velocities
        # accordingly
        FMP_Update_Interacting_Pairs(model)
        for agent_id in keys(model.agents)
            FMP_Update_Vel(model.agents[agent_id], model)
        end

        # do RL stuff 
        StateTransition(model)
        Reward(model)
        #Action(model)  # if you comment this out it behaves as vanilla FMP
        model.ModelStep += 1
        
        # show progress
        next!(p)

    end

    # plot stuff
    InteractiveDynamics.abm_video(
        "/Users/riomcmahon/Desktop/circle_swap.mp4",
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

    RewardPlot(model)
end 
