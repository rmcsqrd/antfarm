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


## define AgentBasedModel (ABM)

function fmp_model(rl_arch; num_agents=20, num_goals=num_agents, num_steps=1500)
    properties = Dict(:FMP_params=>FMP_Parameter_Init(),
                      :dt => 0.05,
                      :num_agents=>num_agents,
                      :num_goals=>num_goals,
                      :num_steps=>num_steps,
                      :step_inc=>2,
                      :SS=>StateSpace(zeros(Bool, num_agents, num_goals),  # GA
                                      zeros(Bool, num_agents, num_goals),  # GO
                            ),
                      :Actions=>[(1,0), (0,1), (-1,0), (0,-1)],
                      :Agents2RL=>Dict{Int64, Int64}(),  # dict to map Agents.jl agent_ids to RL formulation id values
                      :Goals=>Dict{Int64, Tuple{Float64, Float64}}(),  # dict to map RL formulation goal id's to position of Agents.jl goal (agent.type == :T)
                      :ModelStep=>1,
                      :RL=>rl_arch,
                     )

    space2d = ContinuousSpace((1,1); periodic = true)
    model = ABM(FMP_Agent, space2d, properties=properties)
    return model
end
