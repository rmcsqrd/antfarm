# Wrapper function fot the Agents.jl module
# Implementation inspired by 
#   https://github.com/JuliaDynamics/Agents.jl/blob/master/examples/social_distancing.jl

using Agents, Random, AgentsPlots, Plots

mutable struct Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    mass::Float64
end


function model_space(; speed=0.002)

    # initialize state space
    space2d = ContinuousSpace(2; periodic=true, extend=(1,1))
    model = ABM(Agent, space2d, properties = Dict(:dt=>1.0))

    # add agents
    Random.seed!(42)
    for ind in 1:10
        pos = Tuple(rand(2))
        vel = sincos(2π*rand()).*speed
        add_agent!(pos, model, vel, 1.0)
    end

    index!(model)
    return model
end

gr()
cd(@__DIR__)

model = model_space()
agent_step!(agent, model) = move_agent!(agent, model, model.dt)
nothing

e = model.space.extend
anim = @animate for i in 1:2:100
    p1 = plotabm(
        model,
        as = 4,
        showaxis = false,
        grid = false,
        xlims = (0, e[1]),
        ylims = (0, e[2]),
    )

    title!(p1, "step $(i)")
    step!(model, agent_step!, 2)
end
gif(anim, "socialdist1.gif", fps = 25)
println("I did stuff")


function stuff()
    println("dog")
end
