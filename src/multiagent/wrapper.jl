#Wrapper function fot the Agents.jl module
# Implementation inspired by 
#   https://github.com/JuliaDynamics/Agents.jl/blob/master/examples/social_distancing.jl

using Agents, Random, AgentsPlots, Plots, ProgressMeter

mutable struct Agent <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    mass::Float64
end


function model_space(; speed=0.002)

    # initialize state space
    space2d = ContinuousSpace(2; periodic=true, extend=(1,1))
    #Agent(id, pos, vel; mass, happy) = Agent(id, pos, vel, mass, happy)
    model = ABM(Agent, space2d, properties = Dict(:dt=>1.0, :rho=>7.5e6, :r=>10))

    # add agents
    num_agents = 1000
    radius = 0.001
    for ind in 1:num_agents
        thetap = ind*2*3.1415/num_agents
        #x = radius*cos(thetap)
        #y = radius*sin(thetap)
        e = model.space.extend
        #x = 0.5*e[1]  # this is a conveinent way to work within model extents
        x = ind/num_agents*e[1]
        y = ind/num_agents*e[2]
        pos = Tuple((x,y))
        #vel = sincos(2π*rand()).*(speed)
        vel = Tuple((-speed, speed))
        add_agent!(pos, model, vel, 1.0)
    end

    index!(model)
    return model
end

function create_gif()
    gr()
    cd(@__DIR__)

    model = model_space()
    agent_step!(agent, model) = move_agent!(agent, model, model.dt)
    nothing

    e = model.space.extend  # this gives dimensions of model space I think
    
    num_steps = 100
    p = Progress(round(Int,num_steps/2))
    anim = @animate for i in 1:2:num_steps
        p1 = plotabm(
            model,
            as = 4,
            #showaxis = false,
            #grid = false,
            xlims = (0, e[1]),
            ylims = (0, e[1]),
        )

        title!(p1, "step $(i)")
        step!(model, agent_step!, 2)
        next!(p)
    end
    gif(anim, "output/socialdist1.gif", fps = 25)
end


function stuff()
    println(" test")
end
