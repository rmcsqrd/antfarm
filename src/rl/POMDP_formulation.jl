# add a struct for the POMDP
# add in state space, obsevation space, etc


mutable struct StateSpace
    # Goal awareness vector for agent
    # GA(i, g) = agent i is interacting with goal g
    GA::Array{Bool,2} 

    # Goal occupation vector for agent
    # GO(i, g) = agent i knows location of goal g
    GO::Array{Bool,2} 

    # Goal intention vector of agent
    # GI(i, g) = agent i intends to interact with goal g
    GI::Array{Bool,2} 

    # Agent interaction vector of agent
    # AI(i, j) = agent i is interacting with agent j
    AI::Array{Bool,2} 
end

mutable struct ObservationSpace
    # Goal Awareness Belief  
    # GAB(i, j, g) = agent i believes agent j knows location of goal g
    GAB::Array{Bool, 3} # 

    # Goal intention belief
    # GIB(i, j, g) = agent i belief that agent j is heading to goal g
    GIB::Array{Bool, 3}

    # Agent interaction belief
    # AI(i, j) = agent i believes that it is interacting with agent j
    AIB::Array{Bool, 2}
end

mutable struct POMDP
    State
    Action
    Observation
    StateTransition
    Reward
    ObservationTransition
    γ
end

function RL_Wrapper(model)
    ComputeReward(model)
    StateTransitionUpdate(model)  # GO, GA, AI
    ObservationTransitionUpdate(model) # GAB, GIB, AIB
    ActionSelection(model) # GI


    # 

end

function StateTransitionUpdate(model)
    # Goal Occupation  
    # clear goal interactions then update
    model.SS.GO = zeros(Bool, size(model.SS.GO))
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            for goal_id in model.agents[i].Gi
                g = model.GoalHash[hash(goal_id)]
                model.SS.GO[i,g] = 1
            end
        end
    end
    
    # Goal Awareness/Agent Interaction
    # clear agent interactions and goal awareness then update 
    model.SS.AI = zeros(Bool, size(model.SS.AI))
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            for neighbor_id in model.agents[i].Ni

                # first update agent interactions
                j = model.AgentHash[hash(neighbor_id)]
                model.SS.AI[i,j] = 1

                # next update goal awareness using bitwise or
                # to simulate information exchange
                model.SS.GA[i, :] = model.SS.GA[i, :] .| model.SS.GA[j, :]  # period makes it elementwise
                model.SS.GA[j, :] = model.SS.GA[j, :] .| model.SS.GA[i, :]  # period makes it elementwise
            end
        end
    end
end

function ObservationTransitionUpdate(model)


    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            for neighbor_id in model.agents[i].Ni

                # first update goal intention according to some function
                gi_update = GoalIntentionUpdate(model.SS.GI[i, :],model.SS.GI[j, :])
                model.SS.GI[i, :] = gi_update
                model.SS.GI[j, :] = gi_update
            end
        end
    end
end

function GoalIntentionUpdate(GI1, GI2)
    # BONE, this is random as placeholder
    # first figure out how GI arrays are different 
end

function NeighboringGoals(model)

end

function UpdateGoals(model)

end
