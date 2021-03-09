mutable struct MDP
    State
    Action
    Transition
    Discount
    Reward
end

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

function StateTransition(model)
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
                
                # update state history
                model.A3C[i].StateHist[:, model.ModelStep] = GetSubstate(model, i)
            end
        end
    end

end

function Reward(model)
    # compute team performance scaling factors
    team_performance = sum(model.SS.GO, dims=1)
    opt_performance = [model.num_agents/model.num_goals for x in 1:model.num_goals]
    team_performance = reshape(team_performance, length(team_performance))
    opt_performance = reshape(opt_performance, length(opt_performance))
    alpha = 1/max(1,norm(team_performance-opt_performance))

    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            if length(model.agents[i].Gi) > 0
                println("i = $i, GO = ")
                println(model.SS.GO[i, :], "\n")
                println(model.agents[i].Gi, "\n")
            end

            # give reward for communication
            for neighbor_id in model.agents[i].Ni
                j = model.AgentHash[hash(neighbor_id)]
                info_exchange = xor.(model.SS.GA[i, :], model.SS.GA[j, :])
                beta = sum(info_exchange)/model.num_goals
                model.A3C[i].RewardHist[model.ModelStep] += 10*beta
            end

            # get reward for goal occupation
            model.A3C[i].RewardHist[model.ModelStep] += sum(model.SS.GO[i,:]) 
            model.A3C[i].RewardHist[model.ModelStep] += sum(model.SS.GO[i,:])*alpha
        end
    end
end

function Action(model)
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            selected_action = PolicyEvaluate(model, i)

            # if random selected, give random target
            if selected_action == :random
                action = Tuple(rand(2))
            else

                # if agent knows location of target (which is also represented
                # as an agent), then give it the target location
                goal_id = model.GoalHash[hash(selected_action)]
                if model.SS.GA[i, selected_action] == 1
                    action = model.agents[goal_id].pos
                else
                    # else more randomly
                    action = Tuple(rand(2))
                end
            end
            model.agents[agent_id].tau = action
            model.A3C[i].ActionHist[model.ModelStep] = action
        end
    end
end
