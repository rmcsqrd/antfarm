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
    # clear goal interactions then update goal occupation/goal awareness
    model.SS.GO = zeros(Bool, size(model.SS.GO))
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            for goal_id in model.agents[i].Gi
                g = model.GoalHash[hash(goal_id)]
                model.SS.GO[i,g] = 1
                model.SS.GA[i,g] = 1
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
            # update state history
            model.agents[agent_id].State = GetSubstate(model, i)
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

            # give reward for communication
            for neighbor_id in model.agents[i].Ni
                j = model.AgentHash[hash(neighbor_id)]
                info_exchange = xor.(model.SS.GA[i, :], model.SS.GA[j, :])
                beta = sum(info_exchange)/model.num_goals
                model.agents[agent_id].Reward += 10*beta
            end

            # get reward for goal occupation
            model.agents[agent_id].Reward += sum(model.SS.GO[i,:]) 
            model.agents[agent_id].Reward += sum(model.SS.GO[i,:])*alpha
        end
    end
end

function Action(model)
    for agent_id in keys(model.agents)
        if model.agents[agent_id].type == :A
            i = model.AgentHash[hash(agent_id)]
            selected_action, pi_sa, vi_s  = PolicyEvaluate(model, agent_id)
            
            # Symbols don't play nice with data recording:
            #   a number larger than model.num_goals implies random
            #   (see PolicyEvaluate)
            if selected_action > model.num_goals
                model.agents[agent_id].tau = Tuple(rand(2))
                #println("Random action: ",model.agents[agent_id].tau)
            else

                # if agent knows location of target (which is also represented
                # as an agent), then give it the target location
                if model.SS.GA[i, selected_action] == 1
                    model.agents[agent_id].tau = model.Goals[selected_action]
                else
                    # else stay in current position and incur penalty
                    model.agents[agent_id].Reward -= 1
                    model.agents[agent_id].tau = model.agents[agent_id].pos
                end
            end
            model.agents[agent_id].Action = selected_action
            model.agents[agent_id].PiAction = pi_sa
            model.agents[agent_id].Value = vi_s
        end
    end
end
