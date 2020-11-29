using LinearAlgebra 

function GridlockDispersalAnalyze(agent_df, model)
    # first sort by agent type - sorts alphabetically and agents are :A so they are on top
    sort!(agent_df, :type)  

    # next loop through positions of agents and compute the L2 norm
    #   (euclidean distance) from center of state space and the agent
    #   position. Store this in a vector then take the mean of the vector
    #   for average distance from assumed gridlock centroid.
    
    centroid = model.space.extend ./ 2  # elementwise division by 2 b/c its a tuple
    norm_vect = Vector{Float64}(undef, model.num_agents)

    for i in 1:model.num_agents
        # when computing the norm not that positions are tuples so do operations elementwise
        norm_vect[i] = norm(centroid .- agent_df.pos[i])
    end

    norm_mean = sum(norm_vect)/model.num_agents
    return norm_mean
end
