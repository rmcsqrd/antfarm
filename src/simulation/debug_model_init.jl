function reward_debug(model)
    pos1 = (0.8, 0.5)
    pos2 = (0.2, 0.5)
    # add agents
    add_agent!(pos1,
               model,
               (0.0,0.0),
               pos1,
               "#FF0000",
               :A,
               model.FMP_params.d/2,
               model.space.extent,
               [], [])
    add_agent!(pos1,
               model,
               (0.0,0.0),
               pos1,
               "#FF0000",
               :A,
               model.FMP_params.d/2,
               model.space.extent,
               [], [])
    # add goals
    add_agent!(pos2,
               model,
               (0.0,0.0),
               pos2,
               "#FF0000",
               :T,
               model.FMP_params.d/2,
               model.space.extent,
               [], [])
    add_agent!(pos2,
               model,
               (0.0,0.0),
               pos2,
               "#FF0000",
               :T,
               model.FMP_params.d/2,
               model.space.extent,
               [], [])
end
