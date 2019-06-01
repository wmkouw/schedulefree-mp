export EdgeDelta

mutable struct EdgeDelta
    "
    Edge for an observation
    "
    # Distribution parameters
    data

    # Factor graph properties
    id
    node_id

    function message()
        "Outgoing message is updated variational parameters"
        return Delta(data)
    end
end

function probability(x)
    "Non-zero for observed value"
    if x == data
        return 1
    else
        return 0
    end
end