export EdgeDelta

mutable struct EdgeDelta
    "
    Edge for an observation
    "
    # Distribution parameters
    params::Dict{String, Float64}

    # Factor graph properties
    nodes::Dict{String, Symbol}
    id::String

    function EdgeDelta(data::Float64, nodes::Dict{String, Symbol}, id::String)
        "Outgoing message is updated variational parameters"

        # Location of spike is mean parameter
        params = Dict{String, Float64}("mean" => data)

        self = new(params, nodes, id)
        return self
    end
end

function entropy(edge::EdgeDelta)
    "Entropy of Normal(⋅, 0.0) evaluates to negative infinity"
    return -Inf
end

function message(edge::EdgeDelta)
    "Edge is distribution"
    # TODO: create a delta distribution
    return Normal(edge.params["mean"], 0.0)
end

function act(edge::EdgeDelta, message)
    "Outgoing message is the delta spike itself"

    # Put message in queue of connecting nodes
    enqueue!(eval(edge.nodes["top"]).new_messages, (message, "data"))
    # TODO: avoid hard-coding node key

    return Nothing
end

function react(edge::EdgeDelta)
    "React to incoming messages"

    # Message from edge to nodes
    act(edge, message(edge))

    return Nothing
end
