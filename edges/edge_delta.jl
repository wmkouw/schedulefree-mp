export EdgeDelta

using Distributions: Normal, logpdf

mutable struct EdgeDelta
    "
    Edge for an observation
    "
    # Distribution parameters
    params::Dict{String, Float64}
    prediction_error::Float64

    # Message bookkeeping
    messages::Dict{String, Normal}
    incoming_messages::Dict{String, Queue{Normal}}

    # Factor graph properties
    nodes::Dict{String, Symbol}
    id::String

    function EdgeDelta(data::Float64, nodes::Dict{String, Symbol}, id::String)
        "Outgoing message is updated variational parameters"

        # Location of spike is mean parameter
        params = Dict{String, Float64}("data" => data)

        # Initialize prediction error
        prediction_error = 1e12

        # Initialize messages
        messages = Dict{String, Normal}("top" => Normal())

        # Initialize new message queue
        incoming_messages = Dict{String, Queue{Normal}}("top" => Queue{Normal}())

        self = new(params, prediction_error, messages, incoming_messages, nodes, id)
        return self
    end
end

function entropy(edge::EdgeDelta)
    "Entropy of Normal(⋅, 0.0) evaluates to negative infinity"
    return -Inf
end

function free_energy(edge::EdgeGaussian)
    "Compute free energy of edge and connecting nodes"

    # Query nodes for energy
    U = 0
    for key in keys(edge.nodes)
        U += energy(eval(edge.nodes[key]))
    end

    # Entropy is -Inf, but is disregarded for observed nodes

    # Return FE
    return U
end

function prediction_error(edge::EdgeDelta, message)
    "Compute precision-weighted prediction error"
    return -logpdf(message, edge.params["data"])
end

function message(edge::EdgeDelta)
    "Edge is distribution"
    # TODO: create a delta distribution
    return Normal(edge.params["data"], 0.0)
end

function act(edge::EdgeDelta, message, pred_error)
    "Outgoing message is the delta spike itself"

    # Outgoing signal consists of the message, its error and the edge id
    outgoing_message = (message, pred_error, "data")

    # Put message in queue of connecting nodes
    enqueue!(eval(edge.nodes["top"]).incoming_messages, outgoing_message)
    # TODO: avoid hard-coding node key

    return Nothing
end

function react(edge::EdgeDelta)
    "React to incoming messages"

    # Iterate through remaining messages
    pred_error = 0
    for i = 1:length(edge.incoming_messages["top"])

        # Pop incoming messages
        edge.messages["top"] = dequeue!(edge.incoming_messages["top"])

        # Update variational distribution
        pred_error += prediction_error(edge, edge.messages["top"])

    end

    # Keep track of prediction error
    edge.prediction_error = pred_error

    # Message from edge to nodes
    act(edge, message(edge), pred_error)

    return Nothing
end
