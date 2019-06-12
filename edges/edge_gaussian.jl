export EdgeGaussian

using Distributions: Normal, mean, std, params
using DataStructures: Queue, enqueue!, dequeue!

mutable struct EdgeGaussian
    "
    Edge with a Gaussian recognition distribution
    "
    # Recognition distribution parameters
    params::Dict{String, Float64}
    change_entropy::Float64

    # Message bookkeeping
    messages::Dict{String, Normal}
    new_messages::Dict{String, Queue{Normal}}

    # Factor graph properties
    nodes::Dict{String, Symbol}
    state_type::String
    id::String

    function EdgeGaussian(mean::Float64,
                          precision::Float64,
                          state_type::String,
                          nodes::Dict{String, Symbol},
                          id::String)

        # Check valid precision
        if precision <= 0
            throw("Exception: non-positive precision.")
        end

        # TODO: check keys of node dictionary (valid={"left", "right", "bottom"})

        # Set recognition distribution parameters
        params = Dict{String, Float64}("mean" => mean, "precision" => precision)

        # Initial change in entropy
        change_entropy = Inf

        # Initialize messages
        messages = Dict{String, Normal}("data" => Normal(0.0, 1.0),
                                        "mean" => Normal(0.0, 1.0))

        # Initialize new message queue
        new_messages = Dict{String, Queue{Normal}}("data" => Queue{Normal}(),
                                                   "mean" => Queue{Normal}())

        # Construct instance
        self = new(params, change_entropy, messages, new_messages, nodes, state_type, id)
        return self
    end
end

function update(edge::EdgeGaussian, message_left::Normal, message_right::Normal)
    "Update recognition distribution as the product of messages"

    # Compute entropy of edge before update
    entropy_old = entropy(edge)

    # Extract parameters of messages
    mean_l, precision_l = params(message_left)
    mean_r, precision_r = params(message_right)

    # Update variational parameters
    precision = (precision_l + precision_r)
    mean = inv(precision)*(precision_l*mean_l + precision_r*mean_r)

    # Update attributes
    edge.params["precision"] = precision
    edge.params["mean"] = mean

    # Compute new entropy
    entropy_new = entropy(edge)

    # Store change in entropy
    edge.change_entropy = entropy_old - entropy_new

    return Nothing
end

function message(edge::EdgeGaussian)
    "Outgoing message"
    return Normal(edge.params["mean"], edge.params["precision"])
end

function entropy(edge::EdgeGaussian)
    "Compute entropy of Gaussian distribution"
    return log(2*pi)/2 - log(edge.params["precision"])/2
end

function free_energy(edge::EdgeGaussian)
    "Compute free energy of edge and connecting nodes"

    # Query nodes for energy
    for key in keys(edge.nodes)
        U = energy(eval(edge.nodes[key]))
    end

    # Compute own entropy
    H = entropy(edge)

    # Return FE
    return U - H
end

function act(edge::EdgeGaussian, message)
    "Outgoing message is updated variational parameters"

        # State type determines which nodes are connected
        if edge.state_type == "previous"

            # Put message in queue of connecting nodes
            enqueue!(eval(edge.nodes["right"]).new_messages, (message, "mean"))
            # TODO: avoid hard-coding node key

        elseif edge.state_type == "current"

            # Put message in queue of connecting nodes
            enqueue!(eval(edge.nodes["left"]).new_messages, (message, "data"))
            enqueue!(eval(edge.nodes["bottom"]).new_messages, (message, "mean"))
            # TODO: avoid hard-coding node key

        else
            throw("Exception: state_type unknown.")
        end
    return Nothing
end

function react(edge::EdgeGaussian)
    "React to incoming messages"

    # Check lengths of queues
    l1 = length(edge.new_messages["data"])
    l2 = length(edge.new_messages["mean"])

    # Iterate through the longer queue until both queues are equally long
    for i = 1:abs(l1 - l2)

        if l1 > l2
            edge.messages["data"] = dequeue!(edge.new_messages["data"])
        else
            edge.messages["mean"] = dequeue!(edge.new_messages["mean"])
        end

        # Update variational distribution
        update(edge, edge.messages["mean"], edge.messages["data"])
    end

    # Iterate through remaining messages
    for i = 1:min(l1, l2)

        # Pop incoming messages
        edge.messages["mean"] = dequeue!(edge.new_messages["mean"])
        edge.messages["data"] = dequeue!(edge.new_messages["data"])

        # Update variational distribution
        update(edge, edge.messages["mean"], edge.messages["data"])

        # TODO: track change in entropy
    end

    # Message from edge to nodes
    act(edge, message(edge))

    return Nothing
end
