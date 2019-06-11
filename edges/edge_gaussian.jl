export EdgeGaussian

using Distributions: Normal, mean, std
using DataStructures: Queue, enqueue!, dequeue!

mutable struct EdgeGaussian
    "
    Edge with a Gaussian recognition distribution
    "
    # Recognition distribution parameters
    params::Dict{String, Float64}
    change_entropy::Float64

    # Messages
    messages::Dict{String, Normal}
    new_messages::Dict{String, Queue{Normal}}

    # Factor graph properties
    nodes::Dict{String, Symbol}
    state_type::String
    id::Symbol

    function EdgeGaussian(mean::Float64, 
                          precision::Float64, 
                          state_type::String,
                          nodes::Dict{String,Symbol},
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

        # Construct instance
        self = new(id, nodes, messages, params, change_entropy)
        return self
    end
end

function update(edge::Type{EdgeGaussian}, message_left, message_right)
    "Update recognition distribution as the product of messages"

    # Compute entropy of edge before update
    entropy_old = entropy(edge)

    # Means
    mean_l = message_left.params["mean"]
    mean_r = message_right.params["mean"]

    # Precisions
    precision_l = message_left.params["precision"]
    precision_r = message_right.params["precision"]

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

function entropy(edge::Type{EdgeGaussian})
    "Compute entropy of Gaussian distribution"
    return log(2*pi)/2 + log(edge.params["precision"]^2)/2
end

function act(edge::Type{EdgeGaussian}, message)
    "Outgoing message is updated variational parameters"

        # State type determines which nodes are connected
        if edge.state_type == "previous"

            # Put message in queue of connecting nodes
            enqueue!(edge.nodes["right"].new_messages, (message, "mean"))

        elseif edge.state_type == "current"
            
            # Put message in queue of connecting nodes
            enqueue!(edge.nodes["left"].new_messages, (message, "mean"))
            enqueue!(edge.nodes["bottom"].new_messages, (message, "mean"))

        else
            throw("Exception: state_type unknown.")
        end
    return Nothing
end

function react(edge::Type{EdgeGaussian})
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
    message = Normal(edge.params["mean"], edge.params["precision"])
    act(edge, message)

    return Nothing
end
