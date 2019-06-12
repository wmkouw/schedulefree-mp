export NodeGaussian

using Distributions: Normal, params
using DataStructures: Queue, enqueue!, dequeue!

mutable struct NodeGaussian

    # Factor graph properties
    id::String
    edges::Dict{String, Symbol}

    # Message bookkeeping
    messages::Dict{String, Normal}
    new_messages::Queue{Tuple}

    # Node properties
    transition::Float64
    precision::Float64
    threshold::Float64
    verbose::Bool

    function NodeGaussian(edge_data_id::Symbol,
                          edge_mean_id::Symbol,
                          transition::Float64,
                          precision::Float64,
                          id::String;
                          threshold=0.0001,
                          verbose=false)

        # Connect node to specific edges
        edges = Dict{String, Symbol}("data" => edge_data_id,
                                     "mean" => edge_mean_id)

        # Keep track of incoming messages
        messages = Dict{String, Normal}("data" => Normal(0.0, 1.0),
                                        "mean" => Normal(0.0, 1.0))

        # Initialize new message queue
        new_messages = Queue{Tuple}()

        # Create instance
        self = new(id, edges, messages, new_messages, transition, precision, threshold, verbose)
        return self
    end
end

function energy(node::NodeGaussian)
    "Compute internal energy of node"

    # Expected mean
    Em = node.transition * eval(node.edges["mean"]).params["mean"]

    # Expected data
    Ex = eval(node.edges["data"]).params["mean"]

    # -log-likelihood of Gaussian with expected parameters
    return 1/2 *log(2*pi) - log(node.precision) + 1/2 *(Ex - Em)'*node.precision*(Ex - Em)
end

function message(node::NodeGaussian, edge::String)
    "Compute outgoing message"

    if edge == "data"

        # Expected mean
        Em = node.transition * node.edges["mean"].params[1]

        # Supply sufficient parameters for normal as output message
        message = Normal(Em, node.precision)

    elseif edge == "mean"

        # Expected data
        Ex = node.edges["data"].params[1]

        # Supply sufficient parameters for normal as output message
        message = Normal(Ex, node.precision)

    else
        throw("Exception: edge unknown.")
    end

    return message
end

function act(node::NodeGaussian, edge::String)
    "Send out message for one of the connecting edges"

    if edge == "data"

        # TODO: hard-block when data edge is a delta dist

        # Compute message for data parameter
        message_data = message(node, "data")

        # Put message in queue of edge for data parameter
        enqueue!(node.edge["data"].new_messages["data"], message_data)

    elseif edge == "mean"

        # Compute message for mean parameter
        message_mean = message(node, "mean")

        # Put message in queue of edge for mean parameter
        enqueue!(node.edge["mean"].new_messages["mean"], message_mean)

    else
        throw("Exception: edge unknown.")
    end

    return Nothing
end

function react(node::NodeGaussian)
    "Decide to react based on delta Free Energy"

    # Number of new messages
    num_new = length(node.new_messages)

    # Compute current internal energy
    F_old = energy(node)

    for n = 1:num_new

        # Check current message
        message, entropy, edge = dequeue!(node.new_messages)

        # Update edge
        node.messages[edge] = message

        # New energy
        F_new = energy(node) - entropy

        # Compute change in internal energy
        dF = abs(F_new - F_old)

        if node.verbose
            println("dU = "*string(dU))
        end

        # Check if change in energy is sufficient to fire
        if dU > node.threshold
            act(node)
        end
    end

    return Nothing
end
