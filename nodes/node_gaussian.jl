export NodeGaussian

using Distributions: Normal, params, mean
using DataStructures: Queue, enqueue!, dequeue!

mutable struct NodeGaussian

    # Node properties
    transition::Float64
    control::Float64
    precision::Float64
    threshold::Float64

    # Bookkeeping of recognition distributions
    beliefs::Dict{String, Normal}

    # Identifiers of edges/nodes in factor graph
    id::String
    id_mean::String
    id_data::String

    # Additional properties
    verbose::Bool

    function NodeGaussian(id,
                          id_mean,
                          id_data;
                          transition=1.0,
                          control=0.0,
                          precision=1.0,
                          threshold=0.0001,
                          verbose=false)

        # Keep track of recognition distributions
        beliefs = Dict{String, Normal}(id_mean => Normal(),
                                       id_data => Normal())

        # Create instance
        self = new(id,
                   id_mean,
                   id_data,
                   beliefs,
                   transition,
                   control,
                   precision,
                   threshold,
                   verbose)
        return self
    end
end

function energy(node::NodeGaussian)
    "Compute internal energy of node"

    # Expected mean
    Em = node.transition * mean(node.beliefs[id_mean]) + node.control

    # Expected data
    Ex = mean(node.beliefs[id_data])

    # -log-likelihood of Gaussian with expected parameters
    return 1/2 *log(2*pi) - log(node.precision) + 1/2 *(Ex - Em)'*node.precision*(Ex - Em)
end

function forward_message(node::NodeGaussian, edge_id::String)
    "Compute message forward to x"

    if edge_id == "data"

        # Expected mean
        Em = node.transition * mean(node.messages["mean"])

        # Supply sufficient parameters for normal as output message
        message = Normal(Em, node.precision)

    elseif edge_id == "mean"

        # Expected data
        Ex = mean(node.messages["data"])

        # Supply sufficient parameters for normal as output message
        message = Normal(Ex, node.precision)

    else
        throw("Exception: edge id unknown.")
    end

    return message
end

function act(node::NodeGaussian, edge_id::String)
    "Send out message for one of the connecting edges"

    # Compute message for a particular edge
    outgoing_message = message(node, edge_id)

    if node.edges[edge_id] == :x_t
        #TODO: avoid hard-coding queue keys of edges

        # Push message in queue of connected edge
        enqueue!(eval(node.edges[edge_id]).incoming_messages["left"], outgoing_message)

    elseif node.edges[edge_id] == :y_t

        # Push message in queue of connected edge
        enqueue!(eval(node.edges[edge_id]).incoming_messages["top"], outgoing_message)

    elseif node.edges[edge_id] == :z_t

        # Push message in queue of connected edge
        enqueue!(eval(node.edges[edge_id]).incoming_messages["right"], outgoing_message)

    else
        throw("Exception: Unknown edge.")
    end

    return Nothing
end

function react(node::NodeGaussian)
    "Decide to react based on delta Free Energy"

    for n = 1:length(node.incoming_messages)

        # Check current message
        incoming_message, delta_free_energy, edge_in = dequeue!(node.incoming_messages)

        # Update edge
        node.messages[edge_in] = incoming_message

        # Report dFE
        if node.verbose
            println("dFE = "*string(delta_free_energy))
        end

        # Check if change in energy is sufficient to fire
        if abs(delta_free_energy) > node.threshold

            for edge_out in setdiff(Set(keys(node.edges)), [edge_in])
                act(node, edge_out)
            end
        end
    end

    return Nothing
end
