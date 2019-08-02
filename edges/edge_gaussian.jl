export EdgeGaussian

using Distributions: Normal, mean, std, params
using DataStructures: Queue, enqueue!, dequeue!

mutable struct EdgeGaussian
    "
    Edge with a Gaussian recognition distribution
    "
    # Recognition distribution parameters
    params::Dict{String, Float64}
    free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Normal}
    incoming_messages::Dict{String, Queue{Normal}}

    # Factor graph properties
    nodes::Dict{String, Symbol}
    state_type::String
    id::String

    function EdgeGaussian(mean::Float64,
                          precision::Float64,
                          state_type::String,
                          nodes::Dict{String, Symbol},
                          id::String;
                          free_energy=1e12)

        # Check valid precision
        if precision <= 0
            throw("Exception: non-positive precision.")
        end

        # TODO: check keys of node dictionary (valid={"left", "right", "bottom"})

        # Set recognition distribution parameters
        params = Dict{String, Float64}("mean" => mean,
                                       "precision" => precision)

        # Initialize messages
        messages = Dict{String, Normal}()
        for key in keys(nodes)
            messages[key] = Normal()
        end

        # Initialize new message queue
        incoming_messages = Dict{String, Queue{Normal}}()
        for key in keys(nodes)
            incoming_messages[key] = Queue{Normal}()
        end

        # Construct instance
        self = new(params,
                   free_energy,
                   messages,
                   incoming_messages,
                   nodes,
                   state_type,
                   id)
        return self
    end
end

function update(edge::EdgeGaussian, message_left::Normal, message_right::Normal)
    "Update recognition distribution as the product of messages"

    # Extract parameters of messages
    mean_l, precision_l = params(message_left)
    mean_r, precision_r = params(message_right)

    # Update variational parameters
    precision = (precision_l + precision_r)
    mean = inv(precision)*(precision_l*mean_l + precision_r*mean_r)

    # Update attributes
    edge.params["precision"] = precision
    edge.params["mean"] = mean

    return Nothing
end

function message(edge::EdgeGaussian)
    "Outgoing message"
    return Normal(edge.params["mean"], edge.params["precision"])
end

function entropy(edge::EdgeGaussian)
    "Compute entropy of Gaussian distribution"
    return log(2*π * ℯ / edge.params["precision"])/2
end

function free_energy(edge::EdgeGaussian)
    "Compute free energy of edge and connecting nodes"

    # Query nodes for energy
    U = 0
    for key in keys(edge.nodes)
        U += energy(eval(edge.nodes[key]))
    end

    # Compute own entropy
    H = entropy(edge)

    # Return FE
    return U - H
end

function act(edge::EdgeGaussian, out_message, delta_free_energy::Float64)
    "Outgoing message is updated variational parameters"

        # State type determines which nodes are connected
        if edge.state_type == "previous"

            # Tuple of message, change in free energy of message and edge id
            T = (out_message, delta_free_energy, "mean")

            # Put message in queue of connecting nodes
            enqueue!(eval(edge.nodes["right"]).incoming_messages, T)
            # TODO: avoid hard-coding node key

        elseif edge.state_type == "current"

            # Tuples of message, change in free energy of message and edge id
            T_left = (out_message, delta_free_energy, "data")
            T_bottom = (out_message, delta_free_energy, "mean")
            # TODO: edge and node id's are confusing

            # Put message in queue of connecting nodes
            enqueue!(eval(edge.nodes["left"]).incoming_messages, T_left)
            enqueue!(eval(edge.nodes["bottom"]).incoming_messages, T_bottom)
            # TODO: avoid hard-coding node key

        else
            throw("Exception: state_type unknown.")
        end
    return Nothing
end

function react(edge::EdgeGaussian)
    "React to incoming messages"

    # Check lengths of queues
    l1 = length(edge.incoming_messages["left"])
    l2 = length(edge.incoming_messages["bottom"])

    # Iterate through the longer queue until both queues are equally long
    for i = 1:abs(l1 - l2)

        if l1 > l2
            edge.messages["left"] = dequeue!(edge.incoming_messages["left"])
        else
            edge.messages["bottom"] = dequeue!(edge.incoming_messages["bottom"])
        end

        # Update variational distribution
        update(edge, edge.messages["left"], edge.messages["bottom"])
    end

    # Iterate through remaining messages
    for i = 1:min(l1, l2)

        # Pop incoming messages
        edge.messages["left"] = dequeue!(edge.incoming_messages["left"])
        edge.messages["bottom"] = dequeue!(edge.incoming_messages["bottom"])

        # Update variational distribution
        update(edge, edge.messages["left"], edge.messages["bottom"])
    end

    # Compute delta free energy
    delta_free_energy = free_energy(edge) - edge.free_energy

    # Update edge's free energy
    edge.free_energy = free_energy(edge)

    # Message from edge to nodes
    act(edge, message(edge), delta_free_energy)

    return Nothing
end
