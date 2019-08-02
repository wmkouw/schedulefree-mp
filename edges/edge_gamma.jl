export EdgeGamma

using Distributions: Gamma, mean, params
using DataStructures: Queue, enqueue!, dequeue!
using SpecialFunctions: gamma, digamma

mutable struct EdgeGamma
    "
    Edge with a Gamma recognition distribution
    "
    # Recognition distribution parameters
    params::Dict{String, Float64}
    local_free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Gamma}
    incoming_messages::Dict{String, Queue{Gamma}}

    # Factor graph properties
    nodes::Dict{String, Symbol}
    id::String

    function EdgeGamma(shape::Float64,
                       rate::Float64,
                       nodes::Dict{String, Symbol},
                       id::String;
                       local_free_energy=1e12)

        # Check valid parameters
        if shape < 0
            throw("Exception: shape parameter should be larger than 0.")
        end
        if rate < 0
            throw("Exception: rate parameter should be larger than 0.")
        end

        # Set recognition distribution parameters
        params = Dict{String, Float64}("shape" => shape,
                                       "rate" => rate)

        # Initialize messages
        messages = Dict{String, Gamma}()
        for key in keys(nodes)
            messages[key] = Gamma()
        end

        # Initialize new message queue
        incoming_messages = Dict{String, Queue{Gamma}}("bottom" => Queue{Gamma}())

        # Construct instance
        self = new(params,
                   local_free_energy,
                   messages,
                   incoming_messages,
                   nodes,
                   id)
        return self
    end
end

function update(edge::EdgeGamma, message_left::Gamma, message_right::Gamma)
    "Update recognition distribution as the product of messages"

    # Extract parameters of messages
    alpha_l, beta_l = params(message_left)
    alpha_r, beta_r = params(message_right)

    # Update attributes
    edge.params["shape"] = alpha_l + alpha_r - 1
    edge.params["rate"] = beta_l + beta_r

    return Nothing
end

function message(edge::EdgeGamma)
    "Outgoing message"
    return Gamma(edge.params["shape"], edge.params["rate"])
end

function entropy(edge::EdgeGamma)
    "Compute entropy of Gamma distribution"

    # Parameters
    alpha = edge.params["shape"]
    beta = edge.params["rate"]

    return alpha - log(beta) + log(gamma(alpha)) + (1-alpha)*digamma(alpha)
end

function free_energy(edge::EdgeGamma)
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

function act(edge::EdgeGamma, out_message, delta_free_energy)
    "Outgoing message is updated variational parameters"

    # Put message in queue of connecting nodes
    for key in keys(edge.nodes)
        enqueue!(eval(edge.nodes[key]).incoming_messages, (out_message, delta_free_energy, key))
    end

    return Nothing
end

function react(edge::EdgeGamma)
    "React to incoming messages"

    # Check lengths of queues
    l1 = length(edge.incoming_messages["left"])
    l2 = length(edge.incoming_messages["right"])

    # Iterate through the longer queue until both queues are equally long
    for i = 1:abs(l1 - l2)

        if l1 > l2
            edge.messages["left"] = dequeue!(edge.incoming_messages["left"])
        else
            edge.messages["right"] = dequeue!(edge.incoming_messages["right"])
        end

        # Update variational distribution
        update(edge, edge.messages["left"], edge.messages["right"])
    end

    # Iterate through remaining messages
    for i = 1:min(l1, l2)

        # Pop incoming messages
        edge.messages["left"] = dequeue!(edge.incoming_messages["left"])
        edge.messages["right"] = dequeue!(edge.incoming_messages["right"])

        # Update variational distribution
        update(edge, edge.messages["left"], edge.messages["right"])
    end

    # Compute delta free energy
    delta_free_energy = free_energy(edge) - edge.local_free_energy

    # Update edge's free energy
    edge.local_free_energy = free_energy(edge)

    # Message from edge to nodes
    act(edge, message(edge), delta_free_energy)

    return Nothing
end
