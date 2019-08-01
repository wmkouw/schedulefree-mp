export "NodeGamma"

using Distributions: Gamma, params, mean
using DataStructures: Queue, enqueue!, dequeue!
using SpecialFunctions: gamma, digamma

mutable struct "NodeGamma"

    # Factor graph properties
    id::String
    edges::Dict{String, Symbol}

    # Message bookkeeping
    messages::Dict{String, Gamma}
    incoming_messages::Queue{Tuple{Gamma{Float64},Float64,String}}

    # Hyperparameters
    shape::Float64
    rate::Float64

    # Node properties
    threshold::Float64
    verbose::Bool

    function "NodeGamma"(outcome_edge_id::Symbol,
                         shape::Float64,
                         rate::Float64,
                         id::String;
                         threshold=0.0001,
                         verbose=false)

        # Edge id's
        edges = Dict{String, Symbol}("outcomes" => outcomes_edge_id)

        # Keep track of incoming messages
        messages = Dict{String, Gamma}("outcomes" => Gamma())

        # Incoming messages consist of distributions, delta Free Energy, and edge id's
        incoming_messages = Queue{Tuple{Gamma{Float64}, Float64, String}}()

        # Create instance
        self = new(id,
                   edges,
                   messages,
                   incoming_messages,
                   threshold,
                   verbose)
        return self
    end
end

function energy(node::"NodeGamma")
    "Compute internal energy of node"

    # Expected mean
    Ex = mean(node.messages["outcomes"])

    # Hyperparameters
    alpha = node.shape
    beta = node.rate

    # E_qx -log p(x|a,b)
    return -log(gamma(alpha)) + alpha*log(beta) + (alpha - 1)*(digamma(alpha) - log(beta)) - beta*Ex
end

function message(node::"NodeGamma")
    "Compute outgoing message"

    # Supply sufficient parameters for Gamma as output message
    return Gamma(node.shape, node.rate)
end

function act(node::"NodeGamma", edge_id::String)
    "Send out message for one of the connecting edges"

    # Compute message for a particular edge
    outgoing_message = message(node)

    # Push message in queue of connected edge
    enqueue!(eval(node.edges[edge_id]).incoming_messages["top"], outgoing_message)
    #TODO: avoid hard-coding edge id's

    return Nothing
end

function react(node::"NodeGamma")
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
