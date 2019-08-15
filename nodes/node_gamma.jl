export NodeGamma

using Distributions: Gamma, params, mean
using DataStructures: Queue, enqueue!, dequeue!
using SpecialFunctions: gamma, digamma
include("../util.jl")

mutable struct NodeGamma
    """Gamma distribution node"""

    # Identifiers of edges/nodes in factor graph
    id::String
    beliefs::Dict{String, Any}
    connected_edges::Dict{String, String}

    # Reaction parameters
    incoming::Queue{Tuple}
    threshold::Float64

    # Additional properties
    verbose::Bool

    function NodeGamma(id::String;
                       edge_data=1.0,
                       edge_shape=1.0,
                       edge_rate=1.0,
                       threshold=0.0,
                       verbose=false)

       # Keep track of recognition distributions
       beliefs = Dict{String, Any}()
       connected_edges = Dict{String, String}()

       # Check for set parameters vs recognition distributions
       if isa(edge_data, Float64)
           beliefs["data"] = edge_data
       else
           connected_edges["data"] = edge_data
           beliefs["data"] = Gamma()
       end
       if isa(edge_shape, Float64)
           if edge_shape >= 0.0
               beliefs["shape"] = edge_shape
           else
               error("Exception: shape should be non-negative.")
           end
       else
           connected_edges["shape"] = edge_shape
           beliefs["shape"] = Gamma()
       end
       if isa(edge_rate, Float64)
           if edge_rate >= 0.0
               beliefs["rate"] = edge_rate
           else
               error("Exception: rate should be non-negative.")
           end
       else
           connected_edges["rate"] = edge_rate
           beliefs["rate"] = Gamma()
       end

       # Initialize queue for incoming messages
       incoming = Queue{Tuple}()

       # Create instance
       self = new(id, beliefs, connected_edges, incoming, threshold, verbose)
       return self
    end
end

function energy(node::NodeGamma)
    "Compute internal energy of node"

    # Expectations over beliefs
    Ea = mean(node.beliefs["shape"])
    Eb = mean(node.beliefs["rate"])
    Ex = mean(node.beliefs["data"])

    # E_qx E_qa E_qb -log p(x|a,b)
    return -log(gamma(Ea)) + Ea*log(Eb) + (Ea - 1)*(digamma(Ea) - log(Eb)) - Eb*Ex
end

function message(node::NodeGamma, edge_id::String)
    "Compute message to each edge"

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Expectations over beliefs
    Ea = mean(node.beliefs["shape"])
    Eb = mean(node.beliefs["rate"])
    Ex = mean(node.beliefs["data"])

    if edge_name == "data"

        # Supply sufficient statistics
        message = Gamma(Ea, Eb)

    elseif edge_name == "shape"
        # Supply sufficient statistics
        error("Exception: not implemented yet.")
    elseif edge_name == "rate"
        # Supply sufficient statistics
        error("Exception: not implemented yet.")
    else
        throw("Exception: edge id unknown.")
    end

    return message
end

function act(node::NodeGamma, edge_id::String, graph::MetaGraph)
    "Send out message for one of the connecting edges"

    # Compute message for a particular edge
    outgoing_message = message(node, edge_id)

    # Pass message to edge
    eval(graph[graph[edge_id, :id], :object]).messages[node.id] = outgoing_message

    return Nothing
end

function react(node::NodeGamma, graph::MetaGraph)
    "React to incoming messages from edges"

    # Find edges attached to node
    N = neighbors(graph, graph[node.id, :id])
    edge_ids = [graph[n, :id] for n in N]

    # Loop over all edges that have produced incoming messages
    for n = 1:length(node.incoming)

        # Extract the id of the edge and the communicated delta free energy
        edge_id, delta_free_energy = dequeue!(node.incoming)

        # Check if change in free energy is sufficient to fire
        if abs(delta_free_energy) >= node.threshold

            if length(edge_ids) == 1
                act(node, edge_ids[1], graph)
            else
                # Loop over other edges
                for edge_out in setdiff(Set(edge_ids), Set([edge_id]))
                    act(node, edge_out, graph)
                end
            end
        end
    end

    return Nothing
end
