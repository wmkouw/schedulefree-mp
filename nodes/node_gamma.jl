export NodeGamma

using Distributions: Gamma, params, mean
using DataStructures: Queue, enqueue!, dequeue!
using SpecialFunctions: gamma, digamma
include("../util.jl")

mutable struct NodeGamma
    """Gamma distribution node"""

    # Identifiers of edges/nodes in factor graph
    id::String
    time::Int64
    beliefs::Dict{String, Any}
    connected_edges::Dict{String, String}

    # Reaction parameters
    incoming::Queue{Tuple}
    heuristics::Dict{String,Any}
    threshold::Float64
    silent::Bool

    # Additional properties
    verbose::Bool

    function NodeGamma(id::String;
                       time=0,
                       edge_data=1.0,
                       edge_shape=1.0,
                       edge_scale=1.0,
                       heuristics=Dict("backwards_in_time" => false),
                       threshold=0.0,
                       silent=false,
                       verbose=false)

       # Keep track of recognition distributions
       beliefs = Dict{String, Any}()
       connected_edges = Dict{String, String}()

       # Check for set parameters vs recognition distributions
       if isa(edge_data, Float64)
           beliefs["data"] = Delta(edge_data)
       else
           connected_edges["data"] = edge_data
           beliefs["data"] = Gamma()
       end
       if isa(edge_shape, Float64)
           if edge_shape >= 0.0
               beliefs["shape"] = Delta(edge_shape)
           else
               error("Exception: shape should be non-negative.")
           end
       else
           connected_edges["shape"] = edge_shape
           beliefs["shape"] = Gamma()
       end
       if isa(edge_scale, Float64)
           if edge_scale >= 0.0
               beliefs["scale"] = Delta(edge_scale)
           else
               error("Exception: scale should be non-negative.")
           end
       else
           connected_edges["scale"] = edge_scale
           beliefs["scale"] = Gamma()
       end

       # Initialize queue for incoming messages
       incoming = Queue{Tuple}()

       # Create instance
       self = new(id, time, beliefs, connected_edges, incoming, heuristics, threshold, silent, verbose)
       return self
    end
end

function energy(node::NodeGamma)
    """Compute internal energy of node."""

    #For now, this function assumes that parameters a,θ are clamped to particular values.
    # If a,θ are non-gamma distributions, then E_q(a)[log(a)] =/= ψ(a) + log(θ) e.g.
    if ~isa(node.beliefs["shape"], Delta) and ~isa(node.beliefs["scale"], Delta)
        error("Error: energy functional not known for belief on gamma parameters.")
    end

    # Expectations over beliefs
    Ea = mean(node.beliefs["shape"])
    Eθ = mean(node.beliefs["scale"])
    Ex = mean(node.beliefs["data"])

    # Parameters of data distribution
    shape = node.beliefs["data"].shape
    scale = node.beliefs["data"].scale

    # E_qx E_qa E_qθ -log Γ(x|a,θ)
    return log(gamma(Ea)) + Ea*log(Eθ) - (Ea - 1)*(digamma(shape) + log(scale)) + Ex/Eθ
end

function grad_energy(node::NodeGamma, edge_id::String)
    """Compute gradient of internal energy with respect to a particular edge."""

    # For now, this function assumes that parameters a,θ are clamped to particular values.
    # If a,θ are non-gamma distributions, then E_q(a)[log(a)] =/= ψ(a) + log(θ) e.g.
    if ~isa(node.beliefs["shape"], Delta) and ~isa(node.beliefs["scale"], Delta)
        error("Error: energy functional not known for belief on gamma parameters.")
    end

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Expectations over beliefs
    Ea = mean(node.beliefs["shape"])
    Eθ = mean(node.beliefs["scale"])
    Ex = mean(node.beliefs["data"])

    # Partial derivative with respect to x
    if edge_name == "data"

        # Parameters of data distribution
        shape, scale = params(node.beliefs["data"])

        # Partial derivative with respect to shape parameter of data belief
        partial_shape = -(Ea - 1)*polygamma(1, shape) + scale / Eθ

        # Partial derivative with respect to shape parameter of data belief
        partial_scale = -(Ea - 1)/scale + shape/Eθ

        return (partial_shape, partial_scale)
    end
end

function message(node::NodeGamma, edge_id::String)
    "Compute message to an edge"

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Expectations over beliefs
    Ea = mean(node.beliefs["shape"])
    Eθ = mean(node.beliefs["scale"])
    Ex = mean(node.beliefs["data"])

    if edge_name == "data"

        # Pass message based on belief over parameters
        message = Gamma(Ea, Eθ)

    elseif edge_name == "shape"

        # Pass message based on belief over parameters
        error("Exception: not implemented yet.")

    elseif edge_name == "scale"

        # Pass message based on belief over parameters
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

    # Find edge based on edge id
    edge = eval(Symbol(graph[edge_id, :id]))

    # Check if edge is blocked
    if !edge.block

        # Check heuristics for blocking
        if node.heuristics["backwards_in_time"] | (edge.time >= node.time)

            # Pass message to edge
            edge.messages[node.id] = outgoing_message
        end
    end

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
            # Mark that node has fired
            node.silent = false
        else
            # Mark that node has gone silent
            node.silent = true
        end
    end

    return Nothing
end
