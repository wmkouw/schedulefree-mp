export TransitionGaussian

using Distributions: Normal, Gamma, mean
using DataStructures: Queue, enqueue!, dequeue!
include("../util.jl")

mutable struct TransitionGaussian
    """
    Composite node for state transitions.

    Contains a multiplication node with transition coefficients,
    an addition node with control coefficients, and
    a Gaussian process noise term.
    """

    # Identifiers of edges/nodes in factor graph
    id::String
    beliefs::Dict{String, Any}
    connected_edges::Dict{String, String}

    # Reaction parameters
    incoming::Queue{Tuple}
    threshold::Float64

    # Additional properties
    verbose::Bool

    function TransitionGaussian(id;
                                edge_data=0.0,
                                edge_mean=0.0,
                                edge_precision=1.0,
                                edge_transition=1.0,
                                edge_control=0.0,
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
            beliefs["data"] = Normal()
        end
        if isa(edge_mean, Float64)
            beliefs["mean"] = edge_mean
        else
            connected_edges["mean"] = edge_mean
            beliefs["mean"] = Normal()
        end
        if isa(edge_precision, Float64)
            if edge_precision > 0.0
                beliefs["precision"] = edge_precision
            else
                error("Exception: precision should be positive.")
            end
        else
            connected_edges["precision"] = edge_precision
            beliefs["precision"] = Gamma()
        end

        # Check for transition and controls
        if isa(edge_transition, Float64)
            beliefs["transition"] = edge_transition
        else
            connected_edges["transition"] = edge_transition
            beliefs["transition"] = Normal()
        end
        if isa(edge_control, Float64)
            beliefs["control"] = edge_control
        else
            connected_edges["control"] = edge_control
            beliefs["control"] = Normal()
        end

        # Initialize queue for incoming messages
        incoming = Queue{Tuple}()

        # Create instance
        self = new(id, beliefs, connected_edges, incoming, threshold, verbose)
        return self
    end
end

function energy(node::TransitionGaussian)
    """
    Compute internal energy of node.

    Assumes Gaussian distributions for x,m,a,u and Gamma for γ.
    """
    
    # Moments of transition belief
    Ea = mean(node.beliefs["transition"])
    if isa(node.beliefs["transition"], Float64)
        Va = 0.0
    else
        Va = var(node.beliefs["transition"])
    end

    # Moments of control belief
    Eu = mean(node.beliefs["control"])
    if isa(node.beliefs["control"], Float64)
        Vu = 0.0
    else
        Vu = var(node.beliefs["control"])
    end

    # Moments of mean belief
    Em = mean(node.beliefs["mean"])
    if isa(node.beliefs["mean"], Float64)
        Vm = 0.0
    else
        Vm = var(node.beliefs["mean"])
    end

    # Moments of data belief
    Ex = mean(node.beliefs["data"])
    if isa(node.beliefs["data"], Float64)
        Vx = 0.0
    else
        Vx = var(node.beliefs["data"])
    end

    # Moments of precision belief
    Eγ = mean(node.beliefs["precision"])

    # Check whether precision is clamped
    if isa(node.beliefs["precision"], Gamma)

        # Extract parameters
        shape, scale = params(node.beliefs["precision"])

        # -log-likelihood of Gaussian with expected parameters
        return -1/2*log(2*pi) + 1/2*(digamma(shape) + log(scale)) - Eγ/2*((Vx+Ex^2) -2*Ex*Em*Ea + (Vm+Em^2)*(Va+Ea^2))

    else
        # -log-likelihood of Gaussian with expected parameters
        return -1/2*log(2*pi) + 1/2*log(Eγ) - Eγ/2*((Vx+Ex^2) -2*Ex*Em*Ea + (Vm+Em^2)*(Va+Ea^2))
    end
    # TODO: incorporate control
end

function message(node::TransitionGaussian, edge_id::String)
    """
    Compute messages to each edge.

    Assumes Gaussian distributions for x,m,a,u and Gamma for γ.
    """

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Moments of transition belief
    Ea = mean(node.beliefs["transition"])
    if isa(node.beliefs["transition"], Float64)
        Va = 0.0
    else
        Va = var(node.beliefs["transition"])
    end

    # Moments of control belief
    Eu = mean(node.beliefs["control"])
    if isa(node.beliefs["control"], Float64)
        Vu = 0.0
    else
        Vu = var(node.beliefs["control"])
    end

    # Moments of mean belief
    Em = mean(node.beliefs["mean"])
    if isa(node.beliefs["mean"], Float64)
        Vm = 0.0
    else
        Vm = var(node.beliefs["mean"])
    end

    # Moments of data belief
    Ex = mean(node.beliefs["data"])
    if isa(node.beliefs["data"], Float64)
        Vx = 0.0
    else
        Vx = var(node.beliefs["data"])
    end

    # Moments of precision belief
    Eγ = mean(node.beliefs["precision"])

    if edge_name == "data"

        # Supply sufficient statistics
        message = Normal(Ea*Em, inv(Eγ))

    elseif edge_name == "mean"

        # Supply sufficient statistics
        message = Normal(Ex*Ea / (Va + Ea^2), inv(Eγ*(Va + Ea^2)))

    elseif edge_name == "precision"

        # Supply sufficient statistics
        rate = (inv(Vx) + inv(Vm) + (Ex - Em)^2)/2
        message = Gamma(3/2, 1/rate)

    elseif edge_name == "transition"

        # Supply sufficient statistics
        message = Normal(Ex*Em/(Vm + Em^2), Eγ*(Vm + Em^2))

    elseif edge_name == "control"

        error("Exception: not implemented yet.")

    else
        throw("Exception: edge id unknown.")
    end

    return message
end

function act(node::TransitionGaussian, edge_id::String, graph::MetaGraph)
    "Send out message for one of the connecting edges"

    # Compute message for a particular edge
    outgoing_message = message(node, edge_id)

    # Pass message to edge
    eval(graph[graph[edge_id, :id], :object]).messages[node.id] = outgoing_message

    return Nothing
end

function react(node::TransitionGaussian, graph::MetaGraph)
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

            for edge_out in setdiff(Set(edge_ids), Set([edge_id]))
                act(node, edge_out, graph)
            end
        end
    end

    return Nothing
end
