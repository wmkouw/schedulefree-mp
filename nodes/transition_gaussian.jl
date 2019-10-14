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

    function TransitionGaussian(id::String;
                                time=0,
                                edge_data=0.0,
                                edge_mean=0.0,
                                edge_precision=1.0,
                                edge_transition=1.0,
                                edge_control=0.0,
                                heuristics=Dict{String,Any}("backwards_in_time" => false),
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
            beliefs["data"] = Normal()
        end
        if isa(edge_mean, Float64)
            beliefs["mean"] = Delta(edge_mean)
        else
            connected_edges["mean"] = edge_mean
            beliefs["mean"] = Normal()
        end
        if isa(edge_precision, Float64)
            if edge_precision > 0.0
                beliefs["precision"] = Delta(edge_precision)
            else
                error("Exception: precision should be positive.")
            end
        else
            connected_edges["precision"] = edge_precision
            beliefs["precision"] = Gamma()
        end

        # Check for transition and controls
        if isa(edge_transition, Float64)
            beliefs["transition"] = Delta(edge_transition)
        else
            connected_edges["transition"] = edge_transition
            beliefs["transition"] = Normal()
        end
        if isa(edge_control, Float64)
            beliefs["control"] = Delta(edge_control)
        else
            connected_edges["control"] = edge_control
            beliefs["control"] = Normal()
        end

        # Initialize queue for incoming messages
        incoming = Queue{Tuple}()

        # Create instance
        self = new(id, time, beliefs, connected_edges, incoming, heuristics, threshold, silent, verbose)
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
    Va = var(node.beliefs["transition"])

    # Moments of control belief
    Eu = mean(node.beliefs["control"])
    Vu = var(node.beliefs["control"])

    # Moments of mean belief
    Em = mean(node.beliefs["mean"])
    Vm = var(node.beliefs["mean"])

    # Moments of data belief
    Ex = mean(node.beliefs["data"])
    Vx = var(node.beliefs["data"])

    # Moments of precision belief
    Eγ = mean(node.beliefs["precision"])

    # Moments of precision belief
    Eγ = mean(node.beliefs["precision"])

    # Check whether precision is clamped
    if isa(node.beliefs["precision"], Gamma)

        # Extract parameters
        shape, scale = params(node.beliefs["precision"])

        # Expectation of log γ
        Elogγ = digamma(shape) + log(scale)
    else
        # Expectation of log of fixed γ
        Elogγ = log(Eγ)
    end

    # Energy of Gaussian with expected parameters
    return 1/2*log(2*pi) - 1/2*Elogγ + Eγ/2*(Vx+Ex^2 -2*Ex*Em*Ea - 2*Ex*Eu + 2*Ea*Em*Eu + (Vm+Em^2)*(Va+Ea^2) + Vu+Eu^2)
end

function grad_energy(node::TransitionGaussian, edge_id::String)
    """
    Compute gradient of internal energy with respect to a particular edge.

    Assumes Gaussian distributions for x,m,a and Gamma for γ.
    """

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Moments of transition belief
    Ea = mean(node.beliefs["transition"])
    Va = var(node.beliefs["transition"])

    # Moments of control belief
    Eu = mean(node.beliefs["control"])
    Vu = var(node.beliefs["control"])

    # Moments of mean belief
    Em = mean(node.beliefs["mean"])
    Vm = var(node.beliefs["mean"])

    # Moments of data belief
    Ex = mean(node.beliefs["data"])
    Vx = var(node.beliefs["data"])

    # Moments of precision belief
    Eγ = mean(node.beliefs["precision"])

    if edge_name == "data"

        # Partial derivative with respect to mean of data belief
        partial_mean = Eγ*(Ex - (Ea*Em + Eu))

        # Partial derivative with respect to variance of data belief
        partial_variance = Eγ/2

        return (partial_mean, partial_variance)

    elseif edge_name == "mean"

        # Partial derivative with respect to mean of belief over mean
        partial_mean = Eγ*(Em*(Ea^2 + Va) + Ea*(Eu - Ex))

        # Partial derivative with respect to variance of belief over mean
        partial_variance = Eγ/2*(Va + Ea^2)

        return (partial_mean, partial_variance)

    elseif edge_name == "transition"

        # Partial derivative with respect to mean of belief over transition coefficients
        partial_mean = Eγ*(Ea*(Em^2 + Vm) + Em*(Eu - Ex))

        # Partial derivative with respect to variance of belief over transition coefficients
        partial_variance = Eγ/2*(Vm + Em^2)

        return (partial_mean, partial_variance)

    elseif edge_name == "control"

        # Partial derivative with respect to mean of belief over control
        partial_mean = Eγ*(Ea*Em - Ex + Eu)

        # Partial derivative with respect to variance of belief over control
        partial_variance = Eγ/2

        return (partial_mean, partial_variance)

    elseif edge_name == "precision"

        # Extract parameters
        shape, scale = params(node.beliefs["precision"])

        # Partial derivative with respect to shape of precision belief
        partial_shape = -1/2*polygamma(1, shape) + 1/2*scale*(Vx+Ex^2 -2*Ex*Em*Ea - 2*Ex*Eu + 2*Ea*Em*Eu + (Vm+Em^2)*(Va+Ea^2) + Vu+Eu^2)

        # Partial derivative with respect to shape of precision belief
        partial_scale = -1/(2*scale) + 1/2*shape*(Vx+Ex^2 -2*Ex*Em*Ea - 2*Ex*Eu + 2*Ea*Em*Eu + (Vm+Em^2)*(Va+Ea^2) + Vu+Eu^2)

        return (partial_shape, partial_scale)
    end
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
    Va = var(node.beliefs["transition"])

    # Moments of control belief
    Eu = mean(node.beliefs["control"])
    Vu = var(node.beliefs["control"])


    # Moments of mean belief
    Em = mean(node.beliefs["mean"])
    Vm = var(node.beliefs["mean"])

    # Moments of data belief
    Ex = mean(node.beliefs["data"])
    Vx = var(node.beliefs["data"])

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
        inv_scale = (Vx + Ex^2 - 2*Ex*Ea*Em + (Em^2 + Vm)*(Ea^2 + Va))/2

        message = Gamma(3/2., inv(inv_scale))

    elseif edge_name == "transition"

        # Supply sufficient statistics
        message = Normal(Ex*Em/(Vm + Em^2), inv(Eγ*(Vm + Em^2)))

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

    # Extract edge from graph
    edge = eval(graph[graph[edge_id, :id], :object])

    # Check if edge is blocked
    if edge.block == false

        # Check heuristics for blocking
        if node.heuristics["backwards_in_time"] | (edge.time >= node.time)

            # Pass message to edge
            edge.messages[node.id] = outgoing_message
        end
    end

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
            # Mark that node has fired
            node.silent = false
        else
            # Mark that node has gone silent
            node.silent = true
        end
    end

    return Nothing
end
