export LikelihoodGaussian

using Distributions: Normal, Gamma, mean
using DataStructures: Queue, enqueue!, dequeue!
include("../util.jl")

mutable struct LikelihoodGaussian
    """
    Composite node for observation likelihoods.

    Contains a multiplication node with emission coefficients,
    and a Gaussian measurement noise term.
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

    function LikelihoodGaussian(id::String;
                                time=0,
                                edge_mean=0.0,
                                edge_data=0.0,
                                edge_precision=1.0,
                                edge_emission=1.0,
                                heuristics=Dict("backwards_in_time" => false),
                                threshold=0.0,
                                silent=false)

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

        # Check for emission and controls
        if isa(edge_emission, Float64)
            beliefs["emission"] = Delta(edge_emission)
        else
            connected_edges["emission"] = edge_emission
            beliefs["emission"] = Normal()
        end

        # Initialize queue for incoming messages
        incoming = Queue{Tuple}()

        # Create instance
        self = new(id, time, beliefs, connected_edges, incoming, heuristics, threshold, silent)
        return self
    end
end

function energy(node::LikelihoodGaussian)
    """
    Compute internal energy of node.

    Assumes Gaussian distributions for x,m,b and Gamma for γ.
    """

    # Moments of emission belief
    Eb = mean(node.beliefs["emission"])
    Vb = var(node.beliefs["emission"])

    # Moments of mean belief
    Em = mean(node.beliefs["mean"])
    Vm = var(node.beliefs["mean"])

    # Moments of data belief
    Ex = mean(node.beliefs["data"])
    Vx = var(node.beliefs["data"])

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
    return 1/2*log(2*pi) - 1/2*Elogγ + Eγ/2*(Vx+Ex^2 -2*Ex*Em*Eb + (Vm+Em^2)*(Vb+Eb^2))
end

function grad_energy(node::LikelihoodGaussian, edge_id::String)
    """
    Compute gradient of internal energy with respect to a particular edge.

    Assumes Gaussian distributions for x,m and Gamma for γ.
    """

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Moments of emission belief
    Eb = mean(node.beliefs["emission"])
    Vb = var(node.beliefs["emission"])

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
        partial_mean = Eγ*(Ex - Em*Eb)

        # Partial derivative with respect to variance of data belief
        partial_variance = -Eγ/2

        return (partial_mean, partial_variance)

    elseif edge_name == "mean"

        # Partial derivative with respect to mean of belief over mean
        partial_mean = Eγ*(Em*(Eb^2 + Vb) - Ex*Eb)

        # Partial derivative with respect to variance of belief over mean
        partial_variance = Eγ/2*(Vb + Eb^2)

        return (partial_mean, partial_variance)

    elseif edge_name == "emission"

        # Partial derivative with respect to mean of belief over emission coefficients
        partial_mean = Eγ*(Eb*(Em^2 + Vm) - Ex*Em)

        # Partial derivative with respect to variance of belief over emission coefficients
        partial_variance = Eγ/2*(Vm + Em^2)

        return (partial_mean, partial_variance)

    elseif edge_name == "precision"

        # Extract parameters
        shape, scale = params(node.beliefs["precision"])

        # Partial derivative with respect to shape of precision belief
        partial_shape = -1/2*polygamma(1, shape) + 1/2*scale*(Vx+Ex^2 -2*Ex*Em*Eb + (Vm+Em^2)*(Vb+Eb^2))

        # Partial derivative with respect to shape of precision belief
        partial_scale = -1/(2*scale) + 1/2*shape*(Vx+Ex^2 -2*Ex*Em*Eb + (Vm+Em^2)*(Vb+Eb^2))

        return (partial_shape, partial_scale)
    end
end

function message(node::LikelihoodGaussian, edge_id::String)
    """
    Compute messages to each edge.

    Assumes Gaussian distributions for x,m,a,u and Gamma for γ.
    """

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Moments of emission belief
    Eb = mean(node.beliefs["emission"])
    Vb = var(node.beliefs["emission"])

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
        message = Normal(Eb*Em, inv(Eγ))

    elseif edge_name == "mean"

        # Supply sufficient statistics
        message = Normal(Ex*Eb / (Vb + Eb^2), inv(Eγ*(Vb + Eb^2)))

    elseif edge_name == "precision"

        # Supply sufficient statistics
        inv_scale = (Vx + Ex^2 - 2*Ex*Eb*Em + (Em^2 + Vm)*(Eb^2 + Vb))/2

        message = Gamma(3/2., inv(inv_scale))

    elseif edge_name == "emission"

        # Supply sufficient statistics
        message = Normal(Ex*Em/(Vm + Em^2), inv(Eγ*(Vm + Em^2)))

    else
        throw("Exception: edge id unknown.")
    end

    return message
end

function act(node::LikelihoodGaussian, edge_id::String, graph::MetaGraph)
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

function react(node::LikelihoodGaussian, graph::MetaGraph)
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
