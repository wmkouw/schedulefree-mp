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
    beliefs::Dict{String, Any}
    connected_edges::Dict{String, String}

    # Reaction parameters
    incoming::Queue{Tuple}
    threshold::Float64

    # Additional properties
    verbose::Bool

    function LikelihoodGaussian(id;
                                edge_mean=0.0,
                                edge_data=0.0,
                                edge_precision=1.0,
                                edge_emission=1.0,
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
            belief["mean"] = edge_mean
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

        # Check for emission and controls
        if isa(edge_emission, Float64)
            beliefs["emission"] = edge_emission
        else
            connected_edges["emission"] = edge_emission
            beliefs["emission"] = Normal()
        end

        # Initialize queue for incoming messages
        incoming = Queue{Tuple}()

        # Create instance
        self = new(id, beliefs, connected_edges, incoming, threshold, verbose)
        return self
    end
end

function energy(node::LikelihoodGaussian)
    "Compute internal energy of node"

    # Expected emission coefficients
    EB = mean(node.beliefs["emission"])

    # Expected mean
    Em = mean(node.beliefs["mean"])

    # Expected data
    Ex = mean(node.beliefs["data"])

    # Expected precision
    Et = mean(node.beliefs["precision"])

    # -log-likelihood of Gaussian with expected parameters
    return 1/2 *log(2*pi) - log(Et) + 1/2 *(Ex - EB*Em)'*Et*(Ex - EB*Em)
end

function message(node::LikelihoodGaussian, edge_id::String)
    "Compute message forward to x"

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge_id)

    # Expected emission coefficients
    EB = mean(node.beliefs["emission"])

    # Expected mean
    Em = mean(node.beliefs["mean"])

    # Expected data
    Ex = mean(node.beliefs["data"])

    # Expected precision
    Et = mean(node.beliefs["precision"])

    if edge_name == "data"

        # Supply sufficient statistics
        message = Normal(EB*Em, inv(Et))

    elseif edge_name == "mean"

        # Supply sufficient statistics
        message = Normal(Ex, inv(Et))

    elseif edge_name == "precision"

        # Supply sufficient statistics
        error("Exception: not implemented yet.")

    elseif edge_name == "emission"

        error("Exception: not implemented yet.")

    else
        throw("Exception: edge id unknown.")
    end

    return message
end

function act(node::LikelihoodGaussian, edge_id::String, graph::MetaGraph)
    "Send out message for one of the connecting edges"

    # Compute message for a particular edge
    outgoing_message = message(node, edge_id)

    # Pass message to edge
    eval(graph[graph[edge_id, :id], :object]).messages[node.id] = outgoing_message

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
        end
    end

    return Nothing
end
