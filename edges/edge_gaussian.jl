export EdgeGaussian

import LinearAlgebra: norm
import DataStructures: Queue, enqueue!, dequeue!
import Distributions: Normal, mean, std, var, params
include("../util.jl")

mutable struct EdgeGaussian
    """Edge with a Gaussian recognition distribution"""

    # Factor graph properties
    id::String
    time::Int64
    block::Bool
    silent::Bool

    # Recognition distribution parameters
    mean::Float64
    precision::Float64
    free_energy::Float64
    grad_free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Normal}

    function EdgeGaussian(id::String;
                          time=0,
                          mean=0.0,
                          precision=1.0,
                          free_energy=1e12,
                          grad_free_energy=1e12,
                          block=false,
                          silent=false)

        # Check valid precision
        if precision <= 0
            throw("Exception: non-positive precision.")
        end

        # Initialize messages
        messages = Dict{String, Normal}()

        # Construct instance
        self = new(id, time, block, silent, mean, precision, free_energy, grad_free_energy, messages)
        return self
    end
end

function params(edge::EdgeGaussian)
    "Parameters of current belief"
    return edge.mean, inv(edge.precision)
end

function mean(edge::EdgeGaussian)
    "Mean of current belief"
    return edge.mean
end

function var(edge::EdgeGaussian)
    "Variance of current belief"
    return inv(edge.precision)
end

function moments(edge::EdgeGaussian)
    "First two moments of current belief"
    return mean(edge), var(edge)
end

function update(edge::EdgeGaussian)
    "Update recognition distribution as the product of messages"

    if !edge.block

        if length(edge.messages) > 0

            # Extract list of messages
            messages = collect(values(edge.messages))

            # Updated belief is the product of incoming messages
            new_belief = prod(messages)

            # Store new parameters
            edge.mean, stddev = params(new_belief)
            edge.precision = inv(stddev^2)
        end

        return Nothing
    end
end

function belief(edge::EdgeGaussian)
    "Outgoing message is belief over variable (recognition distribution)"
    return Normal(edge.mean, sqrt(inv(edge.precision)))
end

function entropy(edge::EdgeGaussian)
    "Entropy of Gaussian distribution"

    # Variance
    edge_variance = inv(edge.precision)

    # Entropy
    return 1/2*log(2*π * ℯ * edge_variance)
end

function grad_entropy(edge::EdgeGaussian)
    "Gradient of entropy of Gaussian evaluated for supplied parameters"

    # Variance
    edge_variance = inv(edge.precision)

    # Partial derivative with respect to mean
    partial_mean = 0.0

    # Partial derivative with respect to variance
    partial_variance = 1/(2*edge_variance)

    # Return tuple of partial derivatives
    return (partial_mean, partial_variance)
end

function free_energy(edge::EdgeGaussian, graph::MetaGraph)
    "Compute free energy of edge and connecting nodes"

    # Initialize node energies
    U = 0

    # Extract connecting nodes
    neighbours = neighbors(graph, graph[edge.id, :id])

    # Update marginal at connected nodes
    for neighbour in neighbours

        # Collect node variable via graph
        node = eval(Symbol(graph[neighbour, :id]))

        # Add to total energy
        U += energy(node)
    end

    # Compute own entropy
    H = entropy(edge)

    # Return free energy
    return U - H
end

function grad_free_energy(edge::EdgeGaussian, graph::MetaGraph)
    "Compute gradient of local free energy."

    # Extract connecting nodes
    neighbours = neighbors(graph, graph[edge.id, :id])

    # Initialize free energies partial derivatives
    free_energy_mean = 0.
    free_energy_variance = 0.

    # Update marginal at connected nodes
    for neighbour in neighbours

        # Collect node variable via graph
        node = eval(Symbol(graph[neighbour, :id]))

        # Gradient of node energy evaluated at current belief parameters
        energy_mean, energy_variance = grad_energy(node, edge.id)

        # Add evaluated energy gradients
        free_energy_mean += energy_mean
        free_energy_variance += energy_variance
    end

    # Compute gradient of own entropy
    entropy_mean, entropy_variance = grad_entropy(edge)

    # Subtract evaluated entropy gradients
    free_energy_mean -= entropy_mean
    free_energy_variance -= entropy_variance

    # Return norm of free energy gradient
    return norm([free_energy_mean, free_energy_variance])
end

function act(edge::EdgeGaussian,
             belief::Normal,
             delta_free_energy::Float64,
             graph::MetaGraph)
    "Pass belief to connected nodes."

    # Extract connecting nodes
    neighbours = neighbors(graph, graph[edge.id, :id])

    # Loop over neighbouring nodes
    for neighbour in neighbours

        # Collect node variable via graph
        node = eval(Symbol(graph[neighbour, :id]))

        # Get edge name from edge id
        edge_name = key_from_value(node.connected_edges, edge.id)

        # Update belief distribution
        node.beliefs[edge_name] = belief

        # Tell node that it has received a new message
        enqueue!(node.incoming, (edge.id, delta_free_energy))
    end

    return Nothing
end

function react(edge::EdgeGaussian, graph::MetaGraph)
    "React to incoming messages"

    # Update variational distribution
    update(edge)

    # Check whether edge should remain silent
    if !edge.silent

        # Compute free energy after update
        edge.free_energy = free_energy(edge, graph)

        # Compute gradient of free energy after update
        edge.grad_free_energy = grad_free_energy(edge, graph)

        # Message from edge to nodes
        act(edge, belief(edge), edge.grad_free_energy, graph)
    end

    return Nothing
end
