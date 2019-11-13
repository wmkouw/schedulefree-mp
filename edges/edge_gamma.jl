export EdgeGamma

import LinearAlgebra: norm
import Distributions: Gamma, mean, std, var, params
import DataStructures: Queue, enqueue!, dequeue!
import SpecialFunctions: gamma, digamma, polygamma
include("../util.jl")

mutable struct EdgeGamma
    """
    Edge with a Gamma recognition distribution
    """
    # Factor graph properties
    id::String
    time::Int64
    block::Bool
    silent::Bool

    # Recognition distribution parameters
    shape::Float64
    scale::Float64
    free_energy::Float64
    grad_free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Gamma}

    function EdgeGamma(id::String;
                       time=0,
                       shape=1.0,
                       scale=1.0,
                       free_energy=1e12,
                       grad_free_energy=1e12,
                       block=false,
                       silent=false)

        # Check valid parameters
        if shape < 0
            throw("Exception: shape parameter should be larger than 0.")
        end
        if scale < 0
            throw("Exception: scale parameter should be larger than 0.")
        end

        # Initialize messages
        messages = Dict{String, Gamma}()

        # Construct instance
        self = new(id, time, block, silent, shape, scale, free_energy, grad_free_energy, messages)
        return self
    end
end

function params(edge::EdgeGamma)
    "Parameters of current belief"
    return edge.shape, edge.scale
end

function mean(edge::EdgeGamma)
    "Mean of current belief"
    return edge.shape * edge.scale
end

function var(edge::EdgeGamma)
    "Variance of current belief"
    return edge.shape * edge.scale^2
end

function moments(edge::EdgeGamma)
    "First two moments of current belief"
    return mean(edge), var(edge)
end

function update(edge::EdgeGamma)
    "Update recognition distribution as the product of messages"

    if !edge.block

        # Initialize message
        belief = Gamma(1., Inf)

        # Loop over all incoming messages
        for key in keys(edge.messages)

            # Multiply the incoming beliefs
            belief = belief * edge.messages[key]
        end

        # Store parameters
        edge.shape, edge.scale = params(belief)

        return Nothing
    end
end

function belief(edge::EdgeGamma)
    "Outgoing message"
    return Gamma(edge.shape, edge.scale)
end

function entropy(edge::EdgeGamma)
    "Compute entropy of Gamma distribution"

    # Parameters
    a = edge.shape
    θ = edge.scale

    # Entropy of a univariate Gamma
    return a + log(θ) + log(gamma(a)) + (1-a)*digamma(a)
end

function grad_entropy(edge::EdgeGamma)
    "Gradient of entropy of Gamma evaluated for supplied parameters"

    # Parameters
    a = edge.shape
    θ = edge.scale

    # Partial derivative with respect to shape
    partial_shape = 1 + (1 + a)*polygamma(1, a)

    # Partial derivative with respect to scale
    partial_scale = 1/θ

    # Return tuple of partial derivatives
    return (partial_shape, partial_scale)
end

function free_energy(edge::EdgeGamma, graph::MetaGraph)
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

function grad_free_energy(edge::EdgeGamma, graph::MetaGraph)
    "Compute free energy of edge and connecting nodes"

    # Extract connecting nodes
    neighbours = neighbors(graph, graph[edge.id, :id])

    # Initialize node energies
    free_energy_shape = 0.
    free_energy_scale = 0.

    # Update marginal at connected nodes
    for neighbour in neighbours

        # Collect node variable via graph
        node = eval(Symbol(graph[neighbour, :id]))

        # Gradient of node energy evaluated at current belief parameters
        energy_shape, energy_scale = grad_energy(node, edge.id)

        # Add evaluated energy gradients
        free_energy_shape += energy_shape
        free_energy_scale += energy_scale
    end

    # Compute gradient of own entropy
    entropy_shape, entropy_scale = grad_entropy(edge)

    # Subtract evaluated entropy gradients
    free_energy_shape -= entropy_shape
    free_energy_scale -= entropy_scale

    # Return norm of free energy gradient
    return norm([free_energy_shape, free_energy_scale])
end

function act(edge::EdgeGamma,
             belief::Gamma,
             delta_free_energy::Float64,
             graph::MetaGraph)
    "Pass belief to connected nodes."

    # Extract connecting nodes
    neighbours = neighbors(graph, graph[edge.id, :id])

    # Update marginal at connected nodes
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

function react(edge::EdgeGamma, graph::MetaGraph)
    "React to incoming messages"

    # Update variational distribution
    update(edge)

    # Check whether edge should remain silent
    if !edge.silent

        # Compute gradient of free energy after update
        edge.grad_free_energy = grad_free_energy(edge, graph)

        # Message from edge to nodes
        act(edge, belief(edge), edge.grad_free_energy, graph)
    end

    return Nothing
end
