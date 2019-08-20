export EdgeGamma

using LinearAlgebra: norm
using Distributions: Gamma, mean, params
using DataStructures: Queue, enqueue!, dequeue!
using SpecialFunctions: gamma, digamma, polygamma

mutable struct EdgeGamma
    """
    Edge with a Gamma recognition distribution
    """
    # Factor graph properties
    id::String
    block::Bool

    # Recognition distribution parameters
    shape::Float64
    scale::Float64
    free_energy::Float64
    grad_free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Gamma}

    function EdgeGamma(id; shape=1.0, scale=1.0, free_energy=1e12, grad_free_energy=1e12, block=false)

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
        self = new(id, block, shape, scale, free_energy, grad_free_energy, messages)
        return self
    end
end

function update(edge::EdgeGamma)
    "Update recognition distribution as the product of messages"

    # Number of messages
    num_messages = length(edge.messages)

    if num_messages == 0

        return Nothing

    elseif num_messages == 1

        # Extract parameters
        edge.shape, edge.scale = params(collect(values(edge.messages))[1])

    elseif num_messages >= 2

        new_shape = 0
        new_scale = 0
        for key in keys(edge.messages)

            # Extract parameters
            shape, scale = params(edge.messages[key])

            # Sum shape shape and scale parameters of each message
            new_shape += shape
            new_scale += inv(scale)

        end

        # Correct shape update
        edge.shape = new_shape - (num_messages - 1)
        edge.scale = inv(new_scale)

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

    # Extract connecting nodes
    N = neighbors(graph, graph[edge.id, :id])
    node_ids = [graph[n, :id] for n in N]

    # Initialize node energies
    U = 0

    # Update marginal at connected nodes
    for node_id in node_ids

        # Collect node variable via graph
        node = eval(get_prop(graph, graph[node_id, :id], :object))

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
    N = neighbors(graph, graph[edge.id, :id])
    node_ids = [graph[n, :id] for n in N]

    # Initialize node energies
    free_energy_shape = 0.
    free_energy_scale = 0.

    # Update marginal at connected nodes
    for node_id in node_ids

        # Collect node variable via graph
        node = eval(get_prop(graph, graph[node_id, :id], :object))

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
    N = neighbors(graph, graph[edge.id, :id])
    node_ids = [graph[n, :id] for n in N]

    # Update marginal at connected nodes
    for node_id in node_ids

        # Collect node variable via graph
        node = eval(get_prop(graph, graph[node_id, :id], :object))

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

    # Compute gradient of free energy after update
    edge.grad_free_energy = grad_free_energy(edge, graph)

    # Message from edge to nodes
    act(edge, belief(edge), edge.grad_free_energy, graph)

    return Nothing
end
