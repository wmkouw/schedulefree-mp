export EdgeGaussian

using Distributions: Normal, mean, std, params
using DataStructures: Queue, enqueue!, dequeue!
using LightGraphs, MetaGraphs

mutable struct EdgeGaussian
    """Edge with a Gaussian recognition distribution"""

    # Factor graph properties
    id::String
    block::Bool

    # Recognition distribution parameters
    mean::Float64
    precision::Float64
    free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Normal}

    function EdgeGaussian(id; mean=0.0, precision=1.0, free_energy=1e12, block=false)

        # Check valid precision
        if precision <= 0
            throw("Exception: non-positive precision.")
        end

        # Initialize messages
        messages = Dict{String, Normal}()

        # Construct instance
        self = new(id, block, mean, precision, free_energy, messages)
        return self
    end
end

function update(edge::EdgeGaussian)
    "Update recognition distribution as the product of messages"

    # Number of messages
    num_messages = length(edge.messages)

    if num_messages == 1

        # Collect parameters from single message
        mean, var = params(collect(values(edge.messages))[1])

        # Update parameters
        edge.mean = mean
        edge.precision = inv(var)

    elseif num_messages >= 2

        # Loop over stored messages
        total_precision = 0
        weighted_mean = 0
        for key in keys(edge.messages)

            # Extract parameters
            mean, var = params(edge.messages[key])

            # Sum over precisions
            total_precision += inv(var)

            # Compute weighted means
            weighted_mean += inv(var)*mean
        end

        # Update variational parameters
        edge.precision = total_precision
        edge.mean = inv(total_precision)*weighted_mean

    end
    return Nothing
end

function belief(edge::EdgeGaussian)
    "Outgoing message is belief over variable (recognition distribution)"
    return Normal(edge.mean, inv(edge.precision))
end

function entropy(edge::EdgeGaussian)
    "Entropy of Gaussian distribution"
    return 1/2*log(2*π * ℯ / edge.precision)
end

function gradient_entropy(edge::EdgeGaussian)
    "Gradient of entropy of Gaussian evaluated for supplied parameters"

    # Partial derivative with respect to mean
    partial_mean = 0.0

    # Partial derivative with respect to precision
    partial_precision = -1/(2*edge.precision)

    # Return tuple of partial derivatives
    return (partial_mean, partial_precision)
end

function free_energy(edge::EdgeGaussian, graph::MetaGraph)
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

function act(edge::EdgeGaussian,
             belief::Normal,
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

function react(edge::EdgeGaussian, graph::MetaGraph)
    "React to incoming messages"

    # Update variational distribution
    update(edge)

    # Compute free energy after update
    new_free_energy = free_energy(edge, graph)

    # Compute change in free energy after update
    delta_free_energy = new_free_energy - edge.free_energy

    # Message from edge to nodes
    act(edge, belief(edge), delta_free_energy, graph)

    # Store new free energy
    edge.free_energy = new_free_energy

    return Nothing
end
