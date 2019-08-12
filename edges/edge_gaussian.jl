export EdgeGaussian

using Distributions: Normal, mean, std, params
using DataStructures: Queue, enqueue!, dequeue!

mutable struct EdgeGaussian
    "
    Edge with a Gaussian recognition distribution
    "
    # Recognition distribution parameters
    mean::Float64
    precision::Float64
    free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Normal}

    # Edge id in factor graph
    id::String

    function EdgeGaussian(id;
                          mean=0.0,
                          precision=1.0,
                          free_energy=1e12)

        # Check valid precision
        if precision <= 0
            throw("Exception: non-positive precision.")
        end

        # Initialize messages
        messages = Dict{String, Normal}()

        # Construct instance
        self = new(mean, precision, messages, id, free_energy)
        return self
    end
end

function update(edge::EdgeGaussian)
    "Update recognition distribution as the product of messages"

    # Loop over stored messages
    new_precision = 0
    weighted_mean = 0
    for key in keys(edge.messages)

        # Extract parameters
        mean, var = params(edge.messages[key])

        # Sum over precisions
        new_precision += inv(var)

        # Compute weighted means
        weighted_mean += inv(var)*mean
    end

    # Update variational parameters
    edge.precision = new_precision
    edge.mean = edge.precision*weighted_mean

    return Nothing
end

function message(edge::EdgeGaussian)
    "Outgoing message"
    return Normal(edge.mean, edge.precision)
end

function entropy(edge::EdgeGaussian)
    "Compute entropy of Gaussian distribution"
    return log(2*π * ℯ / edge.precision)/2
end

function free_energy(edge::EdgeGaussian, graph::MetaGraph)
    "Compute free energy of edge and connecting nodes"

    # Extract connecting nodes
    N = neighbors(graph, graph[edge.id, :id])

    # Query nodes for energy
    U = 0
    for n in N
        U += energy(get_prop!(graph, :id, graph[n, :id], :object))
        #TODO: check
    end

    # Compute own entropy
    H = entropy(edge)

    # Return FE
    return U - H
end

function act(edge::EdgeGaussian,
             variational_distribution::Normal,
             delta_free_energy::Float64,
             graph::MetaGraph)
    "Pass variational distribution to connected nodes."

    # Extract connecting nodes
    N = neighbors(graph, graph[edge.id, :id])

    # Update marginal at connected nodes
    for neighbor in N
        node = get_prop!(G[neigbor, :id], :object)
        node.messages[edge.id] = variational_distribution
    end

    # Tuple of message, change in free energy of message and edge id
    T = (marginal, delta_free_energy)



    return Nothing
end

function react(edge::EdgeGaussian, G::MetaGraph)
    "React to incoming messages"

    # Update variational distribution
    update(edge, edge.messages["left"], edge.messages["right"])

    # Compute delta free energy
    delta_free_energy = free_energy(edge) - edge.free_energy

    # Update edge's free energy
    edge.free_energy = free_energy(edge)

    # Message from edge to nodes
    act(G, edge, message(edge), delta_free_energy)

    return Nothing
end
